"""Solve replenishment scenarios and compare frozen policies on realized holdout sales.

Run ``python -m supply_chain_opt_engine.optimization`` after reconciliation.
Only checksummed committed inputs are read; this stage does not require PostgreSQL.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from supply_chain_opt_engine.history import ingestion_manifest, load_history
from supply_chain_opt_engine.inventory_data import (
    extend_weekly,
    forecast_matrix,
    generate_scenario,
    save_scenario,
)
from supply_chain_opt_engine.inventory_model import solve_store
from supply_chain_opt_engine.m5 import checksum
from supply_chain_opt_engine.metrics import QUANTILES, quantile_column
from supply_chain_opt_engine.rolling import rolling_schedule
from supply_chain_opt_engine.simulation import evaluate_policy, order_up_to, purchase_orders

LOGGER = logging.getLogger(__name__)
MILP = "lightgbm_mint_milp"
MEDIAN = "lightgbm_p50_order_up_to"
UPPER = "lightgbm_p90_order_up_to"
# Same P90 forecast, reviewed weekly: batches orders like a typical planner.
WEEKLY = "lightgbm_p90_weekly_order_up_to"
REVIEW_PERIODS = {WEEKLY: 7}
SEASONAL = "seasonal_naive_p50_order_up_to"
ETS = "autoets_static_safety_stock"
ROLLING = "_rolling"


@dataclass(frozen=True)
class OptimizationConfig:
    service_quantile: float = 0.9
    minimum_fill_rate: float = 0.0
    shortage_penalty_multiplier: float = 1.0
    time_limit_seconds: float = 60.0
    mip_relative_gap: float = 0.01
    capacity_multiplier: float = 1.0
    lead_time_shock_days: int = 0
    tail_days: int = 14
    seed: int = 42
    num_threads: int = 1
    safety_stock_z: float = 1.2815515655446004
    # Days between rolling replans on realized state; 0 keeps frozen plans only.
    rolling_review_days: int = 0
    # Share of each store's capacity that planners may fill. Replay and every
    # storage check still use the full capacity, so a share below 1 leaves
    # headroom for demand that comes in below forecast. Applies to all policies.
    planning_capacity_share: float = 1.0

    def __post_init__(self):
        if self.service_quantile not in QUANTILES:
            raise ValueError("Choose one of the six available forecast quantiles.")
        for name in ["time_limit_seconds", "capacity_multiplier", "shortage_penalty_multiplier"]:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if (
            isinstance(self.minimum_fill_rate, bool)
            or not np.isfinite(self.minimum_fill_rate)
            or not 0 <= self.minimum_fill_rate <= 1
        ):
            raise ValueError("Minimum fill rate must be a finite fraction between 0 and 1.")
        if (
            isinstance(self.planning_capacity_share, bool)
            or not np.isfinite(self.planning_capacity_share)
            or not 0 < self.planning_capacity_share <= 1
        ):
            raise ValueError("Planning capacity share must lie in (0, 1].")
        if not np.isfinite(self.mip_relative_gap) or not 0 <= self.mip_relative_gap < 1:
            raise ValueError("MIP relative gap must lie in [0,1).")
        for name, low, high in [
            ("lead_time_shock_days", 0, 4),
            ("tail_days", 9, 365),
            ("seed", 0, 2**31 - 1),
            ("num_threads", 1, 1),
            ("rolling_review_days", 0, 28),
        ]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer between {low} and {high}.")
        if not np.isfinite(self.safety_stock_z) or self.safety_stock_z < 0:
            raise ValueError("Safety stock multiplier must be finite and nonnegative.")


def checked_bundle(root, directory, status, path=None):
    if path is None:
        candidates = []
        for p in (root / directory).glob("*/manifest.json"):
            if json.loads(p.read_text()).get("status") == status:
                candidates.append(p)
        if not candidates:
            raise ValueError(f"Complete {directory} before inventory planning.")
        path = max(candidates, key=lambda p: p.stat().st_mtime)
    document = json.loads(path.read_text())
    if document.get("status") != status:
        raise ValueError(f"Incomplete upstream bundle: {path}")
    for name, digest in document["artifact_sha256"].items():
        if checksum(path.parent / name) != digest:
            raise ValueError(f"Upstream artifact checksum mismatch: {name}")
    return path, document


class PlanningFailure(ValueError):
    def __init__(self, diagnostics, bundle=None):
        super().__init__("No validated feasible MILP schedule; no purchase orders exported.")
        self.diagnostics = diagnostics
        self.bundle = bundle


def comparable_savings(metric):
    """Compare cost only when both policies meet the same service and capacity tests."""
    metric = metric.copy()
    metric["comparison_eligible"] = metric.capacity_feasible & metric.service_feasible
    metric["cost_saving_vs_global_p50"] = np.nan
    for evaluation in metric.evaluation.unique():
        for scope in metric.scope.unique():
            mask = metric.evaluation.eq(evaluation) & metric.scope.eq(scope)
            reference = metric[mask & metric.policy.eq(MEDIAN)]
            if (
                len(reference)
                and reference.total_cost.iloc[0] > 0
                and reference.comparison_eligible.iloc[0]
            ):
                good = mask & metric.comparison_eligible
                metric.loc[good, "cost_saving_vs_global_p50"] = (
                    1 - metric.loc[good, "total_cost"] / reference.total_cost.iloc[0]
                )
    return metric


def solve_stores(data, config, log_dir):
    orders = np.zeros((data.n, data.decision_days), dtype=int)
    diagnostics = []
    stores = sorted(data.capacity)
    remaining = config.time_limit_seconds
    for index, store in enumerate(stores):
        allowance = max(0.001, remaining / (len(stores) - index))
        LOGGER.info(
            "Solving %s: %d leaves, %.2fs solver budget",
            store,
            data.catalog.store_id.eq(store).sum(),
            allowance,
        )
        result = solve_store(
            data.for_store(store),
            time_limit=allowance,
            mip_gap=config.mip_relative_gap,
            seed=config.seed,
            threads=config.num_threads,
            log_path=log_dir / f"{store}.log",
        )
        diag = {"store_id": store, **result.diagnostics, "time_limit_seconds": allowance}
        diagnostics.append(diag)
        remaining -= result.diagnostics["solve_seconds"]
        if result.orders is None:
            raise PlanningFailure(diagnostics)
        orders[data.catalog.store_id.eq(store).to_numpy()] = result.orders
        LOGGER.info(
            "%s: %s, objective %.4f, gap %s",
            store,
            diag["termination"],
            diag["incumbent_objective"],
            diag["relative_gap"],
        )
    return orders, diagnostics


def raised_to_committed(sub):
    """Lift a store's planning capacity to the stock it already has committed.

    After below-forecast demand, on-hand stock plus the pipeline can exceed a
    store's (buffered) planning capacity with no new orders at all. Planning
    is then infeasible, yet ordering nothing for the whole store only adds
    stockouts. Returns the subproblem with each such store's capacity raised to
    its zero-order forecast peak, or None when committed stock fits everywhere.
    """
    from supply_chain_opt_engine.simulation import inventory_trace

    trace = inventory_trace(sub, np.zeros((sub.n, sub.decision_days), dtype=int))
    volume = sub.catalog.unit_volume_m3.to_numpy()
    stores = sub.catalog.store_id.to_numpy()
    capacity = dict(sub.capacity)
    raised = False
    for store, limit in sub.capacity.items():
        peak = float((volume[stores == store] @ trace["inventory"][stores == store]).max())
        if peak > limit + 1e-9:
            capacity[store] = peak * (1 + 1e-9) + 1e-9
            raised = True
    return replace(sub, capacity=capacity) if raised else None


def milp_review_planner(config, log_root):
    """Plan one rolling review.

    If infeasible: first lift capacity to already-committed stock, then relax
    the service floor, and only then order nothing.
    """

    def plan(sub, review):
        directory = log_root / f"review_{review:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        try:
            orders, stores = solve_stores(sub, config, directory)
            return orders, {"status": "optimized", "stores": stores}
        except PlanningFailure as error:
            failure = error.diagnostics
        raised = raised_to_committed(sub)
        if raised is not None:
            lifted = directory / "capacity_raised"
            lifted.mkdir(exist_ok=True)
            info = {"raised_capacity_m3": raised.capacity}
            try:
                orders, stores = solve_stores(raised, config, lifted)
                return orders, {
                    "status": "capacity_raised_to_committed_stock",
                    "stores": stores,
                    **info,
                }
            except PlanningFailure:
                sub = raised
        if sub.minimum_fill_rate > 0:
            relaxed = directory / "service_floor_relaxed"
            relaxed.mkdir(exist_ok=True)
            try:
                orders, stores = solve_stores(replace(sub, minimum_fill_rate=0.0), config, relaxed)
                return orders, {"status": "service_floor_relaxed", "stores": stores}
            except PlanningFailure:
                pass
        zero = np.zeros((sub.n, sub.decision_days), dtype=int)
        return zero, {"status": "no_feasible_plan_zero_orders", "failure": failure}

    return plan


def order_up_to_review_planner(safety, review_period=1):
    """Re-run an order-up-to rule at a review, lifting capacity to committed stock."""

    def plan(sub, review):
        try:
            return order_up_to(sub, safety, review_period=review_period), {"status": "planned"}
        except ValueError as error:
            raised = raised_to_committed(sub)
            if raised is not None:
                try:
                    orders = order_up_to(raised, safety, review_period=review_period)
                    return orders, {
                        "status": "capacity_raised_to_committed_stock",
                        "raised_capacity_m3": raised.capacity,
                    }
                except ValueError:
                    pass
            zero = np.zeros((sub.n, sub.decision_days), dtype=int)
            return zero, {
                "status": "committed_stock_exceeds_capacity_zero_orders",
                "error": str(error),
            }

    return plan


def safety_stock(policy, inputs, config):
    if policy != ETS:
        return None
    return (
        config.safety_stock_z
        * inputs.catalog.recent_std.to_numpy()
        * np.sqrt(inputs.catalog.lead_time_days.to_numpy() + 1)
    )


def run_phase(inputs, policy_demands, config, directory, *, actual=None, scenario_id=None):
    """Solve and validate all plans before writing any purchase-order exports.

    Frozen schedules are fixed at the origin. With realized demand and a positive
    ``rolling_review_days``, every policy is also replanned at weekly reviews from
    realized on-hand stock, as ``<policy>_rolling``.

    Every planner sees ``planning_capacity_share`` of each store's capacity;
    replay and all storage checks below use the full capacity.
    """
    directory.mkdir(parents=True, exist_ok=False)
    if inputs.minimum_fill_rate != config.minimum_fill_rate:
        raise ValueError("Scenario and optimizer minimum fill rates differ.")
    start = time.perf_counter()
    share = config.planning_capacity_share
    plan = replace(inputs, capacity={s: c * share for s, c in inputs.capacity.items()})
    orders, diagnostics = solve_stores(plan, config, directory)
    schedules = {MILP: orders}
    for policy, demand in policy_demands.items():
        LOGGER.info("Planning %s", policy)
        schedules[policy] = order_up_to(
            plan.with_demand(demand),
            safety_stock(policy, inputs, config),
            review_period=REVIEW_PERIODS.get(policy, 1),
        )
    rolling_reviews = {}
    if actual is not None and config.rolling_review_days > 0:
        period = config.rolling_review_days
        planners = {MILP: (plan.demand, milp_review_planner(config, directory / "rolling"))}
        for policy, demand in policy_demands.items():
            planners[policy] = (
                demand,
                order_up_to_review_planner(
                    safety_stock(policy, inputs, config), REVIEW_PERIODS.get(policy, 1)
                ),
            )
        for policy, (forecast, planner) in planners.items():
            LOGGER.info("Rolling replan %s every %d days", policy, period)
            schedules[policy + ROLLING], rolling_reviews[policy + ROLLING] = rolling_schedule(
                plan, forecast, actual, planner, period=period
            )
        (directory / "rolling_reviews.json").write_text(
            json.dumps(rolling_reviews, indent=2, default=str) + "\n"
        )
    metrics = []
    traces = []
    exports = []
    for policy, orders in schedules.items():
        # Different policies may PLAN on different forecasts, but all schedules
        # are scored against one common demand scenario for a fair comparison.
        scores, trace = evaluate_policy(
            inputs,
            orders,
            policy=policy,
            evaluation="forecast",
            allow_capacity_breach=policy != MILP,
        )
        metrics.extend(scores)
        traces.append(trace)
        if actual is not None:
            # Actual replay does not change a single order. Capacity violations
            # are exposed rather than silently discarding goods or hiding costs.
            scores, trace = evaluate_policy(
                inputs.with_demand(actual),
                orders,
                policy=policy,
                evaluation="realized",
                allow_capacity_breach=True,
            )
            metrics.extend(scores)
            traces.append(trace)
        exports.append(
            purchase_orders(
                inputs, orders, policy=policy, scenario_id=scenario_id or directory.name
            )
        )
    metric = comparable_savings(pd.DataFrame(metrics))
    metric.to_csv(directory / "metrics.csv", index=False)
    pd.concat(traces, ignore_index=True).to_parquet(
        directory / "inventory_trace.parquet", index=False
    )
    nonempty = [frame for frame in exports if len(frame)]
    exported = pd.concat(nonempty, ignore_index=True) if nonempty else exports[0]
    exported.to_csv(directory / "purchase_orders.csv", index=False)
    np.savez_compressed(
        directory / "order_schedules.npz",
        **schedules,
        node_ids=inputs.catalog.unique_id.to_numpy(dtype=str),
    )
    (directory / "solver_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return metric, {
        "phase": directory.name,
        "seconds": time.perf_counter() - start,
        "stores": diagnostics,
        "decision_days": inputs.decision_days,
        "evaluation_days": inputs.h,
        "rolling_review_days": config.rolling_review_days if rolling_reviews else 0,
        "policies": list(schedules),
    }


def run_optimization(
    root,
    config,
    *,
    reconciliation_path=None,
    baseline_path=None,
    phases=("holdout", "latest"),
    output_root=None,
):
    recpath, rec = checked_bundle(
        root, "artifacts/forecasts/reconciliation", "reconciled", reconciliation_path
    )
    globalpath = root / "artifacts/models/lightgbm" / rec["upstream_run_id"] / "manifest.json"
    if checksum(globalpath) != rec["upstream_manifest_sha256"]:
        raise ValueError("Global model lineage differs from reconciliation.")
    globalpath, glob = checked_bundle(root, "artifacts/models/lightgbm", "forecasted", globalpath)
    basepath, base = checked_bundle(
        root, "artifacts/metrics/baselines", "backtested", baseline_path
    )
    ingestpath = root / "data/interim" / rec["source"]["ingestion_run_id"] / "manifest.json"
    ingestpath, ingest = ingestion_manifest(root, ingestpath)
    if checksum(ingestpath) != rec["source"]["manifest_sha256"]:
        raise ValueError("Reconciliation and ingestion lineage differ.")
    if base["source"]["ingestion_run_id"] != rec["source"]["ingestion_run_id"]:
        raise ValueError("Statistical baseline and reconciliation scopes differ.")
    econpath = ingestpath.parent / "dim_sku_economics.csv"
    if checksum(econpath) != ingest["artifact_sha256"][econpath.name]:
        raise ValueError("Economics artifact checksum mismatch.")
    history, source = load_history(root, manifest_path=ingestpath)
    economics = pd.read_csv(econpath)
    nodes = pd.read_csv(recpath.parent / "hierarchy_nodes.csv").query("level == 'store_item'")
    nodes = nodes.sort_values("unique_id").reset_index(drop=True)
    ids = nodes.unique_id.tolist()
    if len(ids) != rec["bottom_series"] or set(ids) != set(history.unique_id):
        raise ValueError("Forecast hierarchy and observed scope differ.")
    holdout = next(f for f in rec["folds"] if f["role"] == "holdout")
    baseline = pd.read_parquet(basepath.parent / "predictions.parquet")
    frozen = [f for f in base["folds"] if f["role"] == "holdout"]
    if frozen != [holdout]:
        raise ValueError("Statistical baseline holdout dates differ from reconciled forecasts.")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output_root = output_root or root / "artifacts/orders/optimization"
    output_root.mkdir(parents=True, exist_ok=True)
    scenario_root = root / "data/synthetic/inventory" / run_id
    scenario_root.mkdir(parents=True, exist_ok=False)
    metrics = []
    timings = []
    with tempfile.TemporaryDirectory(prefix=".inventory-", dir=output_root) as temp:
        staging = Path(temp)
        try:
            for phase in phases:
                if phase == "holdout":
                    panel = pd.read_parquet(recpath.parent / "predictions.parquet")
                    panel = panel[panel.fold_id.eq(holdout["fold_id"])]
                    raw = pd.read_parquet(globalpath.parent / "predictions.parquet")
                    raw = raw[raw.fold_id.eq(holdout["fold_id"])]
                    origin = pd.Timestamp(holdout["origin"])
                elif phase == "latest":
                    panel = pd.read_parquet(recpath.parent / "latest_forecasts.parquet")
                    raw = pd.read_parquet(globalpath.parent / "latest_forecasts.parquet")
                    origin = pd.Timestamp(rec["latest_origin"])
                else:
                    raise ValueError("Choose holdout or latest planning phase.")
                panel = panel[
                    panel.model.eq(rec["config"]["primary_method"]) & panel.unique_id.isin(ids)
                ]
                horizon = 28
                dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon)
                if not pd.to_datetime(panel.forecast_origin).eq(origin).all():
                    raise ValueError("Forecast origins differ from the selected inventory phase.")
                demand = forecast_matrix(
                    panel, ids, dates, quantile_column(config.service_quantile)
                )
                inputs, metadata, pipeline = generate_scenario(
                    history,
                    economics,
                    nodes,
                    origin,
                    demand,
                    tail_days=config.tail_days,
                    seed=config.seed,
                    capacity_multiplier=config.capacity_multiplier,
                    lead_time_shock=config.lead_time_shock_days,
                    minimum_fill_rate=config.minimum_fill_rate,
                    shortage_penalty_multiplier=config.shortage_penalty_multiplier,
                )
                metadata.update(
                    scenario_id=f"{run_id}/{phase}",
                    ingestion_run_id=ingest["run_id"],
                    economics_sha256=checksum(econpath),
                    reconciliation_run_id=rec["run_id"],
                )
                save_scenario(inputs, metadata, pipeline, scenario_root / phase)
                policies = {
                    MEDIAN: extend_weekly(
                        forecast_matrix(raw, ids, dates, "q50"), config.tail_days
                    ),
                    UPPER: extend_weekly(forecast_matrix(raw, ids, dates, "q90"), config.tail_days),
                }
                policies[WEEKLY] = policies[UPPER]
                actual = None
                if phase == "holdout":
                    for policy, model in [(SEASONAL, "SeasonalNaive"), (ETS, "AutoETS")]:
                        f = baseline[
                            baseline.fold_id.eq(holdout["fold_id"]) & baseline.model.eq(model)
                        ]
                        policies[policy] = extend_weekly(
                            forecast_matrix(f, ids, dates, "point"), config.tail_days
                        )
                    observed = history[history.ds.isin(inputs.dates)]
                    actual = forecast_matrix(observed, ids, inputs.dates, "y")
                    labels = forecast_matrix(panel, ids, dates, "y")
                    if not np.array_equal(actual[:, :horizon], labels):
                        raise ValueError("Holdout labels differ from observed sales.")
                LOGGER.info(
                    "Planning %s: %d leaves, %d decision + %d tail days",
                    phase,
                    inputs.n,
                    horizon,
                    config.tail_days,
                )
                metric, timing = run_phase(
                    inputs,
                    policies,
                    config,
                    staging / phase,
                    actual=actual,
                    scenario_id=f"{run_id}/{phase}",
                )
                metrics.append(metric.assign(phase=phase))
                timings.append(timing)
            status = "optimized"
            pd.concat(metrics, ignore_index=True).to_csv(
                staging / "policy_metrics.csv", index=False
            )
        except PlanningFailure as error:
            status = "failed"
            error_diagnostics = error.diagnostics
            (staging / "failure.json").write_text(json.dumps(error.diagnostics, indent=2) + "\n")
            LOGGER.error("%s", error)
        manifest = {
            "run_id": run_id,
            "status": status,
            "config": asdict(config),
            "source": source,
            "reconciliation_run_id": rec["run_id"],
            "reconciliation_manifest_sha256": checksum(recpath),
            "global_run_id": glob["run_id"],
            "baseline_run_id": base["run_id"],
            "baseline_manifest_sha256": checksum(basepath),
            "scenario_directory": str(scenario_root.relative_to(root)),
            "scenario_manifest_sha256": {
                p.parent.name: checksum(p) for p in scenario_root.glob("*/manifest.json")
            },
            "phases": timings,
            "point_in_time": (
                "Frozen policies are fixed at the origin. *_rolling policies replan every "
                f"{config.rolling_review_days} days from realized on-hand stock, using only "
                "sales before each review and the unrefreshed origin forecast"
                if config.rolling_review_days
                else "All policies frozen before any realized-demand evaluation"
            ),
            "objective": (
                "Fixed fees + daily holding + per-unit lost-sales penalties; purchase cost excluded"
            ),
            "comparison": (
                f"Shared inventory, pipeline, capacity, economics and "
                f"{config.tail_days}-day no-order tail"
            ),
            "capacity_policy": (
                "End-of-day storage; realized overflow explicitly invalidates capacity feasibility"
            ),
            "service_policy": (
                "Per-store minimum unit fill on forecast demand over both windows; "
                "not a realized fill-rate guarantee"
            ),
            "forecast_comparison": "All policies replayed on the same selected quantile demand",
            "economics_basis": "Synthetic assumptions; not measured Walmart costs or savings",
            "evaluation_context": (
                "Exploratory policy redesign after reviewing historical holdout results; "
                "new prospective validation required before a deployment claim"
            ),
            "versions": {n: version(n) for n in ["pyomo", "highspy", "numpy", "pandas"]},
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "solver_threads": config.num_threads,
            },
            "code_sha256": {
                name: checksum(Path(__file__).parent / name)
                for name in [
                    "optimization.py",
                    "inventory_data.py",
                    "inventory_model.py",
                    "simulation.py",
                    "rolling.py",
                ]
            },
            "artifact_sha256": {
                str(p.relative_to(staging)): checksum(p) for p in staging.rglob("*") if p.is_file()
            },
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        bundle = output_root / run_id
        staging.rename(bundle)
    if status != "optimized":
        raise PlanningFailure(error_diagnostics, bundle=bundle)
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--reconciliation-manifest", type=Path)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--phase", choices=["both", "holdout", "latest"], default="both")
    parser.add_argument("--minimum-fill-rate", type=float)
    parser.add_argument("--shortage-penalty-multiplier", type=float)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        document = yaml.safe_load(args.config.read_text())
        opts = document["optimization"]
        if opts["solver"] != "appsi_highs" or opts["horizon_days"] != 28:
            raise ValueError("Use appsi_highs with the frozen 28-day horizon.")
        config = OptimizationConfig(
            **{
                k: v
                for k, v in opts.items()
                if k in OptimizationConfig.__dataclass_fields__ and k != "seed"
            },
            seed=document["seed"],
        )
        from dataclasses import replace

        overrides = {
            name: getattr(args, name)
            for name in ["minimum_fill_rate", "shortage_penalty_multiplier"]
            if getattr(args, name) is not None
        }
        config = replace(config, **overrides)
        phases = ("holdout", "latest") if args.phase == "both" else (args.phase,)
        root = args.config.resolve().parent.parent
        bundle = run_optimization(
            root,
            config,
            reconciliation_path=args.reconciliation_manifest,
            baseline_path=args.baseline_manifest,
            phases=phases,
            output_root=root / document["artifacts"]["root_dir"] / "orders/optimization",
        )
        print(bundle / "manifest.json")
        print(
            pd.read_csv(bundle / "policy_metrics.csv")
            .query("evaluation == 'realized'")[
                ["policy", "scope", "fill_rate", "total_cost", "capacity_feasible"]
            ]
            .to_string(index=False)
        )
        return 0
    except (ValueError, OSError, KeyError, ImportError, yaml.YAMLError) as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
