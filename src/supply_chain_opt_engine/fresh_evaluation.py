"""Pre-registered fresh-period evaluation on unused 2016 windows.

Run ``python -m supply_chain_opt_engine.fresh_evaluation`` after the pinned
upstream runs exist. The protocol is ``docs/fresh_evaluation.md``.

At each fresh origin this module refits every model with **frozen** settings on
data up to that origin only, exactly as the fold-4 holdout was produced:
LightGBM quantiles with the folds 1-3 hyperparameters, SeasonalNaive / AutoETS /
TSB, and MinT reconciliation. It then runs the inventory phase (frozen and
weekly-rolling plans for every policy) against realized sales, and scores the
pre-registered primary comparison. Nothing is tuned on the fresh windows.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import tempfile
import time
import uuid
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from supply_chain_opt_engine.baseline import BaselineConfig, forecast_fold
from supply_chain_opt_engine.evaluation import Fold, split_fold
from supply_chain_opt_engine.feature_store import FeatureEncoder, load_feature_store
from supply_chain_opt_engine.hierarchy import Hierarchy
from supply_chain_opt_engine.history import ingestion_manifest, load_history
from supply_chain_opt_engine.inventory_data import (
    extend_weekly,
    forecast_matrix,
    generate_scenario,
    save_scenario,
)
from supply_chain_opt_engine.learning import (
    ForecastConfig,
    fit_quantiles,
    forecast_frame,
    predict_raw,
    prediction_dataset,
    training_dataset,
)
from supply_chain_opt_engine.m5 import checksum
from supply_chain_opt_engine.metrics import QUANTILES, quantile_column
from supply_chain_opt_engine.optimization import (
    ETS,
    MEDIAN,
    SEASONAL,
    UPPER,
    WEEKLY,
    OptimizationConfig,
    checked_bundle,
    run_phase,
)
from supply_chain_opt_engine.reconciliation import (
    ReconciliationConfig,
    load_forecast_bundle,
    prediction_rows,
    reconcile_origin,
)
from supply_chain_opt_engine.reconciliation_data import prepare_reconciliation_data

LOGGER = logging.getLogger(__name__)
PROTOCOL = Path("docs/fresh_evaluation.md")


def fresh_folds(origins, first_fold_id, horizon, tail_days, *, after, last_observed):
    """Fresh windows: chronological, non-overlapping, after ``after``, fully observed."""
    origins = [date.fromisoformat(str(value)) for value in origins]
    if not origins or origins != sorted(set(origins)):
        raise ValueError("Fresh origins must be distinct and chronologically ordered.")
    span = timedelta(days=horizon + tail_days)
    if origins[0] < after:
        raise ValueError("The first fresh window starts before earlier evaluation data ends.")
    for left, right in zip(origins, origins[1:], strict=False):
        if left + span > right:
            raise ValueError("Fresh windows, including their tails, must not overlap.")
    if origins[-1] + span > last_observed:
        raise ValueError("The last fresh window extends past the observed sales.")
    return [
        Fold(first_fold_id + index, origin, horizon, "fresh", "fresh")
        for index, origin in enumerate(origins)
    ]


def leaf_scores(frame, model):
    """Leaf WMAPE and central P05-P95 coverage over one window."""
    y = frame.y.to_numpy(dtype=float)
    point = frame.point.to_numpy(dtype=float)
    lower = frame[quantile_column(0.05)].to_numpy(dtype=float)
    upper = frame[quantile_column(0.95)].to_numpy(dtype=float)
    return {
        "model": model,
        "wmape": float(np.abs(y - point).sum() / y.sum()) if y.sum() else np.nan,
        "coverage_90": float(((y >= lower) & (y <= upper)).mean()),
        "rows": len(frame),
    }


def optimization_config(document):
    options = document["optimization"]
    return OptimizationConfig(
        **{
            k: v
            for k, v in options.items()
            if k in OptimizationConfig.__dataclass_fields__ and k != "seed"
        },
        seed=document["seed"],
    )


def baseline_config(document):
    options = document["backtest"]
    return BaselineConfig(
        horizon=options["horizon_days"],
        season_length=options["season_length"],
        conformal_windows=options.get("conformal_windows", 2),
        tsb_alpha_d=options.get("tsb_alpha_d", 0.1),
        tsb_alpha_p=options.get("tsb_alpha_p", 0.1),
        n_jobs=options.get("n_jobs", 1),
    )


def primary_comparison(metrics, policy, baseline, window):
    rows = metrics[
        metrics.evaluation.eq("realized")
        & metrics.scope.eq("including_tail")
        & metrics.policy.isin([policy, baseline])
    ].set_index("policy")
    mine, theirs = rows.loc[policy], rows.loc[baseline]
    eligible = bool(mine.comparison_eligible and theirs.comparison_eligible)
    return {
        "window": window,
        "policy_cost": float(mine.total_cost),
        "baseline_cost": float(theirs.total_cost),
        "policy_fill": float(mine.fill_rate),
        "baseline_fill": float(theirs.fill_rate),
        "policy_lowest_store_fill": float(mine.minimum_store_fill_rate),
        "baseline_lowest_store_fill": float(theirs.minimum_store_fill_rate),
        "policy_eligible": bool(mine.comparison_eligible),
        "baseline_eligible": bool(theirs.comparison_eligible),
        "both_eligible": eligible,
        "saving": float(1 - mine.total_cost / theirs.total_cost) if eligible else np.nan,
    }


def verdict(comparison):
    """The pre-registered success criterion from docs/fresh_evaluation.md."""
    both = comparison[comparison.both_eligible]
    pooled = float(1 - both.policy_cost.sum() / both.baseline_cost.sum()) if len(both) else np.nan
    supports = bool(comparison.policy_eligible.all() and len(both) and pooled > 0)
    return {
        "policy_eligible_all_windows": bool(comparison.policy_eligible.all()),
        "windows_both_eligible": int(len(both)),
        "pooled_saving": pooled,
        "supports_holdout_finding": supports,
    }


def run_fresh_evaluation(root: Path, document: dict) -> Path:
    fresh = document["fresh_evaluation"]
    opt = optimization_config(document)
    horizon = document["forecast"]["horizon_days"]
    ingest_path, ingest = ingestion_manifest(
        root, root / "data/interim" / fresh["ingestion_run_id"] / "manifest.json"
    )
    store = load_feature_store(root, None, ingest_path)
    history, source = load_history(root, manifest_path=ingest_path)
    lgb_path, upstream = load_forecast_bundle(
        root, root / "artifacts/models/lightgbm" / fresh["lightgbm_run_id"] / "manifest.json"
    )
    rec_path, rec = checked_bundle(
        root,
        "artifacts/forecasts/reconciliation",
        "reconciled",
        root
        / "artifacts/forecasts/reconciliation"
        / fresh["reconciliation_run_id"]
        / "manifest.json",
    )
    base_path, base = checked_bundle(
        root,
        "artifacts/metrics/baselines",
        "backtested",
        root / "artifacts/metrics/baselines" / fresh["baseline_run_id"] / "manifest.json",
    )
    if rec["upstream_run_id"] != upstream["run_id"]:
        raise ValueError("Pinned reconciliation was not built on the pinned LightGBM run.")
    for name, document_source in [
        ("LightGBM", upstream),
        ("reconciliation", rec),
        ("baseline", base),
    ]:
        if document_source["source"]["ingestion_run_id"] != ingest["run_id"]:
            raise ValueError(f"Pinned {name} run uses a different ingestion scope.")
    bconfig = baseline_config(document)
    if asdict(bconfig) != base["config"]:
        raise ValueError("Baseline settings differ from the pinned baseline run.")
    rconfig = ReconciliationConfig(**rec["config"])
    fconfig = ForecastConfig(**upstream["config"])
    holdout = next(f for f in upstream["folds"] if f["role"] == "holdout")
    after = date.fromisoformat(holdout["validation_end"]) + timedelta(days=opt.tail_days)
    folds = fresh_folds(
        fresh["origins"],
        fresh["first_fold_id"],
        horizon,
        opt.tail_days,
        after=after,
        last_observed=history.ds.max().date(),
    )
    economics_path = ingest_path.parent / "dim_sku_economics.csv"
    if checksum(economics_path) != ingest["artifact_sha256"][economics_path.name]:
        raise ValueError("Economics artifact checksum mismatch.")
    economics = pd.read_csv(economics_path)
    nodes = pd.read_csv(rec_path.parent / "hierarchy_nodes.csv").query("level == 'store_item'")
    nodes = nodes.sort_values("unique_id").reset_index(drop=True)
    ids = nodes.unique_id.tolist()
    catalog = (
        store.history[["unique_id", "store_id", "item_id"]]
        .drop_duplicates()
        .merge(store.items, on="item_id", validate="many_to_one")
    )
    hierarchy = Hierarchy.from_catalog(catalog)
    encoder = FeatureEncoder(upstream["_categories"])

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output_root = root / "artifacts/metrics/fresh_evaluation"
    output_root.mkdir(parents=True, exist_ok=True)
    scenario_root = root / "data/synthetic/fresh" / run_id
    forecasts, scores, metrics, comparisons, timings = [], [], [], [], []
    with tempfile.TemporaryDirectory(prefix=".fresh-", dir=output_root) as temporary:
        staging = Path(temporary)
        for index, fold in enumerate(folds, 1):
            window = f"F{index}"
            begin = time.perf_counter()
            origin = pd.Timestamp(fold.origin)
            dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon)
            split_fold(history, fold, 29 + 2 * horizon)
            LOGGER.info("%s: refitting frozen LightGBM at %s", window, fold.origin)
            train = training_dataset(store, encoder, fold.origin, fconfig)
            data = prediction_dataset(store, encoder, fold.origin, fconfig, with_labels=True)
            model = fit_quantiles(train, upstream["best_params"], fconfig)
            raw = forecast_frame(predict_raw(model, data.X), data, fold.fold_id)
            LOGGER.info("%s: statistical baselines", window)
            statistical, _, _ = forecast_fold(history, fold, bconfig)
            LOGGER.info("%s: MinT reconciliation", window)
            prepared = prepare_reconciliation_data(
                store, hierarchy, fold.origin, raw, upstream, None
            )
            results, _, checks = reconcile_origin(
                hierarchy, prepared, rconfig, rconfig.seed + fold.fold_id
            )
            point, quantiles, _ = results[rconfig.primary_method]
            reconciled = prediction_rows(
                hierarchy,
                point,
                quantiles,
                prepared.future_dates,
                fold.origin,
                rconfig.primary_method,
                fold.fold_id,
            )
            reconciled = reconciled[reconciled.unique_id.isin(ids)].merge(
                raw[["unique_id", "ds", "y"]], on=["unique_id", "ds"], validate="one_to_one"
            )
            columns = ["unique_id", "ds", "y", "point", *[quantile_column(q) for q in QUANTILES]]
            for name, frame in [
                ("LightGBM", raw),
                (rconfig.primary_method, reconciled),
                *[(m, statistical[statistical.model.eq(m)]) for m in ["AutoETS", "SeasonalNaive"]],
            ]:
                scores.append({"window": window, **leaf_scores(frame, name)})
                forecasts.append(frame[columns].assign(model=name, window=window))
            LOGGER.info("%s: inventory scenario", window)
            demand = forecast_matrix(reconciled, ids, dates, quantile_column(opt.service_quantile))
            inputs, metadata, pipeline = generate_scenario(
                history,
                economics,
                nodes,
                origin,
                demand,
                tail_days=opt.tail_days,
                seed=opt.seed,
                capacity_multiplier=opt.capacity_multiplier,
                lead_time_shock=opt.lead_time_shock_days,
                minimum_fill_rate=opt.minimum_fill_rate,
                shortage_penalty_multiplier=opt.shortage_penalty_multiplier,
            )
            metadata.update(
                scenario_id=f"{run_id}/{window}",
                ingestion_run_id=ingest["run_id"],
                economics_sha256=checksum(economics_path),
                reconciliation_run_id=rec["run_id"],
            )
            save_scenario(inputs, metadata, pipeline, scenario_root / window)
            policies = {
                MEDIAN: extend_weekly(forecast_matrix(raw, ids, dates, "q50"), opt.tail_days),
                UPPER: extend_weekly(forecast_matrix(raw, ids, dates, "q90"), opt.tail_days),
            }
            policies[WEEKLY] = policies[UPPER]
            for policy, name in [(SEASONAL, "SeasonalNaive"), (ETS, "AutoETS")]:
                frame = statistical[statistical.model.eq(name)]
                policies[policy] = extend_weekly(
                    forecast_matrix(frame, ids, dates, "point"), opt.tail_days
                )
            observed = history[history.ds.isin(inputs.dates)]
            actual = forecast_matrix(observed, ids, inputs.dates, "y")
            if not np.array_equal(actual[:, :horizon], forecast_matrix(raw, ids, dates, "y")):
                raise ValueError("Fresh labels differ from observed sales.")
            metric, timing = run_phase(
                inputs,
                policies,
                opt,
                staging / window,
                actual=actual,
                scenario_id=f"{run_id}/{window}",
            )
            metrics.append(metric.assign(window=window, origin=str(fold.origin)))
            comparisons.append(
                primary_comparison(
                    metric, fresh["primary_policy"], fresh["primary_baseline"], window
                )
            )
            timings.append(
                {
                    "window": window,
                    "origin": str(fold.origin),
                    "seconds": time.perf_counter() - begin,
                    "inventory_seconds": timing["seconds"],
                    "max_coherence_error": max(c["point_coherence_error"] for c in checks),
                    "stores": timing["stores"],
                }
            )
            LOGGER.info("%s finished in %.1fs: %s", window, timings[-1]["seconds"], comparisons[-1])
        comparison = pd.DataFrame(comparisons)
        outcome = verdict(comparison)
        pd.concat(forecasts, ignore_index=True).to_parquet(
            staging / "forecasts.parquet", index=False
        )
        pd.DataFrame(scores).to_csv(staging / "forecast_scores.csv", index=False)
        pd.concat(metrics, ignore_index=True).to_csv(staging / "policy_metrics.csv", index=False)
        comparison.to_csv(staging / "primary_comparison.csv", index=False)
        manifest = {
            "run_id": run_id,
            "status": "evaluated",
            "protocol": str(PROTOCOL),
            "protocol_sha256": checksum(root / PROTOCOL),
            "fresh_evaluation": {k: str(v) for k, v in fresh.items()},
            "folds": [fold.metadata() for fold in folds],
            "optimization_config": asdict(opt),
            "baseline_config": asdict(bconfig),
            "reconciliation_config": asdict(rconfig),
            "pinned_manifests_sha256": {
                "ingestion": checksum(ingest_path),
                "lightgbm": checksum(lgb_path),
                "reconciliation": checksum(rec_path),
                "baseline": checksum(base_path),
            },
            "source": source,
            "scenario_directory": str(scenario_root.relative_to(root)),
            "verdict": outcome,
            "timings": timings,
            "information_policy": (
                "Models refitted with frozen settings on data up to each origin; realized "
                "sales enter only rolling on-hand state before each weekly review."
            ),
            "economics_basis": "Synthetic assumptions; not measured Walmart costs or savings",
            "versions": {n: version(n) for n in ["pyomo", "highspy", "lightgbm", "statsforecast"]},
            "hardware": {"platform": platform.platform(), "processor": platform.processor()},
            "code_sha256": {
                name: checksum(Path(__file__).parent / name)
                for name in [
                    "fresh_evaluation.py",
                    "optimization.py",
                    "rolling.py",
                    "inventory_model.py",
                    "simulation.py",
                    "learning.py",
                    "reconciliation.py",
                    "baseline.py",
                ]
            },
        }
        manifest["artifact_sha256"] = {
            str(p.relative_to(staging)): checksum(p) for p in staging.rglob("*") if p.is_file()
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        bundle = output_root / run_id
        staging.rename(bundle)
    return bundle


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        document = yaml.safe_load(args.config.read_text())
        root = args.config.resolve().parent.parent
        bundle = run_fresh_evaluation(root, document)
        manifest = json.loads((bundle / "manifest.json").read_text())
        print(pd.read_csv(bundle / "primary_comparison.csv").to_string(index=False))
        print(json.dumps(manifest["verdict"], indent=2))
        print(bundle / "manifest.json")
    except (ValueError, OSError, KeyError, yaml.YAMLError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
