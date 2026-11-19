"""EXPLORATORY: planning-capacity buffer sensitivity on already-used windows.

The pre-registered fresh-period test (docs/fresh_evaluation.md) found that
every P90 policy exceeded storage when demand came in below forecast. This
module tests one remedy: planners may fill only ``share`` of each store's
capacity, while replay and every storage check use the full capacity.

All windows here have been inspected before, and any share chosen from these
results is selected in-sample. Results are therefore exploratory and must not
be reported as an independent test. Every grid cell is kept, including
failures.

The fresh windows reuse the registered run's checksummed forecasts and rebuild
each scenario, verifying that the regenerated scenario files match the
registered ones byte for byte. The holdout reruns the standard optimization.

Run ``python -m supply_chain_opt_engine.storage_headroom``.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from supply_chain_opt_engine.fresh_evaluation import optimization_config
from supply_chain_opt_engine.history import ingestion_manifest, load_history
from supply_chain_opt_engine.inventory_data import (
    extend_weekly,
    forecast_matrix,
    generate_scenario,
    save_scenario,
)
from supply_chain_opt_engine.m5 import checksum
from supply_chain_opt_engine.metrics import quantile_column
from supply_chain_opt_engine.optimization import (
    ETS,
    MEDIAN,
    MILP,
    SEASONAL,
    UPPER,
    WEEKLY,
    PlanningFailure,
    run_optimization,
    run_phase,
)

LOGGER = logging.getLogger(__name__)
SHARES = (1.0, 0.95, 0.9, 0.85, 0.8)
POLICY = MILP + "_rolling"
BASELINE = WEEKLY + "_rolling"
SCENARIO_FILES = [
    "catalog.csv",
    "store_capacity.csv",
    "starting_inventory.csv",
    "pipeline_receipts.csv",
]


def fresh_run(root, run_id):
    path = root / "artifacts/metrics/fresh_evaluation" / run_id / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "evaluated":
        raise ValueError("The fresh-period run is not complete.")
    for name in ["forecasts.parquet", "policy_metrics.csv"]:
        if checksum(path.parent / name) != manifest["artifact_sha256"][name]:
            raise ValueError(f"Fresh-period artifact checksum mismatch: {name}")
    return path, manifest


def fresh_window(root, document, run_id, window):
    """Rebuild one fresh window's scenario, policies and realized demand."""
    _, manifest = fresh_run(root, run_id)
    index = int(window.removeprefix("F")) - 1
    origin = pd.Timestamp(manifest["folds"][index]["origin"])
    pinned = manifest["fresh_evaluation"]
    ingest_path, ingest = ingestion_manifest(
        root, root / "data/interim" / pinned["ingestion_run_id"] / "manifest.json"
    )
    history, _ = load_history(root, manifest_path=ingest_path)
    economics = pd.read_csv(ingest_path.parent / "dim_sku_economics.csv")
    rec_dir = root / "artifacts/forecasts/reconciliation" / pinned["reconciliation_run_id"]
    nodes = pd.read_csv(rec_dir / "hierarchy_nodes.csv").query("level == 'store_item'")
    nodes = nodes.sort_values("unique_id").reset_index(drop=True)
    ids = nodes.unique_id.tolist()
    forecasts = pd.read_parquet(
        root / "artifacts/metrics/fresh_evaluation" / run_id / "forecasts.parquet"
    )
    forecasts = forecasts[forecasts.window.eq(window)]
    opt = optimization_config(document)
    horizon = document["forecast"]["horizon_days"]
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon)

    def frame(model):
        return forecasts[forecasts.model.eq(model)]

    demand = forecast_matrix(
        frame("mint_shrink"), ids, dates, quantile_column(opt.service_quantile)
    )
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
    registered = root / manifest["scenario_directory"] / window / "manifest.json"
    expected = json.loads(registered.read_text())["artifact_sha256"]
    with tempfile.TemporaryDirectory() as temporary:
        save_scenario(inputs, metadata, pipeline, Path(temporary) / window)
        for name in SCENARIO_FILES:
            if checksum(Path(temporary) / window / name) != expected[name]:
                raise ValueError(f"{window}: regenerated {name} differs from the registered run.")
    raw = frame("LightGBM")
    policies = {
        MEDIAN: extend_weekly(forecast_matrix(raw, ids, dates, "q50"), opt.tail_days),
        UPPER: extend_weekly(forecast_matrix(raw, ids, dates, "q90"), opt.tail_days),
    }
    policies[WEEKLY] = policies[UPPER]
    for policy, name in [(SEASONAL, "SeasonalNaive"), (ETS, "AutoETS")]:
        policies[policy] = extend_weekly(
            forecast_matrix(frame(name), ids, dates, "point"), opt.tail_days
        )
    observed = history[history.ds.isin(inputs.dates)]
    actual = forecast_matrix(observed, ids, inputs.dates, "y")
    if not np.array_equal(actual[:, :horizon], forecast_matrix(raw, ids, dates, "y")):
        raise ValueError(f"{window}: saved labels differ from observed sales.")
    return inputs, policies, actual, opt


def run_cell(root, document, run_id, window, share, output):
    """One (window, share) cell. Failures are returned, never dropped."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    directory = output / f"{window}_share_{share:.2f}"
    try:
        if window == "holdout":
            pinned = document["fresh_evaluation"]
            config = replace(optimization_config(document), planning_capacity_share=share)
            bundle = run_optimization(
                root,
                config,
                reconciliation_path=root
                / "artifacts/forecasts/reconciliation"
                / pinned["reconciliation_run_id"]
                / "manifest.json",
                baseline_path=root
                / "artifacts/metrics/baselines"
                / pinned["baseline_run_id"]
                / "manifest.json",
                phases=("holdout",),
                output_root=directory,
            )
            metric = pd.read_csv(bundle / "policy_metrics.csv")
            source = str(bundle.relative_to(root))
        else:
            inputs, policies, actual, opt = fresh_window(root, document, run_id, window)
            config = replace(opt, planning_capacity_share=share)
            metric, _ = run_phase(inputs, policies, config, directory, actual=actual)
            source = str(directory.relative_to(root))
        return {
            "window": window,
            "share": share,
            "status": "solved",
            "source": source,
            "metrics": metric.to_dict(orient="records"),
        }
    except (PlanningFailure, ValueError) as error:
        return {"window": window, "share": share, "status": "failed", "error": str(error)}


def summarize(cells):
    rows = []
    for cell in cells:
        base = {"window": cell["window"], "share": cell["share"], "status": cell["status"]}
        if cell["status"] != "solved":
            rows.append({**base, "error": cell["error"]})
            continue
        metric = pd.DataFrame(cell["metrics"])
        metric = metric[metric.evaluation.eq("realized") & metric.scope.eq("including_tail")]
        metric = metric.set_index("policy")
        mine, theirs = metric.loc[POLICY], metric.loc[BASELINE]
        both = bool(mine.comparison_eligible and theirs.comparison_eligible)
        rows.append(
            {
                **base,
                "milp_cost": mine.total_cost,
                "baseline_cost": theirs.total_cost,
                "milp_fill": mine.fill_rate,
                "baseline_fill": theirs.fill_rate,
                "milp_lowest_store_fill": mine.minimum_store_fill_rate,
                "baseline_lowest_store_fill": theirs.minimum_store_fill_rate,
                "milp_max_overflow_m3": mine.max_capacity_violation_m3,
                "baseline_max_overflow_m3": theirs.max_capacity_violation_m3,
                "milp_eligible": bool(mine.comparison_eligible),
                "baseline_eligible": bool(theirs.comparison_eligible),
                "both_eligible": both,
                "saving": 1 - mine.total_cost / theirs.total_cost if both else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["share", "window"], ascending=[False, True])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--fresh-run", default="20261119T164107Z-8bcce3f6")
    parser.add_argument("--shares", type=float, nargs="+", default=list(SHARES))
    parser.add_argument("--windows", nargs="+", default=["holdout", "F1", "F2", "F3"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        document = yaml.safe_load(args.config.read_text())
        root = args.config.resolve().parent.parent
        fresh_path, _ = fresh_run(root, args.fresh_run)
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        output = root / "artifacts/metrics/storage_headroom" / run_id
        (output / "cells").mkdir(parents=True)
        jobs = [(w, s) for s in args.shares for w in args.windows]
        cells = []
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = {
                pool.submit(run_cell, root, document, args.fresh_run, w, s, output / "cells"): (
                    w,
                    s,
                )
                for w, s in jobs
            }
            for future in as_completed(futures):
                cell = future.result()
                cells.append(cell)
                LOGGER.info("%s share %.2f: %s", cell["window"], cell["share"], cell["status"])
        summary = summarize(cells)
        summary.to_csv(output / "summary.csv", index=False)
        (output / "cells.json").write_text(json.dumps(cells, indent=2, default=str) + "\n")
        manifest = {
            "run_id": run_id,
            "status": "exploratory",
            "label": (
                "EXPLORATORY. Windows were inspected before; any share chosen from these "
                "results is selected in-sample and is not an independent test."
            ),
            "fresh_run_id": args.fresh_run,
            "fresh_manifest_sha256": checksum(fresh_path),
            "shares": args.shares,
            "windows": args.windows,
            "workers": args.workers,
            "primary": {"policy": POLICY, "baseline": BASELINE, "scope": "realized, 42 days"},
            "config": asdict(optimization_config(document)),
            "code_sha256": {
                name: checksum(Path(__file__).parent / name)
                for name in [
                    "storage_headroom.py",
                    "optimization.py",
                    "rolling.py",
                    "inventory_model.py",
                    "simulation.py",
                ]
            },
            "created": date.today().isoformat(),
            "artifact_sha256": {
                name: checksum(output / name) for name in ["summary.csv", "cells.json"]
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(summary.to_string(index=False))
        print(output / "manifest.json")
    except (ValueError, OSError, KeyError, yaml.YAMLError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
