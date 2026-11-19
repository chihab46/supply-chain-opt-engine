"""Reproducible service/cost sensitivity experiments on pinned forecast evidence.

Run ``python -m supply_chain_opt_engine.inventory_tradeoff --phase holdout``.
Targets are scenario assumptions, never selected automatically from holdout scores.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import uuid
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from supply_chain_opt_engine.m5 import checksum
from supply_chain_opt_engine.optimization import (
    OptimizationConfig,
    PlanningFailure,
    checked_bundle,
    run_optimization,
)


def run_tradeoff(
    root,
    config,
    *,
    minimum_fill_rates=(0.0, 0.8, 0.9, 0.95),
    penalty_multipliers=(1.0,),
    capacity_multipliers=None,
    phase="holdout",
    runner=run_optimization,
):
    """Freeze sources once; save failures as evidence and retain every successful replay."""
    if phase not in {"holdout", "latest"}:
        raise ValueError("Choose holdout or latest.")
    capacities = (
        (config.capacity_multiplier,) if capacity_multipliers is None else capacity_multipliers
    )
    configs = list(
        dict.fromkeys(
            replace(
                config,
                minimum_fill_rate=floor,
                shortage_penalty_multiplier=penalty,
                capacity_multiplier=capacity,
            )
            for floor, penalty, capacity in itertools.product(
                minimum_fill_rates, penalty_multipliers, capacities
            )
        )
    )
    if not configs:
        raise ValueError("Specify at least one service, penalty and capacity scenario.")
    recpath, _ = checked_bundle(root, "artifacts/forecasts/reconciliation", "reconciled")
    basepath, _ = checked_bundle(root, "artifacts/metrics/baselines", "backtested")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = root / "artifacts/metrics/inventory_tradeoff" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    experiments = []
    for index, selected in enumerate(configs):
        logging.info("Sensitivity %d/%d: %s", index + 1, len(configs), asdict(selected))
        experiment = {"scenario": index, "config": asdict(selected)}
        try:
            bundle = runner(
                root, selected, reconciliation_path=recpath, baseline_path=basepath, phases=(phase,)
            )
        except PlanningFailure as error:
            # Infeasible is a proof; a time limit without an incumbent is not.
            proven = any(d.get("termination") == "infeasible" for d in error.diagnostics)
            experiment.update(
                status="infeasible" if proven else "no_feasible_incumbent",
                diagnostics=error.diagnostics,
                bundle=str(error.bundle.relative_to(root)) if error.bundle else None,
            )
        else:
            manifest_path = bundle / "manifest.json"
            metric = pd.read_csv(bundle / "policy_metrics.csv")
            metric["scenario"] = index
            for key in ["minimum_fill_rate", "shortage_penalty_multiplier", "capacity_multiplier"]:
                metric[key] = getattr(selected, key)
            metric["bundle"] = str(bundle.relative_to(root))
            rows.extend(metric.to_dict("records"))
            experiment.update(
                status="optimized",
                bundle=str(bundle.relative_to(root)),
                manifest_sha256=checksum(manifest_path),
            )
        experiments.append(experiment)
        # Keep progress durable even if a later experiment is interrupted.
        (directory / "experiments.json").write_text(json.dumps(experiments, indent=2) + "\n")
    metric = (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(
            columns=[
                "scenario",
                "phase",
                "policy",
                "evaluation",
                "scope",
                "minimum_fill_rate",
                "shortage_penalty_multiplier",
                "capacity_multiplier",
                "total_cost",
                "fill_rate",
                "minimum_store_fill_rate",
                "capacity_feasible",
                "service_feasible",
                "comparison_eligible",
                "bundle",
            ]
        )
    )
    metric.to_csv(directory / "tradeoff_metrics.csv", index=False)
    document = {
        "run_id": run_id,
        "status": "evaluated",
        "phase": phase,
        "reconciliation_manifest": str(recpath.relative_to(root)),
        "reconciliation_manifest_sha256": checksum(recpath),
        "baseline_manifest": str(basepath.relative_to(root)),
        "baseline_manifest_sha256": checksum(basepath),
        "scenario_assumptions": "Synthetic economics and inventory; no measured Walmart savings",
        "interpretation": (
            "Exploratory sensitivity after historical holdout review. Do not select or tune "
            "a deployment policy on these scores; validate on a new period. "
            "Projected service floors do not guarantee realized service."
        ),
        "experiments": experiments,
        "code_sha256": {"inventory_tradeoff.py": checksum(Path(__file__))},
        "artifact_sha256": {
            name: checksum(directory / name)
            for name in ["experiments.json", "tradeoff_metrics.csv"]
        },
    }
    (directory / "manifest.json").write_text(json.dumps(document, indent=2) + "\n")
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--phase", choices=["holdout", "latest"], default="holdout")
    parser.add_argument("--minimum-fill-rates", type=float, nargs="+", default=[0, 0.8, 0.9, 0.95])
    parser.add_argument("--penalty-multipliers", type=float, nargs="+", default=[1.0])
    parser.add_argument("--capacity-multipliers", type=float, nargs="+")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    document = yaml.safe_load(args.config.read_text())
    config = OptimizationConfig(
        **{
            k: v
            for k, v in document["optimization"].items()
            if k in OptimizationConfig.__dataclass_fields__ and k != "seed"
        },
        seed=document["seed"],
    )
    print(
        run_tradeoff(
            args.config.resolve().parent.parent,
            config,
            phase=args.phase,
            minimum_fill_rates=args.minimum_fill_rates,
            penalty_multipliers=args.penalty_multipliers,
            capacity_multipliers=args.capacity_multipliers,
        )
    )


if __name__ == "__main__":
    main()
