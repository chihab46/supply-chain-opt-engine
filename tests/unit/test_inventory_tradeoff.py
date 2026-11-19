import json
from dataclasses import replace

import pandas as pd
import pytest

from supply_chain_opt_engine.inventory_tradeoff import run_tradeoff
from supply_chain_opt_engine.optimization import OptimizationConfig, PlanningFailure


def test_sensitivity_pins_sources_and_records_infeasible_and_unproven_failures(tmp_path):
    for directory, status in [
        ("artifacts/forecasts/reconciliation", "reconciled"),
        ("artifacts/metrics/baselines", "backtested"),
    ]:
        p = tmp_path / directory / "frozen/manifest.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"status": status, "artifact_sha256": {}}))
    requests = []

    def runner(root, config, **kwargs):
        requests.append((config, kwargs))
        if config.minimum_fill_rate == 1:
            raise PlanningFailure([{"termination": "infeasible"}])
        if config.minimum_fill_rate == 0.95:
            raise PlanningFailure([{"termination": "maxTimeLimit"}])
        bundle = root / f"artifacts/orders/optimization/run-{len(requests)}"
        bundle.mkdir(parents=True)
        (bundle / "manifest.json").write_text("{}")
        pd.DataFrame([{"phase": "holdout", "policy": "example", "total_cost": 123}]).to_csv(
            bundle / "policy_metrics.csv", index=False
        )
        return bundle

    result = run_tradeoff(
        tmp_path, OptimizationConfig(), minimum_fill_rates=[0.8, 0.8, 0.95, 1], runner=runner
    )
    document = json.loads((result / "manifest.json").read_text())
    assert [r["status"] for r in document["experiments"]] == [
        "optimized",
        "no_feasible_incumbent",
        "infeasible",
    ]
    assert len(requests) == 3
    assert len({str(kwargs["reconciliation_path"]) for _, kwargs in requests}) == 1
    assert all(kwargs["phases"] == ("holdout",) for _, kwargs in requests)
    assert "prospective" not in document["status"]
    scores = pd.read_csv(result / "tradeoff_metrics.csv")
    assert scores.minimum_fill_rate.tolist() == [0.8]
    assert "new period" in document["interpretation"]

    # A grid with no successful solves still produces an inspectable table.
    failed = run_tradeoff(tmp_path, OptimizationConfig(), minimum_fill_rates=[1], runner=runner)
    assert pd.read_csv(failed / "tradeoff_metrics.csv").empty


def test_sensitivity_validates_entire_grid_before_running(tmp_path):
    with pytest.raises(ValueError, match="fill rate"):
        run_tradeoff(tmp_path, OptimizationConfig(), minimum_fill_rates=[0.9, 1.5])
    with pytest.raises(ValueError, match="positive"):
        replace(OptimizationConfig(), shortage_penalty_multiplier=0)
    with pytest.raises(ValueError, match="at least one"):
        run_tradeoff(tmp_path, OptimizationConfig(), minimum_fill_rates=[])
