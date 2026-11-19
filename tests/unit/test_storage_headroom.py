import numpy as np
import pytest
from test_rolling import scenario, uniform

from supply_chain_opt_engine.dashboard_data import saved_setting
from supply_chain_opt_engine.optimization import MILP, WEEKLY, OptimizationConfig, run_phase
from supply_chain_opt_engine.simulation import inventory_trace
from supply_chain_opt_engine.storage_headroom import BASELINE, POLICY, summarize


def peak_volume(inputs, orders, demand):
    """Largest end-of-day stock volume for the single test store."""
    trace = inventory_trace(inputs.with_demand(demand), orders)
    return float((inputs.catalog.unit_volume_m3.to_numpy() @ trace["inventory"]).max())


def test_buffer_constrains_planners_but_not_replay_capacity(tmp_path):
    inputs = scenario(uniform(4.0, h=20), decision_days=14, capacity=40.0, initial=[2, 2])
    actual = np.array(uniform(1.0, h=20))
    config = OptimizationConfig(
        time_limit_seconds=5, rolling_review_days=7, planning_capacity_share=0.5
    )
    metric, _ = run_phase(inputs, {WEEKLY: inputs.demand}, config, tmp_path / "half", actual=actual)
    schedules = np.load(tmp_path / "half" / "order_schedules.npz")
    for policy in [MILP, WEEKLY]:
        assert peak_volume(inputs, schedules[policy], inputs.demand) <= 20.0 + 1e-6
    # Replay checks storage against the full 40 m3, not the 20 m3 planning limit.
    realized = metric[
        metric.evaluation.eq("realized")
        & metric.policy.eq(MILP)
        & metric.scope.eq("including_tail")
    ]
    peak = peak_volume(inputs, schedules[MILP], actual)
    assert realized.max_capacity_violation_m3.iloc[0] == pytest.approx(
        max(0.0, peak - 40.0), abs=1e-6
    )
    assert bool(realized.capacity_feasible.iloc[0]) == (peak <= 40.0 + 1e-5)
    full, _ = run_phase(
        inputs,
        {WEEKLY: inputs.demand},
        OptimizationConfig(time_limit_seconds=5, rolling_review_days=7),
        tmp_path / "full",
        actual=actual,
    )
    full_orders = np.load(tmp_path / "full" / "order_schedules.npz")[MILP]
    assert peak_volume(inputs, full_orders, inputs.demand) > 20.0


@pytest.mark.parametrize("share", [0, -0.1, 1.01, float("nan"), True])
def test_planning_capacity_share_is_validated(share):
    with pytest.raises(ValueError, match="Planning capacity share"):
        OptimizationConfig(planning_capacity_share=share)


def test_saved_runs_without_newer_settings_match_their_defaults():
    assert saved_setting({}, "planning_capacity_share") == 1.0
    assert saved_setting({}, "rolling_review_days") == 0
    assert saved_setting({"planning_capacity_share": 0.9}, "planning_capacity_share") == 0.9
    assert saved_setting({}, "not_a_setting") is None


def cell(window, share, milp, base):
    rows = []
    for policy, (cost, ok) in [(POLICY, milp), (BASELINE, base)]:
        rows.append(
            {
                "policy": policy,
                "evaluation": "realized",
                "scope": "including_tail",
                "total_cost": cost,
                "fill_rate": 0.98,
                "minimum_store_fill_rate": 0.97,
                "max_capacity_violation_m3": 0.0 if ok else 5.0,
                "comparison_eligible": ok,
            }
        )
    return {"window": window, "share": share, "status": "solved", "metrics": rows}


def test_summary_defines_savings_only_when_both_are_eligible():
    cells = [
        cell("F1", 0.9, (70.0, True), (100.0, True)),
        cell("F2", 0.9, (70.0, True), (100.0, False)),
        {"window": "F3", "share": 0.9, "status": "failed", "error": "No incumbent"},
    ]
    summary = summarize(cells).set_index("window")
    assert summary.loc["F1", "saving"] == pytest.approx(0.30)
    assert np.isnan(summary.loc["F2", "saving"])
    assert summary.loc["F3", "status"] == "failed" and "No incumbent" in summary.loc["F3", "error"]


def test_committed_overflow_lifts_capacity_instead_of_ordering_nothing(tmp_path):
    from supply_chain_opt_engine.optimization import (
        milp_review_planner,
        order_up_to_review_planner,
        raised_to_committed,
    )

    # Item 0's stock alone exceeds the 30 m3 limit; item 1 has none and needs orders.
    sub = scenario(uniform(3.0, h=20), decision_days=14, capacity=30.0, initial=[60, 0])
    raised = raised_to_committed(sub)
    assert raised is not None and raised.capacity["S"] == pytest.approx(57.0, rel=1e-6)
    assert raised_to_committed(scenario(uniform(3.0, h=20), decision_days=14)) is None
    config = OptimizationConfig(time_limit_seconds=5)
    for planner in [milp_review_planner(config, tmp_path), order_up_to_review_planner(None, 7)]:
        orders, info = planner(sub, 0)
        assert info["status"] == "capacity_raised_to_committed_stock"
        assert orders[1].sum() > 0 and orders[0].sum() == 0
        peak = peak_volume(raised, orders, sub.demand)
        assert peak <= raised.capacity["S"] + 1e-6
