import json

import numpy as np
import pandas as pd
import pytest

from supply_chain_opt_engine.inventory_data import InventoryInputs
from supply_chain_opt_engine.optimization import (
    MILP,
    OptimizationConfig,
    order_up_to_review_planner,
    run_phase,
)
from supply_chain_opt_engine.rolling import review_days, rolling_schedule, state_at_review
from supply_chain_opt_engine.simulation import inventory_trace, order_up_to


def scenario(demand, *, decision_days, lead=2, moq=1, capacity=1000.0, initial=None):
    d = np.array(demand, dtype=float)
    n, h = d.shape
    catalog = pd.DataFrame(
        {
            "unique_id": [f"S/i{i}" for i in range(n)],
            "store_id": ["S"] * n,
            "item_id": [f"i{i}" for i in range(n)],
            "lead_time_days": [lead] * n,
            "minimum_order_quantity": [moq] * n,
            "unit_volume_m3": [1.0] * n,
            "daily_holding_cost": [0.1] * n,
            "stockout_penalty": [5.0] * n,
            "fixed_order_cost": [1.0] * n,
            "unit_purchase_cost": [10.0] * n,
            "recent_std": [1.0] * n,
        }
    )
    origin = pd.Timestamp("2020-01-01")
    return InventoryInputs(
        catalog,
        pd.date_range(origin + pd.Timedelta(days=1), periods=h),
        origin,
        decision_days,
        d,
        np.full(n, 6) if initial is None else np.array(initial),
        np.zeros((n, h), dtype=int),
        {"S": capacity},
    )


def uniform(rate, n=2, h=20):
    return [[rate] * h for _ in range(n)]


def test_review_days_cover_the_decision_window():
    assert review_days(14, 7) == [0, 7]
    assert review_days(15, 7) == [0, 7, 14]
    with pytest.raises(ValueError):
        review_days(14, 0)


def test_state_ignores_sales_on_and_after_the_review_day():
    inputs = scenario(uniform(3.0), decision_days=14)
    actual = np.array(uniform(2.0))
    committed = np.zeros((2, 14), dtype=int)
    committed[:, 0] = 10
    changed = actual.copy()
    changed[:, 7:] = 50
    first = state_at_review(inputs, actual, committed, 7)
    second = state_at_review(inputs, changed, committed, 7)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_committed_orders_are_never_affected_by_later_sales():
    inputs = scenario(uniform(3.0), decision_days=14)
    actual = np.array(uniform(2.0))
    planner = order_up_to_review_planner(None)
    base, _ = rolling_schedule(inputs, inputs.demand, actual, planner, period=7)
    changed = actual.copy()
    changed[:, 7:] = 0.0
    other, _ = rolling_schedule(inputs, inputs.demand, changed, planner, period=7)
    # Week one is decided before any of the changed days are observed.
    np.testing.assert_array_equal(base[:, :7], other[:, :7])


def test_review_state_hands_off_realized_stock_and_committed_pipeline():
    inputs = scenario(uniform(3.0), decision_days=14)
    actual = np.array(uniform(1.0))
    seen = {}

    def planner(sub, review):
        seen[review] = sub
        orders = np.zeros((sub.n, sub.decision_days), dtype=int)
        if review == 0:
            orders[:, 6] = 9  # Placed in week one, arrives on day 8.
        return orders, {"status": "test"}

    committed, reviews = rolling_schedule(inputs, inputs.demand, actual, planner, period=7)
    assert [r["review_day"] for r in reviews] == [0, 7]
    assert reviews[1]["review_date"] == str(inputs.dates[7].date())
    later = seen[7]
    assert later.decision_days == 7 and later.h == inputs.h - 7
    assert later.origin == inputs.origin + pd.Timedelta(days=7)
    np.testing.assert_array_equal(later.demand, inputs.demand[:, 7:])
    # Six units on hand, one sold on each of days 0..6: none left by day 6.
    np.testing.assert_array_equal(later.initial, [0, 0])
    assert later.receipts[:, 1].tolist() == [9, 9]
    assert committed[:, 6].tolist() == [9, 9]


def test_rolling_order_up_to_orders_less_when_demand_falls_short_of_forecast():
    inputs = scenario(uniform(4.0), decision_days=14, capacity=60.0)
    actual = np.array(uniform(1.0))
    frozen = order_up_to(inputs, None)
    rolling, _ = rolling_schedule(
        inputs, inputs.demand, actual, order_up_to_review_planner(None), period=7
    )
    assert rolling.sum() < frozen.sum()
    frozen_peak = inventory_trace(inputs.with_demand(actual), frozen)["inventory"].sum(0).max()
    rolling_peak = inventory_trace(inputs.with_demand(actual), rolling)["inventory"].sum(0).max()
    assert rolling_peak < frozen_peak


def test_run_phase_adds_rolling_policies_only_with_realized_demand(tmp_path):
    inputs = scenario(uniform(3.0), decision_days=14, capacity=60.0)
    actual = np.array(uniform(2.0))
    config = OptimizationConfig(time_limit_seconds=5, rolling_review_days=7)
    metric, timing = run_phase(inputs, {}, config, tmp_path / "holdout", actual=actual)
    assert set(metric.policy) == {MILP, MILP + "_rolling"}
    assert timing["rolling_review_days"] == 7
    reviews = json.loads((tmp_path / "holdout" / "rolling_reviews.json").read_text())
    assert [r["review_day"] for r in reviews[MILP + "_rolling"]] == [0, 7]
    schedules = np.load(tmp_path / "holdout" / "order_schedules.npz")
    assert schedules[MILP + "_rolling"].shape == (2, 14)
    frozen, _ = run_phase(inputs, {}, config, tmp_path / "latest")
    assert set(frozen.policy) == {MILP}


def test_weekly_baseline_orders_only_on_review_days_and_batches(tmp_path):
    from supply_chain_opt_engine.optimization import UPPER, WEEKLY

    inputs = scenario(uniform(3.0, h=42), decision_days=28, lead=2, initial=[2, 2])
    daily = order_up_to(inputs, None, review_period=1)
    weekly = order_up_to(inputs, None, review_period=7)
    ordering_days = set(np.flatnonzero(weekly.any(axis=0)))
    assert ordering_days <= {0, 7, 14, 21, 27}
    assert (weekly > 0).sum() < (daily > 0).sum()
    config = OptimizationConfig(time_limit_seconds=5, rolling_review_days=7)
    demands = {UPPER: inputs.demand, WEEKLY: inputs.demand}
    metric, _ = run_phase(
        inputs, demands, config, tmp_path / "holdout", actual=np.array(uniform(2.0, h=42))
    )
    assert {WEEKLY, WEEKLY + "_rolling"} <= set(metric.policy)
    schedules = np.load(tmp_path / "holdout" / "order_schedules.npz")
    np.testing.assert_array_equal(schedules[WEEKLY], weekly)
    rolled = set(np.flatnonzero(schedules[WEEKLY + "_rolling"].any(axis=0)))
    assert rolled <= {0, 7, 14, 21, 27}


def test_rolling_review_period_is_validated():
    with pytest.raises(ValueError):
        OptimizationConfig(rolling_review_days=-1)
    with pytest.raises(ValueError):
        OptimizationConfig(rolling_review_days=29)
