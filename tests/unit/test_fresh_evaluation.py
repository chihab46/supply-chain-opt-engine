from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from supply_chain_opt_engine.fresh_evaluation import (
    fresh_folds,
    optimization_config,
    primary_comparison,
    verdict,
)

AFTER = date(2016, 1, 10)
LAST = date(2016, 5, 22)


def test_preregistered_windows_are_valid_and_non_overlapping():
    document = yaml.safe_load(Path("configs/default.yaml").read_text())
    fresh = document["fresh_evaluation"]
    tail = optimization_config(document).tail_days
    folds = fresh_folds(
        fresh["origins"], fresh["first_fold_id"], 28, tail, after=AFTER, last_observed=LAST
    )
    assert [f.fold_id for f in folds] == [5, 6, 7]
    assert {f.role for f in folds} == {"fresh"}
    assert [str(f.validation_end) for f in folds] == ["2016-02-07", "2016-03-20", "2016-05-01"]


@pytest.mark.parametrize(
    "origins, message",
    [
        (["2016-01-09"], "before earlier evaluation"),
        (["2016-01-10", "2016-02-20"], "must not overlap"),
        (["2016-04-11"], "past the observed"),
        (["2016-02-21", "2016-01-10"], "chronologically"),
    ],
)
def test_fresh_windows_reject_leaky_or_invalid_schedules(origins, message):
    with pytest.raises(ValueError, match=message):
        fresh_folds(origins, 5, 28, 14, after=AFTER, last_observed=LAST)


def metrics(policy_cost, baseline_cost, policy_ok=True, baseline_ok=True):
    rows = []
    for policy, cost, ok in [
        ("milp", policy_cost, policy_ok),
        ("base", baseline_cost, baseline_ok),
    ]:
        for scope in ["decision_window", "including_tail"]:
            rows.append(
                {
                    "policy": policy,
                    "evaluation": "realized",
                    "scope": scope,
                    "total_cost": cost if scope == "including_tail" else 1.0,
                    "fill_rate": 0.95,
                    "minimum_store_fill_rate": 0.93,
                    "comparison_eligible": ok,
                }
            )
    return pd.DataFrame(rows)


def test_primary_comparison_uses_the_full_window_and_requires_both_eligible():
    row = primary_comparison(metrics(70.0, 100.0), "milp", "base", "F1")
    assert row["saving"] == pytest.approx(0.30)
    assert row["both_eligible"]
    row = primary_comparison(metrics(70.0, 100.0, baseline_ok=False), "milp", "base", "F1")
    assert np.isnan(row["saving"]) and not row["both_eligible"] and row["policy_eligible"]


def test_verdict_follows_the_preregistered_rule():
    rows = [
        primary_comparison(metrics(70.0, 100.0), "milp", "base", "F1"),
        primary_comparison(metrics(120.0, 100.0), "milp", "base", "F2"),
        primary_comparison(metrics(50.0, 100.0, baseline_ok=False), "milp", "base", "F3"),
    ]
    outcome = verdict(pd.DataFrame(rows))
    # Pooled over windows where both are eligible: 1 - 190 / 200.
    assert outcome["pooled_saving"] == pytest.approx(0.05)
    assert outcome["windows_both_eligible"] == 2
    assert outcome["supports_holdout_finding"]
    rows[1] = primary_comparison(metrics(80.0, 100.0, policy_ok=False), "milp", "base", "F2")
    outcome = verdict(pd.DataFrame(rows))
    assert not outcome["policy_eligible_all_windows"]
    assert not outcome["supports_holdout_finding"]
