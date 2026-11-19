"""Rolling-horizon replanning with realized-state feedback for holdout replay.

A frozen policy fixes every order at the origin. A rolling policy reviews the
plan every ``period`` days: it observes realized on-hand stock at the end of the
previous day, keeps every order already committed, re-solves the remaining
horizon and commits only the orders up to the next review.

Information rules:

- Realized sales enter only through the on-hand state, and only for days before
  the review. Later realized sales are masked before the state is computed.
- Planning demand is the policy's origin forecast for the remaining days. The
  forecast is not refreshed at reviews, so this isolates state feedback.
- Orders placed in earlier weeks stay fixed and arrive as pipeline receipts.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from supply_chain_opt_engine.simulation import inventory_trace


def review_days(decision_days: int, period: int) -> list[int]:
    if not isinstance(period, int) or period < 1:
        raise ValueError("Review period must be a positive integer number of days.")
    return list(range(0, decision_days, period))


def state_at_review(inputs, actual, committed, review):
    """On-hand stock at the end of day ``review - 1`` and receipts from ``review``.

    Realized demand on or after the review day is zeroed first, so the state
    cannot depend on sales that were not yet observed.
    """
    actual = np.asarray(actual, dtype=float)
    if actual.shape != (inputs.n, inputs.h):
        raise ValueError("Realized demand must match the scenario shape.")
    observed = np.where(np.arange(inputs.h)[None, :] < review, actual, 0.0)
    trace = inventory_trace(inputs.with_demand(observed), committed)
    initial = inputs.initial.astype(float) if review == 0 else trace["inventory"][:, review - 1]
    # Arrivals combine the original pipeline with already committed orders.
    return np.rint(initial), np.rint(trace["arrivals"][:, review:])


def subproblem(inputs, forecast, initial, receipts, review):
    """The scenario seen at a review: remaining days, decisions and state."""
    return replace(
        inputs,
        dates=inputs.dates[review:],
        origin=pd.Timestamp(inputs.origin) + pd.Timedelta(days=review),
        decision_days=inputs.decision_days - review,
        demand=np.asarray(forecast, dtype=float)[:, review:],
        initial=initial,
        receipts=receipts,
    )


def rolling_schedule(inputs, forecast, actual, planner, *, period=7):
    """Replan at each review and return committed orders plus review records.

    ``planner(sub, review)`` returns ``(orders, info)`` where ``orders`` covers
    the subproblem's remaining decision days.
    """
    committed = np.zeros((inputs.n, inputs.decision_days), dtype=int)
    reviews = []
    for review in review_days(inputs.decision_days, period):
        initial, receipts = state_at_review(inputs, actual, committed, review)
        sub = subproblem(inputs, forecast, initial, receipts, review)
        plan, info = planner(sub, review)
        plan = np.asarray(plan)
        if plan.shape != (inputs.n, sub.decision_days):
            raise ValueError("Rolling planner returned a schedule with the wrong shape.")
        end = min(review + period, inputs.decision_days)
        committed[:, review:end] = plan[:, : end - review]
        reviews.append(
            {
                "review_day": review,
                "review_date": str(sub.dates[0].date()),
                "committed_order_units": int(plan[:, : end - review].sum()),
                "starting_units": float(initial.sum()),
                **info,
            }
        )
    return committed, reviews
