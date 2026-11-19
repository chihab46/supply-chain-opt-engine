# Storage headroom: exploratory analysis

**Status: exploratory.** Every window here had already been inspected, and the
grid was designed after the [pre-registered fresh test](fresh_evaluation.md)
failed. Nothing in this document is an independent test, and no setting chosen
from it should be reported as validated.

## Question

The fresh test failed because every P90 policy exceeded store storage when
demand came in below forecast. Does planning against only part of each store's
capacity fix that, and does the optimizer keep its cost advantage when it does?

## Method

- **Remedy tested:** `planning_capacity_share` in `OptimizationConfig`. Every
  planner (the MILP and all order-up-to rules, frozen and rolling) plans
  against `share × capacity`; replay and every storage check use the full
  capacity. The default, 1.0, is the original behaviour.
- **Grid:** shares 1.0, 0.95, 0.90, 0.85 and 0.80 on the December 2015 holdout
  and the three 2016 windows F1–F3, with weekly rolling replans as before.
  Every cell is reported.
- **Same inputs:** the fresh windows reuse the registered run's checksummed
  forecasts; each rebuilt scenario was checked byte for byte against the
  registered one. The holdout reruns the standard optimization.
- **Comparison:** rolling MILP against rolling weekly-review P90
  order-up-to, realized 42-day cost, saving defined only when both meet the 90%
  store fill target and stay within storage.
- **Runs:** `artifacts/metrics/storage_headroom/20261119T180619Z-3f9332dc` (all
  shares) and `20261119T184112Z-613944f3` (0.85 and 0.80 after the fallback
  change below). Command:
  `uv run python -m supply_chain_opt_engine.storage_headroom`.

## Finding 1: the plans were never at the limit

On the P90 forecast the rolling MILP's end-of-day stock peaks at **84%** of
capacity and the weekly baseline's at **62–66%**. Realized peaks reach
**103–108%** for both. A buffer only changes a plan if the plan reaches it, so
shares of 0.85 and above did not bind:

| Share | MILP max overflow, F1 / F2 / F3 | Baseline max overflow | MILP eligible | Holdout saving |
| --- | --- | --- | --- | --- |
| 1.00 | 17.9 / 14.2 / 18.7 m³ | 12.3 / 13.4 / 5.8 m³ | none | 29.8% |
| 0.95 | 17.9 / 14.1 / 18.7 m³ | same | none | 29.8% |
| 0.90 | 17.9 / 14.1 / 18.7 m³ | same | none | 29.8% |
| 0.85 | 17.9 / 14.1 / 18.8 m³ | same | none | 29.8% |
| 0.80 | 17.7 / 13.7 / 18.0 m³ | same | none | 29.8% |

The weekly baseline's results were identical to the cent at every share; its
plans never come near 80%. Rows 0.85 and 0.80 are from the second run.

This corrects the fresh-test write-up, which first said the MILP planned
storage "to the limit". The overflow comes from **stock ordered for P90 demand
that does not sell**, which both policies do. The MILP holds more stock (84%
against 62–66% planned), so it overflows sooner and also inside the decision
window.

## Finding 2: a binding buffer only trades overflow for stockouts

At 0.80 the buffer starts to bind, but only after a low-demand week has already
left stock above 80%. The weekly review then has no feasible plan:

- **First run, original fallback: order nothing for the whole store.** Storage
  was respected in every window, but fill collapsed. The MILP's 42-day cost rose
  to $29,866, $25,958 and $43,965 (from $16,255, $15,873 and $18,142), with the
  lowest store at 90.1%, 90.3% and **74.2%**. It was eligible in F1 and F2 only
  because it barely cleared 90%, and it then cost more than the baseline.
- **Second run, new fallback: plan around stock already committed.** When
  committed stock alone exceeds a store's planning limit, that store's limit is
  lifted to the committed peak and planning continues (status
  `capacity_raised_to_committed_stock`). Service and cost recover, and the
  overflow returns (table above, row 0.80).

Neither is a fix. The excess is created by orders already placed against P90
demand, so capping the planning target afterwards cannot remove it.

## Changes kept in the code

- `planning_capacity_share` stays available, default 1.0, so the original
  behaviour and every registered result are unchanged.
- The rolling planners no longer order nothing when committed stock exceeds
  capacity; they lift that store's limit to its committed peak first, then relax
  the service floor, and only then order nothing. In the original grid this
  path was reached only when the buffer was 0.85 or below.
- Older saved runs that predate a setting now match it at its default in the
  dashboard, so adding settings does not hide saved references.

## Solver variability

The shares that never bind give slightly different results from each other
and from the registered run (F2 MILP: $15,873 and 24.6 m³ registered, $16,847
and 14.2 m³ here). These are time-limited solves with 31–55% gaps, run four at a
time, so plans vary with timing. The conclusions do not depend on these
differences.

## What would address the mechanism

Untested here; each would again be exploratory on these windows:

- **Order less for the tail.** The last review sizes orders for the 14-day
  no-order tail at P90 and is where the baseline's overflow occurs. Planning the
  tail on P50, or allowing reviews to continue into it, targets that directly.
- **Plan storage on a lower demand scenario.** Require stock to fit capacity if
  demand comes in at P50 (or on sampled demand paths), while service is still
  planned on P90. This is a second inventory trajectory in the MILP.
- **Price storage instead of capping it.** An overflow cost per m³-day lets the
  optimizer trade overflow risk against stockouts rather than treating capacity
  as a hard line on one forecast.
