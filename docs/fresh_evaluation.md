# Fresh-period evaluation protocol

**Status: pre-registered on 2026-11-19, before any forecast, inventory plan or
score was computed on the windows below.** Results are reported in a separate
section once the run completes. Nothing in this protocol may change after that
run; any later variation is labelled exploratory.

## Why

The stockout penalty, service floor, rolling replan and weekly-review baseline
were all chosen after the December 2015 holdout (fold 4) had been inspected.
That holdout can no longer test them independently. M5 sales after the holdout
comparison window are still unused.

## Windows

Three non-overlapping 42-day windows: 28 order-decision days plus the shared
14-day no-order tail. They start after the last day used by any earlier
evaluation (2016-01-10, the end of the fold-4 tail) and end before the last
observed day (2016-05-22).

| Window | Origin (last observed day) | Decision days | Tail ends |
| --- | --- | --- | --- |
| F1 | 2016-01-10 | 2016-01-11 to 2016-02-07 | 2016-02-21 |
| F2 | 2016-02-21 | 2016-02-22 to 2016-03-20 | 2016-04-03 |
| F3 | 2016-04-03 | 2016-04-04 to 2016-05-01 | 2016-05-15 |

## Frozen inputs

Every input is pinned and checksummed in the run manifest. None is refitted or
re-tuned on fresh data except where noted.

| Input | Pinned value |
| --- | --- |
| Ingestion (series, sales, economics) | `20261119T152927Z-596696fe` |
| LightGBM hyperparameters and feature schema | `20261119T153241Z-d725af75` (tuned on folds 1–3 only) |
| Reconciliation settings | `20261119T154145Z-2da2199f` (MinT-shrink, 501 paths) |
| Statistical baseline settings | `20261119T154311Z-cf01203f` |
| Inventory settings | `configs/default.yaml` `optimization` block as committed with this protocol |

At each fresh origin, models are **refitted with the frozen settings on data up
to that origin only**, exactly as the fold-4 holdout was: six LightGBM quantile
models, AutoETS and SeasonalNaive, and the reconciliation's aggregate models and
residual covariance. Scenario state (starting stock, pipeline, capacity) comes
from the 56 days ending at each origin. Realized sales enter only the rolling
replan's on-hand state, and only for days before each weekly review.

## Primary comparison

- **Policy:** `lightgbm_mint_milp_rolling` (MinT P90 MILP, 90% store fill floor,
  weekly replanning on realized stock).
- **Baseline:** `lightgbm_p90_weekly_order_up_to_rolling` (the same P90 forecast,
  weekly-review order-up-to, weekly replanning).
- **Metric:** realized total cost over the full 42 days (holding + stockout
  penalty + fixed order fees).
- **Eligibility:** a policy is eligible in a window only if every store reaches
  at least 90% realized fill and storage is never exceeded over the 42 days.
  A saving is defined only when both policies are eligible in that window.

## Success criterion

The fresh-period result **supports** the holdout finding if the MILP is eligible
in all three windows **and** its pooled 42-day cost (sum over windows where both
policies are eligible) is lower than the baseline's. Otherwise it does **not**
support it. Either way, the per-window savings, the pooled saving and every
eligibility failure are reported as they come out.

## Secondary results

Reported, but not part of the success criterion:

- per-window savings, fill rates and lowest-store fill for every policy;
- frozen versus rolling storage overflow;
- the daily-review P90 rule and the statistical baselines;
- leaf WMAPE and P05–P95 coverage for LightGBM, MinT and AutoETS;
- MILP optimality gaps and run times.

## Run

```bash
uv run python -m supply_chain_opt_engine.fresh_evaluation
```

Outputs go to `artifacts/metrics/fresh_evaluation/<run_id>/`, with one inventory
bundle per window and a manifest that pins every input above and this file's
checksum.

## Results

Run `20261119T164107Z-8bcce3f6`, started 2026-11-19 16:41 UTC, after the
protocol commit `6df56e5` (16:40 UTC). No setting was changed. The
protocol section above is unchanged; only this section was added.

### Verdict: does not support the holdout finding

The MILP was not eligible in any window, and neither was the weekly baseline,
so no window has a defined saving and the pooled saving is undefined. Both
policies met the 90% store fill target everywhere; **both failed storage.**

| Window | MILP fill (lowest store) | Baseline fill (lowest store) | MILP max overflow | Baseline max overflow | MILP cost | Baseline cost |
| --- | --- | --- | --- | --- | --- | --- |
| F1 | 98.69% (98.35%) | 98.51% (98.08%) | 17.9 m³ | 12.3 m³ | $16,255 | $23,580 |
| F2 | 98.59% (97.68%) | 98.37% (97.57%) | 24.6 m³ | 13.4 m³ | $15,873 | $23,794 |
| F3 | 98.68% (98.23%) | 98.67% (98.27%) | 18.7 m³ | 5.8 m³ | $18,142 | $25,194 |

Store capacities are 152–231 m³. Costs are realized 42-day totals for the
rolling versions of both policies.

### Why both failed

- **The no-order tail.** The last weekly review (day 21) sizes orders to cover
  the 14-day tail at the P90 forecast. When actual tail demand is lower, stock
  accumulates with no review left to stop it. This explains every baseline
  overflow, all on tail days (F1 Feb 12–13, F2 Mar 25–26, F3 May 6).
- **The MILP leaves no headroom.** It also overflows **inside** the 28-day
  decision window in every window (2.4, 9.1 and 1.9 m³; F1 Jan 16, F2 Feb 27
  and Mar 18–20, F3 Apr 9), where the weekly baseline stays within storage. The
  MILP plans end-of-day stock up to the storage limit on the P90 forecast, so a
  week of below-forecast demand is enough to push it over. The baseline's
  simpler rule happens to leave slack.

The December 2015 holdout did not show this: with rolling reviews, both
policies stayed within storage there. That result did not generalize.

### Secondary results

These do not change the verdict.

- **Forecasting replicates.** Leaf WMAPE for LightGBM versus AutoETS was 48.5%
  vs 51.8% (F1), 52.1% vs 55.9% (F2) and 48.7% vs 52.9% (F3), a 6–8% relative
  reduction, compared with 9% on the holdout. LightGBM's P05–P95 coverage was
  87.9–89.8%, against 64.8–66.9% for AutoETS. MinT reconciliation again cost a
  little accuracy (49.5%, 52.5%, 49.3% WMAPE) and coverage (84.7–87.2%).
- **Cost gap, descriptive only.** Ignoring the storage failures, the MILP's
  42-day cost was 31%, 33% and 28% below the baseline's (31% pooled), at equal or
  higher fill. Because both broke storage, this is not a valid saving.
- **Other policies.** Rolling AutoETS with safety stock was the **only policy
  eligible in all three windows** (lowest store fill 91.0–91.6%, within
  storage), at $61,308, $62,562 and $66,855: 3.7–3.9 times the MILP's cost. Its
  lower fill left the slack the P90 policies lacked. P50 and SeasonalNaive rules
  never met the floor. Every frozen P90 policy overflowed by 142–166 m³.
- **Solver.** Every review found a plan; frozen-plan gaps were 31–41%. Each
  window took 9–10 minutes end to end.
- **Not run.** `scripts/validate_inventory_run.py` reads the optimization
  runner's manifest format and was not applied to these bundles. The schedules
  were still validated by `run_phase`'s physical replay.

### What this means

Under the economics used here, the optimizer's cost advantage is real on
paper but was bought partly by running storage at the limit. A deployable
version needs storage headroom against forecast error, for example a capacity
buffer, a storage constraint on a higher demand quantile or on sampled demand
paths, or reviews that continue into the tail. All of the M5 data is now used,
so any such fix can only be evaluated on these windows and would be
exploratory, not another independent test.
