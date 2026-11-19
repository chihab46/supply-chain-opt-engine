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
