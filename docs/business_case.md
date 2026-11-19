# Inventory planning business brief

## Bottom line

On a December 2015 holdout of 600 store-item series, a forecast-driven
optimizer that replans weekly cost 30% less than a weekly-review reorder rule
at a higher fill rate (98.3% versus 97.8%), with both inside storage and above
a 90% per-store fill target.

**That saving did not hold up in a clean test.** On three unused 2016 windows,
evaluated under a protocol fixed before the run, both the optimizer and the
reorder rule exceeded store storage in every window, so no saving can be
claimed. The optimizer stayed 28–33% cheaper on paper, but it overflowed even
within the planning window: it fills storage to the limit on a high forecast
and has no headroom when demand comes in lower. The forecasts themselves did
replicate: LightGBM beat the statistical benchmark again on all three windows.

The next step is to plan with storage headroom against forecast error, then
validate on genuinely new data with real cost inputs. Costs here are synthetic,
and the optimizer's plans are good but not proven optimal. See the
[fresh-period results](fresh_evaluation.md#results).

## Decision and user

A retail replenishment planner needs to decide which products to order, in what
quantities and on which days. The planning problem balances availability against
ordering fees and inventory carrying costs, subject to lead times, minimum order
quantities and finite store storage. This prototype covers 600 store-item series
in three California stores using the public Walmart M5 sales dataset.

## What the application does

The forecasting pipeline estimates demand and uncertainty over 28 days. The
planner selects a demand scenario and a minimum store fill requirement, then
compares replenishment schedules under common inventory, receipts, capacity and
economics. Orders stop after day 28, with a shared 14-day tail included to expose
late stockouts and excess inventory. On historical periods, each policy can also
be replanned weekly from the stock actually on hand, as a planner would. The
dashboard supports forecast drill-down, cost and service comparisons and
validated purchase-order downloads.

The service requirement applies to the proportion of forecast units fulfilled in
each store over both windows. It is not a guarantee for each SKU, each day or
actual future demand. Starting stock, costs, lead times and capacity are synthetic
assumptions, not observed Walmart operations.

## What the evidence establishes

**Forecasting.** On the frozen holdout, LightGBM has 59.20% WMAPE versus 65.04%
for AutoETS, an 8.99% relative reduction, and its central 90% intervals cover
89.55% of observations. Errors are high because most series are daily,
store-level and intermittent. Hierarchical reconciliation makes store and
category totals consistent with item forecasts, but worsens item-level holdout
WMAPE to 61.08%: a measured tradeoff for consistency.

**Inventory: three findings, in the order they were found.**

1. *The first objective was wrong.* A cost-only optimizer placed no orders and
   served 23% of holdout demand. The cause was the synthetic stockout penalty,
   worth only 3–8 days of holding cost per lost unit, when a lost sale forfeits
   at least its gross margin. The penalty is now lost margin plus 5–20% of price
   as goodwill. Under it, high service is also the cheapest plan: 0%, 80%, 90%
   and 95% service floors produce the same schedule.
2. *Plans fixed on day one break storage.* Planned once on the P90 forecast, the
   optimizer reached 98.6% fill on actual sales but exceeded storage by up to
   134 m³, because actual demand came in below forecast. Every frozen P90 rule
   failed the same way. Replanning weekly from realized stock removed the
   overflow for every policy, at a cost of 0.3 to 0.9 points of fill.
3. *A weak baseline overstates the gain.* Against a daily-review reorder rule,
   which pays an ordering fee nearly every day, the optimizer looked 69% cheaper.
   Against a weekly-review rule on the same forecast, the fair comparison, the
   gain is 30%.

Results over all 42 days of actual sales, every policy replanned weekly:

| Policy | Fill rate | Lowest store | Orders | Total cost | Meets target and storage |
| --- | --- | --- | --- | --- | --- |
| **Optimizer (MILP)** | **98.33%** | **97.97%** | **1,582** | **$17,305** | **Yes** |
| Reorder rule, weekly review | 97.83% | 97.06% | 2,587 | $24,659 | Yes |
| Reorder rule, daily review | 97.62% | 96.90% | 7,063 | $56,255 | Yes |
| Statistical forecast + safety stock | 87.42% | 86.69% | 6,435 | $68,704 | No (fill) |

Against the weekly rule, the optimizer saves $7,222 in ordering fees and $940 in
lost-sale penalties, and spends $808 more on holding stock. Savings are only
compared between policies that meet the service target in every store and never
exceed storage, over the same window. Purchase expenditure is excluded from the
objective, and all costs are synthetic.

The optimizer's solver reports gaps of 34–55% at the first three weekly reviews
(under 1% at the last), so better plans may exist. A holdout run takes about
five minutes, well above the brief's 1–2 second interactive target. The
latest-origin plan meets its projected service and storage constraints (98.9%
projected fill); it has no observed future result.

## How to assess a scenario

1. Check that a feasible plan exists and inspect its optimality gap. A large gap
   means a better plan may exist; a time limit alone does not prove
   infeasibility.
2. Inspect projected and realized service separately, including the lowest store
   fill rate. Evaluate the complete window, not just the first 28 days.
3. Reject cost comparisons where either policy fails storage or the declared
   service target. A cheaper policy with lower service is a tradeoff, not a
   saving at the same service standard.
4. Compare against the strongest reasonable baseline, with the same information
   and the same replanning rhythm. A weak or frozen baseline inflates the gain.
5. Compare service targets while holding economics and capacity fixed. Treat
   penalty or capacity changes as separate scenarios. The dashboard does this
   filtering for saved service-target points.

## What is needed before a real deployment

- **Storage headroom, then new data.** The pre-registered 2016 test showed that
  both plans overflow storage when demand comes in below the P90 forecast. Plan
  storage against forecast error (a capacity buffer, a higher-quantile or
  scenario-based storage constraint, or reviews that continue into the tail),
  then test on data not yet seen. All M5 data is now used, so that test needs
  new data, ideally a real retailer's.
- **Real economics.** Agree the service requirement with an operations owner,
  then obtain actual stock, supplier calendars, lead-time distributions,
  ordering costs, storage constraints and the business value of a lost sale.
- **Demand that sales hide.** Historical sales understate demand when items were
  out of stock; this prototype does not observe those lost sales.
- **Fresher forecasts and faster solves.** Weekly replans currently reuse the
  original forecast, and each review takes about a minute (20 seconds per
  store).
- **Operational limits.** Check purchase budgets, supplier limits and real
  ordering cadence, then run a shadow-planning pilot with human review.

See the [README](../README.md#headline-result) for the full results and
[inventory definitions](optimization.md) for the model, rolling rules and
commands. [Benchmark evidence](benchmarks/README.md) still holds the
pre-revision runs.
