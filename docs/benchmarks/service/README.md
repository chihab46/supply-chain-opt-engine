# Service and cost sensitivity evidence

Run `20261112T104754Z-c33775d5` evaluated four declared service targets on the
same 600 store-item holdout series. All scenarios use reconciled P90 demand,
unchanged synthetic penalties, baseline capacity and the shared 14-day no-order
tail. Targets are assumptions for exploratory policy redesign after the old
holdout was reviewed, not newly independent test results or tuned deployment
settings.

All four MILP plans satisfy their forecast-input service and storage constraints.
The complete **realized 42-day** results are:

| Forecast service floor | Actual aggregate fill | Actual lowest store fill | Scenario cost | Storage feasible | Actual service floor met |
| --- | --- | --- | --- | --- | --- |
| 0% | 15.66% | 15.03% | $810.73 | Yes | Yes, zero floor |
| 80% | 79.59% | 79.02% | $10,770.25 | No | No |
| 90% | 90.30% | 89.53% | $13,384.73 | No | No |
| 95% | 95.85% | 95.49% | $14,674.42 | No | Yes |

Costs on storage-infeasible replays are diagnostics, not feasible operating-policy
results or savings claims. The 90% scenario exceeds end-of-day capacity by up to
80.72 m³. Its aggregate fill exceeds 90%, but one store misses the requirement.
These results establish that a projected service floor alone does not establish
a deployable replenishment policy. Robust storage planning and fresh-period
validation remain work for a future business-approved specification.

The cost-only control has a combined certified gap of 0.25%. Service-constrained
incumbents have gaps of 56.44% to 60.36%; the configured 1% goal was not reached.
Their phase runtimes are 79–85 seconds, including model construction, forecast-only
incumbent generation and policy replay around the configured solver budget.
Do not describe these plans as optimal or promise a one-second interaction.

The separate latest-origin run `20261112T105218Z-1b0faa3e` meets projected service
and capacity in both windows. Its lowest store has 90.001% projected fill over
28 days and 90.000% over 42 days. These are projected values; future actual
fulfillment is unavailable.

The stress scenario in `infeasible/` requests 100% service, 75% capacity and a
10x shortage penalty. HiGHS proves the first store infeasible, and the failed
bundle has no purchase-order export. A synthetic unit test separately checks
that a time limit without an incumbent is retained as unproven, rather than
misclassified as infeasible.

## Files

| File | Evidence |
| --- | --- |
| `manifest.json` | Original sensitivity manifest with pinned sources, settings and hashes |
| `experiments.json` | All four configurations and optimization bundle references |
| `tradeoff_metrics.csv` | Every policy's common-forecast and realized scores for both windows |
| `independent_validation.json` | Independent row-level replay of all four optimization bundles |
| `latest_run.json`, `latest_metrics.csv`, `latest_validation.json` | Latest-origin plan provenance, projected KPIs and replay checks |
| `infeasible/` | Original stress-test manifest, failure diagnostics and empty metrics schema |

Full traces, schedules and scenario inputs stay under the artifact paths in the
manifests. Independent validation recomputes physical flow, costs, store service,
capacity, export quantities and eligibility for cost comparisons. It also checks
that forecasts and actual demand are shared across policies and that synthetic
penalty multipliers match the saved economics. All five successful bundles passed.

Reproduce the successful grid with:

```bash
uv run python -m supply_chain_opt_engine.inventory_tradeoff --phase holdout \
  --minimum-fill-rates 0 0.8 0.9 0.95
```

The sources and assumptions are reproducible; time-limited incumbent schedules
can vary with hardware and timing. See [the inventory model](../../optimization.md)
and [business brief](../../business_case.md) before interpreting results.
