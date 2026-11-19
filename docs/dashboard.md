# Dash planning workbench

Step 7 provides a local, two-tab application backed by the validated forecast and
inventory artifacts. It does not query PostgreSQL or fit forecasting models when
users change chart selections.

```bash
uv sync --extra dashboard
uv run --extra dashboard python -m app.dashboard
```

Open [localhost:8050](http://127.0.0.1:8050). Start from the repository root.
`DASH_HOST`, `DASH_PORT` and `PROJECT_CONFIG` can be set in `.env`; command-line
`--host`, `--port` and `--root` override the corresponding launch defaults. Debug
mode and the development reloader are disabled. The default host is localhost.

Complete pipeline steps 1 through 6 first. With missing or invalid source data,
the app shows an actionable empty state rather than fabricated sample results.
Restart the application after producing a new forecast snapshot.

## Demand forecast

The explorer opens on the latest saved origin. All four historical folds are
also available. Select a store, then a category, then an item. Selecting a higher
level resets dependent controls. The selections map to actual saved Total,
Store, Store-Category and Store-Item nodes; aggregate quantiles are never created
by adding leaf quantiles.

The chart shows 42 days of history, a 28-day coherent point forecast, and P10–P95
and P10–P90 uncertainty ranges. Historical selections also show their realized
validation demand. These later observations are displayed for evaluation and
never used by the planning inputs. Latest-origin forecasts have no future actuals.

P10–P95 has nominal coverage of 85%. The coherent center is distinct from the
marginal median. The daily table includes both center and P50 alongside P10,
P90 and P95. The footer identifies the forecast origin, node and reconciliation
run.

SHAP bars show model-wide mean absolute SHAP values from the saved production
or holdout P50 model. They explain associations, not causal effects or an
individual item's particular forecast. These are upstream global-model explanations;
they do not attribute the later MinT adjustment. Earlier tuning folds explicitly show
that no SHAP sample was saved; they do not inherit a later model's explanation.

## Replenishment simulator

Choose the historical holdout or the latest-origin scenario, then set:

- Demand quantile from the six available forecasts, P05, P10, P50, P80, P90, P95.
- Store capacity from 10% to 150% of the common synthetic baseline.
- Supplier lead-time delay from zero to four days.
- Minimum store fill rate from 0% to 100%, separate from the demand quantile.
- A positive multiplier on the synthetic per-unit shortage penalty.

The project default is a 90% minimum store fill target with unmodified synthetic
penalties. The model enforces this on forecast demand for every store, over both
28 days and the complete window. Set the target to zero for cost-only planning.
It is not a promise about actual future service. Impossible service targets fail
without relaxing the requirement. A time limit with no incumbent is reported as
no feasible plan found, rather than proof of infeasibility.

The quantile slider uses six discrete positions. It never interpolates forecasts
that were not produced upstream. The UI accepts Dash's integer-valued float
slider events and validates each position before mapping it to a quantile.

Press **Run optimizer** to submit a background scenario. Status progresses through
pending, running, completed or failed. Controls and duplicate submissions are
disabled while the matching job is active. The demand forecast view remains usable.
A one-second poll reads persisted job status; it does not repeatedly redraw an
unchanged result.

Each result is tied to its exact phase, full configuration, upstream manifests
and optimizer implementation hashes. Identical requests reuse completed results
or the same active job. Validated reference runs can appear immediately with a
**Saved reference** label. Changing any scenario control clears the displayed
result and disables download until the new settings have a matching result.

The window selector switches between the 28-day decision window and the complete
window including the common 14-day tail. It changes reporting, not the solved
schedule. See [inventory definitions](optimization.md) for the objective,
lead-time convention, initial stock, pipeline, capacity and terminal treatment.

## Results and exports

The result opens with the **release decision**: "Ready to release" or "Hold: do
not release" with the reason in words, for example the largest storage excess in
m³ or the lowest store's fill against the target. Below it are the total cost,
fill rate with the lowest store, average inventory value, the order count with
its purchase value, and the purchase-order download.

**Storage headroom** plots each store's end-of-day volume as a share of its
capacity, with overflow days marked. It reads item volumes and store capacities
from the result's checksummed scenario files and says so when a result has none.
Further views show stock and receipts, the policy comparison, cost against
service, cost components and the solver gap.

Historical holdout results are labeled **Realized fill rate**. Latest-origin
results are labeled **Projected fulfillment** and explicitly describe a
forecast-demand replay. They do not claim an observed future service level.

The archived cost-only plan selects zero new orders and serves only 23.06% of
28-day holdout demand under the original synthetic economics. The new default
enforces a separate service floor. Zero-order results remain visible, and P90
demand alone still does not impose a minimum service constraint.
Comparison policies that exceed realized storage are marked. Their scores remain
diagnostic, and a primary result with a capacity failure cannot be downloaded
for the affected reporting window.

The policy table shows the lowest store fill rate and whether the service target
was met. The cost–service scatter plots those store minima against total scenario
cost, with a dotted service target. Each policy keeps one marker shape in the
chart and the table key; filled markers pass service and storage, open markers
fail one. Stars show saved MILP service targets only when the
remaining settings and forecast snapshot match. Changing penalty, capacity or
lead time does not mix incomparable scenarios. The command in
[inventory documentation](optimization.md) generates the service sensitivity grid.

All forecast comparisons now use one common selected demand scenario, even when
the policy used another forecast to plan. All holdout comparisons use the same
observed sales. Savings require both policies to meet service and capacity checks.
These are synthetic scenario economics; the screen makes no claim of measured
Walmart savings. Historical service experiments follow review of the old holdout
and require fresh-period validation before selecting a deployment policy.

CSV downloads contain only the MILP policy, with integer quantities, placement
and arrival dates, unit costs, purchase values, fixed fees and scenario IDs.
Zero new orders produce a valid CSV with headers and no data rows. Failed,
unverified and stale scenarios cannot be downloaded. A realized replay that misses
the store service target also disables download for that reporting window.

## Background execution and persistence

`dashboard_jobs.py` launches a separate Python worker process. Files under
`artifacts/dashboard/jobs/` persist requests, status and logs. Atomic writes and
file locks deduplicate requests across browser sessions. A shared worker lock
serializes optimizer execution on the local host. Waiting jobs stay pending and
do not occupy a Dash request worker. A stopped worker is detected and reported
as failed; the same scenario can then be retried.

The worker calls the existing inventory workflow with pinned reconciliation and
baseline manifests. Result bundles and synthetic inputs retain the Step 6
locations and audit format. Failed solves produce no downloadable result.
Starting or importing the application never starts an optimizer job. Jobs already
submitted continue independently of tab navigation or browser closure.

This queue targets the project's local Linux environment. It uses filesystem
locks and a single shared filesystem; it is not a distributed worker service.
The existing roughly 30–40 seconds of recorded solve time is shown honestly.
The app does not promise the brief's unachieved 1–2 second scenario target.

## Implementation and checks

- `app/dashboard.py`: layout, filters, callbacks, source-bound result selection and download.
- `app/presentation.py`: native Plotly charts, tables and result labels.
- `app/assets/`: responsive warehouse palette, locally served Lato fonts and hidden-tab chart resize handling.
- `dashboard_data.py`: pinned, checksummed forecast snapshots and validated result views.
- `dashboard_jobs.py`: persisted requests, deduplication, worker lifecycle and failure handling.
- `tests/unit/test_dashboard.py`: real Flask/Dash callback tests and worker-state fixtures.

The layout follows the complete forecast and simulator concepts generated for
this step. The implementation uses real saved values in place of illustrative
concept values. It also adds the explicit reporting-window selector, source
provenance and uncertainty/feasibility explanations needed to interpret the data.
Charts, controls, tables and text are native components, not screenshots.

The callback regression suite covers hierarchy resets, invalid selections,
source checksums, model-wide SHAP scope, stale-result protection, future-versus-
realized labeling, zero-order CSVs, duplicate jobs, active-control locking,
worker completion/failure, retry and stopped-worker detection. Whole-number
float slider events have a dedicated regression test.

## Service-control verification

The current suite passes 124 tests, with five live PostgreSQL integration tests
skipped because `TEST_DATABASE_URL` is unset. Lint, formatting and Python
compilation pass. Real-M5 Flask/Dash requests verify the default 90% scenario,
four saved service targets, visible service/storage failures and the tradeoff
chart response. Five full-size successful inventory bundles pass independent
row-level replay; a sixth stress case is proven infeasible and exports no orders.

Rendered visual verification of the new controls is incomplete. The Browser
plugin is unavailable, and regular Playwright cannot launch Chromium because
the environment blocks sockets. Starting a local Dash server is blocked by the
same restriction. No screenshot, desktop or mobile visual-pass claim is made
for this update. Callback tests do not substitute for rendered visual QA.

## Archived dashboard verification

The following records verification of the previous dashboard, before the new
service controls and cost–service chart.

The regression suite passed 109 tests, with five live PostgreSQL tests skipped
because `TEST_DATABASE_URL` was unset. Lint, formatting, Python compilation and
JavaScript syntax checks passed. A time-limited feasible result with no finite
lower bound reports an undefined certified gap rather than inventing zero gap.

The Chrome/Playwright flow passed 16 checks on the real M5 artifacts: page
identity, hierarchy drill-down, exact saved holdout KPIs, CSV download, projected
future labels, stale-result clearing, background completion, active-control
locking, use of the forecast tab during a solve, infeasible-scenario handling,
download rejection and mobile overflow. No application console errors remained.
A separate visual pass checked the actual Plotly canvas bounds and both tabs on
desktop and mobile. Screenshots were compared with both design concepts.

The browser tests ran through the normal host terminal because the agent's
sandbox blocked sockets and Chrome, and the connected browser host was
unavailable. This was rendered application verification, not a static layout
substitute. Chrome was tested; other browser engines remain untested.

The review checked navigation and copy, the green/white palette, locally served
humanist typography, chart/table proportions, tab resizing, and the mobile
stack. Material fixes included discrete keyboard slider steps, handling numeric
float events, removing legend crowding and fitting charts initialized in hidden
tabs. Real values and complete policy comparisons intentionally replace the
concepts' illustrative data. The extra reporting-window selector and explicit
feasibility labels preserve the inventory evaluation's meaning.

The final visual pass used the concepts' 1487×1058 desktop canvas and a 390×844
mobile viewport. The earlier full functional run also covered 1440×1000. Both
concepts and final screenshots were inspected directly; no material layout
mismatch remained after the measured-canvas correction and mobile cleanup.

## Visual system

The dashboard uses a planning-sheet style: paper background, white sheets with
hairline rules, tabular figures and monospace identifiers. Tokens live at the
top of `app/assets/dashboard.css`; chart colours mirror them in
`app/presentation.py`.

- Dispatch blue marks the plan and forecasts; ink marks actual sales.
- Vermillion is reserved for exceptions: storage breaches, missed targets and
  the hold decision.
- No state relies on colour alone. Statuses read "✓ Within limit" or
  "✕ Exceeds capacity +57.3 m³"; policies differ by marker shape; stores differ
  by line style; the forecast line is dashed. The palette avoids green–yellow
  pairs.
- Missing values are not shown as zero: an unobserved outcome reads "Not yet
  observed", undefined rates read "N/A".
- Keyboard focus shows a 2px accent outline; animation stops under
  `prefers-reduced-motion`.
