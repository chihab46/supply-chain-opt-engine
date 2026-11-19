# Pipeline guide

Full setup, stage-by-stage commands, measured results and implementation notes.
The [README](../README.md) is the short overview.

## Evaluation

Forecast scores average four frozen seasonal backtests across 600 store-item
series; LightGBM's four-fold score includes its three tuning folds (holdout-only
scores are below). Inventory scores use the untouched fourth fold, replayed on
actual sales over the 28 decision days plus a shared 14-day no-order tail, with
common synthetic starting stock, receipts, capacity and economics. Every
inventory row uses weekly replanning on realized stock.

| Forecast + inventory policy | Forecast WMAPE | 90% coverage | 42-day fill | Holding | Stockout | Order fees | Total cost | Eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Seasonal Naive + P50 order-up-to | 72.93% | 64.90% | 79.42% | $678 | $35,175 | $36,246 | $72,099 | No (fill) |
| AutoETS + static safety stock | 60.12% | 65.32% | 87.42% | $798 | $21,829 | $46,078 | $68,704 | No (fill) |
| LightGBM P90 + daily order-up-to | 55.22% | 88.65% | 97.62% | $1,658 | $4,815 | $49,782 | $56,255 | Yes |
| LightGBM P90 + weekly order-up-to | 55.22% | 88.65% | 97.83% | $1,997 | $4,475 | $18,187 | $24,659 | Yes |
| **LightGBM MinT P90 + MILP** | 55.76% | 84.89% | **98.33%** | $2,805 | $3,535 | $10,965 | **$17,305** | **Yes** |

"Eligible" means the policy kept every store at or above the 90% fill target and
never exceeded storage on actual sales; only eligible policies may be compared
on cost. Purchase cost is excluded from the objective. Coverage for the
statistical models uses training-only conformal P05–P95 intervals, which
under-cover on this dataset. Reconciled WMAPE uses coherent projected centers;
coverage uses marginal sample quantiles.

## Current state

Step 1 is implemented: M5 download support, deterministic dataset scoping,
validated daily sales, synthetic SKU economics, artifact manifests, and
transactional PostgreSQL loading. Step 2 implements origin-safe SQL feature views
and a Parquet export command. Step 3 implements seasonal statistical backtests
with training-only conformal intervals. The full baseline run produced 201,600
forecast observations and saved fold-level, series-level, and holdout metrics.
Step 4 adds horizon-conditioned global LightGBM quantiles through MLForecast,
seeded Optuna search on tuning folds only, saved model inference, and SHAP
diagnostics. Step 5 adds the 613-node hierarchy, Bottom-Up, Top-Down and
MinT-shrink reconciliation, coherent joint paths, and scores at every level.
Step 6 implements the bounded Pyomo/HiGHS model, origin-safe inventory scenarios,
validated purchase orders, and realized policy replay. Step 7 adds the two-tab
Dash workbench, saved forecast drill-down, background scenario jobs, matched
result KPIs, and validated purchase-order downloads. The real M5 files have been downloaded and validated. Scoping produced 600 store-item
series across 354 distinct items, with 660,000 daily observations from 2013-05-19
through 2016-05-22. PostgreSQL loading completed successfully, with 1,969 calendar
rows, 354 SKU economics rows, 145,228 weekly price rows, and 660,000 daily sales
rows. The ingestion manifest records the committed load and source checksums.

Step 8 revised the economics and planning after reviewing the holdout: the
stockout penalty became lost margin plus goodwill, the cost-only store solve
gained a capacity-repaired, heuristic warm start, and every policy now also runs
with weekly replanning on realized stock alongside a weekly-review P90 baseline.
The full pipeline was rerun from a fresh database load (ingestion run
`20261119T152927Z-596696fe`). Selected series, sales and every forecast score
reproduced the earlier runs exactly; only the stockout penalty changed.

## Local setup

Use Python 3.12, [uv](https://docs.astral.sh/uv/guides/projects/), and Docker Compose.
The Python version is deliberately restricted to 3.12 for the initial development
environment. Expand support only after checking the forecasting and solver stack.

```bash
cp -n .env.example .env
uv sync --extra dashboard
docker compose up -d postgres
docker compose ps
```

The database is published on localhost port 5433, with PostgreSQL using port 5432
inside the container. This avoids the existing service on host port 5432.
If you change the port, username,
password, or database in `.env`, update `DATABASE_URL` to match. The example
credentials are for local development. No credentials belong in version control.
The ingestion entry point loads `.env` without overriding existing environment
variables.

The database creates the four tables in `sql/01_schema.sql` when its volume is
first initialized. Ingestion also applies that script before loading. It includes
the upgrade for the new availability column; other schema changes need migrations.
`sql/02_feature_view.sql` installs feature views through the feature export command.
Stop PostgreSQL with `docker compose down`; the named volume retains its data.

For a smaller install, `uv sync` omits the dashboard. Development tools are included
by default. Optional `uv sync --extra data` adds the Kaggle download client; the
required input files can also be downloaded manually. `pyproject.toml` is the
dependency manifest and `uv.lock` records the resolved environment. Commit the
lockfile for reproducible installs. A Python 3.12 virtual environment is installed
in this workspace. Do not maintain a separate hand-written `requirements.txt`.

## Input data

Accept the M5 competition terms on Kaggle and download these files into `data/raw/`:

```text
sales_train_evaluation.csv
calendar.csv
sell_prices.csv
```

The [M5 competition data page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
requires a Kaggle account. Accept the competition rules and configure credentials
using [Kaggle's API setup](https://www.kaggle.com/docs/api). Keep credentials in
your account's Kaggle configuration or environment, never in this repository.
Then download only the required files:

```bash
uv sync --extra data
uv run supply-chain-ingest download
```

The download command preserves existing CSVs, handles Kaggle's per-file ZIPs,
and never extracts unrelated archive members. Raw data and generated artifacts
are excluded from version control. Manual CSV download works too.

Defaults select at most 600 store-item series in CA_1, CA_2, and CA_3 over roughly
1,100 days. This counts store-item pairs, so 600 distinct products sold in three
stores would exceed the default scope. Sampling must represent all three
categories and both fast-moving and intermittent series.

## Scope and load

Choose a selection cutoff no later than the earliest planned evaluation origin.
The example below reserves all sales after February 2015 from the series ranking.
It is an explicit example, not a finalized four-fold backtest schedule.

```bash
uv run supply-chain-ingest load --selection-cutoff 2015-02-28
# Validate and export without requiring PostgreSQL:
uv run supply-chain-ingest load --selection-cutoff 2015-02-28 --prepare-only
```

You can also set `data.selection_cutoff` in `configs/default.yaml`. Paths in that
configuration resolve relative to the project root, the parent of `configs/`.
For an alternate configuration, keep the same directory structure or use absolute
data paths. The equivalent module command is
`uv run python -m supply_chain_opt_engine.ingestion load ...`.

The loader reads sales in chunks and retains the final `history_days` sales
columns, regardless of future dates present in the calendar. It ranks candidates
using only days through the cutoff. Sampling first covers every store/category
cell, then interleaves fast and intermittent demand strata by velocity. An
intermittent item has at least 50% zero sales during listed training days.
Products with no positive listed training demand are excluded. If fewer than
600 eligible series exist, the manifest records the smaller panel. Unlisted
periods retain their zero sales with `is_available=false`; positive sales without
a matching weekly price fail validation.

Each run exports the selected IDs, four table CSVs, checksums, training-only
selection statistics, synthetic-cost assumptions, and dependency versions into
`data/interim/<run_id>/`. Economics also go to `data/synthetic/<run_id>/`.
`manifest.json` reports `prepared` until all database writes commit, then `loaded`.
If loading fails, the prepared bundle remains available for inspection. A rerun
revalidates the sources and creates a new bundle.

Calendar rows and all price weeks for selected store-items are retained to
support known future calendar features and historical price references. Raw future
prices remain subject to the feature availability rules in the requirements.
Economics use only prices from weeks beginning by the selection cutoff.

Loading uses COPY into temporary tables, validates existing scope and economics,
then upserts all four tables in one transaction. Exact reruns do not duplicate
rows. If a new scope would leave old facts outside its panel, or would change
existing SKU economics, loading fails without replacing the existing dataset.
Use a fresh database for a different scope or synthetic-economics seed.

## SQL feature engineering

After ingestion, install the SQL views and export the latest 28-day feature panel:

```bash
uv run python -m supply_chain_opt_engine.features
```

The command reads `.env`, installs or replaces the SQL objects in a transaction,
then reads features and source counts from a read-only repeatable database
snapshot. It saves `features.parquet` and `manifest.json` under
`data/processed/features/<run_id>/`. The manifest records the origin, SQL hash,
price policy, column names, missing-value counts, source counts, and library
versions. No realized target demand is included in the export.

Use an explicit origin for historical modeling or backtests:

```bash
uv run python -m supply_chain_opt_engine.features --origin 2015-02-28
# An explicit scenario where future weekly prices are known at the origin:
uv run python -m supply_chain_opt_engine.features --origin 2015-02-28 --known-future-prices
# Install views without exporting a panel:
uv run python -m supply_chain_opt_engine.features --install-only
```

The row key is `(unique_id, forecast_origin, ds)`. Each row includes its horizon
from 1 to 28. Demand lags are relative to the origin; rolling demand windows end
on that origin, inclusive. All target dates in a forecast share the same demand
snapshot, while their calendar features differ. Weekly prices carry forward from
the origin by default. Unlisted periods are excluded from demand statistics,
with sample counts retained; undefined values remain missing.

For SQL clients, `model_forecast_features` exposes the latest-origin 28-day panel.
Historical queries must use the parameterized function:

```sql
SELECT * FROM build_forecast_features(DATE '2015-02-28', 28, FALSE);
```

See [feature definitions](features.md) for window boundaries, weekly price
handling, holiday semantics, and the checks required before training.

## Statistical baselines

Run the four seasonal, 28-day folds with SeasonalNaive, AutoETS, and TSB:

```bash
uv run python -m supply_chain_opt_engine.baseline
# Reproduce from the checksummed CSVs committed by ingestion, without a database:
uv run python -m supply_chain_opt_engine.baseline --source ingestion
```

The schedule in `configs/default.yaml` covers spring, summer, back-to-school,
and winter holidays in 2015. Folds 1 to 3 are available for tuning; fold 4 is the
holdout. Series selection must precede the first origin. Both point estimates and
training-only conformal quantiles are evaluated. The runner reports WMAPE, RMSSE,
pinball loss at six quantiles, and central 90% coverage, with undefined-denominator
counts and separate tuning, holdout, and all-fold summaries.

Each run saves predictions, training scales, series/fold/summary metrics, runtimes,
and a checksummed manifest in `artifacts/metrics/baselines/<run_id>/`. AutoETS model
selection and nested calibration are the slowest part; the default uses two
workers. See [baseline definitions](baselines.md) for the exact dates,
observed-sales policy, calibration assumptions, formulas, and artifact schemas.

Measured forecast comparison, with RMSSE averaged equally across defined series-folds:

| Model | Four-fold WMAPE | Mean RMSSE | 90% coverage | Mean pinball loss | Holdout WMAPE |
| --- | --- | --- | --- | --- | --- |
| Seasonal Naive | 72.93% | 0.964 | 64.90% | 1.606 | 83.20% |
| AutoETS | 60.12% | 0.788 | 65.32% | 1.308 | 65.04% |
| TSB | 61.59% | 0.764 | 65.09% | 1.315 | 69.08% |
| LightGBM | 55.22% | 0.752 | 88.65% | 1.055 | 59.20% |

Among the statistical baselines, AutoETS has the lowest WMAPE on the tuning folds
and the reported four-fold summary. TSB has their lowest mean RMSSE. Two-window conformal intervals have
limited calibration support and achieved about 65% coverage at the nominal 90%
level; this result must remain visible when evaluating later quantile models.
The reference run used two workers. AutoETS fitting, calibration, and prediction
totaled about 10.8 minutes across all four folds; Seasonal Naive and TSB each
totaled about one second. These are local model timings, not runtime guarantees.

The compact [benchmark evidence](benchmarks/README.md) includes saved summary
and fold CSVs, runtimes, run provenance, and independent validation. The full local
bundle is `artifacts/metrics/baselines/20261119T154311Z-cf01203f/`; its summary
is identical to the earlier run `20261111T162554Z-2f637d4b`.

## Global LightGBM quantiles

Fit the six global quantile models, tune on folds 1 to 3, evaluate the frozen
holdout, and refit at the latest origin:

```bash
uv run python -m supply_chain_opt_engine.forecasting
# Use the checksummed committed extracts without PostgreSQL access:
uv run python -m supply_chain_opt_engine.forecasting --source ingestion
```

The direct model uses horizon-safe origin features plus target-calendar fields.
Thirteen non-overlapping origin blocks supply a year of training labels. Median
pinball loss selects the tree parameters; all quantiles reuse the frozen winning
settings. The holdout is excluded from Optuna. A declared repair policy preserves
P50 while ordering the lower and upper quantiles. SHAP explains the raw P50 model
and includes an additivity check.

Outputs under `artifacts/models/lightgbm/<run_id>/` include native holdout and
production models, historical and latest forecasts, trial history and SQLite
study, scores, SHAP values, runtime measurements, and a checksummed manifest.
See [forecasting definitions](forecasting.md) for source consistency,
training-label cutoffs, parameter ranges, categorical schemas, and model loading.

The full 30-trial reference run completed 23 trials and pruned 7. Its untouched
holdout achieved **59.20% WMAPE**, **0.726 RMSSE**, **89.55% central-90% coverage**,
and **0.984 mean pinball loss**. AutoETS holdout WMAPE was 65.04%, so LightGBM's
relative WMAPE reduction was 8.99%. Its four-fold scores include parameter-selection
folds and should not be treated as an independent test estimate.

Each quantile-model set used 218,400 training examples; fitting and forecasting
took 46 to 56 seconds on this machine. The latest-origin refit phase took about
57 seconds, including saved models and SHAP diagnostics. The full Optuna search
is a separate upfront cost. Runtime measurements are not guarantees.

The largest holdout median-model SHAP drivers were 7-day and 14-day rolling demand,
series identity, item identity, and 28-day rolling demand. These are model
dependence diagnostics, not causal effects. Forecast nonnegativity/order repair
altered 46,872 of the 67,200 historical rows; P50 was preserved except for clipping
negative values at zero. Coverage and pinball scores include that fixed repair.

Reference scores, trial history, provenance, runtime measurements, SHAP importance,
and independent validation are in [the benchmark evidence](benchmarks/README.md).
The full local bundle is
`artifacts/models/lightgbm/20261119T153241Z-d725af75/` (identical summary to the
earlier `20261111T165106Z-52250620`), with 67,200 historical
forecast rows and 16,800 latest-origin forecast rows.

## Hierarchical reconciliation

Step 5 evaluates Bottom-Up, Top-Down and MinT-shrink over Total → Store →
Store-Category → Store-Item. Independent weekly AutoETS models forecast the 13
aggregate nodes; the frozen LightGBM models supply the 600 leaves. Covariance
and allocation shares use only pre-origin observations. MinT is the declared
primary method; holdout scores do not select the method.

```bash
uv run python -m supply_chain_opt_engine.reconciliation
# Reproduce from the committed ingestion extracts:
uv run python -m supply_chain_opt_engine.reconciliation --source ingestion
```

The full run `20261119T154145Z-2da2199f` (matching the earlier
`20261111T175055Z-ad5ca98d` to within 1e-13) saved all four seasonal backtests and
latest-origin forecasts with 501 coherent joint paths per method. Holdout point
WMAPE is:

| Level | Independent base | Bottom-Up | Top-Down | MinT-shrink |
| --- | --- | --- | --- | --- |
| Total | 12.60% | 19.22% | 12.60% | 15.09% |
| Store | 14.19% | 19.36% | 16.92% | 15.49% |
| Store-Category | 20.08% | 22.09% | 22.52% | 20.18% |
| Store-Item | 59.20% | 59.20% | 73.82% | 61.08% |

All three reconciliation methods produce coherent points and paths. MinT
improves upper-level WMAPE relative to Bottom-Up, but does not improve holdout
WMAPE relative to the independent base at any level. Leaf P05-P95 coverage falls
from 89.55% to 84.46%. Coherence therefore comes with a measured accuracy and
coverage tradeoff in this configuration; the brief's suggested gain is not an
observed result. Bottom-Up preserves all original leaf points and quantiles.

Independent matrix checks found maximum point/path summation differences below
`7e-12`, within the declared `1e-8` tolerance. Marginal quantiles need not add,
even when their underlying paths are coherent. Negative projected leaves are
clipped and aggregates recomputed; this is not a constrained MinT optimum.

The inventory stage reads 16,800 primary-method rows from
`artifacts/forecasts/reconciliation/20261119T154145Z-2da2199f/latest_leaf_forecasts.parquet`.
Full covariance, projections, sample paths and metrics remain in the local
artifact bundle. See [reconciliation definitions](reconciliation.md) and
[portable benchmark evidence](benchmarks/README.md) for assumptions,
checksums and validation details.

## Inventory optimization and holdout simulation

```bash
uv run python -m supply_chain_opt_engine.optimization
# Run just the holdout or latest-origin scenario:
uv run python -m supply_chain_opt_engine.optimization --phase holdout
uv run python -m supply_chain_opt_engine.optimization --phase latest
```

Step 6 uses reconciled leaf P90 demand by default. It implements integer orders,
MOQ triggers, lead-time arrivals, end-of-day store capacity, and work-conserving
lost sales, with a 90% minimum store fill rate on forecast demand. Synthetic
state comes from the 56 observed days ending at each origin. Every policy uses
the same starting inventory, pipeline and capacity. The solver checks item
subproblem bounds and combined capacity, with a warm-started coupled store MILP
fallback. It validates every extracted schedule by physical replay.

Each policy runs twice on the holdout. The **frozen** version fixes all orders at
the origin. The **rolling** version (`rolling_review_days: 7`) replans on days
0, 7, 14 and 21 from realized on-hand stock, using only sales observed before
each review, keeps orders already placed, and commits one week at a time. The
origin forecast is reused, not refreshed. Run
`20261119T161809Z-b00ebe51` gave these 42-day results on actual sales:

| Policy | Version | Fill | Lowest store | Total cost | Orders | Max overflow | Eligible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MinT P90 MILP | Frozen | 98.58% | 98.32% | $18,322 | 1,687 | 134.2 m³ | No |
| MinT P90 MILP | **Rolling** | **98.33%** | **97.97%** | **$17,305** | **1,582** | **0** | **Yes** |
| P90 weekly order-up-to | Frozen | 98.59% | 98.17% | $26,142 | 2,830 | 130.2 m³ | No |
| P90 weekly order-up-to | Rolling | 97.83% | 97.06% | $24,659 | 2,587 | 0 | Yes |
| P90 daily order-up-to | Frozen | 98.54% | 98.11% | $80,661 | 10,625 | 130.3 m³ | No |
| P90 daily order-up-to | Rolling | 97.62% | 96.90% | $56,255 | 7,063 | 0 | Yes |
| P50 daily order-up-to | Rolling | 71.60% | 68.89% | $80,816 | 4,766 | 0 | No (fill) |

Every frozen P90 plan overflows storage once actual demand falls below the
forecast; replay keeps those units rather than assuming free disposal. Rolling
fixes that for every policy at a small fill cost. Among eligible plans, the
rolling MILP is cheapest: 30% below weekly order-up-to and 69% below the daily
rule.

The holdout run took 319 seconds in total, including four weekly MILP reviews of
63, 62, 62 and 25 seconds, each with the full 60-second solver budget. The
brief's 1–2 second scenario target has not been achieved. Frozen MILP gaps were
34–42% per store. The separate run `20261119T154350Z-191214db` also planned the
latest origin (75 seconds): projected fill is 98.85% with the lowest store at
98.64%, within storage. Future actual sales are unknown, so it has no realized
fill rate.

Outputs are under `artifacts/orders/optimization/<run_id>/`, with side inputs
under `data/synthetic/inventory/<run_id>/`. Each holdout bundle saves
`rolling_reviews.json` with every review's status and committed units.
Purchase-order exports include a policy column, dates, integer quantities, costs
and scenario IDs. `scripts/validate_inventory_run.py` independently replays every
schedule and recomputes its costs; both runs above pass. See
[optimization definitions](optimization.md). The
[portable benchmark evidence](benchmarks/README.md) still holds the
pre-revision runs.

## Interactive planning dashboard

```bash
uv sync --extra dashboard
uv run --extra dashboard python -m app.dashboard
```

Open [localhost:8050](http://127.0.0.1:8050). The demand forecast view drills from
Total to Store, Category and Item, with saved uncertainty bands, daily values and
clearly scoped SHAP explanations. It supports the latest origin and all four
historical folds.

The replenishment simulator accepts six demand quantiles, capacity from 10% to
150%, lead-time delays from zero to four days, a minimum store fill requirement,
and a synthetic shortage-penalty multiplier. Jobs run in separate processes
with persistent pending/running/completed/failed states and deduplication. The
forecast tab remains available during a solve. Matching validated reference
results load immediately; changed settings clear stale KPIs and downloads.

Holdout KPIs use realized demand. Future scenarios report projected fulfillment,
with no claim of an observed future fill rate. Zero orders and storage failures
remain visible. Downloads contain only validated MILP orders; a zero-order
result produces a header-only CSV. A result missing the selected window's service
or capacity requirements cannot be downloaded. Use the reporting selector to inspect either
the 28-day window or the common tail.

See [dashboard usage and architecture](dashboard.md) for startup settings,
job persistence, source lineage, chart semantics and validation coverage.

## Repository layout

```text
app/
  dashboard.py               Two-tab Dash workbench and callbacks
  presentation.py            Native Plotly figures, tables and KPIs
  assets/                    Responsive CSS, local fonts and chart resizing
artifacts/
  forecasts/                 Forecasts and reconciled scenarios
  metrics/                   Fold metrics, policy KPIs, runtime measurements
  models/                    Fitted models and training metadata
  orders/                    Purchase-order exports
configs/default.yaml         Scope, forecasting, and solver defaults
data/
  raw/                       Original M5 CSVs
  interim/                   Scoped and validated extracts
  processed/                 Modeling tables
  synthetic/                 Economics, capacity, starting stock, pipeline receipts
docs/requirements.md         Functional requirements and acceptance criteria
sql/01_schema.sql            Four normalized PostgreSQL tables and indexes
sql/02_feature_view.sql      Origin-safe demand, weekly price and calendar features
src/supply_chain_opt_engine/
  ingestion.py               Download/load CLI and artifact manifests
  m5.py                      M5 validation and deterministic panel selection
  metadata.py                Synthetic SKU economics from training prices
  database.py                PostgreSQL schema bootstrap and COPY/upsert loading
  features.py                SQL feature installation, validation and Parquet export
  evaluation.py              Frozen seasonal folds, daily-grid checks, training scales
  metrics.py                 WMAPE, RMSSE, pinball and interval coverage
  history.py                 Provenance-checked PostgreSQL or committed CSV history
  baseline.py                SeasonalNaive, AutoETS, TSB, backtests and metrics
  forecasting.py             MLForecast, LightGBM quantiles, Optuna and SHAP
  feature_store.py           SQL/offline feature snapshots and fixed categorical schema
  learning.py                Direct training matrices, quantile fitting and native model IO
  tuning.py                  Seeded Optuna study restricted to folds 1-3
  explanations.py            P50 TreeSHAP diagnostics and additivity checks
  hierarchy.py               Stable node ordering, summing matrix and coherence checks
  reconcilers.py             Nixtla projections and joint rank-block sampling
  reconciliation_data.py     Origin-bounded fitted errors and aggregate base forecasts
  reconciliation.py          Four-fold reconciliation, metrics and scenario artifacts
  inventory_data.py          Validated arrays and pre-origin scenario generation
  inventory_model.py         Pyomo/HiGHS model, bounds, gaps and feasibility checks
  optimization.py            Scenario runner, lineage checks and purchase-order bundles
  simulation.py              Forecast-only policy planning and realized-demand replay
  dashboard_data.py          Pinned forecast snapshots and checked result views
  dashboard_jobs.py          Persistent scenario queue and worker lifecycle
scripts/validate_inventory_run.py  Independent saved-evidence replay
tests/unit/                  Ingestion, forecasts, reconciliation and inventory tests
tests/integration/           PostgreSQL rerun, rollback and scope-safety tests
tests/sql/                   Real PostgreSQL feature, boundary and leakage tests
compose.yaml                 Local PostgreSQL service
pyproject.toml               Dependencies, packaging, lint and test settings
uv.lock                      Resolved direct and transitive dependencies
```

The `src/` package layout makes imports independent of the working directory and
avoids top-level module name collisions. The application stays separate from the
installable analytics package.

## Development checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m compileall -q src app
uv run pytest -m "not integration and not slow"
# Requires a PostgreSQL database where the test user may create schemas:
TEST_DATABASE_URL="$DATABASE_URL" uv run pytest -m integration
```

Integration tests create and remove a unique schema per test. Set
`TEST_DATABASE_URL` explicitly; tests skip database access when it is absent.
Export `DATABASE_URL` first if using the example above; pytest does not load `.env`
automatically. `tests/README.md` describes current and planned coverage. Imports
must not connect to PostgreSQL, train models, or start a server.

`tests/sql/` runs the feature SQL in a temporary single-user PostgreSQL cluster.
It needs local PostgreSQL server binaries and a non-root user, and skips when
those are absent. It does not require Docker, network access, or a running server.

## Implementation order

1. Freeze the scope and four seasonal cutoffs, ingest the M5 subset, and generate
   reproducible economics, capacities, starting inventory, and pipeline receipts.
2. Implement feature availability rules and chronological evaluation, then save
   statistical baseline metrics.
3. Train quantile models, tune on folds 1 to 3, and reconcile the full hierarchy.
4. Build and test the inventory model, then compare policies on realized demand.
5. Connect validated artifacts and background optimization to the Dash simulator.

See [requirements](requirements.md) for definitions and completion criteria.
Setup follows uv's [project workflow](https://docs.astral.sh/uv/guides/projects/).
The solver uses Pyomo's
[HiGHS interface](https://pyomo.readthedocs.io/en/stable/api/pyomo.contrib.appsi.solvers.highs.Highs.html).
