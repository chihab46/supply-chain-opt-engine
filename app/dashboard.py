"""Local two-tab Dash planning workbench. Start with python -m app.dashboard."""

from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import asdict
from pathlib import Path

import yaml
from dash import Dash, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate
from dotenv import load_dotenv

from app.presentation import (
    blank_results,
    empty_figure,
    forecast_figure,
    forecast_summary,
    forecast_table,
    result_view,
    scenario_results,
    shap_figure,
)
from supply_chain_opt_engine.dashboard_data import ALL, DashboardRepository
from supply_chain_opt_engine.dashboard_jobs import ACTIVE, JobManager, selected_config
from supply_chain_opt_engine.optimization import OptimizationConfig

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
QUANTILE_CHOICES = (0.05, 0.1, 0.5, 0.8, 0.9, 0.95)
SIMULATOR_SUBTITLE = "Test a replenishment plan against common inventory and capacity."
QUANTILE_NOTE = "P90 demand is a planning scenario, not a minimum fill rate."
TAIL_LABEL = "Including 14-day tail"
GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}


def quantile_at(position):
    if (
        isinstance(position, bool)
        or not isinstance(position, (int, float))
        or not math.isfinite(position)
        or position != int(position)
        or not 0 <= position < len(QUANTILE_CHOICES)
    ):
        raise ValueError("Choose one of the six demand quantiles.")
    return QUANTILE_CHOICES[int(position)]


def dropdown(id, options, value, **kwargs):
    return dcc.Dropdown(
        id=id, options=options, value=value, clearable=False, searchable=True, **kwargs
    )


def field(label, component):
    return html.Div([html.Label(label, htmlFor=component.id, className="field-label"), component])


def config_from_project(root):
    path = Path(os.environ.get("PROJECT_CONFIG", root / "configs/default.yaml"))
    if not path.is_absolute():
        path = root / path
    doc = yaml.safe_load(path.read_text())
    opts = {
        k: v
        for k, v in doc["optimization"].items()
        # Rolling replans multiply solve time; interactive jobs stay frozen.
        if k in OptimizationConfig.__dataclass_fields__ and k not in {"seed", "rolling_review_days"}
    }
    return OptimizationConfig(**opts, seed=doc["seed"])


def head(title, note=None, id=None):
    heading = html.H2(title, id=id) if id else html.H2(title)
    return html.Div(
        [heading, *([html.Span(note, className="section-note")] if note else [])],
        className="section-head",
    )


def slider_field(label, id, value_id, value_text, slider):
    return html.Div(
        [
            html.Label(
                [label, html.Span(value_text, id=value_id, className="slider-value")],
                htmlFor=id,
                className="field-label",
            ),
            slider,
        ],
        className="slider-field",
    )


def layout(repository, defaults=None):
    defaults = defaults or OptimizationConfig()
    stores = sorted(repository.leaves.store_id.unique())
    return html.Div(
        [
            dcc.Store(id="job-id", storage_type="session"),
            dcc.Store(id="job-state"),
            dcc.Interval(id="job-poll", interval=1000),
            dcc.Download(id="order-download"),
            html.Header(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("Supply Chain Engine", className="wordmark"),
                                html.Span("Replenishment planning", className="wordmark-sub"),
                            ],
                            className="brand",
                        ),
                        dcc.Tabs(
                            id="workspace-tab",
                            value="forecast",
                            className="workspace-tabs",
                            parent_className="workspace-tabs-parent",
                            children=[
                                dcc.Tab(
                                    label="Demand forecast",
                                    value="forecast",
                                    className="workspace-tab",
                                    selected_className="workspace-tab-selected",
                                ),
                                dcc.Tab(
                                    label="Replenishment simulator",
                                    value="scenario",
                                    className="workspace-tab",
                                    selected_className="workspace-tab-selected",
                                ),
                            ],
                        ),
                        html.Div(
                            [
                                "Walmart M5 · ",
                                html.Span(" ".join(stores), className="mono"),
                                f" · {len(repository.leaves):,} store-items · synthetic costs",
                            ],
                            className="dataset",
                        ),
                    ],
                    className="header-inner",
                ),
                className="app-header",
            ),
            html.Main(
                [
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.H1("Demand forecast"),
                                    html.P(
                                        "Expected daily sales for every store, category and item, "
                                        "with reconciled uncertainty bands."
                                    ),
                                ],
                                className="page-head",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            field(
                                                "As of",
                                                dropdown(
                                                    "forecast-origin",
                                                    [
                                                        {"label": v, "value": k}
                                                        for k, v in repository.origin_labels.items()
                                                    ],
                                                    "latest",
                                                ),
                                            ),
                                            html.Div(className="filter-divider"),
                                            field(
                                                "Store",
                                                dropdown(
                                                    "store-filter",
                                                    [
                                                        {"label": "All stores", "value": ALL},
                                                        *[{"label": s, "value": s} for s in stores],
                                                    ],
                                                    ALL,
                                                ),
                                            ),
                                            field(
                                                "Category",
                                                dropdown(
                                                    "category-filter",
                                                    [{"label": "All categories", "value": ALL}],
                                                    ALL,
                                                    disabled=True,
                                                ),
                                            ),
                                            field(
                                                "Item",
                                                dropdown(
                                                    "item-filter",
                                                    [{"label": "All items", "value": ALL}],
                                                    ALL,
                                                    disabled=True,
                                                ),
                                            ),
                                        ],
                                        className="filter-bar",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    head(
                                                        "Total demand",
                                                        "Units per day · 42 days of history "
                                                        "and a 28-day forecast",
                                                        id="forecast-title",
                                                    ),
                                                    dcc.Graph(
                                                        id="forecast-chart",
                                                        responsive=True,
                                                        style={"height": "300px", "width": "100%"},
                                                        figure=empty_figure(
                                                            "Loading saved forecasts.", 300
                                                        ),
                                                        config=GRAPH_CONFIG,
                                                    ),
                                                    html.P(
                                                        "Shaded bands are P10–P95 (nominal 85% "
                                                        "coverage) and P10–P90. Point forecasts "
                                                        "add across the hierarchy; marginal "
                                                        "quantiles do not.",
                                                        className="chart-caption",
                                                    ),
                                                ],
                                                className="chart-main",
                                            ),
                                            html.Aside(
                                                [
                                                    html.H2("Reading this forecast"),
                                                    html.Div(id="forecast-summary"),
                                                ],
                                                className="summary-rail",
                                            ),
                                        ],
                                        className="forecast-grid",
                                    ),
                                ],
                                className="sheet",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            head("Daily forecast", "Next 28 days"),
                                            html.Div(id="forecast-table"),
                                        ],
                                        className="sheet sheet-pad",
                                    ),
                                    html.Div(
                                        [
                                            head("Model drivers", "Median model, all series"),
                                            dcc.Graph(
                                                id="shap-chart",
                                                responsive=True,
                                                style={"height": "260px", "width": "100%"},
                                                figure=empty_figure(
                                                    "Loading model explanation.", 260
                                                ),
                                                config=GRAPH_CONFIG,
                                            ),
                                            html.Div(id="shap-note", className="chart-caption"),
                                        ],
                                        className="sheet sheet-pad",
                                    ),
                                ],
                                className="pair",
                            ),
                            html.Div(id="forecast-source", className="source-line"),
                        ],
                        id="forecast-panel",
                    ),
                    html.Section(
                        [
                            html.Div(
                                [html.H1("Replenishment simulator"), html.P(SIMULATOR_SUBTITLE)],
                                className="page-head",
                            ),
                            html.Div(
                                [
                                    html.Aside(
                                        [
                                            html.Fieldset(
                                                [
                                                    html.Legend("Scenario"),
                                                    field(
                                                        "Evaluation window",
                                                        dropdown(
                                                            "scenario-phase",
                                                            [
                                                                {"label": v, "value": k}
                                                                for k, v in (
                                                                    repository.phase_labels.items()
                                                                )
                                                            ],
                                                            "holdout",
                                                        ),
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label(
                                                                "Demand quantile",
                                                                htmlFor="scenario-quantile",
                                                                className="field-label",
                                                            ),
                                                            dcc.Slider(
                                                                id="scenario-quantile",
                                                                min=0,
                                                                max=5,
                                                                step=1,
                                                                marks={
                                                                    i: f"P{round(q * 100):02}"
                                                                    for i, q in enumerate(
                                                                        QUANTILE_CHOICES
                                                                    )
                                                                },
                                                                value=4,
                                                                included=False,
                                                            ),
                                                            html.P(
                                                                QUANTILE_NOTE,
                                                                id="quantile-note",
                                                                className="control-note",
                                                            ),
                                                        ],
                                                        className="slider-field",
                                                    ),
                                                ]
                                            ),
                                            html.Fieldset(
                                                [
                                                    html.Legend("Service policy"),
                                                    field(
                                                        "Minimum store fill rate, %",
                                                        dcc.Input(
                                                            id="scenario-service",
                                                            type="number",
                                                            min=0,
                                                            max=100,
                                                            step=1,
                                                            value=defaults.minimum_fill_rate * 100,
                                                            className="scenario-number",
                                                        ),
                                                    ),
                                                    html.P(
                                                        "Required on forecast demand in every "
                                                        "store, over 28 days and the full window. "
                                                        "Actual service can differ. 0% plans on "
                                                        "cost alone.",
                                                        className="control-note",
                                                    ),
                                                    field(
                                                        "Shortage penalty multiplier",
                                                        dcc.Input(
                                                            id="scenario-penalty",
                                                            type="number",
                                                            min=0.01,
                                                            # "any": with min 0.01, step 0.1 made
                                                            # the default 1 an invalid value.
                                                            step="any",
                                                            value=(
                                                                defaults.shortage_penalty_multiplier
                                                            ),
                                                            className="scenario-number",
                                                        ),
                                                    ),
                                                    html.P(
                                                        "Costs, starting stock and capacity are "
                                                        "synthetic assumptions, not Walmart data.",
                                                        className="control-note",
                                                    ),
                                                ]
                                            ),
                                            html.Fieldset(
                                                [
                                                    html.Legend("Supply constraints"),
                                                    slider_field(
                                                        "Store capacity",
                                                        "scenario-capacity",
                                                        "capacity-value",
                                                        "100%",
                                                        dcc.Slider(
                                                            id="scenario-capacity",
                                                            min=10,
                                                            max=150,
                                                            step=5,
                                                            marks={
                                                                v: f"{v}%"
                                                                for v in [10, 50, 100, 150]
                                                            },
                                                            value=100,
                                                        ),
                                                    ),
                                                    slider_field(
                                                        "Lead-time delay",
                                                        "scenario-delay",
                                                        "delay-value",
                                                        "+0 days",
                                                        dcc.Slider(
                                                            id="scenario-delay",
                                                            min=0,
                                                            max=4,
                                                            step=1,
                                                            marks={v: str(v) for v in range(5)},
                                                            value=0,
                                                        ),
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                [
                                                    html.Button(
                                                        "Run optimizer",
                                                        id="run-optimizer",
                                                        n_clicks=0,
                                                        className="button-primary run-button",
                                                    ),
                                                    html.Div(
                                                        id="submit-error",
                                                        role="alert",
                                                        className="submission-error",
                                                    ),
                                                ],
                                                className="form-footer",
                                            ),
                                        ],
                                        className="scenario-form",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        "Ready",
                                                        id="scenario-status",
                                                        className="status-text",
                                                        role="status",
                                                        **{"aria-live": "polite"},
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label(
                                                                "Report",
                                                                htmlFor="result-scope",
                                                                className="inline-label",
                                                            ),
                                                            dropdown(
                                                                "result-scope",
                                                                [
                                                                    {
                                                                        "label": "28-day window",
                                                                        "value": "decision_window",
                                                                    },
                                                                    {
                                                                        "label": TAIL_LABEL,
                                                                        "value": "including_tail",
                                                                    },
                                                                ],
                                                                "decision_window",
                                                            ),
                                                        ],
                                                        className="scope-field",
                                                    ),
                                                ],
                                                className="status-line",
                                            ),
                                            html.Div(
                                                blank_results(
                                                    "Loading validated reference results."
                                                ),
                                                id="scenario-results",
                                            ),
                                        ],
                                        className="results-column",
                                    ),
                                ],
                                className="scenario-layout",
                            ),
                        ],
                        id="scenario-panel",
                        style={"display": "none"},
                    ),
                ],
                className="workspace",
            ),
        ]
    )


def create_app(root=ROOT, repository=None, manager=None):
    root = Path(root).resolve()
    app = Dash(
        __name__,
        title="Supply Chain Engine",
        assets_folder=str(ROOT / "app/assets"),
        suppress_callback_exceptions=True,
        update_title="Updating · Supply Chain Engine",
    )
    try:
        repository = repository or DashboardRepository(root)
        manager = manager or JobManager(
            root, repository.recpath, repository.basepath, config_from_project(root)
        )
    except (ValueError, OSError, KeyError) as error:
        LOGGER.error("Dashboard source unavailable: %s", error)
        app.layout = html.Main(
            [
                html.H1("Forecast data is not ready"),
                html.P(
                    "Complete the ingestion, forecasting, reconciliation and inventory steps, "
                    "then restart this dashboard."
                ),
                html.P("See README.md for pipeline commands."),
            ],
            className="setup panel",
        )
        return app
    app.layout = layout(repository, manager.defaults)
    app.repository = repository
    app.job_manager = manager

    @app.callback(
        Output("forecast-panel", "style"),
        Output("scenario-panel", "style"),
        Input("workspace-tab", "value"),
    )
    def switch_tab(tab):
        return (
            ({"display": "block"}, {"display": "none"})
            if tab == "forecast"
            else ({"display": "none"}, {"display": "block"})
        )

    @app.callback(
        Output("category-filter", "options"),
        Output("category-filter", "value"),
        Output("category-filter", "disabled"),
        Input("store-filter", "value"),
    )
    def categories(store):
        opts = [
            {"label": "All categories", "value": ALL},
            *[{"label": c.title(), "value": c} for c in repository.category_options(store)],
        ]
        return opts, ALL, store == ALL

    @app.callback(
        Output("item-filter", "options"),
        Output("item-filter", "value"),
        Output("item-filter", "disabled"),
        Input("store-filter", "value"),
        Input("category-filter", "value"),
    )
    def items(store, category):
        options = (
            repository.item_options(store, category) if store != ALL and category != ALL else []
        )
        return [{"label": "All items", "value": ALL}, *options], ALL, not options

    @app.callback(
        Output("forecast-title", "children"),
        Output("forecast-chart", "figure"),
        Output("forecast-table", "children"),
        Output("shap-chart", "figure"),
        Output("shap-note", "children"),
        Output("forecast-source", "children"),
        Input("forecast-origin", "value"),
        Input("store-filter", "value"),
        Input("category-filter", "value"),
        Input("item-filter", "value"),
    )
    def update_forecast(origin, store, category, item):
        try:
            node, date, observed, forecast = repository.forecast_view(origin, store, category, item)
        except ValueError:
            return (
                "Choose a hierarchy selection",
                empty_figure("Updating the hierarchy selection."),
                [],
                empty_figure(""),
                "",
                "",
            )
        title = (
            "Total demand"
            if node == "Total"
            else f"{store} demand"
            if category == ALL
            else f"{store} · {category.title()}"
            if item == ALL
            else f"{store} · {item.split('/')[-1]}"
        )
        importance, role = repository.shap_importance(origin)
        note = (
            f"Model-wide mean |SHAP| · {'production' if role == 'latest' else 'holdout'} P50. "
            "Associations are not causal effects."
            if role
            else "No explanation sample was saved for this tuning fold."
        )
        source = (
            f"Reconciliation {repository.rec['run_id']} · "
            f"Origin {date:%b %d, %Y} · 28 days · Node {node}"
        )
        return (
            title,
            forecast_figure(date, observed, forecast),
            forecast_table(forecast),
            shap_figure(importance),
            note,
            source,
        )

    @app.callback(
        Output("forecast-summary", "children"),
        Input("forecast-origin", "value"),
        Input("store-filter", "value"),
        Input("category-filter", "value"),
        Input("item-filter", "value"),
    )
    def update_summary(origin, store, category, item):
        try:
            _, date, observed, forecast = repository.forecast_view(origin, store, category, item)
        except ValueError:
            return html.P("Choose a store, category and item.", className="muted")
        return forecast_summary(date, observed, forecast)

    @app.callback(
        Output("capacity-value", "children"),
        Output("delay-value", "children"),
        Output("quantile-note", "children"),
        Input("scenario-capacity", "value"),
        Input("scenario-delay", "value"),
        Input("scenario-quantile", "value"),
    )
    def control_labels(capacity, delay, quantile):
        try:
            q = quantile_at(quantile)
            note = f"P{round(q * 100):02} demand is a planning scenario, not a minimum fill rate."
        except ValueError:
            note = "Choose one of the six available forecast quantiles."
        return f"{capacity:g}%", f"+{delay:g} days", note

    def config_for(q, capacity, delay, service, penalty):
        if isinstance(service, bool) or not isinstance(service, (int, float)):
            raise ValueError("Choose a minimum store fill rate between 0% and 100%.")
        if isinstance(penalty, bool) or not isinstance(penalty, (int, float)):
            raise ValueError("Choose a positive shortage penalty multiplier.")
        return selected_config(
            manager.defaults, quantile_at(q), capacity, delay, service / 100, penalty
        )

    def resolve(state, phase, config):
        key, _ = manager.identity(phase, config)
        if state and state.get("job_id") == key:
            if state["status"] == "completed":
                result = repository.result(str(root / state["result_manifest"]), phase)
                if result["manifest"]["config"] != asdict(config):
                    raise ValueError("Result settings differ from the selected scenario.")
                return result, "Completed"
            return None, state["status"]
        path = repository.reference(phase, asdict(config))
        if path:
            return repository.result(str(path), phase), "Saved reference"
        return None, "ready"

    @app.callback(
        Output("job-id", "data"),
        Output("submit-error", "children"),
        Input("run-optimizer", "n_clicks"),
        State("scenario-phase", "value"),
        State("scenario-quantile", "value"),
        State("scenario-capacity", "value"),
        State("scenario-delay", "value"),
        State("scenario-service", "value"),
        State("scenario-penalty", "value"),
        prevent_initial_call=True,
    )
    def submit(n, phase, q, capacity, delay, service, penalty):
        if not n:
            raise PreventUpdate
        try:
            config = config_for(q, capacity, delay, service, penalty)
            key, _ = manager.identity(phase, config)
            if manager.path(key).exists():
                manager.read(key)
            reference = repository.reference(phase, asdict(config))
            if reference:
                repository.result(str(reference), phase)
            state = manager.submit(phase, config, reference=reference)
            return {"job_id": state["job_id"]}, ""
        except (ValueError, OSError) as error:
            LOGGER.error("Could not submit scenario: %s", error)
            return no_update, "The scenario could not start. Check the selected settings and retry."

    @app.callback(
        Output("job-state", "data"),
        Input("job-id", "data"),
        Input("job-poll", "n_intervals"),
        State("job-state", "data"),
    )
    def poll(job, _, previous):
        if not job:
            return no_update
        try:
            state = manager.read(job["job_id"])
        except (ValueError, OSError, KeyError):
            state = {
                "job_id": job.get("job_id"),
                "status": "failed",
                "error": "This scenario job is unavailable. Run optimizer to retry.",
            }
        return no_update if state == previous else state

    @app.callback(
        Output("scenario-results", "children"),
        Output("scenario-status", "children"),
        Output("scenario-status", "className"),
        Output("run-optimizer", "disabled"),
        Output("run-optimizer", "children"),
        *[
            Output(c, "disabled")
            for c in [
                "scenario-phase",
                "scenario-quantile",
                "scenario-capacity",
                "scenario-delay",
                "scenario-service",
                "scenario-penalty",
            ]
        ],
        Input("job-state", "data"),
        Input("scenario-phase", "value"),
        Input("scenario-quantile", "value"),
        Input("scenario-capacity", "value"),
        Input("scenario-delay", "value"),
        Input("scenario-service", "value"),
        Input("scenario-penalty", "value"),
        Input("result-scope", "value"),
    )
    def show_results(state, phase, q, capacity, delay, service, penalty, scope):
        try:
            if scope not in {"decision_window", "including_tail"}:
                raise ValueError("Unknown result window.")
            config = config_for(q, capacity, delay, service, penalty)
            result, status = resolve(state, phase, config)
            active = status in ACTIVE
            label = (
                "Queued"
                if status == "pending"
                else "Optimizer running…"
                if status == "running"
                else "Run optimizer"
            )
            if result is not None:
                result = {**result, "tradeoff": repository.tradeoff(phase, asdict(config), scope)}
                content = scenario_results(result, phase, scope)
                status_text = (
                    f"{status} · P{round(config.service_quantile * 100):02} · "
                    f"{capacity:g}% capacity · +{delay:g} days"
                    f" · {service:g}% service target · {penalty:g}x shortage penalty"
                )
            elif status == "failed":
                content = blank_results(
                    state.get("error", "The scenario failed. Retry with different settings."), True
                )
                status_text = "Failed · no orders available"
            elif active:
                text = (
                    "Waiting for the available solver worker."
                    if status == "pending"
                    else "Optimization is running in the background. Solver budget is 60 seconds; "
                    "data preparation and replay add time."
                )
                content = blank_results(text)
                status_text = (
                    "Pending · queued for the solver"
                    if status == "pending"
                    else "Running · scenario controls are locked"
                )
            else:
                content = blank_results(
                    "Settings changed. Run optimizer to calculate this scenario."
                )
                status_text = "Ready · no result for these settings"
            return content, status_text, f"status-text {status}", active, label, *([active] * 6)
        except (ValueError, OSError, KeyError) as error:
            LOGGER.error("Could not display scenario: %s", error)
            return (
                blank_results(
                    "The saved result could not be verified. Run a new scenario or restart "
                    "after checking the source files.",
                    True,
                ),
                "Result unavailable",
                "status-text",
                False,
                "Run optimizer",
                False,
                False,
                False,
                False,
                False,
                False,
            )

    @app.callback(
        Output("order-download", "data"),
        Input("download-orders", "n_clicks"),
        State("job-state", "data"),
        State("scenario-phase", "value"),
        State("scenario-quantile", "value"),
        State("scenario-capacity", "value"),
        State("scenario-delay", "value"),
        State("scenario-service", "value"),
        State("scenario-penalty", "value"),
        State("result-scope", "value"),
        prevent_initial_call=True,
    )
    def download(n, state, phase, q, capacity, delay, service, penalty, scope):
        if not n:
            raise PreventUpdate
        try:
            result, _ = resolve(state, phase, config_for(q, capacity, delay, service, penalty))
            if result is None or not result_view(result, phase, scope)["downloadable"]:
                raise PreventUpdate
            name = f"orders_{phase}_{result['manifest']['run_id']}.csv"
            return dcc.send_data_frame(result["orders"].to_csv, name, index=False)
        except (ValueError, OSError, KeyError):
            raise PreventUpdate from None

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    load_dotenv(args.root / ".env", override=False)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    host = args.host or os.environ.get("DASH_HOST", "127.0.0.1")
    port = args.port or int(os.environ.get("DASH_PORT", "8050"))
    create_app(args.root).run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
