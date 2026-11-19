"""Code-native charts and result views for the planning workbench.

Visual rules shared by every view (mirrors the tokens in assets/dashboard.css):

- One accent (dispatch blue) for the optimizer and forecasts; ink for actuals.
- Vermillion is reserved for exceptions: storage breaches and missed targets.
- Status is never colour alone: every state carries a symbol and words, and
  chart series differ by marker shape or line style as well as colour. The
  palette avoids pairing green with yellow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from supply_chain_opt_engine.dashboard_data import POLICY_LABELS
from supply_chain_opt_engine.optimization import MILP

INK = "#17202a"
INK_2 = "#3d4752"
MUTED = "#6b7480"
RULE = "#e3e0da"
GRID = "#eeebe5"
ACCENT = "#1f4f8f"
ALERT = "#b8401c"
BAND_OUTER = "rgba(31,79,143,.10)"
BAND_INNER = "rgba(31,79,143,.20)"
RECEIPTS = "#c9d3df"
FONT = "Lato, system-ui, sans-serif"
GRAPH = {"displayModeBar": False, "responsive": True}

# One fixed style per policy, used by every chart and table: colour, marker
# shape and a short direct label, so no policy is identified by colour alone.
POLICY_STYLE = {
    MILP: (ACCENT, "circle", "MILP"),
    "lightgbm_p90_weekly_order_up_to": (INK, "square", "P90 weekly"),
    "lightgbm_p90_order_up_to": ("#7d8590", "diamond", "P90 daily"),
    "lightgbm_p50_order_up_to": ("#5b9bd5", "triangle-up", "P50 daily"),
    "seasonal_naive_p50_order_up_to": ("#8e6bb0", "triangle-down", "SeasonalNaive"),
    "autoets_static_safety_stock": ("#a4632f", "cross", "AutoETS + SS"),
}
# Table keys use the same shape as the chart marker for each policy.
GLYPH = {
    "circle": "●",
    "square": "■",
    "diamond": "◆",
    "triangle-up": "▲",
    "triangle-down": "▼",
    "cross": "✚",
}
STORE_DASH = ["solid", "dash", "dot", "dashdot"]
STORE_COLOR = [INK, ACCENT, "#7d8590", "#8e6bb0"]


def policy_style(policy):
    base = policy.removesuffix("_rolling")
    return POLICY_STYLE.get(base, ("#7d8590", "circle", POLICY_LABELS.get(base, base)))


def policy_label(policy):
    base = policy.removesuffix("_rolling")
    label = POLICY_LABELS.get(base, base)
    return f"{label} (rolling)" if policy.endswith("_rolling") else label


def theme(fig, height=340):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=12, t=32, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=INK_2),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor="#cdc8bf", font=dict(family=FONT, size=12)),
        legend=dict(
            orientation="h",
            y=1.0,
            yanchor="bottom",
            x=0,
            xanchor="left",
            font=dict(size=12, color=INK_2),
            bgcolor="rgba(0,0,0,0)",
            itemclick=False,
            itemdoubleclick=False,
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        automargin=True,
        showline=True,
        linecolor="#cdc8bf",
        ticks="outside",
        ticklen=4,
        tickcolor="#cdc8bf",
        tickfont=dict(color=MUTED),
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
        automargin=True,
        tickfont=dict(color=MUTED),
        title_font=dict(color=MUTED, size=12),
        separatethousands=True,
    )
    return fig


def empty_figure(message, height=320):
    fig = theme(go.Figure(), height)
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color=MUTED, size=13),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def units(value, digits=0):
    return "—" if value is None or not np.isfinite(value) else f"{value:,.{digits}f}"


def money(value):
    return "—" if value is None or not np.isfinite(value) else f"${value:,.2f}"


def rate(value):
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.2%}"


def status(ok, good_text, bad_text):
    """A state shown as symbol plus words, never colour alone."""
    return html.Span(
        ["✓ " if ok else "✕ ", good_text if ok else bad_text],
        className="state state-ok" if ok else "state state-alert",
    )


# ---------------------------------------------------------------- forecasts


def forecast_figure(origin, observed, forecast):
    fig = go.Figure()
    for upper, fill, name in [("q95", BAND_OUTER, "P10–P95"), ("q90", BAND_INNER, "P10–P90")]:
        fig.add_trace(
            go.Scatter(
                x=forecast.ds,
                y=forecast.q10,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast.ds,
                y=forecast[upper],
                line=dict(width=0),
                fill="tonexty",
                fillcolor=fill,
                name=name,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=observed.ds,
            y=observed.y,
            name="Actual",
            line=dict(color=INK, width=1.5),
            hovertemplate="Actual %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.ds,
            y=forecast.point,
            name="Forecast (coherent center)",
            line=dict(color=ACCENT, width=2.4, dash="dash"),
            customdata=forecast[["q50", "q10", "q95"]].to_numpy(),
            hovertemplate=(
                "Center %{y:,.1f}<br>P50 %{customdata[0]:,.1f}"
                "<br>P10–P95 %{customdata[1]:,.1f}–%{customdata[2]:,.1f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=origin, line=dict(color="#a8a296", width=1))
    fig.add_annotation(
        x=origin,
        y=1,
        xref="x",
        yref="paper",
        text=f"As of {pd.Timestamp(origin):%b %d}",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        xshift=-6,
        font=dict(size=11, color=MUTED),
    )
    theme(fig, 300)
    limits = np.concatenate(
        [observed.y.to_numpy(), forecast[["q10", "q95", "point"]].to_numpy().ravel()]
    )
    lower = max(0.0, float(limits.min()) * 0.95)
    upper = max(lower + 1.0, float(limits.max()) * 1.05)
    fig.update_yaxes(title="Units per day", range=[lower, upper])
    return fig


def forecast_summary(origin, observed, forecast):
    """What this forecast says, in units. Point centers are coherent and add up;
    quantiles do not, so no summed interval is shown."""
    origin = pd.Timestamp(origin)
    ahead = forecast.point.sum()
    before = observed[(observed.ds > origin - pd.Timedelta(days=28)) & (observed.ds <= origin)]
    previous = before.y.sum() if len(before) == 28 else np.nan
    change = ahead / previous - 1 if previous and np.isfinite(previous) else np.nan
    peak = forecast.loc[forecast.point.idxmax()] if len(forecast) else None
    realized = observed[observed.ds > origin]
    rows = [
        ("Forecast, next 28 days", f"{units(ahead)} units", "Sum of daily coherent centers"),
        ("Actual, previous 28 days", f"{units(previous)} units", None),
        (
            "Change",
            "—" if not np.isfinite(change) else f"{change:+.1%}",
            "Forecast against the previous 28 days",
        ),
        (
            "Peak forecast day",
            "—" if peak is None else f"{units(peak.point, 1)} units",
            None if peak is None else f"{peak.ds:%a %b %d}",
        ),
    ]
    if len(realized) == len(forecast) and len(realized):
        merged = forecast.merge(realized, on="ds")
        error = (merged.y - merged.point).abs().sum() / merged.y.sum() if merged.y.sum() else np.nan
        rows.append(("Actual, these 28 days", f"{units(merged.y.sum())} units", "Holdout fold"))
        rows.append(("WMAPE for this node", rate(error), "Absolute error over actual units"))
    else:
        rows.append(("Actual, these 28 days", "Not yet observed", "Latest origin has no outcome"))
    return html.Dl(
        [
            html.Div(
                [
                    html.Dt(label),
                    html.Dd(value, className="figure"),
                    *([html.Dd(note, className="figure-note")] if note else []),
                ],
                className="summary-row",
            )
            for label, value, note in rows
        ],
        className="summary-list",
    )


def shap_figure(frame):
    if frame is None:
        return empty_figure("No explanation sample was saved for this tuning fold.", 260)
    names = {
        "rolling_mean_7": "Recent 7-day mean",
        "rolling_mean_14": "Recent 14-day mean",
        "rolling_mean_28": "Recent 28-day mean",
        "series_id": "Series identity",
        "horizon": "Days ahead",
        "lag_7": "7-day lag",
        "price_relative_to_52w": "Relative price",
        "snap_CA": "SNAP calendar",
        "horizon_day": "Days ahead",
    }
    data = frame.iloc[::-1]
    fig = theme(
        go.Figure(
            go.Bar(
                x=data.mean_absolute_shap,
                y=[names.get(s, s.replace("_", " ").capitalize()) for s in data.feature],
                orientation="h",
                # Muted: explanatory, so it should not compete with the forecast.
                marker_color="#8b9cb4",
                hovertemplate="%{y}: %{x:.3f} units<extra></extra>",
            )
        ),
        260,
    )
    fig.update_layout(margin=dict(l=8, r=12, t=8, b=8), hovermode="closest", bargap=0.4)
    fig.update_xaxes(title="Mean |SHAP|, units", showgrid=True, gridcolor=GRID, showline=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_2))
    return fig


def table(headers, rows, *, caption=None, numeric=None, highlight=None, scroll=True):
    numeric = set(range(1, len(headers))) if numeric is None else set(numeric)
    body = []
    for index, row in enumerate(rows):
        body.append(
            html.Tr(
                [
                    html.Td(value, className="num" if col in numeric else None)
                    for col, value in enumerate(row)
                ],
                className="row-highlight" if highlight == index else None,
            )
        )
    return html.Div(
        html.Table(
            [
                *([html.Caption(caption)] if caption else []),
                html.Thead(
                    html.Tr(
                        [
                            html.Th(h, scope="col", className="num" if c in numeric else None)
                            for c, h in enumerate(headers)
                        ]
                    )
                ),
                html.Tbody(body),
            ]
        ),
        className="table-scroll" if scroll else "table-wrap",
    )


def forecast_table(frame):
    return table(
        ["Date", "Center", "P10", "P50", "P90", "P95"],
        [
            [
                r.ds.strftime("%a %b %d"),
                *[f"{getattr(r, c):,.1f}" for c in ["point", "q10", "q50", "q90", "q95"]],
            ]
            for r in frame.itertuples()
        ],
        caption="Units per day. Quantiles are marginal and do not add across the hierarchy.",
    )


# ---------------------------------------------------------------- inventory


def inventory_figure(trace):
    grouped = trace.groupby("ds", as_index=False)[["inventory", "arrivals"]].sum()
    fig = go.Figure(
        go.Bar(
            x=grouped.ds,
            y=grouped.arrivals,
            name="Receipts",
            marker_color=RECEIPTS,
            hovertemplate="Receipts %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped.ds,
            y=grouped.inventory,
            name="Ending stock",
            line=dict(color=ACCENT, width=2.2),
            hovertemplate="Ending stock %{y:,.0f}<extra></extra>",
        )
    )
    theme(fig, 250)
    fig.update_layout(bargap=0.25)
    fig.update_yaxes(title="Units, all stores", rangemode="tozero")
    return fig


def storage_figure(trace, storage):
    """Daily end-of-day storage use against each store's capacity."""
    if storage is None:
        return None
    joined = trace.merge(storage["catalog"], on="unique_id", validate="many_to_one")
    joined["volume"] = joined.inventory * joined.unit_volume_m3
    used = joined.groupby(["store_id", "ds"], as_index=False).volume.sum()
    fig = go.Figure()
    peak = 1.0
    for index, (store, frame) in enumerate(used.groupby("store_id")):
        capacity = storage["capacity"][store]
        share = frame.volume / capacity
        peak = max(peak, float(share.max()))
        fig.add_trace(
            go.Scatter(
                x=frame.ds,
                y=share,
                name=store,
                line=dict(
                    color=STORE_COLOR[index % len(STORE_COLOR)],
                    width=1.8,
                    dash=STORE_DASH[index % len(STORE_DASH)],
                ),
                customdata=np.c_[frame.volume, np.full(len(frame), capacity)],
                hovertemplate=(
                    f"{store} %{{y:.0%}} · %{{customdata[0]:,.1f}} of "
                    "%{customdata[1]:,.0f} m³<extra></extra>"
                ),
            )
        )
        over = frame[share > 1 + 1e-7]
        if len(over):
            fig.add_trace(
                go.Scatter(
                    x=over.ds,
                    y=over.volume / capacity,
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=10, color=ALERT),
                    name=f"{store} over capacity",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    fig.add_hline(y=1, line=dict(color=ALERT, width=1.2))
    fig.add_annotation(
        x=0,
        y=1,
        xref="paper",
        yref="y",
        text="Capacity",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=11, color=ALERT),
    )
    theme(fig, 250)
    fig.update_yaxes(title="Share of capacity", tickformat=".0%", range=[0, max(1.1, peak * 1.08)])
    return fig


def tradeoff_figure(selected, saved, target, phase):
    """Display observed tradeoffs; a forecast floor is never a realized guarantee."""
    fig = theme(go.Figure(), 360)
    for row in selected.itertuples():
        fill = getattr(row, "minimum_store_fill_rate", row.fill_rate)
        if fill is None or not np.isfinite(fill):
            continue
        eligible = row.capacity_feasible and getattr(row, "service_feasible", True)
        color, symbol, _ = policy_style(row.policy)
        fig.add_trace(
            go.Scatter(
                x=[fill],
                y=[row.total_cost],
                # Shape and legend identify the policy; direct labels collided
                # when baselines cluster, so they are left to the hover text.
                mode="markers",
                name=policy_label(row.policy),
                marker=dict(
                    size=12,
                    color=color,
                    symbol=symbol if eligible else f"{symbol}-open",
                    line=dict(width=2, color=color),
                ),
                hovertemplate=(
                    "%{fullData.name}<br>Lowest store fill %{x:.2%}<br>Cost $%{y:,.2f}"
                    + (
                        "<br>✓ Meets service and storage"
                        if eligible
                        else "<br>✕ Fails service or storage"
                    )
                    + "<extra></extra>"
                ),
            )
        )
    if saved is not None and len(saved):
        for good in [True, False]:
            ok = saved.capacity_feasible & saved.service_feasible
            points = saved[ok] if good else saved[~ok]
            if points.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=points.minimum_store_fill_rate,
                    y=points.total_cost,
                    mode="markers",
                    name="Saved service targets" if good else "Saved targets that failed",
                    customdata=points[["minimum_fill_rate"]].to_numpy(),
                    marker=dict(
                        size=11,
                        symbol="star" if good else "star-open",
                        color=MUTED,
                        line=dict(width=1.5, color=MUTED),
                    ),
                    hovertemplate=(
                        "Planning target %{customdata[0]:.0%}<br>Lowest store fill %{x:.2%}"
                        "<br>Cost $%{y:,.2f}<extra></extra>"
                    ),
                )
            )
    if target:
        fig.add_vline(x=target, line=dict(color="#a8a296", width=1, dash="dot"))
        fig.add_annotation(
            x=target,
            y=1,
            xref="x",
            yref="paper",
            text=f"{target:.0%} target",
            showarrow=False,
            xanchor="right",
            yanchor="top",
            xshift=-4,
            font=dict(size=11, color=MUTED),
        )
    fig.update_layout(
        hovermode="closest",
        showlegend=True,
        legend=dict(orientation="h", y=-0.2, yanchor="top", x=0, xanchor="left"),
        margin=dict(l=8, r=12, t=12, b=8),
    )
    prefix = "Realized" if phase == "holdout" else "Projected"
    fig.update_xaxes(title=f"{prefix} lowest store fill rate", tickformat=".0%", range=[0, 1.04])
    fig.update_yaxes(title="Total cost, USD", rangemode="tozero", tickprefix="$")
    return fig


def result_view(result, phase, scope):
    evaluation = "realized" if phase == "holdout" else "forecast"
    metrics = result["metrics"]
    selected = metrics[metrics.evaluation.eq(evaluation) & metrics.scope.eq(scope)]
    primary = selected[selected.policy.eq(MILP)].iloc[0]
    target = result["manifest"]["config"].get("minimum_fill_rate", 0.0)
    days = int(primary.days)
    trace = result["trace"]
    trace = trace[trace.evaluation.eq(evaluation) & trace.policy.eq(MILP)]
    dates = sorted(trace.ds.unique())[:days]
    trace = trace[trace.ds.isin(dates)]
    ordered = selected.assign(
        _eligible=selected.capacity_feasible
        & selected.get("service_feasible", pd.Series(True, index=selected.index)),
        _plan=selected.policy.eq(MILP),
    ).sort_values(["_plan", "_eligible", "total_cost"], ascending=[False, False, True])
    rows = []
    for r in ordered.itertuples():
        over = getattr(r, "max_capacity_violation_m3", np.nan)
        delta = r.total_cost / primary.total_cost - 1 if primary.total_cost else np.nan
        rows.append(
            [
                html.Span(
                    [
                        html.Span(
                            GLYPH.get(policy_style(r.policy)[1], "●"),
                            className="policy-key",
                            style={"color": policy_style(r.policy)[0]},
                            **{"aria-hidden": "true"},
                        ),
                        policy_label(r.policy),
                    ],
                    className="policy-name",
                ),
                money(r.total_cost),
                "Plan"
                if r.policy == MILP
                else ("—" if not np.isfinite(delta) else f"{delta:+.0%}"),
                rate(r.fill_rate),
                rate(getattr(r, "minimum_store_fill_rate", r.fill_rate)),
                status(
                    bool(r.capacity_feasible),
                    "Within limit",
                    f"Exceeds capacity +{over:,.1f} m³"
                    if np.isfinite(over)
                    else "Exceeds capacity",
                ),
                status(bool(getattr(r, "service_feasible", True)), "Met", "Missed"),
            ]
        )
    comparison = table(
        [
            "Policy",
            "Total cost",
            "vs plan",
            "Fill rate" if phase == "holdout" else "Projected fulfillment",
            "Lowest store",
            "Storage",
            "Service target",
        ],
        rows,
        numeric={1, 2, 3, 4},
        highlight=0,
        scroll=False,
    )
    breakdown = table(
        ["Cost component", "USD"],
        [
            ["Holding", money(primary.holding_cost)],
            ["Stockout penalties", money(primary.stockout_cost)],
            ["Ordering fees", money(primary.fixed_order_cost)],
            [html.Strong("Total"), html.Strong(money(primary.total_cost))],
        ],
        scroll=False,
    )
    solver = table(
        ["Measure", "Result"],
        [
            ["Certified gap", rate(result["gap"])],
            [
                "Capacity",
                status(bool(primary.capacity_feasible), "Within limit", "Exceeds capacity"),
            ],
            ["Decision horizon", "28 days"],
            ["Tail", f"{result['manifest']['config']['tail_days']} days"],
            ["Minimum store fill target", rate(target)],
            [
                "Shortage penalty",
                f"{result['manifest']['config'].get('shortage_penalty_multiplier', 1):g}x",
            ],
        ],
        scroll=False,
    )
    warning = []
    if primary.order_count == 0:
        warning.append(
            "No new orders selected. "
            + (
                "Existing inventory and receipts meet the projected service floor."
                if target
                else "Cost-only planning has no minimum service constraint."
            )
        )
    if not getattr(primary, "service_feasible", True):
        warning.append(
            "The realized plan misses the service target. The forecast constraint "
            "does not guarantee actual fulfillment. Download is disabled for this window."
        )
    if not primary.capacity_feasible:
        warning.append(
            "This realized replay exceeds storage. Download is disabled for this window."
        )
    if not selected.capacity_feasible.all():
        warning.append(
            "A comparison policy exceeds storage in this window. "
            "Its scores are diagnostic, not a feasible operating plan."
        )
    if phase == "latest":
        warning.append(
            "Forecast-demand replay. Realized future demand and fill rate are unavailable."
        )
    return {
        "primary": primary,
        "downloadable": bool(
            primary.capacity_feasible and getattr(primary, "service_feasible", True)
        ),
        "figure": inventory_figure(trace),
        "storage": storage_figure(trace, result.get("storage")),
        "comparison": comparison,
        "tradeoff": tradeoff_figure(selected, result.get("tradeoff"), target, phase),
        "breakdown": breakdown,
        "solver": solver,
        "warning": " ".join(warning),
        "warnings": warning,
        "target": target,
        "fill_label": "Realized fill rate" if phase == "holdout" else "Projected fulfillment",
        "caption": "Holdout, realized demand"
        if phase == "holdout"
        else "Latest origin, forecast demand",
    }


def section(title, body, note=None, aside=None, className=""):
    return html.Section(
        [
            html.Div(
                [
                    html.H2(title),
                    *([html.Span(note, className="section-note")] if note else []),
                    *([aside] if aside is not None else []),
                ],
                className="section-head",
            ),
            body,
        ],
        className=f"sheet-section {className}".strip(),
    )


def decision(verdict, reason, tone, figures, action):
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "✓" if tone == "ok" else "✕" if tone == "hold" else "·",
                                className="verdict-mark",
                                **{"aria-hidden": "true"},
                            ),
                            html.Span(verdict, className="verdict-text"),
                        ],
                        className="verdict-line",
                    ),
                    html.P(reason, className="verdict-reason"),
                ],
                className=f"verdict verdict-{tone}",
            ),
            figures,
            action,
        ],
        className="decision",
    )


def download_action(disabled, note):
    return html.Div(
        [
            html.Button(
                "Download orders",
                id="download-orders",
                disabled=disabled,
                className="button button-primary",
            ),
            html.Div(note, className="action-note"),
        ],
        className="decision-action",
    )


def headline_figures(cost, cost_label, rows):
    return html.Div(
        [
            html.Div(
                [html.Div(cost_label, className="hero-label"), html.Div(cost, className="hero")],
                className="hero-block",
            ),
            html.Dl(
                [
                    html.Div(
                        [
                            html.Dt(label),
                            html.Dd(value, className="figure"),
                            *([html.Dd(note, className="figure-note")] if note else []),
                        ],
                        className="fact",
                    )
                    for label, value, note in rows
                ],
                className="fact-list",
            ),
        ],
        className="decision-figures",
    )


def blank_results(message, failed=False):
    return html.Div(
        [
            decision(
                "Scenario failed" if failed else "No plan for these settings",
                message,
                "hold" if failed else "idle",
                headline_figures(
                    "—",
                    "Total cost",
                    [
                        ("Fill rate", "—", None),
                        ("Avg inventory value", "—", None),
                        ("Orders", "—", None),
                    ],
                ),
                download_action(True, "No current result"),
            ),
            html.Div(
                [
                    html.Div(
                        "Storage headroom, stock and policy comparison", className="empty-title"
                    ),
                    html.Div(
                        "They appear here once the optimizer has a plan for these settings. "
                        "Run it from the scenario panel.",
                        className="empty-text",
                    ),
                ],
                className="empty-state",
            ),
        ],
        className="result-stack",
    )


def scenario_results(result, phase, scope):
    view = result_view(result, phase, scope)
    r = view["primary"]
    service_ok = bool(getattr(r, "service_feasible", True))
    lowest = getattr(r, "minimum_store_fill_rate", r.fill_rate)
    over = getattr(r, "max_capacity_violation_m3", np.nan)
    window = "28-day window" if scope == "decision_window" else "42 days incl. tail"
    reasons = []
    if not service_ok:
        reasons.append(
            f"misses the {rate(view['target'])} fill target (lowest store {rate(lowest)})"
        )
    if not r.capacity_feasible:
        reasons.append(
            f"exceeds storage by up to {over:,.1f} m³" if np.isfinite(over) else "exceeds storage"
        )
    if view["downloadable"]:
        verdict, tone = "Ready to release", "ok"
        reason = (
            f"Every store meets the {rate(view['target'])} fill target and stock stays within "
            f"storage over the {window}."
        )
        if phase == "latest":
            reason += " Projected on forecast demand; actual outcomes are not yet known."
    else:
        verdict, tone = "Hold: do not release", "hold"
        reason = "This plan " + " and ".join(reasons) + f" over the {window}."
    orders = result.get("orders")
    value = (
        orders.purchase_value.sum()
        if orders is not None and "purchase_value" in orders and len(orders)
        else 0.0
    )
    figures = headline_figures(
        money(r.total_cost),
        f"Total cost · {window}",
        [
            (view["fill_label"], rate(r.fill_rate), f"Lowest store {rate(lowest)}"),
            ("Avg inventory value", money(r.average_inventory_value), "Mean daily stock at cost"),
            ("Orders", f"{int(r.order_count):,}", f"{money(value)} purchase value"),
        ],
    )
    storage = (
        dcc.Graph(
            figure=view["storage"],
            responsive=True,
            config=GRAPH,
            className="chart",
            style={"height": "260px"},
        )
        if view["storage"] is not None
        else html.Div(
            "Storage detail is unavailable: this result does not reference verifiable "
            "scenario files.",
            className="empty-text inline-empty",
        )
    )
    return html.Div(
        [
            decision(
                verdict,
                reason,
                tone,
                figures,
                download_action(not view["downloadable"], "Purchase-order CSV for the plan"),
            ),
            *(
                [
                    html.Ul(
                        [html.Li(text) for text in view["warnings"]],
                        className="notes",
                        role="status",
                    )
                ]
                if view["warnings"]
                else []
            ),
            html.Div(
                [
                    section(
                        "Storage headroom",
                        storage,
                        "End-of-day volume as a share of each store's capacity",
                    ),
                    section(
                        "Stock and receipts",
                        dcc.Graph(
                            figure=view["figure"],
                            responsive=True,
                            config=GRAPH,
                            className="chart",
                            style={"height": "260px"},
                        ),
                        view["caption"],
                    ),
                ],
                className="pair",
            ),
            section(
                "Policy comparison",
                view["comparison"],
                "Same inventory, receipts, capacity and economics for every policy",
            ),
            html.Div(
                [
                    section(
                        "Cost against service",
                        html.Div(
                            [
                                dcc.Graph(
                                    figure=view["tradeoff"],
                                    responsive=True,
                                    config=GRAPH,
                                    style={"height": "380px"},
                                ),
                                html.P(
                                    "Filled markers meet the service target and storage; open "
                                    "markers fail at least one. Stars are saved service targets "
                                    "under the same economics and capacity. Costs exclude "
                                    "purchase cost.",
                                    className="chart-caption",
                                ),
                            ]
                        ),
                    ),
                    html.Div(
                        [
                            section("Cost breakdown", view["breakdown"], "Optimizer plan"),
                            section("Solver", view["solver"], "HiGHS MILP"),
                        ],
                        className="stack",
                    ),
                ],
                className="pair pair-wide",
            ),
            html.Div(
                [
                    "Result ",
                    html.Code(result["manifest"]["run_id"]),
                    " · synthetic economics · historical results are exploratory.",
                ],
                className="source-line",
            ),
        ],
        className="result-stack",
    )
