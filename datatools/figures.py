"""
Visualizations for CombinerAnalyzer results.

Two complementary views:

- 'plot_combiner_strips': per-inverter time series (measured + possible kW)
  with a heatmap strip of per-combiner hourly flag categories below. Shows
  *when* each combiner went bad and how it relates to its parent inverter.

- 'plot_combiner_loss_bars': stacked MWh-loss bars per combiner, faceted by
  inverter. Shows *which combiner contributed how much* over the analysis window.
"""

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .inverter_analysis import FLAG_COLORS

HOURLY_CATEGORIES_ORDER = ["curtailment", "offline", "comms", "calibration", "underperform"]
ALL_CATEGORIES_ORDER = HOURLY_CATEGORIES_ORDER + [
    "tracker_east",
    "tracker_center",
    "tracker_west",
]

InverterArg = Optional[Union[str, Sequence[str]]]


def _resolve_inverters(ca, inverters: InverterArg, default_source: str) -> List[str]:
    if inverters == "all":
        return list(ca.inverter_names)
    if isinstance(inverters, str):
        return [inverters]
    if inverters is not None:
        return list(inverters)
    if default_source == "flagged":
        flagged = {inv for (inv, _cmb) in ca.flagged_combiners}
        return [inv for inv in ca.inverter_names if inv in flagged]
    if default_source == "totals":
        invs = ca.to_dataframe().index.get_level_values("inverter").unique()
        return [inv for inv in ca.inverter_names if inv in set(invs)]
    return list(ca.inverter_names)


def _discrete_colorscale(categories: Sequence[str]):
    """Step colorscale + value→band-center map for a categorical heatmap."""
    n = len(categories)
    scale = []
    for i, cat in enumerate(categories):
        color = FLAG_COLORS.get(cat, "#cccccc")
        scale.append([i / n, color])
        scale.append([(i + 1) / n, color])
    cat_to_num = {cat: i + 0.5 for i, cat in enumerate(categories)}
    return scale, cat_to_num, n


def plot_combiner_strips(
    ca,
    inverters: InverterArg = None,
    resample: Optional[str] = "15min",
    height_per_inv: int = 160,
):
    """Per-inverter measured/possible trace with a combiner-flag heatmap strip.

    Each inverter occupies one row in the figure. The kW line trace
    (measured + total possible) is plotted on the primary (left) y-axis;
    the per-combiner flag heatmap is plotted on a secondary (right) y-axis
    with combiner IDs as categorical ticks. The heatmap is added first so
    it sits behind the line.

    Parameters
    ----------
    ca : CombinerAnalyzer
        Must have 'run_analysis()' already called.
    inverters : list[str] | "all" | None
        'None' (default) plots only inverters that have at least one flagged
        combiner. '"all"' plots every inverter.
    resample : str | None
        Pandas freq for downsampling the line traces. Default '"15min"';
        'None' keeps native frequency. The heatmap stays hourly regardless.
    height_per_inv : int
        Pixel height per inverter row.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    inv_names = _resolve_inverters(ca, inverters, default_source="flagged")
    if not inv_names:
        raise ValueError("No inverters to plot (no flagged combiners — pass inverters='all').")

    inv_meas = ca.df_inv_measured_kw
    inv_poss = ca.df_possible_kw.T.groupby(level="inverter").sum().T
    if resample:
        inv_meas = inv_meas.resample(resample).mean()
        inv_poss = inv_poss.resample(resample).mean()

    flags_df = ca.hourly_flags
    colorscale, cat_to_num, n_cat = _discrete_colorscale(HOURLY_CATEGORIES_ORDER)

    n_inv = len(inv_names)
    specs = [[{"secondary_y": True}] for _ in range(n_inv)]
    fig = make_subplots(
        rows=n_inv,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.3 / n_inv,
        specs=specs,
        subplot_titles=[f"<b>{inv}</b>" for inv in inv_names],
    )

    htmpl_heat = "combiner <b>%{y}</b><br>%{x|%Y-%m-%d %H:%M}" "<br>%{customdata}<extra></extra>"

    for k, inv in enumerate(inv_names):
        row = k + 1

        if not flags_df.empty:
            inv_cols = [c for c in flags_df.columns if c[0] == inv]
            if inv_cols:
                cmb_ids = [c[1] for c in inv_cols]
                sub = flags_df[inv_cols]
                z = sub.apply(lambda col: col.map(cat_to_num))
                fig.add_trace(
                    go.Heatmap(
                        x=sub.index,
                        y=cmb_ids,
                        z=z.T.to_numpy(),
                        customdata=sub.T.to_numpy(),
                        zmin=0,
                        zmax=n_cat,
                        colorscale=colorscale,
                        showscale=False,
                        hovertemplate=htmpl_heat,
                        hoverongaps=False,
                        xgap=0,
                        ygap=1,
                        opacity=0.55,
                    ),
                    row=row,
                    col=1,
                    secondary_y=True,
                )

        if inv in inv_poss.columns:
            fig.add_trace(
                go.Scattergl(
                    x=inv_poss.index,
                    y=inv_poss[inv].to_numpy(),
                    name=f"{inv} possible",
                    line=dict(color="#bdbdbd", width=1, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=1,
                secondary_y=False,
            )
        if inv in inv_meas.columns:
            fig.add_trace(
                go.Scattergl(
                    x=inv_meas.index,
                    y=inv_meas[inv].to_numpy(),
                    name=inv,
                    line=dict(color="#1929f8", width=1.2),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=1,
                secondary_y=False,
            )

    legend_traces = [
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=FLAG_COLORS.get(cat, "#cccccc"), symbol="square"),
            name=cat,
            showlegend=True,
        )
        for cat in HOURLY_CATEGORIES_ORDER
    ]
    fig.add_traces(legend_traces, rows=[1] * len(legend_traces), cols=[1] * len(legend_traces))

    fig.update_layout(
        height=height_per_inv * n_inv + 80,
        margin=dict(t=50, l=60, r=20, b=30),
        title=dict(
            text=f"<b>{ca.site.name} — Inverter Power + Combiner Flag Strips</b>",
            x=0.0,
            xanchor="left",
        ),
        plot_bgcolor="#fff",
        paper_bgcolor="#f8f9fa",
        font_size=10,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, font_size=10),
    )
    fig.update_xaxes(linecolor="#ced4da", gridcolor="#eee", tickformat="%m/%d", tickfont_size=9)
    for k in range(n_inv):
        row = k + 1
        fig.update_yaxes(
            title=dict(text="kW", standoff=0),
            row=row,
            col=1,
            secondary_y=False,
            rangemode="nonnegative",
            tickfont_size=9,
            linecolor="#ced4da",
            gridcolor="#eee",
        )
        fig.update_yaxes(
            row=row,
            col=1,
            secondary_y=True,
            autorange="reversed",
            type="category",
            showticklabels=False,
            showgrid=False,
            showline=False,
            ticks="",
        )
    return fig


def plot_combiner_loss_bars(
    ca,
    inverters: InverterArg = None,
    facet_cols: int = 3,
    height_per_row: int = 240,
    sort_combiners: bool = True,
):
    """Stacked MWh-loss bars per combiner, faceted by inverter.

    Reads 'ca.to_dataframe()' (totals MWh per (inv, cmb) x category) and lays
    out one subplot per inverter, with combiners on the x-axis and stacked
    category bars on the y-axis.

    Parameters
    ----------
    ca : CombinerAnalyzer
    inverters : list[str] | "all" | None
        'None' (default) plots only inverters that contributed any flagged
        loss. '"all"' plots every inverter regardless.
    facet_cols : int
        Columns in the subplot grid.
    height_per_row : int
        Pixel height per subplot row.
    sort_combiners : bool
        If True, sort combiners within each inverter by total loss (desc).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = ca.to_dataframe()
    if df.empty:
        raise ValueError("No flags recorded — call run_analysis() first.")

    inv_names = _resolve_inverters(ca, inverters, default_source="totals")
    if not inv_names:
        raise ValueError("No inverters with flagged combiners.")

    cats = [c for c in ALL_CATEGORIES_ORDER if c in df.columns]
    df = df.reindex(columns=cats, fill_value=0.0)

    n = len(inv_names)
    cols = min(facet_cols, n)
    rows = int(np.ceil(n / cols))

    h_spacing = min(0.07, 0.8 / max(cols - 1, 1))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"<b>{inv}</b>" for inv in inv_names],
        horizontal_spacing=h_spacing,
        vertical_spacing=0.4 / rows,
    )

    seen_cats: set = set()
    inv_index = df.index.get_level_values("inverter")
    for k, inv in enumerate(inv_names):
        r, c = k // cols + 1, k % cols + 1
        if inv not in set(inv_index):
            continue
        sub = df.loc[inv_index == inv].copy()
        sub.index = sub.index.get_level_values("combiner")
        if sort_combiners:
            sub = sub.loc[sub.sum(axis=1).sort_values(ascending=False).index]
        x_vals = sub.index.astype(str).tolist()
        for cat in cats:
            y = sub[cat].to_numpy()
            if not np.any(y > 0):
                continue
            show_legend = cat not in seen_cats
            seen_cats.add(cat)
            fig.add_trace(
                go.Bar(
                    x=x_vals,
                    y=y,
                    name=cat,
                    legendgroup=cat,
                    marker_color=FLAG_COLORS.get(cat, "#cccccc"),
                    showlegend=show_legend,
                    hovertemplate=(
                        f"<b>{inv}</b> · combiner %{{x}}"
                        f"<br>{cat}: %{{y:.2f}} MWh<extra></extra>"
                    ),
                ),
                row=r,
                col=c,
            )

    fig.update_layout(
        barmode="stack",
        height=height_per_row * rows + 80,
        margin=dict(t=70, l=60, r=20, b=40),
        title=dict(
            text=f"<b>{ca.site.name} — Combiner Loss by Category (MWh)</b>",
            x=0.02,
            xanchor="left",
            yanchor="bottom",
            y=1.02,
        ),
        plot_bgcolor="#fff",
        paper_bgcolor="#f8f9fa",
        font_size=10,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.95, font_size=10
        ),
    )
    fig.update_xaxes(tickfont_size=8, tickangle=45, linecolor="#ced4da")
    fig.update_yaxes(
        title=dict(text="MWh", standoff=0),
        tickfont_size=9,
        rangemode="tozero",
        gridcolor="#eee",
    )
    return fig
