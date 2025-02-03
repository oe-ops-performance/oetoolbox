import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def event_summary_by_category(df):
    """note: df comes from output(s) of function "load_pi_events" in dataquery/pitools.py"""
    summary_cols = [
        "Site",
        "Category",
        "n_events",
        "n_assets",
        "total_duration",
        "approx_hours",
        "asset_names",
    ]
    summary_data = []
    for site in df.Site.unique():
        dfs_ = df.loc[df.Site.eq(site)].copy()
        for category in dfs_.Category.unique():
            dfs = dfs_.loc[dfs_.Category.eq(category)].copy()
            n_events = dfs.shape[0]
            n_assets = len(dfs.PrimaryElement.unique())
            unique_assets = list(dfs.PrimaryElement.unique())
            n_assets = len(unique_assets)
            asset_names = ", ".join(unique_assets)
            timedelta_ = pd.Timedelta(0)  # init
            # summary_data.append([site, category, n_events, n_assets, asset_names])
            for asset in unique_assets:
                dfsa = dfs.loc[dfs.PrimaryElement.eq(asset)].reset_index(drop=True).copy()
                for e in range(dfsa.shape[0]):
                    e_start, e_end = map(
                        pd.Timestamp, [dfsa.at[e, "EventStart"], dfsa.at[e, "EventEnd"]]
                    )
                    timedelta_ += pd.Timedelta(dfsa.at[e, "EventDuration"])

            summary_data.append(
                [
                    site,
                    category,
                    n_events,
                    n_assets,
                    str(timedelta_),
                    round(timedelta_.total_seconds() / 3600, 1),
                    asset_names,
                ]
            )

    dfsummary_ = pd.DataFrame(summary_data, columns=summary_cols)
    return dfsummary_


def inverter_event_subplots(df, df_events, q=True):
    """Creates subplot figure of time series plots with shaded regions added for corresponding events.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of inverter time-series data with pd.DatetimeIndex. Columns will be filtered
        for format: "OE.ActivePower_<inverter name>" (standard output of pi query). Inverter names
        will be derived from dataframe columns and are used for event matching criteria.
        Optional columns may be included with meter data (from PI, or utility) for comparison
        with sum of inverters - column name for PI meter data must include "OE_MeterMW", and column
        name for utility meter data must include "Utility".
    df_events : pd.DataFrame
        A dataframe of PI event frames corresponding to the inverter assets in df_inv. Must include
        the following columns: ['Site', 'EventStart', 'EventEnd', 'PrimaryElement']
        Note: portions of events outside of the data range will not be included in the output figure.
    q : bool, default True
        A "quiet" parameter to enable/disable status printouts (enabled if q=False)

    Returns
    -------
    plotly figure
        A subplot figure of time series data.
        The first (top) plot is for site-level generation data including sum of inverters, pi meter,
        and utility meter; ranges are highlighted for events with category="Site Offline Event"
        The remaining plots are for inverter-level generation data, with highlighted ranges for
        events with category="Inverter Offline Event"
    """
    qprint = lambda str_, end="\n": None if q else print(str_, end=end)
    meter_cols = [c for c in df.columns if any(i in c for i in ["OE_MeterMW", "Utility"])]
    inv_cols = list(filter(lambda c: c.startswith("OE.ActivePower"), df.columns))
    inv_names = [c.replace("OE.ActivePower_", "") for c in inv_cols]
    df["Sum_Inverter_MW"] = df[inv_cols].sum(axis=1).div(1e3).copy()

    n_rows = 1 + len(inv_cols)  # total gen + inv-level gen

    plot1_title_parts = ["Sum of Inverters"]
    if any("OE_MeterMW" in c for c in meter_cols):
        plot1_title_parts.append("PI Meter")
    if any("Utility" in c for c in meter_cols):
        plot1_title_parts.append("Utility Meter")
    plot1_title = ", ".join(plot1_title_parts) + " (MW)"

    subplot_titles = [plot1_title, *inv_cols]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.3 / n_rows,  # default 0.3/n_rows
    )
    hvtemp = (
        "<b>%{fullData.name}</b><br>Value: %{y:.2f}<br><i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
    )

    # ROW 1 (site-level generation)
    for col in ["Sum_Inverter_MW", *meter_cols]:
        fig.add_trace(
            go.Scattergl(
                x=df.index,
                y=df[col],
                name=col,
                mode="lines",
                hovertemplate=hvtemp,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # ROWS 2+ (inverter-level generation)
    for i, col in enumerate(inv_cols):
        fig.add_trace(
            go.Scattergl(
                x=df.index,
                y=df[col],
                name=col.replace("OE.ActivePower_", ""),
                mode="lines",
                hovertemplate=hvtemp,
                showlegend=False,
            ),
            row=i + 2,
            col=1,
        )

    fig.update_layout(
        font_size=9,
        height=100 * n_rows,
        margin=dict(t=20, b=20, l=20, r=20),
    )

    ymax_ = df[inv_cols].max().max() * 1.05
    for row_ in range(2, n_rows + 1):
        fig.update_yaxes(row=row_, col=1, range=[0, ymax_])

    fig.update_annotations(font_size=12, x=0, xanchor="left", xref="x domain")

    data_start = df.index.min()
    data_end = df.index.max()
    max_timedelta = data_end - data_start

    # find/filter events data using asset names in df
    site = df_events.at[0, "Site"]  # should be same for every event
    search_elements = [site, *inv_names]
    cond_1 = df_events["PrimaryElement"].isin(search_elements)
    cond_2 = df_events["EventStart"].lt(data_end)
    cond_3 = df_events["EventEnd"].gt(data_start)
    dfe = df_events.loc[(cond_1 & cond_2 & cond_3)].copy()
    if dfe.empty:
        qprint("!! no matching events found in supplied dataframe !!\n")
        return fig

    # overwrite "effective" columns (in case event search range does not exactly match data file)
    effective_cols = list(filter(lambda c: c.startswith("effective"), dfe.columns))
    if len(effective_cols) > 0:
        dfe = dfe.drop(columns=effective_cols)
    dfe["effective_start"] = dfe["EventStart"].mask(dfe["EventStart"].lt(data_start), data_start)
    dfe["effective_end"] = dfe["EventEnd"].mask(dfe["EventEnd"].gt(data_end), data_end)
    dfe["effective_duration"] = dfe["effective_end"] - dfe["effective_start"]

    n_events = dfe.shape[0]
    qprint(f">>> found {n_events} matching events")

    dfe = dfe.loc[dfe["effective_duration"].lt(max_timedelta)].copy()
    if dfe.shape[0] < n_events:
        n_removed = n_events - dfe.shape[0]
        qprint(f">>> NOTE: removed {n_removed} events with durations exceeding range of data")
        n_events = dfe.shape[0]
        qprint(f">>> updated number of events: {n_events}")

    vkwargs_ = dict(fillcolor="lightsalmon", layer="below", line_width=0, opacity=0.4)

    # Shaded regions for site-level outage events
    dfee = dfe.loc[dfe["Site"].eq(dfe["PrimaryElement"])].copy()
    if not dfee.empty:
        for start, end in zip(dfee["effective_start"], dfee["effective_end"]):
            fig.add_vrect(row=1, col=1, x0=start, x1=end, **vkwargs_)
        qprint(f"      - Site Outages (n={dfee.shape[0]})")

    for i, inv in enumerate(inv_names):
        c1_ = dfe["PrimaryElement"].eq(inv)
        c2_ = dfe["Category"].str.contains("Inverter Offline")
        dfee = dfe.loc[(c1_ & c2_)].copy()
        if not dfee.empty:
            subplot_row = i + 2
            for start, end in zip(dfee["effective_start"], dfee["effective_end"]):
                fig.add_vrect(row=subplot_row, col=1, x0=start, x1=end, **vkwargs_)
        qprint(f"      - {inv} Outage (n={dfee.shape[0]})")
    qprint(">>> done.\n")

    return fig
