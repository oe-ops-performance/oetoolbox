import calendar
import numpy as np
import pandas as pd
from pathlib import Path
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


config_ = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "zoom2d",
        "zoomIn2d",
        "zoomOut2d",
        "pan2d",
        "select2d",
        "lasso2d",
        "autoScale2d",
    ],
}
colors_ = plotly.colors.qualitative.Plotly

HOVERTEMPLATES = {
    "bar": (
        "<b>%{fullData.name}</b><br>Generation: %{y:.2f} MWh<br>"
        "<i>%{x|%Y-%m-%d}</i><extra></extra>"
    ),
    "timeseries": (
        "<b>%{fullData.name}</b><br>Power: %{y:.2f} kW<br>"
        "<i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
    ),
    "scatter": (
        "<b>%{fullData.name}</b><br>POA: %{x:.2f} kW/m2<br>"
        "Power: %{y:.2f} kW<br><i>%{customdata|%Y-%m-%d %H:%M}</i><extra></extra>"
    ),
}


SCATTER_KWARGS = {
    "Possible_MW": dict(
        opacity=0.3,
        marker=dict(color="#777", size=8, symbol="square", line=dict(color="#333", width=1)),
    ),
    "Inv_Total_MW": dict(
        opacity=1,
        marker=dict(
            color="#1929f8", size=8, symbol="square-open", line=dict(color="#1929f8", width=1.5)
        ),
    ),
    "PI_Meter_MW": dict(
        opacity=0.9,
        marker=dict(
            color=colors_[2], size=6, symbol="x-thin", line=dict(color="limegreen", width=1.5)
        ),
    ),
    "Util_Meter_MW": dict(
        opacity=1,
        marker=dict(color=colors_[1], size=6, symbol="circle", line=dict(color="#000", width=0.5)),
    ),
}

# def solar_scatter_site_level(df: pd.DataFrame, poa_data: pd.Series) -> go.Scattergl:
#     """data should already be resampled (normally using 15min)"""
#     if not all(c in df.columns for c in SCATTER_KWARGS.keys()):
#         raise KeyError("One or more columns missing from data.")
#     elif len(df) != len(poa_data):
#         raise ValueError("Length mismatch between generation data and poa data.")
#     return go.Scattergl(

#     )


# for solar sites w/ flashreport data
def solar_summary_subplots(site: str, year: int, month: int, data: dict[str, pd.DataFrame]):
    """data is from output of FlashReportGenerator.get_data_for_summary_plot
    -> keys: ["inverters", "meter", "pvlib", "site_level", "kpis"] (+ "utility" optional)
    """
    # validate data input
    required_keys = ["inverters", "meter", "pvlib", "site_level", "kpis"]
    if not all(k in data.keys() for k in required_keys):
        raise KeyError(f"Invalid data input; missing one or more keys. {required_keys = }")

    # check for optional utility meter data (hourly)
    df_util = data.get("utility", pd.DataFrame())  # utility meter data is optional

    # create site totals df from original data (i.e. not resampled)
    dfi, dfm, dfp = map(lambda key: data[key].copy(), required_keys[:-1])
    if not len(set(pd.infer_freq(df.index) for df in [dfi, dfm, dfp])) > 1:
        raise ValueError("One or more dataframes have different frequencies.")
    poacol = "POA" if "POA" in dfp.columns else "POA_DTN"
    if poacol not in dfp.columns:
        raise ValueError("Could not find valid poa column in pvlib data.")

    dfi = dfi.filter(like="ActivePower")
    dfm = dfm.filter(like="Meter")
    if dfi.shape[1] != dfp.filter(like="Possible_Power").shape[1]:
        raise ValueError("Mismatch between pvlib and inverter data columns.")

    poss_col, act_col, pi_col, util_col = [
        "Possible_MW",
        "Inv_Total_MW",
        "PI_Meter_MW",
        "Util_Meter_MW",
    ]

    df_site = dfm.copy()
    df_site.columns = [pi_col]
    df_site[act_col] = dfi.sum(axis=1).div(1e3)
    df_site[poss_col] = dfp.filter(like="Possible_Power").sum(axis=1).div(1e3)
    df_site[poacol] = dfp[poacol].copy()

    native_freq = df_site.index.diff().min()

    # create hourly dataframe for site-level generation time series
    df_site_hourly = df_site.resample("1h").mean()
    if not df_util.empty():
        df_site_hourly[util_col] = df_util[site].copy()

    # convert all interval data to hourly frequency
    df_inv = dfi.resample("1h").mean().copy()
    df_meter = dfm.resample("1h").mean().copy()
    df_pvlib = dfp.resample("1h").mean().copy()
    df_kpis = data["kpis"]

    """PLOT
        row 1: kpi table
        row 2: site-level generation bar chart (daily totals)
        row 3: site-level generation time series
        remaining rows: inverter-level 
    """
    # define subplot figure
    n_inv = len(df_inv.columns)
    n_rows = n_inv + 3  # top row (scatter, kpi table) + meter bar chart + meter time series
    plot1_height = 380
    mplot_height = 100  # for rows 2 and 3
    plotX_height = 80  # for inverter plots
    total_height = plot1_height + mplot_height + mplot_height + plotX_height * (n_inv)

    p1_ratio = plot1_height / total_height
    p2_ratio = mplot_height / total_height
    pX_ratio = plotX_height / total_height
    row_height_list = [p1_ratio, p2_ratio, p2_ratio] + [pX_ratio] * (n_inv)

    fig = make_subplots(
        rows=n_rows,
        row_heights=row_height_list,
        cols=3,
        column_widths=[0.1, 0.5, 0.4],
        horizontal_spacing=0.035,  # default 0.2/n_cols
        vertical_spacing=0.2 / n_rows,  # default 0.3/n_rows
        specs=[[{"colspan": 2}, None, {"type": "table"}]]
        + [[{}, {"colspan": 2}, None]] * (n_inv + 2),
    )

    colors_ = plotly.colors.qualitative.Plotly
    htmp_ = (
        "<b>%{fullData.name}</b><br>Generation: %{y:.2f} MWh<br>"
        "<i>%{x|%Y-%m-%d}</i><extra></extra>"
    )
    htemplate = (
        "<b>%{fullData.name}</b><br>Power: %{y:.2f} kW<br>"
        "<i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
    )
    htemplate_sc = (
        "<b>%{fullData.name}</b><br>POA: %{x:.2f} kW/m2<br>"
        "Power: %{y:.2f} kW<br><i>%{customdata|%Y-%m-%d %H:%M}</i><extra></extra>"
    )

    # ROW 1, COL 1: scatter plot -- actual v. possible v. meter
    if native_freq < pd.Timedelta(minutes=15):
        df_sc = df_site.resample("15min").mean().copy()
    else:
        df_sc = df_site.copy()

    params_1 = dict(
        poss_col=dict(
            name=poss_col,
            opacity=0.3,
            marker=dict(color="#777", size=8, symbol="square", line=dict(color="#333", width=1)),
        ),
        act_col=dict(
            name=act_col,
            opacity=1,
            marker=dict(
                color="#1929f8", size=8, symbol="square-open", line=dict(color="#1929f8", width=1.5)
            ),
        ),
        pi_col=dict(
            name=pi_col,
            opacity=0.9,
            marker=dict(
                color=colors_[2], size=6, symbol="x-thin", line=dict(color="limegreen", width=1.5)
            ),
        ),
        util_col=dict(
            name=util_col,
            marker=dict(
                color=colors_[1], size=6, symbol="circle", line=dict(color="#000", width=0.5)
            ),
            mode="markers",  # needed b/c doesn't use kwargs_1
            hovertemplate=htemplate_sc,  # needed b/c doesn't use kwargs_1
        ),
    )
    kwargs_1 = dict(
        x=df_sc[poacol],
        customdata=df_sc.index,
        mode="markers",
        hovertemplate=htemplate_sc,
        legendgroup="scatter",
    )
    for col, col_kwargs in params_1.items():
        if col == util_col:
            continue
        fig.add_trace(go.Scattergl(y=df_sc[col], **kwargs_1, **col_kwargs), row=1, col=1)

    # add utility meter data to scatter plot if exists (use hourly poa data for trace)
    if util_col in df_site_hourly.columns:
        col_kwargs = params_1[util_col] | dict(
            x=df_site_hourly[poacol], y=df_site_hourly[util_col], customdata=df_site_hourly.index
        )
        fig.add_trace(go.Scattergl(**kwargs_1, **col_kwargs), row=1, col=1)

    # ROW 1, COL 3: kpi table
    fig.add_trace(
        go.Table(
            columnwidth=[2.6, 1],
            header=dict(
                values=list(df_kpis.columns),
                align=["left", "right"],
                height=24,
                font_size=11,
                font_color=["white", "black"],
                line_color="black",
                fill_color=["gray"] + ["silver"] * 3 + ["limegreen"],
            ),
            cells=dict(
                values=[df_kpis[c].tolist() for c in df_kpis.columns],
                align=["left", "right"],
                height=22,
                font_size=10,
                format=["", ".2f"],
                line_color="gray",
                fill_color=["gainsboro"] + ["whitesmoke"] * 3 + ["palegreen"],
            ),
        ),
        row=1,
        col=3,
    )

    # REMAINING ROWS - INVERTER LEVEL GENERATION (scatter + time series)

    # get plotly default colors & use to ensure subplot trace colors match
    pltcolors = plotly.colors.qualitative.Dark24
    c_num = len(pltcolors)
    colorlist = [pltcolors[n % c_num] for n in range(len(invcols))]

    # remaining subplots (scatter / time series)
    common_idx = dfH.index.copy()

    sc_kwargs = dict(
        mode="markers", showlegend=False, customdata=common_idx, hovertemplate=htemplate_sc
    )
    ts_kwargs = dict(x=common_idx, mode="lines")
    pvl_kwargs = dict(
        fill="tozeroy",
        fillcolor="#eee",
        hoverinfo="skip",
        hovertemplate=None,
        # fillpattern=dict(shape='/', size=3, solidity=0.05),
        line=dict(color="#bfbfbf", dash="dot", width=1),
    )
    inv_kwargs = dict(fill="tozeroy", line_width=1, hovertemplate=htemplate)

    # ROW 2, COL 1: scatter plots -- site-level generation
    sc_trace_possible = go.Scatter(
        x=dfH[poacol],
        y=dfH[poss_col],
        name=poss_col,
        marker=dict(color="#777", size=3),
        **sc_kwargs,
    )
    sc_trace_actual = go.Scatter(
        x=dfH[poacol],
        y=dfH[act_col],
        name=act_col,
        marker=dict(color="#1929f8", size=3),
        **sc_kwargs,
    )
    sc_trace_pi_meter = go.Scatter(
        x=dfH[poacol],
        y=dfH[pi_col],
        name=pi_col,
        marker=dict(color=colors_[2], size=2),
        **sc_kwargs,
    )
    # temp: duplicating scatter plots for meter bar + line graphs
    for row in (2, 3):
        fig.add_trace(sc_trace_possible, row=row, col=1)
        fig.add_trace(sc_trace_actual, row=row, col=1)
        fig.add_trace(sc_trace_pi_meter, row=row, col=1)
        if site in dfu.columns:
            sc_trace_util = go.Scatter(
                x=dfH[poacol],
                y=dfu[site],
                name=util_col,
                marker=dict(color=colors_[1], size=2),
                **sc_kwargs,
            )
            fig.add_trace(sc_trace_util, row=row, col=1)

    # ROW 2, COL 2: stacked bar charts -- daily generation
    bar_cols = [poss_col, act_col, pi_col]
    if data_freq == "1min":
        dfbar = df[bar_cols].resample("h").mean().resample("D").sum().copy()
    else:
        dfbar = df[bar_cols].resample("D").sum().copy()
    bar_max = dfbar.max().max() * 1.05  # used for yaxis in layout
    dfbar = dfbar.rename_axis("Date").reset_index(drop=False)

    bar_trace_possible = go.Bar(
        x=dfbar["Date"],
        y=dfbar[poss_col],
        name=poss_col,
        hovertemplate=htmp_,
        marker_color="#777",
        showlegend=False,
    )
    bar_trace_actual = go.Bar(
        x=dfbar["Date"],
        y=dfbar[act_col],
        name=act_col,
        hovertemplate=htmp_,
        marker_color="#1929f8",
        showlegend=False,
    )
    bar_trace_pi_meter = go.Bar(
        x=dfbar["Date"],
        y=dfbar[pi_col],
        name=pi_col,
        hovertemplate=htmp_,
        marker_color=colors_[2],
        showlegend=False,
    )
    fig.add_trace(bar_trace_possible, row=2, col=2)
    fig.add_trace(bar_trace_actual, row=2, col=2)
    fig.add_trace(bar_trace_pi_meter, row=2, col=2)
    if site in dfu.columns:
        dfubar = dfu[[site]].resample("D").sum()
        bar_trace_util = go.Bar(
            x=dfbar["Date"],
            y=dfubar[site],
            name=util_col,
            hovertemplate=htmp_,
            marker_color=colors_[1],
            showlegend=False,
        )
        fig.add_trace(bar_trace_util, row=2, col=2)

    # ROW 3, COL 2: time-series subplot of site-level generation
    ts_trace_possible = go.Scattergl(
        y=dfH[poss_col],
        name=poss_col,
        **ts_kwargs,
        **pvl_kwargs,
        legendgroup="timeseries",
    )
    ts_trace_actual = go.Scattergl(
        y=dfH[act_col],
        name=act_col,
        **ts_kwargs,
        line_color="#1929f8",
        line_width=1.5,
        legendgroup="timeseries",
    )
    ts_trace_pi_meter = go.Scattergl(
        y=dfH[pi_col],
        name=pi_col,
        **ts_kwargs,
        line_color=colors_[2],
        line_width=1.5,
        legendgroup="timeseries",
    )
    for trc in [ts_trace_possible, ts_trace_actual, ts_trace_pi_meter]:
        fig.add_trace(trc, row=3, col=2)
    if site in dfu.columns:
        ts_trace_util = go.Scattergl(
            y=dfu[site],
            name=util_col,
            **ts_kwargs,
            line_color=colors_[1],
            line_width=1.5,
            legendgroup="timeseries",
        )
        fig.add_trace(ts_trace_util, row=3, col=2)

    # ROWS 4 to END
    for i, col in enumerate(invcols):
        scatter_trace = go.Scattergl(
            x=dfH[poacol],
            y=dfH[col],
            name=invnames[i],
            marker_color=colorlist[i],
            marker_size=2,
            **sc_kwargs,
        )
        fig.add_trace(scatter_trace, row=i + 4, col=1)

        pvl_trace = go.Scattergl(
            y=dfH[pvlinvcols[i]],
            name=pvlnames[i],
            **ts_kwargs,
            **pvl_kwargs,
            showlegend=False,
        )
        fig.add_trace(pvl_trace, row=i + 4, col=2)

        inv_trace = go.Scattergl(
            y=dfH[col].fillna(0),
            name=invnames[i],
            line_color=colorlist[i],
            **ts_kwargs,
            **inv_kwargs,
            showlegend=False,
        )
        fig.add_trace(inv_trace, row=i + 4, col=2)

    # FORMATTING
    title_txt = f'<b>{site} - {dfH.index[0].date().strftime("%B %Y")}</b>'
    fig.update_layout(
        height=total_height,
        width=1500,
        font_size=10,
        title=dict(
            text=title_txt,
            font_size=16,
            x=0,
            xref="paper",
            y=1,
            yref="paper",
            yanchor="bottom",
            pad_b=8,
        ),
        hoverlabel_font=dict(size=12),  # showlegend=False,
        margin=dict(t=30, r=20, b=20, l=20),
        paper_bgcolor="#f8f9fa",
        plot_bgcolor="#fff",
        legend=dict(
            x=0.02,
            xref="paper",
            y=0.992,
            yref="paper",
            yanchor="top",
            font_size=11,
            bordercolor="#ced4da",
            borderwidth=1,
            tracegroupgap=10,
            groupclick="toggleitem",
        ),
        barcornerradius=4,
        barmode="group",
        bargap=0.4,  # gap between bars of adjacent location coordinates
        bargroupgap=0.3,  # gap between bars of the same location coordinate
    )
    fig.update_xaxes(linecolor="#ced4da", tickfont_size=9)
    fig.update_xaxes(col=1, minallowed=0)

    xkwargs_ = dict(
        fixedrange=True,
        showticklabels=True,
        tickformat="%m/%d",
        dtick="86400000",
    )

    fig.update_xaxes(row=2, col=2, **xkwargs_)
    fig.update_xaxes(row=3, col=2, **xkwargs_)

    for row_ in range(3, n_rows + 1):
        fig.update_xaxes(
            row=row_,
            col=2,
            ticklabelmode="period",
            tick0=common_idx[0],
            gridcolor="#eee",
            **xkwargs_,
        )

    ymax_ = max([dfH[poss_col].max(), dfH[act_col].max(), dfH[pi_col].max()]) * 1.05

    fig.update_yaxes(linecolor="#ced4da", rangemode="tozero", tickfont_size=9)

    fig.update_yaxes(
        row=2,
        col=2,
        fixedrange=True,
        range=[0, bar_max],
        showgrid=False,
        title=dict(text=f"<b>Meter</b>", standoff=0),
    )
    fig.update_yaxes(
        row=3,
        col=2,
        fixedrange=True,
        range=[0, ymax_],
        showgrid=False,
        title=dict(text=f"<b>Meter</b>", standoff=4),
    )

    for row in (2, 3):
        fig.update_yaxes(row=row, col=1, fixedrange=True, range=[0, ymax_])
        fig.update_xaxes(row=row, col=1, fixedrange=True)

    ymax = max([dfH[pvlinvcols].max().max(), dfH[invcols].max().max()]) * 1.05
    for i, inv in enumerate(invnames):
        rw = i + 4  # start at row 4
        fig.update_yaxes(
            row=rw,
            col=2,
            fixedrange=True,
            range=[0, ymax],
            showgrid=False,
            title=dict(text=f"<b>{inv}</b>", standoff=0),
        )
        fig.update_yaxes(row=rw, col=1, fixedrange=True, range=[0, ymax])
        fig.update_xaxes(row=rw, col=1, fixedrange=True)

    # add vertical day separators to bar chart (as shapes)
    ## adding vertical lines to bar chart (subplot #3)
    vline_kwargs = dict(
        line=dict(width=0.5, color="#ced4da"),
        xref="x3",
        yref="y3 domain",
        y0=0,
        y1=1,
    )
    for tstamp in dfbar["Date"][:-1]:
        x_ = tstamp + pd.Timedelta(hours=12)
        fig.add_shape(type="line", x0=x_, x1=x_, **vline_kwargs)

    if save_html:
        sfolder_ = (
            oepaths.frpath(year, month, ext="solar", site=site) if (savepath is None) else savepath
        )
        sfile_ = f"{site}_{calendar.month_abbr[month]}{year}_summary_plots.html"
        if len(frfpaths) == 0:
            sfile_ = sfile_.split(".html")[0] + "_rev0.html"
        spath_ = Path(sfolder_, sfile_)
        spath_ = oepaths.validated_savepath(spath_)
        fig.write_html(spath_, config=config_)
        disppath = (
            str(spath_) if (savepath is not None) else ".." + str(spath_).split("Operations")[-1]
        )
        print(f'Saved file "{spath_.name}"\n>> path: {disppath}')

    if return_df_and_fpaths:
        return fig, output[1]
    return fig
