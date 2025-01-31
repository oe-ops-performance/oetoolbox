import numpy as np
import pandas as pd
from pathlib import Path
import itertools, calendar

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utilities import oepaths, oemeta
from ..datatools import utilitymeterdata as umd
from .tools import (
    load_monthly_query_files,
    load_meter_historian,
    get_flashreport_summary_table,
    solar_fr_fpath_dict,
    get_flashreport_kpis,
    get_kpis_from_tracker,
    get_solar_budget_values,
)


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


"""
INVERTER VS. PVLIB SUBPLOTS
"""


def inv_pvlib_subplots(site, year=None, month=None, dfinv=None, dfpvl=None, fig_height=None):
    if (year is not None) and (month is not None):
        qfile_types = ["Inverters", "PVLib"]
        df_dict = load_monthly_query_files(
            site, year, month, types_=qfile_types, separate_dfs=True, q=True
        )
        if not all(k in df_dict for k in qfile_types):
            print(
                f"Missing the following file(s): {[k for k in qfile_types if k not in df_dict]}\n>> exiting.."
            )
            return

        dfi_, dfp_ = map(lambda k: df_dict[k], qfile_types)
    elif all(isinstance(z, pd.DataFrame) for z in [dfinv, dfpvl]):
        dfi_, dfp_ = dfinv.copy(), dfpvl.copy()
    else:
        print("missing args\n>> exiting..")
        return

    inv_cols = list(filter(lambda c: "ActivePower" in c, dfi_.columns))
    pvlib_cols = list(filter(lambda c: "Possible_Power" in c, dfp_.columns))
    invnames = [c.replace("OE.ActivePower_", "") for c in inv_cols]
    pvlnames = [f"{i}_PVLib" for i in invnames]

    n_rows = len(inv_cols)
    dfi = dfi_[inv_cols].copy()
    dfp = dfp_[pvlib_cols].copy()

    if pd.infer_freq(dfi.index) != "h":
        dfi = dfi.resample("h").mean()

    if pd.infer_freq(dfp.index) != "h":
        dfp = dfp.resample("h").mean()

    htemplate = (
        "<b>%{fullData.name}</b><br>Power: %{y:.2f} kW<br><i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
    )

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"<b>{inv}</b>" for inv in invnames],
        vertical_spacing=0.2 / n_rows,
    )
    # print(f'\n{pvlib_cols = }\n\n{inv_cols  = }\n')
    common_idx = dfi.index.copy()
    for i, col in enumerate(dfi.columns):
        kwargs_ = dict(x=common_idx, mode="lines", showlegend=False, hovertemplate=htemplate)
        trace1_kwargs = kwargs_ | dict(
            y=dfp[pvlib_cols[i]],
            name=pvlnames[i],
            fill="tozeroy",
            fillcolor="#f7f7f7",
            fillpattern=dict(shape="/", size=3, solidity=0.05),
            line=dict(color="#bfbfbf", dash="dot", width=1),
        )  # pvlib
        y_vals = dfi[col].fillna(0).copy()
        trace2_kwargs = kwargs_ | dict(
            y=y_vals, name=invnames[i], fill="tozeroy", fillpattern=dict(), line_width=1
        )  # actual

        for trace in [trace1_kwargs, trace2_kwargs]:
            fig.add_trace(trace, row=i + 1, col=1)

    fig.update_layout(
        title=dict(
            text=f'<b>{site} - {dfi.index[0].date().strftime("%B %Y")}</b>',
            font=dict(size=14),
            x=0,
            xref="paper",
        ),
        height=60 * n_rows if not fig_height else fig_height,
        margin=dict(t=40, r=100, b=20, l=20),
        paper_bgcolor="#fff",
        plot_bgcolor="#fff",
        clickmode="none",
        hoverlabel_font=dict(size=11),
    )

    fig.update_xaxes(
        tickformat="%m/%d",
        tick0=common_idx[0],
        dtick="86400000",  # one day (milliseconds)
        tickfont=dict(size=10),
        ticklabelmode="period",  # tickangle=-30,
        fixedrange=True,
        linecolor="#aaa",
        gridcolor="#e2e2e2",
    )

    fig.update_yaxes(
        rangemode="tozero",
        nticks=2,
        tick0=0,
        tickfont=dict(size=10),
        showgrid=False,
        fixedrange=True,
        linecolor="#aaa",
    )

    # format subplot titles (are actually annotations in subplot figures)
    for i in range(len(fig.layout.annotations)):
        yref = "y domain" if i == 0 else f"y{i+1} domain"
        fig.layout.annotations[i].update(
            font=dict(size=12),
            x=1.005,
            xref="paper",
            xanchor="left",
            y=0.5,
            yref=yref,
            yanchor="middle",
        )

    return fig


"""
FLASHREPORT SUMMARY PLOTS
"""


# for solar sites w/ flashreport data
def monthly_summary_subplots(site, year, month, save_html=False, savepath=None, df_util=None):
    # error handling for invalid savepath
    if savepath is not None:
        if not Path(savepath).exists():
            print("Error: invalid savepath.\nExiting..")
            return None
        elif not Path(savepath).is_dir():
            print("Error: savepath must be a folder (not a file).\nExiting..")
            return None

    qfile_types = ["Inverters", "PVLib", "Meter"]

    ## get flashreport-related files
    df_dict = load_monthly_query_files(
        site, year, month, types_=qfile_types, separate_dfs=True, q=True
    )
    if not all(k in df_dict for k in qfile_types):
        print(
            f"Missing the following file(s): {[k for k in qfile_types if k not in df_dict]}\n>> exiting.."
        )
        return

    dfi, dfp, dfm = map(lambda k: df_dict[k], qfile_types)

    poacol = "POA" if "POA" in dfp.columns else "POA_DTN"
    invcols = list(filter(lambda c: "ActivePower" in c, dfi.columns))
    pvlinvcols = list(filter(lambda c: "Possible_Power" in c, dfp.columns))
    invnames = [c.replace("OE.ActivePower_", "") for c in invcols]
    pvlnames = [f"{i}_PVLib" for i in invnames]

    # combine inverter, meter, and pvlib data
    df = dfi.join(dfm).join(dfp).copy()

    # rename "Actual_MW" column to "Inv_Total_MW'
    df = df.rename(columns={"Actual_MW": "Inv_Total_MW"})

    # create variables for data column names (for totals)
    poss_col, act_col, pi_col, util_col = [
        "Possible_MW",
        "Inv_Total_MW",
        "PI_Meter_MW",
        "Util_Meter_MW",
    ]

    # utility meter data (if exists)
    if isinstance(df_util, pd.DataFrame):
        dfu = df_util.dropna(axis=1, how="all").copy()
    elif df_util == "skip":
        dfu = pd.DataFrame()
    else:
        dfu = load_meter_historian(year=year, month=month)
        dfu = dfu.dropna(axis=1, how="all")

    # historical kpis
    t0 = pd.Timestamp(year=year, month=month, day=1)
    t1 = t0 - pd.DateOffset(months=1)
    t2 = t0 - pd.DateOffset(months=2)
    t3 = t0 - pd.DateOffset(months=3)
    ymlist = [(t.year, t.month) for t in [t3, t2, t1]]

    dftbl_ = get_kpis_from_tracker([site], ymlist)
    dfk_ = dftbl_.sort_values(by="Combo Date", ascending=True).copy()
    new_idx = dfk_.Year.astype(str) + "-" + dfk_.Month.astype(str).str.zfill(2)
    keepcols_ = [
        "POA Insolation (kWh/m2)",
        "Budgeted POA (kWh/m2)",
        "DC/System Health Loss (MWh)",
        "Downtime Loss (MWh)",
        "Curtailment - Total (MWh)",
        "Possible Generation (MWh)",
        "Inverter Generation (MWh)",
        "Meter Generation (MWh)",
        "Budgeted Production (MWh)",
        "Inverter Uptime Availability (%)",
        "Performance Availability (%)",
    ]
    dfk = dfk_[keepcols_].set_index(new_idx).T.copy()
    dfk = dfk.rename_axis("Reporting Metric / KPI").reset_index(drop=False)

    # load budget values
    dfb_ = get_solar_budget_values(year, month)
    dfb = dfb_.loc[dfb_["Project"].eq(site)].reset_index(drop=True).copy()

    # load kpis from flashreport for current year/month & merge with budget values
    dfkpi = get_flashreport_kpis(site, year, month, q=True)
    dfkpi = dfkpi.merge(dfb, how="left", on="Project").set_index("Project")
    dfkpi = dfkpi.apply(pd.to_numeric).fillna(0)

    val_ = lambda col_: dfkpi.at[site, col_]
    dfkpi["Performance Availability (%)"] = 1 + (
        (
            (
                val_("Meter Generation (MWh)")
                + val_("Snow Derate Loss (MWh)")
                + val_("Insurance BI Adjustment (MWh)")
            )
            + (val_("Curtailment - Total (MWh)") - val_("Budgeted Curtailment (MWh)"))
            - (val_("POA Insolation (kWh/m2)") / val_("Budgeted POA (kWh/m2)"))
            * val_("Budgeted Production (MWh)")
        )
        / val_("Budgeted Production (MWh)")
    )

    # add current kpis from report (b/c probably haven't been transferred to tracker yet)
    newcol = f"{year}-{month:02d}"
    dfk[newcol] = 0.0  # init
    for kcol in dfk.iloc[:, 0].values:
        dfk.loc[dfk["Reporting Metric / KPI"].eq(kcol), newcol] = val_(kcol)

    """PLOT"""
    # define subplot figure
    n_inv = len(invcols)
    n_rows = n_inv + 2  # changed; +added new subplot of meter data (before inverters)
    plot1_height = 380  # 400
    plot2_height = 100
    plotX_height = 80
    total_height = plot1_height + plot2_height + plotX_height * (n_inv)

    p1_ratio = plot1_height / total_height
    p2_ratio = plot2_height / total_height
    pX_ratio = plotX_height / total_height
    row_height_list = [p1_ratio] + [p2_ratio] + [pX_ratio] * (n_inv)

    fig = make_subplots(
        rows=n_rows,
        row_heights=row_height_list,
        cols=3,
        column_widths=[0.1, 0.5, 0.4],
        horizontal_spacing=0.035,  # default 0.2/n_cols
        vertical_spacing=0.2 / n_rows,  # default 0.3/n_rows
        specs=[[{"colspan": 2}, None, {"type": "table"}]]
        + [[{}, {"colspan": 2}, None]] * (n_inv + 1),
    )

    colors_ = plotly.colors.qualitative.Plotly
    htmp_ = (
        "<b>%{fullData.name}</b><br>Generation: %{y:.2f} MWh<br><i>%{x|%Y-%m-%d}</i><extra></extra>"
    )
    htemplate = (
        "<b>%{fullData.name}</b><br>Power: %{y:.2f} kW<br><i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
    )
    htemplate_sc = "<b>%{fullData.name}</b><br>POA: %{x:.2f} kW/m2<br>Power: %{y:.2f} kW<br><i>%{customdata|%Y-%m-%d %H:%M}</i><extra></extra>"

    # ROW 1: scatter plot -- actual v. possible v. meter
    df_ = df.resample("15min").mean().copy()
    p1_cols = [poss_col, act_col, pi_col]
    p1_opacity = [0.3, 1, 0.9]
    p1_markers = [
        dict(color="#777", size=8, symbol="square", line=dict(color="#333", width=1)),
        dict(color="#1929f8", size=8, symbol="square-open", line=dict(color="#1929f8", width=1.5)),
        dict(color=colors_[2], size=6, symbol="x-thin", line=dict(color="limegreen", width=1.5)),
    ]
    p1_kwargs = dict(
        x=df_[poacol], customdata=df_.index, mode="markers", hovertemplate=htemplate_sc
    )
    for i, col in enumerate(p1_cols):
        kwargs = p1_kwargs | dict(
            y=df_[col],
            name=col,
            opacity=p1_opacity[i],
            marker=p1_markers[i],
            legendgroup="scatter",
        )
        fig.add_trace(go.Scattergl(**kwargs), row=1, col=1)

    # ROW 1: kpi table
    fig.add_trace(
        go.Table(
            columnwidth=[2.6, 1],
            header=dict(
                values=list(dfk.columns),
                align=["left", "right"],
                height=24,
                font_size=11,
                font_color=["white", "black"],
                line_color="black",
                fill_color=["gray"] + ["silver"] * 3 + ["limegreen"],
            ),
            cells=dict(
                values=[dfk[c].tolist() for c in dfk.columns],
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

    # resample to hourly for remaining subplots
    dfH = df.resample("h").mean().copy()

    # add utility meter data if exists (use hourly poa data for trace)
    if site in dfu.columns:
        um_kwargs = dict(
            x=dfH[poacol],
            y=dfu[site].values,
            name=util_col,
            mode="markers",
            customdata=dfH.index,
            marker=dict(
                color=colors_[1], size=6, symbol="circle", line=dict(color="#000", width=0.5)
            ),
            hovertemplate=htemplate_sc,
            legendgroup="scatter",
        )
        fig.add_trace(go.Scattergl(**um_kwargs), row=1, col=1)

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
    fig.add_trace(sc_trace_possible, row=2, col=1)
    fig.add_trace(sc_trace_actual, row=2, col=1)
    fig.add_trace(sc_trace_pi_meter, row=2, col=1)
    if site in dfu.columns:
        sc_trace_util = go.Scatter(
            x=dfH[poacol],
            y=dfu[site],
            name=util_col,
            marker=dict(color=colors_[1], size=2),
            **sc_kwargs,
        )
        fig.add_trace(sc_trace_util, row=2, col=1)

    # ROW 2, COL 2: stacked bar charts -- daily generation
    bar_cols = [poss_col, act_col, pi_col]
    dfbar = df[bar_cols].resample("h").mean().resample("D").sum().copy()
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

    # ROW 2: add time series traces
    # fig.add_trace(
    #     go.Scatter(
    #         y=dfH[poss_col],
    #         name=poss_col,
    #         **ts_kwargs,
    #         **pvl_kwargs,
    #         legendgroup='timeseries',
    #     ),
    #     row=2,
    #     col=2,
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         y=dfH[act_col],
    #         name=act_col,
    #         line_color=colors_[0],
    #         opacity=0.2,
    #         **ts_kwargs,
    #         **inv_kwargs,
    #         legendgroup='timeseries',
    #     ),
    #     row=2,
    #     col=2,
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         y=dfH[pi_col].fillna(0),
    #         name=pi_col,
    #         line_color=colors_[2],
    #         fillpattern=dict(shape='\\', size=3, solidity=0.4),
    #         opacity=0.4,
    #         **ts_kwargs,
    #         **inv_kwargs,
    #         legendgroup='timeseries',
    #     ),
    #     row=2,
    #     col=2,
    # )
    # if site in dfu.columns:
    #     fig.add_trace(
    #         go.Scatter(
    #             y=dfu[site].fillna(0),
    #             name=util_col,
    #             line_color=colors_[1],
    #             fillpattern=dict(shape='/', size=3, solidity=0.4),
    #             **ts_kwargs,
    #             **inv_kwargs,
    #             legendgroup='timeseries',
    #         ),
    #         row=2,
    #         col=2,
    #     )

    # ROWS 3 to END
    for i, col in enumerate(invcols):
        scatter_trace = go.Scattergl(
            x=dfH[poacol],
            y=dfH[col],
            name=invnames[i],
            marker_color=colorlist[i],
            marker_size=2,
            **sc_kwargs,
        )
        fig.add_trace(scatter_trace, row=i + 3, col=1)

        pvl_trace = go.Scattergl(
            y=dfH[pvlinvcols[i]],
            name=pvlnames[i],
            **ts_kwargs,
            **pvl_kwargs,
            showlegend=False,
        )
        fig.add_trace(pvl_trace, row=i + 3, col=2)

        inv_trace = go.Scattergl(
            y=dfH[col].fillna(0),
            name=invnames[i],
            line_color=colorlist[i],
            **ts_kwargs,
            **inv_kwargs,
            showlegend=False,
        )
        fig.add_trace(inv_trace, row=i + 3, col=2)

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

    fig.update_yaxes(row=2, col=1, fixedrange=True, range=[0, ymax_])
    fig.update_xaxes(row=2, col=1, fixedrange=True)

    ymax = max([dfH[pvlinvcols].max().max(), dfH[invcols].max().max()]) * 1.05
    for i, inv in enumerate(invnames):
        rw = i + 3  # start at row 3
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
        spath_ = Path(sfolder_, sfile_)
        spath_ = oepaths.validated_savepath(spath_)
        fig.write_html(spath_, config=config_)
        disppath = (
            str(spath_) if (savepath is not None) else ".." + str(spath_).split("Operations")[-1]
        )
        print(f'Saved file "{spath_.name}"\n>> path: {disppath}')

    return fig


def fr_historical_summary_plot(site, yearmonthlist, savehtml=False, q=True):
    qprint = lambda msg: None if q else print(msg)

    ## load existing kpis from tracker
    dfs_ = get_kpis_from_tracker([site], yearmonthlist)
    dfs_ = dfs_.sort_values(by="Combo Date").reset_index(drop=True)
    matching_cols = {
        "Combo Date": "Date",
        "POA Insolation (kWh/m2)": "Insolation",
        "Possible Generation (MWh)": "Possible",
        "Inverter Generation (MWh)": "Inverter_Sum",
        "Meter Generation (MWh)": "Util_Meter",
    }
    dfs = dfs_[[*matching_cols]].rename(columns=matching_cols).copy()

    ## fill in any missing values using flashreport data (if exists)
    has_missing_values = dfs.eq(0).any(axis=1)
    dates_with_missing_vals = [pd.Timestamp(d) for d in dfs.loc[has_missing_values, "Date"].values]

    ## get missing values from flashreports
    if dates_with_missing_vals:
        qprint("getting missing values from flashreports..")
    for date_ in dates_with_missing_vals:
        year, month = date_.year, date_.month
        dff = get_flashreport_kpis(site, year, month, q=False)
        if dff is None:
            qprint("")
            continue

        ## find associated columns with missing values
        dfsT = dfs[dfs.Date.eq(date_)].T.copy()
        missing_cols = dfsT.loc[dfsT.iloc[:, 0].eq(0)].index.values

        if "Possible" in missing_cols:
            qprint("\n!! found report with kpis that have not been transferred to tracker !!\n")

        ## get values from matching column(s) & update "dfs"
        for col in missing_cols:
            eq_col = {y: x for x, y in matching_cols.items()}.get(col)
            value_ = dff.at[0, eq_col]
            dfs.loc[dfs.Date.eq(date_), col] = value_

        qprint("")
    qprint("done!")

    ## get pi meter data
    pi_meter_vals = []
    for y, m in yearmonthlist:
        dfpi = load_monthly_query_files(site, y, m, types_=["Meter"], separate_dfs=False, q=True)
        mcol = dfpi.columns[0]
        pi_total = dfpi.resample("h").mean()[mcol].sum()
        pi_meter_vals.append(pi_total)

    dfs["PI_Meter"] = pd.Series(pi_meter_vals)

    ## create plots
    x_vals = list(range(len(yearmonthlist)))
    months_ = [int(f"{y}{m:02d}") for y, m in yearmonthlist]
    month_labels = [f"{calendar.month_abbr[m]} '{str(y)[-2:]}" for y, m in yearmonthlist]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=dfs["Possible"],
            name="Possible",
            offset=-0.4,
            opacity=0.15,
            marker_color="#000",
        )
    )
    fig.add_trace(go.Bar(x=x_vals, y=dfs["Inverter_Sum"], name="Inverter Sum"))
    fig.add_trace(go.Bar(x=x_vals, y=dfs["PI_Meter"], name="PI Meter"))
    fig.add_trace(go.Bar(x=x_vals, y=dfs["Util_Meter"], name="Util Meter"))
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=dfs["Insolation"].mask(dfs["Insolation"].eq(0), np.nan),
            name="Insolation",
            yaxis="y2",
            marker_color="#000",
            line_color="#000",
        ),
    )
    fig.update_layout(
        barmode="group",
        height=400,
        width=110 * len(months_) if len(months_) >= 8 else 800,
        title=dict(text=f"<b>{site}</b>", xref="paper", x=0, y=0.95),
        legend=dict(orientation="h", xanchor="right", x=1, y=1.1),
        margin=dict(t=20, r=20, b=20, l=20),
        xaxis=dict(tickvals=x_vals, ticktext=month_labels, tickfont_size=12),
        yaxis=dict(
            title=dict(text="Monthly Generation (MWh)", font_size=11, standoff=5), tickfont_size=10
        ),
        yaxis2=dict(
            side="right",
            overlaying="y",
            rangemode="tozero",
            showgrid=False,
            title=dict(text="Insolation (kWh/m2)", font_size=11),
            tickfont_size=10,
        ),
    )

    if savehtml:
        downloads_ = Path.home.joinpath("Downloads")
        fname_ = f"frHistGen_{site}_{months_[0]}to{months_[-1]}.html"
        spath_ = Path(downloads_, fname_)
        fig.write_html(oepaths.validated_savepath(spath_))

    return fig


def poa_comparison_fig(dfpoa):

    htemplate = (
        "<b>%{fullData.name}</b><br>Value: %{y:.2f}<br><i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
    )

    fig = go.Figure()

    poacols = list(dfpoa.columns)[:-4]


"""
METER DATA COMPARISON DATA / FIGURES
"""


## get df for meter comparison plot "meter_time_series_compare" (below)
def get_site_combined_meter_df(site, year, month, df_util=None):
    # if 'Inverters' not in oemeta.data['AF_Solar_V3'].get(site):
    #     return pd.DataFrame()
    dfm = load_monthly_query_files(site, year, month, types_=["Meter", "Inverters"], q=False)
    keepcols_ = {"PI_Meter_MW": "PI_Meter", "Actual_MW": "Inverter_Sum"}
    df = dfm[[c for c in dfm.columns if c in keepcols_]].copy()
    df = df.rename(columns={c: keepcols_[c] for c in df.columns})
    df = df.resample("h").mean()
    if isinstance(df_util, str):
        if df_util == "skip":
            return df
    elif df_util is None:
        dfum = load_meter_historian(year, month)
    else:
        dfum = df_util.copy()
    dfu = dfum.dropna(axis=1, how="all").copy()
    if site in dfu.columns:
        df["Utility"] = dfu[site].copy()
    return df


## for solar flashreports > pi meter vs. utility meter vs. inv sum
def meter_time_series_compare(df, title_=None, height_=575):
    ## df with datetime index & columns for comparison (possible columns: ['Utility_Meter', 'PI_Meter', 'Inverter_Sum'])
    c1, err1 = (not df.empty), "df cannot be empty"
    c2, err2 = (type(df.index) == pd.DatetimeIndex), "df must have datetime index"
    c3, err3 = (
        any(c in df.columns for c in ["Utility_Meter", "PI_Meter", "Inverter_Sum"]),
        "no valid columns in df",
    )
    for req, msg in zip([c1, c2, c3], [err1, err2, err3]):
        if not req:
            print(f"error: {msg}.\nexiting..")
            return

    ## define figure object
    fig = go.Figure()

    htemplate = (
        "<b>%{fullData.name}</b><br>Power: %{y:.2f} MW<br><i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
    )
    colors_ = plotly.colors.qualitative.G10
    blu_, red_, grn_ = colors_[0], colors_[1], colors_[3]

    common_kwargs = dict(x=df.index, mode="lines", fill="tozeroy", hovertemplate=htemplate)

    inv_kwargs = common_kwargs | dict(
        name="Inverter_Sum",
        line=dict(color=blu_, width=2),
        fillpattern=dict(bgcolor="rgba(0,0,0,0)", fgcolor=blu_, shape="|", size=3, solidity=0.3),
    )
    pi_kwargs = common_kwargs | dict(
        name="PI_Meter",
        line=dict(color=grn_, width=2),
        fillpattern=dict(bgcolor="rgba(0,0,0,0)", fgcolor=grn_, shape="-", size=4, solidity=0.2),
    )
    util_kwargs = common_kwargs | dict(
        name="Utility",
        line=dict(color=red_, width=1.5),
        fillpattern=dict(bgcolor="rgba(0,0,0,0)", fgcolor=red_, shape="/", size=3, solidity=0.2),
    )

    includes_ = lambda c: (c in df.columns)

    df[df.lt(0)] = 0  # remove negative values

    if includes_("Inverter_Sum"):
        fig.add_trace(go.Scatter(**inv_kwargs, y=df["Inverter_Sum"].fillna(0)))

    if includes_("PI_Meter"):
        fig.add_trace(go.Scatter(**pi_kwargs, y=df["PI_Meter"].fillna(0)))

    if includes_("Utility"):
        fig.add_trace(go.Scatter(**util_kwargs, y=df["Utility"].fillna(0)))

    title_txt = title_ if title_ else "Monthly Generation Data Comparison"
    fig.update_layout(
        height=height_,
        margin=dict(t=50, r=20, b=20, l=20),
        title=dict(
            text=f"<b>{title_txt}</b>",
            xref="paper",
            x=0,
            yref="paper",
            yanchor="bottom",
            y=1,
            pad_b=10,
        ),
        legend=dict(orientation="h", xanchor="right", x=1, yanchor="bottom", y=1),
    )
    fig.update_xaxes(
        showticklabels=True,
        tickformat="%m/%d",
        tick0=df.index[0],
        dtick="86400000",
        ticklabelmode="period",
        gridcolor="#eee",
        range=[df.index.min(), df.index.max()],
    )
    fig.update_yaxes(
        linecolor="#ced4da",
        rangemode="tozero",
        tickfont_size=9,
        showgrid=False,
        title=dict(text=f"<b>Generation (MWh)</b>", standoff=0),
    )
    return fig


def site_inv_meter_compare_plot(site, year, month, height_=575):
    df = get_site_combined_meter_df(site, year, month, df_util="skip")
    title_ = f"{site.upper()} {calendar.month_abbr[month]}-{year} meter vs. inverters"
    fig = meter_time_series_compare(df, title_=title_, height_=height_)
    return fig


def flashreport_meter_comparison_plot(
    site, year, month, df_util=None, save_file=False, q=True, height_=350
):
    df = get_site_combined_meter_df(site, year, month, df_util=df_util)
    title_ = f"{site}  |  Meter Data ({calendar.month_abbr[month]} {year})"
    fig = meter_time_series_compare(df, title_=title_, height_=height_)
    if not fig:
        print("error\nexiting..")
        return

    if save_file:
        path_ = Path(oepaths.frpath(year, month, ext="Solar"), site)
        fname_ = f"meterComparisonPlot_{site}_{month:02d}-{year}.html"
        savepath_ = oepaths.validated_savepath(path_.joinpath(fname_))
        fig.write_html(savepath_, config=config_)
        if not q:
            print(f'saved file: "{savepath_.name}"\nfull path: {str(savepath_)}')

    return fig


## meter data comparison subplots (developed for stlmtui sites)
def meter_comparison_subplots(
    year, month, sitelist, original_files=True, df_util=None, save_file=False, q=True, height_=None
):

    ## check requirements for sites in list
    frfpath_dict = solar_fr_fpath_dict(year, month)
    fr_sites = [s for s in sitelist if s in frfpath_dict]
    c1 = lambda site: any("PIQuery_Inverters" in fp.name for fp in frfpath_dict[site])
    c2 = lambda site: any("PIQuery_Meter" in fp.name for fp in frfpath_dict[site])
    isvalid_ = lambda site: (c1(site) and c2(site))
    valid_sites = [s for s in fr_sites if isvalid_(s)]
    if not valid_sites:
        print("None of the specified sites have the required files for plotting.\nExiting..")
        return

    ## utility meter data
    if not original_files:
        if df_util is None:
            dfum = load_meter_historian(year, month)
        else:
            dfum = df_util.copy()
        dfu = dfum.dropna(axis=1, how="all").copy()
    else:
        dfu = umd.load_meter_data(year, month, sitelist=valid_sites, q=q)

    queryfile_dict = {}
    for site in valid_sites:
        dfm = load_monthly_query_files(site, year, month, types_=["Meter", "Inverters"], q=q)
        if not q:
            print("")
        keepcols_ = {"PI_Meter_MW": "PI_Meter", "Actual_MW": "Inverter_Sum"}
        df_ = dfm[[c for c in dfm.columns if c in keepcols_]].copy()
        df_ = df_.rename(columns={c: keepcols_[c] for c in df_.columns})
        df_ = df_.resample("h").mean()
        if site in dfu.columns:
            df_["Utility"] = dfu[site].copy()

        queryfile_dict[site] = df_.copy()

    ## plot
    title_ = f"<b>Meter Data Comparison - {calendar.month_name[month]} {year}</b>"

    n_rows = len(valid_sites)

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        vertical_spacing=0.25 / n_rows,  # default 0.3/n_rows
        subplot_titles=valid_sites,
    )

    htemplate = (
        "<b>%{fullData.name}</b><br>Power: %{y:.2f} MW<br><i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
    )
    colors_ = plotly.colors.qualitative.G10
    blu_, red_, grn_ = colors_[0], colors_[1], colors_[3]
    inv_col, pi_col, util_col = ["Inverter_Sum", "PI_Meter", "Utility"]

    for i, item_ in enumerate(queryfile_dict.items()):
        site, df = item_
        df[df.lt(0)] = 0  # remove negative values
        kwargs_ = dict(
            x=df.index, mode="lines", fill="tozeroy", hovertemplate=htemplate, showlegend=False
        )

        inv_kwargs = kwargs_ | dict(
            line=dict(color=blu_, width=2),
            fillpattern=dict(
                bgcolor="rgba(0,0,0,0)", fgcolor=blu_, shape="|", size=3, solidity=0.3
            ),
        )
        inv_trace = go.Scatter(**inv_kwargs, y=df[inv_col].fillna(0), name=f"Inverters_{site}")
        fig.add_trace(inv_trace, row=i + 1, col=1)

        pi_kwargs = kwargs_ | dict(
            line=dict(color=grn_, width=2),
            fillpattern=dict(
                bgcolor="rgba(0,0,0,0)", fgcolor=grn_, shape="-", size=4, solidity=0.2
            ),
        )
        pi_trace = go.Scatter(**pi_kwargs, y=df[pi_col].fillna(0), name=f"PI_Meter_{site}")
        fig.add_trace(pi_trace, row=i + 1, col=1)

        util_kwargs = kwargs_ | dict(
            line=dict(color=red_, width=1.5),
            fillpattern=dict(
                bgcolor="rgba(0,0,0,0)", fgcolor=red_, shape="/", size=3, solidity=0.2
            ),
        )
        if util_col in df.columns:
            util_trace = go.Scatter(**util_kwargs, y=df[util_col].fillna(0), name=f"Utility_{site}")
            fig.add_trace(util_trace, row=i + 1, col=1)

    fig.update_layout(
        height=120 * n_rows,
        width=1000,
        font_size=9,
        title=dict(
            text=title_,
            font_size=16,
            x=0.5,
            xref="paper",
            xanchor="center",
            y=1,
            yref="paper",
            yanchor="bottom",
            pad_b=8,
        ),
        hoverlabel_font=dict(size=11),
        margin=dict(t=30, r=20, b=20, l=135),
    )
    fig.update_xaxes(
        showticklabels=True,
        tickformat="%m/%d",
        dtick="86400000",
        ticklabelmode="period",
    )
    fig.update_yaxes(linecolor="#ced4da", rangemode="tozero", tickfont_size=8, showgrid=False)

    ## set alignment for subplot titles (annotations in subplot figure)
    for i, annot_ in enumerate(fig.layout.annotations):
        yaxis_id = "y" if i == 0 else f"y{i+1}"
        fig.layout.annotations[i].update(
            font_size=12,
            x=0,
            xref="paper",
            xanchor="right",
            xshift=-20,
            y=0.5,
            yref=f"{yaxis_id} domain",
            yanchor="middle",
        )

    return fig


"""
MISC FLASHREPORT SUMMARY PLOTS
"""


def flashreport_inv_totals_subplots(site, year, month, q=True):
    df_ = get_flashreport_summary_table(site, year, month, inverter_level=True, q=q)
    df = df_.transpose().rename_axis("Inverter").reset_index(drop=False).copy()
    plot_cols = list(df.columns)[1:]
    fig = make_subplots(
        rows=len(plot_cols), cols=1, subplot_titles=[f"<b>{c}</b>" for c in plot_cols]
    )

    for i, col in enumerate(plot_cols):
        bar_trace = go.Bar(x=df.Inverter, y=df[col], name=col, showlegend=False)
        fig.add_trace(bar_trace, row=i + 1, col=1)

    ## set alignment for subplot titles (annotations in subplot figure)
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].update(font_size=12, x=0, xref="paper", xanchor="left")

    fig.update_layout(
        barcornerradius=10,
        height=600,
        font_size=10,
        margin=dict(t=40, b=30, l=20, r=20),
        title=dict(
            font_size=14,
            text=f"{calendar.month_name[month]} {year} - <b>{site}</b> - Inverter Totals",
            x=0.5,
            xref="paper",
            xanchor="center",
        ),
        clickmode="none",
    )
    fig.update_yaxes(fixedrange=True)
    fig.update_xaxes(fixedrange=True)

    return fig


def flashreport_inv_daily_gen_subplots(site, year, month, q=True):
    df_dict = load_monthly_query_files(
        site, year, month, types_=["Inverters", "PVLib"], separate_dfs=True, q=q
    )
    if not all(k in df_dict for k in ["Inverters", "PVLib"]):
        print("one or more files missing.\nexiting..")
        return

    df_inv = df_dict.get("Inverters")
    df_inv.columns = df_inv.columns.map(lambda c: c.replace("OE.ActivePower_", ""))
    dfi_ = df_inv.resample("h").mean().resample("D").sum().copy()
    dfi = dfi_.drop(columns=["Actual_MW"]).copy()
    dfi = dfi.rename_axis("Date").reset_index(drop=False)

    df_pvl = df_dict.get("PVLib")
    pvlcols = [c for c in df_pvl if "Possible_Power" in c]
    dfp_ = df_pvl[pvlcols].resample("h").mean().resample("D").sum().copy()
    dfp = dfp_.rename_axis("Date").reset_index(drop=False).copy()

    x_col = "Date"  # for x axis
    dfp.columns = dfp.columns.map(lambda c: c if c == x_col else c.casefold())

    plot_cols = [c for c in dfi.columns if c != x_col]
    n_rows = len(plot_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=[f"<b>{c}</b>" for c in plot_cols],
        vertical_spacing=0.3 / n_rows,  # default 0.3/n_rows
    )
    for i, col in enumerate(plot_cols):
        pvl_col = f"{col.casefold()}_possible_power"
        pvl_trace = go.Bar(
            x=dfp[x_col],
            y=dfp[pvl_col],
            name=col + "_Possible",
            marker_color="#787878",
            marker_pattern=dict(shape="x", size=3, bgcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        fig.add_trace(pvl_trace, row=i + 1, col=1)

        inv_trace = go.Bar(x=dfi[x_col], y=dfi[col], name=col, showlegend=False)
        fig.add_trace(inv_trace, row=i + 1, col=1)

    fig.update_layout(
        barcornerradius=4,
        barmode="group",
        bargap=0.25,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1,  # gap between bars of the same location coordinate
        clickmode="none",
        font_size=10,
        height=100 * n_rows,
        margin=dict(t=40, b=30, l=20, r=20),
        title=dict(
            font_size=14,
            text=f"{calendar.month_name[month]} {year} - <b>{site}</b> - Daily Generation (kWh) by Inverter",
            x=0.5,
            xref="paper",
            xanchor="center",
        ),
    )
    fig.update_yaxes(fixedrange=True)
    fig.update_xaxes(tickformat="%b-%d", fixedrange=True)

    ## set alignment for subplot titles (annotations in subplot figure)
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].update(font_size=12, x=0, xref="paper", xanchor="left")

    return fig


def historical_kpi_summary_plot(site, yearmonth_list):
    # load kpi tracker data
    dfs_ = get_kpis_from_tracker([site], yearmonth_list)
    dfs_ = dfs_.sort_values(by=["Month"]).reset_index(drop=True)

    matching_cols = {
        "Combo Date": "Date",
        "POA Insolation (kWh/m2)": "Insolation",
        "Budgeted POA (kWh/m2)": "Budgeted_Insolation",
        "Possible Generation (MWh)": "Possible",
        "Inverter Generation (MWh)": "Actual",
        "Meter Generation (MWh)": "Meter",
        "Curtailment - Total (MWh)": "Curtailment",
        "Performance Availability (%)": "Performance_Availability",
    }
    dfs = dfs_[[*matching_cols]].rename(columns=matching_cols).copy()

    ym_list = list(sorted(yearmonth_list))  # sort dates ascending
    x_vals = list(range(len(ym_list)))
    months_ = [int(f"{y}{m:02d}") for y, m in ym_list]
    lbl_ = lambda y, m: f"{calendar.month_abbr[m]} '{str(y)[-2:]}"
    month_labels = [f"<i>{lbl_(y, m)}</i>" for y, m in ym_list[:-1]] + [
        f"<b>{lbl_(*ym_list[-1])}</b>"
    ]

    colors_ = plotly.colors.qualitative.Plotly

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=dfs["Possible"],
            legend="legend2",
            marker_color="#000",
            name="Possible",
            offset=-0.4,
            opacity=0.2,
        ),
    )

    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=dfs["Actual"],
            legend="legend2",
            marker_color=colors_[2],
            marker_pattern=dict(shape="+", size=4),
            name="Actual",
            opacity=0.6,
        ),
    )

    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=dfs["Curtailment"],
            base=dfs["Possible"].sub(dfs["Curtailment"]),
            legend="legend2",
            marker_pattern=dict(bgcolor="#E5ECF5", fgcolor="red", shape="x", size=5),
            name="Curtailment",
            offset=-0.4,
            opacity=0.8,
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=dfs["Budgeted_Insolation"].mask(dfs["Budgeted_Insolation"].eq(0), np.nan),
            line=dict(color="#3d3d3d", dash="dot", width=1),
            marker=dict(color="#3d3d3d", symbol="circle-open"),
            name="Budgeted_Insolation",
            yaxis="y2",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=dfs["Insolation"].mask(dfs["Insolation"].eq(0), np.nan),
            line_color="orange",
            marker_color="orange",
            name="Insolation",
            yaxis="y2",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=dfs["Performance_Availability"].mask(dfs["Performance_Availability"].eq(0), np.nan),
            line_color="black",
            marker=dict(color="black", size=8, symbol="diamond"),
            name="Performance_Availability",
            yaxis="y3",
        ),
    )

    fig.update_layout(
        barmode="group",
        height=375,
        width=850,
        title=dict(text=f"<b>{site}</b>", x=0, xref="paper", y=0.9),
        legend=dict(font_size=10, orientation="h", x=1, xanchor="right", y=1, yanchor="bottom"),
        legend2=dict(font_size=10, orientation="h", x=1, xanchor="right", y=1.08, yanchor="bottom"),
        margin=dict(t=60, r=20, b=20, l=20),
        xaxis=dict(tickfont_size=12, ticktext=month_labels, tickvals=x_vals),
        yaxis=dict(
            tickfont_size=10,
            title=dict(font_size=11, standoff=5, text="Monthly Generation (MWh)"),
        ),
        yaxis2=dict(
            overlaying="y",
            rangemode="tozero",
            showgrid=False,
            side="right",
            tickfont_size=10,
            title=dict(font_size=11, standoff=10, text="Insolation (kWh/m2)"),
        ),
        yaxis3=dict(
            anchor="free",
            autoshift=True,
            overlaying="y",
            rangemode="tozero",
            showgrid=False,
            side="right",
            tickfont_size=10,
            title=dict(font_size=11, standoff=10, text="Perf. Availability (%)"),
        ),
    )

    return fig
