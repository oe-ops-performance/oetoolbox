import numpy as np
import pandas as pd
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


def boxplot(df, keys_=None, resample_=True):
    col_list = (
        list(filter(lambda c: any(k in c for k in keys_), df.columns))
        if keys_
        else list(df.columns)
    )
    if resample_ and (pd.infer_freq(df.index) == "1min"):
        df = df.copy().resample("15min").mean()
    fig = make_subplots(rows=len(col_list), cols=1)
    for i, col in enumerate(col_list):
        fig.add_trace(go.Box(x=df[col], name=col, boxpoints="outliers"), row=i + 1, col=1)
    fig.update_layout(height=50 * len(col_list), margin=dict(t=20, r=60, l=60, b=20))
    return fig


def scatterplot(df, x_col, y_cols, group_col=None, height=400):
    tmplt_ = "<b>%{fullData.name}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>%{customdata}<extra></extra>"
    fig = go.Figure()

    colors_ = plotly.colors.qualitative.Plotly
    tracecolors_ = (colors_[:3] + [colors_[4], colors_[3]]) * 3
    symbols_ = ["circle", "square", "triangle-up", "cross", "diamond"] * 3

    if group_col is not None:
        ## group by unique values of group col
        groups_ = list(df[group_col].unique())
        for i, grp in enumerate(groups_):
            df_ = df.loc[df[group_col].eq(grp)].copy()
            for col in y_cols:
                fig.add_trace(
                    go.Scatter(
                        x=df_[x_col],
                        y=df_[col],
                        customdata=df_[group_col],
                        legendgroup=i,
                        legendgrouptitle_text=grp,
                        name=col,
                        mode="markers",
                        marker_color=tracecolors_[i],
                        marker_symbol=symbols_[i],
                        hovertemplate=tmplt_,
                    ),
                )
    else:
        for col in y_cols:
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[col],
                    customdata=df.index,
                    name=col,
                    mode="markers",
                    hovertemplate=tmplt_,
                ),
            )

    fig.update_layout(height=height, margin=dict(t=20, r=20, l=20, b=20))
    fig.update_xaxes(title_text=x_col)
    return fig


def tsplot(df, keys_=None, resample_=False, height=400):
    col_list = (
        list(filter(lambda c: any(k in c for k in keys_), df.columns))
        if keys_
        else list(df.columns)
    )
    if resample_ and (pd.infer_freq(df.index) == "1min"):
        df = df.copy().resample("15min").mean()
    fig = go.Figure()
    for col in col_list:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, mode="lines"))
    fig.update_layout(height=height, margin=dict(t=20, r=20, l=20, b=20))
    return fig


def histplots(site, df, keys_=None, onlydaylight=False):
    df = df.select_dtypes(include="number").copy()
    col_list = (
        list(filter(lambda c: any(k in c for k in keys_), df.columns))
        if keys_
        else list(df.columns)
    )
    # if onlydaylight:
    #     df = filter_daylight(site, df)  # TODO resolve this
    fig = make_subplots(rows=len(col_list), cols=1)
    for i, col in enumerate(col_list):
        fig.add_trace(go.Histogram(x=df[col], name=col), row=i + 1, col=1)
    fig.update_layout(height=60 * len(col_list), margin=dict(t=20, r=60, l=60, b=20))
    return fig


def ts_subplots(df, row_col_list, row_height_list=[], fig_height=None, resample_=False):
    if resample_:
        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq is not None:
            freq_ = (
                inferred_freq if any(x.isdigit() for x in inferred_freq) else f"1{inferred_freq}"
            )
            if pd.Timedelta(freq_) <= pd.Timedelta(minutes=15):
                df = df.resample("15min").mean()

    n_rows = len(row_col_list)
    if n_rows == 2:
        rheights_ = [0.7, 0.3]
    else:
        rheights_ = [1 / n_rows for i in range(n_rows)]  # default

    if len(row_height_list) == n_rows:
        if sum(row_height_list) <= 1:
            rheights_ = row_height_list

    if fig_height is None:
        fig_height = 125 * n_rows if (n_rows <= 4) else 75 * n_rows

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        row_heights=rheights_,
        vertical_spacing=0.1 / n_rows,  # default 0.3/n_rows
    )

    hvtemp = (
        "<b>%{fullData.name}</b><br>Value: %{y:.2f}<br><i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
    )
    for i, columns_ in enumerate(row_col_list):
        for col in columns_:
            fig.add_trace(
                go.Scattergl(
                    x=df.index,
                    y=df[col],
                    name=col,
                    mode="lines",
                    legendgroup=i,
                    hovertemplate=hvtemp,
                ),
                row=i + 1,
                col=1,
            )

    fig.update_layout(
        font_size=9,
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(
            x=1.01,
            xanchor="left",
            y=0.5,
            yanchor="middle",
            groupclick="toggleitem",
            tracegroupgap=22,
            # tracegroupgap=0.075*fig_height,
        ),
        # hovermode='x unified',
    )
    if fig_height != "skip":
        fig.update_layout(height=fig_height)

    return fig


def time_series_compare(df, title_=None, height_=450, width_=None, trace_colors=None):
    c1, err1 = (not df.empty), "df cannot be empty"
    c2, err2 = (type(df.index) == pd.DatetimeIndex), "df must have datetime index"
    for req, msg in zip([c1, c2], [err1, err2]):
        if not req:
            print(f"error: {msg}.\nexiting..")
            return

    df[df.lt(0)] = 0  # remove negative values

    ## define figure object
    fig = go.Figure()
    htemplate = (
        "<b>%{fullData.name}</b><br>Value: %{y:.2f}<br><i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
    )
    colors_ = plotly.colors.qualitative.G10
    blu_, red_, grn_ = colors_[0], colors_[1], colors_[3]
    color_props = [blu_, grn_, red_]
    if isinstance(trace_colors, list):
        color_props = trace_colors[:3]

    ## currently only supporting up to first three columns of df
    compare_cols = list(df.columns)[:3]
    kwargs_ = dict(x=df.index, mode="lines", fill="tozeroy", hovertemplate=htemplate)
    fpat_kwargs = [
        dict(bgcolor="rgba(0,0,0,0)", shape="|", size=3, solidity=0.3),
        dict(bgcolor="rgba(0,0,0,0)", shape="-", size=4, solidity=0.2),
        dict(bgcolor="rgba(0,0,0,0)", shape="/", size=3, solidity=0.2),
    ]
    for i, col in enumerate(compare_cols):
        trace_kwargs = kwargs_ | dict(
            name=col,
            line_color=color_props[i],
            line_width=1.5,
            fillpattern=fpat_kwargs[i] | dict(fgcolor=color_props[i]),
        )
        fig.add_trace(go.Scatter(**trace_kwargs, y=df[col].fillna(0)))

    title_txt = title_ if title_ else "Time Series Comparison"
    fig.update_layout(
        height=height_,
        margin=dict(t=50, r=20, b=20, l=20),
        font_size=11,
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
    if width_ is not None:
        fig.update_layout(width=width_)

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
