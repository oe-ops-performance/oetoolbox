import calendar
import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_waterfall(x, y, measure, text):
    kwargs = dict(
        # name="20",
        orientation="v",
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    )
    fig = go.Figure(go.Waterfall(measure=measure, x=x, text=text, y=y, **kwargs))
    fig.update_layout(title="test waterfall", showlegend=False, margin=dict(t=80, b=60, l=60, r=60))
    return fig


def combiner_heatmap(site, analysis_data, add_overlay=True):
    if not all(k in analysis_data for k in ["categories", "lost_mwh", "flagged"]):
        raise KeyError("One or more required keys missing from input.")
    df_flag = analysis_data["categories"]
    df_loss = analysis_data["lost_mwh"]

    # zero out losses related to curtailment
    df_loss[df_flag.eq("CURT")] = ""

    # similar for comms/cal
    if any("comms" in dict_ for dict_ in analysis_data["flagged"].values()):
        df_loss = df_loss.replace(0, "")
    elif any("calibration" in dict_ for dict_ in analysis_data["flagged"].values()):
        df_loss = df_loss.replace(0, "")

    xvals = list(df_loss.columns)  # cmb names
    yvals = list(df_loss.index)  # inv names
    zvals = df_loss.values
    textvals = df_flag.values  # replace("-/-", "").values

    fig = go.Figure(
        go.Heatmap(
            x=xvals,
            y=yvals,
            z=zvals,
            colorscale=plotly.colors.sequential.Reds,
            colorbar=dict(
                len=250,
                lenmode="pixels",
                yref="paper",
                y=1,
                yanchor="top",
                title=dict(text="lost MWh", font_size=12),
            ),
            hoverongaps=False,
            hovertemplate="%{y}<br>%{x}<br>Loss: %{z:.2f} MWh<br>Type: %{text}",
            text=textvals,
            texttemplate="%{text}",
            textfont_size=10,
        )
    )

    if add_overlay is True:
        df_nonexist = pd.DataFrame(
            np.nan, index=np.arange(df_flag.shape[0]), columns=np.arange(df_flag.shape[1])
        )
        df_nonexist[(df_flag == "-/-").values] = 0  # ""
        fig.add_trace(
            go.Heatmap(
                z=df_nonexist.values,
                colorscale=[[0, "#ffffff"], [1, "#ffffff"]],
                showscale=False,
                hoverinfo="skip",
            )
        )

    plot_area_width = 43 * len(xvals)
    plot_area_height = 21 * len(yvals)
    margin = dict(t=80, b=150, l=100, r=140)
    fig_height = plot_area_height + margin["t"] + margin["b"]
    fig_width = plot_area_width + margin["l"] + margin["r"]
    fig.update_layout(
        title=dict(text=site, y=1, yanchor="bottom", yref="paper", pad_b=25),
        margin=margin,
        height=fig_height,
        width=fig_width,
    )

    fig.update_xaxes(tickson="boundaries", tickangle=45)
    fig.update_yaxes(tickson="boundaries")

    return fig
