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
