import pandas as pd
import plotly.graph_objects as go

from ..utils.pi import PIDataset
from ..dataquery.pitools import pipoint_list, query_pi_recordedvalues


VOLTAGE_BAND = 4  # kV

VOLTAGE_PIPOINT_DICT = {
    "Route 66": [f"RT66.RTAC.Tenska_ERCOT_VOLT_{x}" for x in ("Target", "POI")],
    "South Plains II": [f"SP2.RTAC.Tenaska_ERCOT_VOLT_{x}" for x in ("Target", "POI")],
}


def query_monthly_voltage_data(site, year, month, freq="1h"):
    if site not in VOLTAGE_PIPOINT_DICT:
        raise ValueError(f"Invalid site specified: {site}")
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.DateOffset(months=1)
    start_date, end_date = map(lambda t: t.strftime("%Y-%m-%d"), [start, end])
    kwargs = dict(start_date=start_date, end_date=end_date, freq=freq, keep_tzinfo=True)
    dataset = PIDataset.from_pipoints(site, VOLTAGE_PIPOINT_DICT[site], **kwargs)
    return dataset.data


def query_monthly_voltage_data_recordedvalues(site, year, month, q=True):
    if site not in VOLTAGE_PIPOINT_DICT:
        raise ValueError(f"Invalid site specified: {site}")
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.DateOffset(months=1)
    start_date, end_date = map(lambda t: t.strftime("%Y-%m-%d"), [start, end])
    pipt_list = pipoint_list(VOLTAGE_PIPOINT_DICT[site])
    df = query_pi_recordedvalues(site, start_date, end_date, pipoint_list=pipt_list, q=q)
    df["Attribute"] = df["Attribute"].astype(str)
    df_poi = df.loc[df.Attribute.str.contains("POI")].copy()
    df_target = df.loc[df.Attribute.str.contains("Target")].copy()
    df_target = df_target.resample("1s").ffill()

    target_tag, poi_tag = VOLTAGE_PIPOINT_DICT[site]

    dfp = df_poi[["Value"]].rename(columns={"Value": poi_tag}).copy()
    dft = df_target[["Value"]].rename(columns={"Value": target_tag}).copy()
    df_query = dfp.join(dft, how="left")
    df_query = df_query[~df_query.index.duplicated()].copy()
    return df_query


def format_query_dataframe(df_query):
    df = df_query.copy()
    df.columns = df.columns.map(lambda c: c.split("ERCOT_")[-1].casefold())
    for col in ["volt_poi", "volt_target"]:
        if df[col].max() > 1e3:
            df[col] = df[col].div(10)
    df["delta"] = df["volt_poi"].sub(df["volt_target"])
    return df


def process_excursion_events(df, threshold=VOLTAGE_BAND, recordedvalues=False):
    """Returns a list of DataFrames where the voltage delta exceeds the voltage band."""
    if recordedvalues:
        df = df.resample("1s").ffill().copy()
    condition = df["delta"].abs().gt(threshold)
    breaks = (~condition).cumsum()
    return [group for _, group in df[condition].groupby(breaks)]


def create_excursion_event_table(df_list):
    columns = [
        "Start Time",
        "Start Setpoint (kV)",
        "End Time",
        "End Setpoint (kV)",
        "Duration (mins)",
        "Max Delta (kV)",
        "Min Delta (kV)",
    ]
    data = []
    for df in df_list:
        start, end = df.index[0], df.index[-1]
        start_sp = df.at[start, "volt_target"]
        end_sp = df.at[end, "volt_target"]
        duration = (end - start).total_seconds() / 60
        max_delta = df["delta"].max()
        min_delta = df["delta"].min()
        data.append([start, start_sp, end, end_sp, duration, max_delta, min_delta])
    return pd.DataFrame(data, columns=columns)


def create_excursion_plot(df_query, site):
    dff = format_query_dataframe(df_query).copy()
    dff["lower_limit"] = -VOLTAGE_BAND
    dff["upper_limit"] = VOLTAGE_BAND

    title_text = f"{site} - Voltage Profile - {dff.index[0].strftime('%b-%Y')}"
    fig = go.Figure()
    for col in ["volt_target", "volt_poi"]:
        fig.add_trace(go.Scattergl(x=dff.index, y=dff[col], mode="lines", name=col))

    fig.update_layout(
        height=500,
        width=1050,
        title=dict(text=title_text, pad_b=15, yanchor="bottom", y=1, yref="paper"),
        legend_x=1.05,
        margin=dict(t=50, b=20, l=70, r=20),
        yaxis_title=dict(text="Voltage (kV)", standoff=10),
        yaxis2=dict(
            title=dict(text="Voltage Delta (kV)", standoff=10),
            overlaying="y",
            side="right",
            showgrid=False,
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=dff.index,
            y=dff["delta"],
            mode="lines",
            line=dict(color="black", width=1),
            yaxis="y2",
            name="delta",
        )
    )
    for limit_col in ["lower_limit", "upper_limit"]:
        fig.add_trace(
            go.Scatter(
                x=dff.index,
                y=dff[limit_col],
                mode="lines",
                name=limit_col,
                line=dict(color="black", width=1, dash="dash"),
                yaxis="y2",
            )
        )

    event_df_list = process_excursion_events(dff)
    for dfg in event_df_list:
        x0, x1 = dfg.index.min(), dfg.index.max()
        timedelta = x1 - x0
        if len(event_df_list) > 50:
            if timedelta < pd.Timedelta("15min"):
                continue
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="lightsalmon",
            opacity=0.4,
            layer="below",
            line_width=0,
        )

    return fig


def run_voltage_analysis(site, year, month, include_plot=False, q=True) -> dict:
    """Queries voltage data for a given site and month, then processes and visualizes the data."""
    df_query = query_monthly_voltage_data_recordedvalues(site, year, month, q=q)
    df = format_query_dataframe(df_query)
    df_list = process_excursion_events(df)
    df_events = create_excursion_event_table(df_list)
    output = {"data": df, "events": df_events}
    if include_plot:
        plot_fig = create_excursion_plot(df_query, site)
        output.update({"plot": plot_fig})
    return output
