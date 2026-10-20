import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pvlib
from sklearn.metrics import r2_score

# from ..dataquery.external import query_DTN, query_fracsun
from ..datatools.qc import get_sun, get_daylight_timestamps
from ..utils.datetime import infer_freq, upsample
from ..utils.helpers import quiet_print_function

BACKFILL_COL_MAPPING = {
    "OE.Ambient_Temp": "dtn_temp_air",
    "OE.POA": "poa_global",
    "OE.Wind_Speed": "dtn_wind_speed",
}


def r2_by_day(df: pd.DataFrame, data_col: str, ref_col: str) -> pd.DataFrame:
    data_list = []  # cols: date, r2_score
    for date in sorted(set(df.index.date)):
        df_ = df[df.index.date == date].dropna(axis=0, how="any").copy()
        if df_[[data_col]].empty:
            r2_ = None
        else:
            r2_ = r2_score(y_true=df_[ref_col].values, y_pred=df_[data_col].values)
        data_list.append([date, r2_])
    df = pd.DataFrame(data_list, columns=["date", "r2_score"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def r2_by_half_day(
    df: pd.DataFrame, data_col: str, ref_col: str, location_kwargs: dict
) -> pd.DataFrame:
    freq_td = infer_freq(df.index)
    is_daylight = _daylight_filter(df, location_kwargs)
    data_list = []  # cols: date, start (0/noon), period (am/pm), r2_score
    for date in sorted(set(df.index.date)):
        sun_ = get_sun(date=date, **location_kwargs)
        noon_ = sun_["noon"]
        noon_str = noon_.strftime("%Y-%m-%d %H:%M:%S")  # avoid potential tz issues
        matching_date = is_daylight & (df.index.date == date)
        is_am = matching_date & (df.index < noon_str)
        is_pm = matching_date & (df.index >= noon_str)
        for period, filter_cond in zip(["am", "pm"], [is_am, is_pm]):
            df_ = df[filter_cond].copy()
            notna = df_[[data_col, ref_col]].notna().all(axis=1)
            df_ = df_[notna]
            if df_[[data_col]].empty:
                masd_ = 1000.0
                r2_ = None
            else:
                mcond = df_.index.hour.isin(range(10, 15))
                masd_ = df_.loc[mcond, data_col].diff().abs().mean()
                r2_ = r2_score(y_true=df_[ref_col].values, y_pred=df_[data_col].values)
            start_ = pd.Timestamp(date) if period == "am" else noon_.round(freq_td)
            data_list.append([date, start_, period, r2_, masd_])
    df = pd.DataFrame(data_list, columns=["date", "start", "period", "r2_score", "MASD"])
    df["date"] = pd.to_datetime(df["date"])
    df["start"] = pd.to_datetime(df["start"])
    return df.set_index("date")


def formatted_r2_series(
    df_r2: pd.DataFrame, ref_index: pd.DatetimeIndex, stat_col="r2_score"
) -> pd.Series:
    if all(c in df_r2.columns for c in ["period", "start"]):
        # need to resample, otherwise index.map misses the 'pm' values
        freq_td = infer_freq(ref_index)
        df = df_r2.set_index("start")[[stat_col]].copy()
        df.loc[df.index[-1].ceil("D")] = df.iloc[-1, :].values
        df = df.resample(freq_td).ffill().iloc[:-1, :].copy()
        return ref_index.tz_localize(None).map(df[stat_col])
    return ref_index.tz_localize(None).normalize().map(df_r2[stat_col])


def _grouped_sensor_cols(columns) -> dict:
    output = {}
    matching_cols_ = lambda key_: [
        c
        for c in columns
        if c.startswith(key_) and not any(x in c.lower() for x in ["_r2_", "average", "masd"])
    ]
    for key in BACKFILL_COL_MAPPING.keys():
        cols = matching_cols_(key)
        if len(cols) > 0:
            output.update({key: cols})
    return output


def _process_backfill_input_data(df_sensors, df_dtn, meter_data=None, q=True) -> tuple:
    """Returns grouped sensor_cols dict and combined dataframe."""
    qprint = quiet_print_function(q=q)

    if any(df.index.tzinfo is None for df in [df_sensors, df_dtn]):
        raise ValueError("All data must be timezone-aware.")
    if meter_data is not None:
        if meter_data.index.tzinfo is None:
            raise ValueError("All data must be timezone-aware.")

    check_dfs = [df_sensors, df_dtn]
    if meter_data is not None:
        check_dfs.append(meter_data)
    if len(set(x.index[0].utcoffset() for x in check_dfs)) != 1:
        raise ValueError("All data must be in the same timezone.")

    sensor_cols = _grouped_sensor_cols(df_sensors.columns)
    # if not sensor_cols:
    #     raise KeyError("No valid sensor columns identified in dataset.")

    for key, dtn_col in BACKFILL_COL_MAPPING.items():
        if key in sensor_cols.keys() and dtn_col not in df_dtn.columns:
            qprint(f"Note: missing reference DTN column '{dtn_col}'; excluding group '{key}'")
            del sensor_cols[key]
    # if not sensor_cols:
    #     raise KeyError("No valid sensor columns identified in dataset.")

    df_dtn = df_dtn[list(BACKFILL_COL_MAPPING.values())]

    if "OE.POA" in sensor_cols.keys() and meter_data is None:
        qprint("Note: optional 'meter_data' not provided - no secondary r2 check for POA sensors.")

    df = _add_dtn_to_sensor_data(df_sensors, df_dtn, meter_data)

    return df, sensor_cols


def _daylight_filter(df, location_kwargs):
    start_date = df.index[0].strftime("%Y-%m-%d")
    end_date = df.index[-1].ceil("D").strftime("%Y-%m-%d")
    freq = infer_freq(df.index)
    daylight_tstamps = get_daylight_timestamps(
        start_date,
        end_date,
        freq=freq,
        **location_kwargs,
    )
    return df.index.tz_localize(None).isin(daylight_tstamps)


def _generate_r2_scores(df_combined: pd.DataFrame, sensor_cols: dict, location_kwargs: dict):
    df = df_combined.copy()
    for stype, cols in sensor_cols.items():
        dtn_col = BACKFILL_COL_MAPPING[stype]
        secondary_r2 = "POA" in stype and "site_meter" in df.columns
        for data_col in cols:
            if df[data_col].isna().sum() / len(df) > 0.95:
                df[f"{data_col}_r2_dtn"] = 0.0
                continue

            df_r2 = r2_by_half_day(df, data_col, ref_col=dtn_col, location_kwargs=location_kwargs)
            df[f"{data_col}_r2_dtn"] = formatted_r2_series(df_r2=df_r2, ref_index=df.index)
            if secondary_r2 is True:
                dfn = df[[data_col, "site_meter"]].copy()
                dfn = (dfn - dfn.min()) / (dfn.max() - dfn.min())  # normalized: 0 to 1
                norm_r2 = r2_by_half_day(
                    dfn, data_col, ref_col="site_meter", location_kwargs=location_kwargs
                )
                df[f"{data_col}_r2_meter"] = formatted_r2_series(df_r2=norm_r2, ref_index=df.index)
                df[f"{data_col}_r2_meter"] = df[f"{data_col}_r2_meter"].clip(lower=0.0)
            df[f"{data_col}_MASD"] = formatted_r2_series(
                df_r2=df_r2, ref_index=df.index, stat_col="MASD"
            )
    return df


def _flagged_ranges(df, flag_condition, min_length=None):
    dfx = df[flag_condition].copy()
    dfx["time_diff"] = dfx.index.diff()
    if dfx.shape[0] > 1:
        dfx.iloc[0, -1] = dfx.index[1] - dfx.index[0]

    step = infer_freq(df.index)
    conditions = dfx["time_diff"].eq(step)
    breaks = dfx["time_diff"].gt(step).cumsum()
    length_condition = lambda df_: True if min_length is None else len(df_) >= min_length
    return [df_.index for _, df_ in dfx[conditions].groupby(breaks) if length_condition(df_)]


def _generate_ranges_to_remove_by_col(df_all, sensor_cols, location_kwargs, r2_threshold):
    output = {}
    df = _generate_r2_scores(df_all.resample("h").mean(), sensor_cols, location_kwargs)
    for columns in sensor_cols.values():
        for col in columns:
            no_data = df[col].isna()
            off_from_dtn = df[f"{col}_r2_dtn"].lt(r2_threshold)
            if f"{col}_r2_meter" in df.columns:
                similar_to_meter = df[f"{col}_r2_meter"].gt(r2_threshold)
                cloudy_conditions = df[f"{col}_MASD"].gt(120.0)  # threshold for hourly data
                close_enough_to_keep = similar_to_meter & cloudy_conditions
            else:
                close_enough_to_keep = df.index.notna()  # should be False everywhere

            backfill_condition = no_data | (off_from_dtn & ~close_enough_to_keep)
            if not backfill_condition.any():
                continue

            ranges_to_remove = _flagged_ranges(df, backfill_condition, min_length=2)
            output[col] = ranges_to_remove

    return output


def backfill_weather_sensor_data(
    df_sensors: pd.DataFrame,
    df_dtn: pd.DataFrame,
    location_kwargs: dict,
    meter_data: pd.Series = None,
    r2_threshold: float = 0.75,
    q: bool = True,
):
    qprint = quiet_print_function(q=q)
    qprint("\nBegin weather data backfill.")
    df, sensor_cols = _process_backfill_input_data(df_sensors, df_dtn, meter_data, q=q)
    ranges_to_remove_by_col = _generate_ranges_to_remove_by_col(
        df, sensor_cols, location_kwargs, r2_threshold, q=q
    )
    for pfx, columns in sensor_cols.items():
        qprint(f"\nSensor group: {pfx}")
        for col in columns:
            ranges_to_remove = ranges_to_remove_by_col.get(col, [])
            if not ranges_to_remove:
                qprint(f" -> {col = }; no data removed.")
                continue
            n_removed = 0
            for idx in ranges_to_remove:
                in_range = (df.index >= idx.min()) & (df.index < idx.max())
                df.loc[in_range, col] = np.nan
                n_removed += in_range.sum()
            if n_removed == len(df):
                qprint(f" -> removed {col = }.")
            else:
                qprint(f" -> {col = }; removed {n_removed} values.")

        df[f"{pfx}_average"] = df[columns].mean(axis=1).copy()
        fill_condition = df[f"{pfx}_average"].isna()
        if not fill_condition.any():
            df[pfx] = df[f"{pfx}_average"].copy()
            qprint(f">>> no backfill needed; using average across sensor data.")
            continue
        n_fill = fill_condition.sum()
        ref_col = BACKFILL_COL_MAPPING[pfx]
        df[pfx] = np.where(fill_condition, df[ref_col], df[f"{pfx}_average"])
        qprint(f">>> backfilled {n_fill} values with DTN data ('{ref_col}')")

    for pfx, ref_col in BACKFILL_COL_MAPPING.items():
        if pfx not in df.columns:
            qprint(f"\n>>> added column '{pfx}' using DTN data ('{ref_col}')")
            df[pfx] = df[ref_col].copy()
    processed_cols = list(BACKFILL_COL_MAPPING.keys())
    qprint("\nEnd backfill.")
    return {"df": df.copy(), "backfilled": df[processed_cols].copy()}


def _add_dtn_to_sensor_data(
    df_sensors: pd.DataFrame, df_dtn: pd.DataFrame, meter_data: pd.Series = None
) -> pd.DataFrame:
    # all input data is tz-aware
    sensor_freq = infer_freq(df_sensors.index)
    dtn_freq = infer_freq(df_dtn.index)  # should be hourly
    if sensor_freq > dtn_freq:
        raise ValueError("Unsupported datetime index for sensor data.")

    # resolve potential timezone mismatches (i.e. named tz vs utc-offset)
    ref_tzinfo = df_sensors.index.tzinfo
    ref_offset = df_sensors.index[0].utcoffset()
    different_tz = lambda df: df.index.tzinfo != ref_tzinfo
    same_offset = lambda df: df.index[0].utcoffset() == ref_offset
    need_to_convert = lambda df: different_tz(df) and same_offset(df)
    if need_to_convert(df_dtn):
        df_dtn = df_dtn.tz_convert(ref_tzinfo)

    # upsample DTN data to match sensor data (if needed)
    if dtn_freq > sensor_freq:
        df_dtn = upsample(df=df_dtn, new_freq=sensor_freq, agg_type="interpolate")
        df_dtn["poa_global"] = df_dtn["poa_global"].fillna(0.0)

    df_list = [df_sensors, df_dtn]

    # similar steps for meter data (if provided)
    if meter_data is not None:
        if need_to_convert(meter_data):
            meter_data = meter_data.tz_convert(ref_tzinfo)
        mfreq = infer_freq(meter_data.index)
        if mfreq > sensor_freq:
            meter_data = upsample(meter_data.to_frame(), sensor_freq).iloc[:, 0]
            meter_data = meter_data.fillna(0.0)
        elif mfreq < sensor_freq:
            meter_data = meter_data.resample(sensor_freq).mean()
        df_meter = meter_data.to_frame(name="site_meter")
        df_list.append(df_meter)

    return pd.concat(df_list, axis=1)


abbr_mapping = {"OE.Ambient_Temp": "AmbTemp", "OE.POA": "POA", "OE.Wind_Speed": "Wind"}


def format_backfill_data_to_match_legacy(df_result):
    # df is from output of backfill_weather_sensor_data
    df = df_result.copy()
    for col in ["OE.Ambient_Temp", "OE.POA", "OE.Wind_Speed"]:
        sensor_abbr = abbr_mapping[col]
        avgcol = f"{col}_average"
        if avgcol in df.columns:
            df[f"Average_Across_{sensor_abbr}"] = df[avgcol].copy()
        df[f"Processed_{sensor_abbr}"] = df[col].copy()
    return df


def _get_backfilled_ranges_by_sensor_type(df_formatted) -> dict[str, list[pd.DatetimeIndex]]:
    # df_formatted is output from format_backfill_data_to_match_legacy
    output = {}
    df = df_formatted.copy()
    for oe_sensor, abbr in abbr_mapping.items():
        pcol = f"Processed_{abbr}"
        if pcol not in df.columns:
            continue
        dtn_col = BACKFILL_COL_MAPPING[oe_sensor]
        dfx = df[df[pcol] == df[dtn_col]].copy()
        dfx["time_diff"] = dfx.index.diff()
        if dfx.shape[0] > 1:
            dfx.iloc[0, -1] = dfx.index[1] - dfx.index[0]

        step = infer_freq(df.index)
        conditions = dfx["time_diff"].eq(step)
        breaks = dfx["time_diff"].gt(step).cumsum()

        output[oe_sensor] = [df_.index for _, df_ in dfx[conditions].groupby(breaks)]
    return output


def backfill_subplots(df, resample=True):
    sensor_cols = _grouped_sensor_cols(df.columns)
    n_rows = len(sensor_cols)
    s_titles = [f"<b>{grp}</b>" for grp in sensor_cols.keys()]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=s_titles,
        vertical_spacing=0.2 / n_rows,
    )
    htemplate = (
        "<b>%{fullData.name}</b><br>Value: %{y:.2f}<br>" "<i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
    )

    backfilled_ranges = _get_backfilled_ranges_by_sensor_type(df)

    if resample is True and infer_freq(df.index) < pd.Timedelta(minutes=15):
        df = df.resample("15min").mean().copy()
    freq_td = infer_freq(df.index)

    shapes = []
    for row, item in enumerate(BACKFILL_COL_MAPPING.items(), start=1):
        grp, dtn_col = item
        if grp not in sensor_cols:
            continue
        grp_kwargs = dict(legendgroup=grp, legendgrouptitle_text=f"<b>{grp}</b>", legendrank=row)

        dtn_trace = go.Scattergl(
            x=df.index,
            y=df[dtn_col],
            name=dtn_col,
            mode="lines",
            line=dict(color="#777777"),
            hovertemplate=htemplate,
            **grp_kwargs,
        )

        sensor_traces = []
        for col in sensor_cols[grp]:
            sensor_traces.append(
                go.Scattergl(
                    x=df.index,
                    y=df[col],
                    name=col,
                    mode="lines",
                    opacity=0.5,
                    **grp_kwargs,
                )
            )

        processed_col = "Processed_" + abbr_mapping[grp]

        data_trace = go.Scattergl(
            x=df.index,
            y=df[processed_col],
            mode="lines",
            name=processed_col,
            line=dict(color="#000000", dash="dot"),
            **grp_kwargs,
        )

        fig.add_traces([dtn_trace, *sensor_traces, data_trace], rows=row, cols=1)

        for idx in backfilled_ranges[grp]:
            shapes.append(
                dict(
                    type="rect",
                    x0=idx.min().floor(freq_td),
                    x1=idx.max().ceil(freq_td),
                    y0=0,
                    y1=1,
                    yref=f"y{row} domain".replace("y1", "y"),
                    fillcolor="lightsalmon",
                    opacity=0.4,
                    line_width=0,
                    layer="below",
                )
            )

    fig.update_xaxes(tickformat="%m/%d", tick0=df.index[0], dtick=str(86400000 * 7))
    fig.update_layout(
        font_size=9,
        paper_bgcolor="#fbfbfb",
        margin=dict(t=30, r=20, b=10, l=20),
        legend=dict(
            x=1.01,
            xanchor="left",
            groupclick="toggleitem",
            grouptitlefont=dict(size=11),
            tracegroupgap=16,
        ),
        shapes=shapes,
    )

    # format subplot titles (are actually annotations in subplot figures)
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].update(font=dict(size=12), x=0, xanchor="left")

    return fig
