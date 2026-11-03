from astral.sun import sun
from astral import LocationInfo
import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pvanalytics.features.daytime as daytime
from pvanalytics.features.orientation import tracking_nrel, fixed_nrel


from ..utils.dataset import Dataset
from ..utils.datetime import get_continuous_ranges, infer_freq
from ..utils.helpers import quiet_print_function


def _get_the_sun(observer, date):
    target_date = pd.Timestamp(date).floor("D")
    date_before = date - pd.Timedelta(days=1)
    date_after = date + pd.Timedelta(days=1)
    for d in (target_date, date_before, date_after):
        try:
            the_sun = sun(observer, date=d)
            return the_sun
        except ValueError:
            continue
        except Exception as e:
            raise e
    raise ValueError


def get_sun(
    date: str, tz: str, lat: float, lon: float, name: str = "", region: str = ""
) -> dict[str, pd.Timestamp]:
    """Returns dictionary with times of the sun on given date for dawn, sunrise, noon, sunset, and dusk.

    Parameters
    ----------
    date : str | pd.Timestamp
        The date for which to calculate sun times.
    tz : str
        The timezone of the desired location.
    lat : float
        The latitude of the desired location.
    lon : float
        The longitude of the desired location.
    name : str, optional
        Location name. Does not affect calculations.
    region : str, optional
        Region location is in. Does not affect calculations.

    Returns
    -------
    dict[str, pd.Timestamp]
        A dictionary with keys = dawn, sunrise, noon, sunset, dusk and corresponding pd.Timestamp values
        with the calculated sun times at the specified location.
    """
    location = LocationInfo(name=name, region=region, timezone=tz, latitude=lat, longitude=lon)
    observer = location.observer
    target_date = pd.Timestamp(date).floor("D")
    the_sun = _get_the_sun(observer, date=target_date)
    output = {}
    for key, dtime in the_sun.items():
        local_tstamp = pd.Timestamp(dtime).tz_convert(tz).tz_localize(None)
        if local_tstamp.date() != dtime.date():
            local_tstamp = pd.Timestamp(dt.datetime.combine(dtime.date(), local_tstamp.time()))
        elif local_tstamp.date() != target_date.date():
            local_tstamp = pd.Timestamp(
                dt.datetime.combine(target_date.date(), local_tstamp.time())
            )
        tstamp = local_tstamp.round("min")  # round to nearest minute
        if pd.Timestamp(date).tzinfo is not None:
            tstamp = tstamp.tz_localize(tz=tz)
        output.update({key: tstamp})
    return output


def get_sun_times(
    start_date: str,
    end_date: str,
    tz: str,
    lat: float,
    lon: float,
    name: str = "",
    region: str = "",
) -> list[dict[str, pd.Timestamp]]:
    """Returns list of dictionaries with dawn and dusk times for each day in specified date range."""
    location_kwargs = dict(tz=tz, lat=lat, lon=lon, name=name, region=region)
    sun_times = []
    for date in pd.date_range(start_date, end_date):
        sun_dict = get_sun(date, **location_kwargs)
        sun_times.append({"dawn": sun_dict["dawn"], "dusk": sun_dict["dusk"]})
    return sun_times


def get_daylight_timestamps(
    start_date: str,
    end_date: str,
    tz: str,
    lat: float,
    lon: float,
    name: str = "",
    region: str = "",
    freq="1min",
    limits: tuple = ("dawn", "dusk"),
):
    """Build a list of daylight timestamps at ``freq`` between ``start_date`` and ``end_date``.

    ``limits`` selects which sun-event keys bound each day's window — ``("dawn", "dusk")``
    (civil twilight, default) or ``("sunrise", "sunset")`` (tighter, geometric horizon).
    For ``freq`` larger than 1 minute the per-day bounds are floored/ceiled to the freq
    grid so the returned timestamps land on grid.
    """
    if limits not in (("dawn", "dusk"), ("sunrise", "sunset")):
        raise ValueError("limits must be ('dawn', 'dusk') or ('sunrise', 'sunset')")
    location_kwargs = dict(tz=tz, lat=lat, lon=lon, name=name, region=region)
    snap_to_grid = pd.Timedelta(freq) > pd.Timedelta(minutes=1)
    timestamp_list = []
    for date in pd.date_range(start_date, end_date):
        sun_times = get_sun(date, **location_kwargs)
        start_, end_ = sun_times[limits[0]], sun_times[limits[1]]
        if snap_to_grid:
            start_ = start_.floor(freq)
            end_ = end_.ceil(freq)
        daylight_range = pd.date_range(start_, end_, freq=freq, inclusive="left")
        timestamp_list.extend(list(daylight_range))
    return list(filter(lambda t: t < pd.Timestamp(end_date), timestamp_list))


def comparison_figure(
    df_inv_0, df_inv_1, resample=True, ignore_no_changes=False, q=True, **location_kwargs
):
    qprint = quiet_print_function(q=q)

    common_cols = [c for c in df_inv_0.columns if c in df_inv_1.columns]
    df0 = df_inv_0[common_cols].copy()
    df1 = df_inv_1[common_cols].copy()

    freq = pd.infer_freq(df0.index)
    if not freq:
        raise ValueError("Dataframe does not have a uniform frequency/interval.")
    freq_timedelta = df0.index.diff().max()

    compare_columns = list(
        set(df0.round(4).compare(df1.round(4)).columns.get_level_values(level=0).to_list())
    )

    if len(compare_columns) == 0:
        print("No changes detected.")
        if ignore_no_changes is False:
            return
        compare_columns = ["all"]

    # resample to 15-min for plot
    if resample is True:
        df0_plot = df0.resample("15min").mean().copy()
        df1_plot = df1.resample("15min").mean().copy()
    else:
        df0_plot, df1_plot = df0.copy(), df1.copy()

    plot_freq = pd.infer_freq(df1_plot.index)
    n_rows = len(compare_columns)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=compare_columns,
        vertical_spacing=0.275 / n_rows,
        specs=[[{"secondary_y": True}]] * n_rows,
    )
    hvtemp = (
        "<b>%{fullData.name}</b><br>Value: %{y:.2f}<br><i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
    )
    kwargs = dict(x=df1_plot.index.copy(), mode="lines", showlegend=False, hovertemplate=hvtemp)

    # add daylight trace
    if all(k in location_kwargs for k in ["tz, lat, lon"]):
        loc_kwargs = {
            k: val for k, val in location_kwargs.items() if k in ["tz, lat, lon", "name", "region"]
        }
        start = df0.index.min().floor("D")
        end = df0.index.max().ceil("D")
        daylight_tstamps = get_daylight_timestamps(start, end, **loc_kwargs)
        df_daylight = pd.DataFrame(
            index=df0.index, data={"is_daytime": df0.index.isin(daylight_tstamps).astype(int)}
        )
        fig.add_trace(
            go.Scattergl(
                x=df_daylight.index,
                y=df_daylight["is_daytime"],
                name="daytime",
                line=dict(color="#999", dash="dot", width=1),
                showlegend=False,
            ),
            secondary_y=True,
            row="all",
            col=1,
        )
    if compare_columns[0] == "all":
        for c in common_cols:
            fig.add_trace(
                go.Scattergl(**kwargs, y=df0_plot[c], name=c), secondary_y=False, row=1, col=1
            )
    else:
        for i, col in enumerate(compare_columns, start=1):
            trace0 = go.Scattergl(
                **kwargs,
                y=df0_plot[col],
                name=f"{col}_ORIGINAL",
                line=dict(color="#333", dash="dot", width=1),
            )
            fig.add_trace(trace0, row=i, col=1, secondary_y=False)
            trace1 = go.Scattergl(**kwargs, y=df1_plot[col], name=f"{col}_NEW")
            fig.add_trace(trace1, row=i, col=1, secondary_y=False)

            # add shaded regions to subplot to highlight removed data
            df_cmp = df0[[col]].round(4).compare(df1[[col]].round(4))
            df_cmp.columns = df_cmp.columns.droplevel(0)  # drop level of multiindex columns
            df_cmp["time_diff"] = df_cmp.index.diff()
            if df_cmp.shape[0] > 1:
                df_cmp.iloc[0, -1] = df_cmp.index[1] - df_cmp.index[0]

            # group rows with continuous date ranges
            data_interval = freq_timedelta
            grp_conditions = df_cmp["time_diff"].eq(data_interval)

            # separate/break groups using cumulative sum of condition: [timedelta > 1min]
            breaks_ = df_cmp["time_diff"].gt(data_interval).cumsum()

            # use groupby function to separate df_cmp into continuous ranges according to the above conditions
            grouped_df_list = [dfg for _, dfg in df_cmp[grp_conditions].groupby(breaks_)]

            vrect_kwargs = dict(
                row=i, line_width=0, opacity=0.4, layer="below", fillcolor="lightsalmon"
            )
            n_ranges = 0
            for dfg in grouped_df_list:
                x0_, x1_ = dfg.index.min(), dfg.index.max()
                if (x1_ - x0_) < pd.Timedelta(hours=2):
                    continue
                x0_plot = x0_.floor(plot_freq)
                x1_plot = x1_.floor(plot_freq)
                fig.add_vrect(x0=x0_plot, x1=x1_plot, **vrect_kwargs)
                n_ranges += 1
            qprint(f"{col = }, {n_ranges = }")

    # set alignment for subplot titles (annotations in subplot figure)
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].update(font_size=12, x=1, xref="paper", xanchor="right")

    fig.update_layout(
        font_size=9,
        height=max(400, 120 * n_rows),
        margin=dict(t=40, b=20, l=0, r=20),
        title=dict(font_size=14, text=f"<b>QC Comparison</b>", x=0.015),
        hoverlabel_font_size=11,
    )
    fig.update_yaxes(showline=True, linewidth=1, linecolor="#777", secondary_y=False)
    fig.update_yaxes(secondary_y=True, side="right", rangemode="tozero")
    fig.update_xaxes(showline=True, linewidth=1, linecolor="#777")

    if all([("OE.ActivePower" in c) for c in compare_columns]) or compare_columns[0] == "all":
        ymax_ = df1.max().max() * 1.05
        fig.update_yaxes(range=[0, ymax_], secondary_y=False)
    else:
        for i, col in enumerate(compare_columns, start=1):
            ymin_ = df1[col].min() * 0.95
            ymax_ = df1[col].max() * 1.05
            fig.update_yaxes(row=i, range=[ymin_, ymax_], secondary_y=False)

    return fig


# outlier detection method: interquartile range (IQR)
def get_iqr_bounds(series, iqr_multiplier=1.5):
    """Returns lower/upper bounds for IQR with specified multiplier."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    iqr = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * iqr
    upper_bound = Q3 + iqr_multiplier * iqr
    return lower_bound, upper_bound


def iqr_outlier_condition(series, iqr_multiplier=1.5):
    """Returns condition for use in outlier removal."""
    lower_bound, upper_bound = get_iqr_bounds(series, iqr_multiplier)
    return series.lt(lower_bound) | series.gt(upper_bound)


# outlier detection method: z_score (for rolling window)
def z_score_outlier_condition(series: pd.Series, z_threshold: int = 3):
    """condition indicates values that exceed z score threshold (# of std. devs. from mean)
    -> window will be determined based on data interval as follows:
        freq=1min -> window=30
        freq=5min -> window=6
        freq=15min -> window=4
        freq=1h -> window=2 (TODO: determine whether this should be supported)
    """
    freq_mins = int(series.index.diff().max().total_seconds() / 60)
    window = {1: 30, 5: 6, 15: 4, 60: 2}.get(freq_mins, None)
    if window is None:
        raise KeyError("Unsupported data interval/frequency detected.")
    rolling_mean = series.rolling(window=window).mean().copy()
    rolling_std = series.rolling(window=window).std().copy()
    rolling_z = (series - rolling_mean) / rolling_std
    return rolling_z.abs().gt(z_threshold)


def remove_outliers(series, method, **kwargs) -> pd.Series:
    """removes outliers from series using selected method; returns number of points removed
    -> NOTE: updates the existing series object; returns separate series with removed outliers.
    """
    if method not in ("iqr", "z_score"):
        raise ValueError(f"Unsupported {method = }.")

    if method == "iqr":
        iqr_multiplier = kwargs.get("iqr_multiplier", 1.5)
        is_outlier = iqr_outlier_condition(series=series, iqr_multiplier=iqr_multiplier)
    elif method == "z_score":
        z_threshold = kwargs.get("z_threshold", 3)
        is_outlier = z_score_outlier_condition(series=series, z_threshold=z_threshold)

    series_outliers = series.loc[is_outlier].copy()
    series.loc[is_outlier] = np.nan
    return series_outliers


def detect_and_remove_outliers(series, method, **kwargs):
    """removes outliers from series using selected method;"""
    if method not in ("iqr", "z_score"):
        raise ValueError(f"Unsupported {method = }.")

    if method == "iqr":
        iqr_multiplier = kwargs.get("iqr_multiplier", 1.5)
        is_outlier = iqr_outlier_condition(series=series, iqr_multiplier=iqr_multiplier)
    elif method == "z_score":
        z_threshold = kwargs.get("z_threshold", 3)
        is_outlier = z_score_outlier_condition(series=series, z_threshold=z_threshold)

    outliers = series.loc[is_outlier].copy()
    new_series = series.copy()
    new_series.loc[is_outlier] = np.nan
    return dict(data=new_series, outliers=outliers)


def _mirrored_data(series, noon_time):
    # where series is one day of data, and split time is noon
    df = series.to_frame(name=series.name if series.name else "original")
    df["mirrored_part1"] = df.iloc[:, 0].copy()
    df["mirrored_part2"] = df.iloc[:, 0].copy()

    _morning = series.index < noon_time
    _afternoon = series.index >= noon_time

    part1 = series[_morning]
    part2 = series[_afternoon]

    mirrored_1_vals = part1.iloc[::-1].values.tolist()
    if len(mirrored_1_vals) > len(part2):
        mirrored_1_vals = mirrored_1_vals[: len(part2)]
    elif len(mirrored_1_vals) < len(part2):
        mirrored_1_vals.extend([np.nan] * (len(part2) - len(mirrored_1_vals)))
    df.loc[_afternoon, "mirrored_part1"] = mirrored_1_vals

    mirrored_2_vals = part2.iloc[::-1].values.tolist()
    if len(mirrored_2_vals) > len(part1):
        n_diff = len(mirrored_2_vals) - len(part1)
        mirrored_2_vals = mirrored_2_vals[n_diff:]
    elif len(mirrored_2_vals) < len(part1):
        n_diff = len(part1) - len(mirrored_2_vals)
        mirrored_2_vals = ([np.nan] * n_diff) + mirrored_2_vals
    df.loc[_morning, "mirrored_part2"] = mirrored_2_vals

    return df


# Assuming 'poa_measured' is a pandas Series with a DatetimeIndex
# and 'solar_noon' is a Series containing the time of solar noon for each day.


def analyze_half_days(poa_measured, solar_noon):
    # Initialize a mask for bad data (True = BAD (remove), False = GOOD (keep))
    qc_mask = pd.Series(False, index=poa_measured.index)

    # Group by date to evaluate day-by-day
    for date, day_data in poa_measured.groupby(poa_measured.index.date):
        noon_time = solar_noon.loc[date].strftime("%Y-%m-%d %H:%M")  # prevent tz mismatch errors
        mirrored_data = _mirrored_data(day_data, noon_time)

        # Split day data into morning and afternoon chunks
        morning = day_data[day_data.index < noon_time]
        afternoon = day_data[day_data.index >= noon_time]

        # --- Evaluate Morning (Mirror the morning curve into the afternoon) ---
        synthetic_morning_day = mirrored_data["mirrored_part1"].copy()

        daytime_mask = daytime.power_or_irradiance(synthetic_morning_day)

        # fixed_nrel returns True if the profile matches a fixed tilt profile
        is_fixed_tilt_am = fixed_nrel(synthetic_morning_day, daytime_mask)
        if is_fixed_tilt_am.all():
            qc_mask.loc[morning.index] = True

        # --- Evaluate Afternoon (Mirror the afternoon curve into the morning) ---
        synthetic_afternoon_day = mirrored_data["mirrored_part2"].copy()

        daytime_mask = daytime.power_or_irradiance(synthetic_afternoon_day)
        is_fixed_tilt_pm = fixed_nrel(synthetic_afternoon_day, daytime_mask)
        if is_fixed_tilt_pm.all():
            qc_mask.loc[afternoon.index] = True

    return qc_mask


def process_poa_sensor_data(df_, location_kwargs):
    df_ = df_.filter(like="OE.POA_").copy()
    if df_.empty:
        raise ValueError("The only columns in the provided dataframe should be OE.POA sensors.")
    poa_columns = list(df_.columns)
    noon_tstamps = {
        date: get_sun(date, **location_kwargs)["noon"] for date in sorted(set(df_.index.date))
    }
    solar_noon = pd.Series(index=noon_tstamps.keys(), data=noon_tstamps.values())

    df = df_.copy()
    for col in poa_columns:
        bad_data_flag = analyze_half_days(poa_measured=df_[col], solar_noon=solar_noon)
        df[f"{col}_flagged"] = bad_data_flag.astype(int)
        df[f"{col}_processed"] = np.where(bad_data_flag, np.nan, df[col])

    return {key: df.filter(like=f"_{key}").copy() for key in ("flagged", "processed")}


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


def _data_group(column):
    contains = lambda *args: all(x.lower() in column.lower() for x in args)
    if contains("activepower"):
        return "inverter"
    elif contains("meter", "mw"):
        return "meter"
    elif contains("wind", "speed"):
        return "wind"
    elif contains("poa") or contains("ghi"):
        return "irradiance"
    elif contains("temp"):
        return "temperature"
    return "unknown"


def flag_interpolated_data(
    series: pd.Series,
    location_kwargs: dict = None,
    min_duration="2h",
    max_plateau="8h",
    tol_rel: float = 1e-6,
    zero_frac: float = 0.01,
    ramp_low_frac: float = 0.05,
    night_pct_limit: float = 0.05,
) -> pd.DataFrame:
    """Finds ranges of interpolated (linear) or held/stuck (flatline) data in a single series.

    Method: a point is "collinear" when the second difference using its two neighbors,
    x[t+1] - 2*x[t] + x[t-1], is ~0 (within ``tol_rel`` * data scale). Real measurements almost
    never stay collinear for hours, but linear interpolation and forward-fill do so exactly (up to
    rounding). A flagged range holds only the *interior* points; the bracketing anchor values on
    either side are real measurements and are not included. Each candidate run must also stay
    within tolerance of the straight line between its anchors (guards against slow curvature that
    individually passes the second-difference test).

    Candidates are then checked for legitimate flat conditions, which are not flagged:
        - flat at/near zero (|value| <= ``zero_frac`` * scale), e.g. nighttime or idle
        - flat non-zero plateau (clipping/curtailment) lasting <= ``max_plateau`` and, if
          ``location_kwargs`` is given, entirely in daylight (<= ``night_pct_limit`` night points)
    - sloped run staying below ``ramp_low_frac`` * scale (e.g. dawn/dusk ramps from zero)
    All other sloped runs are flagged. NOTE: a genuine ramp-rate-limited recovery with a large
    amplitude cannot be told apart from interpolation by shape alone; raise ``min_duration``
    if those are a concern.

    Parameters
    ----------
    series : pd.Series
        Numeric data with a DatetimeIndex. Duplicate/missing timestamps are handled; gaps are
        filled with NaN on the inferred regular grid and break any run.
    location_kwargs : dict, optional
        Keys for ``get_daylight_timestamps`` (tz, lat, lon, ...). Enables the night-time check.
    min_duration : str | pd.Timedelta
        Minimum run length to flag (number of interior points x data interval).
    max_plateau : str | pd.Timedelta
        Longest flat non-zero run still considered a plausible clipping/curtailment plateau.
    tol_rel : float
        Collinearity tolerance relative to data scale (99th percentile of non-collinear values).
    zero_frac : float
        Flat runs with |value| below this fraction of scale are treated as zero.
    ramp_low_frac : float
        Sloped runs whose |value| never exceeds this fraction of scale are not flagged.
    night_pct_limit : float
        Max fraction of a flat non-zero run allowed to fall outside daylight to stay exempt.

    Returns
    -------
    pd.DataFrame
        One row per flagged range: start, end (inclusive, interior points), n_points,
        kind ("linear" | "flatline"), start_value, end_value.
    """
    columns = ["start", "end", "n_points", "kind", "start_value", "end_value"]
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s = s[~s.index.duplicated()].sort_index()
    if s.notna().sum() < 3:
        return pd.DataFrame(columns=columns)

    freq = infer_freq(s.index)
    s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq=freq))
    d2 = s.shift(-1) + s.shift(1) - 2 * s  # centered at t; NaN if any of 3 points is missing

    def _scale(x):
        v = x.dropna().abs().quantile(0.99) if x.notna().any() else 0.0
        return v if v > 0 else 1.0

    # two-pass scale: exclude collinear points so long held/interpolated spans don't skew it
    scale = _scale(s)
    is_collinear = d2.abs().le(tol_rel * scale)
    scale = _scale(s[~is_collinear])
    zero_tol = zero_frac * scale

    # tolerances: relative to scale, but never tighter than the data's own rounding noise
    # (rounding to q adds up to q/2 per point -> 2q in the second difference, ~q in a line fit)
    tol = tol_rel * scale
    fit_tol = 5 * tol
    q = _data_resolution(s.to_numpy())
    if q is not None:
        tol = max(tol, 2.1 * q)
        fit_tol = max(fit_tol, 1.5 * q)
    is_collinear = d2.abs().le(tol).to_numpy()

    min_points = int(np.ceil(pd.Timedelta(min_duration) / freq))
    max_plateau_td = pd.Timedelta(max_plateau)
    values = s.to_numpy()

    # contiguous True runs as [start, stop) positions of interior points; anchors at start-1, stop
    edges = np.flatnonzero(np.diff(np.r_[0, is_collinear.astype(np.int8), 0]))
    stack = [(i0 - 1, i1) for i0, i1 in zip(edges[::2], edges[1::2])]  # (anchor_a, anchor_b)
    segments = []
    while stack:
        a, b = stack.pop()
        n = b - a - 1
        if n < min_points:
            continue
        y = values[a : b + 1]
        dev = np.abs(y - (y[0] + (y[-1] - y[0]) * np.arange(n + 2) / (n + 1)))
        if dev.max() > fit_tol:  # curvature hidden by loose tolerance -> split at worst point
            m = a + int(dev.argmax())
            stack.extend([(a, m), (m, b)])
            continue
        segments.append((a, b))

    rows = []
    for a, b in sorted(segments):
        i0, i1, n = a + 1, b, b - a - 1
        y = values[a : b + 1]
        if np.abs(y).max() <= zero_tol:
            continue  # flat at/near zero (night, idle)
        is_flat = np.ptp(y) <= fit_tol
        if not is_flat and np.abs(y).max() <= ramp_low_frac * scale:
            continue  # low-amplitude ramp (e.g. dawn/dusk), not worth flagging
        if is_flat and n * freq <= max_plateau_td:
            pct_night = 0.0
            if location_kwargs:
                pct_night = 1 - _is_daylight(s.index[i0:i1], freq, location_kwargs).mean()
            if pct_night <= night_pct_limit:
                continue  # plausible clipping/curtailment plateau

        rows.append(
            dict(
                start=s.index[i0],
                end=s.index[i1 - 1],
                n_points=n,
                kind="flatline" if is_flat else "linear",
                start_value=values[i0],
                end_value=values[i1 - 1],
            )
        )
    return pd.DataFrame(rows, columns=columns)


def _data_resolution(values: np.ndarray, max_decimals: int = 6):
    """Returns the decimal resolution (e.g. 0.001) data was rounded to, or None if full precision."""
    v = values[np.isfinite(values)][:100_000]
    if v.size == 0:
        return None
    for k in range(max_decimals + 1):
        if np.mean(np.abs(v - np.round(v, k)) < 1e-9 + 3e-7 * np.abs(v)) >= 0.99:
            return 10.0**-k
    return None


def _is_daylight(index: pd.DatetimeIndex, freq, location_kwargs: dict) -> np.ndarray:
    """Boolean array (dawn to dusk) for a regular datetime index, using local wall time."""
    start_date = index[0].strftime("%Y-%m-%d")
    end_date = (index[-1].normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    daylight = get_daylight_timestamps(start_date, end_date, freq=freq, **location_kwargs)
    return index.tz_localize(None).isin(daylight)


def clean_raw_data(df, location_kwargs):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Dataframe must have datetime index.")

    df = df.select_dtypes(include="number").copy()
    if df.empty:
        raise ValueError("No numeric columns found in dataset.")

    changes = {}

    def _process_changes(**kwargs):
        changes.update({key: val for key, val in kwargs.items() if val})
        return

    # Remove empty columns
    original_cols = list(df.columns)
    df = df.dropna(axis=1, how="all")
    removed_cols = [c for c in original_cols if c not in df.columns]
    _process_changes(removed_cols=removed_cols)

    negatives = {}
    outliers = {}
    iqr_outliers = {}
    interpolated = {}
    for col in df.columns:
        coltype = _data_group(col)

        # Part 1: Replace negative values with zero
        is_negative = df[col].lt(0)
        if is_negative.any():
            negatives[col] = is_negative.sum()
            df.loc[is_negative, col] = 0.0

        # Part 2: Remove outliers -> primarily for transient spikes
        cstats = df[col].describe(percentiles=[0.95])
        p95_, max_ = cstats.loc["95%"], cstats.loc["max"]
        delta_ = ((max_ - p95_) / p95_) if p95_ > 0 else 0

        skip_outlier_detect = coltype in ("wind", "inverter", "meter") and delta_ < 5
        if not skip_outlier_detect:
            outlier_condition = (
                (df[col].shift(-1) - df[col]).abs().rolling(4, center=True).mean().gt(1000)
            )
            if outlier_condition.any():
                outliers[col] = df.loc[outlier_condition, col].copy()
                df.loc[outlier_condition, col] = np.nan

        # Part 2a: Remove IQR outliers
        is_iqr_outlier = iqr_outlier_condition(df[col], iqr_multiplier=1.5)
        if is_iqr_outlier.any():
            iqr_outliers[col] = df.loc[is_iqr_outlier, col].copy()
            df.loc[is_iqr_outlier, col] = np.nan

        # Part 3: Remove interpolated data
        d1_col, d2_col = f"{col}_d1", f"{col}_d2"
        dfc = df[[col]].copy()
        dfc[d1_col] = dfc[col].shift(-1) - dfc[col]
        dfc[d2_col] = dfc[d1_col].shift(-1) - dfc[d1_col]

        min_val = 25 if coltype not in ("meter", "wind") else 0.025
        min_hours = 2  # flagged event duration threshold
        if coltype in ("meter", "temperature"):
            min_hours += 1
        elif coltype == "inverter" and "_Pad_" in col:  # FL4
            min_hours += 3  # because of clipping
        elif coltype == "wind":
            min_hours += 10

        freq_td = infer_freq(df.index)
        min_periods_threshold = pd.Timedelta(hours=min_hours) / freq_td
        window = 15 if freq_td == pd.Timedelta(minutes=1) else 2
        flag_1 = dfc[d2_col].rolling(window, center=True).mean().round(4)

        conditions = flag_1.eq(0)
        if "ActivePower" in col:
            conditions &= dfc[col].gt(1)
        elif "Meter" in col:
            conditions &= dfc[col].gt(0.001)

        grouped_idx_list = get_continuous_ranges(dfc[conditions], native_freq=freq_td)
        flagged_ranges = [idx for idx in grouped_idx_list if len(idx) >= min_periods_threshold]
        interpolated_ranges = []
        for idx in flagged_ranges:
            timespan = idx[-1] - idx[0]
            dfc_ = dfc[dfc.index.isin(idx)].copy()

            # exceptions - daylight-dependent data
            if coltype in ("inverters", "meter", "irradiance", "temperature"):
                daylight_ = _daylight_filter(dfc_, location_kwargs)
                pct_night = (len(dfc_) - daylight_.sum()) / len(dfc_)
                pct_night_limit = 0.7

                df_dark, df_light = dfc_[~daylight_].copy(), dfc_[daylight_].copy()

                if pct_night > pct_night_limit:  # nighttime groups

                    # exception - temperature data
                    if "_Temp" in col:
                        continue

                    # exception - nighttime flatline near/at zero
                    if (df_dark[col].mean() < 5) and (df_dark[col].max() < 5):
                        continue

                elif pct_night > 0.5 and timespan < pd.Timedelta(hours=8):  # mostly-night

                    # exception - nighttime flatline near/at zero
                    if (df_dark[col].mean() < 5) and (df_dark[col].max() < min_val):
                        continue

                elif pct_night < 0.05:  # daylight groups

                    # exception - ac module outage condition
                    if df_light[col].mean() > min_val:
                        continue

                    # exception - originally intended for GHI/POA (semi-linear trend)
                    if timespan < pd.Timedelta(hours=2.5) and dfc_[d1_col].abs().mean() > 1:
                        continue

            # exception - meter outage
            if coltype == "meter" and dfc_[col].mean().round(1) == 0.0:
                continue

            # if no exceptions hit, add index/range to list for data removal
            interpolated_ranges.append(idx)

        if interpolated_ranges:
            remove_index = pd.concat([x.to_frame() for x in interpolated_ranges]).index
            df.loc[df.index.isin(remove_index), col] = np.nan
            interpolated[col] = pd.DataFrame(
                {
                    "start": [x[0] for x in interpolated_ranges],
                    "end": [x[-1] for x in interpolated_ranges],
                }
            )

    _process_changes(
        negatives=negatives, outliers=outliers, iqr_outliers=iqr_outliers, interpolated=interpolated
    )
    return df, changes


DAYLIGHT_TYPES = ("inverter", "meter", "irradiance")  # solar-driven; zero at night
INTERPOLATION_SETTINGS = {  # per data group overrides for flag_interpolated_data
    "meter": dict(min_duration="3h"),
    "temperature": dict(min_duration="3h", max_plateau="12h"),
    "wind": dict(min_duration="12h", max_plateau="24h"),
}
NEGATIVE_POLICY = {  # keep | clip (all negatives -> 0) | clip_small (small -> 0, large -> NaN)
    "meter": "clip",
    "inverter": "clip_small",
    "irradiance": "clip_small",
    "wind": "clip_small",
    "temperature": "keep",  # negative values are legitimate
    "unknown": "keep",
}
IQR_MULTIPLIER = {"temperature": 3.0, "wind": 3.0, "unknown": 3.0}  # default 1.5 otherwise
DEFAULT_STEPS = [
    "drop_empty_columns",
    "flag_interpolated",
    "handle_negatives",
    "remove_spikes",
    "remove_outliers",
]


class QCDataset(Dataset):
    """Raw time series data -> cleaned time series data, with a record of every change.

    Each cleaning step sets bad values to NaN (never silently edits good values), stores the removed
    values in ``removed(col)``, and logs counts in ``summary()``. Steps return ``self`` for chaining:

        qc = QCDataset(df, location_kwargs=dict(tz=..., lat=..., lon=...))
        qc.run()                                   # default pipeline
        qc.flag_interpolated().handle_negatives()  # or step-by-step
        qc.data      # cleaned data      qc.raw  # original data      qc.summary()  # change log

    Parameters
    ----------
    dataframe : pd.DataFrame
        Raw data with a DatetimeIndex; only numeric columns are used.
    location_kwargs : dict, optional
        Keys for ``get_daylight_timestamps`` (tz, lat, lon, ...). Enables daylight-aware logic.
    column_types : dict, optional
        Overrides the data group inferred from column names (see ``_data_group``).
    is_solar : bool
        Set False for wind/gas sites so daylight logic is never applied.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        location_kwargs: dict = None,
        column_types: dict = None,
        is_solar: bool = True,
    ):
        super().__init__()
        if not isinstance(dataframe.index, pd.DatetimeIndex):
            raise TypeError("Dataframe must have a datetime index.")
        data = dataframe.select_dtypes(include="number")
        if data.empty:
            raise ValueError("No numeric columns found in dataset.")

        self.raw = data.copy()
        self.data = data.copy()
        self.location_kwargs = location_kwargs
        self.is_solar = bool(is_solar and location_kwargs)
        overrides = column_types or {}
        self.column_types = {c: overrides.get(c) or _data_group(c) for c in data.columns}
        self.interpolated = {}  # col -> DataFrame of flagged ranges
        self._log = []
        self._removed = {}
        self._daylight = None
        self.regularize_index()

    @property
    def columns(self):
        data = getattr(self, "data", None)
        return [] if data is None else list(data.columns)

    # ---- helpers --------------------------------------------------------------------------
    def _uses_daylight(self, col) -> bool:
        return self.is_solar and self.column_types[col] in DAYLIGHT_TYPES

    def daylight(self) -> pd.Series:
        """Boolean series (dawn to dusk, local wall time) aligned to the data index."""
        if self._daylight is None or not self._daylight.index.equals(self.data.index):
            mask = _is_daylight(self.data.index, self.freq_timedelta, self.location_kwargs)
            self._daylight = pd.Series(mask, index=self.data.index)
        return self._daylight

    def _scale(self, col) -> float:
        v = self.data[col].abs().quantile(0.99)
        return v if pd.notna(v) and v > 0 else 1.0

    def _columns(self, columns):
        return self.columns if columns is None else list(columns)

    def _log_step(self, step, col, n, detail=""):
        self._log.append(dict(step=step, column=col, n_points=int(n), detail=detail))

    def _remove(self, step, col, mask, detail="") -> int:
        """Sets masked (non-NaN) points of ``col`` to NaN, recording their original values."""
        mask = pd.Series(np.asarray(mask, dtype=bool), index=self.data.index)
        mask &= self.data[col].notna()
        n = int(mask.sum())
        if n:
            removed = pd.DataFrame({"value": self.data.loc[mask, col], "step": step})
            self._removed[col] = pd.concat([self._removed.get(col, removed.iloc[:0]), removed])
            self.data.loc[mask, col] = np.nan
        self._log_step(step, col, n, detail)
        return n

    def removed(self, col) -> pd.DataFrame:
        """Original values removed from ``col`` (index = timestamp; columns = value, step)."""
        return self._removed.get(col, pd.DataFrame(columns=["value", "step"])).sort_index()

    def summary(self) -> pd.DataFrame:
        """Points changed per step (rows) and column (columns)."""
        if not self._log:
            return pd.DataFrame()
        log = pd.DataFrame(self._log)
        table = log.pivot_table(
            index="step", columns="column", values="n_points", aggfunc="sum", sort=False
        )
        return table.fillna(0).astype(int)

    # ---- steps: structure -----------------------------------------------------------------
    def regularize_index(self):
        """Sorts, drops duplicate timestamps (keeps first) and fills missing ones with NaN."""
        d = self.data
        dup = d.index.duplicated()
        if dup.any():
            self._log_step("duplicate_timestamps_dropped", "*", dup.sum())
            d = d[~dup]
        d = d.sort_index()
        freq = infer_freq(d.index)
        full = pd.date_range(d.index[0], d.index[-1], freq=freq)
        if not d.index.isin(full).all():
            raise ValueError("Timestamps are not aligned to a regular grid at the inferred interval.")
        if len(full) > len(d):
            self._log_step("missing_timestamps_filled", "*", len(full) - len(d))
        self.data = d.reindex(full)
        self.freq_timedelta = freq
        return self

    def drop_empty_columns(self):
        empty = [c for c in self.columns if self.data[c].isna().all()]
        for col in empty:
            self._log_step("empty_column_dropped", col, 0)
        self.data = self.data.drop(columns=empty)
        return self

    # ---- steps: cleaning ------------------------------------------------------------------
    def flag_interpolated(self, columns=None, **kwargs):
        """Removes interpolated (linear) and held (flatline) ranges; see flag_interpolated_data.
        Should run first: later steps that insert NaNs would break the runs this step detects."""
        for col in self._columns(columns):
            ctype = self.column_types[col]
            loc = self.location_kwargs if self._uses_daylight(col) else None
            opts = {**INTERPOLATION_SETTINGS.get(ctype, {}), **kwargs}
            ranges = flag_interpolated_data(self.data[col], location_kwargs=loc, **opts)
            self.interpolated[col] = ranges
            mask = pd.Series(False, index=self.data.index)
            for r in ranges.itertuples():
                mask.loc[r.start : r.end] = True
            self._remove("interpolated", col, mask, f"{len(ranges)} range(s)")
        return self

    def handle_negatives(self, neg_tol_frac: float = 0.05, policy: dict = None, columns=None):
        """Negative values by data group (see NEGATIVE_POLICY): temperature is left alone; power-like
        data has small negatives (> -neg_tol_frac * scale) clipped to 0 and large ones removed."""
        policy = {**NEGATIVE_POLICY, **(policy or {})}
        for col in self._columns(columns):
            how = policy.get(self.column_types[col], "keep")
            x = self.data[col]
            is_neg = x.lt(0)
            if how == "keep" or not is_neg.any():
                continue
            if how == "clip":
                small, large = is_neg, is_neg & False
            else:  # clip_small
                small = is_neg & x.ge(-neg_tol_frac * self._scale(col))
                large = is_neg & ~small
            self.data.loc[small, col] = 0.0
            self._log_step("negatives_clipped", col, small.sum())
            if large.any():
                self._remove("negatives_removed", col, large)
        return self

    def remove_spikes(self, spike_frac: float = 0.5, columns=None):
        """Removes isolated single-sample spikes: a point differing from BOTH neighbors, in the same
        direction, by more than max(``spike_frac`` * scale, 8 x the column's typical sample-to-sample
        noise) -- so naturally noisy columns (e.g. wind speed) need a much larger jump."""
        for col in self._columns(columns):
            x = self.data[col]
            up, down = x - x.shift(1), x - x.shift(-1)
            noise = 1.4826 * up.abs().median()
            thr = max(spike_frac * self._scale(col), 8 * noise)
            is_spike = (up.abs().gt(thr) & down.abs().gt(thr)) & (np.sign(up) == np.sign(down))
            self._remove("spikes", col, is_spike)
        return self

    def remove_outliers(
        self,
        method: str = "auto",
        iqr_multiplier: float = None,
        columns=None,
        hampel_k: float = 8.0,
        hampel_window="1h",
        hampel_min_frac: float = 0.05,
    ):
        """Outlier removal. Methods:
        - "auto" (default): "hampel" for solar-driven columns, "iqr" for the rest.
        - "hampel": point deviates from the rolling median by > hampel_k * scaled MAD AND by more
          than ``hampel_min_frac`` * scale (so tiny wiggles on clear days are never flagged).
          Daylight points only for solar-driven columns. Local, so real peaks are kept.
        - "iqr": global fences (multiplier IQR_MULTIPLIER, default 1.5; 3.0 for temperature/wind).
          Only appropriate for roughly stationary data. NOT for solar power: its daytime
          distribution is skewed by every morning/evening/cloudy sample, so fences land below real
          peak production (it removed 548 genuine near-nameplate points in the Maricopa sample).
        """
        if method not in ("auto", "iqr", "hampel"):
            raise ValueError("method must be 'auto', 'iqr' or 'hampel'.")
        for col in self._columns(columns):
            x = self.data[col]
            solar = self._uses_daylight(col)
            day = self.daylight() if solar else pd.Series(True, index=x.index)
            how = ("hampel" if solar else "iqr") if method == "auto" else method
            if how == "iqr":
                mult = iqr_multiplier or IQR_MULTIPLIER.get(self.column_types[col], 1.5)
                lo, hi = get_iqr_bounds(x[day], mult)
                is_outlier = day & (x.lt(lo) | x.gt(hi))
                detail = f"fences [{lo:.4g}, {hi:.4g}]"
            else:
                win = max(5, int(pd.Timedelta(hampel_window) / self.freq_timedelta) // 2 * 2 + 1)
                med = x.rolling(win, center=True, min_periods=win // 2).median()
                mad = (x - med).abs().rolling(win, center=True, min_periods=win // 2).median()
                thr = np.maximum(hampel_k * 1.4826 * mad, hampel_min_frac * self._scale(col))
                is_outlier = day & ((x - med).abs() > thr)
                detail = f"window={win}, k={hampel_k}"
            self._remove(f"outliers_{how}", col, is_outlier, detail)
        return self

    def flag_out_of_range(self, limits: dict, columns=None):
        """Removes values outside physical limits. ``limits`` maps a column name or data group to
        (low, high), e.g. {"irradiance": (0, 1500), "temperature": (-50, 60)}; None = no bound.
        No defaults because units (W/m2 vs kW/m2, C vs F) are not known here."""
        for col in self._columns(columns):
            bounds = limits.get(col, limits.get(self.column_types[col]))
            if bounds is None:
                continue
            lo, hi = bounds
            bad = pd.Series(False, index=self.data.index)
            if lo is not None:
                bad |= self.data[col].lt(lo)
            if hi is not None:
                bad |= self.data[col].gt(hi)
            self._remove("out_of_range", col, bad, f"limits={bounds}")
        return self

    def flag_nighttime_values(self, frac: float = 0.02, columns=None):
        """Removes irradiance/inverter values above ``frac`` * scale outside dawn-dusk (opt-in)."""
        for col in self._columns(columns):
            if not self._uses_daylight(col) or self.column_types[col] == "meter":
                continue
            bad = ~self.daylight() & self.data[col].gt(frac * self._scale(col))
            self._remove("nighttime_values", col, bad)
        return self

    def fill_gaps(self, max_gap="1h", columns=None):
        """Time-interpolates NaN gaps up to ``max_gap`` (opt-in; filled points are logged, and are
        interpolated data -- run flag_interpolated again if filled data should not be trusted)."""
        limit = int(pd.Timedelta(max_gap) / self.freq_timedelta)
        for col in self._columns(columns):
            was_na = self.data[col].isna()
            filled = self.data[col].interpolate(method="time", limit=limit, limit_area="inside")
            self.data[col] = filled
            self._log_step("gaps_filled", col, (was_na & filled.notna()).sum())
        return self

    def run(self, steps=None):
        """Runs steps in order. ``steps`` = list of method names or (name, kwargs) tuples."""
        for step in steps or DEFAULT_STEPS:
            name, kwargs = (step, {}) if isinstance(step, str) else step
            getattr(self, name)(**kwargs)
        return self
