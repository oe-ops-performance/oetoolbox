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
    freq_mins = int(series.index.diff.max().total_seconds() / 60)
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
    elif contains("ambient", "temp"):
        return "temperature"
    return "unknown"


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


class QCDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        self.data = dataframe.select_dtypes(include="number").copy()
        self.columns = list(self.data.columns)
        self._validate_dataframe()  # assigns other properties/attributes

    def _validate_dataframe(self) -> None:
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise TypeError("Dataframe must have a datetime index.")

        # validate index (use smallest detected frequency/interval)
        freq_timedelta = self.data.index.to_series().diff().min()
        start_datetime = self.data.index.min()
        end_datetime = self.data.index.max()
        continuous_index = pd.date_range(start_datetime, end_datetime, freq=freq_timedelta)
        if len(self.data.index) != len(continuous_index):
            raise ValueError("Dataframe index must be continuous; found missing timestamp(s).")

        self.start = start_datetime
        self.end = end_datetime
        self.freq_timedelta = freq_timedelta

    def remove_outliers(
        self, method: str, iqr_multiplier=1.5, z_threshold=3
    ) -> dict[str, pd.DataFrame]:
        """Removes outliers using either z-score or interquartile range method.

        Parameters
        ----------
        method : str
            Outlier detection method; either "z_score" or "iqr".

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary of removed outliers by column. Updates self.data
        """
        kwargs = dict(method=method)
        if method == "iqr":
            kwargs.update(dict(iqr_multiplier=iqr_multiplier))
        else:
            kwargs.update(dict(z_threshold=z_threshold))

        outlier_dict = {}
        series_list = []
        for col in self.columns:
            series = self.data[col].copy()
            outliers = remove_outliers(series, **kwargs)
            series_list.append(series)
            if outliers.empty:
                continue
            outlier_dict.update({col: outliers})

        self.data = pd.concat(series_list, axis=1)
        return outlier_dict
