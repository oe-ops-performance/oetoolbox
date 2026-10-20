import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pvanalytics.quality.irradiance as iqc
import pvanalytics.features.daytime as daytime
from pvanalytics.features.orientation import fixed_nrel
from sklearn.metrics import r2_score

from .pvlib import _get_aligned_clearsky_data
from .qc import get_sun
from ..utils import oemeta, oeplots
from ..utils.assets import SolarSite
from ..utils.datetime import infer_freq, upsample


def _get_continuous_ranges(df_segmented: pd.DataFrame, native_freq: pd.Timedelta):
    if type(df_segmented.index) is not pd.DatetimeIndex:
        raise TypeError("Dataframe must have a datetime index.")
    if df_segmented.empty:
        return []
    df = df_segmented.copy()
    df["time_diff"] = df.index.diff()
    if len(df) > 1:
        df.iloc[0, -1] = df.iloc[1, -1]

    step = native_freq
    conditions = df.time_diff.eq(step)
    breaks = df.time_diff.gt(step).cumsum()

    return [x.index for _, x in df[conditions].groupby(breaks)]


def _process_poa_sensor_data(
    poa_measured: pd.Series,
    poa_clearsky: pd.Series,
    apparent_zenith: pd.Series,
    solar_noon: pd.Series = None,
    has_trackers: bool = True,
    r2_min: float = None,  # default 0.94 if hourly, 0.90 if minute-level
):
    # --- Context Assumptions ---
    # df['poa_measured'] : Measured single-axis tracker POA irradiance
    # df['poa_clearsky']  : Transposed clear-sky POA matched to tracker angles

    # 1. Establish the statistical daytime mask to drop nighttime data
    daytime_mask = daytime.power_or_irradiance(poa_clearsky)

    # 2. Define the core daylight period (exclude shoulder periods)
    # Drop anything within 5 degrees of the horizon or very low clear-sky signals
    max_zenith = 85 if has_trackers else 82
    core_daylight_mask = (apparent_zenith <= max_zenith) & (poa_clearsky > 50)

    # 3. Compute instantaneous checks (only strictly enforced during core daylight)
    valid_instantaneous = iqc.clearsky_limits(
        measured=poa_measured,
        clearsky=poa_clearsky,
        csi_max=1.15,
    )

    # 4. Compute daily insolation limits (using tracker-specific upper bounds)
    # Note: daily_insolation_limits handles internal integration and applies back to timestamps
    daily_max = 1.35 if has_trackers else 1.25
    valid_daily = iqc.daily_insolation_limits(
        irrad=poa_measured,
        clearsky=poa_clearsky,
        daily_min=0.40,
        daily_max=daily_max,
    )

    # 5. Combine masks logically
    # Instantaneous validation is only forced when the sun is high enough.
    # Daily validation and general day/night separation apply across the whole set.
    final_qc_mask = daytime_mask & valid_daily

    # Where it is core daylight, it must also pass the instantaneous clear-sky limit check
    final_qc_mask = final_qc_mask & (~core_daylight_mask | valid_instantaneous)

    is_bad = ~final_qc_mask & daytime_mask

    df = (
        poa_clearsky.to_frame()
        .join(
            poa_measured.to_frame(
                name="poa_measured" if not poa_measured.name else poa_measured.name
            )
        )
        .copy()
    )
    df["valid_instantaneous"] = valid_instantaneous.astype(int)
    df["valid_daily"] = valid_daily.astype(int)
    df["final_qc_mask"] = final_qc_mask.astype(int)
    df["is_bad"] = is_bad.astype(int)

    df_bad = df.loc[is_bad].copy()
    if df_bad.empty:
        bad_ranges, ranges_to_remove = [], []
    else:
        bad_ranges = _get_continuous_ranges(df_bad, native_freq=infer_freq(df.index))
        ranges_to_remove = [idx for idx in bad_ranges if (idx[-1] - idx[0]) > pd.Timedelta(hours=2)]

    # 6. Create mask for detecting stowed trackers by half day
    is_stowed = poa_clearsky.index.isna()  # default: False
    stowed_ranges = []
    if has_trackers is True:
        if solar_noon is None:
            raise ValueError("Must provide solar noon reference for stowed tracker detection.")
        is_stowed = analyze_half_days(
            poa_measured=poa_measured, solar_noon=solar_noon, r2_min=r2_min
        )
        df_stow = df.loc[is_stowed].copy()
        if not df_stow.empty:
            stowed_ranges = _get_continuous_ranges(df_stow, native_freq=infer_freq(df.index))
        df["is_stowed"] = is_stowed.astype(int)

    return dict(
        df=df, bad_ranges=bad_ranges, ranges_to_remove=ranges_to_remove, stowed_ranges=stowed_ranges
    )


def _validate_poa_data_input(df: pd.DataFrame):
    refcols = ["poa_clearsky", "apparent_zenith", "azimuth", "zenith"]
    if not all(c in df.columns for c in refcols):
        raise ValueError(f"Missing one or more required columns -> {refcols}")
    poa_cols = list(df.filter(like="OE.POA_").columns)
    if not poa_cols:
        raise ValueError("No 'OE.POA' sensor columns found.")
    return poa_cols


def _get_solar_noon(dates, location_kwargs):
    tstamps = {date: get_sun(date, **location_kwargs)["noon"] for date in dates}
    return pd.Series(index=tstamps.keys(), data=tstamps.values())


def process_poa_data(
    df: pd.DataFrame, location_kwargs: dict, has_trackers: bool, r2_min: float = None
):
    poa_cols = _validate_poa_data_input(df)
    if infer_freq(df.index) < pd.Timedelta(hours=1):
        df = df.resample("1h").mean()
    solar_noon = _get_solar_noon(dates=sorted(set(df.index.date)), location_kwargs=location_kwargs)
    results = {}
    for col in poa_cols:
        results[col] = _process_poa_sensor_data(
            poa_measured=df[col].fillna(0.0),
            poa_clearsky=df["poa_clearsky"],
            apparent_zenith=df["apparent_zenith"],
            solar_noon=solar_noon,
            has_trackers=has_trackers,  # oemeta.data["TrackerCount"].get(site, 0) > 0
            r2_min=r2_min,
        )
    return results


def _plot_bad_data(df, qc_results):
    # qc_results is the output of 'process_poa_data', and df is the same input

    kwargs = dict(row_col_list=[[c] for c in qc_results.keys()])
    if len(qc_results) == 2:
        kwargs.update(dict(row_height_list=[0.5, 0.5]))
    fig = oeplots.ts_subplots(df, **kwargs, fig_height=500)
    for r in range(len(qc_results)):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["poa_clearsky"],
                name="poa_clearsky",
                line_color="#777777",
                legendgroup="reference",
                showlegend=(r == 0),
            ),
            row=(r + 1),
            col=1,
        )
    fig.update_traces(zorder=-1, selector=dict(name="poa_clearsky"))

    vkwargs = dict(type="rect", y0=0, y1=1, line_width=0, opacity=0.5, layer="below")
    shapes = []
    for row, dict_ in enumerate(qc_results.values(), start=1):
        yref = f"y{row} domain".replace("y1", "y")
        for idx in dict_["bad_r2_ranges"]:
            shapes.append(
                dict(x0=idx[0], x1=idx[-1], **vkwargs, yref=yref, fillcolor="navajowhite")
            )
        for idx in dict_["ranges_to_remove"]:
            shapes.append(
                dict(x0=idx[0], x1=idx[-1], **vkwargs, yref=yref, fillcolor="lightsalmon")
            )
        for idx in dict_["stowed_ranges"]:
            shapes.append(dict(x0=idx[0], x1=idx[-1], **vkwargs, yref=yref, fillcolor="turquoise"))
    fig.update_layout(shapes=shapes)
    return fig


def _generate_flag_column(
    ref_index: pd.DatetimeIndex, col: str, ranges_by_flag: dict[str, list[pd.DatetimeIndex]]
):
    df = pd.DataFrame(index=ref_index)
    df[f"{col}_FLAG"] = ""
    for flag, ranges in ranges_by_flag.items():
        if len(ranges) == 0:
            continue
        index_to_remove = pd.concat([idx.to_frame() for idx in ranges]).index
        condition = df.index.isin(index_to_remove)
        df.loc[condition, f"{col}_FLAG"] = flag
    return df[f"{col}_FLAG"]


def _add_flag_columns(df_, qc_results):
    df = df_.copy()
    for col, results in qc_results.items():
        df[f"{col}_FLAG"] = ""

        # bad data -> outside of normal irradiance range (and longer than 2 hours -> 1h data)
        flag = "bad" if "bad_ranges" in results else "ref_r2"
        ranges_to_remove = results["ranges_to_remove"]
        if ranges_to_remove:
            index_to_remove = pd.concat([idx.to_frame() for idx in ranges_to_remove]).index
            is_bad = df.index.isin(index_to_remove)
            df.loc[is_bad, f"{col}_FLAG"] = flag

        if "is_stowed" in results["df"].columns:
            is_stowed = results["df"]["is_stowed"].eq(1)
            df.loc[is_stowed, f"{col}_FLAG"] = "stow"

    return df


def _remove_flagged_ranges(df_native, df_flags):
    flag_cols = [c for c in df_flags.columns if c.endswith("_FLAG")]
    data_cols = [c.replace("_FLAG", "") for c in flag_cols]

    freq_td = infer_freq(df_native.index)
    if infer_freq(df_flags.index) > freq_td:
        df_flags = upsample(df_flags[flag_cols], new_freq=freq_td, agg_type="ffill")
    else:
        df_flags = df_flags[flag_cols].copy()

    df = df_native[data_cols].copy()
    removed_ranges_by_col = {}
    for col, flag in zip(data_cols, flag_cols):
        is_flagged = df_flags[flag].ne("")
        df.loc[is_flagged, col] = np.nan
        removed_ranges_by_col[col] = _get_continuous_ranges(df[is_flagged], freq_td)
    return df, removed_ranges_by_col


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


def analyze_half_days(poa_measured, solar_noon, r2_min=0.94):
    # Initialize a mask for bad data (True = BAD (remove), False = GOOD (keep))
    qc_mask = pd.Series(False, index=poa_measured.index)

    if r2_min is None:
        r2_min = 0.94 if infer_freq(poa_measured.index) == pd.Timedelta(hours=1) else 0.90

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
        is_fixed_tilt_am = fixed_nrel(synthetic_morning_day, daytime_mask, r2_min=r2_min)
        if is_fixed_tilt_am.all():
            qc_mask.loc[morning.index] = True

        # --- Evaluate Afternoon (Mirror the afternoon curve into the morning) ---
        synthetic_afternoon_day = mirrored_data["mirrored_part2"].copy()

        daytime_mask = daytime.power_or_irradiance(synthetic_afternoon_day)
        is_fixed_tilt_pm = fixed_nrel(synthetic_afternoon_day, daytime_mask, r2_min=r2_min)
        if is_fixed_tilt_pm.all():
            qc_mask.loc[afternoon.index] = True

    return qc_mask


def r2_by_half_day(
    df: pd.DataFrame, data_col: str, ref_col: str, solar_noon: pd.Series, has_trackers: bool
) -> pd.DataFrame:
    freq_td = infer_freq(df.index)

    # 1. Establish the statistical daytime mask to drop nighttime data
    daytime_mask = daytime.power_or_irradiance(df[ref_col])

    # 2. Define the core daylight period (exclude shoulder periods)
    # Drop anything within 5 degrees of the horizon or very low clear-sky signals
    max_zenith = 85 if has_trackers else 82
    min_val = 0.05 if ref_col == "site_meter" else 25  # normalized data

    data_list = []  # cols: date, start (0/noon), period (am/pm), r2_score
    for date, day_data in df.groupby(df.index.date):
        noon_ = solar_noon.loc[date]
        noon_str = noon_.strftime("%Y-%m-%d %H:%M")  # prevent tz mismatch errors
        core_daylight_mask = (day_data["apparent_zenith"] <= max_zenith) & (
            day_data[ref_col] > min_val
        )
        daylight_mask = daytime.power_or_irradiance(day_data[ref_col])

        morning = day_data[daylight_mask & (day_data.index < noon_str)]
        morning = morning[morning[[data_col, ref_col]].notna().all(axis=1)]
        if morning.empty:
            masd, r2 = 1000.0, None
        else:
            masd = morning[data_col].diff().abs().mean()
            r2 = r2_score(y_true=morning[ref_col].values, y_pred=morning[data_col].values)
        am_start = pd.Timestamp(date)
        data_list.append([date, am_start, "am", r2, masd])

        afternoon = day_data[daylight_mask & (day_data.index >= noon_str)]
        afternoon = afternoon[afternoon[[data_col, ref_col]].notna().all(axis=1)]
        if afternoon.empty:
            masd, r2 = 1000.0, None
        else:
            masd = afternoon[data_col].diff().abs().mean()
            r2 = r2_score(y_true=afternoon[ref_col].values, y_pred=afternoon[data_col].values)
        pm_start = noon_.round(freq_td)
        data_list.append([date, pm_start, "pm", r2, masd])

    df = pd.DataFrame(data_list, columns=["date", "start", "period", "r2_score", "MASD"])
    df["date"] = pd.to_datetime(df["date"])
    df["start"] = pd.to_datetime(df["start"])
    return df.set_index("date")


def _formatted_r2_series(
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


def generate_r2_flags(
    series: pd.Series,
    dtn_series: pd.Series,
    apparent_zenith: pd.Series,
    location_kwargs: dict,
    has_trackers: bool,
    meter_data: pd.Series = None,
    clearsky_series: pd.Series = None,
):
    data_col = series.name if series.name else "data"
    ref_col = dtn_series.name if dtn_series.name else "reference"
    solar_noon = _get_solar_noon(
        dates=sorted(set(series.index.date)), location_kwargs=location_kwargs
    )
    df = series.to_frame(name=data_col).copy()
    df[ref_col] = dtn_series.values
    df["apparent_zenith"] = apparent_zenith.values
    if meter_data is not None:
        df["site_meter"] = meter_data.values

    if clearsky_series is not None:
        df["ref_clearsky"] = clearsky_series.values

    if (series.isna().sum() / len(series)) > 0.95:
        df[f"{data_col}_r2_dtn"] = 0.0
        return

    kwargs = dict(data_col=data_col, solar_noon=solar_noon, has_trackers=has_trackers)
    df_r2 = r2_by_half_day(df=df, ref_col=ref_col, **kwargs)
    df[f"{data_col}_r2_dtn"] = _formatted_r2_series(df_r2=df_r2, ref_index=df.index)

    if "site_meter" in df.columns:
        dfn = df[[data_col, "site_meter"]].copy()
        dfn = (dfn - dfn.min()) / (dfn.max() - dfn.min())  # normalized: 0 to 1
        dfn["apparent_zenith"] = df["apparent_zenith"].values
        df_r2_norm = r2_by_half_day(df=dfn, ref_col="site_meter", **kwargs)
        df[f"{data_col}_r2_meter"] = _formatted_r2_series(df_r2=df_r2_norm, ref_index=df.index)

    if "ref_clearsky" in df.columns:
        df_r2_cs = r2_by_half_day(df=df, ref_col="ref_clearsky", **kwargs)
        df[f"{data_col}_r2_clearsky"] = _formatted_r2_series(df_r2=df_r2_cs, ref_index=df.index)

    df[f"{data_col}_MASD"] = _formatted_r2_series(df_r2=df_r2, ref_index=df.index, stat_col="MASD")
    return df


def process_poa_data_via_r2(
    df: pd.DataFrame, location_kwargs: dict, has_trackers: bool, r2_min: float = 0.75
):
    poa_cols = list(df.filter(like="OE.POA_").columns)
    if not poa_cols:
        raise ValueError("No POA sensor columns found.")
    if not all(c in df.columns for c in ["apparent_zenith", "poa_global"]):
        raise KeyError("Missing one or more required columns.")
    kwargs = dict(
        apparent_zenith=df["apparent_zenith"],
        location_kwargs=location_kwargs,
        has_trackers=has_trackers,
    )
    if "site_meter" in df.columns:
        kwargs.update(dict(meter_data=df["site_meter"]))
    if "poa_clearsky" in df.columns:
        kwargs.update(dict(clearsky_series=df["poa_clearsky"]))

    results = {}
    for col in poa_cols:
        df_flags = generate_r2_flags(series=df[col], dtn_series=df["poa_global"], **kwargs)

        # conditions
        has_no_data = df_flags[col].isna()
        off_from_dtn = df_flags[f"{col}_r2_dtn"].lt(r2_min)
        off_from_clearsky = df_flags[f"{col}_r2_clearsky"].lt(r2_min)
        if f"{col}_r2_meter" in df_flags.columns:
            similar_to_meter = df_flags[f"{col}_r2_meter"].gt(r2_min)
            cloudy_conditions = df_flags[f"{col}_MASD"].gt(120.0)  # threshold for hourly data
            close_enough_to_keep = similar_to_meter & cloudy_conditions
        else:
            close_enough_to_keep = df_flags.index.notna()  # should be False everywhere

        bfill_condition = has_no_data | (off_from_dtn & ~close_enough_to_keep)
        bfill_condition &= ~off_from_clearsky

        df_bfill = df.loc[bfill_condition].copy()
        bfill_ranges = _get_continuous_ranges(df_bfill, native_freq=infer_freq(df.index))
        ranges_to_remove = [idx for idx in bfill_ranges if len(idx) >= 2]
        results[col] = dict(df=df_flags, ranges_to_remove=ranges_to_remove)

    return results


class POADataset:
    def __init__(
        self,
        site: str,
        dataframe: pd.DataFrame,
        ref_poa: pd.Series,
        ref_meter: pd.Series = None,
        r2_min_ref: float = 0.75,
        r2_min_stow: float = None,  # default 0.94 for hourly & 0.90 for minute-level data
        analysis_freq: str = "1h",
    ):
        self.site = SolarSite(site)
        self.r2_min_ref = r2_min_ref
        self.r2_min_stow = r2_min_stow
        self.has_trackers = oemeta.data["TrackerCount"].get(site, 0) > 0
        if analysis_freq not in ("1h", "1min"):
            raise ValueError("Unsupported frequency.")
        self.analysis_freq = pd.Timedelta(analysis_freq)

        self.qc_changes = {}  # init
        self._process_input_data(df=dataframe, ref_poa=ref_poa, ref_meter=ref_meter)

        self._run_poa_qc()

        self.df_clean = None
        self.df_final = None
        self.removed_ranges = {}

    def _process_input_data(self, df, ref_poa, ref_meter):
        if ref_poa.name != "poa_global":
            raise ValueError("Unexpected column name in reference POA data.")
        poa_cols_ = list(df.filter(like="OE.POA_").columns)
        if not poa_cols_:
            raise ValueError("No 'OE.POA' sensor columns found.")
        poa_cols = list(df.filter(like="OE.POA_").dropna(axis=1, how="all").columns)
        if not poa_cols:
            raise ValueError("No valid data found for 'OE.POA' sensor column(s).")

        self.native_freq = infer_freq(df.index)
        self.df_native = self._clean_native_data(df[poa_cols])
        if self.df_native.empty:
            raise ValueError(
                f"No valid data found for 'OE.POA' sensor column(s).\n{self.qc_changes}"
            )
        self.poa_columns = list(self.df_native.columns)

        # create dataframe for analysis
        if self.analysis_freq == self.native_freq:
            self.df = self.df_native.copy()
        elif self.analysis_freq > self.native_freq:
            self.df = self.df_native.resample(self.analysis_freq).mean().copy()
        else:
            raise ValueError("Unsupported frequency.")

        # create hourly dataframe for analysis
        # if self.freq_td == pd.Timedelta(hours=1):
        #     self.df = self.df_native.copy()
        # elif self.freq_td < pd.Timedelta(hours=1):
        #     self.df = self.df_native.resample("1h").mean().copy()
        # else:
        #     raise ValueError("Unsupported frequency.")

        # add supporting data to df
        poa_freq = infer_freq(ref_poa.index)
        if poa_freq == self.analysis_freq:
            self.df["poa_global"] = ref_poa.values
        elif poa_freq > self.analysis_freq:
            df_poa = upsample(ref_poa.to_frame(), new_freq=self.analysis_freq)
            self.df["poa_global"] = df_poa.iloc[:, 0].values
        else:
            raise ValueError("Invalid frequency for provided external poa data.")

        if ref_meter is not None:
            meter_freq = infer_freq(ref_meter.index)
            if meter_freq == self.analysis_freq:
                self.df["site_meter"] = ref_meter.values
            elif meter_freq < self.analysis_freq:
                self.df["site_meter"] = ref_meter.resample("1h").mean().values
            else:
                raise ValueError("Invalid frequency for provided meter data.")
            self.df["site_meter"] = self.df["site_meter"].clip(lower=0.0)

        if "poa_clearsky" not in self.df.columns:
            df_clearsky = _get_aligned_clearsky_data(
                self.site.name, self.df.index, include_solar_position=True
            )
            self.df = self.df.join(df_clearsky)

    def _clean_native_data(self, df_native):
        df = df_native.dropna(axis=1, how="all").copy()
        empty_cols = [c for c in df_native.columns if c not in df.columns]
        zeroed_cols = []
        for col in df.columns:
            if df[col].sum() < 1e-5:
                zeroed_cols.append(col)
        remove_cols = empty_cols + zeroed_cols
        if remove_cols:
            df = df.drop(columns=remove_cols)
            rc_ = {k: v for k, v in zip(["all_na", "all_zero"], [empty_cols, zeroed_cols]) if v}
            self.qc_changes.update({"removed_columns": rc_})

        n_negative_by_col = df.lt(0).sum().to_dict()
        if sum(n_negative_by_col.values()) > 0:
            df[df.lt(0)] = 0.0
            self.qc_changes.update({"negative_values": n_negative_by_col})

        # remove extended ranges of zero values (flatlined at zero)
        flatlined_at_zero_ranges = {}
        for col in df.columns:
            df_zero = df[df[col].eq(0)].copy()
            zero_ranges = _get_continuous_ranges(df_zero, self.native_freq)
            flatlined_ranges = [x for x in zero_ranges if (x[-1] - x[0]) > pd.Timedelta(days=1)]
            if flatlined_ranges:
                remove_index = pd.concat([x.to_frame() for x in flatlined_ranges]).index
                df.loc[df.index.isin(remove_index), col] = np.nan
                flatlined_at_zero_ranges.update({col: flatlined_ranges})
        if flatlined_at_zero_ranges:
            self.qc_changes.update({"zero_ranges": flatlined_at_zero_ranges})

        return df

    def _run_poa_qc(self):
        kwargs = dict(
            location_kwargs=self.site.location_kwargs,
            has_trackers=self.has_trackers,
        )
        results_1 = process_poa_data(df=self.df, r2_min=self.r2_min_stow, **kwargs)
        results_2 = process_poa_data_via_r2(df=self.df, r2_min=self.r2_min_ref, **kwargs)
        self.results = {
            col: {
                "df1": dict_["df"],
                "bad_data_ranges": dict_["ranges_to_remove"],
                "stowed_ranges": dict_["stowed_ranges"],
                "df2": results_2[col]["df"],
                "bad_r2_ranges": results_2[col]["ranges_to_remove"],
            }
            for col, dict_ in results_1.items()
        }
        self._combine_flagged_ranges()
        self._generate_df_flags()

    def _combine_flagged_ranges(self):
        flagged_ranges = {}
        for col, dict_ in self.results.items():
            all_flagged_ranges_list = [
                dict_["bad_data_ranges"] + dict_["bad_r2_ranges"] + dict_["stowed_ranges"]
            ]
            all_ranges = [rng_ for list_ in all_flagged_ranges_list for rng_ in list_]
            idx_list = []
            for idx in all_ranges:
                idx_freq = infer_freq(idx)
                if idx_freq > self.analysis_freq:
                    idx2 = pd.date_range(idx[0], idx[-1] + idx_freq, freq=idx_freq)
                    idx = (
                        idx.to_frame().reindex(idx2).resample(self.analysis_freq).ffill()[:-1].index
                    )
                idx_list.append(idx)

            df_idx = pd.concat([x.to_frame() for x in all_ranges]).drop_duplicates().sort_index()
            flagged_ranges[col] = _get_continuous_ranges(df_idx, self.native_freq)
        self.flagged_ranges = flagged_ranges

    def _generate_df_flags(self):
        series_list = []
        for col, results in self.results.items():
            ranges_by_flag = {
                "ref_r2": results["bad_r2_ranges"],
                "bad": results["bad_data_ranges"],
                "stow": results["stowed_ranges"],
            }
            flag_data_list = []
            for flag, ranges in ranges_by_flag.items():
                flag_data_ = _generate_flag_column(
                    ref_index=self.df.index, col=col, ranges_by_flag={flag: ranges}
                )
                flag_data_list.append(flag_data_.to_frame(name=flag))
            flag_data = pd.concat(flag_data_list, axis=1).agg(
                lambda row: ", ".join([str(val) for val in row if val != ""]), axis=1
            )
            series_list.append(pd.Series(flag_data, name=f"{col}_FLAG"))
        self.df_flags = pd.concat(series_list, axis=1)

    def _get_flags_for_removal(self, secondary_stow_condition):
        if secondary_stow_condition is False:
            return self.df_flags.copy()
        return self.df_flags.replace("stow", "").copy()  # i.e. needs another flag -> r2 flag

    def remove_flagged_ranges(self, secondary_stow_condition=False):
        df_flags = self._get_flags_for_removal(secondary_stow_condition)
        self.df_final, self.removed_ranges = _remove_flagged_ranges(self.df_native, df_flags)
        self.df_final["OE.POA_average"] = self.df_final[self.poa_columns].mean(axis=1)

    def plot_results(self):
        if self.df_final is None:
            raise ValueError("Run 'remove_flagged_ranges' before calling this function.")
        df = self.df_final.resample("1h").mean().copy()
        kwargs = dict(row_col_list=[[c] for c in df.columns], fig_height=200 * len(df.columns))
        if len(df.columns) == 2:
            kwargs.update(dict(row_height_list=[0.5, 0.5], fig_height=500))
        fig = oeplots.ts_subplots(df, **kwargs)
        fig.update_traces(line_color="#000000", selector=dict(name=df.columns[-1]))

        # add clearsky poa trace + dtn poa trace to last row
        tkwargs = dict(x=self.df.index, legendgroup="reference", opacity=0.5, hoverinfo="skip")
        for zorder, col in enumerate(["poa_clearsky", "poa_global"], start=-2):
            line_color = "#777777" if col == "poa_global" else "#aaaaaa"
            col_kwargs = tkwargs | dict(line_color=line_color)
            if col == "poa_clearsky":
                col_kwargs.update(dict(line_dash="dot"))
            fig.add_trace(
                go.Scatter(y=self.df[col], name=col, showlegend=True, **col_kwargs),
                row=len(df.columns),
                col=1,
            )
            fig.update_traces(zorder=zorder, selector=dict(name=col))

        vkwargs = dict(type="rect", y0=0, y1=1, line_width=0, opacity=0.2, layer="below")
        yref = lambda row: f"y{row} domain".replace("y1", "y")
        shapes = []
        for row, col in enumerate(df.columns[:-1], start=1):
            for idx in self.removed_ranges[col]:
                shapes.append(
                    dict(x0=idx[0], x1=idx[-1], yref=yref(row), fillcolor="lightsalmon", **vkwargs)
                )
        fig.update_layout(shapes=shapes)
        return fig

    def plot(self, shade_by_flag=True, resample=True):
        # generate subplot figure
        kwargs = dict(row_col_list=[[c] for c in self.results.keys()])
        if len(self.results) == 2:
            kwargs.update(dict(row_height_list=[0.5, 0.5]))

        dfplot = self.df.copy()
        dfflags = self.df_flags.copy()
        if resample is True:
            dfplot = dfplot.resample("1h").mean()
            dfflags = dfflags.resample("1h").first()

        fig = oeplots.ts_subplots(dfplot, **kwargs, fig_height=500)

        for col in self.results.keys():
            fig.update_traces(
                customdata=dfflags[f"{col}_FLAG"],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Value: %{y:.2f}<br>"
                    "<i>%{x|%Y-%m-%d %H:%M}</i><br>"
                    "Flags: %{customdata}<extra></extra>"
                ),
                selector=dict(name=col),
            )

        # add clearsky poa trace + dtn poa trace to each row
        tkwargs = dict(x=dfplot.index, legendgroup="reference", opacity=0.7, hoverinfo="skip")
        for zorder, col in enumerate(["poa_clearsky", "poa_global"], start=-2):
            line_color = "#777777" if col == "poa_global" else "#aaaaaa"
            col_kwargs = tkwargs | dict(line_color=line_color)
            if col == "poa_clearsky":
                col_kwargs.update(dict(line_dash="dot"))
            for r in range(len(self.results)):
                fig.add_trace(
                    go.Scatter(y=dfplot[col], name=col, showlegend=(r == 0), **col_kwargs),
                    row=(r + 1),
                    col=1,
                )
            fig.update_traces(zorder=zorder, selector=dict(name=col))

        # add shaded regions for qc flags
        vkwargs = dict(type="rect", y0=0, y1=1, line_width=0, opacity=0.2, layer="below")
        yref = lambda row: f"y{row} domain".replace("y1", "y")
        colors = {
            "bad_r2": "rgb(255, 210, 0)",
            "bad_data": "rgb(247, 37, 133)",
            "stowed": "rgb(0, 180, 216)",
        }
        shapes = []
        if shade_by_flag is True:
            for row, dict_ in enumerate(self.results.values(), start=1):
                for pfx, color in colors.items():
                    for idx in dict_[f"{pfx}_ranges"]:
                        x0, x1 = idx[0], idx[-1]
                        if resample is True:
                            x0 = x0.floor("h")
                            x1 = x1.ceil("h")
                        shapes.append(
                            dict(x0=x0, x1=x1, yref=yref(row), fillcolor=color, **vkwargs)
                        )
        else:
            vkwargs.update(dict(fillcolor="lightsalmon"))
            for row, col in enumerate(self.results.keys(), start=1):
                for idx in self.flagged_ranges[col]:
                    x0, x1 = idx[0], idx[-1]
                    if resample is True:
                        x0 = x0.floor("h")
                        x1 = x1.ceil("h")
                    shapes.append(dict(x0=x0, x1=x1, yref=yref(row), **vkwargs))
        fig.update_layout(shapes=shapes)

        return fig
