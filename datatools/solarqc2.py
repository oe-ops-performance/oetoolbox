"""POA sensor quality control with coarse-resolution scoring and native flag mapping."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pvanalytics.quality.irradiance as iqc
import pvanalytics.features.daytime as daytime
from pvanalytics.features.orientation import fixed_nrel
from sklearn.metrics import r2_score

from .pvlib import _get_aligned_clearsky_data
from .qc import get_sun
from ..utils import oemeta, oeplots
from ..utils.assets import SolarSite
from ..utils.datetime import infer_freq, upsample


@dataclass
class POAQCConfig:
    """Thresholds and analysis options for POA sensor quality control."""

    r2_min_ref: float = 0.75
    r2_min_stow: Optional[float] = None
    analysis_freq: str = "1h"
    min_samples_per_half_day: int = 3
    min_sample_coverage: float = 0.5
    min_bad_duration: Union[str, pd.Timedelta] = "2h"
    max_abs_relative_bias: Optional[float] = None
    gain_bounds: Optional[tuple[float, float]] = None
    meter_min_samples: int = 5
    meter_min_coverage: float = 0.6
    meter_min_correlation: float = 0.7

    def __post_init__(self):
        if self.analysis_freq not in ("1h", "1min"):
            raise ValueError("Unsupported analysis frequency.")
        if not 0.0 <= self.r2_min_ref <= 1.0:
            raise ValueError("r2_min_ref must be between 0 and 1.")
        if self.r2_min_stow is not None and not 0.0 <= self.r2_min_stow <= 1.0:
            raise ValueError("r2_min_stow must be between 0 and 1.")
        if self.min_samples_per_half_day < 2:
            raise ValueError("min_samples_per_half_day must be at least 2.")
        if not 0.0 <= self.min_sample_coverage <= 1.0:
            raise ValueError("min_sample_coverage must be between 0 and 1.")
        if self.meter_min_samples < 3:
            raise ValueError("meter_min_samples must be at least 3.")
        if not 0.0 <= self.meter_min_coverage <= 1.0:
            raise ValueError("meter_min_coverage must be between 0 and 1.")
        if not -1.0 <= self.meter_min_correlation <= 1.0:
            raise ValueError("meter_min_correlation must be between -1 and 1.")
        duration = pd.Timedelta(self.min_bad_duration)
        if pd.isna(duration) or duration < pd.Timedelta(0):
            raise ValueError("min_bad_duration must be a non-negative duration.")
        if self.max_abs_relative_bias is not None and (
            not np.isfinite(self.max_abs_relative_bias) or self.max_abs_relative_bias < 0
        ):
            raise ValueError("max_abs_relative_bias must be finite and non-negative.")
        if self.gain_bounds is not None and (
            len(self.gain_bounds) != 2
            or not all(np.isfinite(value) for value in self.gain_bounds)
            or self.gain_bounds[0] > self.gain_bounds[1]
        ):
            raise ValueError("gain_bounds must be a finite, ordered (minimum, maximum) pair.")


def _get_continuous_ranges(df_segmented: pd.DataFrame, native_freq: pd.Timedelta):
    if not isinstance(df_segmented.index, pd.DatetimeIndex):
        raise TypeError("Dataframe must have a datetime index.")
    if len(df_segmented.index) == 0:
        return []
    index = df_segmented.index
    if not index.is_monotonic_increasing:
        raise ValueError("Datetime index must be sorted to find continuous ranges.")
    if len(index) == 1:
        return [index]

    breaks = np.flatnonzero((index[1:] - index[:-1]) != native_freq) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(index)]
    return [index[start:end] for start, end in zip(starts, ends) if end > start]


def _indices_in_ranges(ref_index, ranges, default_step=None):
    """Return timestamps covered by discrete ranges, including the final sample."""
    selected = np.zeros(len(ref_index), dtype=bool)
    if len(ref_index) == 0 or not ranges:
        return selected

    for index in ranges:
        if len(index) == 0:
            continue
        if len(index) > 1:
            step = infer_freq(index)
        else:
            step = default_step or infer_freq(ref_index)
        start = ref_index.searchsorted(index[0], side="left")
        stop = ref_index.searchsorted(index[-1] + step, side="left")
        selected[start:stop] = True
    return selected


def _ranges_at_least_duration(ranges, cadence, minimum_duration):
    return [index for index in ranges if (index[-1] - index[0] + cadence) >= minimum_duration]


def _align_series(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Align a timestamped reference to the target index without positional matching."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Reference series must have a DatetimeIndex.")
    if not series.index.is_monotonic_increasing or series.index.has_duplicates:
        raise ValueError("Reference series index must be sorted and unique.")
    if str(series.index.tz) != str(index.tz):
        if series.index[0].utcoffset() == index[0].utcoffset():
            series = series.tz_convert(index.tz)
        else:
            raise ValueError("Reference and target indexes must use the same timezone.")
    return series.reindex(index)


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

    is_bad = ~final_qc_mask & daytime_mask & poa_measured.notna()

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
    is_stowed = pd.Series(False, index=poa_clearsky.index)
    stowed_ranges = []
    if has_trackers is True:
        if solar_noon is None:
            raise ValueError("Must provide solar noon reference for stowed tracker detection.")
        is_stowed = analyze_half_days(
            poa_measured=poa_measured, solar_noon=solar_noon, r2_min=r2_min
        )
        is_stowed &= core_daylight_mask & daytime_mask & poa_measured.notna()
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
            poa_measured=df[col],
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
    flags = pd.Series("", index=ref_index, dtype=object, name=f"{col}_FLAG")
    for flag, ranges in ranges_by_flag.items():
        condition = _indices_in_ranges(ref_index, ranges)
        flags.iloc[condition] = flag
    return flags


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
    df_flags = (
        df_flags[flag_cols]
        .reindex(df_native.index.union(df_flags.index).sort_values())
        .ffill()
        .reindex(df_native.index)
    )

    df = df_native[data_cols].copy()
    removed_ranges_by_col = {}
    for col, flag in zip(data_cols, flag_cols):
        is_flagged = df_flags[flag].fillna("").ne("")
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
    df: pd.DataFrame,
    data_col: str,
    ref_col: str,
    solar_noon: pd.Series,
    has_trackers: bool,
    min_samples: int = 3,
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    freq_td = infer_freq(df.index)
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2.")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be between 0 and 1.")

    # 2. Define the core daylight period (exclude shoulder periods)
    # Drop anything within 5 degrees of the horizon or very low clear-sky signals
    max_zenith = 85 if has_trackers else 82
    min_val = 0.05 if ref_col == "site_meter" else 25

    data_list = []
    for date, day_data in df.groupby(df.index.date):
        noon_ = solar_noon.loc[date]
        noon_str = noon_.strftime("%Y-%m-%d %H:%M")  # prevent tz mismatch errors
        core_daylight_mask = (day_data["apparent_zenith"] <= max_zenith) & (
            day_data[ref_col] > min_val
        )
        valid = core_daylight_mask & day_data[[data_col, ref_col]].notna().all(axis=1)
        periods = (
            ("am", day_data.index < noon_str, pd.Timestamp(date)),
            ("pm", day_data.index >= noon_str, noon_.round(freq_td)),
        )
        for period, side_mask, start in periods:
            samples = day_data.loc[valid & side_mask, [data_col, ref_col]]
            n_samples = len(samples)
            n_expected = int((side_mask & day_data["apparent_zenith"].le(max_zenith)).sum())
            n_reference = int((side_mask & core_daylight_mask).sum())
            coverage = n_samples / n_expected if n_expected else np.nan
            reference_coverage = n_reference / n_expected if n_expected else np.nan
            if n_samples < min_samples or coverage < min_coverage:
                r2, masd, bias, relative_bias, gain = (np.nan,) * 5
            else:
                measured = samples[data_col]
                reference = samples[ref_col]
                r2 = (
                    r2_score(y_true=reference, y_pred=measured)
                    if reference.nunique() > 1
                    else np.nan
                )

                adjacent = samples.index.to_series().diff().eq(freq_td)
                changes = measured.diff().abs().where(adjacent)
                masd = changes.mean()
                residual = measured - reference
                bias = residual.mean()
                relative_bias = (residual / reference).median()
                reference_energy = reference.pow(2).sum()
                gain = (
                    (reference * measured).sum() / reference_energy
                    if reference_energy > 0
                    else np.nan
                )
                n_samples = len(samples)
            data_list.append(
                [
                    date,
                    start,
                    period,
                    r2,
                    masd,
                    bias,
                    relative_bias,
                    gain,
                    n_samples,
                    n_expected,
                    coverage,
                    reference_coverage,
                ]
            )

    df = pd.DataFrame(
        data_list,
        columns=[
            "date",
            "start",
            "period",
            "r2_score",
            "MASD",
            "bias",
            "relative_bias",
            "gain",
            "n_samples",
            "n_expected",
            "coverage",
            "reference_coverage",
        ],
    )
    df["date"] = pd.to_datetime(df["date"])
    df["start"] = pd.to_datetime(df["start"])
    return df.set_index("date")


def meter_shape_correlation_by_half_day(
    df: pd.DataFrame,
    data_col: str,
    meter_col: str,
    clearsky_col: str,
    solar_noon: pd.Series,
    has_trackers: bool,
    min_samples: int = 5,
    min_coverage: float = 0.6,
) -> pd.DataFrame:
    """Compare sensor and meter clear-sky-index shape for each half-day."""
    if min_samples < 3:
        raise ValueError("Meter min_samples must be at least 3.")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("Meter min_coverage must be between 0 and 1.")

    max_zenith = 85 if has_trackers else 82
    freq_td = infer_freq(df.index)
    data = []
    for date, day_data in df.groupby(df.index.date):
        noon = solar_noon.loc[date]
        noon_str = noon.strftime("%Y-%m-%d %H:%M")
        daylight = day_data["apparent_zenith"].le(max_zenith) & day_data[clearsky_col].gt(50)
        meter_valid = (
            day_data["meter_valid"].fillna(False).astype(bool)
            if "meter_valid" in day_data.columns
            else pd.Series(True, index=day_data.index)
        )
        valid = (
            daylight
            & meter_valid
            & day_data[[data_col, meter_col, clearsky_col]].notna().all(axis=1)
        )
        periods = (
            ("am", day_data.index < noon_str, pd.Timestamp(date)),
            ("pm", day_data.index >= noon_str, noon.round(freq_td)),
        )
        for period, side_mask, start in periods:
            samples = day_data.loc[valid & side_mask, [data_col, meter_col, clearsky_col]]
            n_samples = len(samples)
            n_expected = int((daylight & side_mask).sum())
            coverage = n_samples / n_expected if n_expected else np.nan
            correlation = np.nan
            if n_samples >= min_samples and coverage >= min_coverage:
                clearsky = samples[clearsky_col]
                sensor_csi = samples[data_col] / clearsky
                meter_csi = samples[meter_col] / clearsky
                if sensor_csi.nunique() > 1 and meter_csi.nunique() > 1:
                    correlation = sensor_csi.rank().corr(meter_csi.rank())
            data.append([date, start, period, correlation, n_samples, coverage])

    result = pd.DataFrame(
        data,
        columns=[
            "date",
            "start",
            "period",
            "meter_shape_corr",
            "meter_n_samples",
            "meter_coverage",
        ],
    )
    result["date"] = pd.to_datetime(result["date"])
    result["start"] = pd.to_datetime(result["start"])
    return result.set_index("date")


def _formatted_r2_series(
    df_r2: pd.DataFrame, ref_index: pd.DatetimeIndex, stat_col="r2_score"
) -> pd.Series:
    if all(c in df_r2.columns for c in ["period", "start"]):
        stats = df_r2.set_index("start")[stat_col]
        boundaries = stats.index
        positions = boundaries.searchsorted(ref_index.tz_localize(None), side="right") - 1
        values = np.full(len(ref_index), np.nan)
        valid = (positions >= 0) & (positions < len(stats))
        values[valid] = stats.to_numpy()[positions[valid]]
        return pd.Series(values, index=ref_index)
    return ref_index.tz_localize(None).normalize().map(df_r2[stat_col])


def generate_r2_flags(
    series: pd.Series,
    dtn_series: pd.Series,
    apparent_zenith: pd.Series,
    location_kwargs: dict,
    has_trackers: bool,
    meter_data: pd.Series = None,
    meter_valid_series: pd.Series = None,
    clearsky_series: pd.Series = None,
    solar_noon: pd.Series = None,
    min_samples_per_half_day: int = 3,
    min_sample_coverage: float = 0.5,
    meter_min_samples: int = 5,
    meter_min_coverage: float = 0.6,
):
    data_col = series.name if series.name else "data"
    ref_col = dtn_series.name if dtn_series.name else "reference"
    if solar_noon is None:
        solar_noon = _get_solar_noon(
            dates=sorted(set(series.index.date)), location_kwargs=location_kwargs
        )
    df = series.to_frame(name=data_col).copy()
    df[ref_col] = _align_series(dtn_series, df.index)
    df["apparent_zenith"] = _align_series(apparent_zenith, df.index)
    if meter_data is not None:
        df["site_meter"] = _align_series(meter_data, df.index)
        if meter_valid_series is not None:
            df["meter_valid"] = _align_series(meter_valid_series, df.index).fillna(False)
    elif meter_valid_series is not None:
        raise ValueError("meter_valid_series requires meter_data.")

    if clearsky_series is not None:
        df["ref_clearsky"] = _align_series(clearsky_series, df.index)

    kwargs = dict(data_col=data_col, solar_noon=solar_noon, has_trackers=has_trackers)
    df_r2 = r2_by_half_day(
        df=df,
        ref_col=ref_col,
        min_samples=min_samples_per_half_day,
        min_coverage=min_sample_coverage,
        **kwargs,
    )
    for stat in (
        "r2_score",
        "MASD",
        "bias",
        "relative_bias",
        "gain",
        "n_samples",
        "n_expected",
        "coverage",
        "reference_coverage",
    ):
        df[f"{data_col}_{stat if stat != 'r2_score' else 'r2_dtn'}"] = _formatted_r2_series(
            df_r2=df_r2, ref_index=df.index, stat_col=stat
        )

    if "site_meter" in df.columns and "ref_clearsky" in df.columns:
        df_meter_shape = meter_shape_correlation_by_half_day(
            df=df,
            data_col=data_col,
            meter_col="site_meter",
            clearsky_col="ref_clearsky",
            solar_noon=solar_noon,
            has_trackers=has_trackers,
            min_samples=meter_min_samples,
            min_coverage=meter_min_coverage,
        )
        for stat in ("meter_shape_corr", "meter_n_samples", "meter_coverage"):
            df[f"{data_col}_{stat}"] = _formatted_r2_series(
                df_r2=df_meter_shape, ref_index=df.index, stat_col=stat
            )

    if "ref_clearsky" in df.columns:
        df_r2_cs = r2_by_half_day(
            df=df,
            ref_col="ref_clearsky",
            min_samples=min_samples_per_half_day,
            min_coverage=min_sample_coverage,
            **kwargs,
        )
        df[f"{data_col}_r2_clearsky"] = _formatted_r2_series(df_r2=df_r2_cs, ref_index=df.index)

    return df


def process_poa_data_via_r2(
    df: pd.DataFrame,
    location_kwargs: dict,
    has_trackers: bool,
    r2_min: float = 0.75,
    min_samples_per_half_day: int = 3,
    min_sample_coverage: float = 0.5,
    min_bad_duration: Union[str, pd.Timedelta] = "2h",
    reference_poa: pd.Series = None,
    max_abs_relative_bias: float = None,
    gain_bounds: tuple[float, float] = None,
    meter_min_samples: int = 5,
    meter_min_coverage: float = 0.6,
    meter_min_correlation: float = 0.7,
):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("POA data must have a DatetimeIndex.")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("POA data index must be sorted and unique.")
    if df.empty:
        raise ValueError("POA data cannot be empty.")
    if not 0.0 <= r2_min <= 1.0:
        raise ValueError("r2_min must be between 0 and 1.")
    if min_samples_per_half_day < 2:
        raise ValueError("min_samples_per_half_day must be at least 2.")
    if not 0.0 <= min_sample_coverage <= 1.0:
        raise ValueError("min_sample_coverage must be between 0 and 1.")
    if meter_min_samples < 3:
        raise ValueError("meter_min_samples must be at least 3.")
    if not 0.0 <= meter_min_coverage <= 1.0:
        raise ValueError("meter_min_coverage must be between 0 and 1.")
    if not -1.0 <= meter_min_correlation <= 1.0:
        raise ValueError("meter_min_correlation must be between -1 and 1.")
    min_bad_duration = pd.Timedelta(min_bad_duration)
    if pd.isna(min_bad_duration) or min_bad_duration < pd.Timedelta(0):
        raise ValueError("min_bad_duration must be a non-negative duration.")
    if max_abs_relative_bias is not None and (
        not np.isfinite(max_abs_relative_bias) or max_abs_relative_bias < 0
    ):
        raise ValueError("max_abs_relative_bias must be finite and non-negative.")
    if gain_bounds is not None and (
        len(gain_bounds) != 2
        or not all(np.isfinite(value) for value in gain_bounds)
        or gain_bounds[0] > gain_bounds[1]
    ):
        raise ValueError("gain_bounds must be a finite, ordered (minimum, maximum) pair.")
    poa_cols = list(df.filter(like="OE.POA_").columns)
    if not poa_cols:
        raise ValueError("No POA sensor columns found.")
    if not all(c in df.columns for c in ["apparent_zenith", "poa_global"]):
        raise KeyError("Missing one or more required columns.")
    analysis_df = df
    if infer_freq(df.index) < pd.Timedelta(hours=1):
        analysis_df = df.resample("1h").mean()
    if reference_poa is not None:
        analysis_df = analysis_df.copy()
        analysis_df["poa_global"] = _align_series(reference_poa, analysis_df.index)
    solar_noon = _get_solar_noon(
        dates=sorted(set(analysis_df.index.date)), location_kwargs=location_kwargs
    )
    results = {}
    for col in poa_cols:
        df_flags = generate_r2_flags(
            series=analysis_df[col],
            dtn_series=analysis_df["poa_global"],
            apparent_zenith=analysis_df["apparent_zenith"],
            meter_data=analysis_df.get("site_meter"),
            meter_valid_series=analysis_df.get("meter_valid"),
            clearsky_series=analysis_df.get("poa_clearsky"),
            solar_noon=solar_noon,
            location_kwargs=location_kwargs,
            has_trackers=has_trackers,
            min_samples_per_half_day=min_samples_per_half_day,
            min_sample_coverage=min_sample_coverage,
            meter_min_samples=meter_min_samples,
            meter_min_coverage=meter_min_coverage,
        )

        cadence = infer_freq(analysis_df.index)
        missing = analysis_df[col].isna()
        enough_paired_samples = df_flags[f"{col}_n_samples"].ge(min_samples_per_half_day)
        enough_coverage = df_flags[f"{col}_coverage"].ge(min_sample_coverage)
        reference_disagrees = df_flags[f"{col}_r2_dtn"].lt(r2_min)

        has_meter = "site_meter" in analysis_df.columns
        if has_meter:
            meter_corr = df_flags.get(
                f"{col}_meter_shape_corr",
                pd.Series(np.nan, index=df_flags.index),
            )
            meter_assessable = meter_corr.notna()
            meter_supports_sensor = meter_corr.ge(meter_min_correlation) & meter_assessable
        else:
            meter_assessable = pd.Series(False, index=df_flags.index)
            meter_supports_sensor = pd.Series(False, index=df_flags.index)

        # A meter can corroborate the sensor when the POA reference disagrees.
        # Clear-sky fit remains diagnostic; cloudiness is not inferred from sensor variability.
        reference_mismatch = (
            enough_paired_samples
            & enough_coverage
            & reference_disagrees
            & ((not has_meter) | (meter_assessable & ~meter_supports_sensor))
        )
        calibration_mismatch = pd.Series(False, index=df_flags.index)
        if max_abs_relative_bias is not None:
            calibration_mismatch |= df_flags[f"{col}_relative_bias"].abs().gt(max_abs_relative_bias)
        if gain_bounds is not None:
            calibration_mismatch |= df_flags[f"{col}_gain"].lt(gain_bounds[0]) | df_flags[
                f"{col}_gain"
            ].gt(gain_bounds[1])
        calibration_mismatch &= enough_paired_samples & enough_coverage

        missing_ranges = _get_continuous_ranges(analysis_df.loc[missing], native_freq=cadence)
        mismatch_ranges = _get_continuous_ranges(
            analysis_df.loc[reference_mismatch], native_freq=cadence
        )
        calibration_ranges = _get_continuous_ranges(
            analysis_df.loc[calibration_mismatch], native_freq=cadence
        )
        max_zenith = 85 if has_trackers else 82
        daylight = analysis_df["apparent_zenith"].le(max_zenith) & (
            analysis_df["poa_global"].isna() | analysis_df["poa_global"].gt(25)
        )
        inconclusive = (~enough_paired_samples | ~enough_coverage) & daylight
        if has_meter:
            inconclusive |= (
                enough_paired_samples & enough_coverage & reference_disagrees & ~meter_assessable
            )
        inconclusive_ranges = _get_continuous_ranges(
            analysis_df.loc[inconclusive], native_freq=cadence
        )

        missing_ranges = _ranges_at_least_duration(missing_ranges, cadence, min_bad_duration)
        mismatch_ranges = _ranges_at_least_duration(mismatch_ranges, cadence, min_bad_duration)
        calibration_ranges = _ranges_at_least_duration(
            calibration_ranges, cadence, min_bad_duration
        )
        inconclusive_ranges = _ranges_at_least_duration(
            inconclusive_ranges, cadence, min_bad_duration
        )
        removal_mask = _indices_in_ranges(
            analysis_df.index,
            missing_ranges + mismatch_ranges + calibration_ranges,
            default_step=cadence,
        )
        ranges_to_remove = _get_continuous_ranges(
            pd.DataFrame(index=analysis_df.index[removal_mask]), native_freq=cadence
        )
        results[col] = dict(
            df=df_flags,
            ranges_to_remove=ranges_to_remove,
            missing_ranges=missing_ranges,
            reference_mismatch_ranges=mismatch_ranges,
            calibration_mismatch_ranges=calibration_ranges,
            inconclusive_ranges=inconclusive_ranges,
        )

    return results


class POADataset:
    """Assess POA sensor quality against irradiance and plant-meter references.

    ``meter_valid_mask`` is an optional boolean series indexed like ``ref_meter``;
    True marks meter samples usable for corroboration. Without it, meter samples
    are assumed operationally valid because this class cannot infer curtailment,
    clipping, or outage status from power values alone.
    """

    def __init__(
        self,
        site: str,
        dataframe: pd.DataFrame,
        ref_poa: pd.Series,
        ref_meter: pd.Series = None,
        config: POAQCConfig = None,
        cleaning_changes: dict = None,
        meter_valid_mask: pd.Series = None,
    ):
        self.site = SolarSite(site)
        self.config = deepcopy(config) if config is not None else POAQCConfig()
        self.has_trackers = oemeta.data["TrackerCount"].get(site, 0) > 0
        self.analysis_freq = pd.Timedelta(self.config.analysis_freq)

        self.qc_changes = {}  # init
        self.cleaning_changes = deepcopy(cleaning_changes) if cleaning_changes else {}
        if self.cleaning_changes:
            self.qc_changes["input_cleaning_changes"] = deepcopy(self.cleaning_changes)
        self._process_input_data(
            df=dataframe,
            ref_poa=ref_poa,
            ref_meter=ref_meter,
            meter_valid_mask=meter_valid_mask,
        )

        self._run_poa_qc()

        self.df_clean = None
        self.df_final = None
        self.removed_ranges = {}

    def _process_input_data(self, df, ref_poa, ref_meter, meter_valid_mask=None):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("POA sensor data must have a DatetimeIndex.")
        if not df.index.is_monotonic_increasing or df.index.has_duplicates:
            raise ValueError("POA sensor data index must be sorted and unique.")
        if ref_poa.empty:
            raise ValueError("Reference POA data cannot be empty.")
        _align_series(ref_poa, ref_poa.index)
        if ref_meter is not None:
            _align_series(ref_meter, ref_meter.index)
        if meter_valid_mask is not None:
            if ref_meter is None:
                raise ValueError("meter_valid_mask requires ref_meter.")
            if not pd.api.types.is_bool_dtype(meter_valid_mask.dtype):
                raise TypeError("meter_valid_mask must contain boolean values.")
            _align_series(meter_valid_mask, ref_meter.index)
        if ref_poa.name != "poa_global":
            raise ValueError("Unexpected column name in reference POA data.")
        poa_cols_ = list(df.filter(like="OE.POA_").columns)
        if not poa_cols_:
            raise ValueError("No 'OE.POA' sensor columns found.")
        poa_cols = list(df.filter(like="OE.POA_").dropna(axis=1, how="all").columns)
        if not poa_cols:
            raise ValueError("No valid data found for 'OE.POA' sensor column(s).")

        self.native_freq = infer_freq(df.index)
        reference_freq = infer_freq(ref_poa.index)
        if reference_freq < pd.Timedelta(hours=1):
            self.ref_poa_hourly = ref_poa.resample("1h").mean()
        elif reference_freq > pd.Timedelta(hours=1):
            self.ref_poa_hourly = upsample(ref_poa.to_frame(), new_freq="1h").iloc[:, 0]
        else:
            self.ref_poa_hourly = ref_poa.copy()

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

        # Preserve the hourly reference for scoring; hold it across sub-hourly samples.
        if self.analysis_freq == pd.Timedelta(hours=1):
            self.df["poa_global"] = _align_series(self.ref_poa_hourly, self.df.index)
        else:
            ref_poa_analysis = self.ref_poa_hourly.resample(self.analysis_freq).ffill()
            self.df["poa_global"] = _align_series(ref_poa_analysis, self.df.index)

        if ref_meter is not None:
            meter_freq = infer_freq(ref_meter.index)
            if meter_freq == self.analysis_freq:
                self.df["site_meter"] = _align_series(ref_meter, self.df.index)
            elif meter_freq < self.analysis_freq:
                meter_resampled = ref_meter.resample(self.analysis_freq).mean()
                self.df["site_meter"] = _align_series(meter_resampled, self.df.index)
            else:
                raise ValueError("Invalid frequency for provided meter data.")
            self.df["site_meter"] = self.df["site_meter"].clip(lower=0.0)

        if "poa_clearsky" not in self.df.columns:
            df_clearsky = _get_aligned_clearsky_data(
                self.site.name, self.df.index, include_solar_position=True
            )
            self.df = self.df.join(df_clearsky)

        self.df_hourly = self.df_native.resample("1h").mean()
        self.df_hourly["poa_global"] = _align_series(self.ref_poa_hourly, self.df_hourly.index)
        if ref_meter is not None:
            meter_hourly = ref_meter.resample("1h").mean().clip(lower=0.0)
            self.df_hourly["site_meter"] = _align_series(meter_hourly, self.df_hourly.index)
            if meter_valid_mask is not None:
                valid_hourly = (
                    meter_valid_mask.reindex(ref_meter.index)
                    .fillna(False)
                    .resample("1h")
                    .min()
                    .astype(bool)
                )
                self.df_hourly["meter_valid"] = _align_series(
                    valid_hourly, self.df_hourly.index
                ).fillna(False)
        hourly_clearsky = _get_aligned_clearsky_data(
            self.site.name, self.df_hourly.index, include_solar_position=True
        )
        self.df_hourly = self.df_hourly.join(hourly_clearsky)

    def _clean_native_data(self, df_native):
        df = df_native.dropna(axis=1, how="all").copy()
        empty_cols = [c for c in df_native.columns if c not in df.columns]
        zeroed_cols = []
        for col in df.columns:
            if df[col].abs().max() < 1e-5:
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
            df_zero = pd.DataFrame(index=df.index[df[col].eq(0)])
            zero_ranges = _get_continuous_ranges(df_zero, self.native_freq)
            flatlined_ranges = [x for x in zero_ranges if (x[-1] - x[0]) > pd.Timedelta(days=1)]
            if flatlined_ranges:
                df.loc[_indices_in_ranges(df.index, flatlined_ranges), col] = np.nan
                flatlined_at_zero_ranges.update({col: flatlined_ranges})
        if flatlined_at_zero_ranges:
            self.qc_changes.update({"zero_ranges": flatlined_at_zero_ranges})

        return df

    def _run_poa_qc(self):
        kwargs = dict(
            location_kwargs=self.site.location_kwargs,
            has_trackers=self.has_trackers,
        )
        results_1 = process_poa_data(df=self.df_hourly, r2_min=self.config.r2_min_stow, **kwargs)
        results_2 = process_poa_data_via_r2(
            df=self.df_hourly,
            r2_min=self.config.r2_min_ref,
            min_samples_per_half_day=self.config.min_samples_per_half_day,
            min_sample_coverage=self.config.min_sample_coverage,
            min_bad_duration=self.config.min_bad_duration,
            reference_poa=self.ref_poa_hourly,
            max_abs_relative_bias=self.config.max_abs_relative_bias,
            gain_bounds=self.config.gain_bounds,
            meter_min_samples=self.config.meter_min_samples,
            meter_min_coverage=self.config.meter_min_coverage,
            meter_min_correlation=self.config.meter_min_correlation,
            **kwargs,
        )
        self.results = {
            col: {
                "df1": dict_["df"],
                "bad_data_ranges": dict_["ranges_to_remove"],
                "stowed_ranges": dict_["stowed_ranges"],
                "df2": results_2[col]["df"],
                "bad_r2_ranges": results_2[col]["reference_mismatch_ranges"],
                "missing_ranges": results_2[col]["missing_ranges"],
                "calibration_mismatch_ranges": results_2[col]["calibration_mismatch_ranges"],
                "inconclusive_ranges": results_2[col]["inconclusive_ranges"],
            }
            for col, dict_ in results_1.items()
        }
        self._combine_flagged_ranges()
        self._generate_df_flags()

    def _combine_flagged_ranges(self):
        flagged_ranges = {}
        for col, dict_ in self.results.items():
            ranges = (
                dict_["bad_data_ranges"]
                + dict_["bad_r2_ranges"]
                + dict_["missing_ranges"]
                + dict_["calibration_mismatch_ranges"]
                + dict_["stowed_ranges"]
            )
            is_flagged = _indices_in_ranges(
                self.df.index, ranges, default_step=infer_freq(self.df_hourly.index)
            )
            flagged_ranges[col] = _get_continuous_ranges(
                pd.DataFrame(index=self.df.index[is_flagged]), infer_freq(self.df.index)
            )
        self.flagged_ranges = flagged_ranges

    def _generate_df_flags(self):
        series_list = []
        for col, results in self.results.items():
            ranges_by_flag = {
                "ref_r2": results["bad_r2_ranges"],
                "bad": results["bad_data_ranges"],
                "missing": results["missing_ranges"],
                "calibration": results["calibration_mismatch_ranges"],
                "stow": results["stowed_ranges"],
                "inconclusive": results["inconclusive_ranges"],
            }
            flag_data = pd.Series("", index=self.df.index, dtype=object)
            for flag, ranges in ranges_by_flag.items():
                condition = _indices_in_ranges(
                    self.df.index,
                    ranges,
                    default_step=infer_freq(self.df_hourly.index),
                )
                addition = flag_data.where(flag_data.eq(""), flag_data + ", ") + flag
                flag_data = flag_data.mask(condition, addition)
            series_list.append(pd.Series(flag_data, name=f"{col}_FLAG"))
        self.df_flags = pd.concat(series_list, axis=1)

    def _get_flags_for_removal(self, include_stow):
        excluded_flags = {"inconclusive"}
        if not include_stow:
            excluded_flags.add("stow")

        def retain_removal_flags(value):
            return ", ".join(flag for flag in value.split(", ") if flag not in excluded_flags)

        return self.df_flags.apply(lambda column: column.map(retain_removal_flags))

    def remove_flagged_ranges(self, secondary_stow_condition=False, include_stow=None):
        if include_stow is None:
            include_stow = not secondary_stow_condition
        df_flags = self._get_flags_for_removal(include_stow)
        self.df_final, self.removed_ranges = _remove_flagged_ranges(self.df_native, df_flags)
        self.df_final["OE.POA_average"] = self.df_final[self.poa_columns].mean(axis=1)

        # backfill
        dfref = self.ref_poa_hourly.to_frame().copy()
        if infer_freq(dfref.index) > self.analysis_freq:
            dfref = upsample(dfref, new_freq=self.analysis_freq)
        refpoa = dfref.iloc[:, 0].copy()
        self.df_final["Processed_POA"] = np.where(
            self.df_final["OE.POA_average"].isna(),
            refpoa.values,
            self.df_final["OE.POA_average"].values,
        )

    def plot_results(self):
        if self.df_final is None:
            raise ValueError("Run 'remove_flagged_ranges' before calling this function.")
        df = self.df_final.resample("1h").mean().copy()
        kwargs = dict(row_col_list=[[c] for c in df.columns], fig_height=200 * len(df.columns))
        if len(df.columns) == 2:
            kwargs.update(dict(row_height_list=[0.5, 0.5], fig_height=500))
        fig = oeplots.ts_subplots(df, **kwargs)
        fig.update_traces(line_color="#000000", selector=dict(name=df.columns[-2]))
        fig.update_traces(line=dict(color="#8D00C0"), selector=dict(name=df.columns[-1]))

        # add clearsky poa trace + dtn poa trace to second-to-last row
        tkwargs = dict(x=self.df.index, legendgroup="reference", opacity=0.5, hoverinfo="skip")
        for zorder, col in enumerate(["poa_global", "poa_clearsky"], start=-2):
            line_color = "#777777" if col == "poa_global" else "#aaaaaa"
            col_kwargs = tkwargs | dict(line_color=line_color)
            if col == "poa_clearsky":
                col_kwargs.update(dict(line_dash="dot"))
            trace = go.Scatter(y=self.df[col], name=col, showlegend=True, **col_kwargs)

            fig.add_trace(trace, row=len(df.columns) - 1, col=1)
            fig.add_trace(trace, row=len(df.columns), col=1)
            fig.update_traces(zorder=zorder, selector=dict(name=col))

        vkwargs = dict(type="rect", y0=0, y1=1, line_width=0, layer="below")
        yref = lambda row: f"y{row} domain".replace("y1", "y")
        shapes = []
        for row, col in enumerate(df.columns, start=1):
            shkwargs = dict(yref=yref(row), fillcolor="lightsalmon", **vkwargs)
            if col in self.removed_ranges:
                for idx in self.removed_ranges[col]:
                    shapes.append(dict(x0=idx[0], x1=idx[-1], opacity=0.4, **shkwargs))
            else:
                rngs = _get_continuous_ranges(df[df["OE.POA_average"].isna()], infer_freq(df.index))
                for idx in rngs:
                    shapes.append(dict(x0=idx[0], x1=idx[-1], opacity=0.2, **shkwargs))

        fig.update_layout(shapes=shapes)
        return fig

    def plot_result(self):
        df0 = self.df_native.copy()
        df1 = self.df_final.copy()
        *poacols, avg_col, final_col = df1.columns

        df = df1.copy()
        for col in poacols:
            df[f"{col}_removed"] = np.where(df[col].isna(), df0[col], np.nan)

        if infer_freq(df.index) != pd.Timedelta(hours=1):
            df = df.resample("h").mean()

        # create clearsky poa trace + dtn poa trace to add to each row
        tkwargs = dict(x=df.index, opacity=0.5, hoverinfo="skip")
        cs_col, ref_col = "poa_clearsky", "poa_global"
        clearsky_trace_kwargs = dict(
            **tkwargs,
            y=self.df_hourly[cs_col],
            name=cs_col,
            line=dict(color="#aaaaaa", width=1.5, dash="dot"),
        )
        ref_trace_kwargs = dict(
            **tkwargs,
            y=self.df_hourly[ref_col],
            name=ref_col,
            line=dict(color="#777777", width=1.5),
        )

        n_rows = len(df1.columns)
        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1 / n_rows,  # default 0.3/n_rows
        )

        hvtemp = (
            "<b>%{fullData.name}</b><br>Value: %{y:.2f}<br>"
            "<i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
        )
        lkwargs = dict(x=df.index, mode="lines", hovertemplate=hvtemp)

        shapes = []
        skwargs = dict(type="rect", y0=0, y1=1, line_width=0, layer="below")
        yref = lambda row: f"y{row} domain".replace("y1", "y")

        for rw, col in enumerate(poacols, start=1):
            fig.add_trace(go.Scatter(**ref_trace_kwargs, showlegend=False), row=rw, col=1)
            color = plotly.colors.qualitative.G10[rw - 1]
            cleaned_trace = go.Scattergl(
                **lkwargs, y=df[col], name=col, legendgroup=rw, line_color=color
            )
            fig.add_trace(cleaned_trace, row=rw, col=1)
            removed_vals_trace = go.Scattergl(
                **lkwargs,
                y=df[f"{col}_removed"],
                name=f"{col}_removed",
                line=dict(color=color, dash="dot"),
                opacity=0.7,
                legendgroup=rw,
            )
            fig.add_trace(removed_vals_trace, row=rw, col=1)

            shkwargs = dict(yref=yref(rw), fillcolor="lightsalmon", opacity=0.4, **skwargs)
            for idx in self.removed_ranges.get(col, []):
                shapes.append(dict(x0=idx[0].floor("h"), x1=idx[-1].ceil("h"), **shkwargs))

        for rw, col in enumerate([avg_col, final_col], start=n_rows - 1):
            color = "#303030" if col == final_col else plotly.colors.qualitative.G10[4]
            trace = go.Scattergl(**lkwargs, y=df[col], name=col, line_color=color, legendgroup=rw)
            fig.add_trace(trace, row=rw, col=1)

            lgdkwargs = dict(showlegend=rw == n_rows - 1, legendgroup=rw)
            if rw < n_rows:
                fig.add_trace(go.Scatter(**clearsky_trace_kwargs, **lgdkwargs), row=rw, col=1)
            fig.add_trace(go.Scatter(**ref_trace_kwargs, **lgdkwargs), row=rw, col=1)

            shkwargs = dict(yref=yref(rw), fillcolor="lightsalmon", opacity=0.2, **skwargs)
            rngs = _get_continuous_ranges(df[df["OE.POA_average"].isna()], infer_freq(df.index))
            for idx in rngs:
                shapes.append(dict(x0=idx[0], x1=idx[-1], **shkwargs))

        fig.update_traces(zorder=-2, selector=dict(name=cs_col))
        fig.update_traces(zorder=-1, selector=dict(name=ref_col))

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
            ),
            shapes=shapes,
            height=max([225 * n_rows, 500]),
        )
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
            "missing": "rgb(255, 127, 80)",
            "calibration_mismatch": "rgb(148, 103, 189)",
            "inconclusive": "rgb(130, 130, 130)",
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
