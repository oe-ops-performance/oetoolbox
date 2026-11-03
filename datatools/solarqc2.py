"""POA sensor quality control against geometry-aware references.

Each POA sensor is compared, at analysis resolution (default 5 min), with the POA it *should* read:

    expected(geometry) = clear-sky POA(geometry) x cloud index x sensor gain

- ``geometry`` is the array's own geometry ("tracker", or the fixed array) plus, for trackers, a set of
  candidate *stuck/stowed* rotations (flat, and +/- steps up to the tracker's max angle). All are
  clear-sky transpositions computed with pvlib.
- ``cloud index`` comes from two independent references: the normalized plant meter (1-minute, so it
  sees passing clouds) and the hourly DTN POA (smooth, but independent of plant operations).
  A window only counts against a sensor when it disagrees with *both* -- a curtailed meter or an
  hour-averaged DTN each fail on their own, a stowed/faulty sensor fails both.
- ``gain`` is the sensor's calibration vs. the clear-sky model, fit per day against the best-fitting
  geometry so that a sensor that is stowed most of the time still gets a sound estimate.

A window is labelled "ok" when the tracking geometry fits, "stow" when it does not but a fixed
rotation does (and the two geometries are physically distinguishable in that window), and "bad"
when nothing fits. Windows where stow and tracking are indistinguishable (around solar noon) take
the label of the nearest distinguishable window.
"""

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Optional, Union

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pvlib import irradiance, tracking
from pvlib.location import Location

from .pvlib import design_database, get_site_location
from ..utils.assets import SolarSite
from ..utils.datetime import infer_freq

TRACKER = "tracker"  # geometry key for the array's own orientation (tracking, or fixed tilt)
FLAG_COLORS = {
    "stow": "rgb(0, 180, 216)",
    "bad": "rgb(247, 37, 133)",
    "missing": "rgb(255, 127, 80)",
    "calibration": "rgb(148, 103, 189)",
    "inconclusive": "rgb(130, 130, 130)",
}
CLIPPED_CSI = 0.95  # meter cloud index at/above this is treated as clipped (censored)
VALUE_HOVER = (
    "<b>%{fullData.name}</b><br>Value: %{y:.2f}<br><i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
)
FLAG_HOVER = VALUE_HOVER.replace("<extra>", "<br>Flags: %{customdata}<extra>")  # needs customdata
REMOVAL_FLAGS = ("bad", "missing", "calibration")  # "stow" is added on request


@dataclass
class POAQCConfig:
    """Thresholds for POA sensor quality control.

    Errors are mean absolute differences over ``window``, as a fraction of the mean clear-sky POA in
    that window. ``tol`` is the allowed error under steady skies; it grows with cloud variability
    (``cloud_tol_factor`` x std of the meter cloud index; ``dtn_tol_factor`` x the sensor's own
    5-min variability for the smoother DTN reference).
    """

    analysis_freq: str = "5min"
    window: Union[str, pd.Timedelta] = "90min"  # scoring window (centered)
    smooth: Union[str, pd.Timedelta] = "15min"  # smoothing before comparing (point sensor vs plant)
    tol: float = 0.12
    cloud_tol_factor: float = 0.6
    dtn_tol_factor: float = 1.5
    tracker_fail_score: float = 1.5  # tracking fit worse than this x tol -> not "ok"
    stow_margin: float = 0.5  # fixed geometry must beat tracking by this x tol to be "stow"
    min_discrimination: float = 0.12  # geometries must differ by this fraction of clear-sky POA
    min_stow_ratio: float = 0.08  # windowed sensor/expected-tracking below this is "bad", not stow
    stow_transition_tol: float = 0.07  # sensor still this far BELOW tracking next to a stow = stow
    stow_transition_max: Union[str, pd.Timedelta] = "2h"  # longest transition attached to a stow run
    bad_min_peak_frac: float = 0.7  # 'bad' needs window clear-sky >= this x the day's peak (low sun)
    stow_max_roughness: float = 0.006  # meter stow must be smooth: median 1-min step / level <= this
    stow_requires_wind: bool = False  # if True, a plant-wide (meter) stow must coincide with wind
    stow_min_wind: float = 8.0  # ...of at least this (hourly mean, m/s); always used to report windy
    storm_min_precip: float = 0.1  # ...and no weather explaining the drop: precipitation >= this
    storm_max_sunshine: float = 45.0  # ...or sunshine below this (minutes per hour)
    storm_min_cloud: float = 60.0  # ...or cloud cover above this % (only if sunshine is absent)
    weather_lookaround: Union[str, pd.Timedelta] = "30min"  # weather is judged within this of a bin
    shape_max_roughness: float = 0.02  # shape test needs a steady sky: sensor/clear-sky 5-min step <=
    shape_max_rmse: float = 0.08  # best fixed geometry must fit the half-day this well (RMSE / mean POA)
    shape_max_ratio: float = 0.3  # ...and at most this x the tracking geometry's RMSE
    shape_min_track_rmse: float = 0.1  # ...which must itself clearly misfit (flat top vs bell)
    shape_min_hours: float = 2.0  # daylight hours needed for a half-day verdict
    dtn_only_max_curvature: float = 0.02  # without the meter, windows rougher than this are not judged
    max_stow_score: float = 1.0  # a stow run must fit ONE geometry with mean score <= this
    stow_rotation_step: int = 15  # degrees between candidate stuck rotations
    max_zenith: float = 80.0
    min_signal: float = 50.0  # W/m2 clear-sky POA below which a bin is not scored
    min_bad_duration: Union[str, pd.Timedelta] = "30min"
    min_stow_duration: Union[str, pd.Timedelta] = "30min"
    bridge_stow_gap: Union[str, pd.Timedelta] = "6h"  # "ok" gap between stow runs (midday overlap)
    ambiguity_fill: Union[str, pd.Timedelta] = "3h"
    gain_bounds: tuple = (0.8, 1.25)  # plausible sensor-vs-model gain when fitting
    flag_gain_bounds: Optional[tuple] = None  # if set, sensors whose gain is outside are flagged
    meter_envelope_quantile: float = 0.95
    fill_max_gap: Union[str, pd.Timedelta, None] = "15min"  # interpolate gaps in Processed_POA up to this
    anchor_min_bins: int = 36  # fewer trusted anchor bins than this -> use the sensors' own gain
    anchor_min_signal: float = 0.8  # backfill gain: only bins >= this x peak clear-sky POA count
    anchor_reach: Union[str, pd.Timedelta] = "1h"  # one-sided anchors fade to the default over this
    anchor_buffer: Union[str, pd.Timedelta] = "1h"  # ...and >= this far from removed data
    anchor_max_adjust: float = 0.05  # local backfill gain may differ from the default by this much
    drop_removed_fraction: Optional[float] = 0.7  # sensor with this share of daylight removed is dropped
    level_bounds: Optional[tuple] = None  # if set, sensors whose level vs. DTN POA is outside: "calibration"
    level_min_days: int = 5  # sunny days needed to judge a sensor's level
    shape_trend_window: Union[str, pd.Timedelta, None] = "2h"  # meter slopes slower than this are not cloud shape
    meter_level_tol: Optional[float] = 0.15  # meter level (vs DTN) is used where sensors agree within this
    meter_level_reach: Union[str, pd.Timedelta] = "6h"  # ...and within this of such an agreeing hour

    def __post_init__(self):
        for name in ("analysis_freq", "window", "smooth", "min_bad_duration", "min_stow_duration"):
            if pd.isna(pd.Timedelta(getattr(self, name))):
                raise ValueError(f"{name} must be a valid duration.")
        if pd.Timedelta(self.analysis_freq) < pd.Timedelta("1min"):
            raise ValueError("analysis_freq must be at least 1 minute.")
        if self.fill_max_gap is not None and pd.isna(pd.Timedelta(self.fill_max_gap)):
            raise ValueError("fill_max_gap must be a valid duration (or None to disable).")
        if self.stow_max_roughness <= 0:
            raise ValueError("stow_max_roughness must be positive.")
        if self.stow_min_wind < 0:
            raise ValueError("stow_min_wind must be non-negative.")
        if self.tol <= 0:
            raise ValueError("tol must be positive.")
        if not 0 < self.meter_envelope_quantile <= 1:
            raise ValueError("meter_envelope_quantile must be in (0, 1].")
        if len(self.gain_bounds) != 2 or self.gain_bounds[0] > self.gain_bounds[1]:
            raise ValueError("gain_bounds must be an ordered (min, max) pair.")
        if self.drop_removed_fraction is not None and not 0 < self.drop_removed_fraction <= 1:
            raise ValueError("drop_removed_fraction must be in (0, 1] (or None to disable).")
        if self.level_bounds is not None and (
            len(self.level_bounds) != 2 or self.level_bounds[0] > self.level_bounds[1]
        ):
            raise ValueError("level_bounds must be an ordered (min, max) pair.")
        if self.flag_gain_bounds is not None and (
            len(self.flag_gain_bounds) != 2 or self.flag_gain_bounds[0] > self.flag_gain_bounds[1]
        ):
            raise ValueError("flag_gain_bounds must be an ordered (min, max) pair.")

    def __str__(self):
        """One configuration parameter per line, with its current value."""
        width = max(len(f.name) for f in fields(self))
        lines = [f"{f.name:<{width}} = {getattr(self, f.name)!r}" for f in fields(self)]
        return "\n".join([f"{type(self).__name__}:", *("  " + line for line in lines)])


# ---- range helpers -----------------------------------------------------------------------------
def _get_continuous_ranges(df_segmented: pd.DataFrame, native_freq: pd.Timedelta):
    if not isinstance(df_segmented.index, pd.DatetimeIndex):
        raise TypeError("Dataframe must have a datetime index.")
    index = df_segmented.index
    if len(index) == 0:
        return []
    if not index.is_monotonic_increasing:
        raise ValueError("Datetime index must be sorted to find continuous ranges.")
    if len(index) == 1:
        return [index]
    breaks = np.flatnonzero((index[1:] - index[:-1]) != native_freq) + 1
    starts, ends = np.r_[0, breaks], np.r_[breaks, len(index)]
    return [index[s:e] for s, e in zip(starts, ends) if e > s]


def _ranges_from_mask(mask: pd.Series, freq: pd.Timedelta):
    return _get_continuous_ranges(pd.DataFrame(index=mask.index[mask.to_numpy(bool)]), freq)


def _run_lengths(mask: pd.Series) -> pd.Series:
    """Length (in samples) of the run of identical values each sample belongs to."""
    run_id = (mask != mask.shift()).cumsum()
    return mask.groupby(run_id).transform("size")


def _to_bins(duration, step) -> int:
    return max(1, int(round(pd.Timedelta(duration) / pd.Timedelta(step))))


def _as_tz(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Reference series -> same timezone-awareness as the target index (instants are preserved)."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Reference series must have a DatetimeIndex.")
    if (series.index.tz is None) != (index.tz is None):
        raise ValueError("Reference and sensor data must both be timezone-aware (or both naive).")
    if not series.index.is_monotonic_increasing or series.index.has_duplicates:
        raise ValueError("Reference series index must be sorted and unique.")
    return series.tz_convert(index.tz) if index.tz is not None else series


# ---- geometry ----------------------------------------------------------------------------------
def _design(site: str) -> dict:
    dict_inv = list(design_database(site).values())[0]
    is_tracker = dict_inv.get("Racking Type") == "Tracker"
    return dict(
        has_trackers=is_tracker,
        tilt=int(dict_inv["Tilt Angle (Deg)"]),  # max rotation for trackers, fixed tilt otherwise
        azimuth=float(dict_inv.get("Azimuth Angle (Deg)", 180)),
        gcr=dict_inv.get("Racking GCR"),
    )


def _stow_rotations(max_angle: int, step: int) -> list[int]:
    angles = {0, int(max_angle)}
    angles.update(range(step, int(max_angle) + 1, step))
    return sorted({s * a for a in angles for s in (-1, 1)})


def _orientations(solpos: pd.DataFrame, design: dict, rotation_step: int) -> dict:
    """(tilt, azimuth) per geometry: the array's own (a tracker's varies over time) plus, for
    trackers, the candidate stuck rotations."""
    if not design["has_trackers"]:
        return {TRACKER: (design["tilt"], design["azimuth"])}
    profile = tracking.singleaxis(
        solpos["apparent_zenith"],
        solpos["azimuth"],
        max_angle=design["tilt"],
        backtrack=True,
        gcr=design["gcr"],
    )
    out = {TRACKER: (profile["surface_tilt"].fillna(0.0), profile["surface_azimuth"].fillna(180.0))}
    for theta in _stow_rotations(design["tilt"], rotation_step):
        orient = tracking.calc_surface_orientation(float(theta), 0.0, 0.0)
        out[f"rot{theta:+d}"] = tuple(
            float(np.ravel(orient[k])[0]) for k in ("surface_tilt", "surface_azimuth")
        )
    return out


def _transpose(orientations, solpos, dni, ghi, dhi, dni_extra, index) -> pd.DataFrame:
    """POA (W/m2) per geometry for the given sky components (Perez)."""
    cols = {}
    for name, (tilt, azimuth) in orientations.items():
        total = irradiance.get_total_irradiance(
            tilt,
            azimuth,
            solpos["apparent_zenith"],
            solpos["azimuth"],
            dni,
            ghi,
            dhi,
            dni_extra=dni_extra,
            model="perez",
        )
        cols[name] = total["poa_global"].fillna(0.0).set_axis(index)
    return pd.DataFrame(cols)


def build_geometries(
    location: Location,
    index: pd.DatetimeIndex,
    design: dict,
    step: pd.Timedelta,
    rotation_step: int,
    sky_ratios: pd.DataFrame = None,
):
    """Clear-sky POA (W/m2) per geometry at bin centers, plus the solar position.

    With ``sky_ratios`` (columns ``dni`` and ``dhi``: measured/clear-sky ratio on ``index``) a third
    frame is returned: the POA per geometry under that *real* sky, transposed from separately
    scaled beam and diffuse components. A tracker is beam-dominated and a flat/stowed sensor is
    not, so on hazy days the two respond very differently -- one scalar cloud index cannot capture it.
    """
    center = index + step / 2
    solpos = location.get_solarposition(center)
    clearsky = location.get_clearsky(center, solar_position=solpos)
    dni_extra = irradiance.get_extra_radiation(center)
    orientations = _orientations(solpos, design, rotation_step)
    geoms = _transpose(
        orientations, solpos, clearsky["dni"], clearsky["ghi"], clearsky["dhi"], dni_extra, index
    )
    solpos = solpos.set_axis(index)
    if sky_ratios is None:
        return geoms, solpos
    dni = clearsky["dni"] * sky_ratios["dni"].reindex(index).fillna(1.0).to_numpy()
    dhi = clearsky["dhi"] * sky_ratios["dhi"].reindex(index).fillna(1.0).to_numpy()
    ghi = (dni * np.cos(np.radians(solpos["apparent_zenith"].to_numpy())).clip(min=0) + dhi).where(
        solpos["apparent_zenith"].to_numpy() < 90, 0.0
    )
    return geoms, solpos, _transpose(orientations, solpos.set_axis(center), dni, ghi, dhi, dni_extra, index)


# ---- cloud-index references --------------------------------------------------------------------
def meter_cloud_index(
    meter: pd.Series,
    clearsky: pd.Series,
    daylight: pd.Series,
    step: pd.Timedelta,
    valid: pd.Series = None,
    quantile: float = 0.95,
    bin_width: float = 50.0,
) -> pd.Series:
    """Meter output relative to its clear-sky output (1 = unclipped clear sky), at ``clearsky.index``.

    The clear-sky output curve is the upper envelope of meter vs. clear-sky POA (``quantile`` per
    ``bin_width`` bin), which absorbs plant capacity, clipping and temperature without a plant model.
    """
    meter = _as_tz(meter, clearsky.index).clip(lower=0.0)
    if valid is not None:
        meter = meter.where(_as_tz(valid, meter.index).reindex(meter.index).fillna(False))
    m = meter.resample(step).mean().reindex(clearsky.index)

    sel = daylight & m.notna() & (clearsky > 10)
    if sel.sum() < 100:
        return pd.Series(np.nan, index=clearsky.index)
    bins = (clearsky[sel] // bin_width).astype(int)
    grouped = m[sel].groupby(bins)
    env = grouped.quantile(quantile)[grouped.size() >= 10]
    if len(env) < 3:
        return pd.Series(np.nan, index=clearsky.index)
    expected = pd.Series(
        np.interp(clearsky, (env.index.to_numpy() + 0.5) * bin_width, env.to_numpy()),
        index=clearsky.index,
    )
    return (m / expected.where(expected > 0.02 * expected.max())).clip(0.0, 1.3).where(daylight)


def model_cloud_index(
    csi_dtn: pd.Series,
    csi_meter: pd.Series,
    clearsky: pd.Series,
    ref_freq: pd.Timedelta,
    step: pd.Timedelta,
    min_coverage: float = 0.5,
    trend: Optional[pd.Timedelta] = None,
) -> pd.Series:
    """DTN's cloud index, reshaped within each reference interval by the meter's.

    The level of every interval comes from DTN alone: the meter's *magnitude* (performance, stow,
    haze it cannot tell from clouds) never reaches the model. The meter only redistributes DTN's
    interval mean over the interval's bins -- its cloud index divided by its own clear-sky-weighted
    interval mean, so the interval energy is preserved. Intervals where fewer than ``min_coverage``
    of the daylight bins have a trusted meter value (NaN in ``csi_meter``) keep DTN's flat index.

    With ``trend``, the meter index is first divided by its own rolling median over that window: a
    slow slope (a stowed or drifting plant) would otherwise be re-normalised hour by hour into a
    sawtooth, while passing clouds -- shorter than half the window -- survive.
    """
    if ref_freq <= step:
        return csi_dtn
    if trend is not None:
        win = max(3, _to_bins(trend, step))
        smooth = csi_meter.rolling(win, center=True, min_periods=max(3, win // 4)).median()
        csi_meter = (csi_meter / smooth.where(smooth > 0.05)).clip(0.0, 3.0)
    index = csi_dtn.index
    block = pd.Series((index.asi8 - index[0].value) // ref_freq.value, index=index)
    trusted = csi_meter.notna() & (clearsky > 0)
    weighted = (clearsky * csi_meter).where(trusted, 0.0).groupby(block).transform("sum")
    weight = clearsky.where(trusted, 0.0).groupby(block).transform("sum")
    covered = weight / clearsky.where(clearsky > 0, 0.0).groupby(block).transform("sum").replace(0, np.nan)
    mean_meter = weighted / weight.replace(0, np.nan)
    shape = (csi_meter / mean_meter.where(mean_meter > 0.05)).where(covered >= min_coverage)
    return (csi_dtn * shape.fillna(1.0)).clip(0.0, 1.3)


def meter_level_factor(
    csi_dtn: pd.Series,
    csi_meter: pd.Series,
    kept: pd.Series,
    clearsky: pd.Series,
    default_gain: float,
    ref_freq: pd.Timedelta,
    step: pd.Timedelta,
    tol: float,
    reach: pd.Timedelta,
    min_coverage: float = 0.5,
) -> pd.Series:
    """Factor scaling DTN's cloud index to the meter's level -- where the kept sensors vouch for it.

    DTN's hourly index can miss clouds entirely (it says clear while sensors *and* meter agree it is
    overcast), so its level alone is wrong on such days. The meter's level (its hourly index relative
    to DTN's) is believed only in the neighbourhood (``reach``) of an hour in which the kept sensors
    sit within ``tol`` of it; an hour in which they contradict it (a stowed or drifting meter, with
    sensors still tracking) withdraws the trust, and the nearest verdict in time wins. Elsewhere the
    factor is 1: DTN sets the level. The factor never exceeds 1 -- clouds DTN missed only lower it.
    """
    one = pd.Series(1.0, index=csi_dtn.index)
    if ref_freq <= step:
        return one
    index = csi_dtn.index
    block = pd.Series((index.asi8 - index[0].value) // ref_freq.value, index=index)
    strong = clearsky > 0.2 * clearsky.max()

    def hourly(num, den, mask):
        n = num.where(mask, 0.0).groupby(block).transform("sum")
        d = den.where(mask, 0.0).groupby(block).transform("sum")
        cover = mask.astype(float).groupby(block).transform("sum") / (
            (clearsky > 0).astype(float).groupby(block).transform("sum").replace(0, np.nan)
        )
        return (n / d.replace(0, np.nan)).where(cover >= min_coverage)

    m_ok = csi_meter.notna() & strong
    f_raw = hourly(clearsky * csi_meter, clearsky, m_ok) / hourly(clearsky * csi_dtn, clearsky, m_ok)
    # the meter index is normalised to its own clear-day envelope, DTN's carries the clear-sky model's
    # bias: on clear hours their ratio is not 1, so express it relative to that
    clear = hourly(clearsky * csi_dtn, clearsky, m_ok) > 0.93
    scale = f_raw[clear].median()
    scale = scale if np.isfinite(scale) and scale > 0 else 1.0
    # a meter reading *above* DTN is not a cloud effect (stow, performance): never above 1
    f_m = (f_raw / scale).clip(0.0, 1.0)
    s_ok = kept.notna() & strong
    r_s = hourly(kept, clearsky * csi_dtn * default_gain, s_ok)

    state = pd.Series(np.nan, index=index)  # 1 = sensors corroborate the meter's level, 0 = contradict
    known = f_m.notna() & r_s.notna()
    state[known] = ((r_s - f_m).abs() <= tol)[known].astype(float)

    t = pd.Series(index, index=index)
    t_known = t.where(state.notna())
    prev_t, next_t = t_known.ffill(), t_known.bfill()
    prev_s, next_s = state.ffill(), state.bfill()
    d_prev = (t - prev_t).fillna(pd.Timedelta.max)
    d_next = (next_t - t).fillna(pd.Timedelta.max)
    nearest = prev_s.where(d_prev <= d_next, next_s)
    trusted = (nearest == 1.0) & (np.minimum(d_prev, d_next) <= reach)
    return f_m.where(trusted & f_m.notna(), 1.0)


def reference_cloud_index(ref_poa: pd.Series, clearsky: pd.Series, step: pd.Timedelta) -> pd.Series:
    """Reference POA (e.g. hourly DTN, timestamp = interval start) / its interval-mean clear-sky POA."""
    ref = _as_tz(ref_poa, clearsky.index).clip(lower=0.0)
    ref_freq = infer_freq(ref.index)
    if ref_freq <= step:
        csi = ref.resample(step).mean().reindex(clearsky.index) / clearsky.where(clearsky > 20)
    else:
        cs_ref = clearsky.resample(ref_freq).mean()
        csi_ref = ref / cs_ref.where(cs_ref > 20).reindex(ref.index)
        full = csi_ref.reindex(csi_ref.index.union(clearsky.index)).ffill()
        csi = full.reindex(clearsky.index)
    return csi.clip(0.0, 1.3)


# ---- per-sensor assessment ---------------------------------------------------------------------
def _csi_for(csi, geom):
    """Cloud index for one geometry: a Series applies to every geometry, a DataFrame is per geometry
    (columns = geometry names, as returned by ``build_geometries``)."""
    return csi[geom] if isinstance(csi, pd.DataFrame) else csi


def sensor_level(x, ref, csi, clearsky, use, min_days=5, min_csi=0.93, min_peak=0.6) -> float:
    """Sensor / reference POA level: median over sunny days of the daily median ratio.

    Only near-peak bins on days when the reference index says the sky is clear count, so clouds and
    sun-angle effects do not move it. ``x`` should hold trusted (tracking) data only. NaN when fewer
    than ``min_days`` days qualify. Healthy sensors sit near 1 (DTN's transposition has its own bias)."""
    sel = use & x.notna() & (ref > 0) & (csi > min_csi) & (clearsky > min_peak * clearsky.max())
    if not sel.any():
        return np.nan
    daily = (x[sel] / ref[sel]).groupby(x.index[sel].date).median()
    return float(daily.median()) if len(daily) >= min_days else np.nan


def estimate_gain(x, geometries, csi, use, cloud_var, bounds=(0.8, 1.25), min_bins=36) -> float:
    """Calibration factor vs. the clear-sky model.

    For each day, fit a free scale against every geometry and keep the best fit whose scale is
    plausible; the gain is the median over low-residual, steady-sky days. Fitting per day against
    the best geometry keeps the estimate sound for sensors that are stowed much of the time.
    """
    gains, quality = [], []
    for _, idx in x.groupby(x.index.date).groups.items():
        m = (use.loc[idx] & x.loc[idx].notna() & _csi_for(csi, TRACKER).loc[idx].notna()).to_numpy()
        if m.sum() < min_bins:
            continue
        xd = x.loc[idx].to_numpy()[m]
        best = None
        for geom in geometries.columns:
            e = (geometries[geom].loc[idx] * _csi_for(csi, geom).loc[idx]).to_numpy()[m]
            denom = (e * e).sum()
            if denom <= 0:
                continue
            g = (xd * e).sum() / denom
            if not bounds[0] <= g <= bounds[1]:
                continue
            resid = np.abs(xd - g * e).sum() / xd.sum()
            if best is None or resid < best[1]:
                best = (g, resid)
        if best:
            gains.append(best[0])
            quality.append((best[1], cloud_var.loc[idx].to_numpy()[m].mean()))
    if not gains:
        return 1.0
    gains, quality = np.array(gains), np.array(quality)
    steady = (quality[:, 0] < 0.1) & (quality[:, 1] < 0.15)
    return float(np.median(gains[steady] if steady.any() else gains))


def _fill_ambiguous(
    lab: pd.Series, max_gap: pd.Timedelta, tracking_fits: pd.Series = None
) -> pd.Series:
    """Resolves 'amb' windows (stow and tracking indistinguishable).

    Where tracking also fits (``tracking_fits``), the sensor is behaving: the window is 'stow' only
    when stow bounds it on *both* sides within ``max_gap`` (e.g. midday of a stowed day), else 'ok'
    -- a stow starting later must not leak backward over a healthy sensor. Where tracking does not
    fit, the window takes the label of the nearest 'ok'/'stow' window within ``max_gap``, else 'ok'.
    """
    known = lab.where(lab.isin(["ok", "stow"]))
    t = pd.Series(lab.index, index=lab.index)
    t_known = t.where(known.notna())
    prev_t, next_t = t_known.ffill(), t_known.bfill()
    prev_l, next_l = known.ffill(), known.bfill()
    d_prev, d_next = t - prev_t, next_t - t
    use_prev = (d_prev <= d_next) | next_l.isna()
    cand = prev_l.where(use_prev, next_l)
    dist = d_prev.where(use_prev, d_next)
    is_amb = lab == "amb"
    take = is_amb & cand.notna() & (dist <= max_gap)
    out = lab.copy()
    out[take] = cand[take]
    out[is_amb & ~take] = "ok"
    if tracking_fits is not None:
        bounded = (prev_l == "stow") & (next_l == "stow") & (d_prev <= max_gap) & (d_next <= max_gap)
        behaving = is_amb & tracking_fits.reindex(lab.index, fill_value=False)
        out[behaving] = np.where(bounded[behaving], "stow", "ok")
    return out


def _drop_short_runs(lab: pd.Series, value: str, min_bins: int, replace: str = "ok") -> pd.Series:
    run_id = (lab != lab.shift()).cumsum()
    short = lab.groupby(run_id).transform("size") < min_bins
    return lab.mask((lab == value) & short, replace)


def _bridge_stow_gaps(lab: pd.Series, max_bins: int) -> pd.Series:
    """'ok' runs shorter than max_bins sandwiched between 'stow' runs become 'stow'.
    (Around solar noon a stowed sensor and the tracking geometry nearly coincide.)"""
    run_id = (lab != lab.shift()).cumsum()
    runs = lab.groupby(run_id).agg(["first", "size"])
    out = lab.copy()
    for i in runs.index[1:-1]:
        if (
            runs.at[i, "first"] == "ok"
            and runs.at[i, "size"] <= max_bins
            and runs.at[i - 1, "first"] == "stow"
            and runs.at[i + 1, "first"] == "stow"
        ):
            out[run_id == i] = "stow"
    return out


def _enforce_single_geometry(
    lab: pd.Series,
    score: pd.DataFrame,
    max_score: float,
    afternoon: pd.Series = None,
    min_bins: int = 1,
):
    """A stuck tracker holds one angle, so each stow run must fit a single fixed geometry.

    Each contiguous stow run is judged separately for the morning and afternoon (split at solar
    noon, ``afternoon`` = True after it) so a good half cannot hide a bad one. A half takes the
    geometry with the lowest mean score; if even that exceeds ``max_score`` (e.g. a fouled sensor
    that matches different rotations at different times of day) the half becomes 'bad'. A half
    shorter than ``min_bins`` is too short to judge alone and follows the other half of its run.
    Returns (labels, per-bin stow geometry).
    """
    geometry = pd.Series(None, index=lab.index, dtype=object)
    if score.shape[1] < 2:
        return lab, geometry
    fixed = score.drop(columns=TRACKER)
    is_stow = lab == "stow"
    run_id = (lab != lab.shift()).cumsum()
    half = afternoon if afternoon is not None else pd.Series(False, index=lab.index)
    out = lab.copy()

    for _, run in lab[is_stow].groupby(run_id[is_stow]).groups.items():
        parts = {h: run[half.loc[run].to_numpy() == h] for h in (False, True)}
        parts = {h: idx for h, idx in parts.items() if len(idx)}
        verdict = {}
        for h, idx in parts.items():
            if len(idx) >= min_bins or len(parts) == 1:
                mean = fixed.loc[idx].mean()
                verdict[h] = mean.idxmin() if mean.min() <= max_score else None
        for h, idx in parts.items():
            v = verdict[h] if h in verdict else next(iter(verdict.values()))  # short half follows
            if v is None:
                out.loc[idx] = "bad"
            else:
                geometry.loc[idx] = v
    return out, geometry


def _trim_stow_transitions(
    lab: pd.Series, geometry: pd.Series, deviates: pd.Series, max_bins: int
):
    """Extend each stow run into adjacent 'ok' bins while the sensor still reads below tracking.

    A tracker entering or leaving stow takes a while to settle onto the tracking curve; sensor data
    in that transition is biased relative to ideal tracking, so it belongs to the stow. Walks
    outward from each run edge, up to ``max_bins``, stopping at the first bin that is not 'ok' or
    that has converged (``deviates`` False or unknown). Extended bins take the run's geometry.
    """
    out, geom = lab.copy(), geometry.copy()
    is_stow = (lab == "stow").to_numpy()
    ok = (lab == "ok").to_numpy()
    dev = deviates.fillna(False).to_numpy(bool)
    n = len(lab)
    edges_after = np.flatnonzero(is_stow[:-1] & ~is_stow[1:])  # last stow bin of each run
    edges_before = np.flatnonzero(~is_stow[:-1] & is_stow[1:]) + 1  # first stow bin of each run
    for edges, direction in ((edges_after, 1), (edges_before, -1)):
        for edge in edges:
            g = geometry.iloc[edge]
            j = edge + direction
            for _ in range(max_bins):
                if not (0 <= j < n) or not (ok[j] and dev[j]):
                    break
                out.iloc[j] = "stow"
                geom.iloc[j] = g
                j += direction
    return out, geom


def _shape_stow_halves(
    x: pd.Series,
    geometries: pd.DataFrame,
    use: pd.Series,
    cfg: POAQCConfig,
    afternoon: pd.Series = None,
):
    """Half-days (morning / afternoon, split at solar noon) on which the sensor's *shape* says it is
    stuck at a fixed geometry, independent of any cloud reference.

    A tracker's output on a steady day is a flat-topped curve; a stuck or stowed sensor's is the
    clear-sky curve of its fixed geometry (a bell for a flat one). Per half-day, each geometry's
    clear-sky curve is fit with a free scale; the half is a shape-stow when the sensor is smooth (a
    steady sky -- clouds would make any shape fit meaningless), the best fixed geometry fits tightly
    with a plausible scale, and tracking clearly does not fit. DTN's hourly cloud index is uncertain
    by about the same amount as the stow-vs-tracking difference, so this evidence supersedes it.
    Halves are judged separately so a sensor stuck in the morning that tracks again in the afternoon
    only gets the morning. Returns (bool Series over x.index, per-bin best geometry name or None).
    """
    flag = pd.Series(False, index=x.index)
    geom = pd.Series(None, index=x.index, dtype=object)
    if geometries.shape[1] < 2:
        return flag, geom
    step = pd.Timedelta(cfg.analysis_freq)
    clearsky = geometries[TRACKER]
    sel_all = use & x.notna() & (clearsky > 0.3 * clearsky.max())
    min_bins = int(round(cfg.shape_min_hours * pd.Timedelta(hours=1) / step))
    half = afternoon if afternoon is not None else pd.Series(False, index=x.index)
    keys = [x.index.date, half.to_numpy()]
    for _, idx in x.groupby(keys).groups.items():
        sel = sel_all.loc[idx]
        if sel.sum() < min_bins:
            continue
        xs, cs = x.loc[idx][sel], clearsky.loc[idx][sel]
        if (xs / cs).diff().abs().median() > cfg.shape_max_roughness:
            continue  # clouds: shape fit is meaningless
        xv, ref = xs.to_numpy(), cs.mean()
        fits = {}
        for g in geometries.columns:
            e = geometries[g].loc[idx][sel].to_numpy()
            k = (xv * e).sum() / (e * e).sum()
            fits[g] = (np.sqrt(((xv - k * e) ** 2).mean()) / ref, k)
        track = fits[TRACKER][0]
        fixed = {
            g: v for g, v in fits.items() if g != TRACKER and cfg.gain_bounds[0] <= v[1] <= cfg.gain_bounds[1]
        }
        if not fixed:
            continue
        best = min(fixed, key=lambda g: fixed[g][0])
        rmse = fixed[best][0]
        if (
            rmse <= cfg.shape_max_rmse
            and rmse <= cfg.shape_max_ratio * track
            and track >= cfg.shape_min_track_rmse
        ):
            flag.loc[idx] = True
            geom.loc[idx] = best
    return flag, geom


def classify_sensor(
    x: pd.Series,
    geometries: pd.DataFrame,
    refs: dict,
    use: pd.Series,
    cfg: POAQCConfig,
    gain: float,
    afternoon: pd.Series = None,
) -> dict:
    """Label each analysis bin of one sensor: 'ok' | 'stow' | 'bad' | '' (not assessable).

    ``refs`` maps reference name -> (cloud index, tolerance function ``f(sensor_hf) -> Series``); the
    cloud index is a Series (same for every geometry) or a DataFrame with one column per geometry.
    """
    step = pd.Timedelta(cfg.analysis_freq)
    win, smooth = _to_bins(cfg.window, step), _to_bins(cfg.smooth, step)
    roll = lambda s: s.rolling(win, center=True, min_periods=max(1, win // 2)).mean()
    sm = lambda s: s.rolling(smooth, center=True, min_periods=1).mean()

    clearsky = geometries[TRACKER]
    cs_win = roll(clearsky.where(use))
    xs = sm(x.where(use))
    ratio_x = (x / clearsky.where(clearsky > 0)).where(use)
    hf = roll(ratio_x.diff().abs()).fillna(0.0)
    curvature = roll(ratio_x.diff().diff().abs()).fillna(0.0)  # ~0 for a smooth bell or plateau

    # error of each geometry vs. each reference, in units of that reference's tolerance;
    # a geometry's score is its error against the *better* reference
    score = None
    for ref_name, (csi, tol_fn) in refs.items():
        err = pd.DataFrame(
            {
                g: roll(
                    (xs - sm(geometries[g] * _csi_for(csi, g) * gain))
                    .abs()
                    .where(use & xs.notna() & _csi_for(csi, g).notna())
                )
                / cs_win
                for g in geometries.columns
            }
        )
        s_ref = err.div(tol_fn(hf), axis=0)
        score = s_ref if score is None else s_ref.where(s_ref.le(score) | score.isna(), score)

    # Shape evidence (independent of the cloud references) supersedes their scores on shape-stow days:
    # tracking is forced to misfit and the best fixed geometry to fit, so the usual machinery decides
    # which bins are distinguishable (a stuck sensor that matches tracking by afternoon stays ok).
    trk_ref = score[TRACKER].copy()  # tracking's fit to the references, before any shape override
    shape_day, shape_geom = _shape_stow_halves(x, geometries, use, cfg, afternoon)
    if shape_day.any() and score.shape[1] > 1:
        on = shape_day & use & x.notna()
        score.loc[on, TRACKER] = np.maximum(score.loc[on, TRACKER].fillna(0.0), 2 * cfg.tracker_fail_score)
        for g in shape_geom[on].dropna().unique():
            gon = on & shape_geom.eq(g)
            score.loc[gon, g] = np.minimum(score.loc[gon, g].fillna(0.0), 0.5)
            # the geometry the shape test chose must be the best fixed one (a near neighbour that
            # also fits a hidden-in-DTN stow would be indistinguishable from tracking)
            others = [c for c in score.columns if c not in (TRACKER, g)]
            score.loc[gon, others] = np.maximum(score.loc[gon, others].fillna(10.0), 1.0)

    trk = score[TRACKER]
    valid = trk.notna()
    trk_ok = trk <= cfg.tracker_fail_score
    lab = pd.Series("", index=x.index, dtype=object)

    if score.shape[1] > 1:
        fixed = score.drop(columns=TRACKER)
        best_idx = fixed.to_numpy().argmin(axis=1)
        best_score = pd.Series(fixed.to_numpy()[np.arange(len(fixed)), best_idx], index=x.index)
        diff = pd.DataFrame(
            {g: roll((geometries[g] - clearsky).abs().where(use)) / cs_win for g in fixed.columns}
        )
        best_diff = pd.Series(diff.to_numpy()[np.arange(len(diff)), best_idx], index=x.index)
        fits_fixed = best_score <= 1.0
        distinct = best_diff >= cfg.min_discrimination
        # a stowed sensor still sees sky: reading ~nothing means dead/covered/shaded, not stow
        # (judged against the more lenient reference, like the scores)
        ratio = pd.concat(
            [
                roll(xs) / roll((clearsky * _csi_for(csi, TRACKER) * gain).where(use))
                for csi, _ in refs.values()
            ],
            axis=1,
        ).max(axis=1)
        plausible = ratio >= cfg.min_stow_ratio
        is_stow = ~trk_ok & fits_fixed & distinct & plausible & (best_score + cfg.stow_margin < trk)
        ambiguous = fits_fixed & ~distinct
    else:
        is_stow = pd.Series(False, index=x.index)
        ambiguous = pd.Series(False, index=x.index)

    lab[valid & trk_ok] = "ok"
    lab[valid & ambiguous] = "amb"  # stow and tracking indistinguishable here
    lab[valid & is_stow] = "stow"
    lab[valid & (lab == "")] = "bad"
    # Low-sun shoulders cannot support a 'bad' verdict: there the comparison is dominated by model
    # and plane details (a sensor can sit well above both references near sunset), not faults.
    # 'bad' needs absolute signal and a window carrying a real share of that day's peak.
    day_peak = cs_win.groupby(cs_win.index.date).transform("max")
    weak = (cs_win < 4 * cfg.min_signal) | (cs_win < cfg.bad_min_peak_frac * day_peak)
    lab[(lab == "bad") & weak] = "ok"

    # where stow and tracking read alike the sensor measures what ideal tracking would, so judge
    # 'tracking fits' by the references, not by the shape override
    lab = _fill_ambiguous(lab, pd.Timedelta(cfg.ambiguity_fill), valid & (trk_ref <= 1.0))
    lab = lab.where(valid, "")
    lab = _drop_short_runs(lab, "bad", _to_bins(cfg.min_bad_duration, step))
    lab = _drop_short_runs(lab, "stow", _to_bins(cfg.min_stow_duration, step))
    lab = _bridge_stow_gaps(
        lab.where(lab != "", "none"), _to_bins(cfg.bridge_stow_gap, step)
    ).replace("none", "")
    lab, geometry = _enforce_single_geometry(
        lab, score, cfg.max_stow_score, afternoon, _to_bins(cfg.min_stow_duration, step)
    )
    lab = _drop_short_runs(lab, "bad", _to_bins(cfg.min_bad_duration, step))

    # sensor/expected-tracking ratio (against whichever reference it agrees with more)
    expected = pd.concat(
        [sm(clearsky * _csi_for(csi, TRACKER) * gain) for csi, _ in refs.values()], axis=1
    )
    ratios = expected.rdiv(sm(x.where(use)), axis=0).where(expected > 2 * cfg.min_signal)
    pick = (ratios - 1).abs().to_numpy()
    pick = np.where(np.isnan(pick), np.inf, pick).argmin(axis=1)
    ratio = pd.Series(ratios.to_numpy()[np.arange(len(ratios)), pick], index=x.index)
    ratio = ratio.rolling(6, center=True, min_periods=3).median()
    # tracking maximizes plane-of-array irradiance, so an unsettled/stowed sensor reads *below* it;
    # reading above expectation is reference error, never a transition
    deviates = (ratio - 1) < -cfg.stow_transition_tol
    lab, geometry = _trim_stow_transitions(
        lab, geometry, deviates, _to_bins(cfg.stow_transition_max, step)
    )

    # Hourly DTN cannot resolve passing clouds, so where the minute-scale meter is unavailable a rough
    # (cloudy) window cannot support a 'stow' or 'bad' verdict -- a healthy tracker would be flagged
    # for disagreeing with a reference that cannot see what the sky did. (Shape-confirmed stow is
    # smooth by construction and exempt.) Such bins stay unflagged: reported as 'inconclusive'.
    if "meter" in refs and cfg.dtn_only_max_curvature is not None:
        no_meter = _csi_for(refs["meter"][0], TRACKER).isna()
        blind = use & no_meter & (curvature > cfg.dtn_only_max_curvature) & ~shape_day
        lab = lab.mask(blind & lab.isin(["stow", "bad"]), "")
        geometry = geometry.mask(lab == "")
    return dict(label=lab, geometry=geometry, score=score, gain=gain)


WEATHER_COLUMNS = {  # DTN column names used when present in ``ref_weather``
    "wind": "dtn_wind_speed",
    "sunshine": "sunshineDuration",
    "cloud": "effectiveCloudCover",
    "precip": "precipAmount",
}


def _to_grid(series: pd.Series, index: pd.DatetimeIndex, step: pd.Timedelta) -> pd.Series:
    """Series on the analysis grid. Interval-start timestamps: coarser data is held over its
    interval (an unknown value stays unknown), finer data is averaged."""
    series = _as_tz(series, index).astype(float)
    freq = infer_freq(series.index)
    if freq <= step:
        return series.resample(step).mean().reindex(index)
    return series.reindex(index, method="ffill", tolerance=freq - pd.Timedelta(1, "ns"))


def weather_on_grid(weather: pd.DataFrame, index: pd.DatetimeIndex, step: pd.Timedelta) -> pd.DataFrame:
    """Recognized weather columns on the analysis grid (see ``_to_grid``)."""
    cols = {k: c for k, c in WEATHER_COLUMNS.items() if c in weather.columns}
    if not cols:
        raise ValueError(f"ref_weather has none of the expected columns {list(WEATHER_COLUMNS.values())}.")
    return pd.DataFrame({key: _to_grid(weather[col], index, step) for key, col in cols.items()})


SKY_COLUMNS = ("dtn_dni", "dtn_dhi", "dni_clearsky", "dhi_clearsky")  # DTN beam/diffuse + clear-sky


def sky_ratios(weather: pd.DataFrame, index: pd.DatetimeIndex, step: pd.Timedelta):
    """DTN measured/clear-sky ratio of beam (``dni``) and diffuse (``dhi``) on the analysis grid,
    or None unless ``weather`` has all of ``SKY_COLUMNS``. NaN where the clear-sky value is too small
    for a meaningful ratio (low sun); the scalar cloud index is used there."""
    if weather is None or not all(c in weather.columns for c in SKY_COLUMNS):
        return None
    dni = (weather["dtn_dni"] / weather["dni_clearsky"].where(weather["dni_clearsky"] > 20)).clip(0, 1.3)
    dhi = (weather["dtn_dhi"] / weather["dhi_clearsky"].where(weather["dhi_clearsky"] > 5)).clip(0, 3.0)
    return pd.DataFrame({"dni": _to_grid(dni, index, step), "dhi": _to_grid(dhi, index, step)})


def weather_flags(weather: pd.DataFrame, cfg: POAQCConfig):
    """(windy, stormy) bool series, each None when the needed columns are absent.

    ``windy``: wind >= ``stow_min_wind`` within ``weather_lookaround``. ``stormy``: precipitation, low
    sunshine (or, lacking it, high cloud cover) within ``weather_lookaround``. Unknown values never
    count as evidence either way: windy is True and stormy False where the data is missing.
    """
    look = _to_bins(cfg.weather_lookaround, pd.Timedelta(cfg.analysis_freq))
    near = lambda s, how: getattr(s.rolling(2 * look + 1, center=True, min_periods=1), how)()
    windy = stormy = None
    if "wind" in weather:
        peak = near(weather["wind"], "max")
        windy = (peak >= cfg.stow_min_wind) | peak.isna()
    evidence = []
    if "precip" in weather:
        evidence.append(near(weather["precip"], "max") >= cfg.storm_min_precip)
    if "sunshine" in weather:
        evidence.append(near(weather["sunshine"], "min") < cfg.storm_max_sunshine)
    elif "cloud" in weather:
        evidence.append(near(weather["cloud"], "max") > cfg.storm_min_cloud)
    if evidence:
        stormy = pd.concat(evidence, axis=1).any(axis=1)
    return windy, stormy


def _robust_roughness(x: pd.Series) -> float:
    """Median absolute 1-sample step relative to the median level; NaN if it cannot be judged."""
    x = x.dropna()
    level = x.median() if len(x) > 10 else np.nan
    if not level > 0:
        return np.nan
    return float(x.diff().abs().median() / level)


def _smooth_runs(mask: pd.Series, native: pd.Series, step: pd.Timedelta, max_roughness: float):
    """True on runs of ``mask`` whose native-resolution series is smooth (see detect_meter_stow).
    A run that cannot be judged (too few samples, zero output) is not smooth."""
    keep = pd.Series(False, index=mask.index)
    run_id = (mask != mask.shift()).cumsum()
    for _, idx in mask[mask].groupby(run_id[mask]).groups.items():
        segment = native.loc[idx[0] : idx[-1] + step - pd.Timedelta(1, "ns")]
        if _robust_roughness(segment) <= max_roughness:  # NaN compares False
            keep.loc[idx] = True
    return keep


def detect_meter_stow(
    meter: pd.Series,
    geometries: pd.DataFrame,
    csi_dtn,
    use: pd.Series,
    cfg: POAQCConfig,
    afternoon: pd.Series,
    valid: pd.Series = None,
    windy: pd.Series = None,
    stormy: pd.Series = None,
) -> pd.Series:
    """True where the plant meter looks stowed: its output follows a fixed-rotation geometry.

    Plant output follows the same geometry as irradiance, so the meter is scaled to POA-like units
    and judged like a sensor -- against the (plant-independent) DTN reference only, since the
    meter's own cloud index is what a stowed plant would corrupt. Returns False everywhere when
    the plant is not a tracker or there is too little meter data to judge.

    A stowed plant's output is a smooth curve (a fixed geometry times the sky), while passing clouds
    make it jagged at 1 minute. Each run must therefore be smooth: its robust roughness (median
    1-minute step / median level of the native meter) must be <= ``cfg.stow_max_roughness``; clear
    days sit near 0.0005 and cloudy ones at 0.02-0.06. A jagged collapse is treated as real sky --
    the meter, like the sensors, is then reporting the true POA conditions -- even when DTN (hourly)
    shows sun. Bins with ``stormy`` weather (precipitation, low sunshine) are never stow either.
    ``windy`` (passed only when ``cfg.stow_requires_wind``) additionally requires wind somewhere
    within each run. Missing weather inputs are skipped. Single-sensor stow (a stuck tracker) is
    not weather- or roughness-gated.
    """
    index = geometries.index
    never = pd.Series(False, index=index)
    if geometries.shape[1] < 2:
        return never
    step = pd.Timedelta(cfg.analysis_freq)
    meter = _as_tz(meter, index).clip(lower=0.0)
    if valid is not None:
        meter = meter.where(_as_tz(valid, meter.index).reindex(meter.index).fillna(False))
    m = meter.resample(step).mean().reindex(index)
    clearsky = geometries[TRACKER]
    sel = use & m.notna() & (m > 0)
    if sel.sum() < 100:
        return never
    scale = m[sel].quantile(0.95) / clearsky[sel].quantile(0.95)
    if not np.isfinite(scale) or scale <= 0:
        return never
    x = m / scale

    win = _to_bins(cfg.window, step)
    dtn_var = (
        _csi_for(csi_dtn, TRACKER)
        .where(use)
        .rolling(win, center=True, min_periods=max(1, win // 2))
        .std()
    )
    gain = estimate_gain(x, geometries, csi_dtn, use, dtn_var.fillna(0.0), bounds=(0.4, 2.5))
    refs = {"dtn": (csi_dtn, lambda hf: cfg.tol + cfg.dtn_tol_factor * hf)}
    label = classify_sensor(x, geometries, refs, use, cfg, gain, afternoon)["label"]
    stow = label.eq("stow")
    if stormy is not None:
        stow &= ~stormy.reindex(stow.index, fill_value=False)
    stow &= _smooth_runs(stow, meter, step, cfg.stow_max_roughness)
    if windy is not None:
        run_id = (stow != stow.shift()).cumsum()
        stow &= windy.reindex(stow.index, fill_value=True).groupby(run_id).transform("any")
    return stow


def assess_poa_sensors(
    sensors: pd.DataFrame,
    ref_poa: pd.Series,
    location: Location,
    design: dict,
    ref_meter: pd.Series = None,
    meter_valid: pd.Series = None,
    config: POAQCConfig = None,
    ref_weather: pd.DataFrame = None,
) -> dict:
    """Assess every column of ``sensors`` (native resolution, tz-aware) against the references.

    Returns dict with ``analysis`` (frame on the analysis grid: sensors, geometries, cloud indexes,
    ``use`` mask), ``solpos`` and ``sensors`` (col -> classify_sensor output + labels incl. missing).
    """
    cfg = config or POAQCConfig()
    step = pd.Timedelta(cfg.analysis_freq)
    if sensors.index.tz is None:
        raise TypeError("POA sensor data must have a timezone-aware DatetimeIndex.")

    x = sensors.clip(lower=0.0).resample(step).mean()
    sky = sky_ratios(ref_weather, x.index, step)  # None unless DTN beam/diffuse are supplied
    if sky is None:
        geoms, solpos = build_geometries(location, x.index, design, step, cfg.stow_rotation_step)
        geoms_sky = None
    else:
        geoms, solpos, geoms_sky = build_geometries(
            location, x.index, design, step, cfg.stow_rotation_step, sky_ratios=sky
        )
    clearsky = geoms[TRACKER]
    use = (solpos["apparent_zenith"] < cfg.max_zenith) & (clearsky > cfg.min_signal)

    # after local solar noon (time of peak elevation each day)
    elevation = solpos["apparent_elevation"]
    noon = elevation.groupby(elevation.index.date).transform(lambda e: e.idxmax())
    afternoon = pd.Series(x.index >= noon.to_numpy(), index=x.index)

    csi_dtn = reference_cloud_index(ref_poa, clearsky, step)
    csi_dtn_geom = None  # per-geometry DTN index (haze-aware), when beam/diffuse are available
    if geoms_sky is not None:
        known = sky.notna().all(axis=1)
        csi_dtn_geom = (geoms_sky / geoms.where(geoms > 10)).clip(0, 1.5)
        for g in csi_dtn_geom.columns:  # scalar index where the sky ratios or geometry are unusable
            csi_dtn_geom[g] = csi_dtn_geom[g].where(known & csi_dtn_geom[g].notna(), csi_dtn)
        csi_dtn = csi_dtn_geom[TRACKER]
    weather = None if ref_weather is None else weather_on_grid(ref_weather, x.index, step)
    windy, stormy = (None, None) if weather is None else weather_flags(weather, cfg)
    csi_meter = pd.Series(np.nan, index=x.index)
    meter_stow = pd.Series(False, index=x.index)
    if ref_meter is not None:
        # a stowed plant's output no longer tracks the cloud-free tracking model, so stowed periods
        # are dropped from the meter reference (and from its clear-sky envelope)
        meter_stow = detect_meter_stow(
            ref_meter,
            geoms,
            csi_dtn if csi_dtn_geom is None else csi_dtn_geom,
            use,
            cfg,
            afternoon,
            meter_valid,
            windy if cfg.stow_requires_wind else None,
            stormy,
        )
        meter_index = _as_tz(ref_meter, x.index).index
        valid = (~meter_stow).reindex(meter_index, method="ffill").fillna(True).astype(bool)
        if meter_valid is not None:
            valid &= _as_tz(meter_valid, x.index).reindex(meter_index).fillna(False).astype(bool)
        csi_meter = meter_cloud_index(
            ref_meter,
            clearsky,
            use | (clearsky > 10),
            step,
            valid,
            cfg.meter_envelope_quantile,
        )

    # A clipped/curtailed meter only says irradiance is *at least* the clipping level, so its index
    # is censored near 1: there, cap it with the DTN index (below clipping the meter is kept as-is,
    # which preserves its 1-minute view of passing clouds)
    censored = csi_meter >= CLIPPED_CSI
    csi_meter = csi_meter.where(~censored, pd.concat([csi_meter, csi_dtn], axis=1).min(axis=1))

    # primary cloud index (gain fit, variability, backfill): meter unless it reads dead while DTN
    # says it is sunny (outage), then DTN
    meter_dead = (csi_meter < 0.1) & (csi_dtn > 0.6) & (clearsky > 300)
    csi_primary = csi_meter.where(~meter_dead).fillna(csi_dtn)
    # a bin beside a dead run mixes dead and live minutes, and a plant coming back climbs for a while
    # (a restart ramp) before settling: neither is a trustworthy cloud index. 2 bins before a dead
    # run, 8 after it.
    dead = meter_dead.astype(float)
    near_dead = (dead.rolling(9, min_periods=1).max() > 0) | (
        dead[::-1].rolling(3, min_periods=1).max()[::-1] > 0
    )
    csi_model = model_cloud_index(
        csi_dtn,
        csi_meter.where(~near_dead),
        clearsky,
        infer_freq(ref_poa.index),
        step,
        trend=None if cfg.shape_trend_window is None else pd.Timedelta(cfg.shape_trend_window),
    )
    win = _to_bins(cfg.window, step)
    cloud_var = (
        csi_primary.where(use)
        .rolling(win, center=True, min_periods=max(1, win // 2))
        .std()
        .fillna(0.0)
    )
    meter_var = (
        csi_meter.where(use)
        .rolling(win, center=True, min_periods=max(1, win // 2))
        .std()
        .fillna(0.0)
    )
    if csi_dtn_geom is None:
        expand = lambda c: c  # one cloud index for every geometry
    else:
        # the meter and primary indexes are tracker-level; other geometries differ by DTN's haze factor
        rel = csi_dtn_geom.div(csi_dtn.where(csi_dtn > 0.05), axis=0).clip(0.2, 5.0).fillna(1.0)
        expand = lambda c: rel.mul(c, axis=0)
    refs = {
        "meter": (expand(csi_meter), lambda hf: cfg.tol + cfg.cloud_tol_factor * meter_var),
        "dtn": (
            csi_dtn if csi_dtn_geom is None else csi_dtn_geom,
            lambda hf: cfg.tol + cfg.dtn_tol_factor * hf,
        ),
    }
    csi_primary_geom = expand(csi_primary)

    ref_grid = _to_grid(ref_poa, x.index, step).clip(lower=0.0)
    out = {}
    for col in x.columns:
        gain = estimate_gain(x[col], geoms, csi_primary_geom, use, cloud_var, cfg.gain_bounds)
        res = classify_sensor(x[col], geoms, refs, use, cfg, gain, afternoon)
        lab = res["label"]

        # sensor gaps / unassessable daylight / out-of-range calibration
        gap = x[col].isna() & (use | (clearsky > cfg.min_signal))
        long_gap = gap & (_run_lengths(gap) >= _to_bins(cfg.min_bad_duration, step))
        lab = lab.mask(long_gap, "missing")
        lab = lab.mask((lab == "") & use & x[col].notna(), "inconclusive")
        if cfg.flag_gain_bounds and not cfg.flag_gain_bounds[0] <= gain <= cfg.flag_gain_bounds[1]:
            lab = lab.mask(use & ~lab.isin(["missing"]), "calibration")
        # magnitude vs. the (plant-independent) DTN POA: two sensors can agree with each other and
        # the meter's shape yet both read low (soiling, miscalibration)
        res["level"] = sensor_level(
            x[col].where(lab.eq("ok")), ref_grid, csi_dtn, clearsky, use, cfg.level_min_days
        )
        if cfg.level_bounds and np.isfinite(res["level"]):
            if not cfg.level_bounds[0] <= res["level"] <= cfg.level_bounds[1]:
                lab = lab.mask(use & ~lab.isin(["missing"]), "calibration")
        res["label"] = lab
        out[col] = res

    analysis = x.join(geoms.add_prefix("cs_")).assign(
        csi_meter=csi_meter,
        csi_dtn=csi_dtn,
        csi_primary=csi_primary,
        csi_model=csi_model,
        primary_is_dtn=csi_meter.isna() | meter_dead,
        use=use,
        meter_stow=meter_stow,
    )
    if weather is not None:
        analysis = analysis.join(weather.add_prefix("wx_"))
        if windy is not None:
            analysis["windy"] = windy
        if stormy is not None:
            analysis["stormy"] = stormy
    return dict(analysis=analysis, solpos=solpos, sensors=out)


# ---- flag mapping / removal --------------------------------------------------------------------
def _fill_short_gaps(series: pd.Series, max_gap, freq: pd.Timedelta):
    """Time-interpolate each NaN run of at most ``max_gap`` that has data on both sides.

    Returns (filled series, bool Series of the bins that were filled). Longer runs are untouched.
    ``max_gap`` of None or zero fills nothing.
    """
    nothing = pd.Series(False, index=series.index)
    if max_gap is None or pd.Timedelta(max_gap) <= pd.Timedelta(0):
        return series.copy(), nothing
    limit = _to_bins(max_gap, freq)
    isna = series.isna()
    run_id = (isna != isna.shift()).cumsum()
    short = isna & (isna.groupby(run_id).transform("size") <= limit)
    filled = series.interpolate(method="time", limit_area="inside")
    out = series.where(~short, filled)
    return out, short & out.notna()


def _labels_to_native(
    lab: pd.Series, native_index: pd.DatetimeIndex, step: pd.Timedelta
) -> pd.Series:
    """Analysis-bin labels (timestamp = bin start) -> native timestamps."""
    return lab.reindex(native_index, method="ffill", tolerance=step - pd.Timedelta(1, "ns")).fillna(
        ""
    )


def _remove_flagged_ranges(df_native: pd.DataFrame, df_flags: pd.DataFrame, flags_to_remove):
    flags_to_remove = set(flags_to_remove)
    freq = infer_freq(df_native.index)
    df = df_native.copy()
    removed = {}
    for col in df_native.columns:
        is_flagged = df_flags[f"{col}_FLAG"].isin(flags_to_remove)
        df.loc[is_flagged, col] = np.nan
        removed[col] = _get_continuous_ranges(df[is_flagged], freq)
    return df, removed


class POADataset:
    """Assess POA sensor quality against irradiance and plant-meter references.

    Parameters
    ----------
    site : str
        Solar site name (location and tracker design come from the site database).
    dataframe : pd.DataFrame
        Sensor data with timezone-aware DatetimeIndex; ``OE.POA_*`` columns are used.
    ref_poa : pd.Series
        Reference POA (transposed DTN GHI); interval-start timestamps, typically hourly.
    ref_meter : pd.Series, optional
        Plant meter output (any units). Resolves passing clouds that the hourly reference averages
        away; sensors are only penalized when they disagree with *both* references.
    meter_valid_mask : pd.Series, optional
        Boolean series (indexed like ``ref_meter``): True where the meter is usable as a reference
        (e.g. not curtailed). Without it the DTN reference covers meter problems.
    ref_weather : pd.DataFrame, optional
        Hourly DTN weather (interval-start index); recognized columns: ``dtn_wind_speed`` (m/s),
        ``sunshineDuration`` (min/h), ``effectiveCloudCover`` (%), ``precipAmount``; others are
        ignored, so the full DTN frame can be passed. A plant-wide stow seen in the meter is never
        called in precipitation or low sunshine (storms look the same in the meter); wind is
        reported by ``meter_stow_summary`` and only required if ``config.stow_requires_wind``.

    After construction: ``df_flags`` (native resolution, one ``<sensor>_FLAG`` column each with
    "", "stow", "bad", "missing", "calibration" or "inconclusive"), ``results`` (per sensor labels,
    stow geometry, scores, gain) and ``analysis``. ``remove_flagged_ranges`` produces ``df_final``.
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
        ref_weather: pd.DataFrame = None,
    ):
        self.site = SolarSite(site)
        self.config = deepcopy(config) if config is not None else POAQCConfig()
        self.design = _design(site)
        self.has_trackers = self.design["has_trackers"]
        self.analysis_freq = pd.Timedelta(self.config.analysis_freq)
        self.ref_poa = ref_poa
        self.ref_meter = ref_meter
        self.ref_weather = ref_weather

        self.qc_changes = {}
        self.cleaning_changes = deepcopy(cleaning_changes) if cleaning_changes else {}
        if self.cleaning_changes:
            self.qc_changes["input_cleaning_changes"] = deepcopy(self.cleaning_changes)

        self._process_input_data(dataframe, ref_poa, ref_meter, meter_valid_mask)
        self._run_poa_qc(ref_poa, ref_meter, meter_valid_mask, get_site_location(site), ref_weather)

        self.df_final = None
        self.removed_ranges = {}
        self.dropped_sensors = {}
        self.poa_source = None

    # ---- input / cleaning ----
    def _process_input_data(self, df, ref_poa, ref_meter, meter_valid_mask):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("POA sensor data must have a DatetimeIndex.")
        if df.index.tz is None:
            raise TypeError("POA sensor data must have a timezone-aware DatetimeIndex.")
        if not df.index.is_monotonic_increasing or df.index.has_duplicates:
            raise ValueError("POA sensor data index must be sorted and unique.")
        if ref_poa.empty:
            raise ValueError("Reference POA data cannot be empty.")
        if meter_valid_mask is not None:
            if ref_meter is None:
                raise ValueError("meter_valid_mask requires ref_meter.")
            if not pd.api.types.is_bool_dtype(meter_valid_mask.dtype):
                raise TypeError("meter_valid_mask must contain boolean values.")
        poa_cols = list(df.filter(like="OE.POA_").columns)
        if not poa_cols:
            raise ValueError("No 'OE.POA' sensor columns found.")

        self.native_freq = infer_freq(df.index)
        if self.native_freq > self.analysis_freq:
            raise ValueError("Sensor data interval must not exceed the analysis frequency.")
        self.df_native = self._clean_native_data(df[poa_cols])
        if self.df_native.empty:
            raise ValueError(
                f"No valid data found for 'OE.POA' sensor column(s).\n{self.qc_changes}"
            )
        self.poa_columns = list(self.df_native.columns)

    def _clean_native_data(self, df_native):
        df = df_native.dropna(axis=1, how="all").copy()
        empty_cols = [c for c in df_native.columns if c not in df.columns]
        zeroed_cols = [c for c in df.columns if df[c].abs().max() < 1e-5]
        remove_cols = empty_cols + zeroed_cols
        if remove_cols:
            df = df.drop(columns=remove_cols)
            rc_ = {k: v for k, v in zip(["all_na", "all_zero"], [empty_cols, zeroed_cols]) if v}
            self.qc_changes["removed_columns"] = rc_

        n_negative_by_col = df.lt(0).sum().to_dict()
        if sum(n_negative_by_col.values()) > 0:
            df[df.lt(0)] = 0.0
            self.qc_changes["negative_values"] = n_negative_by_col

        # extended ranges of zero values (flatlined at zero)
        flatlined = {}
        for col in df.columns:
            zero_ranges = _ranges_from_mask(df[col].eq(0), self.native_freq)
            long_ranges = [r for r in zero_ranges if (r[-1] - r[0]) > pd.Timedelta(days=1)]
            if long_ranges:
                for r in long_ranges:
                    df.loc[r, col] = np.nan
                flatlined[col] = long_ranges
        if flatlined:
            self.qc_changes["zero_ranges"] = flatlined
        return df

    # ---- QC ----
    def _run_poa_qc(self, ref_poa, ref_meter, meter_valid_mask, location, ref_weather=None):
        assessed = assess_poa_sensors(
            sensors=self.df_native,
            ref_poa=ref_poa,
            location=location,
            design=self.design,
            ref_meter=ref_meter,
            meter_valid=meter_valid_mask,
            config=self.config,
            ref_weather=ref_weather,
        )
        self.analysis = assessed["analysis"]
        self.results = assessed["sensors"]
        self.gains = {col: res["gain"] for col, res in self.results.items()}

        flags = {}
        self.flagged_ranges = {}
        for col, res in self.results.items():
            flag = _labels_to_native(res["label"], self.df_native.index, self.analysis_freq)
            flag = flag.where(flag.isin(list(FLAG_COLORS)), "")
            flag = flag.mask(self.df_native[col].isna() & flag.eq("inconclusive"), "")
            flags[f"{col}_FLAG"] = flag
            self.flagged_ranges[col] = _ranges_from_mask(
                flag.isin(REMOVAL_FLAGS + ("stow",)), self.native_freq
            )
        self.df_flags = pd.DataFrame(flags, index=self.df_native.index)

    def flag_summary(self) -> pd.DataFrame:
        """Hours flagged per sensor and flag."""
        hours = self.native_freq / pd.Timedelta(hours=1)
        counts = {
            c.removesuffix("_FLAG"): self.df_flags[c].value_counts().drop("", errors="ignore")
            * hours
            for c in self.df_flags.columns
        }
        return pd.DataFrame(counts).fillna(0.0).round(1)

    def meter_stow_summary(self) -> pd.DataFrame:
        """Runs where the plant meter looks stowed (and so is excluded from the cloud index).

        One row per run: start, end, hours, and (when ``ref_weather`` was given) the peak wind, the
        windy flag (wind >= ``stow_min_wind``) -- to separate wind stow from other causes -- and the
        mean sunshine. Meter-stow periods are backfilled from DTN's cloud index instead.
        """
        a = self.analysis
        ms = a["meter_stow"]
        run_id = (ms != ms.shift()).cumsum()
        rows = []
        for _, idx in ms[ms].groupby(run_id[ms]).groups.items():
            row = dict(start=idx[0], end=idx[-1] + self.analysis_freq, hours=len(idx) * self.analysis_freq / pd.Timedelta(hours=1))
            if "wx_wind" in a:
                row["peak_wind"] = a.loc[idx, "wx_wind"].max()
                row["windy"] = bool(a.loc[idx, "windy"].any())
            if "wx_sunshine" in a:
                row["mean_sunshine"] = a.loc[idx, "wx_sunshine"].mean()
            rows.append(row)
        return pd.DataFrame(rows)

    def daily_flags(self) -> pd.DataFrame:
        """Flag(s) present per sensor per local day (quick overview)."""
        join = lambda s: ",".join(sorted(set(v for v in s if v)))
        return self.df_flags.groupby(self.df_flags.index.date).agg(join)

    # ---- removal / backfill ----
    def _anchored_model(self, average: pd.Series, anchor="2h", max_gap="8h") -> pd.Series:
        """Modelled tracking POA on the analysis grid: clear-sky x cloud index x gain.

        The gain is the kept-sensor/model ratio, measured only where it is trustworthy: near peak sun
        (``anchor_min_signal`` x the peak clear-sky POA -- the sensor/model ratio drifts with sun angle,
        so low-sun ratios must not be carried to midday) and at least ``anchor_buffer`` away from any
        removed data (a sensor is still converging onto the tracking curve next to a stow). The
        whole-record median of those ratios is the default gain. A local median over ``anchor``
        replaces it near the sensor data: interpolated across a gap with trusted data on both sides
        (up to ``max_gap``), and, on one side only, faded back to the default over ``anchor_reach``.
        The local gain may differ from the default by at most ``anchor_max_adjust``.
        """
        cfg, step = self.config, self.analysis_freq
        a = self.analysis
        cs = a[f"cs_{TRACKER}"]
        kept = average.resample(step).mean().reindex(a.index)
        # Level from DTN, within-interval shape from the meter (``model_cloud_index``): the meter's
        # magnitude, stow or performance never reaches the model.
        if "primary_is_dtn" in a:
            from_dtn = a["primary_is_dtn"]
        else:
            from_dtn = a["csi_meter"].isna() if "csi_meter" in a else pd.Series(True, index=a.index)
        csi = a["csi_model"] if "csi_model" in a else a["csi_primary"]
        base = (cs * csi.where(a["use"])).fillna(0.0)

        removed = a["use"] & kept.isna()
        buffer = _to_bins(cfg.anchor_buffer, step)
        near_removed = removed.astype(float).rolling(2 * buffer + 1, center=True, min_periods=1).max() > 0
        rough = pd.Series(False, index=a.index)
        if cfg.dtn_only_max_curvature is not None:
            # Where the cloud index is hourly DTN (no meter) a rough, cloudy window cannot calibrate
            # the gain: DTN cannot see the clouds, so the sensor reads far below the model there.
            # With QC removing the smooth stowed days, such windows can be all that is left.
            win = _to_bins(cfg.window, step)
            r = (kept / cs.where(cs > 0)).where(a["use"])
            curvature = r.diff().diff().abs().rolling(win, center=True, min_periods=max(1, win // 2)).mean()
            rough = curvature > cfg.dtn_only_max_curvature

        def level(base, trusted, fallback):
            ratio = (kept / base.where(trusted)).clip(0.5, 1.5)
            # too little trustworthy data to set the level: use the fallback gain
            default = ratio.median() if ratio.notna().sum() >= cfg.anchor_min_bins else fallback
            local = ratio.rolling(anchor, min_periods=6).median()

            # distance (time) to the nearest anchor on each side, and its value
            t = pd.Series(a.index, index=a.index)
            t_anchor = t.where(local.notna())
            prev_t, next_t = t_anchor.ffill(), t_anchor.bfill()
            prev_v, next_v = local.ffill(), local.bfill()
            d_prev, d_next = (t - prev_t), (next_t - t)
            reach = pd.Timedelta(cfg.anchor_reach)

            fade = lambda v, d: default + (v - default) * (1 - d / reach).clip(lower=0)
            one_sided = fade(prev_v, d_prev).where(d_prev.notna(), np.nan)
            after = fade(next_v, d_next).where(d_next.notna(), np.nan)
            # nearest side wins when only one is within reach; both-sided gap: interpolate in time
            near_prev = d_prev.fillna(pd.Timedelta.max) <= d_next.fillna(pd.Timedelta.max)
            faded = one_sided.where(near_prev, after)
            both = prev_v.notna() & next_v.notna() & ((d_prev + d_next) <= pd.Timedelta(max_gap))
            span = (d_prev + d_next).dt.total_seconds().replace(0, np.nan)
            weight = (d_prev.dt.total_seconds() / span).fillna(0.0)
            between = prev_v + (next_v - prev_v) * weight
            gain = faded.where(~both, between).fillna(default)
            adjust = cfg.anchor_max_adjust
            return default, gain.clip(default * (1 - adjust), default * (1 + adjust))

        # a window DTN cannot see clouds in (no meter shape there) must not set the level
        trusted = ~near_removed & ~(rough & from_dtn) & (base >= cfg.anchor_min_signal * cs.max())
        own = float(np.median(list(self.gains.values())))  # too few trusted bins: sensors' own gain
        default, gain = level(base, trusted, own)
        if cfg.meter_level_tol is not None and "csi_meter" in a and "csi_dtn" in a and getattr(self, "ref_poa", None) is not None:
            # DTN cannot see every cloud: where the kept sensors agree with the meter's level, use it
            csi_meter = a["csi_meter"].where(~from_dtn)
            factor = meter_level_factor(
                a["csi_dtn"], csi_meter, kept, cs, default, infer_freq(self.ref_poa.index), step,
                cfg.meter_level_tol, pd.Timedelta(cfg.meter_level_reach),
            )
            base = base * factor
            trusted = ~near_removed & ~(rough & from_dtn) & (base >= cfg.anchor_min_signal * cs.max())
            default, gain = level(base, trusted, own)
        return base * gain

    def _mostly_removed_sensors(self) -> dict:
        """Sensors (col -> removed share) with ``config.drop_removed_fraction`` of the daylight bins
        removed in ``df_final``: what is left is mostly night-time zeros, so they carry no information.
        A sensor is never dropped if that would leave none."""
        threshold = self.config.drop_removed_fraction
        use = self.analysis["use"]
        if threshold is None or not use.any():
            return {}
        step = self.analysis_freq
        removed = {}
        for col in self.poa_columns:
            kept = self.df_final[col].resample(step).mean().reindex(use.index)
            removed[col] = float((kept.isna() & use).sum() / use.sum())
        dropped = {c: f for c, f in removed.items() if f >= threshold}
        return dropped if len(dropped) < len(self.poa_columns) else {}

    def remove_flagged_ranges(self, secondary_stow_condition=False, include_stow=None):
        """Sets flagged data to NaN and builds ``df_final`` (sensors, OE.POA_average, Processed_POA).

        ``include_stow`` (default True unless ``secondary_stow_condition``) controls whether stowed
        data is removed. Gaps in the sensor average are backfilled with the modelled POA: tracker
        clear-sky POA x cloud index (meter where valid, else DTN) x the median gain of the sensors
        -- so passing clouds seen by the meter survive, unlike with an hourly reference. Gaps of at
        most ``config.fill_max_gap`` with good data on both sides are interpolated instead.
        ``poa_source`` records, per timestamp, "sensor", "interpolated" or "model".
        """
        if include_stow is None:
            include_stow = not secondary_stow_condition
        remove = REMOVAL_FLAGS + (("stow",) if include_stow else ())
        self.df_final, self.removed_ranges = _remove_flagged_ranges(
            self.df_native, self.df_flags, remove
        )
        self.dropped_sensors = self._mostly_removed_sensors()
        if self.dropped_sensors:
            self.df_final = self.df_final.drop(columns=list(self.dropped_sensors))
            self.qc_changes["dropped_sensors"] = dict(self.dropped_sensors)
        else:
            self.qc_changes.pop("dropped_sensors", None)
        kept_columns = [c for c in self.poa_columns if c not in self.dropped_sensors]
        average = self.df_final[kept_columns].mean(axis=1)
        self.df_final["OE.POA_average"] = average

        modelled_bins = self._anchored_model(average)
        modelled = (
            modelled_bins.reindex(modelled_bins.index.union(self.df_final.index))
            .interpolate(method="time", limit_area="inside")
            .reindex(self.df_final.index)
            .fillna(0.0)
        )
        # a short gap with good sensor data on both sides is bridged from the sensor itself, not
        # spliced with the model (OE.POA_average keeps the gap: it is not a calibration anchor)
        bridged, interpolated = _fill_short_gaps(average, self.config.fill_max_gap, self.native_freq)
        missing = bridged.isna()
        self.df_final["Processed_POA"] = bridged.where(~missing, modelled)
        self.poa_source = pd.Series(
            np.select([missing, interpolated], ["model", "interpolated"], "sensor"),
            index=average.index,
        )
        return self.df_final

    # ---- plots ----
    def _flags_at(self, col: str, index: pd.DatetimeIndex) -> pd.Series:
        """Flag(s) of a sensor at each timestamp of ``index`` (a bin's distinct flags, comma-joined,
        when ``index`` is coarser than the native interval)."""
        flag = self.df_flags[f"{col}_FLAG"]
        if len(index) > 1 and infer_freq(index) > self.native_freq:
            join = lambda s: ",".join(sorted(set(v for v in s if v)))
            return flag.resample(infer_freq(index)).agg(join).reindex(index).fillna("")
        return flag.reindex(index, method="ffill").fillna("")

    def _shapes(self, rows: dict, flags=FLAG_COLORS, opacity=0.25, freq=None, extra=None):
        """Shaded rectangles for each flagged run; ``rows`` maps sensor column -> subplot row.

        With ``freq`` (a duration larger than the native interval), flags are resampled to that
        interval -- each bin takes its most common value, "ok"/unflagged winning ties -- so the
        shading lines up with data resampled to the same bins.

        ``extra`` maps a key in ``rows`` to its own label series in place of a sensor's
        ``<col>_FLAG`` column.
        """
        extra = extra or {}
        shapes = []
        for col, row in rows.items():
            if col in extra:
                flag, native = extra[col], infer_freq(extra[col].index)
            else:
                flag, native = self.df_flags[f"{col}_FLAG"], self.native_freq
            step = native
            if freq is not None and pd.Timedelta(freq) > native:
                step = pd.Timedelta(freq)
            if step != native:
                share = pd.DataFrame({n: flag.eq(n).resample(step).mean() for n in flags})
                flag = share.idxmax(axis=1).where(share.max(axis=1) > 1 - share.sum(axis=1))
            for name, color in flags.items():
                for idx in _ranges_from_mask(flag.eq(name), step):
                    shapes.append(
                        dict(
                            type="rect",
                            x0=idx[0],
                            x1=idx[-1] + step,
                            y0=0,
                            y1=1,
                            yref=f"y{row} domain".replace("y1 ", "y "),
                            fillcolor=color,
                            opacity=opacity,
                            line_width=0,
                            layer="below",
                        )
                    )
        return shapes

    def _add_reference_traces(self, fig, rows, step=None, hovertemplate=None):
        """Overlay clear-sky, DTN and (once ``remove_flagged_ranges`` has run) anchored model POA.

        Each reference gets one legend entry (on the first of ``rows``) whose toggle applies to all
        of its copies. ``step`` resamples (mean) any reference with a finer interval.
        """
        a = self.analysis
        dtn = _as_tz(self.ref_poa, a.index).loc[a.index[0] : a.index[-1]]
        references = [
            ("clear-sky POA", a[f"cs_{TRACKER}"], dict(color="#aaaaaa", width=1.2)),
            ("DTN POA", dtn, dict(color="#2E86AB", width=1.2, shape="hv")),
        ]
        if self.df_final is not None:
            anchored = self._anchored_model(self.df_final["OE.POA_average"])
            references.append(
                ("anchored model POA", anchored, dict(color="#E07A1F", width=1.2, dash="dot"))
            )
        for name, y, line in references:
            if step is not None and step > infer_freq(y.index):
                y = y.resample(step).mean()
            for row in rows:
                fig.add_trace(
                    go.Scattergl(
                        x=y.index,
                        y=y,
                        name=name,
                        line=line,
                        opacity=0.8,
                        legendgroup=name,
                        showlegend=row == rows[0],
                        hoverinfo="all" if row == rows[0] else "skip",
                        hovertemplate=(hovertemplate or VALUE_HOVER) if row == rows[0] else None,
                    ),
                    row=row,
                    col=1,
                )

    def _add_meter_trace(self, fig, row, step=None, hovertemplate=None):
        """Reference meter output (a series of zeros if none was provided)."""
        index = self.analysis.index
        if self.ref_meter is None:
            meter = pd.Series(0.0, index=index)
        else:
            meter = _as_tz(self.ref_meter, index).loc[index[0] : index[-1]]
        if step is not None and step > infer_freq(meter.index):
            meter = meter.resample(step).mean()
        fig.add_trace(
            go.Scattergl(
                x=meter.index,
                y=meter,
                name="reference meter",
                line=dict(color="#5C8A3A", width=1.2),
                hovertemplate=hovertemplate or VALUE_HOVER,
            ),
            row=row,
            col=1,
        )

    @staticmethod
    def _row_heights(n_rows: int, n_short: int = 2, short_ratio: float = 2 / 3):
        """Relative subplot heights: the first ``n_short`` rows are ``short_ratio`` x the others."""
        return [short_ratio] * n_short + [1.0] * (n_rows - n_short)

    def plot(self, freq=None):
        """Reference row, reference meter row, then per-sensor raw data vs. modelled expectation.

        The first row holds the clear-sky, DTN and (if ``remove_flagged_ranges`` has run) anchored
        model POA, which are also overlaid on every sensor row; the second holds the reference
        meter. The sensor rows are shaded by flag (hover shows the flag).

        If ``freq`` is longer than the plotted data's interval (``analysis_freq``), the data is
        resampled to it using the mean and the flag shading is aggregated to the same bins.
        """
        cols = self.poa_columns
        resample = freq is not None and pd.Timedelta(freq) > self.analysis_freq
        step = pd.Timedelta(freq) if resample else None
        n = len(cols) + 2
        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1 / n,
            row_heights=self._row_heights(n),
            subplot_titles=["references", "reference meter"]
            + [f"{c}  (gain {self.gains[c]:.2f})" for c in cols],
        )
        self._add_reference_traces(fig, [1, *range(3, n + 1)], step)
        self._add_meter_trace(fig, 2, step)
        a = self.analysis
        for i, col in enumerate(cols, start=1):
            row = i + 2
            expected = a[f"cs_{TRACKER}"] * a["csi_primary"] * self.gains[col]
            traces = [
                (a[col], col, dict(color=plotly.colors.qualitative.G10[i - 1], width=1.2)),
                (expected.where(a["use"]), "expected (tracking)", dict(color="#777777", width=1)),
            ]
            for y, name, line in traces:
                if resample:
                    y = y.resample(step).mean()
                hover = dict(hovertemplate=VALUE_HOVER)
                if name == col:
                    hover = dict(customdata=self._flags_at(col, y.index), hovertemplate=FLAG_HOVER)
                fig.add_trace(
                    go.Scattergl(
                        x=y.index,
                        y=y,
                        name=name,
                        line=line,
                        legendgroup=name,
                        showlegend=row == 3,
                        mode="lines",
                        **hover,
                    ),
                    row=row,
                    col=1,
                )
        fig.update_layout(
            font_size=9,
            margin=dict(t=30, b=20, l=20, r=20),
            height=max(300, 230 * n - 150),
            shapes=self._shapes(
                {"reference meter": 2, **{c: i for i, c in enumerate(cols, start=3)}},
                freq=step,
                extra={
                    "reference meter": self.analysis["meter_stow"].map({True: "stow", False: ""})
                },
            ),
        )
        return fig

    def plot_result(self, highlight="flags"):
        """Reference rows (clear-sky, DTN, anchored model; reference meter), native raw vs. kept data
        per sensor, then the sensor average and final Processed_POA.

        The POA reference traces are drawn in the first row and overlaid on every row but the meter
        row and the last; each has a single legend entry whose toggle applies to all of its copies.

        ``highlight`` shades the sensor rows and the OE.POA_average row: "flags" by flag type,
        "removed" for all removed ranges in one color, or None for no shading. The sensor rows shade
        each sensor's own flagged/removed ranges. The average row shades only the *backfilled*
        ranges, i.e. where no sensor has good data and Processed_POA is modelled; in "flags" mode
        they take the most severe flag any sensor has there ("missing" if none is flagged).
        """
        if self.df_final is None:
            raise ValueError("Run 'remove_flagged_ranges' before calling this function.")
        if highlight not in ("flags", "removed", None):
            raise ValueError("highlight must be 'flags', 'removed' or None.")
        df0, df1 = self.df_native, self.df_final
        cols = [c for c in self.poa_columns if c in df1.columns]  # dropped sensors are not shown
        step = "5min"
        n = len(cols) + 4
        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1 / n,
            row_heights=self._row_heights(n),
        )
        a = self.analysis
        hv = "<b>%{fullData.name}</b><br>%{y:.1f}<br><i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"

        self._add_reference_traces(fig, [1, *range(3, n)], hovertemplate=hv)
        self._add_meter_trace(fig, 2, hovertemplate=hv)

        for i, col in enumerate(cols, start=1):
            row = i + 2
            color = plotly.colors.qualitative.G10[i - 1]
            removed = df0[col].where(df1[col].isna()).resample(step).mean()
            kept = df1[col].resample(step).mean()
            fig.add_trace(
                go.Scattergl(
                    x=kept.index,
                    y=kept,
                    name=col,
                    line_color=color,
                    legendgroup=col,
                    customdata=self._flags_at(col, kept.index),
                    hovertemplate=FLAG_HOVER,
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scattergl(
                    x=removed.index,
                    y=removed,
                    name=f"{col}_removed",
                    line=dict(color=color, dash="dot"),
                    opacity=0.7,
                    legendgroup=col,
                    customdata=self._flags_at(col, removed.index),
                    hovertemplate=FLAG_HOVER,
                ),
                row=row,
                col=1,
            )
        for row, (col, color) in enumerate(
            [("OE.POA_average", plotly.colors.qualitative.G10[4]), ("Processed_POA", "#303030")],
            start=n - 1,
        ):
            y = df1[col].resample(step).mean()
            fig.add_trace(
                go.Scattergl(
                    x=y.index, y=y, name=col, legendgroup=row, line_color=color, hovertemplate=hv
                ),
                row=row,
                col=1,
            )
        model_only = df1["Processed_POA"].where(self.poa_source.eq("model")).resample(step).mean()
        fig.add_trace(
            go.Scattergl(
                x=model_only.index,
                y=model_only,
                name="modelled backfill",
                legendgroup=n,
                line=dict(color="#8D00C0", width=1.5),
            ),
            row=n,
            col=1,
        )

        shapes = []
        if highlight:
            rows = {c: i for i, c in enumerate(cols, start=3)}
            rows["OE.POA_average"] = n - 1
            backfilled = self.poa_source.eq("model")  # no sensor left: Processed_POA is the model
            if highlight == "removed":
                flags, extra = {"removed": "lightsalmon"}, {}
                for c in cols:
                    extra[c] = pd.Series(
                        np.where(df1[c].isna() & df0[c].notna(), "removed", ""), index=df0.index
                    )
                extra["OE.POA_average"] = backfilled.map({True: "removed", False: ""})
            else:
                flags, extra = FLAG_COLORS, {}
                severity = ("bad", "missing", "calibration", "stow", "inconclusive")
                combined = pd.Series("", index=self.df_flags.index)
                for name in reversed(severity):  # most severe assigned last, so it wins
                    has = self.df_flags.eq(name).any(axis=1)
                    combined = combined.mask(has, name)
                # only where nothing good is left; a backfilled bin with no flag is a data gap
                combined = combined.where(backfilled, "").mask(backfilled & combined.eq(""), "missing")
                extra["OE.POA_average"] = combined
            shapes = self._shapes(rows, flags=flags, opacity=0.2, extra=extra)
        fig.update_layout(
            font_size=9,
            margin=dict(t=20, b=20, l=20, r=20),
            height=max(200 * n - 150, 500),
            legend=dict(x=1.01, xanchor="left", y=0.5, yanchor="middle", tracegroupgap=22),
            shapes=shapes,
        )
        return fig
