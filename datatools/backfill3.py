"""Weather sensor QC + backfill: POA via ``solarqc2.POADataset``; temperature and wind with the same
pattern, without the geometry.

Successor to ``backfill2.backfill_weather_sensor_data``, which judged each sensor by half-day R^2
against DTN and filled the gaps with raw DTN. Here every sensor group follows the POADataset recipe:

- scores are windowed errors on an analysis grid, against two independent references: DTN and the
  sensor's peers (the median of the other sensors in the group). A window only counts against a
  sensor when it disagrees with *both*. DTN timing errors are excused because the peers agree, and a
  drifting sensor is caught because neither reference follows it.
- every analysis bin gets a label ("ok", "bad", "missing", "calibration"). Short "bad" runs are
  dropped, flagged data is removed at native resolution, and a sensor with most of its data removed
  is dropped altogether.
- gaps are filled with an *anchored* reference: DTN corrected by the kept sensors' local
  offset (temperature) or gain (wind). Near the edge of a gap the correction follows the
  sensors, and further into the gap it fades to the whole-record value, so the fill joins the
  sensor data without a step. Short gaps with sensor data on both sides are interpolated instead.
  ``source`` records, per timestamp, whether the value is "sensor", "interpolated" or "model".

All temperature data is standardized to deg C (DTN's unit). Sensors reporting deg F are detected
(see ``SensorQCConfig.units``) and converted on input, so sensor, average and processed columns
are all deg C. The detected input unit is recorded in ``qc_changes[group]["native_units"]``.
Wind sensors are compared through a fitted gain, which absorbs both measurement height and units.
"""

import warnings
from dataclasses import dataclass, fields, replace
from typing import Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .backfill2 import BACKFILL_COL_MAPPING, _grouped_sensor_cols
from .solarqc2 import (
    FLAG_COLORS,
    POADataset,
    POAQCConfig,
    SKY_COLUMNS,
    WEATHER_COLUMNS,
    _as_tz,
    _drop_short_runs,
    _fill_short_gaps,
    _labels_to_native,
    _ranges_from_mask,
    _remove_flagged_ranges,
    _run_lengths,
    _to_bins,
    _to_grid,
)
from ..utils.datetime import infer_freq
from ..utils.helpers import quiet_print_function

POA_GROUP = "OE.POA"
REMOVAL_FLAGS = ("bad", "missing", "calibration")
SOURCE_COLORS = {"model": "lightsalmon", "interpolated": "rgb(255, 214, 102)"}


# ---- configuration -----------------------------------------------------------------------------
@dataclass
class SensorQCConfig:
    """Thresholds for temperature / wind sensor QC (see ``default_sensor_config`` for each group).

    ``mode`` sets the error model. With "offset" (temperature) a sensor reads DTN + c, and errors
    and tolerances are absolute, in deg C. With "gain" (wind) a sensor reads DTN x c, and errors and
    tolerances are relative to the reference level in the window.

    A window is "bad" when its error exceeds the tolerance against *both* references:
    ``tol + dtn_tol_factor x (hourly rate of change of DTN)`` for DTN, since an hourly reference
    is least reliable when conditions change fast, and ``peer_tol`` for the peers.
    """

    mode: str = "offset"  # "offset" | "gain"
    units: Optional[str] = "auto"  # temperature input: "auto" | "C" | "F"; output is always deg C
    analysis_freq: Union[str, pd.Timedelta] = "15min"
    window: Union[str, pd.Timedelta] = "6h"  # scoring window (centered)
    smooth: Union[str, pd.Timedelta] = (
        "1h"  # smoothing before comparing (point sensor vs. hourly DTN)
    )
    tol: float = 3.0  # allowed error vs. DTN under steady conditions
    dtn_tol_factor: float = 1.0  # ...plus this x DTN's rate of change (per hour)
    peer_tol: float = 2.0  # allowed error vs. the peer median
    level_floor: float = 0.0  # gain mode: relative errors use at least this level (DTN units)
    min_ref_level: float = 0.0  # gain mode: gain is fit / anchored only where DTN exceeds this
    min_bad_duration: Union[str, pd.Timedelta] = "2h"
    min_missing_duration: Union[str, pd.Timedelta] = "1h"
    value_bounds: Optional[tuple] = (-50.0, 60.0)  # physical limits (deg C / DTN units); None = off
    max_flatline: Union[str, pd.Timedelta, None] = (
        "1D"  # identical values longer than this: removed
    )
    stuck_window: Union[str, pd.Timedelta] = "6h"  # constant this long while DTN moves: "bad"
    stuck_min_ref_range: Optional[float] = 3.0  # ...DTN range over the run at least this
    stuck_min_ref_mean: Optional[float] = None  # ...or DTN mean over the run at least this
    calibration_bounds: Optional[tuple] = (-5.0, 5.0)  # sensor offset/gain vs. DTN outside: flagged
    drop_removed_fraction: Optional[float] = 0.7  # sensor with this share removed is dropped
    fill_max_gap: Union[str, pd.Timedelta, None] = "1h"  # interpolate gaps up to this
    anchor: Union[str, pd.Timedelta] = "6h"  # local correction = median over this window
    anchor_reach: Union[str, pd.Timedelta] = "6h"  # one-sided anchors fade to the default over this
    anchor_max_gap: Union[str, pd.Timedelta] = (
        "24h"  # anchors on both sides: interpolate up to this
    )
    anchor_max_adjust: float = 3.0  # local correction may differ from the default by this
    # (gain mode: as a fraction of the default)

    def __post_init__(self):
        if self.mode not in ("offset", "gain"):
            raise ValueError("mode must be 'offset' or 'gain'.")
        if self.units not in (None, "auto", "C", "F"):
            raise ValueError("units must be 'auto', 'C', 'F' or None.")
        if self.units is not None and self.mode != "offset":
            raise ValueError("units conversion is only supported for mode='offset' (temperature).")
        for name in (
            "analysis_freq",
            "window",
            "smooth",
            "min_bad_duration",
            "min_missing_duration",
            "stuck_window",
            "anchor",
            "anchor_reach",
            "anchor_max_gap",
        ):
            if pd.isna(pd.Timedelta(getattr(self, name))):
                raise ValueError(f"{name} must be a valid duration.")
        if self.tol <= 0 or self.peer_tol <= 0:
            raise ValueError("tol and peer_tol must be positive.")
        for name in ("value_bounds", "calibration_bounds"):
            b = getattr(self, name)
            if b is not None and len(b) != 2:
                raise ValueError(f"{name} must be a (min, max) pair (None for no bound).")
        if self.drop_removed_fraction is not None and not 0 < self.drop_removed_fraction <= 1:
            raise ValueError("drop_removed_fraction must be in (0, 1] (or None to disable).")

    def __str__(self):
        width = max(len(f.name) for f in fields(self))
        lines = [f"{f.name:<{width}} = {getattr(self, f.name)!r}" for f in fields(self)]
        return "\n".join([f"{type(self).__name__}:", *("  " + line for line in lines)])


def default_sensor_config(group: str) -> SensorQCConfig:
    """Fresh default config for ``OE.Ambient_Temp`` or ``OE.Wind_Speed``."""
    if group == "OE.Ambient_Temp":
        return SensorQCConfig()
    if group == "OE.Wind_Speed":
        # Deliberately lenient: point anemometers disagree with hourly 10 m DTN (and with each
        # other: stall thresholds, siting) far more than temperature sensors do, and wind matters
        # less downstream. Aimed at clear failures: stuck/frozen, out of range, gross disagreement.
        return SensorQCConfig(
            mode="gain",
            units=None,
            window="12h",
            tol=0.75,
            dtn_tol_factor=1.0,
            peer_tol=0.5,
            level_floor=4.0,  # relative errors at low wind are unstable: floor the level (DTN m/s)
            min_ref_level=2.0,
            value_bounds=(0.0, 75.0),
            stuck_min_ref_range=None,
            stuck_min_ref_mean=4.0,
            calibration_bounds=(0.2, 4.0),  # wide: hub height and mph vs m/s both move the gain
            anchor_max_adjust=0.3,
        )
    raise KeyError(f"No default sensor QC config for group '{group}'.")


def _resolve_config(group: str, override) -> SensorQCConfig:
    base = default_sensor_config(group)
    if override is None:
        return base
    if isinstance(override, SensorQCConfig):
        return override
    if isinstance(override, dict):
        return replace(base, **override)
    raise TypeError("sensor_configs values must be SensorQCConfig or dict of field overrides.")


# ---- reference alignment -----------------------------------------------------------------------
def _ref_on_index(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Reference series on ``index``. Coarser data (interval-start timestamps, e.g. hourly DTN) is
    linearly interpolated between interval midpoints. A single missing interval is bridged (or, at
    the end of the record, the last value is held), while longer outages stay NaN. Finer data is
    averaged."""
    s = _as_tz(series, index).astype(float)
    freq, target = infer_freq(s.index), infer_freq(index)
    if freq <= target:
        return s.resample(target).mean().reindex(index)
    s = s.dropna()
    if s.empty:
        return pd.Series(np.nan, index=index)
    held = s.reindex(index, method="ffill", tolerance=2 * freq - pd.Timedelta(1, "ns"))
    mid = s.copy()
    mid.index = mid.index + freq / 2
    full = mid.reindex(mid.index.union(index)).interpolate(method="time", limit_area="inside")
    out = full.reindex(index).where(held.notna())
    return out.fillna(held)  # first/last half interval: hold the value


def _detect_fahrenheit(x: pd.DataFrame, ref_c: pd.Series) -> bool:
    """True when the sensors are closer to DTN (deg C) converted to deg F than to DTN itself."""
    vals = x.median(axis=1)
    ok = vals.notna() & ref_c.notna()
    if ok.sum() < 10:
        return bool(vals.median() > 45)  # no overlap: a typical temperature above 45 is not deg C
    err_c = (vals[ok] - ref_c[ok]).abs().median()
    err_f = (vals[ok] - (ref_c[ok] * 1.8 + 32)).abs().median()
    return bool(err_f < err_c)


# ---- per-sensor assessment ---------------------------------------------------------------------
def _fit_correction(x: pd.Series, ref: pd.Series, cfg: SensorQCConfig, min_bins: int = 8) -> float:
    """Sensor offset (mode "offset") or gain (mode "gain") vs. the reference: median over days of
    the daily value. NaN when no day has ``min_bins`` usable bins."""
    ok = x.notna() & ref.notna()
    if cfg.mode == "gain":
        ok &= ref > cfg.min_ref_level
    if not ok.any():
        return np.nan
    df = pd.DataFrame({"x": x[ok], "r": ref[ok]})
    by_day = df.groupby(df.index.date)
    counts = by_day.size()
    if cfg.mode == "offset":
        daily = (df["x"] - df["r"]).groupby(df.index.date).median()
    else:
        sums = by_day.sum()
        daily = (sums["x"] / sums["r"]).where(sums["r"] > 0)
    daily = daily[counts >= min_bins].dropna()
    return float(daily.median()) if len(daily) else np.nan


def _apply(ref, corr, mode):
    return ref + corr if mode == "offset" else ref * corr


def _remove_correction(x, corr, mode):
    return x - corr if mode == "offset" else x / corr


def _stuck_bins(x: pd.Series, ref: pd.Series, cfg: SensorQCConfig, step: pd.Timedelta) -> pd.Series:
    """True on runs of identical values at least ``stuck_window`` long during which DTN moves
    (range >= ``stuck_min_ref_range``) or is high (mean >= ``stuck_min_ref_mean``). This catches
    frozen sensors that agree with each other, which the peer check alone would excuse."""
    never = pd.Series(False, index=x.index)
    if cfg.stuck_min_ref_range is None and cfg.stuck_min_ref_mean is None:
        return never
    xr = x.round(4)
    run_id = (xr != xr.shift()).cumsum()
    long_run = x.notna() & (_run_lengths(xr.fillna(np.inf)) >= _to_bins(cfg.stuck_window, step))
    if not long_run.any():
        return never
    grouped = ref.groupby(run_id)
    moved = pd.Series(False, index=x.index)
    if cfg.stuck_min_ref_range is not None:
        rng = grouped.transform("max") - grouped.transform("min")
        moved |= rng >= cfg.stuck_min_ref_range
    if cfg.stuck_min_ref_mean is not None:
        moved |= grouped.transform("mean") >= cfg.stuck_min_ref_mean
    return long_run & moved


def assess_sensor_group(sensors: pd.DataFrame, ref: pd.Series, cfg: SensorQCConfig) -> dict:
    """Label each analysis bin of each sensor: "ok" | "bad" | "missing" | "calibration" | ""
    (not assessable).

    ``sensors`` and ``ref`` share one regular analysis grid, and the units must already match. For
    each sensor the error is computed against (a) DTN corrected by the sensor's own offset/gain and
    (b) the median of its peers, each corrected into the sensor's frame. Each error is divided by
    its tolerance, and the score is the better (lower) of the two. Returns dict with ``analysis``
    (frame on the grid) and ``sensors`` (col -> label, correction, score, score_dtn, score_peer).
    """
    step = infer_freq(sensors.index)
    win = _to_bins(cfg.window, step)
    roll = lambda s: s.rolling(win, center=True, min_periods=max(1, win // 2)).mean()
    n_sm = _to_bins(cfg.smooth, step)
    sm = lambda s: s.rolling(n_sm, center=True, min_periods=max(1, n_sm // 2)).mean()
    per_hour = pd.Timedelta(hours=1) / step
    rate = roll(ref.diff().abs()) * per_hour  # DTN rate of change, units per hour
    default = 0.0 if cfg.mode == "offset" else 1.0

    corr = {}
    for col in sensors.columns:
        c = _fit_correction(sensors[col], ref, cfg)
        corr[col] = c if np.isfinite(c) and (cfg.mode == "offset" or c > 0) else default
    neutral = pd.DataFrame(
        {c: _remove_correction(sensors[c], corr[c], cfg.mode) for c in sensors.columns}
    )

    out = {}
    for col in sensors.columns:
        x, c = sensors[col], corr[col]
        expected = _apply(ref, c, cfg.mode)
        others = [o for o in sensors.columns if o != col]
        with warnings.catch_warnings():  # rows where every peer is NaN: NaN, without the warning
            warnings.simplefilter("ignore", RuntimeWarning)
            peer_median = neutral[others].median(axis=1) if others else None
        peer = _apply(peer_median, c, cfg.mode) if others else pd.Series(np.nan, index=x.index)
        xs = sm(x).where(x.notna())
        err_dtn, err_peer = roll((xs - sm(expected)).abs()), roll((xs - sm(peer)).abs())
        tol_dtn = cfg.tol + cfg.dtn_tol_factor * rate
        if cfg.mode == "gain":
            level = roll(expected).clip(lower=cfg.level_floor * c)
            err_dtn, err_peer = err_dtn / level, err_peer / level
            tol_dtn = cfg.tol + cfg.dtn_tol_factor * rate * c / level
        score_dtn = (err_dtn / tol_dtn).where(x.notna())
        score_peer = (err_peer / cfg.peer_tol).where(x.notna())
        score = pd.concat([score_dtn, score_peer], axis=1).min(axis=1)  # the better reference

        lab = pd.Series("", index=x.index, dtype=object)
        assessed = score.notna()
        lab[assessed] = np.where(score[assessed] > 1.0, "bad", "ok")
        lab = _drop_short_runs(lab, "bad", _to_bins(cfg.min_bad_duration, step))
        if cfg.value_bounds is not None:
            lo, hi = cfg.value_bounds
            out_of_range = pd.Series(False, index=x.index)
            if lo is not None:
                out_of_range |= x.lt(lo)
            if hi is not None:
                out_of_range |= x.gt(hi)
            lab[out_of_range] = "bad"
        lab[_stuck_bins(x, ref, cfg, step)] = "bad"
        gap = x.isna()
        lab[gap & (_run_lengths(gap) >= _to_bins(cfg.min_missing_duration, step))] = "missing"
        if cfg.calibration_bounds is not None:
            lo, hi = cfg.calibration_bounds
            if (lo is not None and c < lo) or (hi is not None and c > hi):
                lab = lab.mask(x.notna() & lab.ne("missing"), "calibration")

        out[col] = dict(
            label=lab, correction=c, score=score, score_dtn=score_dtn, score_peer=score_peer
        )

    analysis = sensors.assign(ref=ref, ref_rate=rate)
    for col, res in out.items():
        analysis[f"{col}_score"] = res["score"]
    return dict(analysis=analysis, sensors=out)


# ---- backfill model ----------------------------------------------------------------------------
def _anchored_correction(
    kept: pd.Series, ref: pd.Series, cfg: SensorQCConfig, default: float
) -> pd.Series:
    """Correction (offset or gain) of the kept-sensor average vs. DTN on the analysis grid, the
    module-level counterpart of ``POADataset._anchored_model``.

    The local correction is the median over ``anchor`` of the average vs. DTN, measured where the
    sensors have data. Across a gap with anchors on both sides (up to ``anchor_max_gap``) it is
    interpolated in time. With an anchor on one side only, it fades back to ``default`` (the
    whole-record correction) over ``anchor_reach``. It never differs from ``default`` by more than
    ``anchor_max_adjust``.
    """
    index = kept.index
    if cfg.mode == "offset":
        d = kept - ref
    else:
        d = (kept / ref.where(ref > cfg.min_ref_level)).clip(0.05, 20.0)
    win = _to_bins(cfg.anchor, infer_freq(index))
    local = d.rolling(win, center=True, min_periods=max(2, win // 4)).median().where(d.notna())

    t = pd.Series(index, index=index)
    t_anchor = t.where(local.notna())
    prev_t, next_t = t_anchor.ffill(), t_anchor.bfill()
    prev_v, next_v = local.ffill(), local.bfill()
    d_prev, d_next = t - prev_t, next_t - t
    reach = pd.Timedelta(cfg.anchor_reach)

    fade = lambda v, dist: default + (v - default) * (1 - dist / reach).clip(lower=0)
    before, after = fade(prev_v, d_prev), fade(next_v, d_next)
    near_prev = d_prev.fillna(pd.Timedelta.max) <= d_next.fillna(pd.Timedelta.max)
    faded = before.where(near_prev, after)
    both = prev_v.notna() & next_v.notna() & ((d_prev + d_next) <= pd.Timedelta(cfg.anchor_max_gap))
    span = (d_prev + d_next).dt.total_seconds().replace(0, np.nan)
    weight = (d_prev.dt.total_seconds() / span).fillna(0.0)
    between = prev_v + (next_v - prev_v) * weight
    corr = faded.where(~both, between).fillna(default)
    adjust = cfg.anchor_max_adjust if cfg.mode == "offset" else cfg.anchor_max_adjust * default
    return corr.clip(default - adjust, default + adjust)


def _grid_to_native(series: pd.Series, native: pd.DatetimeIndex) -> pd.Series:
    """Analysis-bin values (timestamp = bin start) -> native timestamps, interpolated between bin
    centers (for slowly varying series such as a correction)."""
    step = infer_freq(series.index)
    s = series.copy()
    s.index = s.index + step / 2
    full = s.reindex(s.index.union(native)).interpolate(method="time", limit_area="inside")
    return full.reindex(native).ffill().bfill()


# ---- group pipelines ---------------------------------------------------------------------------
def _clean_native(df: pd.DataFrame, cfg: SensorQCConfig, freq: pd.Timedelta):
    """Drops empty, all-zero and constant columns, clips negative values when the lower physical
    bound is 0, and removes flatlines longer than ``max_flatline``. Returns (df, changes)."""
    changes = {}
    empty = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=empty)
    zeroed = [c for c in df.columns if df[c].abs().max() < 1e-5]
    constant = [c for c in df.columns if c not in zeroed and df[c].nunique() <= 1]
    df = df.drop(columns=zeroed + constant)
    removed = {
        k: v for k, v in (("all_na", empty), ("all_zero", zeroed), ("constant", constant)) if v
    }
    if removed:
        changes["removed_columns"] = removed

    if cfg.value_bounds is not None and cfg.value_bounds[0] == 0:
        n_neg = df.lt(0).sum()
        if n_neg.sum() > 0:
            df = df.clip(lower=0.0)
            changes["negative_values"] = {k: int(v) for k, v in n_neg.items() if v}

    if cfg.max_flatline is not None:
        limit = pd.Timedelta(cfg.max_flatline)
        flat = {}
        for col in df.columns:
            x = df[col]
            run_id = (x != x.shift()).cumsum()
            same = x.notna() & (_run_lengths(run_id) * freq > limit)
            if same.any():
                flat[col] = _ranges_from_mask(same, freq)
                df.loc[same, col] = np.nan
        if flat:
            changes["flatlined"] = flat
    return df, changes


def _empty_group_result(index, ref_native, group, reason):
    """Group without usable sensors: processed = raw DTN (or NaN without DTN)."""
    processed = ref_native if ref_native is not None else pd.Series(np.nan, index=index)
    source = pd.Series("model" if ref_native is not None else "", index=index)
    return dict(
        processed=processed.rename(group),
        average=pd.Series(np.nan, index=index, name=f"{group}_average"),
        cleaned=pd.DataFrame(index=index),
        flags=pd.DataFrame(index=index),
        source=source,
        qc_changes={"note": reason},
        details=None,
    )


def backfill_sensor_group(
    df_native: pd.DataFrame, ref_native: pd.Series, cfg: SensorQCConfig, group: str = ""
) -> dict:
    """QC + backfill for one temperature/wind group. ``df_native`` holds the group's sensor columns
    (tz-aware, native resolution), and ``ref_native`` is the DTN reference on the same index.

    Returns dict: ``processed`` (backfilled series), ``average`` (kept-sensor mean),
    ``cleaned`` (sensors with flagged data removed and dropped sensors removed), ``flags``
    (``<col>_FLAG`` at native resolution), ``source``, ``qc_changes``, ``details`` (assessment)
    and ``raw`` (input sensors in output units). Outputs are in DTN's units: temperature is always
    deg C (deg F sensors are converted; ``qc_changes["native_units"]`` records the input unit).
    """
    index = df_native.index
    freq = infer_freq(index)
    step = max(pd.Timedelta(cfg.analysis_freq), freq)
    df, changes = _clean_native(df_native, cfg, freq)
    if df.empty:
        res = _empty_group_result(index, ref_native, group, "no valid sensor data; using DTN")
        res["qc_changes"].update(changes)
        return res

    # units: everything is assessed and returned in DTN's units (deg C for temperature)
    to_ref = lambda v: v
    ref_grid = ref_native.resample(step).mean()
    if cfg.units is not None:
        units = cfg.units
        if units == "auto":
            units = "F" if _detect_fahrenheit(df.resample(step).mean(), ref_grid) else "C"
        if units == "F":
            to_ref = lambda v: (v - 32) / 1.8
        changes["native_units"] = units

    x_native = to_ref(df)
    x = x_native.resample(step).mean()
    ref_grid = ref_grid.reindex(x.index)
    assessed = assess_sensor_group(x, ref_grid, cfg)

    flags = {}
    for col, res in assessed["sensors"].items():
        flag = _labels_to_native(res["label"], index, step)
        flag = flag.where(flag.isin(REMOVAL_FLAGS), "")
        flags[f"{col}_FLAG"] = flag
    df_flags = pd.DataFrame(flags, index=index)
    cleaned, removed_ranges = _remove_flagged_ranges(x_native, df_flags, REMOVAL_FLAGS)
    changes["corrections"] = {c: round(r["correction"], 3) for c, r in assessed["sensors"].items()}
    changes["removed_ranges"] = {c: r for c, r in removed_ranges.items() if r}

    # sensors that are mostly removed carry no information
    if cfg.drop_removed_fraction is not None:
        share = {c: float(cleaned[c].resample(step).mean().isna().mean()) for c in cleaned.columns}
        dropped = {c: round(f, 3) for c, f in share.items() if f >= cfg.drop_removed_fraction}
        if dropped and len(dropped) < cleaned.shape[1]:
            cleaned = cleaned.drop(columns=list(dropped))
            changes["dropped_sensors"] = dropped

    average = cleaned.mean(axis=1)
    kept = average.resample(step).mean().reindex(x.index)
    default = _fit_correction(kept, ref_grid, cfg)
    if not np.isfinite(default):
        default = float(np.median([r["correction"] for r in assessed["sensors"].values()]))
    corr = _anchored_correction(kept, ref_grid, cfg, default)
    model = _apply(ref_native, _grid_to_native(corr, index), cfg.mode)
    if cfg.mode == "gain" or (cfg.value_bounds is not None and cfg.value_bounds[0] == 0):
        model = model.clip(lower=0.0)

    bridged, interpolated = _fill_short_gaps(average, cfg.fill_max_gap, freq)
    missing = bridged.isna()
    processed = bridged.where(~missing, model)
    source = pd.Series(
        np.select([missing & processed.notna(), interpolated], ["model", "interpolated"], "sensor"),
        index=index,
    ).where(processed.notna(), "")
    changes["default_correction"] = round(default, 3)

    assessed["correction"] = corr
    return dict(
        processed=processed.rename(group),
        average=average.rename(f"{group}_average"),
        cleaned=cleaned,
        flags=df_flags,
        source=source,
        qc_changes=changes,
        details=assessed,
        raw=to_ref(df_native),
    )


def _backfill_poa(
    site, df_poa, df_dtn, ref_native, meter_data, meter_valid_mask, config, include_stow, qprint
) -> dict:
    """POA group via POADataset. Falls back to raw DTN POA when no sensor has valid data."""
    index = df_poa.index
    cfg = POAQCConfig() if config is None else replace(config)
    native = infer_freq(index)
    if native > pd.Timedelta(cfg.analysis_freq):
        qprint(f"Note: POA sensor interval {native} > analysis_freq; using {native}.")
        cfg = replace(cfg, analysis_freq=native)
    has_weather = any(
        c in df_dtn.columns for c in list(WEATHER_COLUMNS.values()) + list(SKY_COLUMNS)
    )
    try:
        ds = POADataset(
            site,
            df_poa,
            ref_poa=df_dtn["poa_global"].dropna(),
            ref_meter=meter_data,
            config=cfg,
            meter_valid_mask=meter_valid_mask,
            ref_weather=df_dtn if has_weather else None,
        )
    except ValueError as e:
        if "No valid data" not in str(e) and "No 'OE.POA' sensor" not in str(e):
            raise
        qprint(f"Note: POADataset found no usable POA data ({str(e).splitlines()[0]}); using DTN.")
        res = _empty_group_result(index, ref_native, POA_GROUP, "no valid POA data; using DTN")
        res["poa_dataset"] = None
        return res

    df_final = ds.remove_flagged_ranges(include_stow=include_stow)
    cleaned = df_final[[c for c in ds.poa_columns if c in df_final.columns]]
    return dict(
        processed=df_final["Processed_POA"].rename(POA_GROUP),
        average=df_final["OE.POA_average"].rename(f"{POA_GROUP}_average"),
        cleaned=cleaned,
        flags=ds.df_flags,
        source=ds.poa_source,
        qc_changes=ds.qc_changes,
        details=None,
        poa_dataset=ds,
    )


# ---- main entry point --------------------------------------------------------------------------
def _validate_inputs(df_sensors, df_dtn, meter_data):
    for name, obj in (("df_sensors", df_sensors), ("df_dtn", df_dtn), ("meter_data", meter_data)):
        if obj is None:
            continue
        if not isinstance(obj.index, pd.DatetimeIndex) or obj.index.tz is None:
            raise ValueError(f"{name} must have a timezone-aware DatetimeIndex.")
        if not obj.index.is_monotonic_increasing or obj.index.has_duplicates:
            raise ValueError(f"{name} index must be sorted and unique.")


def backfill_weather_sensor_data(
    site: str,
    df_sensors: pd.DataFrame,
    df_dtn: pd.DataFrame,
    meter_data: Union[pd.Series, pd.DataFrame] = None,
    meter_valid_mask: pd.Series = None,
    poa_config: POAQCConfig = None,
    sensor_configs: dict = None,
    include_stow: bool = True,
    q: bool = True,
) -> dict:
    """QC and backfill the weather sensor groups (``OE.POA``, ``OE.Ambient_Temp``, ``OE.Wind_Speed``).

    Parameters
    ----------
    site : str
        Solar site name (POADataset gets location and tracker design from the site database).
    df_sensors : pd.DataFrame
        Sensor data, tz-aware, regular interval; columns ``OE.POA_*``, ``OE.Ambient_Temp_*``,
        ``OE.Wind_Speed_*``. Other columns are passed through unchanged.
    df_dtn : pd.DataFrame
        Hourly DTN data (interval-start, tz-aware) with ``poa_global``, ``dtn_temp_air``
        (deg C) and ``dtn_wind_speed``. Recognized weather columns (sunshine, precipitation, cloud
        cover, beam/diffuse + clear-sky) are passed on to POADataset.
    meter_data : pd.Series or single-column pd.DataFrame, optional
        Plant meter, the second POA reference (resolves passing clouds, detects plant stow).
    meter_valid_mask : pd.Series, optional
        Boolean mask indexed like ``meter_data``: True where the meter is a valid reference.
    poa_config : POAQCConfig, optional
    sensor_configs : dict, optional
        Group -> ``SensorQCConfig`` or dict of field overrides on ``default_sensor_config(group)``.
    include_stow : bool
        Remove POA stow so that ``OE.POA`` reflects ideal tracking (default True).

    Returns
    -------
    dict
        ``df``: sensor data with flagged data removed, plus ``<group>_average``, ``<group>``
        (processed), the DTN columns and ``site_meter``. All ``OE.Ambient_Temp*`` columns are
        deg C (deg F sensors are converted). ``backfilled``: the processed group columns.
        ``flags``: ``<sensor>_FLAG`` per sensor (native resolution). ``source``: "sensor" /
        "interpolated" / "model" per group. ``raw``: the input sensor columns, unflagged, in the
        same units as ``df``.
        ``qc_changes``: per group. ``details``: per-group assessment (temperature/wind).
        ``poa_dataset``: the POADataset (or None).
    """
    qprint = quiet_print_function(q=q)
    qprint("\nBegin weather data backfill (v3).")
    if isinstance(meter_data, pd.DataFrame):
        if meter_data.shape[1] != 1:
            raise ValueError("meter_data must be a Series or a single-column DataFrame.")
        meter_data = meter_data.iloc[:, 0]
    _validate_inputs(df_sensors, df_dtn, meter_data)
    sensor_configs = sensor_configs or {}
    unknown = set(sensor_configs) - (set(BACKFILL_COL_MAPPING) - {POA_GROUP})
    if unknown:
        raise KeyError(f"sensor_configs has unknown group(s): {sorted(unknown)}")

    index = df_sensors.index
    sensor_cols = _grouped_sensor_cols(df_sensors.columns)
    dtn_native = {
        col: _ref_on_index(df_dtn[col], index)
        for col in BACKFILL_COL_MAPPING.values()
        if col in df_dtn
    }
    if "poa_global" in dtn_native:
        dtn_native["poa_global"] = dtn_native["poa_global"].fillna(0.0).clip(lower=0.0)

    df = df_sensors.copy()
    results = {}
    for group, dtn_col in BACKFILL_COL_MAPPING.items():
        cols = sensor_cols.get(group, [])
        ref = dtn_native.get(dtn_col)
        qprint(f"\nSensor group: {group} ({len(cols)} sensor(s))")
        if ref is None:
            if not cols:
                qprint(f" -> no sensors and no DTN column '{dtn_col}'; skipped.")
                continue
            qprint(f" -> missing DTN column '{dtn_col}'; no QC, using the sensor average.")
            avg = df_sensors[cols].mean(axis=1)
            res = dict(
                processed=avg.rename(group),
                average=avg.rename(f"{group}_average"),
                cleaned=df_sensors[cols],
                flags=pd.DataFrame(index=index),
                source=pd.Series("sensor", index=index).where(avg.notna(), ""),
                qc_changes={"note": f"missing DTN column '{dtn_col}'; no QC"},
                details=None,
            )
        elif group == POA_GROUP:
            if not cols:
                res = _empty_group_result(index, ref, group, "no sensors; using DTN")
                res["poa_dataset"] = None
            else:
                res = _backfill_poa(
                    site,
                    df_sensors[cols],
                    df_dtn,
                    ref,
                    meter_data,
                    meter_valid_mask,
                    poa_config,
                    include_stow,
                    qprint,
                )
        elif not cols:
            res = _empty_group_result(index, ref, group, "no sensors; using DTN")
        else:
            cfg = _resolve_config(group, sensor_configs.get(group))
            res = backfill_sensor_group(df_sensors[cols], ref, cfg, group)
        results[group] = res

        for col in cols:  # flagged data removed; dropped/unusable sensors become all-NaN
            df[col] = res["cleaned"][col] if col in res["cleaned"] else np.nan
        df[f"{group}_average"] = res["average"]
        df[group] = res["processed"]
        _report_group(qprint, group, cols, res)

    for col, series in dtn_native.items():
        df[col] = series
    if meter_data is not None:
        df["site_meter"] = _to_grid(meter_data, index, infer_freq(index))

    processed_cols = [g for g in BACKFILL_COL_MAPPING if g in results]
    qprint("\nEnd backfill.")
    return dict(
        df=df,
        backfilled=df[processed_cols].copy(),
        flags=pd.concat([r["flags"] for r in results.values()], axis=1),
        source=pd.DataFrame({g: r["source"] for g, r in results.items()}),
        raw=pd.concat(  # input sensors in output units (temperature converted to deg C)
            [r.get("raw", df_sensors[sensor_cols.get(g, [])]) for g, r in results.items()]
            + [pd.DataFrame(index=index)],
            axis=1,
        ),
        qc_changes={g: r["qc_changes"] for g, r in results.items()},
        details={g: r["details"] for g, r in results.items() if r.get("details") is not None},
        poa_dataset=results.get(POA_GROUP, {}).get("poa_dataset"),
    )


def _report_group(qprint, group, cols, res):
    flags = res["flags"]
    for col in cols:
        fcol = f"{col}_FLAG"
        if col not in res["cleaned"]:
            qprint(f" -> {col}: dropped / unusable.")
        elif fcol in flags and flags[fcol].ne("").any():
            counts = flags[fcol][flags[fcol].ne("")].value_counts().to_dict()
            qprint(f" -> {col}: flagged {counts}")
        else:
            qprint(f" -> {col}: no data removed.")
    counts = res["source"].value_counts().to_dict()
    qprint(f">>> {group} source counts: {counts}")


# ---- plots -------------------------------------------------------------------------------------
def backfill_subplots(result: dict, freq: Union[str, pd.Timedelta, None] = "15min"):
    """One row per group: DTN reference, raw sensors (hover shows flags), processed series.
    Rows are shaded where the processed value is modelled (salmon) or interpolated (yellow).

    ``freq`` resamples (mean) for plotting when coarser than the data (None: native resolution).
    """
    df, raw, flags, source = result["df"], result["raw"], result["flags"], result["source"]
    groups = list(source.columns)
    n = len(groups)
    if n == 0:
        raise ValueError("Nothing to plot.")
    native = infer_freq(df.index)
    step = native if freq is None or pd.Timedelta(freq) <= native else pd.Timedelta(freq)
    agg = (lambda s: s) if step == native else (lambda s: s.resample(step).mean())
    join = lambda s: ",".join(sorted(set(v for v in s if v)))

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2 / max(n, 2),
        subplot_titles=[f"<b>{g}</b>" for g in groups],
    )
    hover = (
        "<b>%{fullData.name}</b><br>Value: %{y:.2f}<br><i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
    )
    flag_hover = hover.replace("<extra>", "<br>Flags: %{customdata}<extra>")
    shapes = []
    for row, group in enumerate(groups, start=1):
        legend = dict(legendgroup=group, legendgrouptitle_text=f"<b>{group}</b>", legendrank=row)
        dtn_col = BACKFILL_COL_MAPPING[group]
        if dtn_col in df:
            y = agg(df[dtn_col])
            fig.add_trace(
                go.Scattergl(
                    x=y.index,
                    y=y,
                    name=dtn_col,
                    mode="lines",
                    line=dict(color="#777777"),
                    hovertemplate=hover,
                    **legend,
                ),
                row=row,
                col=1,
            )
        for col in [c for c in raw.columns if c.startswith(group)]:
            y = agg(raw[col])
            fcol = f"{col}_FLAG"
            f = flags[fcol] if fcol in flags else pd.Series("", index=df.index)
            f = f.resample(step).agg(join).reindex(y.index).fillna("") if step != native else f
            fig.add_trace(
                go.Scattergl(
                    x=y.index,
                    y=y,
                    name=col,
                    mode="lines",
                    opacity=0.5,
                    customdata=f,
                    hovertemplate=flag_hover,
                    **legend,
                ),
                row=row,
                col=1,
            )
        y = agg(df[group])
        fig.add_trace(
            go.Scattergl(
                x=y.index,
                y=y,
                name=f"{group} (processed)",
                mode="lines",
                line=dict(color="#000000", dash="dot"),
                hovertemplate=hover,
                **legend,
            ),
            row=row,
            col=1,
        )
        for name, color in SOURCE_COLORS.items():
            for idx in _ranges_from_mask(source[group].eq(name), native):
                shapes.append(
                    dict(
                        type="rect",
                        x0=idx[0],
                        x1=idx[-1] + native,
                        y0=0,
                        y1=1,
                        yref=f"y{row} domain".replace("y1 ", "y "),
                        fillcolor=color,
                        opacity=0.4,
                        line_width=0,
                        layer="below",
                    )
                )

    fig.update_layout(
        font_size=9,
        paper_bgcolor="#fbfbfb",
        height=max(350, 280 * n),
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
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=12), x=0, xanchor="left")
    return fig


def flag_summary(result: dict) -> pd.DataFrame:
    """Hours flagged per sensor (columns) and flag (rows)."""
    flags = result["flags"]
    if flags.empty:
        return pd.DataFrame()
    hours = infer_freq(flags.index) / pd.Timedelta(hours=1)
    counts = {
        c.removesuffix("_FLAG"): flags[c].value_counts().drop("", errors="ignore") * hours
        for c in flags.columns
    }
    return pd.DataFrame(counts).fillna(0.0).round(1)
