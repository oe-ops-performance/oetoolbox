"""
Hourly inverter underperformance analysis with site-meter-aware comms detection.

Faster, vectorized successor to 'inverter_analysis.InverterAnalyzer'. Detection
runs on hourly aggregates computed once across the entire (time * inverter)
matrix, instead of looping per (inverter, date). Comms vs offline is decided
per hour from the site-level shortfall ratio: if site_possible - site_meter is
much smaller than site_possible - sum(site_inverters), the missing power is
reaching the meter (comms outage); otherwise it's a real outage.

Hourly categories: curtailment, offline, comms, calibration, underperform.
Tracker (east/west/center) is still detected per day from the AM/noon/PM
power-delta pattern.

Outputs
-------
- 'hourly_flags'      : DataFrame[hour * inverter] of category strings
- 'hourly_loss_kwh'   : DataFrame[hour * inverter] of loss in kWh
- 'daily_analysis'    : {inverter: {date: {category: loss_mwh}}}
- 'flags'             : {inverter: {category: total_mwh}}
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd

from ..datatools.qc import get_sun, get_daylight_timestamps
from ..utils import oemeta
from ..utils.assets import SolarSite
from ..utils.datetime import infer_freq, upsample


def get_representative_solar_noon(site, start_date, end_date) -> pd.Timestamp:
    site_obj = site if isinstance(site, SolarSite) else SolarSite(site)
    tz = site_obj.timezone
    lat, lon = site_obj.coordinates
    total_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
    mid_tstamp = pd.Timestamp(start_date, tz=tz) + pd.Timedelta(days=int(np.floor(total_days / 2)))
    ref_sun = get_sun(date=mid_tstamp, tz=tz, lat=lat, lon=lon)
    return ref_sun.get("noon", None)


@dataclass
class InverterUnderperformanceConfig:
    """Detection thresholds for the hourly analyzer.

    Power thresholds are expressed as fractions of per-inverter AC nameplate
    capacity (kW). They resolve to absolute kW via 'resolve()'. Each fractional
    default has a paired '*_kw' override; when set, the override wins.
    """

    # --- power thresholds (fractions of per-inverter cap) ---
    offline_power_range_frac: float = 0.005  # max within-hour range to call "stuck"
    offline_power_frac: float = 0.01  # mean below this → offline candidate
    comms_power_frac: float = 0.05  # mean in [offline, comms] → comms candidate
    calibration_offset_frac: float = 0.005  # overnight non-zero threshold
    tracker_stow_power_delta_frac: float = 0.02  # min |PM-AM| delta vs expected for tracker

    # absolute kW overrides (take precedence over fractional defaults)
    offline_power_range_kw: Optional[float] = None
    offline_power_threshold_kw: Optional[float] = None
    comms_power_threshold_kw: Optional[float] = None
    calibration_offset_kw: Optional[float] = None
    tracker_stow_power_delta_kw: Optional[float] = None

    # --- underperform (hourly) ---
    underperform_pct_threshold: float = 0.75
    # only evaluate underperform when hourly possible is at least this fraction of cap
    # (avoids flagging low-sun shoulder hours where small absolute deltas blow up the ratio)
    underperform_min_possible_frac: float = 0.05
    # Shoulder hours = the first/last N daylight hours of each day. They get a separate
    # (typically looser) underperform threshold to avoid flagging the natural ramp-up at
    # sunrise and ramp-down at sunset. Set 'underperform_shoulder_hours = 0' to disable.
    # When 'underperform_shoulder_pct_threshold' is None, the main threshold is used.
    underperform_shoulder_hours: int = 0
    underperform_shoulder_pct_threshold: Optional[float] = None

    # --- curtailment ---
    curtailment_flag_value: int = 0  # site flag > this means curtailed
    pct_curtailed_threshold: float = 0.625  # > this fraction of an hour curtailed → curtail flag

    # --- tracker (daily) ---
    tracker_hours: Tuple[int, int, int] = (9, 12, 15)
    solar_noon_offset_hours: float = 2.0
    # Tracker detection only runs on stable days. A cloudy afternoon mimics a
    # west-stuck pattern, and curtailment depresses measured power for non-mechanical
    # reasons - both produce false positives.
    tracker_require_clearsky_day: bool = True  # set False to skip the clearsky gate
    tracker_clearsky_min_frac: float = 0.7  # min fraction of daylight minutes flagged clearsky
    tracker_max_curtailment_frac: float = 0.1  # max fraction of daylight hours curtailed

    # --- clearsky detection ---
    # measured GHI source for pvlib detect_clearsky: 'dtn' (DTN modeled, from
    # self.data['dtn']['dtn_ghi']) or 'sensors' (on-site GHI sensor data, from
    # self.data['met_stations'][SENSOR_GHI_COL]). The other source is used as a
    # fallback when the chosen one isn't available.
    clearsky_ghi_source: str = "dtn"
    clearsky_detect_window: int = 10  # pvlib detect_clearsky window_length (minutes)

    # --- calibration ---
    calibration_hour: int = 1

    # --- daylight inclusion ---
    # an hour is "daylight" if at least this fraction of its minutes are in daylight
    daylight_min_hour_frac: float = 0.5
    # bounds for the daylight window on each day. ('dawn', 'dusk') uses civil
    # twilight; ('sunrise', 'sunset') uses the tighter geometric-horizon window.
    daylight_limits: Tuple[str, str] = ("dawn", "dusk")

    # --- comms vs offline (site-meter heuristic) ---
    # per hour: comms_share = clip(1 - meter_shortfall / inv_shortfall, 0, 1).
    # If comms_share > this and the inverter is flatlined+low → comms; else offline.
    comms_share_threshold: float = 0.5
    # Minimum site inverter shortfall (kW) to compute the ratio at all; below this
    # the denominator is too noisy and we default to offline.
    comms_min_inv_shortfall_kw: Optional[float] = None  # default = 1% of site capacity

    def resolve(self, inv_capacity_kw: float, site_capacity_kw: float) -> "ResolvedThresholds":
        def _pick(override: Optional[float], frac: float, base_kw: float) -> float:
            return override if override is not None else frac * base_kw

        comms_min = self.comms_min_inv_shortfall_kw
        if comms_min is None:
            comms_min = 0.01 * site_capacity_kw

        return ResolvedThresholds(
            offline_power_range=_pick(
                self.offline_power_range_kw, self.offline_power_range_frac, inv_capacity_kw
            ),
            offline_power_threshold=_pick(
                self.offline_power_threshold_kw, self.offline_power_frac, inv_capacity_kw
            ),
            comms_power_threshold=_pick(
                self.comms_power_threshold_kw, self.comms_power_frac, inv_capacity_kw
            ),
            calibration_offset=_pick(
                self.calibration_offset_kw, self.calibration_offset_frac, inv_capacity_kw
            ),
            tracker_stow_power_delta_threshold=_pick(
                self.tracker_stow_power_delta_kw,
                self.tracker_stow_power_delta_frac,
                inv_capacity_kw,
            ),
            underperform_min_possible=self.underperform_min_possible_frac * inv_capacity_kw,
            comms_min_inv_shortfall=comms_min,
        )


@dataclass(frozen=True)
class ResolvedThresholds:
    offline_power_range: float
    offline_power_threshold: float
    comms_power_threshold: float
    calibration_offset: float
    tracker_stow_power_delta_threshold: float
    underperform_min_possible: float
    comms_min_inv_shortfall: float


class _HourlyAnalyzerBase:
    """Shared setup and helper logic for hourly per-asset analyzers.

    Both InverterAnalyzer (site → inverters) and CombinerAnalyzer (inverter →
    combiners) share identical logic for:
    - frequency detection and normalization across data groups
    - daylight boundary and solar noon resolution
    - daylight and shoulder-hour detection
    - clearsky day and curtailment-fraction daily summaries

    This base class provides all shared methods. Subclasses must define:
    - self.site (SolarSite instance)
    - self.data (dict[str, DataFrame])
    - self.config (config dataclass with tz, lat, lon, daylight_limits, etc.)
    - hourly_index property (pd.DatetimeIndex of hourly results)
    - native_index property (pd.DatetimeIndex of raw-freq data)
    """

    @property
    def hourly_index(self) -> pd.DatetimeIndex:
        raise NotImplementedError("Subclass must implement hourly_index property")

    @property
    def native_index(self) -> pd.DatetimeIndex:
        raise NotImplementedError("Subclass must implement native_index property")

    def _detect_frequencies(self) -> None:
        self.frequencies = {}
        for group, df in self.data.items():
            if len(df) < 2:
                raise ValueError(f"Insufficient data in {group}")
            self.frequencies[group] = infer_freq(df.index)
        self.freq_td = min(self.frequencies.values())

    def _normalize_frequencies(self) -> None:
        matching_grps = [g for g, f in self.frequencies.items() if f == self.freq_td]
        matching_ = [df for g, df in self.data.items() if g in matching_grps]
        if not matching_:
            raise ValueError("Could not find dataframes with matching freq.")

        idx0 = matching_[0].index
        for df in matching_[1:]:
            if not df.index.equals(idx0):
                raise ValueError("One or more dataframes have different datetime index.")
        self.datetime_index = idx0.copy()

        self.start = self.datetime_index.min().floor("D")
        self.end = self.datetime_index.max()
        if self.end.hour > 0:
            self.end = self.end.ceil("D")
        self.start_date = self.start.strftime("%Y-%m-%d")
        self.end_date = self.end.strftime("%Y-%m-%d")

        for group in [g for g in self.data.keys() if g not in matching_grps]:
            self.data[group] = upsample(
                self.data[group], new_freq=self.freq_td, agg_type="interpolate"
            )
            self.frequencies[group] = self.freq_td

    def _generate_analysis_date_attributes(self) -> None:
        self.daylight_timestamps = get_daylight_timestamps(
            start_date=pd.Timestamp(self.start_date, tz=self.tz),
            end_date=pd.Timestamp(self.end_date, tz=self.tz),
            tz=self.tz,
            lat=self.lat,
            lon=self.lon,
            freq=self.freq_td,
            limits=tuple(self.config.daylight_limits),
        )
        self.analysis_dates = pd.date_range(
            self.start_date, self.end_date, freq="D", inclusive="left"
        ).date.tolist()

    def _get_tracker_hours_for_solar_noon(self) -> None:
        try:
            solar_noon = get_representative_solar_noon(
                self.site.name, self.start_date, self.end_date
            )
            if solar_noon is None:
                raise ValueError("solar noon resolution returned None")
            solar_noon_hour = solar_noon.round("H").hour
            offset_hrs = int(self.config.solar_noon_offset_hours)
            am_hour = max(0, solar_noon_hour - offset_hrs)
            pm_hour = min(23, solar_noon_hour + offset_hrs)
            self.tracker_hours = (am_hour, solar_noon_hour, pm_hour)
        except Exception:
            self.tracker_hours = self.config.tracker_hours

    def _compute_daylight_hour_mask(self) -> pd.Series:
        """Bool series over hourly index - True if hour is sufficiently in daylight."""
        hours = self.hourly_index
        if not self.daylight_timestamps:
            return pd.Series(False, index=hours)
        dl_native = pd.Series(False, index=self.native_index)
        dl_idx = pd.DatetimeIndex(self.daylight_timestamps)
        dl_native.loc[dl_native.index.intersection(dl_idx)] = True
        frac = dl_native.astype(float).resample("1H").mean()
        return frac.reindex(hours, fill_value=False) >= self.config.daylight_min_hour_frac

    def _compute_shoulder_hour_mask(self) -> pd.Series:
        """Bool series over hourly index - True for first/last N daylight hours of each day.

        N comes from config.underperform_shoulder_hours. If a day has 2*N or fewer
        daylight hours, every daylight hour on that day is treated as a shoulder.
        """
        hours = self.hourly_index
        out = pd.Series(False, index=hours)
        n = int(self.config.underperform_shoulder_hours)
        if n <= 0:
            return out
        daylight_h = self._compute_daylight_hour_mask()
        daylight_idx = hours[daylight_h.values]
        if len(daylight_idx) == 0:
            return out
        df_dl = pd.DataFrame({"date": daylight_idx.date}, index=daylight_idx)
        asc = df_dl.groupby("date").cumcount()
        total = df_dl.groupby("date")["date"].transform("count")
        is_shoulder = (asc < n) | (asc >= total - n)
        out.loc[daylight_idx[is_shoulder.values]] = True
        return out


INV_ANALYSIS_DATA_KEYS = ["inverters", "meter", "site", "pvlib", "dtn", "met_stations"]
REQUIRED_ANALYSIS_KEYS = ["inverters", "meter", "pvlib"]
SENSOR_GHI_COL = "Processed_GHI"

HOURLY_CATEGORIES = ["curtailment", "offline", "comms", "calibration", "underperform"]
DAILY_TRACKER_CATEGORIES = ["tracker_east", "tracker_west", "tracker_center"]
ALL_CATEGORIES = HOURLY_CATEGORIES + DAILY_TRACKER_CATEGORIES

FLAG_COLORS = {
    "offline": "#9e9e9e",
    "comms": "#f0ad4e",
    "calibration": "#f7dc6f",
    "curtailment": "#5dade2",
    "underperform": "#e74c3c",
    # Tracker family - same purple hue, three lightness steps (east → center → west)
    "tracker_east": "#d2b4de",  # pale lilac
    "tracker_center": "#bb8fce",  # medium purple
    "tracker_west": "#a569bd",  # deep purple
}


class InverterAnalyzer(_HourlyAnalyzerBase):
    """Hourly underperformance analyzer.

    Parameters
    ----------
    site_name : str, optional
        Site name. Required if 'data' is a plain dict.
    data : dict | SolarAnalysisDataset
        Input data. When a dict, must contain 'inverters', 'meter', 'pvlib'
        DataFrames; 'site' (curtailment) is optional.
    inv_col_pattern, pvl_col_pattern : str
        Column patterns used to filter inverter and pvlib columns.
    config : InverterUnderperformanceConfig, optional
    """

    @property
    def hourly_index(self) -> pd.DatetimeIndex:
        return self.hourly_mean_kw.index

    @property
    def native_index(self) -> pd.DatetimeIndex:
        return self.df_measured_kw.index

    def __init__(
        self,
        site_name: str = None,
        data=None,
        inv_col_pattern: str = None,
        pvl_col_pattern: str = None,
        config: Optional[InverterUnderperformanceConfig] = None,
    ):
        from ..utils.solar2 import SolarAnalysisDataset

        if isinstance(data, SolarAnalysisDataset):
            if site_name is None:
                site_name = data.site.name
            data = dict(data.data)

        if data is None:
            data = {}

        if site_name is None:
            raise ValueError("site_name must be provided when data is not a SolarAnalysisDataset.")

        self.site = SolarSite(site_name)
        self.config = config or InverterUnderperformanceConfig()

        self.tz, self.lat, self.lon = self._extract_location_metadata()

        self.ac_capacity_kw = self._get_ac_capacity()
        self.inverter_names = self.site.asset_names_by_group.get("Inverters", [])
        self.n_inv = len(self.inverter_names)
        self.inv_ac_capacity_kw = self.ac_capacity_kw / self.n_inv
        self.thresholds = self.config.resolve(self.inv_ac_capacity_kw, self.ac_capacity_kw)
        self.inv_col_pattern = inv_col_pattern or "OE.ActivePower_"
        self.pvl_col_pattern = pvl_col_pattern or "_Possible_Power"
        self.df_inv_pvl = None
        self.daily_totals = None

        self._process_input_data(data)
        self._detect_frequencies()
        self._normalize_frequencies()

        self._generate_analysis_date_attributes()
        self._process_curtailment_data()
        self._get_tracker_hours_for_solar_noon()

        self._build_native_frames()

        # populated by run_analysis
        self.hourly_flags: pd.DataFrame = pd.DataFrame()
        self.hourly_loss_kwh: pd.DataFrame = pd.DataFrame()
        self.flags: Dict[str, Dict[str, float]] = {}
        self.daily_analysis: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _extract_location_metadata(self) -> Tuple[str, float, float]:
        tz = self.site.timezone
        lat, lon = self.site.coordinates
        if any(x is None for x in (tz, lat, lon)):
            raise ValueError(f"Missing location metadata for site {self.site.name}")
        return tz, lat, lon

    def _get_ac_capacity(self) -> float:
        try:
            capacity_mw = oemeta.data["SystemSize"]["MWAC"].get(self.site.name)
            if capacity_mw is None:
                raise ValueError(f"No AC capacity found for {self.site.name}")
            return float(capacity_mw) * 1000
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Failed to load AC capacity for {self.site.name}: {e}")

    def _process_input_data(self, data):
        valid_data = {}
        for key in INV_ANALYSIS_DATA_KEYS:
            df = data.get(key, None)
            if df is None:
                if key not in REQUIRED_ANALYSIS_KEYS:
                    continue
                raise KeyError(f"No data found for {key}.")
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Provided {key} data is not a dataframe.")
            if df.empty and key not in REQUIRED_ANALYSIS_KEYS:
                continue
            valid_data[key] = df
        self.data = valid_data

    def _process_curtailment_data(self) -> None:
        flag_col, setpoint_col = "OE.CurtailmentStatus", "OE.SiteSetpoint"
        df = self.data.get("site", pd.DataFrame())
        if df.empty or not all(c in df.columns for c in [flag_col, setpoint_col]):
            self.curt_flag_col = None
            self.curt_setpoint_col = None
            return
        self.curt_flag_col = flag_col
        self.curt_setpoint_col = setpoint_col

    def _build_native_frames(self) -> None:
        """Materialize per-inverter measured (kW) and possible (kW) frames at native freq."""
        dfi = self.data["inverters"].filter(like=self.inv_col_pattern).copy()
        dfi.columns = [c.replace(self.inv_col_pattern, "") for c in dfi.columns]
        if not all(inv in dfi.columns for inv in self.inverter_names):
            missing = [inv for inv in self.inverter_names if inv not in dfi.columns]
            raise ValueError(f"Inverter columns missing from data: {missing[:5]}...")
        self.df_measured_kw = dfi[self.inverter_names]

        dfp = self.data["pvlib"].filter(like=self.pvl_col_pattern).copy()
        dfp.columns = [c.replace(self.pvl_col_pattern, "") for c in dfp.columns]
        # Align possible columns to inverter order; fill missing with 0
        self.df_possible_kw = dfp.reindex(columns=self.inverter_names).fillna(0.0)

        meter_col = self.data["meter"].columns[0]
        self.site_meter_mw = self.data["meter"][meter_col].astype(float)

        # Combined wide frame for plotting (inv + inv_possible)
        possible_renamed = self.df_possible_kw.rename(columns=lambda c: f"{c}_possible")
        self.df_inv_pvl = pd.concat([self.df_measured_kw, possible_renamed], axis=1)
        self._process_generation_totals()

    def _process_generation_totals(self) -> None:
        df = self._mwh_from_interval_kw(self.df_inv_pvl)
        for inv in self.inverter_names:
            poss_col = f"{inv}_possible"
            df[f"{inv}_%-loss"] = df[poss_col].sub(df[inv]).div(df[poss_col])
        sorted_cols = sorted(df.columns, key=lambda c: c.replace("%", "z"))
        self.daily_totals = df[sorted_cols].copy()

    def _mwh_from_interval_kw(self, df, freq="D") -> pd.DataFrame:
        # e.g. kW -> kWh
        pwr2nrg_multiplier = infer_freq(df.index) / pd.Timedelta(hours=1)

        # divide by 1,000 to get MW, then convert to MWh via multiplier
        if freq is not None:
            return df.resample("D").sum().div(1e3).mul(pwr2nrg_multiplier)
        return df.sum().div(1e3).mul(pwr2nrg_multiplier)

    def get_total_mwh(self, days: list[str]) -> dict[str, pd.DataFrame]:
        if self.daily_totals is None:
            return {}

        _in_range = self.daily_totals.index.strftime("%Y-%m-%d").isin(days)
        totals = self.daily_totals[_in_range].sum()

        df_list = []
        for lbl, sfx in zip(["mwh", "possible"], ["", "_possible"]):
            filtercols = [f"{inv}{sfx}" for inv in self.inverter_names]
            dff = totals[totals.index.isin(filtercols)].to_frame(lbl).copy()
            dff.index = dff.index.str.replace(sfx, "")
            df_list.append(dff)

        df = pd.concat(df_list, axis=1)

        df["%-loss"] = df["possible"].sub(df["mwh"]).div(df["possible"])
        return df

    # ------------------------------------------------------------------
    # vectorized hourly aggregates
    # ------------------------------------------------------------------
    def _compute_hourly_aggregates(self) -> None:
        """Compute hourly per-inverter aggregates in a single vectorized pass.

        Builds:
            self.hourly_mean_kw       (hours * inv) - mean measured power
            self.hourly_possible_kw   (hours * inv) - mean possible power
            self.hourly_range_kw      (hours * inv) - within-hour max-min
            self.hourly_loss_raw_kwh  (hours * inv) - energy loss for the hour
                                                     (mean_kW * 1 h, clipped ≥ 0)
        """
        m = self.df_measured_kw
        p = self.df_possible_kw

        rs_m = m.resample("1H")
        self.hourly_mean_kw = rs_m.mean()
        self.hourly_range_kw = rs_m.max().sub(rs_m.min())

        self.hourly_possible_kw = p.resample("1H").mean()

        # Each row is a 1-h window, so mean(kW) over the row = kWh for that hour.
        self.hourly_loss_raw_kwh = (self.hourly_possible_kw - self.hourly_mean_kw).clip(lower=0.0)

    def _compute_hourly_site_signals(self) -> None:
        """Compute hourly site-level signals for comms/offline classification.

        Produces:
            self.site_inverters_mw_h  (hourly Series, MW)
            self.site_possible_mw_h   (hourly Series, MW)
            self.site_meter_mw_h      (hourly Series, MW)
            self.comms_share_h        (hourly Series in [0, 1])
            self.curt_hour_mask       (hourly Series, bool) - True means hour curtailed
        """
        # Sum across inverters at native freq, divide by 1000 (kW → MW), then resample
        site_inv_kw = self.df_measured_kw.sum(axis=1)
        site_poss_kw = self.df_possible_kw.sum(axis=1)

        self.site_inverters_mw_h = site_inv_kw.div(1e3).resample("1H").mean()
        self.site_possible_mw_h = site_poss_kw.div(1e3).resample("1H").mean()
        self.site_meter_mw_h = self.site_meter_mw.resample("1H").mean()

        inv_short_kw = (self.site_possible_mw_h - self.site_inverters_mw_h).clip(lower=0.0).mul(1e3)
        meter_short_kw = (self.site_possible_mw_h - self.site_meter_mw_h).clip(lower=0.0).mul(1e3)

        # comms_share = fraction of inv shortfall that is *not* reflected at the meter
        denom = inv_short_kw.where(inv_short_kw >= self.thresholds.comms_min_inv_shortfall, np.nan)
        share = 1.0 - meter_short_kw.div(denom)
        self.comms_share_h = share.clip(0.0, 1.0).fillna(0.0)

        # Curtailment mask per hour
        if self.curt_flag_col is not None:
            flag_series = self.data["site"][self.curt_flag_col] > self.config.curtailment_flag_value
            curt_frac_h = flag_series.astype(float).resample("1H").mean()
            self.curt_hour_mask = curt_frac_h > self.config.pct_curtailed_threshold
        else:
            self.curt_hour_mask = pd.Series(False, index=self.hourly_mean_kw.index)

        # Per-hour curtailment loss per inverter (kWh) - used when curt_hour_mask is True
        self.curtailment_loss_kwh_h = self._compute_hourly_curtailment_loss()

    def _compute_hourly_curtailment_loss(self) -> pd.DataFrame:
        """Per-(hour, inverter) curtailment loss in kWh.

        Loss = min(possible - measured, possible - setpoint), clipped at 0.
        The setpoint term caps it from above (no inverter is "more curtailed"
        than the gap between possible and the cap); the measured term caps it
        from below (an offline inverter's full shortfall is not all curtailment
        - the portion below setpoint stays available for the offline bucket).
        """
        if not self.curt_setpoint_col:
            return self.hourly_loss_raw_kwh

        setpoint_kw_per_inv = (
            self.data["site"][self.curt_setpoint_col].astype(float).mul(1000.0 / self.n_inv)
        )
        sp_kw_h = setpoint_kw_per_inv.resample("1H").mean().reindex(self.hourly_possible_kw.index)

        sp_arr = sp_kw_h.to_numpy()[:, None]
        poss_arr = self.hourly_possible_kw.to_numpy()
        meas_arr = self.hourly_mean_kw.to_numpy()

        setpoint_gap = np.clip(poss_arr - sp_arr, 0.0, None)
        actual_gap = np.clip(poss_arr - meas_arr, 0.0, None)
        loss_arr = np.minimum(actual_gap, setpoint_gap)

        loss = pd.DataFrame(
            loss_arr, index=self.hourly_possible_kw.index, columns=self.inverter_names
        )
        return loss.fillna(0.0)

    def _resolve_measured_ghi(self) -> Optional[pd.Series]:
        """Return the measured GHI series for clearsky detection, honoring config.

        Tries 'config.clearsky_ghi_source' first, falls back to the other source
        if unavailable. Returns 'None' if neither source has usable data.
        """
        source = self.config.clearsky_ghi_source
        if source not in ("dtn", "sensors"):
            raise ValueError(f"Invalid clearsky_ghi_source {source!r}; must be 'dtn' or 'sensors'.")

        def _from_dtn() -> Optional[pd.Series]:
            df = self.data.get("dtn")
            if df is None or df.empty or "dtn_ghi" not in df.columns:
                return None
            return df["dtn_ghi"]

        def _from_sensors() -> Optional[pd.Series]:
            df = self.data.get("met_stations")
            if df is None or df.empty or SENSOR_GHI_COL not in df.columns:
                return None
            return df[SENSOR_GHI_COL]

        order = (_from_dtn, _from_sensors) if source == "dtn" else (_from_sensors, _from_dtn)
        for getter in order:
            series = getter()
            if series is not None:
                return series
        return None

    def _compute_clearsky_day_mask(self) -> Dict:
        """Per-date dict {date: is_clearsky_day}.

        Uses pvlib's 'detect_clearsky' against the DTN 'ghi_clearsky' reference.
        Measured GHI comes from 'config.clearsky_ghi_source' - ''dtn'' reads
        'dtn_ghi' from the dtn group; ''sensors'' reads 'Processed_GHI' from
        the met_stations group. The other source is used as a fallback. A day
        qualifies as clearsky when at least 'tracker_clearsky_min_frac' of its
        daylight minutes pass the test.

        Returns an empty dict when neither source is available - the tracker filter
        treats this as "no clearsky info available" and (when
        'tracker_require_clearsky_day=True') skips tracker detection entirely.
        """
        df_dtn = self.data.get("dtn")
        if df_dtn is None or df_dtn.empty or "ghi_clearsky" not in df_dtn.columns:
            return {}

        measured = self._resolve_measured_ghi()
        if measured is None:
            return {}

        try:
            from pvlib.clearsky import detect_clearsky
        except ImportError:
            return {}

        clearsky_ghi = df_dtn["ghi_clearsky"].fillna(0)
        cs_flag = detect_clearsky(
            measured=measured.reindex(clearsky_ghi.index).fillna(0),
            clearsky=clearsky_ghi,
            window_length=self.config.clearsky_detect_window,
        ).astype(int)

        # Restrict to daylight, then average per date.
        if self.daylight_timestamps:
            dl_idx = pd.DatetimeIndex(self.daylight_timestamps)
            cs_flag = cs_flag.loc[cs_flag.index.intersection(dl_idx)]
        if cs_flag.empty:
            return {}
        daily_frac = cs_flag.groupby(cs_flag.index.date).mean()
        threshold = self.config.tracker_clearsky_min_frac
        return {d: bool(f >= threshold) for d, f in daily_frac.items()}

    def _compute_curtailment_frac_by_date(self) -> Dict:
        """Per-date dict {date: fraction of daylight hours curtailed}."""
        if self.curt_hour_mask is None or len(self.curt_hour_mask) == 0:
            return {}
        daylight_h = self._compute_daylight_hour_mask()
        df = pd.DataFrame({"curt": self.curt_hour_mask, "dl": daylight_h})
        df["date"] = df.index.date
        out = {}
        for date, grp in df.groupby("date"):
            dl_count = int(grp["dl"].sum())
            if dl_count == 0:
                out[date] = 0.0
            else:
                out[date] = float((grp["curt"] & grp["dl"]).sum()) / dl_count
        return out

    def _get_daylight_bounds_by_date(self) -> dict:
        """Return '{date: (daylight_start_ts, daylight_end_ts)}' for shading.

        Cached on first call. Bounds are derived from 'self.daylight_timestamps',
        so they reflect the same dawn/dusk window the analyzer uses.
        """
        cached = getattr(self, "_daylight_bounds_cache", None)
        if cached is not None:
            return cached
        bounds: dict = {}
        if self.daylight_timestamps:
            dl_idx = pd.DatetimeIndex(self.daylight_timestamps)
            df = pd.DataFrame({"ts": dl_idx, "date": dl_idx.date})
            agg = df.groupby("date")["ts"].agg(["min", "max"])
            for row in agg.itertuples():
                # Add freq_td to max so the band ends after the last daylight sample.
                bounds[row.Index] = (row.min, row.max + self.freq_td)
        self._daylight_bounds_cache = bounds
        return bounds

    # ------------------------------------------------------------------
    # detection
    # ------------------------------------------------------------------
    def run_analysis(
        self,
        only_daylight: bool = True,
        detect_tracker: bool = True,
        printouts: bool = False,
    ) -> None:
        """Run hourly + daily detection across all inverters.

        Sets 'self.hourly_flags', 'self.hourly_loss_kwh', 'self.daily_analysis',
        and 'self.flags'.

        Parameters
        ----------
        only_daylight : bool
            Restrict hourly detection to daylight hours.
        detect_tracker : bool
            When False, skip the daily tracker pattern check entirely. No
            'tracker_*' categories will appear in the output and the day's
            'underperform' entries are left in place.
        printouts : bool
        """
        qprint = print if printouts else (lambda *_: None)

        qprint("Computing hourly aggregates...")
        self._compute_hourly_aggregates()
        self._compute_hourly_site_signals()

        qprint("Classifying hourly flags...")
        self._classify_hourly(only_daylight=only_daylight)

        if detect_tracker:
            qprint("Detecting daily tracker patterns...")
            self._detect_tracker_daily()
        else:
            qprint("Skipping tracker detection (detect_tracker=False).")
            self._tracker_by_inv_date = {}

        qprint("Building daily rollup...")
        self._build_daily_rollup()

        qprint("Done.")

    def _classify_hourly(self, only_daylight: bool) -> None:
        """Vectorized hourly classification.

        Priority (highest first):
            1. curtailment   - site curtailment flag majority of hour
            2. offline/comms - flatlined + low; comms iff site comms_share is high
            3. calibration   - overnight (calibration_hour) non-zero
            4. underperform  - measured/possible < threshold (with min-possible gate)
        """
        thr = self.thresholds
        hours = self.hourly_mean_kw.index
        n_inv = len(self.inverter_names)

        daylight_mask = self._compute_daylight_hour_mask()
        analysable = daylight_mask if only_daylight else pd.Series(True, index=hours)

        flags = pd.DataFrame(index=hours, columns=self.inverter_names, dtype=object)
        loss_kwh = pd.DataFrame(0.0, index=hours, columns=self.inverter_names)

        # --- broadcast helpers (1D row-mask → 2D mask of shape (hours, inv)) ---
        def row_mask(s: pd.Series) -> pd.DataFrame:
            arr = np.broadcast_to(s.to_numpy()[:, None], (len(s), n_inv)).copy()
            return pd.DataFrame(arr, index=hours, columns=self.inverter_names)

        analysable_2d = row_mask(analysable)
        curt_hour_2d = row_mask(self.curt_hour_mask & analysable)
        comms_high_2d = row_mask(self.comms_share_h.gt(self.config.comms_share_threshold))

        # Per-inverter "broken, not curtailed" gate: stuck within-hour AND mean below
        # the offline threshold (~1% cap). A healthy inverter complying with curtailment
        # sits at setpoint (typically well above 1% cap), so this gate excludes only
        # essentially-zero inverters - preventing real outages from being painted as
        # curtailment when the site flag is on.
        stuck = self.hourly_range_kw.lt(thr.offline_power_range)
        inv_dead = self.hourly_mean_kw.lt(thr.offline_power_threshold)
        inv_broken_2d = stuck & inv_dead

        # --- curtailment ---
        curt_assign = curt_hour_2d & ~inv_broken_2d
        if curt_assign.any().any():
            flags = flags.mask(curt_assign, "curtailment")
            loss_kwh = loss_kwh.mask(curt_assign, self.curtailment_loss_kwh_h)

        # --- offline / comms ---
        # ~curt_assign (not ~curt_hour_2d): cells excluded from curtailment above can
        # fall through to offline so a broken inverter during a site-curtailment hour
        # lands in the right bucket.
        low_or_dim = self.hourly_mean_kw.lt(thr.comms_power_threshold)
        outage = stuck & low_or_dim & analysable_2d & ~curt_assign
        offline_mask = outage & ~comms_high_2d
        comms_mask = outage & comms_high_2d

        if offline_mask.any().any():
            flags = flags.mask(offline_mask, "offline")
            loss_kwh = loss_kwh.mask(offline_mask, self.hourly_loss_raw_kwh)

        if comms_mask.any().any():
            flags = flags.mask(comms_mask, "comms")
            # comms = power reached the meter; loss reported as 0 (matches v1)
            loss_kwh = loss_kwh.mask(comms_mask, 0.0)

        # --- calibration: at config.calibration_hour, regardless of daylight ---
        calib_hr_1d = pd.Series(hours.hour == self.config.calibration_hour, index=hours)
        calib_hr_2d = row_mask(calib_hr_1d)
        calib_high = self.hourly_mean_kw.gt(thr.calibration_offset)
        calib_assign = calib_high & calib_hr_2d & flags.isna()
        if calib_assign.any().any():
            flags = flags.mask(calib_assign, "calibration")
            # calibration loss is informational only (no MWh tally in v1) → leave 0

        # --- underperform: analysable hours w/ enough sun, no prior flag ---
        # Shoulder hours (first/last N daylight hours) get a looser threshold when
        # configured, since the natural ramp-up/ramp-down depresses the ratio.
        sun_mask = self.hourly_possible_kw.gt(thr.underperform_min_possible)
        ratio = self.hourly_mean_kw.div(self.hourly_possible_kw.where(self.hourly_possible_kw > 0))

        shoulder_thr = (
            self.config.underperform_shoulder_pct_threshold
            if self.config.underperform_shoulder_pct_threshold is not None
            else self.config.underperform_pct_threshold
        )
        shoulder_2d = row_mask(self._compute_shoulder_hour_mask())

        under_main = (
            ratio.lt(self.config.underperform_pct_threshold)
            & ~shoulder_2d
            & sun_mask
            & analysable_2d
            & flags.isna()
        )
        under_shoulder = (
            ratio.lt(shoulder_thr) & shoulder_2d & sun_mask & analysable_2d & flags.isna()
        )
        under_mask = under_main | under_shoulder
        if under_mask.any().any():
            flags = flags.mask(under_mask, "underperform")
            loss_kwh = loss_kwh.mask(under_mask, self.hourly_loss_raw_kwh)

        self.hourly_flags = flags
        self.hourly_loss_kwh = loss_kwh

    def _detect_tracker_daily(self) -> None:
        """Per-(inverter, date) tracker detection on hourly means.

        Compares measured PM-AM delta to expected (possible) PM-AM delta.
        Stuck west: measured delta ≫ expected.
        Stuck east: measured delta ≪ expected.
        Center: all three target hours underperform vs possible by > threshold.
        Stored in 'self._tracker_by_inv_date[inv][date] = ('east'|'west'|'center', loss_mwh)'.
        """
        am_hr, noon_hr, pm_hr = self.tracker_hours
        thr = self.thresholds.tracker_stow_power_delta_threshold

        m_h = self.hourly_mean_kw
        p_h = self.hourly_possible_kw
        idx = m_h.index

        am_m = m_h[idx.hour == am_hr].copy()
        nn_m = m_h[idx.hour == noon_hr].copy()
        pm_m = m_h[idx.hour == pm_hr].copy()
        am_p = p_h[idx.hour == am_hr].copy()
        nn_p = p_h[idx.hour == noon_hr].copy()
        pm_p = p_h[idx.hour == pm_hr].copy()

        # align by date
        for df in (am_m, nn_m, pm_m, am_p, nn_p, pm_p):
            df.index = df.index.date

        # daily perf ratio (MWh basis) - only run tracker on days that underperformed
        daily_meas_mwh = m_h.resample("D").mean().mul(24).div(1e3)  # mean kW * 24 h / 1000
        daily_poss_mwh = p_h.resample("D").mean().mul(24).div(1e3)
        daily_meas_mwh.index = daily_meas_mwh.index.date
        daily_poss_mwh.index = daily_poss_mwh.index.date

        # Day-level gates: tracker pattern detection is unreliable when (a) clouds
        # depress afternoon power (mimics west-stuck) or (b) curtailment depresses
        # measured power for non-mechanical reasons. Skip those days entirely.
        clearsky_by_date = self._compute_clearsky_day_mask()
        curt_frac_by_date = self._compute_curtailment_frac_by_date()
        max_curt = self.config.tracker_max_curtailment_frac
        require_cs = self.config.tracker_require_clearsky_day
        # When the user requires clearsky filtering but DTN data isn't available,
        # `clearsky_by_date` is empty → no day qualifies, so no tracker is detected.
        cs_info_available = bool(clearsky_by_date)

        # Per-inverter disqualifier: any offline/comms/curtailment hour on a date
        # makes the AM/noon/PM samples for that day unreliable for that inverter.
        # An offline-all-day inverter has measured ≈ 0 at every hour, which falsely
        # reads as a stuck-east pattern (meas_delta ≈ 0, exp_delta > 0 → diff < 0).
        disqualifying = {"offline", "comms", "curtailment"}
        inv_skip_dates: Dict[str, set] = {inv: set() for inv in self.inverter_names}
        if not self.hourly_flags.empty:
            for inv in self.inverter_names:
                if inv not in self.hourly_flags.columns:
                    continue
                s = self.hourly_flags[inv].dropna()
                if s.empty:
                    continue
                mask = s.isin(disqualifying)
                inv_skip_dates[inv] = set(s.index[mask].date)

        tracker_result: Dict[str, Dict] = {inv: {} for inv in self.inverter_names}

        for date in self.analysis_dates:
            if curt_frac_by_date.get(date, 0.0) > max_curt:
                continue
            if require_cs:
                if not cs_info_available or not clearsky_by_date.get(date, False):
                    continue
            for inv in self.inverter_names:
                if date in inv_skip_dates[inv]:
                    continue
                poss_d = daily_poss_mwh.at[date, inv] if date in daily_poss_mwh.index else 0.0
                meas_d = daily_meas_mwh.at[date, inv] if date in daily_meas_mwh.index else 0.0
                if poss_d < 1e-3:
                    continue
                # if meas_d / poss_d >= self.config.underperform_pct_threshold:
                #     continue

                a_m = am_m.at[date, inv]
                n_m = nn_m.at[date, inv]
                p_m = pm_m.at[date, inv]
                a_p = am_p.at[date, inv]
                n_p = nn_p.at[date, inv]
                p_p = pm_p.at[date, inv]
                if any(pd.isna(x) for x in (a_m, n_m, p_m, a_p, n_p, p_p)):
                    continue

                # meas_delta = p_m - a_m
                # exp_delta = p_p - a_p
                # diff = meas_delta - exp_delta
                # tracker_type = None
                # if abs(diff) > thr:
                #     tracker_type = "west" if diff > 0 else "east"
                # elif (a_p - a_m > thr) and (n_p - n_m > thr) and (p_p - p_m > thr):
                #     tracker_type = "center"

                am_delta = a_p - a_m
                noon_delta = n_p - n_m
                pm_delta = p_p - p_m

                tracker_type = None
                if (am_delta > noon_delta) and (pm_delta > noon_delta):
                    tracker_type = "center"
                else:
                    tracker_type = "west" if am_delta > pm_delta else "east"

                if tracker_type is not None:
                    loss_mwh = max(poss_d - meas_d, 0.0)
                    tracker_result[inv][date] = (tracker_type, loss_mwh)

        self._tracker_by_inv_date = tracker_result

    def _build_daily_rollup(self) -> None:
        """Aggregate hourly flags → per-(inv, date) and per-inv totals (MWh)."""
        if self.hourly_flags.empty:
            self.daily_analysis = {}
            self.flags = {}
            return

        long_flags = self.hourly_flags.stack(dropna=True).rename("category").to_frame()
        long_loss = self.hourly_loss_kwh.stack().rename("loss_kwh").to_frame()
        long_df = long_flags.join(long_loss, how="left")
        long_df["loss_kwh"] = long_df["loss_kwh"].fillna(0.0)
        long_df.index.names = ["timestamp", "inverter"]
        long_df = long_df.reset_index()
        long_df["date"] = long_df["timestamp"].dt.date

        grouped = (
            long_df.groupby(["inverter", "date", "category"], sort=False)["loss_kwh"]
            .sum()
            .div(1e3)  # kWh → MWh
            .rename("loss_mwh")
            .reset_index()
        )

        daily_analysis: Dict[str, Dict] = {inv: {} for inv in self.inverter_names}
        for row in grouped.itertuples(index=False):
            daily_analysis[row.inverter].setdefault(row.date, {})[row.category] = row.loss_mwh

        # Apply tracker results without double-counting.
        # The detector reports tracker loss as the *daily shortfall* (poss - meas),
        # which already includes any offline/curtailment hours on that day. To keep
        # `sum(categories)` equal to the true daily shortfall, we:
        #   - drop the day's "underperform" entry (tracker absorbs that residual)
        #   - subtract offline + curtailment from the tracker loss before storing
        # comms (0 MWh) and calibration (0 MWh) don't affect the sum.
        for inv, by_date in getattr(self, "_tracker_by_inv_date", {}).items():
            for date, (t_type, daily_shortfall_mwh) in by_date.items():
                day_flags = daily_analysis[inv].setdefault(date, {})
                already_attributed = day_flags.get("offline", 0.0) + day_flags.get(
                    "curtailment", 0.0
                )
                tracker_mwh = max(daily_shortfall_mwh - already_attributed, 0.0)
                day_flags.pop("underperform", None)
                day_flags[f"tracker_{t_type}"] = tracker_mwh

        self.daily_analysis = daily_analysis

        # totals per inverter
        totals: Dict[str, Dict[str, float]] = {}
        for inv, by_date in daily_analysis.items():
            inv_totals: Dict[str, float] = {}
            for date_flags in by_date.values():
                for category, mwh in date_flags.items():
                    inv_totals[category] = inv_totals.get(category, 0.0) + mwh
            if inv_totals:
                totals[inv] = inv_totals
        self.flags = totals

    # ------------------------------------------------------------------
    # public accessors
    # ------------------------------------------------------------------
    @property
    def flagged_inverters(self) -> List[str]:
        return [inv for inv, flags in self.flags.items() if flags]

    def to_dataframe(self) -> pd.DataFrame:
        """Aggregated flags as DataFrame (inverters * categories, MWh)."""
        if not self.flags:
            return pd.DataFrame()
        df = pd.DataFrame(self.flags).fillna(0).T
        df.index.name = "inverter_id"
        return df

    def to_hourly_dataframe(self, dropna: bool = True) -> pd.DataFrame:
        """Long-format hourly flags: columns [timestamp, inverter, category, loss_kwh]."""
        if self.hourly_flags.empty:
            return pd.DataFrame(columns=["timestamp", "inverter", "category", "loss_kwh"])
        flags = self.hourly_flags.stack(dropna=dropna).rename("category").to_frame()
        loss = self.hourly_loss_kwh.stack().rename("loss_kwh")
        out = flags.join(loss, how="left").reset_index()
        out.columns = ["timestamp", "inverter", "category", "loss_kwh"]
        out["loss_kwh"] = out["loss_kwh"].fillna(0.0)
        return out

    # ------------------------------------------------------------------
    # plotting
    # ------------------------------------------------------------------
    def plot_inverter_generation(
        self,
        inverters: Optional[List[str]] = None,
        show_possible: bool = True,
        height_per_row: int = 80,
        resample: Optional[str] = "15min",
        with_flags: bool = True,
    ):
        """Time-series plot of inverter power with hourly flags shaded.

        Flag categories and per-hour loss are surfaced via the measured trace's
        hover (no on-figure annotations - the figure stays clean and the layout
        cost of hundreds of annotations is avoided).

        Optimized for many inverters: traces and shapes are built in batches and
        attached in single calls ('add_traces' + 'update_layout(shapes=...)')
        rather than per-element 'add_trace'/'add_vrect' loops.

        Parameters
        ----------
        inverters : list[str] | "all" | None
            'None' (default) uses :attr:`flagged_inverters`. '"all"' plots every
            inverter. Or pass a list of inverter names.
        show_possible : bool
            Overlay possible-power line.
        height_per_row : int
        resample : str | None
            Pandas freq for plotting downsample. Default '"15min"'. 'None' = native.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if inverters is None:
            inv_names = list(self.flagged_inverters)
        elif inverters == "all":
            inv_names = list(self.inverter_names)
        else:
            for inv in inverters:
                if inv not in self.inverter_names:
                    raise ValueError(f"Invalid inverter name: {inv}")
            inv_names = list(inverters)

        if not inv_names:
            raise ValueError("No inverters to plot (none flagged - pass inverters='all').")

        df = self.df_inv_pvl
        if resample:
            df = df.resample(resample).mean()
        x_arr = df.index.to_numpy()
        x_to_array = {col: df[col].to_numpy() for col in df.columns}

        # Bucket each plot point to its source hour and date for hover-label lookup.
        hour_buckets = pd.DatetimeIndex(df.index).floor("1H")
        date_buckets = pd.DatetimeIndex(df.index).date
        n_points = len(df.index)

        n_rows = len(inv_names)
        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.3 / max(n_rows, 1),
            subplot_titles=[f"<b>{inv}</b>" for inv in inv_names],
        )

        htemp_base = "<b>%{fullData.name}</b><br>%{y:.1f} kW<br><i>%{x|%Y-%m-%d %H:%M}</i>"
        htemp_with_flag = htemp_base + "%{customdata}<extra></extra>"
        htemp_simple = htemp_base + "<extra></extra>"

        tracker_by_inv = getattr(self, "_tracker_by_inv_date", {})

        # ---- batch traces ----
        traces, rows_list, cols_list = [], [], []
        for i, inv in enumerate(inv_names, start=1):
            poss_col = f"{inv}_possible"
            if show_possible and poss_col in x_to_array:
                traces.append(
                    go.Scattergl(
                        x=x_arr,
                        y=x_to_array[poss_col],
                        name=f"{inv} possible",
                        line=dict(color="#979797", width=1, dash="dot"),
                        hovertemplate=htemp_simple,
                        showlegend=False,
                    )
                )
                rows_list.append(i)
                cols_list.append(1)

            hover_labels = self._build_hover_labels(
                inv, hour_buckets, date_buckets, tracker_by_inv, n_points
            )
            traces.append(
                go.Scattergl(
                    x=x_arr,
                    y=x_to_array[inv],
                    name=inv,
                    line=dict(color="#1929f8", width=1.2),
                    customdata=hover_labels,
                    hovertemplate=htemp_with_flag,
                    showlegend=False,
                )
            )
            rows_list.append(i)
            cols_list.append(1)
        fig.add_traces(traces, rows=rows_list, cols=cols_list)

        # ---- batch shapes (hourly bands + daily tracker bands) ----
        shapes: list[dict] = []
        if with_flags is True:
            hour_step = pd.Timedelta(hours=1)
            day_step = pd.Timedelta(days=1)
            tz = self.tz
            daylight_bounds = self._get_daylight_bounds_by_date()

            for i, inv in enumerate(inv_names, start=1):
                xref = "x" if i == 1 else f"x{i}"
                yref = "y domain" if i == 1 else f"y{i} domain"

                if not self.hourly_flags.empty and inv in self.hourly_flags.columns:
                    inv_hourly = self.hourly_flags[inv].dropna()
                    for x0, x1, category in _contiguous_runs(inv_hourly, hour_step):
                        shapes.append(
                            {
                                "type": "rect",
                                "xref": xref,
                                "yref": yref,
                                "x0": x0,
                                "x1": x1,
                                "y0": 0,
                                "y1": 1,
                                "fillcolor": FLAG_COLORS.get(category, "#cccccc"),
                                "opacity": 0.25,
                                "line": {"width": 0},
                                "layer": "below",
                            }
                        )

                for date, (t_type, _loss_mwh) in tracker_by_inv.get(inv, {}).items():
                    # Clip tracker band to that day's daylight window; fall back to
                    # 00:00–24:00 only if daylight bounds aren't available for the date.
                    if date in daylight_bounds:
                        x0, x1 = daylight_bounds[date]
                    else:
                        x0 = pd.Timestamp(date, tz=tz)
                        x1 = x0 + day_step
                    shapes.append(
                        {
                            "type": "rect",
                            "xref": xref,
                            "yref": yref,
                            "x0": x0,
                            "x1": x1,
                            "y0": 0,
                            "y1": 1,
                            "fillcolor": FLAG_COLORS.get(f"tracker_{t_type}", "#cccccc"),
                            "opacity": 0.18,
                            "line": {"width": 0},
                            "layer": "below",
                        }
                    )

        fig.update_layout(
            height=height_per_row * n_rows + 60,
            margin=dict(t=50, l=60, r=20, b=30),
            title=dict(
                text=f"<b>{self.site.name} - Hourly Inverter Power & Flagged Underperformance</b>",
                x=0.0,
                xanchor="left",
            ),
            plot_bgcolor="#fff",
            paper_bgcolor="#f8f9fa",
            font_size=10,
            showlegend=False,
            shapes=shapes,
        )
        fig.update_xaxes(linecolor="#ced4da", gridcolor="#eee", tickformat="%m/%d", tickfont_size=9)
        fig.update_yaxes(
            linecolor="#ced4da",
            rangemode="tozero",
            tickfont_size=9,
            title=dict(text="kW", standoff=0),
        )
        return fig

    def _build_hover_labels(
        self,
        inv: str,
        hour_buckets: pd.DatetimeIndex,
        date_buckets,
        tracker_by_inv: dict,
        n_points: int,
    ) -> np.ndarray:
        """Per-point hover suffix array for the measured trace.

        Each entry is either '""' (no flag at that hour/day) or one or more
        '"<br>⚠ category (loss)"' lines. Fed into 'customdata' and rendered
        via 'hovertemplate''s '%{customdata}' token.
        """
        labels = np.full(n_points, "", dtype=object)

        if not self.hourly_flags.empty and inv in self.hourly_flags.columns:
            flags_h = self.hourly_flags[inv]
            loss_h = (
                self.hourly_loss_kwh[inv]
                if inv in self.hourly_loss_kwh.columns
                else pd.Series(0.0, index=flags_h.index)
            )
            flags_per_pt = flags_h.reindex(hour_buckets).to_numpy()
            loss_per_pt = loss_h.reindex(hour_buckets).to_numpy()

            flagged_idx = np.where(pd.notna(flags_per_pt))[0]
            for k in flagged_idx:
                cat = flags_per_pt[k]
                loss = loss_per_pt[k]
                if loss is not None and not pd.isna(loss) and loss > 0.05:
                    labels[k] = f"<br>⚠ {cat} ({loss:.1f} kWh)"
                else:
                    labels[k] = f"<br>⚠ {cat}"

        # Tracker is per-day - concatenate to whatever hourly label is already there.
        for date, (t_type, mwh) in tracker_by_inv.get(inv, {}).items():
            suffix = f"<br>⚠ tracker_{t_type} ({mwh:.1f} MWh/day)"
            day_idx = np.where(date_buckets == date)[0]
            for k in day_idx:
                labels[k] = labels[k] + suffix

        return labels


def _contiguous_runs(s: pd.Series, step: pd.Timedelta):
    """Yield (start, end, category) tuples for contiguous same-value runs.

    'end' is one 'step' past the last timestamp in the run, so it can be used
    directly as the upper bound of a plotly vrect.
    """
    if s.empty:
        return
    s = s.dropna()
    if s.empty:
        return
    arr = s.values
    idx = s.index
    run_start_i = 0
    for i in range(1, len(arr)):
        same_cat = arr[i] == arr[run_start_i]
        contig = (idx[i] - idx[i - 1]) == step
        if not (same_cat and contig):
            yield idx[run_start_i], idx[i - 1] + step, arr[run_start_i]
            run_start_i = i
    yield idx[run_start_i], idx[-1] + step, arr[run_start_i]
