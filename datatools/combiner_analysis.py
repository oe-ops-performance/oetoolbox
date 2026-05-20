"""
CombinerAnalyzer - hourly vectorized DC health analysis at the combiner level.

Mirrors 'inverter_analysis_2.InverterAnalyzer' one level deeper:
'site -> inverters' becomes 'inverter -> combiners'. Constructed only when
DC combiner current data is available (optional path).

Usage
-----
With a SolarAnalysisDataset (combiner-level pvlib stored under 'pvlib_cmb'):
>>> from oetoolbox.datatools.combiner_analysis import CombinerAnalyzer
>>> ca = CombinerAnalyzer(data=ds, inv_analyzer=ia)
>>> ca.run_analysis()

With a plain dict (DCHealthChecks-style):
>>> ca = CombinerAnalyzer(
...     site_name="FL1",
...     data={"combiners": df_cmb, "inverters": df_inv, "pvlib": df_pvlib_cmb,
...           "curtailment": df_curt},
... )
>>> ca.run_analysis()

Several setup methods ('_detect_and_normalize_frequencies',
'_generate_analysis_date_attributes', '_get_tracker_hours_for_solar_noon',
'_compute_daylight_hour_mask', '_compute_shoulder_hour_mask') duplicate
logic from InverterAnalyzer. These are flagged TODO-mixin; the cleanest
follow-up is to extract a shared base/mixin so both classes inherit.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .dchealth import CombinerMetadata
from .inverter_analysis import InverterAnalyzer, get_representative_solar_noon, _HourlyAnalyzerBase
from .qc import get_daylight_timestamps
from ..utils.assets import SolarSite
from ..utils.datetime import infer_freq, upsample


@dataclass
class CombinerUnderperformanceConfig:
    """Detection thresholds for the hourly combiner analyzer.

    Current thresholds are fractions of per-combiner peak i_mp (amps from the
    combiner-level pvlib model), resolved to absolute amps via per-combiner
    multiplication. Each fractional default has a paired '*_amps' override;
    when set, the override wins.
    """

    # current thresholds (fractions of per-cmb peak i_mp)
    offline_current_range_frac: float = 0.02
    offline_current_frac: float = 0.05
    comms_current_frac: float = 0.10
    calibration_offset_frac: float = 0.05
    tracker_stow_current_delta_frac: float = 0.10

    # absolute amp overrides
    offline_current_range_amps: Optional[float] = None
    offline_current_threshold_amps: Optional[float] = None
    comms_current_threshold_amps: Optional[float] = None
    calibration_offset_amps: Optional[float] = None
    tracker_stow_current_delta_amps: Optional[float] = None

    # ratio-based (unit-free; same numbers as inverter analyzer)
    underperform_pct_threshold: float = 0.75
    underperform_min_possible_frac: float = 0.05
    underperform_shoulder_hours: int = 0
    underperform_shoulder_pct_threshold: Optional[float] = None

    # curtailment
    curtailment_flag_value: int = 0
    pct_curtailed_threshold: float = 0.625

    # tracker (daily)
    tracker_hours: Tuple[int, int, int] = (9, 12, 15)
    solar_noon_offset_hours: float = 2.0
    tracker_require_clearsky_day: bool = True
    tracker_clearsky_min_frac: float = 0.7
    tracker_max_curtailment_frac: float = 0.1

    # clearsky
    clearsky_ghi_source: str = "dtn"
    clearsky_detect_window: int = 10

    # misc
    calibration_hour: int = 1
    daylight_min_hour_frac: float = 0.5
    daylight_limits: Tuple[str, str] = ("dawn", "dusk")

    # comms vs offline (analog: inv vs sum-of-its-cmbs shortfall)
    comms_share_threshold: float = 0.5
    # min sum-of-combiner shortfall (kW) to compute the ratio; below this the
    # denominator is too noisy and we default to offline.
    comms_min_cmb_shortfall_kw: float = 1.0


CMB_REQUIRED_KEYS = ["combiners", "inverters", "pvlib_cmb"]
CMB_OPTIONAL_KEYS = ["dtn", "met_stations", "curtailment", "site"]
CMB_ALL_KEYS = CMB_REQUIRED_KEYS + CMB_OPTIONAL_KEYS

HOURLY_CATEGORIES = ["curtailment", "offline", "comms", "calibration", "underperform"]
DAILY_TRACKER_CATEGORIES = ["tracker_east", "tracker_west", "tracker_center"]

INV_COL_PATTERN = "OE.ActivePower_"


class CombinerAnalyzer(_HourlyAnalyzerBase):
    """Hourly per-combiner underperformance analyzer.

    Internal frames use a 'MultiIndex[(inverter, combiner)]' for columns, so
    inverter-grouped reductions (comms heuristic, parent-inverter disqualifier)
    are one 'frame.T.groupby(level="inverter")' call away.

    Parameters
    ----------
    site_name : str, optional
        Required when 'data' is a plain dict. Inferred from a SolarAnalysisDataset.
    data : dict | SolarAnalysisDataset
        Must contain non-empty 'combiners', 'inverters', and 'pvlib_cmb' (the
        combiner-level pvlib output with 'i_mp_*' columns). Optional:
        'curtailment' (or 'site'), 'dtn', 'met_stations'.
    config : CombinerUnderperformanceConfig, optional
    inv_analyzer : InverterAnalyzer, optional
        When provided, daylight/shoulder/curtailment masks and the parent-inverter
        flag disqualifier are read from it instead of recomputed. Recommended.
    """

    @property
    def hourly_index(self) -> pd.DatetimeIndex:
        return self.hourly_mean_amps.index

    @property
    def native_index(self) -> pd.DatetimeIndex:
        return self.df_measured_amps.index

    def __init__(
        self,
        site_name: Optional[str] = None,
        data=None,
        config: Optional[CombinerUnderperformanceConfig] = None,
        inv_analyzer: Optional[InverterAnalyzer] = None,
    ):
        from ..utils.solar2 import SolarAnalysisDataset

        if isinstance(data, SolarAnalysisDataset):
            if site_name is None:
                site_name = data.site.name
            data = dict(data.data)

        if data is None:
            data = {}
        if site_name is None:
            raise ValueError("site_name required when data is not a SolarAnalysisDataset.")

        self.site = SolarSite(site_name)
        self.meta = CombinerMetadata(site=site_name)
        self.config = config or CombinerUnderperformanceConfig()
        self._inv_analyzer = inv_analyzer

        self.tz = self.site.timezone
        self.lat, self.lon = self.site.coordinates
        self.inverter_names = self.meta.inverter_names

        self._process_input_data(data)
        self._detect_frequencies()
        self._normalize_frequencies()

        self._generate_analysis_date_attributes()
        self._process_curtailment_data()
        self._get_tracker_hours_for_solar_noon()

        self._build_native_frames()
        self._compute_per_combiner_capacity()
        self.thresholds = self._resolve_thresholds()

        # populated by run_analysis()
        self.hourly_flags: pd.DataFrame = pd.DataFrame()
        self.hourly_loss_kwh: pd.DataFrame = pd.DataFrame()
        self.flags: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.daily_analysis: Dict[Tuple[str, str], Dict] = {}
        self._tracker_by_col_date: Dict[Tuple[str, str], Dict] = {}

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _process_input_data(self, data: dict) -> None:
        valid: Dict[str, pd.DataFrame] = {}
        for key in CMB_ALL_KEYS:
            df = data.get(key)
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                if key in CMB_REQUIRED_KEYS:
                    raise KeyError(f"No data for required group: {key!r}")
                continue
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Provided {key!r} data is not a dataframe.")
            valid[key] = df.copy()

        # rename combiner columns to {cmb_id}_{inv} convention via metadata map
        valid["combiners"] = valid["combiners"].rename(columns=self.meta.rename_cols)
        self.data = valid

    def _process_curtailment_data(self) -> None:
        """Resolve which group + column carries the curtailment flag.

        Prefers a 'site' group (InverterAnalyzer convention; also carries
        setpoint), falling back to a 'curtailment' group
        (DCHealthChecks convention; flag only).
        """
        flag_col = "OE.CurtailmentStatus"
        for group in ("site", "curtailment"):
            df = self.data.get(group)
            if df is not None and flag_col in df.columns:
                self.curt_group = group
                self.curt_flag_col = flag_col
                return
        self.curt_group = None
        self.curt_flag_col = None

    # ------------------------------------------------------------------
    # frame construction
    # ------------------------------------------------------------------
    def _build_native_frames(self) -> None:
        """Build (timestamps x MultiIndex[inv, cmb]) frames at native freq.

        Combiner columns in the input data are renamed to '{cmb_id}_{inv}' via
        'CombinerMetadata.rename_cols'. The combiner-level pvlib model output
        uses 'i_mp_{cmb_id}_{inv}' for current and 'possible_kw_{inv}' for
        per-inverter possible power (apportioned to combiners via dc_weight_adj).
        """
        df_cmb = self.data["combiners"].clip(lower=0)
        df_pvl = self.data["pvlib_cmb"]
        df_inv = self.data["inverters"]

        weights = self.meta.dc_weights.set_index(["inv_id", "cmb_id"])
        meas, poss_a, poss_kw = {}, {}, {}
        skipped: List[Tuple[str, str]] = []

        for (inv, cmb), row in weights.iterrows():
            act_col = f"{cmb}_{inv}"
            imp_col = f"i_mp_{cmb}_{inv}"
            inv_pvl_col = f"possible_kw_{inv}"
            if (
                act_col not in df_cmb.columns
                or imp_col not in df_pvl.columns
                or inv_pvl_col not in df_pvl.columns
            ):
                skipped.append((inv, cmb))
                continue
            meas[(inv, cmb)] = df_cmb[act_col]
            poss_a[(inv, cmb)] = df_pvl[imp_col]
            poss_kw[(inv, cmb)] = df_pvl[inv_pvl_col].mul(row["dc_weight_adj"])

        if not meas:
            raise ValueError("No combiner columns matched between data and metadata.")

        self.df_measured_amps = pd.concat(meas, axis=1)
        self.df_possible_amps = pd.concat(poss_a, axis=1)
        self.df_possible_kw = pd.concat(poss_kw, axis=1)
        for f in (self.df_measured_amps, self.df_possible_amps, self.df_possible_kw):
            f.columns.names = ["inverter", "combiner"]

        self.skipped_combiners = skipped

        # inverter measured kW for the comms heuristic
        dfi = df_inv.filter(like=INV_COL_PATTERN).copy()
        dfi.columns = [c.replace(INV_COL_PATTERN, "") for c in dfi.columns]
        present = [inv for inv in self.inverter_names if inv in dfi.columns]
        if not present:
            raise ValueError("No inverter ActivePower columns found in 'inverters' data.")
        self.df_inv_measured_kw = dfi[present]

    def _compute_per_combiner_capacity(self) -> None:
        """Per-combiner peak amps (Series indexed by [inv, cmb])."""
        self.cmb_peak_amps = self.df_possible_amps.max()

    def _resolve_thresholds(self) -> pd.DataFrame:
        """Per-(inv, cmb) absolute thresholds. Rows = (inv, cmb), cols = thresholds."""
        c = self.config
        peak = self.cmb_peak_amps
        return pd.DataFrame(
            {
                "offline_range": (
                    c.offline_current_range_amps
                    if c.offline_current_range_amps is not None
                    else c.offline_current_range_frac * peak
                ),
                "offline": (
                    c.offline_current_threshold_amps
                    if c.offline_current_threshold_amps is not None
                    else c.offline_current_frac * peak
                ),
                "comms": (
                    c.comms_current_threshold_amps
                    if c.comms_current_threshold_amps is not None
                    else c.comms_current_frac * peak
                ),
                "calibration": (
                    c.calibration_offset_amps
                    if c.calibration_offset_amps is not None
                    else c.calibration_offset_frac * peak
                ),
                "tracker_delta": (
                    c.tracker_stow_current_delta_amps
                    if c.tracker_stow_current_delta_amps is not None
                    else c.tracker_stow_current_delta_frac * peak
                ),
                "under_min_poss": c.underperform_min_possible_frac * peak,
            }
        )

    # ------------------------------------------------------------------
    # hourly aggregates + per-inverter signals
    # ------------------------------------------------------------------
    def _compute_hourly_aggregates(self) -> None:
        rs_m = self.df_measured_amps.resample("1H")
        self.hourly_mean_amps = rs_m.mean()
        self.hourly_range_amps = rs_m.max().sub(rs_m.min())
        self.hourly_possible_amps = self.df_possible_amps.resample("1H").mean()
        self.hourly_possible_kw = self.df_possible_kw.resample("1H").mean()

        # Measured kW = (measured / possible) amps ratio * possible kW. Each row
        # is a 1-h window, so mean(kW) over the row equals kWh for that hour.
        denom = self.hourly_possible_amps.where(self.hourly_possible_amps > 0)
        meas_kw = self.hourly_mean_amps.div(denom).mul(self.hourly_possible_kw).fillna(0.0)
        self.hourly_measured_kw_equiv = meas_kw
        self.hourly_loss_raw_kwh = (self.hourly_possible_kw - meas_kw).clip(lower=0.0)

    def _compute_per_inverter_signals(self) -> None:
        """Comms-share per inverter per hour (analog of InverterAnalyzer's site heuristic).

        If the combiners read low but the parent inverter is still generating,
        the missing combiner current is likely a comms outage. If both inverter
        and combiners are low together, it's a real outage.
        """
        cmb_meas_by_inv = self.hourly_measured_kw_equiv.T.groupby(level="inverter").sum().T
        cmb_poss_by_inv = self.hourly_possible_kw.T.groupby(level="inverter").sum().T

        inv_kw_h = self.df_inv_measured_kw.resample("1H").mean()
        inv_kw_h = inv_kw_h.reindex(columns=cmb_poss_by_inv.columns)

        cmb_short = (cmb_poss_by_inv - cmb_meas_by_inv).clip(lower=0.0)
        inv_short = (cmb_poss_by_inv - inv_kw_h).clip(lower=0.0)

        denom = cmb_short.where(cmb_short >= self.config.comms_min_cmb_shortfall_kw, np.nan)
        share = 1.0 - inv_short.div(denom)
        self.comms_share_h = share.clip(0.0, 1.0).fillna(0.0)  # hours x inv

    def _compute_curtailment_mask(self) -> pd.Series:
        """Hourly bool series: True = hour mostly curtailed.

        Prefers the inverter analyzer's mask when available (already computed
        with consistent thresholds).
        """
        if self._inv_analyzer is not None:
            mask = getattr(self._inv_analyzer, "curt_hour_mask", None)
            if mask is not None:
                return mask.reindex(self.hourly_mean_amps.index, fill_value=False)
        if self.curt_group is None:
            return pd.Series(False, index=self.hourly_mean_amps.index)
        flag_series = (
            self.data[self.curt_group][self.curt_flag_col] > self.config.curtailment_flag_value
        )
        curt_frac_h = flag_series.astype(float).resample("1H").mean()
        return curt_frac_h > self.config.pct_curtailed_threshold

    # ------------------------------------------------------------------
    # detection
    # ------------------------------------------------------------------
    def run_analysis(
        self,
        only_daylight: bool = True,
        detect_tracker: bool = True,
        printouts: bool = False,
    ) -> None:
        qprint = print if printouts else (lambda *_: None)

        qprint("Computing hourly aggregates...")
        self._compute_hourly_aggregates()
        self._compute_per_inverter_signals()

        self.daylight_h = self._compute_daylight_hour_mask()
        self.shoulder_h = self._compute_shoulder_hour_mask()
        self.curt_hour_mask = self._compute_curtailment_mask()

        qprint("Classifying hourly flags...")
        self._classify_hourly(only_daylight=only_daylight)

        if detect_tracker:
            qprint("Detecting daily tracker patterns...")
            self._detect_tracker_daily()
        else:
            self._tracker_by_col_date = {}

        qprint("Building daily rollup...")
        self._build_daily_rollup()
        qprint("Done.")

    def _classify_hourly(self, only_daylight: bool) -> None:
        """Vectorized hourly classification at (hours x MultiIndex[inv, cmb]).

        Priority (highest first):
            1. curtailment   - site curtailment flag majority of hour
            2. offline/comms - flatlined + low; comms iff parent inverter
                               still generating (high comms_share)
            3. calibration   - non-zero current at calibration_hour
            4. underperform  - measured/possible ratio < threshold, sun-gated,
                               with shoulder-hour leniency

        Combiners whose parent inverter is itself flagged offline or comms
        for that hour are reset to NaN (the reading is uninformative).
        """
        hours = self.hourly_mean_amps.index
        cols = self.hourly_mean_amps.columns

        def col_thr(name: str) -> pd.DataFrame:
            arr = np.broadcast_to(
                self.thresholds[name].to_numpy()[None, :], (len(hours), len(cols))
            )
            return pd.DataFrame(arr, index=hours, columns=cols)

        def row_mask(s: pd.Series) -> pd.DataFrame:
            arr = np.broadcast_to(s.to_numpy()[:, None], (len(s), len(cols))).copy()
            return pd.DataFrame(arr, index=hours, columns=cols)

        analysable = self.daylight_h if only_daylight else pd.Series(True, index=hours)
        analysable_2d = row_mask(analysable)
        curt_2d = row_mask(self.curt_hour_mask)

        # broadcast per-inverter comms_share across (inv, cmb) cols
        inv_level = cols.get_level_values("inverter")
        share_per_col = self.comms_share_h.reindex(columns=inv_level.unique())[inv_level]
        share_per_col.columns = cols
        comms_high_2d = share_per_col.gt(self.config.comms_share_threshold)

        flags = pd.DataFrame(index=hours, columns=cols, dtype=object)
        loss_kwh = pd.DataFrame(0.0, index=hours, columns=cols)

        # 1. curtailment - all (inv, cmb) flagged on curtailed hours
        # Without per-combiner setpoint info, loss is reported as full possible.
        # TODO: when 'site' group has OE.SiteSetpoint, apportion via dc_weight_adj.
        curt_rows = self.curt_hour_mask & analysable
        if curt_rows.any():
            flags.loc[curt_rows, :] = "curtailment"
            loss_kwh.loc[curt_rows, :] = self.hourly_possible_kw.loc[curt_rows, :].to_numpy()

        # 2. offline / comms
        stuck = self.hourly_range_amps.lt(col_thr("offline_range"))
        low_or_dim = self.hourly_mean_amps.lt(col_thr("comms"))
        outage = stuck & low_or_dim & analysable_2d & ~curt_2d
        offline_mask = outage & ~comms_high_2d
        comms_mask = outage & comms_high_2d
        if offline_mask.any().any():
            flags = flags.mask(offline_mask, "offline")
            loss_kwh = loss_kwh.mask(offline_mask, self.hourly_loss_raw_kwh)
        if comms_mask.any().any():
            flags = flags.mask(comms_mask, "comms")
            loss_kwh = loss_kwh.mask(comms_mask, 0.0)

        # 3. calibration (only at calibration_hour, regardless of daylight)
        calib_hr_1d = pd.Series(hours.hour == self.config.calibration_hour, index=hours)
        calib_hr_2d = row_mask(calib_hr_1d)
        calib_high = self.hourly_mean_amps.gt(col_thr("calibration"))
        calib_assign = calib_high & calib_hr_2d & flags.isna()
        if calib_assign.any().any():
            flags = flags.mask(calib_assign, "calibration")

        # 4. underperform (with shoulder leniency)
        sun_mask = self.hourly_possible_amps.gt(col_thr("under_min_poss"))
        ratio = self.hourly_mean_amps.div(
            self.hourly_possible_amps.where(self.hourly_possible_amps > 0)
        )
        shoulder_thr = (
            self.config.underperform_shoulder_pct_threshold
            if self.config.underperform_shoulder_pct_threshold is not None
            else self.config.underperform_pct_threshold
        )
        shoulder_2d = row_mask(self.shoulder_h)
        under_main = (
            ratio.lt(self.config.underperform_pct_threshold)
            & ~shoulder_2d
            & sun_mask
            & analysable_2d
            & flags.isna()
        )
        under_shld = ratio.lt(shoulder_thr) & shoulder_2d & sun_mask & analysable_2d & flags.isna()
        under_mask = under_main | under_shld
        if under_mask.any().any():
            flags = flags.mask(under_mask, "underperform")
            loss_kwh = loss_kwh.mask(under_mask, self.hourly_loss_raw_kwh)

        # 5. disqualify combiners under an offline/comms parent inverter this hour
        if self._inv_analyzer is not None and not self._inv_analyzer.hourly_flags.empty:
            inv_flags = self._inv_analyzer.hourly_flags.reindex(index=hours)
            inv_disq = inv_flags.isin({"offline", "comms"})
            inv_disq = inv_disq.reindex(columns=inv_level.unique())
            disq_per_col = inv_disq[inv_level]
            disq_per_col.columns = cols
            disq_per_col = disq_per_col.fillna(False).astype(bool)
            flags = flags.mask(disq_per_col, np.nan)
            loss_kwh = loss_kwh.mask(disq_per_col, 0.0)

        self.hourly_flags = flags
        self.hourly_loss_kwh = loss_kwh

    def _detect_tracker_daily(self) -> None:
        """Per-(inv, cmb, date) tracker pattern check on hourly amps.

        Compares AM/noon/PM expected-minus-measured deltas to flag stuck-east /
        stuck-west / center-stow. Result stored in
        'self._tracker_by_col_date[(inv, cmb)][date] = (kind, loss_mwh)'.
        Skips days that fail the curtailment or clearsky gates (when applicable
        and available from the InverterAnalyzer).
        """
        am_hr, noon_hr, pm_hr = self.tracker_hours
        m_h = self.hourly_mean_amps
        p_h = self.hourly_possible_amps
        kw_h = self.hourly_possible_kw
        meas_kw_h = self.hourly_measured_kw_equiv
        idx = m_h.index

        def by_hour(df: pd.DataFrame, hr: int) -> pd.DataFrame:
            sub = df[idx.hour == hr].copy()
            sub.index = sub.index.date
            return sub

        am_m, nn_m, pm_m = by_hour(m_h, am_hr), by_hour(m_h, noon_hr), by_hour(m_h, pm_hr)
        am_p, nn_p, pm_p = by_hour(p_h, am_hr), by_hour(p_h, noon_hr), by_hour(p_h, pm_hr)

        # daily MWh totals per (inv, cmb) for loss reporting
        daily_meas_mwh = meas_kw_h.resample("D").mean().mul(24).div(1e3)
        daily_poss_mwh = kw_h.resample("D").mean().mul(24).div(1e3)
        daily_meas_mwh.index = daily_meas_mwh.index.date
        daily_poss_mwh.index = daily_poss_mwh.index.date

        # day-level gates from the inverter analyzer (when present)
        clearsky_by_date: Dict = {}
        curt_frac_by_date: Dict = {}
        if self._inv_analyzer is not None:
            try:
                clearsky_by_date = self._inv_analyzer._compute_clearsky_day_mask()
            except Exception:
                clearsky_by_date = {}
            try:
                curt_frac_by_date = self._inv_analyzer._compute_curtailment_frac_by_date()
            except Exception:
                curt_frac_by_date = {}

        require_cs = self.config.tracker_require_clearsky_day
        max_curt = self.config.tracker_max_curtailment_frac
        cs_info_available = bool(clearsky_by_date)

        # per-column disqualifier: skip days where this combiner was already
        # flagged offline/comms/curtailment
        disqualifying = {"offline", "comms", "curtailment"}
        col_skip_dates: Dict[Tuple[str, str], set] = {col: set() for col in m_h.columns}
        if not self.hourly_flags.empty:
            for col in m_h.columns:
                if col not in self.hourly_flags.columns:
                    continue
                s = self.hourly_flags[col].dropna()
                if s.empty:
                    continue
                col_skip_dates[col] = set(s.index[s.isin(disqualifying)].date)

        result: Dict[Tuple[str, str], Dict] = {col: {} for col in m_h.columns}
        for date in self.analysis_dates:
            if curt_frac_by_date.get(date, 0.0) > max_curt:
                continue
            if require_cs:
                if not cs_info_available or not clearsky_by_date.get(date, False):
                    continue
            for col in m_h.columns:
                if date in col_skip_dates[col]:
                    continue
                poss_d = daily_poss_mwh.at[date, col] if date in daily_poss_mwh.index else 0.0
                meas_d = daily_meas_mwh.at[date, col] if date in daily_meas_mwh.index else 0.0
                if poss_d < 1e-3:
                    continue

                try:
                    a_m, n_m, p_m = am_m.at[date, col], nn_m.at[date, col], pm_m.at[date, col]
                    a_p, n_p, p_p = am_p.at[date, col], nn_p.at[date, col], pm_p.at[date, col]
                except KeyError:
                    continue
                if any(pd.isna(x) for x in (a_m, n_m, p_m, a_p, n_p, p_p)):
                    continue

                am_delta = a_p - a_m
                noon_delta = n_p - n_m
                pm_delta = p_p - p_m

                if (am_delta > noon_delta) and (pm_delta > noon_delta):
                    kind = "center"
                else:
                    kind = "west" if am_delta > pm_delta else "east"

                loss_mwh = max(poss_d - meas_d, 0.0)
                result[col][date] = (kind, loss_mwh)

        self._tracker_by_col_date = result

    def _build_daily_rollup(self) -> None:
        """Aggregate hourly flags into per-(inv, cmb, date) and per-(inv, cmb) totals (MWh)."""
        if self.hourly_flags.empty:
            self.daily_analysis = {}
            self.flags = {}
            return

        long_flags = self.hourly_flags.stack(level=["inverter", "combiner"], dropna=True)
        long_flags = long_flags.rename("category").to_frame()
        long_loss = self.hourly_loss_kwh.stack(level=["inverter", "combiner"]).rename("loss_kwh")
        long_df = long_flags.join(long_loss, how="left")
        long_df["loss_kwh"] = long_df["loss_kwh"].fillna(0.0)
        long_df.index.names = ["timestamp", "inverter", "combiner"]
        long_df = long_df.reset_index()
        long_df["date"] = long_df["timestamp"].dt.date

        grouped = (
            long_df.groupby(["inverter", "combiner", "date", "category"], sort=False)["loss_kwh"]
            .sum()
            .div(1e3)
            .rename("loss_mwh")
            .reset_index()
        )

        daily: Dict[Tuple[str, str], Dict] = {col: {} for col in self.hourly_flags.columns}
        for row in grouped.itertuples(index=False):
            key = (row.inverter, row.combiner)
            daily.setdefault(key, {}).setdefault(row.date, {})[row.category] = row.loss_mwh

        # apply tracker results: tracker loss = daily shortfall; subtract any
        # offline+curtailment already attributed so sums stay consistent
        for col, by_date in self._tracker_by_col_date.items():
            for date, (kind, daily_shortfall_mwh) in by_date.items():
                day_flags = daily.setdefault(col, {}).setdefault(date, {})
                already = day_flags.get("offline", 0.0) + day_flags.get("curtailment", 0.0)
                tracker_mwh = max(daily_shortfall_mwh - already, 0.0)
                day_flags.pop("underperform", None)
                day_flags[f"tracker_{kind}"] = tracker_mwh

        self.daily_analysis = daily

        totals: Dict[Tuple[str, str], Dict[str, float]] = {}
        for col, by_date in daily.items():
            t: Dict[str, float] = {}
            for date_flags in by_date.values():
                for cat, mwh in date_flags.items():
                    t[cat] = t.get(cat, 0.0) + mwh
            if t:
                totals[col] = t
        self.flags = totals

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------
    @property
    def flagged_combiners(self) -> List[Tuple[str, str]]:
        return [col for col, cats in self.flags.items() if cats]

    def to_dataframe(self) -> pd.DataFrame:
        """Aggregated flags as DataFrame ((inv, cmb) x categories, MWh)."""
        if not self.flags:
            return pd.DataFrame()
        df = pd.DataFrame(self.flags).fillna(0).T
        df.index.names = ["inverter", "combiner"]
        return df

    def to_hourly_dataframe(self, dropna: bool = True) -> pd.DataFrame:
        """Long-format hourly flags: [timestamp, inverter, combiner, category, loss_kwh]."""
        if self.hourly_flags.empty:
            return pd.DataFrame(
                columns=["timestamp", "inverter", "combiner", "category", "loss_kwh"]
            )
        flags = self.hourly_flags.stack(level=["inverter", "combiner"], dropna=dropna)
        flags = flags.rename("category").to_frame()
        loss = self.hourly_loss_kwh.stack(level=["inverter", "combiner"]).rename("loss_kwh")
        out = flags.join(loss, how="left").reset_index()
        out.columns = ["timestamp", "inverter", "combiner", "category", "loss_kwh"]
        out["loss_kwh"] = out["loss_kwh"].fillna(0.0)
        return out
