import numpy as np
import pandas as pd
from pvlib.clearsky import detect_clearsky

from .assets import SolarSite
from .config import CACHE_DIR
from .datetime import infer_freq, upsample
from .helpers import quiet_print_function
from .pi import PIDataset
from .solar import SolarDataset
from ..datatools.qc import get_sun


def _range_id(start_date, end_date):
    start, end = map(pd.Timestamp, [start_date, end_date])
    return f"{start:%Y%m%d}_{end:%Y%m%d}"


def _cache_fpath(site: str, start_date: str, end_date: str, file_type: str, ext: str = "parquet"):
    folder = CACHE_DIR / "analysis_data"
    if not folder.exists():
        folder.mkdir()
    site_str = site.lower().replace(" ", "").replace("-", "")
    ftype_str = file_type.lower()
    rng_id = get_range_id(start_date, end_date)
    return folder / f"{site_str}_{ftype_str}_{rng_id}.{ext}"


# for use with PI queries
ANALYSIS_QUERY_GROUPS = ["combiners", "curtailment", "inverters", "met_stations", "meter"]

ANALYSIS_QUERY_FREQUENCIES = {
    "combiners": "1h",
    "curtailment": "1m",
    "inverters": "1m",
    "met_stations": "1m",
    "meter": "1m",
}


class SolarAnalysisDataset(SolarDataset):
    """Dataset for solar analysis spanning multiple data sources and use cases.

    Overrides the `data` attribute from SolarDataset to be a dict of DataFrames
    keyed by group name, rather than a single DataFrame. Supports two analysis paths:

    1. Before/after comparison analysis - format_combined / split_for_comparison feed
       into the filtering and scatter analysis in scratch.comparison_analysis.
    2. DC health analysis - can be passed directly to DCHealthChecks as a data source
       in place of a raw dict.

    For pvlib, two separate runs are stored because each analysis needs a different format:
        data["pvlib"]     -- inverter-level (columns: *_Possible_Power), for comparison analysis
        data["pvlib_cmb"] -- combiner-level (columns: possible_kw_*, i_mp_*), for DCHealthChecks

    Typical comparison analysis workflow
    -------------------------------------
    >>> ds = SolarAnalysisDataset.from_data_dict("FL1", data, "2026-03-05", "2026-04-27")
    >>> ds.run_pvlib()
    >>> ds.get_daylight_timestamps()
    >>> df = ds.format_combined()
    >>> before, after = ds.split_for_comparison("2026-04-06", "2026-04-08").values()

    Typical DC health workflow
    --------------------------
    >>> ds = SolarAnalysisDataset.from_data_dict("FL1", data, "2026-03-05", "2026-04-27")
    >>> ds.run_pvlib_cmb()
    >>> dch = DCHealthChecks(data=ds)   # site name inferred from ds.site
    >>> dch.run_combiner_analysis()

    Attributes
    ----------
    data : dict[str, pd.DataFrame]
        DataFrames keyed by group. Supported keys: inverters, met_stations, meter,
        combiners, dtn, ghi, pvlib, pvlib_cmb, curtailment.
    combined_df : pd.DataFrame or None
        Formatted output of format_combined(); populated on first call.
    """

    DATA_KEYS = [
        "inverters",
        "met_stations",
        "meter",
        "combiners",
        "dtn",
        "ghi",
        "pvlib",
        "pvlib_cmb",
        "curtailment",
    ]

    # hourly → 1-min upsample constants (used in format_combined)
    _ANALYSIS_FREQ = pd.Timedelta("1min")
    _UPSAMPLE_SHIFT = pd.Timedelta(hours=1)

    def __init__(self, site: SolarSite, start_date: str = None, end_date: str = None):
        super().__init__(site, start_date, end_date)
        self.data = {}
        self.combined_df = None
        self._daylight_timestamps = None
        self._solar_noon = None
        self.datetime_index = None

    @property
    def columns(self):
        return []

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    @classmethod
    def from_data_dict(
        cls,
        site_name: str,
        data: dict,
        start_date: str = None,
        end_date: str = None,
    ):
        """Initialize from an existing dict of DataFrames.

        Parameters
        ----------
        site_name : str
        data : dict[str, pd.DataFrame]
            Keys should be group names such as 'inverters', 'dtn', 'pvlib', 'ghi', etc.
        start_date : str, optional
            If omitted, inferred from the first non-empty DataFrame in data.
        end_date : str, optional
        """
        site = SolarSite(site_name)
        dataset = cls(site, start_date, end_date)
        dataset.data = {k: v.copy() for k, v in data.items() if isinstance(v, pd.DataFrame)}
        if start_date is None or end_date is None:
            dataset._infer_date_range()
            dataset._format_data_index_for_merge()
        return dataset

    def _infer_date_range(self):
        """Infer start/end dates from first non-empty DataFrame in self.data."""
        for grp, df in self.data.items():
            if grp == "dtn":
                continue
            if not df.empty:
                self.start_date = df.index.min().floor("D").strftime("%Y-%m-%d")
                self.end_date = df.index.max().ceil("D").strftime("%Y-%m-%d")
                self.datetime_index = pd.date_range(
                    self.start_date,
                    self.end_date,
                    freq=self._ANALYSIS_FREQ,
                    tz=df.index.tzinfo,
                    inclusive="left",
                )
                return

    def _format_data_index_for_merge(self):
        # updates timezone -- query files have UTC-offset, and DTN file has tz name
        if self.datetime_index is None:
            return
        if self.datetime_index[0].utcoffset() is None:
            return
        offset_ = self.datetime_index[0].utcoffset()
        for grp in self.data.keys():
            if offset_ == self.data[grp].index[0].utcoffset():
                self.data[grp] = self.data[grp].tz_convert(self.datetime_index.tzinfo)

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    def query_data(self, groups=None, freq=None, overwrite=False, q=True):
        """Query data from PI/DTN for the specified groups and store in self.data.

        Parameters
        ----------
        groups : list[str], optional
            Group names to query. Defaults to ['inverters', 'dtn'].
            Supported: 'inverters', 'met_stations', 'meter', 'combiners',
            'dtn', 'curtailment'.
        freq : str, optional
            Data interval for PI queries. Defaults to '1h' for 'combiners', otherwise '1m'.
        overwrite : bool, optional
            Re-query even if data is already loaded. Default False.
        q : bool, optional
        """
        qprint = quiet_print_function(q=q)
        if groups is None:
            groups = ["inverters", "dtn"]
        if freq is not None:
            freq_list = [freq] * len(groups)
        else:
            freq_list = ["1h" if g in ("combiners", "dtn") else "1m" for g in groups]
        for grp, frq in zip(groups, freq_list):
            if grp in self.data and not self.data[grp].empty and not overwrite:
                qprint(f"Found existing data for {grp}; skipping.")
                continue
            qprint(f"Querying data for {grp}...")
            df = self._get_group_data(grp, freq=frq, q=q)
            if df is not None:
                self.data[grp] = df
        self._format_data_index_for_merge()
        qprint("Done.")

    def _get_group_data(self, grp, freq="1h", q=True):
        try:
            if grp == "dtn":
                return SolarDataset.from_dtn(self.site.name, self.start_date, self.end_date, q=q)
            if grp == "curtailment":
                curtailment_path = f"{self.site.af_path}|OE.CurtailmentStatus"
                dataset = PIDataset.from_attribute_paths(
                    attribute_paths=[curtailment_path],
                    site_name=self.site.name,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    freq=freq,
                    keep_tzinfo=True,
                    q=q,
                )
                return dataset.data
            asset_group = grp.replace("_", " ").title()
            kwargs = dict(
                site_name=self.site.name,
                asset_group=asset_group,
                start_date=self.start_date,
                end_date=self.end_date,
                freq=freq,
                keep_tzinfo=True,
                q=q,
                raise_error=False,
            )
            if grp == "combiners":
                kwargs["n_segment"] = 3
            dataset = SolarDataset.from_defined_query_attributes(**kwargs)
            return dataset.data
        except Exception as e:
            print(f"Error querying data for {grp}: {e}")
            return None

    def run_pvlib(self, poa_source="dtn", overwrite=False, q=True):
        """Run the inverter-level pvlib model; stores result in self.data['pvlib'].

        Output columns follow the *_Possible_Power format used by the comparison
        analysis pipeline (format_combined).

        Parameters
        ----------
        poa_source : str
            POA input. Either 'dtn' (poa_global column) or 'met_stations'
            (mean of OE.POA* columns). Default 'dtn'.
        overwrite : bool
        q : bool
        """
        from ..datatools.pvlib import run_pvlib_model

        qprint = quiet_print_function(q=q)
        if "pvlib" in self.data and not self.data["pvlib"].empty and not overwrite:
            qprint("Found existing pvlib data; skipping.")
            return
        poa_data = self._get_poa_data(poa_source)
        qprint("Running inverter-level pvlib model...")
        self.data["pvlib"] = run_pvlib_model(self.site.name, poa_data=poa_data, q=q)

    def run_pvlib_cmb(self, poa_source="dtn", overwrite=False, q=True):
        """Run the combiner-level pvlib model; stores result in self.data['pvlib_cmb'].

        Output columns follow the possible_kw_* / i_mp_* format required by
        DCHealthChecks.run_combiner_analysis.

        Parameters
        ----------
        poa_source : str
            POA input. Either 'dtn' (poa_global column) or 'met_stations'
            (mean of OE.POA* columns). Default 'dtn'.
            NOTE: this should be hourly data
        overwrite : bool
        q : bool
        """
        from ..datatools.pvlib import run_pvlib_combiner_level

        qprint = quiet_print_function(q=q)
        if "pvlib_cmb" in self.data and not self.data["pvlib_cmb"].empty and not overwrite:
            qprint("Found existing combiner-level pvlib data; skipping.")
            return
        poa_data = self._get_poa_data(poa_source)
        if infer_freq(poa_data.index) < pd.Timedelta(hours=1):
            poa_data = poa_data.resample("1h").mean()

        qprint("Running combiner-level pvlib model...")
        self.data["pvlib_cmb"] = run_pvlib_combiner_level(
            site=self.site.name, poa_data=poa_data, return_dict=False, q=q
        )

    def _get_poa_data(self, source="dtn") -> pd.Series:
        if source == "dtn":
            df_dtn = self.data.get("dtn", pd.DataFrame())
            if df_dtn.empty or "poa_global" not in df_dtn.columns:
                raise ValueError("DTN data with 'poa_global' column is required.")
            return pd.Series(df_dtn["poa_global"], name="POA_DTN")
        if source == "met_stations":
            df_met = self.data.get("met_stations", pd.DataFrame())
            if df_met.empty:
                raise ValueError("Met station data is required when poa_source='met_stations'.")
            if "Processed_POA" in df_met.columns:
                return pd.Series(df_met["Processed_POA"], name="POA")
            poa_cols = [c for c in df_met.columns if "OE.POA" in c]
            if not poa_cols:
                raise ValueError("No OE.POA* columns found in met station data.")
            return pd.Series(df_met[poa_cols].mean(axis=1), name="POA")
        raise ValueError(f"Invalid poa_source {source!r}; must be 'dtn' or 'met_stations'.")

    # -----------------------------------------------------------------------
    # Solar position helpers
    # -----------------------------------------------------------------------

    def get_daylight_timestamps(
        self,
        freq="1min",
        limits=("dawn", "dusk"),
        recompute=False,
    ) -> list:
        """Generate (and cache) daylight timestamps for the site and date range.

        Parameters
        ----------
        freq : str
            Timestamp frequency. Default '1min'.
        limits : tuple[str, str]
            Sun event bounds. Either ('dawn', 'dusk') or ('sunrise', 'sunset').
        recompute : bool
            Force recomputation even if cached. Default False.

        Returns
        -------
        list[pd.Timestamp]
        """
        if self._daylight_timestamps is not None and not recompute:
            return self._daylight_timestamps

        if limits not in [("dawn", "dusk"), ("sunrise", "sunset")]:
            raise ValueError("limits must be ('dawn', 'dusk') or ('sunrise', 'sunset')")

        tz = self.site.timezone
        lat, lon = self.site.coordinates
        start = pd.Timestamp(self.start_date, tz=tz)
        end = pd.Timestamp(self.end_date, tz=tz)
        freq_td = pd.Timedelta(freq)

        tstamp_list = []
        for date in pd.date_range(start=start, end=end, freq="D"):
            sun_times = get_sun(date, tz=tz, lat=lat, lon=lon)
            day_start = sun_times[limits[0]]
            day_end = sun_times[limits[1]]
            if freq_td > pd.Timedelta(minutes=1):
                day_start = day_start.floor(freq)
                day_end = day_end.ceil(freq)
            daylight_range = pd.date_range(
                start=day_start, end=day_end, freq=freq, inclusive="left"
            )
            tstamp_list.extend(list(daylight_range))

        self._daylight_timestamps = list(filter(lambda t: t < end, tstamp_list))
        return self._daylight_timestamps

    def get_solar_noon(self, recompute=False) -> pd.Timestamp:
        """Return a representative solar noon for the mid-point of the date range.

        Parameters
        ----------
        recompute : bool
            Force recomputation even if cached. Default False.

        Returns
        -------
        pd.Timestamp
        """
        if self._solar_noon is not None and not recompute:
            return self._solar_noon

        tz = self.site.timezone
        lat, lon = self.site.coordinates
        total_days = (pd.Timestamp(self.end_date) - pd.Timestamp(self.start_date)).days
        mid = pd.Timestamp(self.start_date, tz=tz) + pd.Timedelta(
            days=int(np.floor(total_days / 2))
        )
        ref_sun = get_sun(date=mid, tz=tz, lat=lat, lon=lon)
        self._solar_noon = ref_sun.get("noon")
        return self._solar_noon

    # -----------------------------------------------------------------------
    # Data formatting and splitting
    # -----------------------------------------------------------------------

    def format_combined(self, ghi_clearsky_source="dtn", q=True) -> pd.DataFrame:
        """Format and combine all loaded data into a single analysis DataFrame.

        Aggregates inverter/pvlib generation to site level, extracts DTN/GHI
        columns, and adds a clearsky flag. Result is stored in self.combined_df.

        Hourly data is upsampled to 1-minute resolution to align with inverter data.

        Parameters
        ----------
        ghi_clearsky_source : str
            GHI source for the clearsky flag. Either 'dtn' or 'sensors'. Default 'dtn'.
        q : bool

        Returns
        -------
        pd.DataFrame
            Columns: site_pvlib, site_inverters, poa, dtn_ghi, ghi_clearsky,
            cloud_cover, clearsky_flag, and optionally site_ghi / site_meter.
        """
        df_list = []
        for grp in ("pvlib", "inverters", "meter", "ghi", "dtn"):
            df_ = self.data.get(grp)
            if df_ is None or df_.empty:
                continue
            df_ = self._upsample_if_needed(df_)
            if grp in ("pvlib", "inverters"):
                pattern = "Possible_Power" if grp == "pvlib" else "ActivePower"
                df_list.append(
                    df_.filter(like=pattern).sum(axis=1).div(1e3).to_frame(name=f"site_{grp}")
                )
                if grp == "pvlib":
                    df_list.append(df_.iloc[:, 0].to_frame(name="poa"))
            elif grp in ("meter", "ghi"):
                df_list.append(df_.iloc[:, 0].to_frame(name=f"site_{grp}"))
            elif grp == "dtn":
                dtn_cols = ["dtn_ghi", "ghi_clearsky", "effectiveCloudCover"]
                present = [c for c in dtn_cols if c in df_.columns]
                df_list.append(df_[present].rename(columns={"effectiveCloudCover": "cloud_cover"}))

        df = pd.concat(df_list, axis=1)
        df = df[list(reversed(sorted(df.columns)))]
        self.combined_df = self._add_clearsky_flag(df, ghi_source=ghi_clearsky_source, q=q)
        return self.combined_df

    def _upsample_if_needed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Upsample hourly data to 1-min resolution via interpolation + 1-h shift."""
        if infer_freq(df.index) > pd.Timedelta(self._ANALYSIS_FREQ):
            df = upsample(df, new_freq=pd.Timedelta(self._ANALYSIS_FREQ), agg_type="interpolate")
        return df

    def _add_clearsky_flag(
        self, df: pd.DataFrame, ghi_source: str = "dtn", window: int = 10, q: bool = True
    ) -> pd.DataFrame:
        """Add clearsky_flag column to df using detect_clearsky from pvlib."""
        source_col_map = {"dtn": "dtn_ghi", "sensors": "site_ghi"}
        backup_map = {"dtn": ("sensors", "site_ghi"), "sensors": ("dtn", "dtn_ghi")}

        ghi_col = source_col_map.get(ghi_source)
        if ghi_col not in df.columns:
            backup_source, ghi_col = backup_map[ghi_source]
            if ghi_col not in df.columns:
                raise ValueError(
                    f"Missing GHI column for clearsky flag (tried {ghi_source!r} and backup)."
                )
            if q is False:
                print(f"Note: missing {ghi_source!r} GHI; using {backup_source!r}.")

        if "ghi_clearsky" not in df.columns:
            raise ValueError("Missing 'ghi_clearsky' column required for clearsky flag.")

        df = df.copy()
        df["clearsky_flag"] = detect_clearsky(
            measured=df[ghi_col].fillna(0),
            clearsky=df["ghi_clearsky"].fillna(0),
            window_length=window,
        ).astype(int)
        return df

    def split_for_comparison(self, event_start: str, event_end: str, max_days: int = 30) -> dict:
        """Split combined_df into before/after periods around an event.

        Parameters
        ----------
        event_start : str
            First date of the event (inclusive), format "%Y-%m-%d".
        event_end : str
            Last date of the event (inclusive), format "%Y-%m-%d".
        max_days : int
            Maximum number of days to include before and after the event. Default 30.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys 'before' and 'after'.
        """
        if self.combined_df is None:
            raise ValueError("Call format_combined() before split_for_comparison().")

        df = self.combined_df
        fmt = lambda ts: ts.strftime("%Y-%m-%d")
        offset = pd.Timedelta(days=max_days)

        before_dates = sorted(
            set(
                df[
                    (df.index >= fmt(pd.Timestamp(event_start) - offset)) & (df.index < event_start)
                ].index.date
            )
        )
        after_dates = sorted(
            set(
                df[
                    (df.index >= fmt(pd.Timestamp(event_end) + pd.Timedelta(days=1)))
                    & (df.index < fmt(pd.Timestamp(event_end) + offset))
                ].index.date
            )
        )

        result = {}
        for key, dates in (("before", before_dates), ("after", after_dates)):
            if not dates:
                result[key] = pd.DataFrame()
                continue
            min_date = fmt(dates[0])
            max_date = fmt(pd.Timestamp(dates[-1]) + pd.Timedelta(days=1))
            idx = df[(df.index > min_date) & (df.index < max_date)].index
            result[key] = df[df.index.isin(idx)].copy()

        return result
