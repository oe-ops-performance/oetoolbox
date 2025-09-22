import itertools
import pandas as pd
from pathlib import Path

from .assets import SolarSite
from .dataset import Dataset
from .datetime import remove_tzinfo_and_standardize_index
from .helpers import quiet_print_function
from .oepaths import frpath, latest_file
from .pi import PIDataset


class SolarDataset(Dataset):
    """A dataset loaded from PI containing time series data for a particular solar site
    as well as some other metadata. Inherits from Dataset.

    Additional Parameters
    ---------------------
    site : oetoolbox.utils.assets.SolarSite
        An instance of a solar site, providing relevant attributes/methods for obtaining data

    Additional Attributes
    ---------------------
    site : oetoolbox.utils.assets.SolarSite
        An instance of a solar site, providing relevant attributes/methods for obtaining data
    """

    def __init__(self, site: SolarSite, start_date: str = None, end_date: str = None):
        super().__init__()
        self.site = site
        self.start_date = start_date
        self.end_date = end_date
        self.tz = site.timezone
        self.source_files = []  # init
        self.invalid_items = []  # from pi query

    @property
    def columns(self):
        return []  # not sure if needed here.. but can use for PVLibDataset (i.e. expected columns)

    @classmethod
    def from_existing_data_files(
        cls, site_name: str, asset_group: str, start_date: str = None, end_date: str = None
    ):
        """Loads data from existing query files in flashreport folders

        Parameters
        ----------
        site_name : str
            Name of site (used to initialize SolarSite instance)
        asset_group : str
            The particular asset group to load data for. Includes PVLib.
        start_date : str, optional
            The target start date for the dataset, default = None
        end_date : str, optional
            The target end date for the dataset, default = None
        """
        site = SolarSite(site_name)
        dataset = cls(site, start_date, end_date)
        dataset._load_data_from_files(asset_group)
        return dataset

    def _load_data_from_files(self, asset_group: str):
        # note: if start or end date is None, loads all available data
        valid_groups = self.site.asset_groups + ["pvlib"]
        if asset_group not in valid_groups:
            raise ValueError("Invalid asset group specified")
        group = asset_group.replace(" ", "").lower()
        fpaths_by_period = {
            key_: [fp for fp in fpath_list if group in fp.name.lower()]
            for key_, fpath_list in self.site.data_files_by_period.items()
            if any(group in fp.name.lower() for fp in fpath_list)
        }
        get_year_month = lambda yyyymm: (int(yyyymm[:4]), int(yyyymm[-2:]))
        valid_year_month_list = list(map(get_year_month, fpaths_by_period.keys()))
        if not valid_year_month_list:
            raise ValueError("No files exist for specified asset_group")

        all_dates_with_data = []
        for year, month in valid_year_month_list:
            start_ = pd.Timestamp(year, month, 1)
            end_ = start_ + pd.DateOffset(months=1, days=-1)
            all_dates_with_data.extend(list(pd.date_range(start_, end_)))

        if any(date is None for date in [self.start_date, self.end_date]):
            target_start = pd.Timestamp(*valid_year_month_list[0], 1)
            target_end = pd.Timestamp(*valid_year_month_list[-1], 1) + pd.DateOffset(months=1)
        else:
            target_start = pd.Timestamp(self.start_date).floor("D")
            target_end = pd.Timestamp(self.end_date).ceil("D")

        target_dates = list(pd.date_range(target_start, target_end))
        expected_index = pd.date_range(target_start, target_end, freq="1min", inclusive="left")

        valid_dates = [date for date in target_dates if date in all_dates_with_data]
        data_year_month_list = list(sorted(set([(date.year, date.month) for date in valid_dates])))
        df_list = []
        for year, month in data_year_month_list:
            matching_query_fpaths = fpaths_by_period[f"{year}{month:02d}"]
            latest_fp = latest_file(matching_query_fpaths)
            try:
                df_ = pd.read_csv(latest_fp, index_col=0, parse_dates=True)
                if df_.index.duplicated().any():
                    df_ = df_.loc[~df_.index.duplicated(keep="first")]
                df_ = df_.rename(columns={"POA_DTN": "POA"})
                if "pvlib" in latest_fp.name.lower():
                    df_.columns = df_.columns.map(str.lower)
                elif "meter" in latest_fp.name.lower():
                    df_.columns = df_.columns.str.replace("_", ".")  # should be OE.MeterMW
                df_list.append(df_)
                self.source_files.append(str(latest_fp))
            except:
                raise ValueError("Problem loading data")

        if len(df_list) > 0:
            common_cols = [
                c for c in df_list[0].columns if all((c in df_.columns) for df_ in df_list)
            ]
            formatted_df_list = [df_[common_cols].copy() for df_ in df_list]
            df = pd.concat(formatted_df_list, axis=0)
            if df.index.duplicated().any():
                df = df.loc[~df.index.duplicated(keep="first")]
            df = df.reindex(expected_index)
            self.data = df
        else:
            self.data = pd.DataFrame()

    @classmethod
    def from_defined_query_attributes(
        cls,
        site_name: str,
        start_date: str,
        end_date: str,
        asset_group: str,
        freq: str,
        keep_tzinfo: bool = False,
        data_format: str = "wide",
        q: bool = True,
    ):
        """Queries data from PI for pre-defined attributes used in monthly reporting."""
        site = SolarSite(site_name)
        attribute_paths = site.query_attributes.get(asset_group)

        if asset_group == "Modules":
            common_attpaths = site.get_attribute_paths(attributes=["OE.ModulesOffline"])
            if not attribute_paths:
                attribute_paths = common_attpaths
            else:
                attribute_paths.insert(0, common_attpaths[0])

        if not attribute_paths:
            err_msg = f"No pre-defined attributes found for {site_name=}, {asset_group=}."
            raise KeyError(f"{err_msg}\nSupported asset groups: {[*site.query_attributes]}")

        dataset = cls(site, start_date, end_date)
        dataset._query_data_from_pi(attribute_paths, freq, keep_tzinfo, data_format, q)
        return dataset

    def _query_data_from_pi(
        self, attribute_paths: list, freq: str, keep_tzinfo: bool, data_format: str, q: bool
    ):
        dataset = PIDataset.from_attribute_paths(
            site_name=self.site.name,
            attribute_paths=attribute_paths,
            start_date=self.start_date,
            end_date=self.end_date,
            freq=freq,
            keep_tzinfo=keep_tzinfo,
            data_format=data_format,
            q=q,
        )
        self.data = dataset.data.copy()
        self.invalid_items = dataset.invalid_items

    @classmethod
    def from_pi_for_monthly_report(
        cls,
        site: str,
        year: int,
        month: int,
        asset_group: str,
        freq: str = None,
        q: bool = True,
    ):
        """Queries data for pre-defined query attributes, for use with monthly FlashReports."""
        start = pd.Timestamp(year, month, 1)
        end = start + pd.DateOffset(months=1)
        start_date, end_date = map(lambda d: d.strftime("%Y-%m-%d"), [start, end])
        if freq is None:
            freq = "1m" if asset_group in ["Inverters", "Met Stations", "Meter", "PPC"] else "1h"
        kwargs = dict(start_date=start_date, end_date=end_date, freq=freq, q=q)
        return cls.from_defined_query_attributes(site_name=site, **kwargs, asset_group=asset_group)

    def _load_single_dtn_file(self, filepath):
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        if type(df.index) != pd.DatetimeIndex:
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S%z", utc=True).tz_convert(
                self.site.timezone
            )
        self.source_files.append(str(filepath))
        return df

    def _load_dtn_data_from_files(self, keep_tz: bool = False, q: bool = True):
        """loads data from existing files in the flashreport DTN folders"""
        qprint = quiet_print_function(q=q)

        idx_kwargs = dict(freq="1h", inclusive="left")
        if keep_tz:
            idx_kwargs.update({"tz": self.site.timezone})
        expected_index = pd.date_range(self.start_date, self.end_date, **idx_kwargs)

        required_periods = list(sorted(set((x.year, x.month) for x in expected_index)))
        df_list = []
        for year, month in required_periods:
            dtn_folder = Path(frpath(year, month), "DTN")
            if not dtn_folder.exists():
                qprint(f"No DTN folder found for {year}-{month}.")
                continue
            dtn_fpath = latest_file(list(dtn_folder.glob(f"*{self.site.name}*")))
            if dtn_fpath is None:
                qprint(f"No {year}-{month} DTN file found for {self.site.name}.")
                continue
            dfd = self._load_single_dtn_file(dtn_fpath)
            df_list.append(dfd)
        if not df_list:
            qprint(f"End collection: no data found.")
            return

        # get common columns before concatenation
        is_common = lambda col: all(col in df_.columns for df_ in df_list)
        all_columns = list(
            sorted(set(itertools.chain.from_iterable(df_.columns for df_ in df_list)))
        )
        common_cols = list(filter(is_common, all_columns))
        df = pd.concat([df_[common_cols] for df_ in df_list], axis=0)

        df = df.loc[~df.index.duplicated(keep="first")].copy()

        if not keep_tz:
            df = remove_tzinfo_and_standardize_index(df)

        if len(expected_index) != len(df):
            df = df.drop_duplicates().reindex(expected_index)

        self.data = df.copy()
        return

    @classmethod
    def from_dtn_files(
        cls, site_name: str, start_date: str, end_date: str, keep_tz: bool = False, q: bool = True
    ):
        """loads data from existing files in the flashreport DTN folders"""
        site = SolarSite(site_name)
        target_start = pd.Timestamp(start_date).floor("D")
        target_end = pd.Timestamp(end_date).ceil("D")

        dataset = cls(site, target_start, target_end)
        dataset._load_dtn_data_from_files(keep_tz=keep_tz, q=q)
        return dataset
