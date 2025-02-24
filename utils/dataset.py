from abc import ABC, abstractmethod
import pandas as pd

from .assets import SolarSite
from .oepaths import flashreports as report_dir, sorted_filepaths


class Dataset(ABC):
    """Base class for several types of datasets (e.g. InverterDataset, WeatherDataset, etc.)

    Attributes
    ----------
    data : pandas.DataFrame
        Time series data for dataset with index of type pandas.DatetimeIndex

    """

    def __init__(self):
        self.data = self.empty_dataframe()

    @property
    def start_datetime(self):
        """Earliest timestamp in dataset"""
        if self.data.empty:
            return pd.NaT
        return self.data.index.min()

    @property
    def end_datetime(self):
        """Latest timestamp in dataset"""
        if self.data.empty:
            return pd.NaT
        return self.data.index.max()

    @property
    @abstractmethod
    def columns(self):
        """List of columns to produce empty dataframe"""
        pass  # implemented by subclass

    def empty_dataframe(self):
        """Produces an empty dataframe for the given data format"""
        return pd.DataFrame(columns=self.columns)


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

    def __init__(self, site: SolarSite):
        super().__init__()
        self.site = site
        self.tz = site.timezone
        self.source_files = []  # init

    @property
    def columns(self):
        return []  # not sure if needed here.. but can use for PVLibDataset (i.e. expected columns)

    @classmethod
    def from_existing_query_files(
        cls, site_name: str, asset_group: str, start_date: str = None, end_date: str = None
    ):
        """Loads data from existing query files in flashreport folders

        Parameters
        ----------
        site_name : str
            Name of site (used to initialize SolarSite instance)
        asset_group : str
            The particular asset group to load data for
        start_date : str, optional
            The target start date for the dataset, default = None
        end_date : str, optional
            The target end date for the dataset, default = None
        """
        site = SolarSite(site_name)
        dataset = cls(site)
        dataset._load_data_from_query_files(asset_group, start_date, end_date)
        return dataset

    def _load_data_from_query_files(
        self, asset_group: str, start_date: str = None, end_date: str = None
    ):
        if asset_group not in self.site.asset_groups:
            raise ValueError("Invalid asset group specified")
        group = asset_group.replace(" ", "")
        fpaths_by_period = {
            key_: [fp for fp in fpath_list if group in fp.name]
            for key_, fpath_list in self.site.data_files_by_period.items()
            if any(group in fp.name for fp in fpath_list)
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

        if any(date is None for date in [start_date, end_date]):
            start_date = pd.Timestamp(*valid_year_month_list[0], 1)
            end_date = pd.Timestamp(*valid_year_month_list[-1], 1) + pd.DateOffset(months=1)

        target_start = pd.Timestamp(start_date).floor("D")
        target_end = pd.Timestamp(end_date).floor("D")
        target_dates = list(pd.date_range(target_start, target_end))

        idx_end = pd.Timestamp(end_date).ceil("D")
        expected_index = pd.date_range(target_start, idx_end, freq="1min")[:-1]

        valid_dates = [date for date in target_dates if date in all_dates_with_data]
        data_year_month_list = list(sorted(set([(date.year, date.month) for date in valid_dates])))
        df_list = []
        for year, month in data_year_month_list:
            matching_query_fpaths = fpaths_by_period[f"{year}{month:02d}"]
            latest_fp = sorted_filepaths(matching_query_fpaths)[0]
            try:
                df_ = pd.read_csv(latest_fp, index_col=0, parse_dates=True)
                if df_.index.duplicated().any():
                    df_ = df_.loc[~df_.index.duplicated(keep="first")]
                df_list.append(df_)
                self.source_files.append(str(latest_fp))
            except:
                raise ValueError("Problem loading data")

        common_cols = [c for c in df_list[0].columns if all((c in df_.columns) for df_ in df_list)]
        formatted_df_list = [df_[common_cols].copy() for df_ in df_list]
        df = pd.concat(formatted_df_list, axis=0)
        if df.index.duplicated().any():
            df = df.loc[~df.index.duplicated(keep="first")]
        df = df.reindex(expected_index)
        self.data = df
