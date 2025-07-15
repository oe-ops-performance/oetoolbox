import clr, sys
from numbers import Number
import numpy as np
import pandas as pd
from typing import Literal, Union

from .assets import PISite
from .config import PI_DATABASE_PATH
from .dataset import Dataset
from .datetime import segmented_date_ranges
from .helpers import with_retries

# add reference path for PI AFSDK import
sys.path.append(r"C:\Program Files (x86)\PIPC\AF\PublicAssemblies\4.0")
clr.AddReference("OSIsoft.AFSDK")
import OSIsoft  # type: ignore
from OSIsoft.AF import PISystems  # type: ignore
from OSIsoft.AF.Search import AFEventFrameSearch  # type: ignore
from OSIsoft.AF.Asset import AFAttribute, AFAttributeList, AFSearchMode  # type: ignore
from OSIsoft.AF.PI import (  # type: ignore
    PIPageType,
    PIPagingConfiguration,
    PIPoint,
    PIPointList,
    PIServers,
)
from OSIsoft.AF.Time import (  # type: ignore
    AFTime,
    AFTimeRange,
    AFTimeSpan,
    AFTimeZone,
    AFTimeZoneFormatProvider,
)
from OSIsoft.AF.Data import (  # type: ignore
    AFBoundaryType,
    AFCalculationBasis,
    AFListData,
    AFSummaryTypes,
    AFTimestampCalculation,
)

# reference for enumerations from AF SDK classes (used with query function inputs)
SUMMARY_TYPES = {
    "average": AFSummaryTypes.Average,
    "minimum": AFSummaryTypes.Minimum,
    "maximum": AFSummaryTypes.Maximum,
    "total": AFSummaryTypes.Total,
}

CALCULATION_BASIS = {
    "event_weighted": AFCalculationBasis.EventWeighted,
    "time_weighted": AFCalculationBasis.TimeWeighted,
    "time_weighted_continuous": AFCalculationBasis.TimeWeightedContinuous,
    "time_weighted_discrete": AFCalculationBasis.TimeWeightedDiscrete,
}

TIMESTAMP_CALCULATION = {
    "auto": AFTimestampCalculation.Auto,
    "earliest": AFTimestampCalculation.EarliestTime,
    "most_recent": AFTimestampCalculation.MostRecentTime,
}

PAGING_CONFIGURATION = PIPagingConfiguration(PIPageType.TagCount, 1000)

QUERY_PARAMETERS = {
    "summary_type": SUMMARY_TYPES,
    "calculation_basis": CALCULATION_BASIS,
    "timestamp_calculation": TIMESTAMP_CALCULATION,
}

DEFAULT_QUERY_PARAMETERS = {
    "summary_type": SUMMARY_TYPES["average"],
    "calculation_basis": CALCULATION_BASIS["time_weighted"],
    "timestamp_calculation": TIMESTAMP_CALCULATION["auto"],
    "paging_configuration": PAGING_CONFIGURATION,
}


def validate_query_item_list(item_list: list):
    """Verifies that all items in list are either attribute paths or pipoints (not a mix)
    -> returns item_type (attribute or pipoint)
    """
    if not item_list:
        raise ValueError("No query items/tags specified.")
    all_attpaths = all(item.startswith(PI_DATABASE_PATH) for item in item_list)
    all_pipoints = not any(PI_DATABASE_PATH in item for item in item_list)
    if not (all_attpaths or all_pipoints):
        raise ValueError(
            "All items must be of the same type (i.e. all attribute paths, or all pipoints)"
        )
    return "attribute" if all_attpaths else "pipoint"


class PIDataset(Dataset):
    """A dataset loaded from PI containing time series data for specified attributes/pipoints.
    Can also initialize from a PI site to leverage metadata/methods. Inherits from Dataset.

    Additional Parameters
    ---------------------
    site : oetoolbox.utils.assets.PISite
        An instance of a PI site, providing relevant attributes/methods for obtaining data

    Additional Attributes
    ---------------------
    site : oetoolbox.utils.assets.PISite
        An instance of a PI site, providing relevant attributes/methods for obtaining data
    """

    def __init__(
        self,
        site: PISite,
        original_item_list: list,
        start_date: str,
        end_date: str,
        freq: str,
        keep_tzinfo: bool = False,
    ):
        super().__init__()
        self.site = site
        self.item_type = validate_query_item_list(original_item_list)  # confirms all same type
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.tz = site.timezone if keep_tzinfo else None

        # database related
        self.piserver = PIServers().DefaultPIServer
        self.afserver = PISystems().DefaultPISystem
        self.database = self.afserver.Databases.get_Item("Onward Energy")

        # for tracking items that don't exist in database
        self.item_list = original_item_list  # init
        self.invalid_items = []  # init

    @property
    def expected_index(self):
        """note: freq is formatted for PI (not Pandas)"""
        return pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=pd.Timedelta(self.freq),
            tz=self.tz,
            inclusive="left",
        )

    @property
    def af_time_span(self):
        """Returns AFTimeSpan object for specifying query frequency/interval"""
        return AFTimeSpan.Parse(self.freq)

    @property
    def columns(self):
        return []  # not sure if needed here.. TODO: revisit origin in Dataset class

    @classmethod
    def from_pipoints(
        cls,
        site_name: str,
        pipoint_names: list,
        start_date: str,
        end_date: str,
        freq: str,
        keep_tzinfo: bool = False,
        raise_not_found: bool = False,
        q: bool = True,
        **query_kwargs,
    ):
        """Queries data for given pipoints across specified date range

        Parameters
        ----------
        site_name : str
            Name of site (used to initialize PISite instance) - currently only used for timezone
        pipoint_names : list
            List of pipoint names as strings
        start_date : str
            The start date for which to query data
        end_date : str
            The end date for which to query data
        freq : str
            The frequency/interval, formatted for PI (e.g. "1m", "5m", "1h")
        keep_tzinfo : bool, optional
            Whether to return a tz-aware dataset, by default False
        q : bool, optional
            Quiet parameter (i.e. suppress status printouts), by default True
        """
        site = PISite(site_name)
        dataset = cls(site, pipoint_names, start_date, end_date, freq, keep_tzinfo)
        kwargs = dict(data_format="wide", raise_not_found=raise_not_found, q=q) | query_kwargs
        dataset._run_query(**kwargs)
        return dataset

    @classmethod
    def from_attribute_paths(
        cls,
        site_name: str,
        attribute_paths: list,
        start_date: str,
        end_date: str,
        freq: str,
        keep_tzinfo: bool = False,
        raise_not_found: bool = False,
        data_format: str = "wide",
        q: bool = True,
        **query_kwargs,
    ):
        """Queries data for given attribute paths across specified date range

        Parameters
        ----------
        site_name : str
            Name of site (used to initialize PISite instance) - currently only used for timezone
        attribute_paths : list
            List of full attribute paths as strings
        start_date : str
            The start date for which to query data
        end_date : str
            The end date for which to query data
        freq : str
            The frequency/interval, formatted for PI (e.g. "1m", "5m", "1h")
        keep_tzinfo : bool, optional
            Whether to return a tz-aware dataset, by default False
        data_format : str, optional
            The format of the output dataframe; either "wide" or "long", default "wide"
        q : bool, optional
            Quiet parameter (i.e. suppress status printouts), by default True
        """
        site = PISite(site_name)
        dataset = cls(site, attribute_paths, start_date, end_date, freq, keep_tzinfo)
        kwargs = dict(data_format=data_format, raise_not_found=raise_not_found, q=q) | query_kwargs
        dataset._run_query(**kwargs)
        return dataset

    def create_af_object_list(self, raise_not_found: bool):
        """creates either PIPointList or AFAttributeList object using defined item_list"""
        obj_list = PIPointList() if self.item_type == "pipoint" else AFAttributeList()
        for item in self.item_list:
            try:
                if self.item_type == "pipoint":
                    obj_list.Add(PIPoint.FindPIPoint(self.piserver, item))
                else:
                    obj_list.Add(AFAttribute.FindAttribute(item, self.database))
            except Exception as e:
                if raise_not_found:
                    raise ValueError(f"could not find {item = }; Exception: {e}")
                self.invalid_items.append(item)
        if obj_list.Count > 0:
            return obj_list
        raise ValueError(f"No valid {self.item_type}s found in list.")

    def create_data_object(self, raise_not_found: bool):
        """Returns data object using defined item_list
        -> when raise_not_found=False, invalid items are removed from self.item_list
        """
        af_obj_list = self.create_af_object_list(raise_not_found=raise_not_found)
        for item in self.invalid_items:
            self.item_list.remove(item)
        if self.item_type == "pipoint":
            return af_obj_list
        return af_obj_list.Data

    def get_item_names(self, data_format: str):
        if self.item_type == "pipoint":
            item_names = self.item_list  # list of pipoint names
        else:
            item_names = self._columns_from_attpaths(data_format)
        return item_names

    def get_af_time_range(self, start_date: str, end_date: str):
        """Returns AFTimeRange object for given dates with site- and user- specific timezone info"""
        aftimezone = AFTimeZone().CurrentAFTimeZone
        aftz_fmtprovider = AFTimeZoneFormatProvider(aftimezone)
        start = pd.Timestamp(start_date, tz=self.site.timezone)
        end = pd.Timestamp(end_date, tz=self.site.timezone)
        return AFTimeRange(str(start), str(end), aftz_fmtprovider)

    def get_date_range_list(self, sub_range: int = 10):
        """Returns list of sub date ranges for query if longer than 10 days

        Parameters
        ----------
        sub_range : int
            The number of days for each sub date range.
        """
        start, end = map(pd.Timestamp, [self.start_date, self.end_date])
        total_days = (end - start).days
        if total_days < sub_range:
            sub_range = total_days
        return segmented_date_ranges(start, end, sub_range)

    def _run_query(
        self,
        data_format: str = "wide",
        raise_not_found: bool = False,
        q: bool = True,
        **query_kwargs,
    ):
        """Runs PI query for the specified date range and parameters"""
        qprint = lambda msg, end="\n": None if q else print(msg, end=end)

        # get AFSDK data object for query (checks item_list & removes invalid items)
        data_object = self.create_data_object(raise_not_found=raise_not_found)
        item_names = self.get_item_names(data_format)

        # get AFSDK-specific query parameters
        afsdk_kwargs = DEFAULT_QUERY_PARAMETERS
        afsdk_kwargs["time_span"] = self.af_time_span

        # check for any default parameter overrides
        custom_kwargs = self._validate_query_kwargs(query_kwargs)
        for key in afsdk_kwargs.keys():
            if key in custom_kwargs:
                afsdk_kwargs.update({key: custom_kwargs[key]})

        # additional keyword arguments for query function
        item_metadata = [(name, item) for name, item in zip(item_names, self.item_list)]
        kwargs = dict(data_format=data_format, item_metadata=item_metadata)

        # split date range & run query for sub ranges
        date_range_list = self.get_date_range_list(sub_range=11)
        df_list = []
        for i, sub_range in enumerate(date_range_list):
            qprint(f"Querying range {i+1} of {len(date_range_list)}")
            time_range = self.get_af_time_range(*sub_range)
            df_ = self._query_summaries(data_object, time_range, **afsdk_kwargs, **kwargs)
            df_ = self._format_pi_data(df_, data_format)
            df_list.append(df_)

        df = pd.concat(df_list, axis=0, ignore_index=False)
        if not all(x in df.index for x in self.expected_index):  # validate index
            df = df.reindex(self.expected_index)
        df = df.rename_axis("Timestamp")

        self.data = df.copy()
        qprint(f"Done. {df.shape = }")

    def _format_pi_data(self, dataframe: pd.DataFrame, data_format: str) -> pd.DataFrame:
        """Formats the data output from PI query.

        Parameters
        ----------
        dataframe : pd.DataFrame
            A dataframe consisting of values and timestamps from the .GetValueArrays method
        data_format : str
            The format for the output dataframe; either "wide" or "long"

        Returns
        -------
        pd.DataFrame with a datetime index
        """
        if dataframe.empty:
            return dataframe
        df = dataframe.copy()
        is_object_col = lambda col: col.endswith("_ID") or col in ["Attribute", "PIPoint"]
        numeric_cols = [c for c in df.columns if not is_object_col(c)]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").copy()
        df.index = pd.to_datetime(df.index.astype(str), utc=True)
        df = df.tz_convert(tz=self.site.timezone)
        if self.tz is None:
            df = df.tz_localize(None)
        if data_format == "wide":
            start, end = df.index.min(), df.index.max()
            expected_index = pd.date_range(start, end, freq=self.freq.replace("m", "min"))
            if df.index.duplicated().any():
                df = df.loc[~df.index.duplicated(keep="first")]
            if df.shape[0] != expected_index.shape[0]:
                df = df.reindex(expected_index)
        df = df.rename_axis("Timestamp")
        return df

    def _attpath_meta(self, attpath: str):
        """Returns a dictionary of metadata for given attribute path"""
        path_parts = attpath.replace("|", "\\").split("\\")
        group_index = path_parts.index(self.site.name) + 1
        if f"{self.site.name}|" in attpath:  # if site-level (e.g. Meter)
            group_index -= 1  # make site the group
        related_elements = path_parts[group_index:-1]
        return {
            "parts": path_parts,
            "asset_group": path_parts[group_index],
            "elements": related_elements,
            "level": len(related_elements),
        }

    def _attpath_list_meta(self, attpath_list: list):
        """Returns a dictionary of metadata for given list of attribute paths"""
        all_asset_groups = [self._attpath_meta(p)["asset_group"] for p in attpath_list]
        all_attribute_levels = [self._attpath_meta(p)["level"] for p in attpath_list]
        return {
            "all_same_group": len(set(all_asset_groups)) == 1,
            "all_same_level": len(set(all_attribute_levels)) == 1,
            "max_level": max(all_attribute_levels),
        }

    def _column_name_from_attpath(self, attpath: str, skip_asset_group: bool = True):
        """Returns a column name for use in wide-format dataframe"""
        path_parts = self._attpath_meta(attpath)["parts"]
        group_index = path_parts.index(self.site.name) + 1  # asset group index
        if path_parts[-2] == self.site.name:
            group_index -= 1  # make site the group
        if skip_asset_group:
            group_index += 1
        return "_".join(reversed(path_parts[group_index:]))

    def _columns_from_attpaths(self, data_format: str):
        """Returns a list of columns names for query"""
        skip_group = self._attpath_list_meta(self.item_list)["all_same_group"]
        if data_format == "wide":
            return [self._column_name_from_attpath(p, skip_group) for p in self.item_list]

    @property
    def id_columns(self):
        """Returns a list of necessary ID columns for use with long-format data
        -> note: currently only supports attribute paths
        """
        if self.item_type == "pipoints":
            return []  # long format not currently supported for pipoints
        n_elements_max = self._attpath_list_meta(self.item_list)["max_level"]
        id_columns = ["Group_ID", "Asset_ID", "Subasset_ID", "Subasset2_ID"]
        if n_elements_max <= 4:
            return id_columns[:n_elements_max]
        for i in range(n_elements_max - 4):
            id_columns.append(f"Subasset{i+3}_ID")
        return id_columns

    def _check_for_bad_data(self, values, tstamps, flags):
        """Checks outputs of .GetValueArrays() function in query (returns True if data is bad)"""
        condition_1 = (len(list(values)) == 1) and (list(flags)[0].ToString() == "Bad")
        condition_2 = not any(isinstance(x, Number) for x in list(values))
        return any([condition_1, condition_2])

    @with_retries(n_max=3)
    def _query_summaries(
        self,
        data_object: Union[PIPointList, AFListData],
        time_range: AFTimeRange,
        time_span: AFTimeSpan,
        summary_type: AFSummaryTypes,
        calculation_basis: AFCalculationBasis,
        timestamp_calculation: AFTimestampCalculation,
        paging_configuration: PIPagingConfiguration,
        data_format: Literal["long", "wide"],
        item_metadata: list,
    ) -> pd.DataFrame:
        """queries data using the .Summaries() method from AFSDK on the given data object"""
        id_columns = self.id_columns if data_format == "long" else []
        query_args = [summary_type, calculation_basis, timestamp_calculation, paging_configuration]
        summaries = data_object.Summaries(time_range, time_span, *query_args)
        df_list = []
        for output, item_meta in zip(summaries, item_metadata):
            item_name, item = item_meta
            args = [output, data_format, item_name, self.item_type]
            kwargs = {}
            if data_format == "long" and self.item_type == "attribute":
                item_elements = self._attpath_meta(attpath=item)["elements"]
                if len(item_elements) < len(id_columns):
                    item_elements += [np.nan] * (len(id_columns) - len(item_elements))
                kwargs = dict(item_elements=item_elements, id_columns=id_columns)
            df_ = self._retrieve_query_output_data(*args, **kwargs)
            df_list.append(df_)

        if not df_list:
            raise ValueError("Error retrieving data.")

        if data_format == "wide":
            df = pd.concat(df_list, axis=1)
            # confirm all columns exist in output
            for i, item_meta in enumerate(item_metadata):
                item_name, _ = item_meta
                if item_name not in df.columns:
                    df.insert(i, item_name, np.nan)
        else:  # long format  TODO
            df = pd.concat(df_list, axis=0)
        return df

    def _retrieve_query_output_data(
        self,
        query_output,
        data_format: str,
        item_name: str,
        item_type: str,
        item_elements: list = [],
        id_columns: list = [],
    ):
        """Iterates over the AFValues collection and returns a dataframe with the associated data

        note: the .Summaries() query method returns an object of the following type:
              System.Collections.Generic.IEnumerable[IDictionary[AFSummaryTypes,AFValues]]

        Parameters
        ----------
        query_output : System.Collections.Generic.IDictionary[AFSummaryTypes,AFValues]
            A dictionary-like object with query summary types as keys and data as values.
        data_format : str
            The target output format for the data; either "wide" or "long"
        item_name : str
            The name to use for the data column (if wide format) or asset ID (if long format)
        item_type : str
            The type/origin of data, either "attribute" or "pipoint"
        item_elements : list, optional
            The list of elements associated with attribute; only applies for "long" data
        id_columns : list, optional
            The element ID columns to include; only applies for "long" data from attribute paths

        Returns
        -------
        pd.DataFrame
            A dataframe with a datetime index and single column of data.
            note: dtype is usually float but not guaranteed.
            If all data is bad, returns empty dataframe.
        """
        # get AFValues object from IDictionary (.Values method returns an iterator)
        af_values = [vals for vals in query_output.Values].pop()
        values, tstamps, flags = af_values.GetValueArrays()
        if self._check_for_bad_data(values, tstamps, flags):
            return pd.DataFrame()

        if data_format == "wide":
            return pd.DataFrame({item_name: list(values)}, index=list(tstamps))

        df_ = pd.DataFrame({"Value": list(values)}, index=list(tstamps))
        if item_type == "pipoint":
            df_["PIPoint"] = item_name
            return df_

        df_["Attribute"] = item_name
        for element, id_col in zip(item_elements, id_columns):
            df_[id_col] = element
        return df_

    def _validate_query_kwargs(self, kwargs):
        """Returns dictionary of keys/values that exist in query parameter dict"""
        valid_kwargs = {}
        for key, val in kwargs.items():
            param_dict = QUERY_PARAMETERS.get(key)
            if not param_dict:
                continue
            if val in param_dict:
                valid_kwargs.update({key: val})
        return valid_kwargs
