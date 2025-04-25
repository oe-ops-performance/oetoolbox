import clr, sys
from numbers import Number
import numpy as np
import pandas as pd

from .assets import PISite
from .dataset import Dataset
from .datetime import segmented_date_ranges

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

    def __init__(self, site: PISite):
        super().__init__()
        self.site = site
        self.tz = site.timezone
        self.piserver = PIServers().DefaultPIServer
        self.afserver = PISystems().DefaultPISystem
        self.database = self.afserver.Databases.get_Item("Onward Energy")

    @property
    def default_parameters(self):
        """Returns a dictionary of default values for query function arguments"""
        return {
            "summary_type": SUMMARY_TYPES["average"],
            "calculation_basis": CALCULATION_BASIS["time_weighted"],
            "timestamp_calculation": TIMESTAMP_CALCULATION["auto"],
            "paging_configuration": PAGING_CONFIGURATION,
        }

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
        q: bool = True,
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
        dataset = cls(site)
        dataset.query(
            start_date=start_date,
            end_date=end_date,
            freq=freq,
            pipoint_names=pipoint_names,
            keep_tzinfo=keep_tzinfo,
            q=q,
        )
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
        data_format: str = "wide",
        q: bool = True,
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
        dataset = cls(site)
        dataset.query(
            start_date=start_date,
            end_date=end_date,
            freq=freq,
            attribute_paths=attribute_paths,
            keep_tzinfo=keep_tzinfo,
            data_format=data_format,
            q=q,
        )
        return dataset

    def create_pipoint_list(self, pipoint_names):
        """Returns PIPointList object, which is a collection of PIPoint objects"""
        pipoint_list = PIPointList()
        for pipt in pipoint_names:
            try:
                pipoint_list.Add(PIPoint.FindPIPoint(self.piserver, pipt))
            except Exception as e:
                raise ValueError(f"could not find {pipt = }; Exception: {e}")
        return pipoint_list

    def create_attribute_list(self, attribute_path_list):
        """Returns AttributeList object, which is a collection of AFAttribute objects"""
        attribute_list = AFAttributeList()
        for att in attribute_path_list:
            try:
                attribute_list.Add(AFAttribute.FindAttribute(att, self.database))
            except Exception as e:
                raise ValueError(f"could not find {att = }; Exception: {e}")
        return attribute_list

    def get_data_object(self, item_list: list, item_type: str):
        """Returns data object for use in query (either PIPointList, or AFAttributeList.Data)

        Parameters
        ----------
        item_list : list of str
            A list of either pipoint names or attribute paths
        item_type : str
            The corresponding type of item contained in item_list
            Either "pipoint" or "attribute"

        Returns
        -------
        OSIsoft.AF.PI.PIPointList or OSIsoft.AF.Asset.AttributeList.Data object that is
        used to call query method (e.g. Summaries, RecordedValues)
        """
        if item_type not in ["attribute", "pipoint"]:
            raise ValueError(f"Invalid argument {item_type = }.")
        if item_type == "pipoint":
            return self.create_pipoint_list(item_list)
        return self.create_attribute_list(item_list).Data

    def get_af_time_range(self, start_date: str, end_date: str):
        """Returns AFTimeRange object with site- and user- specific timezone info"""
        aftimezone = AFTimeZone().CurrentAFTimeZone
        aftz_fmtprovider = AFTimeZoneFormatProvider(aftimezone)
        start = pd.Timestamp(start_date, tz=self.tz)
        end = pd.Timestamp(end_date, tz=self.tz)
        return AFTimeRange(str(start), str(end), aftz_fmtprovider)

    def get_af_time_span(self, freq: str):
        """Returns AFTimeSpan object for specifying query frequency/interval"""
        return AFTimeSpan.Parse(freq)

    def get_date_range_list(self, start_date: str, end_date: str, n_days: int = 10):
        """Returns list of sub date ranges for query if longer than 10 days"""
        start, end = map(pd.Timestamp, [start_date, end_date])
        total_days = (end - start).days
        if total_days < n_days:
            n_days = total_days
        return segmented_date_ranges(start, end, n_days)

    def format_pi_dataframe(
        self, dataframe: pd.DataFrame, freq: str, data_format: str, keep_tzinfo: bool = False
    ) -> pd.DataFrame:
        """Formats the data output from PI query.

        Parameters
        ----------
        dataframe : pd.DataFrame
            A dataframe consisting of values and timestamps from the .GetValueArrays method
        freq : str
            The frequency/interval, formatted for PI (e.g. "1m", "5m", "1h")
        data_format : str
            The format for the output dataframe; either "wide" or "long"
        keep_tzinfo : bool, optional
            Whether to return a timezone-aware dataframe

        Returns
        -------
        pd.DataFrame with a datetime index
        """
        df = dataframe.copy()
        is_object_col = lambda col: col.endswith("_ID") or col in ["Attribute", "PIPoint"]
        numeric_cols = [c for c in df.columns if not is_object_col(c)]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").copy()
        df.index = pd.to_datetime(df.index.astype(str), utc=True)
        df = df.tz_convert(tz=self.tz)
        if not keep_tzinfo:
            df = df.tz_localize(None)
        if data_format == "wide":
            start, end = df.index.min(), df.index.max()
            expected_index = pd.date_range(start, end, freq=freq.replace("m", "min"))
            if df.index.duplicated().any():
                df = df.loc[~df.index.duplicated(keep="first")]
            if df.shape[0] != expected_index.shape[0]:
                df = df.reindex(expected_index)
        df = df.rename_axis("Timestamp")
        return df

    def query(
        self,
        start_date: str,
        end_date: str,
        freq: str,
        pipoint_names: list = None,
        attribute_paths: list = None,
        keep_tzinfo: bool = False,
        data_format: str = "wide",
        q: bool = True,
        **query_kwargs,
    ):
        """Runs PI query for the specified date range and parameters"""
        qprint = lambda msg, end="\n": None if q else print(msg, end=end)
        if not any([pipoint_names, attribute_paths]):
            raise ValueError("Must provide either pipoint names or attribute paths")
        if not attribute_paths:
            item_type, item_list = "pipoint", pipoint_names
            item_names = item_list
        else:
            item_type, item_list = "attribute", attribute_paths
            item_names = self._columns_from_attpaths(attribute_paths, data_format)
            id_columns = [] if data_format == "wide" else self._get_long_id_columns(attribute_paths)

        # get query parameters
        data_object = self.get_data_object(item_list=item_list, item_type=item_type)
        time_span = self.get_af_time_span(freq)
        custom_kwargs = self._validate_query_kwargs(query_kwargs)
        query_args = [
            val if key not in custom_kwargs else custom_kwargs[key]
            for key, val in self.default_parameters.items()
        ]
        date_range_list = self.get_date_range_list(start_date, end_date)

        # run query for sub date ranges
        main_df_list = []
        for i, sub_range in enumerate(date_range_list):
            qprint(f"Querying range {i+1} of {len(date_range_list)}")
            time_range = self.get_af_time_range(*sub_range)
            summaries = data_object.Summaries(time_range, time_span, *query_args)

            sub_df_list = []
            for output, item_name, item in zip(summaries, item_names, item_list):
                args = [output, data_format, item_name, item_type]
                kwargs = {}
                if data_format == "long" and item_type == "attribute":
                    item_elements = self._attpath_meta(attpath=item)["elements"]
                    if len(item_elements) < len(id_columns):
                        item_elements += [np.nan] * (len(id_columns) - len(item_elements))
                    kwargs = dict(item_elements=item_elements, id_columns=id_columns)

                df_ = self._retrieve_query_output_data(*args, **kwargs)
                sub_df_list.append(df_)

            df_ = pd.concat(sub_df_list, axis=1)
            df_ = self.format_pi_dataframe(df_, freq, data_format, keep_tzinfo=keep_tzinfo)
            main_df_list.append(df_)

        df = pd.concat(main_df_list, axis=0, ignore_index=False)
        self.data = df.copy()

    def _attpath_meta(self, attpath: str):
        """Returns a dictionary of metadata for given attribute path"""
        path_parts = attpath.replace("|", "\\").split("\\")
        group_index = path_parts.index(self.site.name) + 1
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
        path_parts = attpath.replace("|", "\\").split("\\")
        start_index = path_parts.index(self.site.name) + 1  # asset group index
        if skip_asset_group:
            start_index += 1
        return "_".join(reversed(path_parts[start_index:]))

    def _columns_from_attpaths(self, attpath_list: list, data_format: str):
        """Returns a list of columns names for query"""
        skip_group = self._attpath_list_meta(attpath_list)["all_same_group"]
        if data_format == "wide":
            return [self._column_name_from_attpath(p, skip_group) for p in attpath_list]

    def _get_long_id_columns(self, attpath_list: list):
        """Returns a list of necessary ID columns for use with long-format data"""
        n_elements_max = self._attpath_list_meta(attpath_list)["max_level"]
        id_columns = ["Group_ID", "Asset_ID", "Subasset_ID", "Subasset2_ID"]
        if n_elements_max <= 4:
            return id_columns[:n_elements_max]
        for i in range(n_elements_max - 4):
            id_columns.append(f"Subasset{i+3}_ID")
        return id_columns

    def _get_long_format_kwargs(self, item_list: list, item_type: str):
        """Returns dictionary of keyword arguments for use with long-format data from attributes

        Parameters
        ----------
        item_list : list of str
            A list of either pipoint names or attribute paths
        item_type : str
            The corresponding type of item contained in item_list
            Either "pipoint" or "attribute"
        """
        if item_type == "pipoint":
            return {}

    def _check_for_bad_data(self, values, tstamps, flags):
        """Checks outputs of .GetValueArrays() function in query (returns True if data is bad)"""
        condition_1 = (len(list(values)) == 1) and (list(flags)[0].ToString() == "Bad")
        condition_2 = not any(isinstance(x, Number) for x in list(values))
        return any([condition_1, condition_2])

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
        """
        # TODO: finish building out for long format
        if data_format == "long":
            return pd.DataFrame()

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
