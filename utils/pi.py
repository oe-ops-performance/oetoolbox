import clr, sys
from numbers import Number
import numpy as np
import pandas as pd
from typing import Literal, Union

from .assets import PISite
from .config import PI_DATABASE_PATH
from .dataset import Dataset
from .datetime import segmented_date_ranges
from .helpers import quiet_print_function, with_retries

# add reference path for PI AFSDK import
sys.path.append(r"C:\Program Files (x86)\PIPC\AF\PublicAssemblies\4.0")
clr.AddReference("OSIsoft.AFSDK")
import OSIsoft  # type: ignore
from OSIsoft.AF import PISystems  # type: ignore
from OSIsoft.AF.Search import AFEventFrameSearch  # type: ignore
from OSIsoft.AF.Asset import AFAttribute, AFAttributeList, AFSearchMode, AFValues  # type: ignore
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

# dictionary for referencing OSIsoft objects/enumerations
AFSDK = {
    "boundary_type": {
        "inside": AFBoundaryType.Inside,
        "outside": AFBoundaryType.Outside,
        "interpolated": AFBoundaryType.Interpolated,
    },
    "calculation_basis": {
        "time_weighted": AFCalculationBasis.TimeWeighted,
        "event_weighted": AFCalculationBasis.EventWeighted,
        "time_weighted_continuous": AFCalculationBasis.TimeWeightedContinuous,
        "time_weighted_discrete": AFCalculationBasis.TimeWeightedDiscrete,
    },
    "paging_configuration": {
        "default": PIPagingConfiguration(PIPageType.TagCount, 1000),
    },
    "summary_type": {
        "total": AFSummaryTypes.Total,
        "average": AFSummaryTypes.Average,
        "minimum": AFSummaryTypes.Minimum,
        "maximum": AFSummaryTypes.Maximum,
        "range": AFSummaryTypes.Range,
        "count": AFSummaryTypes.Count,
        "percent_good": AFSummaryTypes.PercentGood,
    },
    "timestamp_calculation": {
        "auto": AFTimestampCalculation.Auto,
        "earliest": AFTimestampCalculation.EarliestTime,
        "most_recent": AFTimestampCalculation.MostRecentTime,
    },
}

DEFAULT_QUERY_PARAMETERS = {
    "summaries": {
        "summary_type": "average",
        "calculation_basis": "time_weighted",
        "timestamp_calculation": "auto",
        "paging_configuration": "default",
    },
    "recorded_values": {
        "boundary_type": "interpolated",  # only affects end points
        "filter_expression": "",
        "include_filtered_values": False,
        "paging_configuration": "default",
        "max_count": 0,  # If > 0, sets the maximum number of values to be returned
    },
}

QUERY_ARGS = {method: list(params.keys()) for method, params in DEFAULT_QUERY_PARAMETERS.items()}


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
        An instance of a PI site, providing relevant attributes/methods for obtaining data.
    original_item_list : list[str]
        A list of pipoint names or attribute paths to query. Must be one or the other, not a mix.
    start_date : str
        The start date for the query range. Format = "%Y-%m-%d".
    end_date : str
        The end date for the query range. Format = "%Y-%m-%d".
    method : Literal["summaries", "recorded_values"]
        The type of query to execute.
    freq : str
        The data interval, formatted for PI (e.g. "1m", "5m", "1h").
        Must specify when using the summaries method.
    keep_tzinfo : bool
        Whether to return a tz-aware dataset.
    data_format : Literal["long", "wide"]
        The format of the output dataframe. Only applies when using the summaries method;
        forced to False when querying recorded_values due to discontinuous nature of readings.
    raise_not_found : bool, optional
        Determines whether to raise an Exception when query tag(s) do not exist. Default = False.

    Additional Attributes
    ---------------------
    site : PISite
    start_date : str
    end_date : str
    method : str
    freq : str
    data_format : str
    tz : str
        The timezone for the output data. If keep_tzinfo=True, uses tz from SolarSite object.
    piserver : OSIsoft.AF.PI.PIServer
        The default PIServer configured for the current client machine.
    afserver : OSIsoft.AF.PISystem
        The default PISystem configured in the AF SDK.
    database : OSIsoft.AF.AFDatabase
        The database that contains data for Onward Energy fleet assets.
    af_data_object : Union[OSIsoft.AF.Data.AFListData, OSIsoft.AF.PI.PIPointList]
        The AFSDK object for the specified pipoints or attribute paths.
    item_type : Literal["attribute", "pipoint"]
        The type of item designated in original_item_list.
    item_list : list[str]
        A list of validated items from original_item_list. Initializes with original list;
        if raise_not_found=False, invalid items are removed from the list.
    invalid_items : list[str]
        A list of invalid items from original_item_list. Initializes as empty list; if
        raise_not_found=False, appends any invalid items to the list.
    """

    def __init__(
        self,
        site: PISite,
        original_item_list: list,
        start_date: str,
        end_date: str,
        method: Literal["summaries", "recorded_values"],
        freq: str,
        keep_tzinfo: bool,
        data_format: Literal["long", "wide"],
        raise_not_found: bool = False,
    ):
        super().__init__()
        self.site = site
        self.start_date = start_date
        self.end_date = end_date
        self.method = method
        self.freq = freq
        self.tz = site.timezone if keep_tzinfo else None
        self.data_format = data_format

        # validate specified query method and data format
        if method not in ["summaries", "recorded_values"]:
            raise KeyError("Invalid method.")
        if data_format not in ["long", "wide"]:
            raise KeyError("Invalid data format.")

        # force data format to long if querying recorded values
        if method == "recorded_values" and data_format == "wide":
            self.data_format = "long"
            print(
                "Note: specified 'wide' format, overriding to 'long' due to "
                "the nature of recorded values (i.e. irregular timestamps)."
            )

        # database related
        self.piserver = PIServers().DefaultPIServer
        self.afserver = PISystems().DefaultPISystem
        self.database = self.afserver.Databases.get_Item("Onward Energy")

        # validate specified query items/tags
        self.item_type = validate_query_item_list(original_item_list)  # confirms all same type
        self.item_list = original_item_list  # init
        self.invalid_items = []  # init

        # get af data object for specified query items (includes validation)
        self.af_data_object = self._create_data_object(raise_not_found=raise_not_found)

    @property
    def expected_index(self):
        """note: freq is formatted for PI (not Pandas)"""
        if self.method == "recorded_values":
            return None
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
        return []  # TODO - evaluate parent class & determine if needed

    def expected_columns(self):
        """Returns list of columns for expected output dataframe"""
        if self.data_format == "wide":
            return self.item_names
        data_col = "PIPoint" if self.item_type == "pipoint" else "Attribute"
        return ["Value", data_col, *self.id_columns]

    @classmethod
    def from_pipoints(
        cls,
        site_name: str,
        pipoint_names: list,
        start_date: str,
        end_date: str,
        method: str = "summaries",
        freq: str = None,
        keep_tzinfo: bool = False,
        data_format: str = "wide",
        raise_not_found: bool = False,
        n_segment: int = 11,
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
        data_format : str, optional
            The format of the output dataframe; either "wide" or "long", default "wide"
        raise_not_found : bool, optional
            Whether to raise an error if any of the specified attributes does not exist.
            Defaults to False, which excludes non-existent items.
        n_segment : int, optional
            The number of days by which to segment the query range. Default = 11.
        q : bool, optional
            Quiet parameter (i.e. suppress status printouts), by default True
        """
        dataset = cls(
            site=PISite(site_name),
            original_item_list=pipoint_names,
            start_date=start_date,
            end_date=end_date,
            method=method,
            freq=freq,
            keep_tzinfo=keep_tzinfo,
            data_format=data_format,
            raise_not_found=raise_not_found,
        )
        dataset._run_query(n_segment=n_segment, q=q, **query_kwargs)
        return dataset

    @classmethod
    def from_attribute_paths(
        cls,
        site_name: str,
        attribute_paths: list,
        start_date: str,
        end_date: str,
        freq: str = None,
        method: str = "summaries",
        keep_tzinfo: bool = False,
        data_format: str = "wide",
        raise_not_found: bool = False,
        n_segment: int = 11,
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
        raise_not_found : bool, optional
            Whether to raise an error if any of the specified attributes does not exist.
            Defaults to False, which excludes non-existent items.
        n_segment : int, optional
            The number of days by which to segment the query range. Default = 11.
        q : bool, optional
            Quiet parameter (i.e. suppress status printouts), by default True
        """
        dataset = cls(
            site=PISite(site_name),
            original_item_list=attribute_paths,
            start_date=start_date,
            end_date=end_date,
            method=method,
            freq=freq,
            keep_tzinfo=keep_tzinfo,
            data_format=data_format,
            raise_not_found=raise_not_found,
        )
        dataset._run_query(n_segment=n_segment, q=q, **query_kwargs)
        return dataset

    def _create_af_object_list(self, raise_not_found: bool):
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

    def _create_data_object(self, raise_not_found: bool):
        """Returns data object using defined item_list
        -> when raise_not_found=False, invalid items are removed from self.item_list
        """
        af_obj_list = self._create_af_object_list(raise_not_found=raise_not_found)
        for item in self.invalid_items:
            self.item_list.remove(item)
        if self.item_type == "pipoint":
            return af_obj_list
        return af_obj_list.Data

    def get_af_time_range(self, start_date: str, end_date: str):
        """Returns AFTimeRange object for given dates with site- and user- specific timezone info"""
        aftimezone = AFTimeZone().CurrentAFTimeZone
        aftz_fmtprovider = AFTimeZoneFormatProvider(aftimezone)
        start = pd.Timestamp(start_date, tz=self.site.timezone)
        end = pd.Timestamp(end_date, tz=self.site.timezone)
        return AFTimeRange(str(start), str(end), aftz_fmtprovider)

    def get_date_range_list(self, n_segment):
        """Returns list of sub date ranges for query.

        Parameters
        ----------
        n_segment : int
            The number of days for each sub date range.
        """
        start, end = map(pd.Timestamp, [self.start_date, self.end_date])
        total_days = (end - start).days
        if total_days < n_segment:
            n_segment = total_days
        return segmented_date_ranges(start, end, n_days=n_segment)

    def _run_query(self, n_segment: int, q: bool = True, **query_kwargs):
        """Runs PI query for the specified date range and parameters"""
        qprint = quiet_print_function(q=q)

        # get AFSDK-specific query parameters
        afsdk_param_vals = DEFAULT_QUERY_PARAMETERS[self.method]
        afsdk_kwargs = {}
        for param, val in afsdk_param_vals.items():
            param_dict = AFSDK.get(param)
            param_val = val if not param_dict else param_dict.get(val)
            afsdk_kwargs[param] = param_val

        if self.method == "summaries":
            afsdk_kwargs["time_span"] = self.af_time_span

        # check for any default parameter overrides
        custom_kwargs = self._validate_query_kwargs(query_kwargs)
        for param, val in custom_kwargs.items():
            afsdk_kwargs.update({param: AFSDK[param][val]})

        # split date range & run query for sub ranges
        date_range_list = self.get_date_range_list(n_segment=n_segment)
        df_list = []
        for i, sub_range in enumerate(date_range_list):
            qprint(f"Querying range {i+1} of {len(date_range_list)}")
            time_range = self.get_af_time_range(*sub_range)
            df_ = self._query_pi_data(time_range, **afsdk_kwargs)
            df_ = self._format_pi_data(df_)
            df_list.append(df_)

        if self.data_format == "wide":
            df_list.insert(0, pd.DataFrame(columns=self.expected_columns()))  # ensures all cols
        df = pd.concat(df_list, axis=0, ignore_index=False)
        if self.method == "summaries":
            if not all(x in df.index for x in self.expected_index):  # validate index
                df = df.reindex(self.expected_index)

        df = df.rename_axis("Timestamp")
        self.data = df.copy()
        qprint(f"Done. {df.shape = }")

    def _format_pi_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Formats the data output from PI query.

        Parameters
        ----------
        dataframe : pd.DataFrame
            A dataframe consisting of values and timestamps from the .GetValueArrays method

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
        if self.data_format == "wide":
            start, end = df.index.min(), df.index.max()
            expected_index = pd.date_range(start, end, freq=self.freq.replace("m", "min"))
            if df.index.duplicated().any():
                df = df.loc[~df.index.duplicated(keep="first")]
            if df.shape[0] != expected_index.shape[0]:
                df = df.reindex(expected_index)
            # missing_cols = [c for c in self.expected_columns() if c not in df.columns]
            # if missing_cols:
            #     for col in missing_cols:
            #         df[col] = np.nan
            #     df = df[self.expected_columns()].copy()  # reorder
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

    @property
    def all_same_asset_group(self) -> bool:
        if self.item_type == "pipoint":
            return False
        all_asset_groups = [self._attpath_meta(p)["asset_group"] for p in self.item_list]
        return len(set(all_asset_groups)) == 1

    @property
    def all_same_attribute_levels(self) -> bool:  # not sure if needed
        if self.item_type == "pipoint":
            return False
        all_attribute_levels = [self._attpath_meta(p)["level"] for p in self.item_list]
        return len(set(all_attribute_levels)) == 1

    @property
    def max_attribute_path_level(self) -> int:
        if self.item_type == "pipoint":
            return 0
        return max([self._attpath_meta(p)["level"] for p in self.item_list])

    @property
    def columns_from_attribute_paths(self) -> list:
        if self.item_type == "pipoint":
            return []
        column_names = []
        for attpath in self.item_list:
            path_parts = self._attpath_meta(attpath)["parts"]
            group_index = path_parts.index(self.site.name) + 1  # asset group index
            if path_parts[-2] == self.site.name:
                group_index -= 1  # make site the group
            if self.all_same_asset_group:
                group_index += 1  # skip asset group in name
            col = "_".join(reversed(path_parts[group_index:]))
            column_names.append(col)
        return column_names

    @property
    def item_names(self) -> list:
        """Returns list equal in length to self.item_list"""
        if self.item_type == "pipoint":
            return self.item_list  # list of pipoint names
        return self.columns_from_attribute_paths  # list of columns derived from attribute paths

    @property
    def id_columns(self) -> list:
        """Returns a list of necessary ID columns for use with long-format data
        -> note: currently only supports attribute paths
        """
        if self.data_format == "wide":
            return []  # only long format
        if self.item_type == "pipoint":
            return []  # does not currently support inferred assets from pipoints
        n_elements_max = self.max_attribute_path_level
        id_columns = ["Group_ID", "Asset_ID", "Subasset_ID", "Subasset2_ID"]
        if n_elements_max <= 4:
            return id_columns[:n_elements_max]
        for i in range(n_elements_max - 4):
            id_columns.append(f"Subasset{i+3}_ID")
        return id_columns

    def _get_item_elements(self, item: str) -> list:
        """Returns list of values corresponding to ID columns for long format data
        -> note: currently only supports attribute paths
        """
        n_cols = len(self.id_columns)
        item_elements = self._attpath_meta(attpath=item)["elements"]
        if len(item_elements) < n_cols:
            item_elements += [np.nan] * (n_cols - len(item_elements))
        return item_elements

    def _check_for_bad_data(self, values, tstamps, flags):
        """Checks outputs of .GetValueArrays() function in query (returns True if data is bad)"""
        condition_1 = (len(list(values)) == 1) and (list(flags)[0].ToString() == "Bad")
        condition_2 = not any(isinstance(x, Number) for x in list(values))
        return any([condition_1, condition_2])

    @with_retries(n_max=3)
    def _query_pi_data(self, time_range: AFTimeRange, **kwargs) -> pd.DataFrame:
        """queries data using the defined method from AFSDK"""
        if self.method == "recorded_values":
            df = self._query_pi_recorded_values(time_range, **kwargs)
        else:
            df = self._query_pi_summaries(time_range, **kwargs)
        return df

    def _query_pi_recorded_values(
        self,
        time_range: AFTimeRange,
        boundary_type: AFBoundaryType,
        filter_expression: str,
        include_filtered_values: bool,
        paging_configuration: PIPagingConfiguration,
        max_count: int,
    ) -> pd.DataFrame:
        """queries data using the .RecordedValues() method from AFSDK on the given data object"""
        # note: should always return long format data, as datetime index will differ for each tag
        recorded_values = self.af_data_object.RecordedValues(
            time_range,
            boundary_type,
            filter_expression,
            include_filtered_values,
            paging_configuration,
            max_count,
        )
        df = self._extract_data_from_query_output(query_output=recorded_values)
        return df

    def _query_pi_summaries(
        self,
        time_range: AFTimeRange,
        time_span: AFTimeSpan,
        summary_type: AFSummaryTypes,
        calculation_basis: AFCalculationBasis,
        timestamp_calculation: AFTimestampCalculation,
        paging_configuration: PIPagingConfiguration,
    ) -> pd.DataFrame:
        """queries data using the .Summaries() method from AFSDK on the given data object"""
        summaries = self.af_data_object.Summaries(
            time_range,
            time_span,
            summary_type,
            calculation_basis,
            timestamp_calculation,
            paging_configuration,
        )
        df = self._extract_data_from_query_output(query_output=summaries)
        return df

    def _extract_data_from_query_output(self, query_output):
        """Iterates over query output object and gets data from AFValues."""
        df_list = []
        for output, item_name, item in zip(query_output, self.item_names, self.item_list):
            # note: type(output) = IDictionary if method="summaries", else AFValues
            df_ = self._retrieve_data(output, item_name, item)
            df_list.append(df_)
        if not df_list:
            raise ValueError("Error retrieving data.")
        # return self._concatenate_data(df_list)
        return pd.concat(df_list, axis=1 if self.data_format == "wide" else 0)

    def _retrieve_data(self, output, item_name: str, item: str):
        """Iterates over the AFValues collection and returns a dataframe with the associated data

        note: the .Summaries() query method returns an object of the following type:
              System.Collections.Generic.IEnumerable[IDictionary[AFSummaryTypes,AFValues]]

        Parameters
        ----------
        query_output : System.Collections.Generic.IDictionary[AFSummaryTypes,AFValues]
            A dictionary-like object with query summary types as keys and data as values.
        item_name : str
            The name to use for the data column (if wide format) or asset ID (if long format)
        item : str
            The attribute path associated with the item name; only applies for "long" data

        Returns
        -------
        pd.DataFrame
            A dataframe with a datetime index and single column of data.
            note: dtype is usually float but not guaranteed.
            If all data is bad, returns empty dataframe.
        """
        if isinstance(output, AFValues):
            # RecordedValues method -> query_output = IEnumerable<AFValues>
            af_values = output
        else:
            # Summaries method -> query_output = IEnumerable<IDictionary<AFSummaryTypes, AFValues>>
            af_values = [vals for vals in output.Values].pop()
        values, tstamps, flags = af_values.GetValueArrays()
        if self._check_for_bad_data(values, tstamps, flags):
            return pd.DataFrame()

        if self.data_format == "wide":
            # item_name is either pipoint, or column name derived from attribute path
            if self.item_type == "pipoint":
                item_name_ = af_values.PIPoint.Name
            else:
                item_name_ = af_values.Attribute.Name
                element_ = af_values.Attribute.Element
                limit_ = (
                    self._attpath_meta(attpath=item)["asset_group"]
                    if self.all_same_asset_group
                    else self.site.name
                )
                while element_.Name != limit_:
                    item_name_ += f"_{element_.Name}"
                    element_ = element_.Parent
            return pd.DataFrame({item_name_: list(values)}, index=list(tstamps))

        # long format data
        df_ = pd.DataFrame({"Value": list(values)}, index=list(tstamps))
        if self.item_type == "pipoint":
            df_["PIPoint"] = item_name
            return df_

        # long format for attribute paths
        df_["Attribute"] = item_name
        for element, id_col in zip(self._get_item_elements(item), self.id_columns):
            df_[id_col] = element
        return df_

    def _concatenate_data(self, df_list: list):
        if self.data_format == "wide":
            df = pd.concat(df_list, axis=1)
            # confirm all columns exist in output
            # for i, item_name in enumerate(self.item_names):
            #     if item_name not in df.columns:
            #         df.insert(i, item_name, np.nan)
        else:  # long format  TODO
            df = pd.concat(df_list, axis=0)
        return df

    def _validate_query_kwargs(self, kwargs):
        """Returns dictionary of keys/values that exist in AFSDK namespace."""
        valid_kwargs = {}
        for key, val in kwargs.items():
            # normalize to lower case
            key = key.lower()

            # case 1: max_count for recorded values query (only parameter with non-string value)
            if key == "max_count":
                if isinstance(val, int):
                    if val > 0:
                        valid_kwargs.update({key: val})
                        continue
                continue

            # confirm value is a string
            if not isinstance(val, str):
                continue

            # normalize to lower case
            val = val.lower()

            # confirm key is valid
            if key not in DEFAULT_QUERY_PARAMETERS[self.method]:
                continue

            # verify value has a defined AFSDK object
            if val not in AFSDK[key]:
                continue

            valid_kwargs.update({key: val})

        return valid_kwargs
