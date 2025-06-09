from itertools import chain
import pandas as pd
from pathlib import Path

from . import oemeta, oepaths
from .config import PI_DATABASE_PATH
from ..datatools.meteoqc import run_meteo_backfill
from ..datatools.oepvlib import run_pvlib_model
from ..datatools.qcutils import run_auto_qc
from ..reporting.flashreports import generate_monthlyFlashReport
from ..reporting.functions import (
    get_kpis_from_flashreport,
    get_kpis_from_tracker,
    get_solar_budget_values,
)
from ..reporting.tools import get_solar_budget_values
from ..reporting.query_attributes import monthly_query_attribute_paths


class PISite:
    """Base class for sites existing in the PI database AF structure; contains
    common attributes and metadata.

    Attributes
    ----------
    name : str
        Name of the site
    fleet : str
        Name of the associated fleet
    fleet_key : str
        Corresponding key to access fleet metadata in oemeta.data reference dict
    """

    def __init__(self, name: str):
        fleet_list = ["gas", "solar", "wind"]
        fleet_keys = [f"AF_{fleet.capitalize()}_V3" for fleet in fleet_list]
        matching_fleet = None
        matching_key = None
        for fleet_, key_ in zip(fleet_list, fleet_keys):
            if name in oemeta.data[key_]:
                matching_fleet = fleet_
                matching_key = key_

        if matching_fleet is None:
            raise ValueError(f"Site = {name} not found.")

        self.name = name
        self.fleet = matching_fleet
        self.fleet_key = matching_key

    @property
    def af_dict(self) -> dict:
        """Returns dictionary of PI AF structure/metadata"""
        return oemeta.data[self.fleet_key].get(self.name)

    @property
    def asset_groups(self) -> list:
        """Returns the top level elements from the PI AF structure
        all groups: ["Inverters", "Met Stations", "Meter", "PPC", "SubSt", "Trackers"]
        """
        return list(self.af_dict.keys())

    @property
    def asset_names_by_group(self) -> dict:
        """Returns dictionary with asset groups as keys, and asset names as values"""
        return {
            group: list(self.af_dict[group][f"{group}_Assets"].keys())
            for group in self.asset_groups
        }

    @property
    def timezone(self) -> str:
        """Returns timezone (if defined)"""
        return oemeta.data["TZ"].get(self.name)

    @property
    def coordinates(self) -> tuple:
        """Returns latitude and longitude"""
        return oemeta.data["LatLong"].get(self.name)

    @property
    def altitude(self):
        """Returns altitude"""
        return oemeta.data["Altitude"].get(self.name)

    @property
    def af_path(self) -> str:
        """Returns path for site-level element in the PI AF structure"""
        if self.fleet == "gas":
            path_parts = ["Gas Fleet"]
        else:
            path_parts = ["Renewable Fleet", f"{self.fleet.capitalize()} Assets"]
        return "\\".join([PI_DATABASE_PATH, *path_parts, self.name])

    def subassets_by_asset(self, asset_group: str):
        """Returns a list or dictionary with asset names as keys and list of subasset names as values"""
        if asset_group not in self.asset_groups:
            return {}
        asset_dict = self.af_dict[asset_group][f"{asset_group}_Assets"]
        if not asset_dict:
            return []
        asset0 = [*asset_dict][0]
        if not asset_dict[asset0].get(f"{asset0}_Subassets"):
            return list(asset_dict.keys())

        asset_hierarchy = {}
        for asset, dict_ in asset_dict.items():
            subasset_dict = dict_.get(f"{asset}_Subassets")
            subasset0 = [*subasset_dict][0]
            if not subasset_dict[subasset0].get(f"{subasset0}_Subassets"):
                asset_hierarchy[asset] = list(subasset_dict.keys())
                continue

            asset_hierarchy[asset] = {}
            for subasset, dict2_ in subasset_dict.items():
                subasset_dict2 = dict2_.get(f"{subasset}_Subassets")
                asset_hierarchy[asset][subasset] = list(subasset_dict2.keys())

        return asset_hierarchy

    def pi_attributes(self, asset_group: str, asset_name: str = "", subasset_name: str = ""):
        """Returns list of attribute names for a given PI element"""
        if asset_group not in self.asset_groups:
            return []

        # check if specified inputs are valid
        element = asset_group  # init
        if asset_name:
            if asset_name not in self.asset_names_by_group[asset_group]:
                return []
            element = asset_name
            if subasset_name:
                if subasset_name not in self.subassets_by_asset(asset_group).get(asset_name):
                    return []
                element = subasset_name

        # get reference dict from oemeta.data
        element_dict = self.af_dict[asset_group].copy()
        if asset_name:
            element_dict = element_dict[f"{asset_group}_Assets"].get(asset_name)
            if subasset_name:
                element_dict = element_dict[f"{asset_name}_Subassets"].get(subasset_name)

        return element_dict.get(f"{element}_Attributes")

    def flashreport_folder(self, year: int, month: int) -> Path:
        """Returns filepath to site flashreport folder for given year/month"""
        return oepaths.frpath(year=year, month=month, ext=self.fleet, site=self.name)


class SolarSite(PISite):
    """Class for solar sites existing in the PI AF structure. Inherits from PISite."""

    def __init__(self, name: str):
        super().__init__(name)
        if len(self.af_dict) == 0:
            raise ValueError(f"site = {name} has not been integrated into PI")

    @property
    def design_database(self) -> dict:
        """Returns design database parameters for site equipment"""
        return oemeta.data["Project_Design_Database"].get(self.name)

    @property
    def query_attributes(self) -> dict:
        """Returns dictionary with asset groups as keys and list of attribute paths as values"""
        try:
            attribute_dict = monthly_query_attribute_paths(self.name)
        except:
            attribute_dict = {}
        return attribute_dict

    @property
    def existing_report_periods(self):
        """Returns list of (year, month) tuples for which a site flashreport folder exists"""
        is_yyyymm = lambda fp: len(fp.name) == 6 and all(x.isdigit() for x in fp.name)
        get_year_month = lambda fp: (int(fp.name[:4]), int(fp.name[-2:]))

        year_month_folders = list(filter(is_yyyymm, oepaths.flashreports.iterdir()))
        year_month_tuples = list(sorted(map(get_year_month, year_month_folders)))
        return [(y, m) for y, m in year_month_tuples if self.flashreport_folder(y, m).exists()]

    @property
    def data_files_by_period(self):
        """Returns a dictionary with strings 'YYYYMM' as keys and list of available PI query
        filepaths (for that reporting period) as values (ascending order)
        """
        return {
            f"{year}{month:02d}": self.get_flashreport_files(year, month, types=["query", "pvlib"])
            for year, month in self.existing_report_periods
            if len(self.get_flashreport_files(year, month, types=["query", "pvlib"])) > 0
        }

    @property
    def existing_report_periods_with_data(self):
        """Returns list of (year, month) tuples for periods where site has query data files"""
        return [(int(key_[:4]), int(key_[-2:])) for key_ in self.data_files_by_period.keys()]

    @property
    def pvlib_files_by_period(self):
        return {
            f"{year}{month:02d}": self.get_flashreport_files(year, month, types=["pvlib"])
            for year, month in self.existing_report_periods
            if len(self.get_flashreport_files(year, month, types=["pvlib"])) > 0
        }

    @property
    def existing_report_periods_with_pvlib_data(self):
        """Returns list of (year, month) tuples for periods where site has query data files"""
        return [(int(key_[:4]), int(key_[-2:])) for key_ in self.pvlib_files_by_period.keys()]

    def existing_periods_with_data_files(self, key: str) -> list:
        valid_keys = list(map(str.lower, self.asset_groups)) + ["pvlib"]
        if key not in valid_keys:
            raise KeyError("Invalid key; must be one of the site asset groups, or 'pvlib'")
        key = key.replace(" ", "").lower()
        fpaths_by_period = {
            ym_str: [fp for fp in fpath_list if key in fp.name.lower()]
            for ym_str, fpath_list in self.data_files_by_period.items()
            if any(key in fp.name.lower() for fp in fpath_list)
        }
        if not fpaths_by_period:
            return []
        get_year_month = lambda ym_str: (int(ym_str[:4]), int(ym_str[-2:]))
        return list(map(get_year_month, fpaths_by_period.keys()))

    # instance method
    def get_flashreport_files(self, year: int, month: int, types=[]) -> dict:
        """Returns a dictionary with categories as keys and list of filepaths as values, sorted
        by date created (newest first)

        If types specified, returns single list of filepaths corresponding to keys in types
        """
        dir = self.flashreport_folder(year, month)
        get_files = lambda str_: oepaths.sorted_filepaths(list(dir.glob(str_)))
        glob_str_dict = {
            "query": "PIQuery_*.csv",
            "pvlib": f"PVLib_InvAC_{self.name}*.csv",
            "report": "*FlashReport*.xlsx",
            "plots": "*.html",
        }
        output_dict = {key_: get_files(str_) for key_, str_ in glob_str_dict.items()}
        captured_files = list(chain.from_iterable(output_dict.values()))
        other_files = [fp for fp in dir.iterdir() if fp.is_file() and fp not in captured_files]
        if other_files:
            output_dict.update({"other": oepaths.sorted_filepaths(other_files)})
        if not types:
            return output_dict
        file_list = []
        for key in types:
            files = output_dict.get(key)
            if files:
                file_list.extend(files)
        return file_list

    # instance method
    def get_flashreport_status(self, year: int, month: int):
        """Returns a dictionary with flashreport steps/requirements as keys and bool values"""
        # get flashreport files
        site_files = self.get_flashreport_files(year, month)
        relevant = lambda target_groups: [g for g in target_groups if g in self.query_attributes]
        files_exist = lambda key, fp_list: any(key.replace(" ", "") in fp.name for fp in fp_list)

        # part 1: pi query requirements (raw data files)
        query_groups = ["Inverters", "Met Stations", "Meter", "PPC"]
        query_status_dict = {
            group: files_exist(group, site_files["query"]) for group in relevant(query_groups)
        }

        # part 2: qc requirements (cleaned data files)
        qc_groups = ["Inverters", "Met Stations", "Meter"]
        cleaned_files = list(filter(lambda fp: "CLEANED" in fp.name, site_files["query"]))
        qc_status_dict = {group: files_exist(group, cleaned_files) for group in relevant(qc_groups)}

        # part 3: backfill requirements (processed data files)
        backfill_groups = ["Met Stations"]
        backfill_files = list(filter(lambda fp: "PROCESSED" in fp.name, cleaned_files))
        backfill_status_dict = {
            group: files_exist(group, backfill_files) for group in relevant(backfill_groups)
        }

        # part 4: pvlib requirements
        pvlib_status = len(site_files["pvlib"]) > 0

        # part 5: flashreport files
        report_status = len(site_files["report"]) > 0

        output_status_dict = {
            "query": query_status_dict,
            "qc": qc_status_dict,
            "backfill": backfill_status_dict,
            "pvlib": pvlib_status,
            "report": report_status,
        }

        if self.name == "Comanche":
            curtailment_report_status = files_exist("Curtailment_Report", site_files["other"])
            output_status_dict.update({"curtailment_report": curtailment_report_status})

        return output_status_dict

    def get_data_filepaths(self, year: int, month: int, asset_group: str):
        """Returns list of Path objects for PIQuery files in flashreport folder"""
        query_fp_list = self.get_flashreport_files(year, month)["query"]
        if not query_fp_list:
            return []
        return [fp for fp in query_fp_list if asset_group in fp.name]

    # functions to get kpis and summary metrics
    def get_monthly_kpis(self, year: int, month: int, source: str):
        """Returns dataframe of kpis loaded from specified source file.

        Parameters
        ----------
        year : int
            The reporting year
        month : int
            The reporting month
        source : str
            The file from which to load kpis; either "flashreport" or "tracker"

        Returns
        -------
        pandas Dataframe
        """
        if source not in ["flashreport", "tracker"]:
            raise ValueError("Invalid source. Must be either 'flashreport' or 'tracker'.")

        if source == "tracker":
            df = get_kpis_from_tracker(sitelist=[self.name], yearmonth_list=[(year, month)])
        else:
            df = get_kpis_from_flashreport(site=self.name, year=year, month=month)
        return df

    def run_flashreport_qc(
        self, year: int, month: int, asset_group: str, save: bool = False, **kwargs
    ):
        """Runs auto qc script on raw PI query files (w/ optional save to flashreport folder)"""
        # check for query files
        query_status = self.get_flashreport_status(year, month)["query"]
        if not query_status.get(asset_group):
            raise ValueError(f"no query data files found for {asset_group = }")

        query_filepaths = self.get_data_filepaths(year, month, asset_group)
        is_raw_fp = lambda fp: not any(x in fp.name.casefold() for x in ["cleaned", "processed"])
        raw_filepath = oepaths.latest_file(list(filter(is_raw_fp, query_filepaths)))
        df_raw = pd.read_csv(raw_filepath, index_col=0, parse_dates=True)
        df_clean = run_auto_qc(df_raw=df_raw, site=self.name, **kwargs)

        if save:
            new_filestem = raw_filepath.stem + "_CLEANED"
            savepath = raw_filepath.with_stem(new_filestem)
            df_clean.to_csv(oepaths.validated_savepath(savepath))

        return df_clean

    def run_flashreport_backfill(
        self, year: int, month: int, asset_group: str, save: bool = False, **kwargs
    ):
        """Runs backfill script on 'cleaned' PI query files (w/ optional save to report folder)"""
        # check qc status
        qc_status = self.get_flashreport_status(year, month)["qc"]
        if not qc_status.get(asset_group):
            raise ValueError(f"no cleaned/qc'd data files found for {asset_group = }")
        if asset_group not in ["Met Stations"]:
            raise ValueError(f"{asset_group = } not currently supported for backfill")

        output = run_meteo_backfill(self.name, year, month, savefile=save, **kwargs)
        df = output[0] if kwargs.get("return_df_and_fpath") else output
        return df

    def run_flashreport_pvlib(self, year: int, month: int, save: bool = False, **kwargs):
        """Runs pvlib script using 'processed' met station PI query file"""
        backfill_status = self.get_flashreport_status(year, month)["backfill"]
        if not backfill_status.get("Met Stations"):
            raise ValueError("Processed meteo file not found")

        output = run_pvlib_model(
            sitename=self.name,
            from_FRfiles=True,
            FR_yearmonth=f"{year}{month:02d}",
            save_files=save,
            **kwargs,
        )
        df = output[0] if kwargs.get("return_df_and_fpath") else output
        return df

    def generate_flashreport(self, year: int, month: int, **kwargs):
        """Runs flashreport generation script for selected year/month
        >> TODO: implement caching of meter historian (for dev, and for dash apps)
        """
        status_dict = self.get_flashreport_status(year, month)
        if any(status is False for status in status_dict["qc"].values()):
            raise ValueError("Must complete data QC before generating report")
        if not status_dict.get("pvlib"):
            raise ValueError("Must create PVLib file before generating report")

        output = generate_monthlyFlashReport(
            sitename=self.name,
            year=year,
            month=month,
            **kwargs,
        )
        if kwargs.get("return_df_and_fpath") or kwargs.get("return_df_and_log_info"):
            return output[0]
        return

    def get_budget_values(self, year: int, month: int):
        """Returns dictionary with budgeted POA, Production, and Curtailment from KPI tracker"""
        budget_vals = get_solar_budget_values(year, month).set_index("Project").loc[self.name]
        matching_column = lambda key: [c for c in budget_vals.index if key in c.casefold()].pop()
        return {
            key: budget_vals.loc[matching_column(key)]
            for key in ["poa", "production", "curtailment"]
        }

    # def calculate_monthly_variance_kpis(self, year: int, month: int, meter_total: float = None):
    #     """Calculates monthly variance KPIs for generation, resource, and availability.

    #     >> data requirements / sources
    #         - total generation (from historian, or PI meter file) <- use meter_total arg to bypass
    #         - budget generation (from kpi tracker file)
    #         - insolation (from PROCESSED met station file, or DTN transposition)
    #         - curtailment (Comanche only - from curtailment report file)

    #     Parameters
    #     ----------
    #     year : int
    #     month : int
    #     meter_total : float, optional
    #         When provided, skips loading of historian file or PI data (for use in loop)

    #     Returns
    #     -------
    #     A dictionary with the following format:
    #         variance_dict = {
    #             "generation": {"value": variance_val, "%": variance_pct},
    #             "resource": {"value": variance_val, "%": variance_pct},
    #             "availability": {"%": variance_pct},
    #         }
    #     """
    #     report_status = self.get_flashreport_status(year, month)

    #     start = pd.Timestamp(year, month, 1)
    #     end = start + pd.DateOffset(months=1)
    #     start_date, end_date = map(lambda t: t.strftime("%Y-%m-%d"), [start, end])

    #     # get meter generation
    #     if meter_total is None:
    #         df_hist = load_meter_historian(year=year, month=month)
    #         if self.name in df_hist.columns:
    #             meter_total = df_hist[self.name].sum()
    #         elif report_status["qc"].get("Meter"):
    #             kwargs = dict(asset_group="Meter", start_date=start_date, end_date=end_date)
    #             dataset = SolarDataset.from_existing_query_files(self.name, **kwargs)
    #             meter_total = dataset.data.iloc[:, 0].sum() / 60  # minute-level data
    #         else:
    #             meter_total = 0.0

    #     # get budget generation
    #     budget_dict = self.get_budget_values(year, month)

    #     # get insolation
    #     if report_status["backfill"].get("Met Stations"):
    #         kwargs2 = dict(asset_group="Met Stations", start_date=start_date, end_date=end_date)
    #         dataset = SolarDataset.from_existing_query_files(self.name, **kwargs2)
    #         poacol = "POA" if "POA" in dataset.data.columns else "POA_DTN"
    #         insolation_total = dataset.data[poacol].sum() / 60  # minute-level data
    #     else:
    #         # TODO: add function to get transposed poa from dtn
    #         pass
    #     return
