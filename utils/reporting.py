import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from typing import Union

from .assets import SolarSite
from .datetime import (
    infer_freq_timedelta,
    localize_naive_datetimeindex,
    remove_tzinfo_and_standardize_index,
)
from .helpers import quiet_print_function
from .oemeta import PI_SITES_BY_FLEET
from .oepaths import date_created, latest_file, sorted_filepaths, validated_savepath
from .solar import SolarDataset
from ..dataquery.external import query_fracsun, get_fracsun_sites
from ..datatools.backfill import process_and_backfill_meteo_data, meteo_backfill_subplots
from ..datatools.pvlib import run_flashreport_pvlib_model, run_pvlib_model
from ..datatools.qcutils import run_auto_qc, qc_compare_fig
from ..datatools.meterhistorian import add_pi_data_files_to_server_folder, load_meter_historian
from ..reporting.curtailment import (
    generate_curtailment_report,
    load_curtailment_report_data,
    curtailment_summary_table,
    sql_curtailment_summary,
)
from ..reporting.flashreports import generate_monthlyFlashReport
from ..reporting.insurance_bi import (
    calculate_insurance_bi_adjustment,
    SOLAR_PERFORMANCE_REPORT_FILEPATH,
)
from ..reporting.kpis import (
    blank_kpi_dataframe,
    format_kpi_data_for_waterfall,
    reconcile_losses,
    validate_kpi_totals,
    ORDERED_KPI_COLUMNS,
    REPORT_KPI_COLUMN_MAPPING,
    KPI_COLUMNS_FOR_PLOT,
)
from ..reporting.solarplots import monthly_summary_subplots

SOLAR_SITES = PI_SITES_BY_FLEET["solar"]


def most_recent_reporting_period() -> tuple[int]:
    today = pd.Timestamp(dt.datetime.now().date())
    last_month = today - pd.DateOffset(months=1)
    return (last_month.year, last_month.month)


def validate_reporting_period(year: int, month: int) -> None:
    """Ensures defined year/month is before the current year/month."""
    max_year, max_month = most_recent_reporting_period()
    if (year, month) > (max_year, max_month):
        raise ValueError("Invalid year/month specified.")
    return


def get_start_and_end_dates(year: int, month: int) -> tuple[pd.Timestamp]:
    start = pd.Timestamp(year, month, 1)
    end = start + pd.DateOffset(months=1)
    return (start, end)


def get_filenames_and_dates(fpath_list: list[Path]) -> str:
    """Returns string for use in class __str__ method."""
    lines = []
    if len(fpath_list) == 0:
        return ""
    for fpath in sorted_filepaths(fpath_list):
        lines.append(f"    {date_created(fpath)}    {fpath.name}")
    lines.append("")
    return "\n".join(lines)


def _get_performance_breakdown(
    dfkpi: pd.DataFrame, calculated_curtailment: float = 0.0, validated: bool = True
) -> dict:
    """Gets totals for waterfall plot; Ensures all buckets sum to zero."""

    def format_kpi_col(col):
        return (
            col.lower().split(" (")[0].replace(" - ", "_").replace(" ", "_").replace("/system", "")
        )

    df = dfkpi.T.copy()
    df.columns = ["value"]
    df.index = df.index.map(format_kpi_col)
    val = lambda kpi: df.at[kpi, "value"]

    # subtract ac module loss & soiling loss from dc health loss
    dc_health_loss = val("dc_health_loss") - val("soiling_loss") - val("ac_module_loss")

    # overwrite curtailment with calculated value (if from other source, e.g. CAISO)
    if calculated_curtailment > 0:
        df.loc["curtailment_total"] = calculated_curtailment

    # calculate estimated ac line losses
    ac_line_losses = val("inverter_generation") - val("meter_generation")

    # ensure insurance bi is not NaN
    insurance_adj = val("insurance_bi_adjustment")
    if pd.isna(insurance_adj):
        insurance_adj = 0

    data = {
        "generation": {
            "possible": val("possible_generation"),
            "inverter": val("inverter_generation"),
            "meter": val("meter_generation"),
        },
        "losses": {
            "soiling_loss": val("soiling_loss"),
            "dc_health_loss": dc_health_loss,
            "ac_module_loss": val("ac_module_loss"),
            "downtime_loss": val("downtime_loss"),
            "curtailment": val("curtailment_total"),
            "ac_line_losses": ac_line_losses,
        },
        "adjustments": {
            "insurance_bi": insurance_adj,
        },
    }
    if validated is False:
        return data

    # validate totals
    possible = data["generation"]["possible"]
    total_losses = sum(data["losses"].values())
    total_adjustments = sum(data["adjustments"].values())
    net = possible - total_losses + total_adjustments
    actual = data["generation"]["meter"]
    assert actual == net

    return data


class FlashReportGenerator:
    """A class containing methods for the entire solar reporting process.

    Parameters
    ----------
    site : str
        The name of a solar site.
    year : int
        The year of the reporting period.
    month : int
        The month of the reporting period.

    Attributes
    ----------
    site : oetoolbox.utils.assets.SolarSite
        An instance of a SolarSite, providing relevant attributes/methods for obtaining data.
    """

    def __init__(self, site: str, year: int, month: int, df_util: pd.DataFrame = None):

        validate_reporting_period(year, month)
        self.site = SolarSite(site)
        self.year = year
        self.month = month

        # get start/end dates as timestamps
        start, end = get_start_and_end_dates(year, month)
        self.start = start
        self.end = end

        # get start/end dates as strings
        start_date, end_date = map(lambda x: x.strftime("%Y-%m-%d"), [start, end])
        self.start_date = start_date
        self.end_date = end_date

        # assign attribute for flashreport folder
        flashreport_folder = self.site.flashreport_folder(year, month)
        if not flashreport_folder.exists():
            flashreport_folder.mkdir(parents=True)
        self.folder = flashreport_folder

        # print warning if utility meter data not provided
        if df_util is None:
            print(
                "Note: recommended to provide df_util if using in loop to for multiple sites.\n"
                "If not provided, meter historian will be loaded if any of the following methods\n"
                'are called: "get_report_status", "create_monthly_report", "create_monthly_subplots".'
            )
        elif not isinstance(df_util, pd.DataFrame):
            if df_util != "ignore_warning":
                raise ValueError("Argument 'df_util' must be a DataFrame.")
            df_util = None

        self.df_util = df_util

    def __str__(self):
        """Prints all reporting files (with date created) that exist in FlashReport folder."""
        fpath_dict = self.flashreport_fpaths
        if not any(fpath_dict.values()):
            return "No files exist for specified reporting period."

        lines = []
        for file_type in ["report", "pvlib"]:
            lines.append(f"{file_type.upper()} FILES:")
            fpath_list = fpath_dict.get(file_type, [])
            lines.append(get_filenames_and_dates(fpath_list))

        lines.append("PI DATA FILES:")
        query_fpaths = fpath_dict.get("query", [])
        group_from_fpath = lambda fp: fp.name.split("_")[1]
        for asset_group in list(sorted(set(map(group_from_fpath, query_fpaths)))):
            matching_fpath = lambda fp: group_from_fpath(fp) == asset_group
            fpath_list = list(filter(matching_fpath, query_fpaths))
            lines.append(get_filenames_and_dates(fpath_list))

        for file_type in ["plots", "other"]:
            ftype = dict(plots="PLOT", other="UNCATEGORIZED")[file_type]
            fpath_list = fpath_dict.get(file_type, [])
            if len(fpath_list) > 0:
                lines.append(f"{ftype} FILES:")
                lines.append(get_filenames_and_dates(fpath_list))

        return "\n".join(lines)

    @property
    def args(self) -> list:
        return [self.site.name, self.year, self.month]

    @property
    def query_attributes(self) -> dict[str, list]:
        """Dictionary of reporting attribute paths for relevant groups using standard OE. attributes."""
        return self.site.get_reporting_query_attributes(validated=False)

    @property
    def query_groups(self) -> list[str]:
        """Returns list of asset groups with defined monthly query attributes."""
        return list(self.query_attributes.keys())

    @property
    def flashreport_fpaths(self) -> dict[str, list]:
        """Returns dictionary with keys: ['query', 'pvlib', 'report', 'plots'] (+ ['other'] if any)"""
        return self.site.get_flashreport_files(self.year, self.month)

    @property
    def query_filepaths_by_group(self) -> dict[str, list]:
        fpath_dict = {}
        for group in self.query_groups:
            grp = group.replace(" ", "")
            fpath_dict[group] = [fp for fp in self.flashreport_fpaths["query"] if grp in fp.name]
        return fpath_dict

    @property
    def existing_filepaths(self) -> dict[str, list]:
        """Returns dict with same keys as flashreport_fpaths, but switches out 'query' for individual asset groups."""
        fpath_dict = self.flashreport_fpaths.copy()
        del fpath_dict["query"]
        return {**self.query_filepaths_by_group, **fpath_dict}

    @property
    def latest_filepaths(self) -> dict[str, list[Path]]:
        fpath_dict = {}
        for key, fpaths in self.existing_filepaths.items():
            if key == "other" or len(fpaths) == 0:
                continue
            fpath_dict.update({key: latest_file(fpaths)})
        return fpath_dict

    def _get_filepaths_for_quick_load(self):
        quick_load_keys = ["Inverters", "Met Stations", "Meter", "pvlib"]
        output = {}
        for key in quick_load_keys:
            fpath = self.latest_filepaths.get(key, None)
            if fpath is not None:
                output[key] = fpath
        return output

    def quick_load_latest_data(
        self, with_tz=False, include_fpaths=False
    ) -> dict[str, pd.DataFrame]:
        """if include fpaths, adds key 'source_files' to output dict with list[Path] as value"""
        data, source_fpaths = {}, []
        for asset_group in ["Inverters", "Met Stations", "Meter", "Modules"]:
            df, fpath = self.load_query_file(
                asset_group=asset_group, version="all", with_tz=with_tz, return_fpath=True
            )
            if not df.empty:
                data[asset_group] = df.copy()
                source_fpaths.append(fpath)

        pvl_fpath = self.latest_filepaths.get("pvlib", None)
        if pvl_fpath is not None:
            data["pvlib"] = self.site._load_file(filepath=pvl_fpath, with_tz=with_tz)
            source_fpaths.append(pvl_fpath)

        if len(source_fpaths) > 0 and include_fpaths:
            data["source_files"] = source_fpaths

        return data

    def load_fracsun_soiling(self, with_tz=False, q: bool = True):
        if self.site.name not in get_fracsun_sites():
            return
        df = query_fracsun(
            site=self.site.name, start_date=self.start_date, end_date=self.end_date, q=q
        )
        df = df[["soiling"]].resample("1h").ffill()
        df["soilingPercent"] = df["soiling"].div(100)
        df = df[["soilingPercent"]]

        expected_index = pd.date_range(self.start_date, self.end_date, freq="h", inclusive="left")
        df = df.reindex(expected_index)
        if with_tz is True:
            df = localize_naive_datetimeindex(df, site=self.site.name, q=q)
        if q is False:
            print("Loaded Fracsun soiling data.")
        return df

    def load_data(self, with_tz: bool = False, return_fpaths: bool = False):
        data_dict = self.quick_load_latest_data(with_tz=with_tz, include_fpaths=True)
        output_data = {
            key.lower().replace(" ", "_"): df
            for key, df in data_dict.items()
            if key != "source_files"
        }

        # get site-level generation interval data
        site_col_dict = {
            "Inverters": "Inv_Total_MW",
            "Meter": "PI_Meter_MW",
            "pvlib": "Possible_MW",
        }
        site_level_data = {}
        for key, col in site_col_dict.items():
            if key not in data_dict:
                continue
            if key == "Meter":
                df = data_dict[key].iloc[:, [0]].copy()
            elif key in ("Inverters", "pvlib"):
                matching_str = "ActivePower" if key == "Inverters" else "Possible_Power"
                df = data_dict[key].filter(like=matching_str).sum(axis=1).div(1e3).to_frame()
            else:
                continue
            df.columns = [col]
            site_level_data[key.lower()] = df.copy()

        # get site-level losses interval data
        if all(key in data_dict for key in ["Modules", "pvlib"]):
            dfm = data_dict["Modules"]
            if dfm.index.diff().max() != pd.Timedelta(hours=1):
                dfm = dfm.resample("h").mean()
            mcol, mloss_col = "OE.ModulesOffline_Percent", "AC_Module_Loss_MW"
            if mcol in dfm.columns:
                dfp = site_level_data["pvlib"]
                if dfp.index.diff().max() != pd.Timedelta(hours=1):
                    dfp = dfp.resample("h").mean()
                df_mloss = dfm[mcol].mul(dfp["Possible_MW"]).to_frame(name=mloss_col)
                site_level_data["ac_module_loss"] = df_mloss

        df_soil = self.load_fracsun_soiling(with_tz=with_tz)
        if df_soil is not None:
            site_level_data["soiling_loss"] = df_soil

        if self.df_util is not None:
            util_data = self.df_util[[self.site.name]].copy()
            site_level_data["utility"] = util_data.rename(columns={self.site.name: "Util_Meter_MW"})

        output_data["site_level"] = site_level_data

        if return_fpaths is True:
            return output_data, data_dict.get("source_files", [])
        return output_data

    def data_filepaths_by_version(self, asset_group: str, version="all") -> Union[dict, list]:
        """Retrieves PI query filepaths from flashreport folder.

        Parameters
        ----------
        year : int
            Year of the reporting period.
        month : int
            Month of the reporting period.
        version : str, optional
            Specific file type to return, by default "all"
            options = [all, raw, cleaned, processed]

        Returns
        -------
        dict if version == "all" else list
            Returns dictionary with versions as keys (raw, cleaned, processed)
            and lists of corresponding filepaths.
            If a single version is specified, returns related list of filepaths.
        """
        if version not in ("all", "raw", "cleaned", "processed"):
            raise ValueError("Invalid version specified.")
        fpaths = self.query_filepaths_by_group.get(asset_group, [])
        processed_fpaths = list(filter(lambda fp: "PROCESSED" in fp.name, fpaths))
        cleaned_fpaths = [f for f in fpaths if "CLEANED" in f.name and f not in processed_fpaths]
        raw_fpaths = [fp for fp in fpaths if fp not in cleaned_fpaths + processed_fpaths]
        fpath_dict = {"raw": raw_fpaths, "cleaned": cleaned_fpaths}
        if asset_group == "Met Stations":
            fpath_dict.update({"processed": processed_fpaths})
        if version != "all":
            return fpath_dict.get(version, [])
        return fpath_dict

    def get_data_filepath(self, asset_group: str, version="all") -> Union[Path, None]:
        """Returns most recent file for given asset_group / version"""
        fpaths = self.data_filepaths_by_version(asset_group, version)
        if version == "all":
            fpath_list = [fp for fplist in fpaths.values() for fp in fplist]
        else:
            fpath_list = fpaths
        return latest_file(fpath_list)

    @property
    def status(self) -> dict:
        return {
            "query": self.query_status,
            "qc": self.qc_status,
            "backfill": self.backfill_status,
            "pvlib": self.pvlib_status,
            "report": self.report_status,
        }

    @property
    def query_status(self) -> dict[str, bool]:
        return {
            query_group: len(fpaths) > 0
            for query_group, fpaths in self.query_filepaths_by_group.items()
        }

    def _validate_asset_group(self, asset_group) -> None:
        if asset_group not in self.query_groups:
            raise KeyError("Invalid asset group specified for query.")
        return

    def _query_filestem(self, asset_group):
        group_str = asset_group.replace(" ", "")
        return f"PIQuery_{group_str}_{self.site.name}_{self.year}-{self.month:02d}"

    def _generate_data_savepath(self, asset_group: str, version: str) -> Path:
        """Generates new savepath for specified asset group/version."""
        filestem = self._query_filestem(asset_group)
        if version in ["cleaned", "processed"]:
            filestem += f"_{version.upper()}"
        filename = filestem + ".csv"
        return validated_savepath(Path(self.folder, filename))

    def run_pi_query(
        self, asset_group, freq=None, skip_if_exists=True, save=True, q=True
    ) -> Union[pd.DataFrame, None]:
        """Runs monthly PI query."""
        qprint = quiet_print_function(q=q)
        self._validate_asset_group(asset_group)
        if skip_if_exists and self.query_status[asset_group] is True and save is True:
            qprint(f"Found existing PI query file for {asset_group}. ({skip_if_exists = })")
            return
        dataset = SolarDataset.from_pi_for_monthly_report(
            site=self.site.name,
            year=self.year,
            month=self.month,
            asset_group=asset_group,
            freq=freq,
            keep_tzinfo=True,
            q=q,
        )
        df = remove_tzinfo_and_standardize_index(dataset.data)
        if save is True:
            savepath = self._generate_data_savepath(asset_group, version="raw")
            df.to_csv(savepath)
            qprint(f"Saved: {str(savepath).split('FlashReports')[-1]}")
        return df

    def load_query_file(
        self, asset_group: str, version="all", with_tz=False, return_fpath=False
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, Path]]:
        """Loads data from csv file; Uses method get_data_filepath."""
        fpath = self.get_data_filepath(asset_group=asset_group, version=version)
        if fpath is None:
            df = pd.DataFrame()
        else:
            df = self.site._load_file(fpath, with_tz=with_tz)
        if return_fpath:
            return df, fpath
        return df

    @property
    def qc_groups(self) -> list[str]:
        return [g for g in self.query_groups if g not in ("Modules", "PPC")]

    @property
    def qc_status(self) -> dict[str, bool]:
        return {
            group: len(self.data_filepaths_by_version(group, version="cleaned")) > 0
            for group in self.qc_groups
        }

    def _validate_qc_group(self, asset_group):
        if asset_group not in self.qc_groups:
            raise KeyError("Invalid asset group specified for QC.")
        elif self.get_data_filepath(asset_group, version="raw") is None:
            raise Exception(f"No raw PI query file exists for {asset_group = }.")
        return

    def run_qc(
        self, asset_group, df_raw=None, skip_if_exists=True, save=True, q=True
    ) -> Union[pd.DataFrame, None]:
        """Runs auto_qc script on data files for select query groups."""
        qprint = quiet_print_function(q=q)
        self._validate_qc_group(asset_group)
        if skip_if_exists and self.qc_status[asset_group] is True and save is True:
            qprint(f"Found existing file for {asset_group}. ({skip_if_exists = })")
            return
        if df_raw is not None:
            if "PROCESSED" in df_raw.columns:
                raise ValueError("Found PROCESSED column in df_raw; expecting raw PI query data.")
            qprint(
                "Warning: when providing df_raw, there are no checks to validate asset_group. "
                "Proceed with caution."
            )
            savepath = self._generate_data_savepath(asset_group, version="cleaned")
        else:
            df_raw, raw_fpath = self.load_query_file(asset_group, version="raw", return_fpath=True)
            qprint(f"Loaded raw query data from file: {raw_fpath.name}")
            savepath = validated_savepath(raw_fpath.with_stem(raw_fpath.stem + "_CLEANED"))

        df_clean = run_auto_qc(df_raw, site=self.site.name)
        n_total_changed = df_clean["PROCESSED"].sum() if "PROCESSED" in df_clean.columns else 0
        qprint(f"QC complete; {n_total_changed = }")
        if save is True:
            df_clean.to_csv(savepath)
            qprint(f"Saved: {str(savepath).split('FlashReports')[-1]}")
        return df_clean

    @property
    def backfill_groups(self) -> list[str]:
        return ["Met Stations"]

    @property
    def backfill_status(self) -> dict[str, bool]:
        return {
            group: len(self.data_filepaths_by_version(group, version="processed")) > 0
            for group in self.backfill_groups
        }

    def _validate_backfill_group(self, asset_group) -> None:
        if asset_group not in self.backfill_groups:
            raise KeyError("Invalid asset group specified for backfill.")
        elif self.get_data_filepath(asset_group, version="cleaned") is None:
            raise Exception(f"No 'cleaned' PI query file exists for {asset_group = }.")
        return

    def run_met_backfill(
        self,
        r2_diff=0.1,
        clean_fpath=None,  # temp
        # df_clean=None,  # TODO
        skip_if_exists=True,
        save=True,
        save_plot=True,
        return_changes=False,
        q=True,
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, dict], None]:
        qprint = quiet_print_function(q=q)
        asset_group = "Met Stations"
        self._validate_backfill_group(asset_group=asset_group)
        if skip_if_exists and self.backfill_status[asset_group] is True and save is True:
            qprint(f"Found existing 'processed' Met Stations file. ({skip_if_exists = })")
            return
        clean_fp = self.get_data_filepath(asset_group, version="cleaned")
        # optional override
        if clean_fpath is not None:
            if Path(clean_fpath).exists():
                clean_fp = clean_fpath
                qprint(
                    "Warning: when providing clean_fpath, there are no checks to validate file. "
                    "Proceed with caution."
                )

        qprint(f"Using data from file: {clean_fp.name}\n\n--- BEGIN BACKFILL ---")
        output = process_and_backfill_meteo_data(
            filepath=clean_fp, site=self.site.name, n_clearsky=5, r2_diff=r2_diff, q=q
        )
        if output is None:
            qprint(f"Error in backfill script (no output).\n--- END BACKFILL ---\n")
            return
        df_processed, changes_dict = output
        qprint(f"--- END BACKFILL ---\n\n{changes_dict = }")
        if save is True:
            processed_fp = validated_savepath(clean_fp.with_stem(clean_fp.stem + "_PROCESSED"))
            df_processed.to_csv(processed_fp)
            qprint(f"Saved: {str(processed_fp).split('FlashReports')[-1]}")
            if save_plot is True:
                fig = meteo_backfill_subplots(df_processed, resample=True)
                fig_savepath = validated_savepath(
                    Path(self.folder, f"meteo_backfill_plots_{self.site.name}.html")
                )
                fig.write_html(fig_savepath)
                qprint(f"Saved: {str(fig_savepath).split('FlashReports')[-1]}")
        if return_changes:
            return df_processed, changes_dict
        return df_processed

    def generate_pvlib_savepath(self, using_dtn: bool = False) -> Path:
        pfx = "PVLib_InvAC" if not using_dtn else "PVLib_InvAC_DTN"
        filename = f"{pfx}_{self.site.name}_{self.year}-{self.month:02d}.csv"
        return validated_savepath(Path(self.folder, filename))

    @property
    def pvlib_status(self) -> bool:
        return len(self.existing_filepaths.get("pvlib", [])) > 0

    def get_pvlib_status(self) -> dict:
        """Checks requirements and returns dictionary indicating missing requirements/next steps."""
        # check if already complete
        if self.pvlib_status is True:
            return dict(status="complete", missing=None, next_step=None)

        # check if ready to run pvlib
        met_fpaths = self.data_filepaths_by_version(asset_group="Met Stations", version="all")
        if len(met_fpaths["processed"]) > 0:
            return dict(status="ready", missing=None, next_step="pvlib")

        # check requirements and determine next step
        if len(met_fpaths["cleaned"]) > 0:
            missing_file, next_step = "processed", "backfill"
        elif len(met_fpaths["raw"]) > 0:
            missing_file, next_step = "cleaned", "auto_qc"
        else:
            missing_file, next_step = "raw", "query"
        return dict(status="not_ready", missing=missing_file, next_step=next_step)

    def expected_pvlib_poa_source(self, processed_fpath=None) -> str:
        """Checks for POA sensor data in Met Station data files & returns 'sensors', or 'dtn'."""
        status_dict = self.get_pvlib_status()

        # if no processed met file, no way to determine whether sensors are valid; default DTN
        if status_dict["status"] == "not_ready":
            return "dtn"

        if processed_fpath is None:
            processed_fpath = self.get_data_filepath(
                asset_group="Met Stations", version="processed"
            )

        columns = list(pd.read_csv(processed_fpath, nrows=0).columns)
        if "Average_Across_POA" not in columns:
            return "dtn"
        return "sensors"

    def run_pvlib(
        self, processed_fpath=None, force_dtn=False, keep_tz=False, save=True, q=True
    ) -> pd.DataFrame:
        """Runs pvlib model via flashreport specific function if fpath not specified, otherwise via general method.
        -> note: when specifying a met station filepath, only the POA data will be used for the model.
        """
        qprint = quiet_print_function(q=q)
        expected_poa_source = self.expected_pvlib_poa_source(processed_fpath=processed_fpath)
        if expected_poa_source == "dtn" and processed_fpath is not None:
            raise ValueError("When filepath is specified, must contain sensor data.")

        if processed_fpath is None:
            # using latest processed met file
            df_dtn = None  # init
            if expected_poa_source == "dtn":
                df_dtn = SolarDataset.get_supporting_data(*self.args, q=q)
            df_pvlib = run_flashreport_pvlib_model(
                *self.args, localized=keep_tz, force_dtn=force_dtn, df_dtn=df_dtn, q=q
            )
        else:
            df = self.site._load_file(processed_fpath, with_tz=True)
            poa_data = pd.Series(df["Processed_POA"], name="POA")
            df_pvlib = run_pvlib_model(
                site=self.site.name,
                datetime_index=poa_data.index,
                poa_data=poa_data,
                q=q,
            )
            if not keep_tz:
                df_pvlib = remove_tzinfo_and_standardize_index(df_pvlib)

        actual_poa_source = "dtn" if "POA_DTN" in df_pvlib.columns else "sensors"
        qprint(f"{expected_poa_source = }\n{actual_poa_source = }")
        if save is True:
            using_dtn = actual_poa_source == "dtn"
            savepath = self.generate_pvlib_savepath(using_dtn=using_dtn)
            df_pvlib.to_csv(savepath)
            qprint(f"\nSaved: {str(savepath).split('FlashReports')[-1]}")

        return df_pvlib

    def run_curtailment_report(
        self, scaling_factor=1, include_sql=True, check_loss=False, q=True
    ) -> Union[None, pd.DataFrame]:
        """Runs script to generate Comanche curtailment report for XCEL.
        -> set check_loss=True to return the daily summary table & get total loss without generating full report
        -> if check_loss=True and include_sql=True, includes sql data in output
        -> if check_loss=False and include_sql=True, includes sql table in report file
        """
        qprint = quiet_print_function(q=q)
        if self.site.name != "Comanche":
            qprint(f"Note: curtailment report not supported for site = {self.site.name}.")
            return

        if check_loss is False:
            _ = generate_curtailment_report(
                self.year, self.month, scaling_factor=scaling_factor, include_sql=include_sql, q=q
            )
            return

        df_data = load_curtailment_report_data(
            self.year, self.month, pvlib_scaling_factor=scaling_factor
        )
        df = curtailment_summary_table(df_data, all_columns=True)  # from PI
        df = df.rename(
            columns={
                "pvlib": "possible_PI",
                "meter": "meter_PI",
                "lost_nrg": "curtailment_PI",
            }
        )
        lost_mw = df["curtailment_PI"].sum() / 1e3
        qprint(f"Total Loss (PI) = {lost_mw:.2f}")

        if include_sql is True:
            qprint("Querying SQL data...", end="\r")
            dfsql = sql_curtailment_summary(self.year, self.month)
            dfsql.index = pd.to_datetime(dfsql[["Year", "Month", "Day"]])
            rename_cols = {
                "Sum_ExpKWh": "possible_SQL",
                "Sum_MeterKWh": "meter_SQL",
                "Sum_CurtKWh": "curtailment_SQL",
            }
            dfsql = dfsql.rename(columns=rename_cols)
            df = df.join(dfsql[list(rename_cols.values())])

            lost_mw_sql = df["curtailment_SQL"].sum() / 1e3
            delta_pct = (lost_mw_sql - lost_mw) / lost_mw
            qprint(f"Total Loss (SQL) = {lost_mw_sql:.2f} (delta: {delta_pct:.1%})")
            ordered_cols = []
            for key in ("possible", "meter", "curtailment"):
                pi_col, sql_col, delta_col = [f"{key}_PI", f"{key}_SQL", f"delta_{key}"]
                df[delta_col] = df[sql_col].sub(df[pi_col])
                ordered_cols.extend([pi_col, sql_col, delta_col])
            df[ordered_cols] = df[ordered_cols].div(1e3)
            other_cols = [c for c in df.columns if c not in ordered_cols]
            df = df[ordered_cols + other_cols]

        return df

    @property
    def report_status(self) -> bool:
        return len(self.existing_filepaths.get("report", [])) > 0

    def get_report_status(self) -> str:
        """Note: this will update self.df_util if it was not originally provided."""
        # get status of flashreport file
        if self.report_status is True:
            status = "complete"
            report_fpath = latest_file(self.existing_filepaths["report"])
            if "Rev0" in report_fpath.name:
                status += "_nometer"
        else:
            status = "ready" if self.pvlib_status is True else "not_ready"

        # get status of utility meter data (load if not provided)
        if self.df_util is None:
            self.df_util = load_meter_historian(self.year, self.month)
        if self.site.name not in self.df_util.columns:
            if self.site.name not in ("FL1", "FL4"):
                status += " (no utility meter data)"

        return status

    def create_monthly_report(
        self, local=False, return_source_filepaths=True, q=True
    ) -> Union[None, Path]:
        """Generates and saves new monthly FlashReport file."""
        qprint = quiet_print_function(q=q)
        if self.df_util is None:
            status = self.get_report_status()
            qprint("Loaded meter historian file.")
            if "not_ready" in status:
                qprint(f"Warning: {status = }")
                return

        output = generate_monthlyFlashReport(
            sitename=self.site.name,
            year=self.year,
            month=self.month,
            q=q,
            local=local,
            df_util=self.df_util,
            return_df_and_log_info=return_source_filepaths,
        )

        if output is None:
            qprint("Encountered an error while generating report.")
            return
        if return_source_filepaths:
            return output[1]  # note: includes flashreport path
        return

    def get_full_status(self) -> dict:
        return {
            "query": self.query_status,
            "qc": self.qc_status,
            "backfill": self.backfill_status,
            "pvlib": self.pvlib_status,
            "report": self.report_status,
        }

    def incomplete_items(self) -> dict:
        incomplete = {}
        for key, status in self.get_full_status().items():
            if isinstance(status, bool):
                if status is False:
                    incomplete.update({key: status})
            else:
                incomplete.update({k: sts for k, sts in status.items() if sts is False})
        return incomplete

    def create_monthly_subplots(self, save_html=True, skip_existing=True, q=True):
        qprint = quiet_print_function(q=q)
        if skip_existing:
            if list(self.folder.glob("*summary*plots*.html")):
                qprint(f"Found existing subplot html file; {skip_existing = }")
                return
        if self.df_util is None:
            _ = self.get_report_status()
            qprint("Loaded meter historian file.")
        try:
            fig = monthly_summary_subplots(*self.args, save_html=save_html, df_util=self.df_util)
        except Exception as e:
            fig = None
            qprint(f"ERROR: {e}")
        return fig

    def get_insurance_bi_adjustment(
        self, source: str, q: bool = True, return_fpath: bool = False
    ) -> dict:
        """Loads insurance claims from file & returns adjustment for given reporting period."""
        if source not in ("file", "tracker"):
            raise ValueError("Invalid source specified. Must be either 'file' or 'tracker'.")
        if source == "file":
            output = calculate_insurance_bi_adjustment(
                site_name=self.site.name,
                year=self.year,
                month=self.month,
                return_claim_and_df=True,
                q=q,
            )
            claim_total = output["total_claim_mwh"]
            source_fpath = SOLAR_PERFORMANCE_REPORT_FILEPATH
        else:
            # otherwise, load from tracker file
            df, source_fpath = self.load_kpis_from_tracker(return_fpath=True)
            claim_total = df.loc[0, "Insurance BI Adjustment (MWh)"]
        if return_fpath:
            return claim_total, source_fpath
        return claim_total

    def load_kpis_from_tracker(self, return_fpath: bool = False):
        """Loads existing KPIs from KPI Tracker document."""
        df, source_fpaths = self.site.get_monthly_kpis(
            self.year, self.month, source="tracker", return_fpaths=True
        )
        if return_fpath:
            return df, source_fpaths[0]
        return df

    def get_kpis(
        self,
        return_fpaths: bool = False,
        return_calculated_curtailment: bool = False,
        remove_gross_if_below_meter: bool = True,  # not sure why would be False; option for now
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, list[Path]]]:
        """Collects KPIs from FlashReport and formats for transfer to KPI tracker file(s)."""
        if self.report_status is False:
            df = blank_kpi_dataframe()
            if return_fpaths is True:
                return df, []
            return df

        df, source_fpaths = self.site.get_monthly_kpis(
            year=self.year, month=self.month, source="flashreport", return_fpaths=True
        )
        df = df.set_index("Metric").rename_axis(None).T.reset_index(drop=True)
        df.columns = df.columns.map(str.strip)
        df = df.rename(columns=REPORT_KPI_COLUMN_MAPPING)
        for i, col in enumerate(ORDERED_KPI_COLUMNS):
            if col not in df.columns:
                df.insert(i, col, np.nan)

        calc_curt = np.nan
        if "Calculated Curtailment [MWh]" in df.columns:
            calc_curt = df.at[0, "Calculated Curtailment [MWh]"]

        if self.site.name == "Comanche":
            df["Curtailment - Compensable (MWh)"] = df["Curtailment - Non_Compensable (MWh)"]
            df["Curtailment - Non_Compensable (MWh)"] = np.nan

        # get curtailment totals
        curt_cols = ["Curtailment - Compensable (MWh)", "Curtailment - Non_Compensable (MWh)"]
        df["Curtailment - Total (MWh)"] = df[curt_cols].max(axis=1)

        # overwrite negative values
        pos_cols = ["DC/System Health Loss (MWh)", "Downtime Loss (MWh)"]
        for col in pos_cols:
            df.loc[df[col].lt(0), col] = 0

        # overwrite gross generation values that are less than net gen.
        if remove_gross_if_below_meter is True:
            gross_ = "Inverter Generation (MWh)"
            net_ = "Meter Generation (MWh)"
            df.loc[df[gross_].lt(df[net_]), gross_] = df[net_]  # does nothing if gross > net

        # limit availability to 100%
        inv_avail_ = "Inverter Uptime Availability (%)"
        df.loc[df[inv_avail_].gt(1), inv_avail_] = 1.00
        df = df[ORDERED_KPI_COLUMNS]

        # get insurance BI adjustment
        source = "file" if (self.year, self.month) == most_recent_reporting_period() else "tracker"
        claim_total, fpath = self.get_insurance_bi_adjustment(source=source, return_fpath=True)
        if claim_total == 0:
            claim_total = np.nan
        df["Insurance BI Adjustment (MWh)"] = claim_total
        source_fpaths.append(fpath)

        # load snow loss from file if exists (TODO: add this to the flashreport)
        snow_fpath = latest_file(list(self.folder.glob("*Snow*Losses*.csv")))
        if snow_fpath is not None:
            df_snow = pd.read_csv(snow_fpath, index_col=0, parse_dates=True)
            # taking the sum of derate and outage losses
            snow_loss = df_snow.filter(like="lostMWh").sum().sum()
            df["Snow Derate Loss (MWh)"] = snow_loss
            source_fpaths.append(snow_fpath)
        else:
            df["Snow Derate Loss (MWh)"] = 0  # i.e. overwriting the NaN value

        if return_fpaths is True:
            if return_calculated_curtailment is False:
                return df, source_fpaths
            return df, source_fpaths, calc_curt
        elif return_calculated_curtailment is True:
            return df, calc_curt
        return df

    def _get_performance_breakdown(
        self, dfkpi: pd.DataFrame, calculated_curtailment: float = 0.0, validated: bool = False
    ) -> dict:
        """Gets totals for waterfall plot; Ensures all buckets sum to zero."""

        def format_kpi_col(col):
            return (
                col.lower()
                .split(" (")[0]
                .replace(" - ", "_")
                .replace(" ", "_")
                .replace("/system", "")
            )

        df = dfkpi.T.copy()
        df.columns = ["value"]
        df.index = df.index.map(format_kpi_col)
        val = lambda kpi: df.at[kpi, "value"]

        # subtract ac module, soiling, and snow losses from dc health loss
        dc_health_loss = (
            val("dc_health_loss")
            - val("soiling_loss")
            - val("ac_module_loss")
            - val("snow_derate_loss")
        )
        if dc_health_loss < 0:
            dc_health_loss = 0

        # overwrite curtailment with calculated value (if from other source, e.g. CAISO)
        if calculated_curtailment > 0:
            df.loc["curtailment_total"] = calculated_curtailment

        # calculate estimated ac line losses
        ac_line_losses = val("inverter_generation") - val("meter_generation")
        # if ac_line_losses < 0:
        #     ac_line_losses = 0

        # ensure insurance bi is not NaN
        insurance_adj = val("insurance_bi_adjustment")
        if pd.isna(insurance_adj):
            insurance_adj = 0

        data = {
            "generation": {
                "possible": val("possible_generation"),
                "inverter": val("inverter_generation"),
                "meter": val("meter_generation"),
            },
            "losses": {
                "soiling": val("soiling_loss"),
                "snow_derate": val("snow_derate_loss"),
                "dc_health": dc_health_loss,
                "ac_module": val("ac_module_loss"),
                "downtime": val("downtime_loss"),
                "curtailment": val("curtailment_total"),
                "ac_line_losses": ac_line_losses,
            },
            "adjustments": {
                "insurance_bi": insurance_adj,
            },
        }
        if validated is False:
            return data

        # validate totals to confirm they sum to zero
        assert validate_kpi_totals(data) is True
        return data

    def get_kpi_data(
        self, adjusted: bool = False, use_calc_curtailment=True, return_fpaths=False, q=True
    ):
        """If adjusted=True, remove_gross_if_below_meter=True & losses are reconciled as much as possible."""
        qprint = quiet_print_function(q=q)
        # load kpis from flashreport
        dfkpi, source_fpaths, calc_curt = self.get_kpis(
            return_fpaths=True,
            return_calculated_curtailment=True,
            remove_gross_if_below_meter=adjusted,
        )
        data = self._get_performance_breakdown(
            dfkpi=dfkpi,
            calculated_curtailment=calc_curt if use_calc_curtailment else 0.0,
            validated=False,  # allow invalid totals, otherwise raises error
        )
        valid, delta = validate_kpi_totals(data, return_delta=True)
        qprint(f"{delta = :.5f} ({valid=})")
        if valid is True:
            if return_fpaths:
                return data, source_fpaths
            return data

        if adjusted is True:
            reconcile_losses(data, q=q)
            valid, delta = validate_kpi_totals(data, return_delta=True)
            qprint(f"new {delta = :.5f} ({valid=})")

        if return_fpaths:
            return data, source_fpaths
        return data

    def create_kpi_waterfall(self, kpi_data) -> go.Figure:
        """Creates waterfall plot using monthly kpi data. TODO: move this to figures.py"""
        waterfall_data = format_kpi_data_for_waterfall(kpi_data=kpi_data)
        x_vals = list(waterfall_data.keys())
        y_vals = list(waterfall_data.values())
        measure = ["absolute" if x in ("inverter", "meter") else "relative" for x in x_vals]
        text = [f"{y:.2f}" if y != 0 else "" for y in y_vals]  # TEMP

        kwargs = dict(
            # name="20",
            orientation="v",
            textposition="outside",
            connector=dict(
                line=dict(
                    color="rgb(63, 63, 63)",
                    width=1,
                ),
                mode="spanning",
            ),
        )
        fig = go.Figure(go.Waterfall(x=x_vals, y=y_vals, measure=measure, text=text, **kwargs))
        fig.update_layout(
            title=f"{self.site.name} - Performance Breakdown for {self.year}-{self.month:02d}",
            showlegend=False,
            margin=dict(t=80, b=60, l=60, r=60),
        )
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
        return fig

    @classmethod
    def load_kpis(
        cls,
        year: int,
        month: int,
        sitelist: list[str] = [],
        df_util: pd.DataFrame = None,
        q: bool = True,
        save: bool = False,
        return_fpaths: bool = False,
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, dict[str, list[Path]]]]:
        """Loads monthly reporting KPIs from most recent flashreport for specified period and sitelist.

        Parameters
        ----------
        year : int
            The year of the reporting period.
        month : int
            The month of the reporting period.
        sitelist : list[str], optional
            A list of solar site names. Defaults to empty list.
            When not provided, collects KPIs for all sites.
        df_util : pd.DataFrame, optional
            A dataframe of utility meter data from historian file.
            When not provided, will be loaded before collection loop.
        q : bool, optional
            Quiet parameter. Defaults to True (no printouts).
            When set to False, enables status printouts in terminal.
        save : bool, optional
            Whether to save an output file to the local Downloads folder.
        return_fpaths : bool, optional
            When True, returns a dictionary of source filepaths for each site.

        Returns
        -------
        pd.DataFrame or tuple[pd.DataFrame, dict[str, list[Path]]]
            A tuple with a dataframe of KPIs & source file dictionary with site names as keys
            and report-related filepaths as values -> i.e. {"site": [fpaths], ... }
            If a subset of sites is provided, the returned dataframe will still have all
            site names in the index (with all NaN values) to facilitate copying over to Excel.
            Note: If subset has a single site, returns KPIs only for that site.
        """
        qprint = quiet_print_function(q=q)
        sites_for_collection = _validated_sitelist(sitelist)
        arg_type = "subset" if len(sitelist) > 0 else "all"
        qprint(f"Collecting FlashReport KPIs for {len(sites_for_collection)} sites. ({arg_type})")
        if df_util is None:
            df_util = load_meter_historian(year, month)
            qprint(">>> loaded meter historian file")

        target_sites = SOLAR_SITES if len(sites_for_collection) != 1 else sites_for_collection
        df_list, source_files = [], {}
        for site in target_sites:
            if site in sites_for_collection:
                generator = cls(site, year, month, df_util)
                dfk, fpaths = generator.get_kpis(return_fpaths=True)
                if len(fpaths) > 0:
                    source_files.update({site: fpaths})
                    qprint(f"{site:<22} - {fpaths[0].name}")
            else:
                dfk = blank_kpi_dataframe()

            dfk.insert(0, "Site", site)
            df_list.append(dfk)

        df = pd.concat(df_list, axis=0, ignore_index=True)
        df = df.set_index("Site")

        if save is True:
            fname = f"solar_flashreport_kpis_{year}-{month:02d}.csv"
            savepath = validated_savepath(Path.home().joinpath("Downloads", fname))
            df.to_csv(savepath, index=True)
            qprint(f">>> saved: {str(savepath)}")

        if return_fpaths:
            return df, source_files
        return df

    def get_historical_kpi_table(self, n_prev_months=3):
        """Returns dataframe with kpis from flashreport & kpis from tracker from previous months.
        -> index = KPI_COLUMNS_FOR_PLOT (name: Metric / KPI)
        -> columns = YYYY-MM for specified months (ascending order)
        """
        # historical kpis
        kpi_list = []
        prev_periods = pd.date_range(
            end=pd.Timestamp(self.year, self.month, 1),
            freq="MS",
            periods=n_prev_months + 1,
            inclusive="left",
        )
        for date in prev_periods:
            dfk = self.site.get_monthly_kpis(date.year, date.month, source="tracker")
            dfk.index = dfk["Combo Date"].dt.strftime("%Y-%m")
            dfk = dfk[KPI_COLUMNS_FOR_PLOT].T
            kpi_list.append(dfk)

        # kpis from specified reporting period (from report)
        df = self.get_kpis()
        budget_vals = self.site.get_budget_values(self.year, self.month)
        budget_key_mapping = {
            "poa": "Budgeted POA (kWh/m2)",
            "production": "Budgeted Production (MWh)",
            "curtailment": "Budgeted Curtailment (MWh)",
        }
        for key, val in budget_vals.items():
            col = budget_key_mapping[key]
            df[col] = val
        df = df[KPI_COLUMNS_FOR_PLOT].T
        df.columns = [f"{self.year}-{self.month:02d}"]
        kpi_list.append(df)

        df_kpis = pd.concat(kpi_list, axis=1)
        df_kpis.index.name = "Mertric / KPI"
        return df_kpis

    def get_data_for_summary_plot(self, return_fpaths=False):
        site_col_dict = {
            "Inverters": "Inv_Total_MW",
            "Meter": "PI_Meter_MW",
            "pvlib": "Possible_MW",
        }
        data_dict = self.quick_load_latest_data(with_tz=False, include_fpaths=True)
        missing_keys = list(filter(lambda k: k not in data_dict, site_col_dict.keys()))
        if len(missing_keys) > 0:
            raise KeyError(f"Missing the following data: {missing_keys}")

        output_data = {key.lower(): df for key, df in data_dict.items() if key in site_col_dict}

        # get site-level interval data
        site_level_data = {}
        for key, col in site_col_dict.items():
            if key == "Meter":
                df = data_dict[key].iloc[:, [0]].copy()
            elif key in ("Inverters", "pvlib"):
                matching_str = "ActivePower" if key == "Inverters" else "Possible_Power"
                df = data_dict[key].filter(like=matching_str).sum(axis=1).div(1e3).to_frame()
            else:
                continue
            df.columns = [col]
            site_level_data[key] = df.copy()

        if self.df_util is not None:
            util_data = self.df_util[[self.site.name]].copy()
            output_data["utility"] = util_data
            site_level_data["utility"] = util_data.rename(columns={self.site.name: "Util_Meter_MW"})

        output_data["site_level"] = site_level_data

        # get historical kpis
        output_data["kpis"] = self.get_historical_kpi_table()
        if return_fpaths is True:
            return output_data, data_dict.get("source_files", [])
        return output_data

    # TODO: refactor the below function (from SolarSite)
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


def _validated_sitelist(sitelist: list[str] = []) -> list[str]:
    """Returns list of valid solar sites. If not provided, returns all solar sites."""
    if len(sitelist) > 0:
        sitelist = list(filter(lambda s: s in SOLAR_SITES, sitelist))
        if not sitelist:
            raise ValueError("No valid solar sites found in specified sitelist.")
        return sitelist
    return SOLAR_SITES
