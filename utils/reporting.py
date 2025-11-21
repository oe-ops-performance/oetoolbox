import datetime as dt
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Union

from .assets import SolarSite
from .datetime import remove_tzinfo_and_standardize_index
from .helpers import quiet_print_function
from .oemeta import PI_SITES_BY_FLEET
from .oepaths import date_created, latest_file, sorted_filepaths, validated_savepath
from .solar import SolarDataset
from ..datatools.backfill import process_and_backfill_meteo_data, meteo_backfill_subplots
from ..datatools.pvlib import run_flashreport_pvlib_model, run_pvlib_model
from ..datatools.qcutils import run_auto_qc, qc_compare_fig
from ..datatools.meterhistorian import add_pi_data_files_to_server_folder, load_meter_historian
from ..reporting.curtailment import (
    generate_curtailment_report,
    load_curtailment_report_data,
    curtailment_summary_table,
)
from ..reporting.flashreports import generate_monthlyFlashReport
from ..reporting.insurance_bi import (
    calculate_insurance_bi_adjustment,
    SOLAR_PERFORMANCE_REPORT_FILEPATH,
)
from ..reporting.kpis import blank_kpi_dataframe, ORDERED_KPI_COLUMNS, REPORT_KPI_COLUMN_MAPPING
from ..reporting.solarplots import monthly_summary_subplots

SOLAR_SITES = PI_SITES_BY_FLEET["solar"]


def most_recent_reporting_period() -> tuple[int]:
    today = pd.Timestamp(dt.datetime.now().date())
    last_month = today - pd.DateOffset(months=1)
    return (last_month.year, last_month.month)


def validate_reporting_period(year: int, month: int):
    """Ensures defined year/month is before the current year/month."""
    max_year, max_month = most_recent_reporting_period()
    if (year, month) > (max_year, max_month):
        raise ValueError("Invalid year/month specified.")
    return


def get_start_and_end_dates(year: int, month: int):
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
            q=q,
        )
        if save is True:
            savepath = self._generate_data_savepath(asset_group, version="raw")
            dataset.data.to_csv(savepath)
            qprint(f"Saved: {str(savepath).split('FlashReports')[-1]}")
        return dataset.data

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
        """
        qprint = quiet_print_function(q=q)
        if self.site.name != "Comanche":
            qprint(f"Note: curtailment report not supported for site = {self.site.name}.")
            return

        if check_loss is True:
            df = load_curtailment_report_data(
                self.year, self.month, pvlib_scaling_factor=scaling_factor
            )
            dfs = curtailment_summary_table(df)
            lost_mw = dfs["lost_nrg"].sum() / 1e3
            qprint(f"Total Loss = {lost_mw:.2f}")
            return dfs

        _ = generate_curtailment_report(
            self.year, self.month, scaling_factor=scaling_factor, include_sql=include_sql
        )
        return

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
        self, return_fpaths: bool = False
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

        # remove gross generation values that are less than net gen.
        gross_ = "Inverter Generation (MWh)"
        net_ = "Meter Generation (MWh)"
        df.loc[df[gross_].lt(df[net_]), gross_] = np.nan

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

        if return_fpaths is True:
            return df, source_fpaths
        return df

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


def _validated_sitelist(sitelist: list[str] = []) -> list[str]:
    """Returns list of valid solar sites. If not provided, returns all solar sites."""
    if len(sitelist) > 0:
        sitelist = list(filter(lambda s: s in SOLAR_SITES, sitelist))
        if not sitelist:
            raise ValueError("No valid solar sites found in specified sitelist.")
        return sitelist
    return SOLAR_SITES
