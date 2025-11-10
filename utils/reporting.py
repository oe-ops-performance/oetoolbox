import datetime as dt
from pathlib import Path
import pandas as pd

from .assets import PISite, SolarSite
from .helpers import quiet_print_function
from .oepaths import latest_file, validated_savepath
from .solar import SolarDataset
from ..datatools.backfill import process_and_backfill_meteo_data, meteo_backfill_subplots
from ..datatools.qcutils import run_auto_qc, qc_compare_fig
from ..datatools.meterhistorian import add_pi_data_files_to_server_folder, load_meter_historian


def validate_reporting_period(year: int, month: int):
    """Ensures defined year/month is before the current year/month."""
    today = pd.Timestamp(dt.datetime.now().date())
    last_month = today - pd.DateOffset(months=1)
    max_year, max_month = last_month.year, last_month.month
    if (year, month) > (max_year, max_month):
        raise ValueError("Invalid year/month specified.")
    return


def get_start_and_end_dates(year: int, month: int):
    start = pd.Timestamp(year, month, 1)
    end = start + pd.DateOffset(months=1)
    return (start, end)


class FlashReportGenerator:
    """A class containing methods for the entire solar reporting process.

    Additional Parameters
    ---------------------
    site : oetoolbox.utils.assets.SolarSite
        An instance of a solar site, providing relevant attributes/methods for obtaining data

    Additional Attributes
    ---------------------
    site : oetoolbox.utils.assets.SolarSite
        An instance of a solar site, providing relevant attributes/methods for obtaining data
    """

    def __init__(self, site: SolarSite, year: int, month: int):

        validate_reporting_period(year, month)
        self.site = site
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
        flashreport_folder = site.flashreport_folder(year, month)
        if not flashreport_folder.exists():
            flashreport_folder.mkdir(parents=True)
        self.folder = flashreport_folder

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
        """Returns dict with same keys as filepath_dict, but switches out 'query' for individual asset groups."""
        fpath_dict = self.flashreport_fpaths.copy()
        del fpath_dict["query"]
        return {**self.query_filepaths_by_group, **fpath_dict}

    def data_filepaths_by_version(self, asset_group: str, version="all"):
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

    def get_data_filepath(self, asset_group: str, version="all"):
        """Returns most recent file for given asset_group / version"""
        return latest_file(self.data_filepaths_by_version(asset_group, version))

    @property
    def query_status(self) -> dict[str, bool]:
        return {
            query_group: len(fpaths) > 0
            for query_group, fpaths in self.query_filepaths_by_group.items()
        }

    def _validate_asset_group(self, asset_group):
        if asset_group not in self.query_groups:
            raise KeyError("Invalid asset group specified for query.")
        return

    def _query_filestem(self, asset_group):
        group_str = asset_group.replace(" ", "")
        return f"PIQuery_{group_str}_{self.site.name}_{self.year}-{self.month:02d}"

    def get_query_file_savepath(self, asset_group):
        """Returns new/unique query file savepath."""
        filename = self._query_filestem(asset_group) + ".csv"
        return validated_savepath(Path(self.folder, filename))

    def run_pi_query(self, asset_group, freq=None, skip_if_exists=True, save=True, q=True):
        """Runs monthly PI query."""
        qprint = quiet_print_function(q=q)
        self._validate_asset_group(asset_group)
        if skip_if_exists and self.query_status[asset_group] is True:
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
            savepath = self.get_query_file_savepath(asset_group)
            dataset.data.to_csv(savepath)
            qprint(f"Saved: {str(savepath).split('FlashReports')[-1]}")
        return dataset.data

    @property
    def qc_groups(self):
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

    def run_qc(self, asset_group, skip_if_exists=True, save=True, q=True):
        """Runs auto_qc script on data files for select query groups."""
        qprint = quiet_print_function(q=q)
        self._validate_qc_group(asset_group)
        if skip_if_exists and self.qc_status[asset_group] is True:
            qprint(f"Found existing file for {asset_group}. ({skip_if_exists = })")
            return
        raw_fpath = self.get_data_filepath(asset_group, version="raw")
        df_raw = self.site._load_file(raw_fpath)
        df_clean = run_auto_qc(df_raw, site=self.site.name)
        n_total_changed = df_clean["PROCESSED"].sum() if "PROCESSED" in df_clean.columns else 0
        qprint(f"QC complete; {n_total_changed = }")
        if save is True:
            clean_fpath = validated_savepath(raw_fpath.with_stem(raw_fpath.stem + "_CLEANED"))
            df_clean.to_csv(clean_fpath)
            qprint(f"Saved: {str(clean_fpath).split('FlashReports')[-1]}")
        return df_clean

    @property
    def backfill_groups(self) -> list[str]:
        return ["Met Stations"]

    @property
    def backfill_status(self) -> bool:
        return {
            group: len(self.data_filepaths_by_version(group, version="processed")) > 0
            for group in self.backfill_groups
        }

    def _validate_backfill_group(self, asset_group):
        if asset_group not in self.backfill_groups:
            raise KeyError("Invalid asset group specified for backfill.")
        elif self.get_data_filepath(asset_group, version="cleaned") is None:
            raise Exception(f"No 'cleaned' PI query file exists for {asset_group = }.")
        return

    def run_met_backfill(
        self,
        r2_diff=0.1,
        skip_if_exists=True,
        save=True,
        save_plot=True,
        return_changes=False,
        q=True,
    ):
        qprint = quiet_print_function(q=q)
        asset_group = "Met Stations"
        self._validate_backfill_group(asset_group=asset_group)
        if skip_if_exists and self.backfill_status[asset_group] is True:
            qprint(f"Found existing 'processed' Met Stations file. ({skip_if_exists = })")
            return
        clean_fp = self.get_data_filepath(asset_group, version="cleaned")
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
                fig_savepath = clean_fp.with_name(f"meteo_backfill_plots_{self.site.name}.html")
                fig.write_html(fig_savepath)
                qprint(f"Saved: {str(fig_savepath).split('FlashReports')[-1]}")
        if return_changes:
            return df_processed, changes_dict
        return df_processed
