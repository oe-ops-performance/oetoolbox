import openpyxl
import calendar
import pandas as pd
import datetime as dt
from pathlib import Path
import shutil

from ..datatools import meter_transforms as transforms
from ..reporting.tools import ALL_SOLAR_SITES
from ..utils import oepaths
from ..utils.helpers import quiet_print_function
from ..utils.meter_paths import get_legacy_meter_filepaths


UTILITY_DATA_SOURCE_DIRS = {
    "solar": Path(oepaths.UTILITY_METER_DIR, "Solar"),
    "wind": Path(oepaths.UTILITY_METER_DIR, "Wind"),
}
UTILITY_DATA_OUTPUT_FOLDER = Path(oepaths.UTILITY_METER_DIR, "Output")
METER_HISTORIAN_FOLDER = Path(oepaths.UTILITY_METER_DIR, "Master_Version")
METER_HISTORIAN_FILEPATH = METER_HISTORIAN_FOLDER.joinpath("Meter_Generation_Historian.xlsm")

PI_DATA_SITES = ["CW-Marin", "Indy I", "Indy II", "Indy III"]

VARSITY_STLMTUI_SITE_IDS = {
    "Adams East": "ADMEST_6_SOLAR",
    "Alamo": "VICTOR_1_SOLAR2",
    "Camelot": "CAMLOT_2_SOLAR1",
    "Catalina II": "CATLNA_2_SOLAR2",
    "CID": "CORCAN_1_SOLAR1",
    "Columbia II": "CAMLOT_2_SOLAR2",
    "CW-Corcoran": "CORCAN_1_SOLAR2",
    "CW-Goose Lake": "GOOSLK_1_SOLAR1",
    "Kansas": "LEPRFD_1_KANSAS",
    "Kent South": "KNTSTH_6_SOLAR",
    "Maricopa West": "MARCPW_6_SOLAR1",
    "Old River One": "OLDRV1_6_SOLAR",
    "West Antelope": "ACACIA_6_SOLAR",
}


def load_meter_historian(year=None, month=None, dropna=True):
    df = pd.read_excel(METER_HISTORIAN_FILEPATH, engine="calamine")
    df = df.iloc[: df.Day.last_valid_index() + 1, :].copy()
    df.index = pd.to_datetime(
        df["Day"].astype(str)
        + " "
        + df["Hour Ending"].astype(int).sub(1).astype(str).str.zfill(2)
        + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
    )
    df = df[[c for c in df.columns if c in ALL_HISTORIAN_SITES]]
    if year is not None:
        df = df[(df.index.year == year)].copy()
    if month is not None:
        df = df[(df.index.month == month)].copy()
    if df.index.duplicated().any():
        df = df.loc[~df.index.duplicated()].copy()
    if dropna:
        df = df.dropna(axis=1, how="all").copy()

    convert_cols = [c for c in df.columns if c not in df.select_dtypes("number").columns]
    if convert_cols:
        df[convert_cols] = df[convert_cols].apply(pd.to_numeric)

    return df


def get_data_output_savepath(year: int, month: int):
    server_folder = Path(UTILITY_DATA_OUTPUT_FOLDER, f"{year}{month:02d}")
    filename = f"utility_meter_data_{year}-{month:02d}_output.csv"
    return oepaths.validated_savepath(Path(server_folder, filename))


def get_data_folder(year: int, month: int, fleet: str):
    source_dir = UTILITY_DATA_SOURCE_DIRS.get(fleet)
    if not source_dir:
        raise ValueError("Invalid fleet")
    return Path(source_dir, f"{year}{month:02d}")


def add_pi_data_files_to_server_folder(year: int, month: int, overwrite: bool = False):
    """Checks for PI meter data files in server folder, then copies over from report folder if exist.

    Returns dict with result status by site ("no_data", "already_exists", "copied_data", "replaced_data")
    """
    output_dict = {}
    server_folder = get_data_folder(year, month, fleet="solar")
    for site in PI_DATA_SITES:
        glob_str = f"PIQuery_Meter_{site}_*.csv"
        reporting_folder = oepaths.frpath(year, month, ext="solar", site=site)
        reporting_files = list(reporting_folder.glob(glob_str))
        if not reporting_files:
            output_dict[site] = "no_data"
            continue

        pi_fpath = oepaths.latest_file(reporting_files)
        server_files = list(server_folder.glob(glob_str))
        if not server_files:
            shutil.copy2(str(pi_fpath), str(server_folder.joinpath(pi_fpath.name)))
            output_dict[site] = "copied_data"
            continue

        if overwrite:
            for fp in server_files:
                fp.unlink()
            shutil.copy2(str(pi_fpath), str(server_folder.joinpath(pi_fpath.name)))
            output_dict[site] = "replaced_data"
            continue

        output_dict[site] = "already_exists"

    return output_dict


def last_day_of_month(year: int, month: int) -> int:
    """returns the last day of a given month"""
    return calendar.monthrange(year, month)[1]


def period_is_valid(year: int, month: int) -> tuple:
    """Determines whether a given year and month are valid"""
    return pd.Timestamp(year, month, 1) <= pd.Timestamp(dt.datetime.now()).floor("D")


def get_utility_filename_patterns(year: int, month: int, by_fleet: bool = False) -> dict:
    """returns dictionary of site names and filename patterns for given year and month"""
    yyyy = str(year)
    yy, mm = yyyy[-2:], f"{month:02d}"
    month_abbr, month_name = calendar.month_abbr[month], calendar.month_name[month]
    last_day = last_day_of_month(year, month)
    next_period = pd.Timestamp(year, month, 1) + pd.DateOffset(months=1)
    next_year, next_month = next_period.year, next_period.month
    individual_solar_filepatterns = {
        "Azalea": f"*{mm}*{yyyy}*Azalea*Generation*",
        "AZ1": f"*AZ*Solar*{mm}-{yyyy}*",
        "Comanche": f"{yyyy}-{mm}*Comanche*",
        "GA3": f"Tanglewood*Solar*Generation*{yyyy}*{mm}*",
        "GA4": f"Twiggs*County*Generation*{mm}*{yyyy}*",
        "Grand View East": f"Grand*View*Solar*{month_abbr.upper()}*{yy}*",
        "Grand View West": f"Grand*View*Solar*{month_abbr.upper()}*{yy}*",
        "Imperial Valley": f"*{mm}*IVSC*{yyyy}*",
        "Maplewood 1": f"*RealTimeEnergyDetails*{yyyy}-{mm}-01*-{last_day}*MW1*.xlsx",
        "Maplewood 2": f"*RealTimeEnergyDetails*{yyyy}-{mm}-01*-{last_day}*MW2*.xlsx",
        "MS3": f"*MSSL*{yy}{mm}*",
        "Mulberry": f"2839*{month_abbr.upper()}*{yy}*",
        "Pavant": f"*Invoice*Pavant*Solar*{next_year}{next_month:02d}*",
        "Richland": f"*Richland*Generation*",
        "Selmer": f"2838*{month_abbr.upper()}*{yy}*",
        "Somers": f"CT*_hourly.csv",
        "Sweetwater": f"Sweetwater*Solar*Invoice*{month_name}*{yyyy}*",
        "Three Peaks": f"Three*Peaks*Power*Invoice*{month_name}*{yyyy}*",
    }

    solar_filepatterns = {}
    for site in ALL_SOLAR_SITES:  # ensures correct order of sites
        if site in individual_solar_filepatterns:
            filepattern = individual_solar_filepatterns[site]
        elif site in PI_DATA_SITES:
            filepattern = f"PIQuery_Meter_{site}_*.csv"  # need last underscore for Indy sites
        elif site in VARSITY_STLMTUI_SITE_IDS:
            filepattern = "*stlmtui*"
        else:
            continue
        solar_filepatterns[site] = filepattern

    wind_filepatterns = {
        "Bingham": f"*Bingham*{month_name}_{yyyy}.xlsx",
        "Hancock": f"*Hancock*{month_name}_{yyyy}.xlsx",
        "Oakfield": f"*Oakfield*{month_name}_{yyyy}.xlsx",
        "Palouse": f"*{month_abbr}*{yyyy}*Palouse*Gen.xlsx",
        "Route 66": "*Shadow*Real*Time*Energy*Imbalance*Detail*RT66*",
        "South Plains II": "*Shadow*Real*Time*Energy*Imbalance*Detail*SP2*",
        "Sunflower": "*SUN*.PRN",
    }

    if by_fleet:
        return {"solar": solar_filepatterns, "wind": wind_filepatterns}

    return solar_filepatterns | wind_filepatterns


ALL_HISTORIAN_SITES = list(get_utility_filename_patterns(2025, 1).keys())


def get_sites_from_stlmtui_file(filepath):
    """Returns dictionary with filenames as keys and list of sites as values"""
    try:
        with open(str(filepath), "r") as file:
            data = file.read()
        return [site for site, id_ in transforms.VARSITY_STLMTUI_SITE_IDS.items() if id_ in data]
    except:
        df = transforms.load_stlmtui(filepath)
        return list(df.columns)


def stlmtui_sites_by_filepath(fpath_list):
    """Returns dictionary with filepaths as keys and lists of related sites as values"""
    return {str(fp): get_sites_from_stlmtui_file(fp) for fp in fpath_list}


def get_site_stlmtui_files(site, fpath_list):
    """Returns list of filepaths that contain data for given site"""
    sites_by_fpath = stlmtui_sites_by_filepath(fpath_list)
    return [Path(fp) for fp, sitelist in sites_by_fpath.items() if site in sitelist]


def get_all_stlmtui_filepaths(year, month):
    """Returns list of all stlmtui filepaths for given year/month"""
    data_folder = get_data_folder(year, month, fleet="solar")
    if data_folder.exists():
        filepath_list = list(data_folder.glob("*stlmtui*"))
        if not filepath_list:
            filepath_list = get_legacy_meter_filepaths(year, month).get("CID")
    else:
        filepath_list = get_legacy_meter_filepaths(year, month).get("CID")
    if not filepath_list:
        return []
    return oepaths.sorted_filepaths(filepath_list)


def get_meter_filepaths(site: str, year: int, month: int):
    """Returns a list of meter filepaths from the associated network folder using site filepattern.

    Parameters
    ----------
    site : str
    year : int
    month : int

    Returns
    -------
    list of pathlib.Path if exist, otherwise empty list
    """
    filepattern_dict = get_utility_filename_patterns(year, month, by_fleet=True)
    fleet = [fleet for fleet, dict_ in filepattern_dict.items() if site in dict_].pop()
    if not fleet:
        raise ValueError("Invalid site")

    filepattern = filepattern_dict[fleet][site]
    data_folder = get_data_folder(year, month, fleet)
    if data_folder.exists():
        meter_filepaths = list(data_folder.glob(filepattern))
        if not meter_filepaths:
            meter_filepaths = get_legacy_meter_filepaths(year, month).get(site)
    else:
        meter_filepaths = get_legacy_meter_filepaths(year, month).get(site)

    if site in VARSITY_STLMTUI_SITE_IDS:
        meter_filepaths = get_site_stlmtui_files(site, meter_filepaths)

    return oepaths.sorted_filepaths(meter_filepaths)


def load_utility_meter_file(site, filepath):
    if "stlmtui" in Path(filepath).name:  # note: use other function whenever possible
        function_name = "load_stlmtui"
    else:
        formatted_site_name = site.lower().replace(" ", "_").replace("-", "_")
        function_name = f"load_{formatted_site_name}"

    load_df_function = getattr(transforms, function_name)
    try:
        dfm = load_df_function(filepath)
    except Exception as e:
        print(e)
        dfm = pd.DataFrame()
    return dfm


def load_all_stlmtui_files(year, month, return_fpaths=False):
    """Loads all stlmtui files for year/month and returns most recent data for each site"""
    fpath_list = get_all_stlmtui_filepaths(year, month)
    fp_dict = {}
    collected_sites = []
    df_list = []
    valid_col = lambda c: c in transforms.VARSITY_STLMTUI_SITE_IDS and c not in collected_sites
    for fpath in fpath_list:
        df_ = transforms.load_stlmtui(fpath, fmt=False)  # retains datetime index (for concat)
        keep_sites = list(filter(valid_col, df_.columns))
        if not keep_sites:
            continue
        collected_sites.extend(keep_sites)
        df_list.append(df_[keep_sites])
        fp_dict[str(fpath)] = keep_sites

    if not df_list:
        df = pd.DataFrame()
    else:
        df = pd.concat(df_list, axis=1)
        ordered_cols = [c for c in transforms.VARSITY_STLMTUI_SITE_IDS if c in df.columns]
        df = transforms.formatted_dataframe(df[ordered_cols])

    if return_fpaths:
        return df, fp_dict
    return df


def load_multiple_meter_files(year, month, sitelist, return_fpaths=False):
    target_sites = [s for s in sitelist if s not in transforms.VARSITY_STLMTUI_SITE_IDS]
    if not target_sites:
        return
    df_list = []
    fpath_dict = {}
    for site in target_sites:
        fp_list = get_meter_filepaths(site, year, month)
        fpath = oepaths.latest_file(fp_list)
        if fpath is None:
            continue
        df_ = load_utility_meter_file(site, fpath)
        if not df_list:  # i.e. first df
            df_ = df_.rename(columns={"MWh": site})
        else:
            df_ = df_[["MWh"]].rename(columns={"MWh": site})

        df_list.append(df_)
        fpath_dict[site] = str(fpath)

    if not df_list:
        df = pd.DataFrame()
    else:
        df = pd.concat(df_list, axis=1)
        ordered_cols = [c for c in ALL_HISTORIAN_SITES if c in df.columns]
        df = df[ordered_cols]

    if return_fpaths:
        return df, fpath_dict
    return df


# function to load utility meter data for any site(s)
def load_meter_data(year, month, sitelist=None, q=True, return_fpaths=False):
    """Load hourly utility meter data for the given year, month, and site(s).

    Parameters
    ----------
    year : int
        Number of year. (4-digit)
    month : int
        Number of month.
    sitelist : list of str, default all sites.
        A list of site names.
    q : bool, default True
        If False, enables status printouts. (includes Utility/PI comparison)

    Returns
    -------
    pandas DataFrame
        A time series of hourly meter generation data for the requested site(s) with
        DatetimeIndex and a column (dtype: float, units: MWh) for each site in list.
        If no data is found for given site(s), returns empty DataFrame.
    """
    qprint = quiet_print_function(q=q)
    if sitelist:
        target_sites = [s for s in sitelist if s in ALL_HISTORIAN_SITES]
        if not sitelist:
            qprint("no valid sites in provided sitelist")
            return pd.DataFrame()
    else:
        target_sites = ALL_HISTORIAN_SITES

    # split sites into stlmtui & non-stlmtui
    is_stlmtui = lambda site: site in transforms.VARSITY_STLMTUI_SITE_IDS
    main_sites, stlmtui_sites = [], []
    for s in target_sites:
        main_sites.append(s) if not is_stlmtui(s) else stlmtui_sites.append(s)

    df_list = []
    fpath_dict = {}

    if main_sites:
        output = load_multiple_meter_files(year, month, main_sites, return_fpaths=True)
        if output is not None:
            df_, fp_dict = output
            df_list.append(df_)
            fpath_dict.update(fp_dict)

    if stlmtui_sites:
        output = load_all_stlmtui_files(year, month, return_fpaths=True)
        if output is not None:
            df_, fp_dict = output
            matching_cols = [c for c in stlmtui_sites if c in df_.columns]
            if matching_cols:
                if not df_list:
                    include_cols = ["Day", "Hour"] + matching_cols
                    df_list.append(df_[include_cols])
                else:
                    df_list.append(df_[matching_cols])
                fpath_dict.update(
                    {
                        s: [fp for fp, sitelist in fp_dict.items() if s in sitelist]
                        for s in matching_cols
                    }
                )

    if not df_list:
        qprint("no data found")
        return pd.DataFrame()

    if len(df_list) == 1:
        df = df_list[0].copy()
    else:
        df = pd.concat(df_list, axis=1)

    foundsites = list(fpath_dict.keys())
    qprint(f"found {len(foundsites)} of {len(target_sites)} sites\n{foundsites}")
    ordered_columns = ["Day", "Hour"] + [s for s in ALL_HISTORIAN_SITES if s in df.columns]
    df = df[ordered_columns]

    if return_fpaths:
        return df, fpath_dict
    return df


def load_data_to_server(df, q=True):
    """Loads data and saves file to directory for given reporting period (year/month from index)"""
    qprint = quiet_print_function(q=q)
    if df.empty:
        qprint("Dataframe is empty. No file saved.")
        return

    year, month = df.index[0].year, df.index[0].month
    savepath = get_data_output_savepath(year, month)
    df.to_csv(savepath)
    qprint(f"file saved: {str(savepath)}")
    return


# openpyxl functions for updating excel files
def get_dataframe_from_worksheet(ws):
    """returns dataframe with contents of Hourly Gen by Project sheet"""
    df = pd.DataFrame(ws.values)
    df.columns = df.loc[0].values
    df = df.iloc[1:, :].rename(columns={"Hour Ending": "Hour"}).copy()
    df["Hour"] = df["Hour"].astype(int)
    df.index = pd.to_datetime(
        df.Day.astype(str).str[:11] + df.Hour.sub(1).astype(str).str.zfill(2) + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
    )
    df["excel_row"] = [i + 2 for i in range(df.shape[0])]
    return df


def get_worksheet_summary(ws, year, month):
    """Returns status summary including existing/remaining sites and excel range"""
    # load worksheet to dataframe
    df_ = get_dataframe_from_worksheet(ws)
    sheet_columns = list(df_.columns)

    # filter to year/month range
    start = pd.Timestamp(year, month, 1)
    end = start + pd.DateOffset(months=1)
    df = df_[(df_.index >= start) & (df_.index < end)].copy()

    if df.empty:
        new_rows = True
        existing, remaining = [], ALL_HISTORIAN_SITES
        prev_tstamp = start - pd.Timedelta(days=1)
        prev_year, prev_month = prev_tstamp.year, prev_tstamp.month
        if df_[(df_.index.year == prev_year) & (df_.index.month == prev_month)].empty:
            return
        last_row = df_["excel_row"].max()
        n_new_rows = len(pd.date_range(start, end, freq="1h", tz="US/Pacific"))  # any tz with dst
        excel_rows = {"start": last_row + 1, "end": last_row + 1 + n_new_rows}

    else:
        # group sites according to whether data exists in the historian file
        site_columns = list(filter(lambda s: s in ALL_HISTORIAN_SITES, df.columns))
        existing, remaining = [], []
        for site in site_columns:
            remaining.append(site) if df[site].isna().all() else existing.append(site)

        # get associated excel row range in file
        new_rows = False
        excel_rows = {"start": df["excel_row"].min(), "end": df["excel_row"].max()}

    return {
        "sheet_columns": sheet_columns,
        "existing_sites": existing,
        "remaining_sites": remaining,
        "row_range": excel_rows,
        "new_rows": new_rows,
    }


def update_meter_historian_file(df, overwrite=False, q=True):
    """Updates the master historian file with data from 'load_meter_data' function output.

    Parameters
    ----------
    df : pandas DataFrame
        A dataframe of meter data for a particular reporting period (from 'load_meter_data')
    overwrite : bool, optional
        When True, writes all data in df_meter to historian (regardless of whether it already exists)
        Defaults to False.

    Returns
    -------
    None
    """
    qprint = quiet_print_function(q=q)
    input_sites = list(df.columns)
    year, month = df.index[0].year, df.index[0].month
    if not overwrite:
        df_hist = load_meter_historian(year, month)
        target_columns = [s for s in input_sites if s not in df_hist.columns]
        if not target_columns:
            qprint("all specified sites exist in file (overwrite=False)")
            return
    else:
        target_columns = input_sites

    # load workbook (note: this function takes 90+ seconds)
    qprint("Loading workbook object", end="... ")
    wb = openpyxl.load_workbook(METER_HISTORIAN_FILEPATH, keep_vba=True)
    qprint("done.")

    ws = wb["Hourly Gen by Project"]
    summary = get_worksheet_summary(ws, year, month)
    if summary is None:
        raise ValueError("Invalid year/month detected in dataframe")

    if summary["new_rows"]:
        qprint(">> note: adding new rows to file for selected year/month")
        date_cols = ["Year", "Month", "Day", "Hour"]
        df[date_cols] = df.index.strftime("%Y-%m-%d-%H").str.split("-").to_list()
        df[date_cols] = df[date_cols].apply(pd.to_numeric)
        target_columns += date_cols

    qprint("Writing new data to worksheet", end="... ")
    column_index = lambda col: summary["sheet_columns"].index(col) + 1
    row_start = summary["row_range"]["start"]
    row_end = summary["row_range"]["end"] + 1
    for i, row in enumerate(range(row_start, row_end)):
        for col in target_columns:
            cell = ws.cell(row=row, column=column_index(col))
            cell.value = df.iloc[i][col]
            if col == "Day":
                cell.number_format = "m/d/yyyy"
    qprint("done.")

    wb.save(METER_HISTORIAN_FILEPATH)
    wb.close()

    added_sites = [s for s in input_sites if s in target_columns]
    qprint(f"File saved. Data added for {len(added_sites)} sites: {added_sites}")

    return
