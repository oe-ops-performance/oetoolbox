import openpyxl
import os, calendar
import pandas as pd
import datetime as dt
from pathlib import Path
import shutil, tempfile, xmltodict

from ..dataquery import pireference as ref, pitools as pt
from ..datatools import mdatatransforms
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
# METER_HISTORIAN_FILEPATH = Path(METER_HISTORIAN_FOLDER, "Meter_Generation_Historian.xlsm")

METER_HISTORIAN_FILEPATH = METER_HISTORIAN_FOLDER.joinpath(
    "Meter_Generation_Historian_TEST-COPY.xlsm"
)

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
    with open(str(filepath), "r") as file:
        data = file.read()
    return [site for site, id_ in VARSITY_STLMTUI_SITE_IDS.items() if id_ in data]


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
    else:
        meter_filepaths = get_legacy_meter_filepaths(year, month).get(site)

    if site in VARSITY_STLMTUI_SITE_IDS:
        meter_filepaths = get_site_stlmtui_files(site, meter_filepaths)

    return oepaths.sorted_filepaths(meter_filepaths)


def parse_stlmtui_file(filepath):
    with open(str(filepath), "r") as file:
        data = file.read()
    data_dict = xmltodict.parse(data)
    worksheet_dict = data_dict["Workbook"].get("Worksheet")
    if not isinstance(worksheet_dict, dict):
        is_meter_sheet = lambda x: x["@ss:Name"] == "Meter Data"
        worksheet_dict = [d for d in worksheet_dict if is_meter_sheet(d)].pop()

    worksheet_rows = worksheet_dict["Table"].get("Row")
    get_row_values = lambda row_dict: [c["Data"].get("#text") for c in row_dict["Cell"]]

    # get column names from second row (skip first b/c blank)
    cols = pd.Series(get_row_values(worksheet_rows[1]))
    w = 1
    while cols.duplicated().any():
        if cols.loc[cols.duplicated()].str[-2].eq("_").all():
            new_cols = cols.loc[cols.duplicated()].str[:-2] + f"_{w}"
        else:
            new_cols = cols.loc[cols.duplicated()] + f"_{w}"
        cols.loc[cols.duplicated()] = new_cols
        w += 1

    # get data from third row to end
    data_list = list(map(get_row_values, worksheet_rows[2:]))

    df = pd.DataFrame(data_list, columns=cols)
    df = df.dropna(axis=1, how="all")
    return df


def format_stlmtui_data(df_):
    column_dict = {
        "Trade Date": "Day",
        "Interval ID": "Hour",
        "Value": "MWh",
        "Resource ID": "Site",
    }
    df = df_.loc[df_["Measurement Type"].eq("GEN")].copy()
    df = df[[*column_dict]].rename(columns=column_dict)
    df = df.replace({"Site": {id_: site for site, id_ in VARSITY_STLMTUI_SITE_IDS.items()}})
    df["Hour"] = pd.to_numeric(df["Hour"]).astype(int)
    df["MWh"] = pd.to_numeric(df["MWh"])
    df["MWh"] = df["MWh"].fillna(0)
    df = df.loc[df.Hour.isin(range(1, 25))].reset_index(drop=True).copy()  # for fall DST
    df["Timestamp"] = pd.to_datetime(
        df["Day"] + " " + df["Hour"].sub(1).astype(str).str.zfill(2) + ":00:00",
        format="%m/%d/%Y %H:%M:%S",
    )
    df = df[["Timestamp", "MWh", "Site"]]
    df = pd.pivot_table(df, index="Timestamp", values="MWh", columns="Site")
    df = df.rename_axis(None, axis=1)
    if df.shape[0] != mdatatransforms.expected_index(df, freq="h").shape[0]:
        df = df.reindex(mdatatransforms.expected_index(df, freq="h"))
    return df


def load_stlmtui_file(filepath):
    df_ = parse_stlmtui_file(filepath)
    return format_stlmtui_data(df_)


def load_utility_meter_file(site, filepath):
    if "stlmtui" in Path(filepath).name:
        return load_stlmtui_file(filepath)

    formatted_site_name = site.lower().replace(" ", "_").replace("-", "_")
    function_name = f"load_{formatted_site_name}"
    load_df_function = getattr(mdatatransforms, function_name)
    try:
        dfm = load_df_function(filepath)
    except Exception as e:
        print(e)
        dfm = pd.DataFrame()
    return dfm


def load_multiple_stlmtui_sites(year: int, month: int, sitelist: list, return_fpaths=False):
    sites_by_file = stlmtui_sites_by_filepath(get_all_stlmtui_filepaths(year, month))
    fpath_dict = {}
    collected_sites = []
    df_list = []
    while len(collected_sites) < len(sitelist):
        for filepath, sites in sites_by_file.items():
            target_sites = [s for s in sitelist if s not in collected_sites and s in sites]
            if not target_sites:
                continue
            df_ = load_stlmtui_file(filepath)
            df_list.append(df_[target_sites])
            collected_sites.extend(target_sites)
            fpath_dict.update({site: filepath for site in target_sites})
        break

    if not df_list:
        return pd.DataFrame()

    return pd.concat(df_list, axis=1)


# TODO: update logic to always include DST data (i.e. for all sites) when available
SUPPORTED_FALL_DST_SITES = ["Route 66", "South Plains II"]


# function to get utility meter data for sites with individual files (ie not grouped/stlmtui)
def load_site_meter_data(site, year, month, q=True, localized=False, return_df_and_fpath=False):
    """Load hourly utility meter data for the given site, year, and month.

    Parameters
    ----------
    site : str
        Name of *site.
    year : int
        Number of year, 4-digit.
    month : int
        Number of month.
    q : bool, optional
        If False, enables status printouts. Defaults to True.
    localized : bool, optional
        Returns timezone-aware data when True. Defaults to False.
    return_df_and_fpath : bool, optional
        Returns dataframe and filepath when True (for logging). Defaults to False.

    *note - there are certain varsity sites that provide data in combined files,
        and will not be able to load. (use "load_varsity_stlmtui_sites" function)

    Returns
    -------
    pandas DataFrame or None
        If meter file is found and successfully loaded, returns a dataframe with
        the following columns/dtypes: ['Day' (datetime), 'Hour' (int), 'MWh' (float)]
        where the "Hour" column represents "Hour Ending" for the interval (range 1 to 24).
        If there is no file or an error loading the file, returns None.
    """
    qprint = quiet_print_function(q=q)
    meter_fpath = oepaths.latest_file(get_meter_filepaths(site, year, month))
    if not meter_fpath:
        qprint("no file found")
        return

    qprint(f'Loading file: "{meter_fpath.name}"', end=" ... ")
    if site in VARSITY_STLMTUI_SITE_IDS:
        df_stlmtui = load_stlmtui_file(meter_fpath)
        if localized:
            pass  # TODO
        dfm = df_stlmtui[[site]]
        if return_df_and_fpath:
            return dfm, meter_fpath
        return dfm

    formatted_site_name = site.lower().replace(" ", "_").replace("-", "_")
    function_name = f"load_{formatted_site_name}"
    load_df_function = getattr(mdatatransforms, function_name)
    args = [meter_fpath]
    if localized and (site in SUPPORTED_FALL_DST_SITES):
        args.append(True)
    try:
        dfm = load_df_function(*args)
        qprint("success!")
    except:
        qprint("ERROR!")
        return

    dfm.index = pd.to_datetime(
        dfm.Day.astype(str) + " " + dfm.Hour.sub(1).astype(str).str.zfill(2) + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
    )
    df = dfm[["MWh"]].copy()
    df.columns = [site]

    if return_df_and_fpath:
        return df, meter_fpath

    return df


def load_multiple_meter_files(year, month, sitelist, q=True, localized=False, return_fpaths=False):
    target_sites = [s for s in sitelist if s not in VARSITY_STLMTUI_SITE_IDS]
    if not target_sites:
        return
    df_list = []
    fpath_dict = {}
    kwargs = dict(q=q, localized=localized, return_df_and_fpath=True)
    for site in target_sites:
        args = [site, year, month]
        output = load_site_meter_data(*args, **kwargs)
        if output is None:
            continue
        dfm, mfp = output
        df_list.append(dfm)
        fpath_dict[site] = str(mfp)

    if not df_list:
        return
    df_meter = pd.concat(df_list, axis=1)
    ordered_columns = [s for s in ALL_HISTORIAN_SITES if s in df_meter.columns]
    df_meter = df_meter[ordered_columns]

    if return_fpaths:
        return df_meter, fpath_dict
    return df_meter


# function to load utility meter data for any site(s)
def load_meter_data(year, month, sitelist=None, q=True, keep_fall_dst=False, return_fpaths=False):
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

    # only allows to keep extra timestamp if all sites in list are wind sites (functionality not built out for solar)
    if keep_fall_dst:
        if month != 11:
            keep_fall_dst = False
        if not all(s in SUPPORTED_FALL_DST_SITES for s in target_sites):
            keep_fall_dst = False

    # split sites into stlmtui & non-stlmtui
    main_sites, stlmtui_sites = [], []
    for s in target_sites:
        main_sites.append(s) if (s not in VARSITY_STLMTUI_SITE_IDS) else stlmtui_sites.append(s)

    df_list = []
    fpath_dict = {}

    if main_sites:
        output = load_multiple_meter_files(
            year, month, main_sites, q=q, localized=keep_fall_dst, return_fpaths=True
        )
        if output is not None:
            df_list.append(output[0])
            fpath_dict.update(output[1])

    if stlmtui_sites:
        df_ = load_multiple_stlmtui_sites(year, month, sitelist=stlmtui_sites)
        df_list.append(df_)

    if not df_list:
        qprint("no data found")
        return pd.DataFrame()

    df_meter = pd.concat(df_list, axis=1)
    foundsites = list(df_meter.columns)
    qprint(f"found {len(foundsites)} of {len(target_sites)} sites\n{foundsites}")
    ordered_columns = [s for s in ALL_HISTORIAN_SITES if s in df_meter.columns]
    df_meter = df_meter[ordered_columns]

    if return_fpaths:
        return df_meter, fpath_dict
    return df_meter


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
