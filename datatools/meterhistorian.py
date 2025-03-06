import openpyxl
import os, calendar
import pandas as pd
import datetime as dt
from pathlib import Path
import shutil, tempfile, xmltodict

from ..dataquery import pireference as ref, pitools as pt
from ..datatools import mdatatransforms
from ..reporting.tools import (
    ALL_SOLAR_SITES,
    ALL_WIND_SITES,
    load_meter_historian,
    date_range,
    site_frpath,
)
from ..utils import oepaths
from ..utils.helpers import quiet_print_function


UTILITY_DATA_OUTPUT_FOLDER = Path(oepaths.UTILITY_METER_DIR, "Output")

UTILITY_DATA_SOURCE_DIRS = {
    "solar": Path(oepaths.UTILITY_METER_DIR, "Solar"),
    "wind": Path(oepaths.UTILITY_METER_DIR, "Wind"),
}

PI_DATA_SITES = ["CW-Marin", "Indy I", "Indy II", "Indy III"]

VARSITY_SITE_IDS = {
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


def get_data_folder(year: int, month: int, fleet: str):
    source_dir = UTILITY_DATA_SOURCE_DIRS[fleet]
    if not source_dir:
        raise ValueError("Invalid fleet")
    period_folder = pd.Timestamp(year, month, 1).strftime("%Y%m%d")
    return Path(source_dir, period_folder)


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
        "Maplewood 1": f"*RealTimeEnergyDetails*{yyyy}-{mm}-01*-{last_day:02d}*MW1*.xlsx",
        "Maplewood 2": f"*RealTimeEnergyDetails*{yyyy}-{mm}-01*-{last_day:02d}*MW2*.xlsx",
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
        elif site in VARSITY_SITE_IDS:
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
    return [site for site, id_ in VARSITY_SITE_IDS.items() if id_ in data]


def _stlmtui_sites_by_filepath(fpath_list):
    """Returns dictionary with filepaths as keys and lists of related sites as values"""
    return {str(fp): get_sites_from_stlmtui_file(fp) for fp in fpath_list}


def get_site_stlmtui_files(site, fpath_list):
    """Returns list of filepaths that contain data for given site"""
    sites_by_fpath = _stlmtui_sites_by_filepath(fpath_list)
    return [Path(fp) for fp, sitelist in sites_by_fpath.items() if site in sitelist]


def get_all_stlmtui_filepaths(year, month):
    """Returns list of all stlmtui filepaths for given year/month"""
    filepath_list = list(get_data_folder(year, month, "solar").glob("*stlmtui*"))
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
    meter_filepaths = list(get_data_folder(year, month, fleet).glob(filepattern))

    if site in VARSITY_SITE_IDS:
        meter_filepaths = get_site_stlmtui_files(site, meter_filepaths)

    return oepaths.sorted_filepaths(meter_filepaths)


def parse_stlmtui_contents(filepath):
    fpath = Path(filepath)
    with tempfile.TemporaryDirectory() as temp_dir:
        xml_fpath = Path(temp_dir, f"{fpath.stem}.xml")
        shutil.copy2(src=fpath, dst=xml_fpath)
        with open(xml_fpath) as xml_file:
            data_dict = xmltodict.parse(xml_file.read())
    worksheets = data_dict["Workbook"]["Worksheet"]
    ws_dict = (
        worksheets
        if isinstance(worksheets, dict)
        else [w for w in worksheets if w["@ss:Name"] == "Meter Data"][0]
    )
    data_list = []
    for row_dict in ws_dict["Table"].get("Row")[1:]:  # skip first row (header w/ no data)
        row_values = [cell_["Data"].get("#text") for cell_ in row_dict["Cell"]]
        data_list.append(row_values)
    uniquecols = []
    for nm in data_list[0]:
        i, col = 1, nm
        while col in uniquecols:
            col = f"{nm}_{i}"
        uniquecols.append(col)
    return pd.DataFrame(data_list[1:], columns=uniquecols)


def load_stlmtui_file(fpath):
    """Loads meter data from combined Varsity files. (of type "stlmtui*.xls")

    Parameters
    ----------
    fpath : str or Path object
        Full filepath to varsity meter file. (from "Varsity Generation DataBase" directory)

    Returns
    -------
    pandas DataFrame or None
        If meter file is successfully loaded, returns a dataframe with datetime index
        and hourly generation data (in MWh) for each site found in file. (columns = sitenames)
    """
    df_ = parse_stlmtui_contents(fpath)
    df_ = df_[df_["Measurement Type"].eq("GEN")].copy()
    df_ = df_[["Trade Date", "Interval ID", "Value", "Resource ID"]].copy()
    df_.columns = ["Day", "Hour", "MWh", "Site"]
    df_ = df_.replace({"Site": {id_: site for site, id_ in VARSITY_SITE_IDS.items()}})
    df_["Hour"] = pd.to_numeric(df_["Hour"]).astype(int)
    df_["MWh"] = pd.to_numeric(df_["MWh"])
    df_["MWh"] = df_["MWh"].fillna(0)  # replace NaN values with zeros
    df_ = df_.loc[df_.Hour.isin(range(1, 25))].reset_index(drop=True)  # for fall DST
    df_["Timestamp"] = pd.to_datetime(
        df_["Day"] + " " + df_["Hour"].sub(1).astype(str).str.zfill(2) + ":00:00",
        format="%m/%d/%Y %H:%M:%S",
    )
    df_ = df_[["Timestamp", "MWh", "Site"]]
    df = pd.pivot_table(df_, index="Timestamp", values="MWh", columns="Site")
    df = df.rename_axis(None, axis=1)

    expected_idx = mdatatransforms.expected_index(df, freq="h")
    # for col in df.columns:
    #     if df[col].last_valid_index() != expected_idx.max():  #TODO: implement in logging
    #         print(f'!! WARNING !! - data for {col} ends at {df[col].last_valid_index()}')
    # if df.index.max() != expected_idx.max():
    #     print(f'!! WARNING !! - incomplete date range detected in stlmtui file: "{fpath.name}"')
    if df.shape[0] != expected_idx.shape[0]:
        df = df.reindex(expected_idx)

    return df


def load_multiple_stlmtui_sites(year: int, month: int, sitelist: list):
    sites_by_file = _stlmtui_sites_by_filepath(get_all_stlmtui_filepaths(year, month))
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
    if site in VARSITY_SITE_IDS:
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
    target_sites = [s for s in sitelist if s not in VARSITY_SITE_IDS]
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
        main_sites.append(s) if (s not in VARSITY_SITE_IDS) else stlmtui_sites.append(s)

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
