import os
import numpy as np
import pandas as pd
import missingno as msno
from pathlib import Path
import matplotlib.pyplot as plt

from ..utils import oepaths, oemeta
from ..utils.assets import SolarSite
from ..utils.solar import SolarDataset
from ..reporting.curtailment import load_external_curtailment_totals
from ..reporting.tools import get_ppa_rate
from ..reporting.openpyxl_monthly import create_monthly_report
from ..dataquery.external import query_fracsun, get_fracsun_sites
from ..datatools.meterhistorian import (
    get_meter_totals_from_historian,
    load_meter_historian,
    METER_HISTORIAN_FILEPATH,
)


def load_PIdatafile(filepath, t1, t2):
    if ".csv" not in filepath:
        print("Incorrect file format - need CSV file")
        return
    df = pd.read_csv(filepath, parse_dates=[0], index_col=0)
    if df.index[0].tzinfo:
        df = df.tz_localize(None)  # remove timezone awareness
    if "Inverters" in filepath:
        df = df.astype(str).apply(pd.to_numeric, errors="coerce")
        if "Imperial Valley" in filepath:
            df = df.round(5)
    if "PPC" in filepath:
        df = df.astype(str).apply(pd.to_numeric, errors="coerce")
        df = df.fillna(0)
    df = df.loc[(df.index >= t1) & (df.index < t2)]  # filter/adjust for selected month
    return df


def get_caiso_filepath(year, month):
    """Returns the filepath to the CAISO file for given year/month"""
    # find caiso file (check frpath & parent)
    frpath = oepaths.frpath(year, month, ext="Solar")
    ympath = oepaths.frpath(year, month)
    glob_str = "*CAISO*.xlsx"
    caiso_fpaths = list(frpath.glob(glob_str))
    if not caiso_fpaths:
        caiso_fpaths = list(ympath.glob(glob_str))
    return oepaths.latest_file(caiso_fpaths)


def _get_caiso_sheetname(site, caiso_fpath):
    """Returns the sheetname in the CAISO file corresponding to the given site"""
    all_sheets = list(pd.read_excel(caiso_fpath, sheet_name=None, nrows=0).keys())
    site_keyword = sorted(site.replace("-", " ").split(), key=len)[-1]  # longest word
    matching_sheets = [s for s in all_sheets if site_keyword in s]
    if len(matching_sheets) != 1:
        return None
    return matching_sheets[0]


def load_caiso_data(site, year, month):
    caiso_fpath = get_caiso_filepath(year, month)
    if not caiso_fpath:
        return
    sheet_name = _get_caiso_sheetname(site, caiso_fpath)
    if sheet_name is None:
        return
    df_caiso = pd.read_excel(caiso_fpath, sheet_name=sheet_name, engine="calamine")
    index_col = "Interval Start Time"
    if index_col not in df_caiso.columns:
        return
    return df_caiso


def format_caiso_data(df_caiso):
    if not isinstance(df_caiso, pd.DataFrame):
        return
    df = df_caiso.copy()
    index_col = "Interval Start Time"
    if not pd.api.types.is_datetime64_any_dtype(df[index_col]):
        df[index_col] = pd.to_datetime(df[index_col])

    df = df.set_index(index_col)
    df = df[["DOT", "SUPP"]]

    year, month = df.index[0].year, df.index[0].month
    start = pd.Timestamp(year, month, 1)
    end = start + pd.DateOffset(months=1)
    expected_index = pd.date_range(start, end, freq="5min")[:-1]
    df = df[(df.index >= start) & (df.index < end)]
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")].copy()
    if df.shape[0] != len(expected_index):
        df = df.reindex(expected_index)

    df["SUPP"] = df["SUPP"].fillna(0)
    return df


def create_curtailment_table_from_caiso_data(site, df_caiso, df_meter, df_pvlib):
    """returns dataframe for populating the 'Curtailment_ONW' sheet in flashreport
    >> uses raw caiso data - output from 'load_caiso_data' function
    """
    dfc = format_caiso_data(df_caiso)
    dfm = df_meter.resample("5min").mean().copy()
    poacol = "POA" if "POA" in df_pvlib.columns else "POA_DTN"
    pvlib_inv_cols = [c for c in df_pvlib.columns if "Possible_Power" in c]
    pvlib_cols = [poacol, *pvlib_inv_cols]
    if pd.infer_freq(df_pvlib.index) == pd.Timedelta(minutes=1):
        dfp = df_pvlib[pvlib_cols].resample("5min").mean().copy()
    else:
        dfp = df_pvlib[pvlib_cols].resample("5min").ffill().copy()

    site_capacity = oemeta.data["SystemSize"]["MWAC"].get(site)

    df = pd.DataFrame(index=dfc.index.copy())
    df["Measured Insolation (Wh/m2)"] = dfp[poacol].copy()
    df["PVLib (MW)"] = dfp[pvlib_inv_cols].sum(axis=1) / 1000
    df["PVLib (MW)"] = df["PVLib (MW)"].mask(df["PVLib (MW)"] > site_capacity, site_capacity)
    df["OE.MeterMW"] = dfm.iloc[:, 0].copy()
    df["Curtailment Rate ($/kWh)"] = get_ppa_rate(site, df.index[0].year, df.index[0].month) / 1e3

    # change this to an equation linked to "CAISO" sheet columns
    df["CAISO_MW_Setpoint"] = dfc["DOT"].mask(dfc["SUPP"] == 0, site_capacity)
    df["SUPP"] = dfc["SUPP"].copy()
    return df


def generate_monthlyFlashReport(
    sitename,
    year,
    month,
    q=True,
    local=False,
    return_df_and_fpath=False,
    df_util=None,
    return_df_and_log_info=False,
):
    """
    creates and saves flash report (Excel file) for specified PI site/project
    using existing data files in corresponding FlashReport folder

        inputs:
            sitename = name of site/project (string)
            year = reporting year (number, 4-digit)
            month = reporting month (number)
            q = 'quiet' enable/disable printouts (boolean)
            local = save report to user downloads folder (boolean)
    """
    qprint = lambda msg, end="\n": None if q else print(msg, end=end)
    qprint(f"\n|| ~~~ BEGIN MONTHLY REPORT GENERATION - {sitename.upper()} ~~~ ||")

    # check if PI data exists for site name
    af_dict = oemeta.data
    site_capacity = af_dict["SystemSize"]["MWAC"][sitename]
    site_capacity_DC = af_dict["SystemSize"]["MWDC"][sitename]  ########NEW
    mod_Tcoeff = af_dict["Equip"]["Mod_Coeff"][sitename]  ########NEW

    # site flash report path & report start/end timestamps for selected year/month
    yyyymm = str(year) + str(month).zfill(2)
    siteFRpath = f"{oepaths.flashreports}\\{yyyymm}\\Solar\\{sitename}"

    t_start = pd.Timestamp(year=year, month=month, day=1, hour=0, minute=0, second=0)
    t_end = t_start + pd.DateOffset(months=1)
    timeRangeH = pd.date_range(t_start, t_end, freq="h")[:-1]

    # check if data exists for selected site & times
    if not os.path.exists(siteFRpath):
        qprint("No flash report folder found for selected site.\nExiting function.")
        return

    site = SolarSite(sitename)
    pvlib_fpath = oepaths.latest_file(site.get_flashreport_files(year, month)["pvlib"])
    if pvlib_fpath is None:
        qprint("!! No PVLib file found !!\nExiting...")
        return

    inv_fpath = oepaths.latest_file(site.get_data_filepaths(year, month, "Inverters"))
    met_fpath = oepaths.latest_file(
        site.get_met_stations_filepaths(year, month, version="processed")
    )

    # create dictionary for log info output (source filepaths)
    source_fpath_dict = {}  # init

    """ GET DTN POA INSOLATION """
    if met_fpath is not None:
        df_ext = site.load_query_file(year, month, asset_group="Met Stations")
        dtn_insolation = df_ext["poa_global"].sum() / 1e3 / 60  # minute-level
    else:
        df_ext = SolarDataset.get_supporting_data(sitename, year, month, keep_tz=False, q=q)
        dtn_insolation = df_ext["poa_global"].sum() / 1e3

    """LOAD PVLIB DATA"""
    dfpvlib = site._load_file(pvlib_fpath)
    qprint(f'Loaded PVLib file ------ "{pvlib_fpath.name}"')
    source_fpath_dict["pvlib"] = pvlib_fpath

    """GET METER DATA"""
    # utility meter data (if exists)
    if df_util is None:
        dfm_ = load_meter_historian(year=year, month=month, dropna=True)
    else:
        dfm_ = df_util.copy()

    nometerdata = dfm_.shape[0] == 0  # check if any data/rows exist for reporting month

    sitenotexist = sitename not in dfm_.columns  # check for site in filtered columns

    if sitenotexist:
        pi_meter_fpath = oepaths.latest_file(site.get_data_filepaths(year, month, "Meter"))
        if pi_meter_fpath is not None:
            df_meter = site.load_query_file(year, month, asset_group="Meter")
            df_meter = df_meter.iloc[:, [0]].resample("h").mean()
            df_meter.columns = [sitename]
            qprint(f'No utility meter found; Loaded PI Meter file ------ "{pi_meter_fpath.name}"')
            meterpath = pi_meter_fpath
        else:
            df_meter = pd.DataFrame({"Timestamp": timeRangeH, sitename: 0})
            qprint(f"!! No PI meter data found for {sitename} !! - replacing with df of zeros.")
            meterpath = None
    else:
        meterpath = METER_HISTORIAN_FILEPATH
        expected_rng = pd.date_range(dfm_.index.min(), dfm_.index.max(), freq="h")
        if len(expected_rng) != dfm_.shape[0]:
            dfm_ = dfm_.reindex(expected_rng)  # for DST (historian removes/adds tstamps)
        dfm_ = dfm_.reset_index(drop=True)  # for row numbers when writing to excel
        df_meter = dfm_.copy()
        qprint(f'Loaded Meter file ------ "{meterpath.name}"')

    source_fpath_dict["meter"] = meterpath

    """determine whether site has utility meter data - used for filename (e.g. Rev0 if no meter)"""
    meterfile = False if (nometerdata or sitenotexist) else True

    """LOAD INVERTER DATA (if available)"""
    if inv_fpath is not None:
        dfi = site._load_file(inv_fpath)
        dfi = dfi.filter(like="ActivePower")

        expected_idx = pd.date_range(t_start, t_end, freq="min")[:-1]
        if dfi.shape[0] != expected_idx.shape[0]:
            dfi = dfi.reindex(expected_idx)

        # check magnitude of power values (looking for kW)
        inv_maxP = dfi.max().max()
        if inv_maxP < 100 and inv_maxP * 1000 < 1000:
            dfi = dfi * 1000

        qprint(f'Loaded Inverter file --- "{inv_fpath.name}"')
    else:
        # invpath = None
        qprint("!! No inverter data found. !! - replacing with df of zeros.")
        dfi = pd.DataFrame(index=timeRangeH)  # blank dataframe
        inv_cols = [
            c for c in dfpvlib.columns if any(id_ in c.casefold() for id_ in ["inv", "poss"])
        ]
        invcols = [col.split("_")[0] for col in inv_cols]  # using pvlib inv names
        for inv in invcols:
            dfi[inv] = 0

    source_fpath_dict["inverters"] = inv_fpath

    """LOAD EXTERNAL CURTAILMENT DATA (if applicable/available)"""
    dfc = pd.DataFrame({"Timestamp": timeRangeH, "Curt_SP": site_capacity})

    # get curtailment values for Comanche, Maplewood 1, Maplewood 2 (returns empty dict otherwise)
    ext_curtailment = load_external_curtailment_totals(sitename, year, month)
    if sitename in ("Comanche", "Maplewood 1", "Maplewood 2"):
        source_fpath = ext_curtailment.get("source", None)
        if source_fpath is None:
            qprint("!! No external curtailment file found. !! - check source path")
        else:
            qprint(f'Loaded curtailment data --- "{source_fpath.name}"')

    """CREATE MISSING DATA FIGURE"""
    fSize = (1.4 * len(dfi.columns), 10)
    msno_fig = msno.matrix(
        dfi, figsize=fSize, fontsize=11, color=((19 / 255), (94 / 255), (135 / 255)), freq="2D"
    )
    fig = msno_fig.get_figure()
    plt.close(fig)

    """ADD TABLE FOR INVERTER HOURLY AVAILABILITY"""
    availcols = [f"{c}_timeAvail" for c in dfi.columns]
    df_avail = dfi.copy()
    kw_threshold = 0.1  # inv kw threshold

    # TODO - add POA threshold condition - if below threshold (25), default to available --- i.e. always 60 unless inv not performing when it should be
    if inv_fpath is not None:
        df_avail = df_avail.mask(df_avail > kw_threshold, 1).copy()
        df_avail = df_avail.mask(df_avail <= kw_threshold, 0).copy()
        df_avail = df_avail.resample("h").sum()
        df_avail.columns = availcols
    else:  # defaults for when inverter data doesn't exist
        for c in df_avail.columns:
            df_avail[c] = 60

    """GET PEAK VALUES FROM INVERTER & PVLIB DATA (MINUTE-LEVEL)"""
    # filter pvlib file for inverter possible power data
    dfpvlib_inv = dfpvlib.filter(regex="Inv|INV|Poss", axis=1).copy()

    peakkWh_dict = {}
    peakkWh_dict["Inverters"] = list(dfi.max())
    peakkWh_dict["PVLib"] = list(dfpvlib_inv.max())

    # resample pvlib & pvlib_inv data to hourly
    dfpvlib_ = dfpvlib.resample("h").mean().copy()
    dfpvlib_inv = dfpvlib_inv.resample("h").mean()

    # resample inverter data to hourly & generate list of missing values/rows per inverter
    df_inv = dfi.resample("h").mean()
    missing_invdata = []
    for col in df_inv.columns:
        num = df_inv.loc[:, col].isna().sum()
        missing_invdata.append(num)

    # fill missing values with zero
    df_inv = df_inv.fillna(0)

    """COMBINE DATA TO SINGLE DATAFRAME FOR REPORT"""
    # confirm pvlib_inv, inv, and avail dfs have same shape
    getrows = lambda df: df.shape
    dflist = [df_inv, dfpvlib_inv, df_avail]
    rowlist = list(map(getrows, dflist))

    # if set has only 1 value, all are equal (no duplicates in sets)
    if len(set(rowlist)) != 1:
        qprint("\n\n!! Check dataframe dimensions before continuing !!\n")
        qprint(f"\t{df_inv.shape = }\n\t{dfpvlib_inv.shape = }\n\t{df_avail.shape = }\n")
        qprint(f"\nNo report created for {sitename}.\nExiting function.")
        return

    # create new dataframe for writing to Excel
    pvlcols = dfpvlib_.columns
    for col in ["POA_all_bad", "module_temperature"]:
        if col not in pvlcols:
            dfpvlib_[col] = 0

    poacols = (["POA"] if "POA" in pvlcols else ["POA_DTN"]) + ["POA_all_bad", "module_temperature"]

    dfPOA = dfpvlib_[poacols].copy()

    # get mods offline data
    mods_fpath = oepaths.latest_file(site.get_data_filepaths(year, month, "Modules"))
    if mods_fpath is None:
        df_acmods = pd.DataFrame(index=dfPOA.index, data={"OE.ModulesOffline_Percent": np.nan})
    else:
        df_acmods = site._load_file(mods_fpath)
        if "OE.ModulesOffline_Percent" not in df_acmods.columns:
            df_acmods = pd.DataFrame(index=dfPOA.index, data={"OE.ModulesOffline_Percent": np.nan})
        else:
            df_acmods = df_acmods.resample("h").mean()
            df_acmods = df_acmods[["OE.ModulesOffline_Percent"]]
            qprint(f'Loaded Modules file ---- "{mods_fpath.name}"')

    # get fracsun soiling data
    if sitename in get_fracsun_sites():
        start_date, end_date = map(lambda x: x.strftime("%Y-%m-%d"), [t_start, t_end])
        df_soil = query_fracsun(sitename, start_date, end_date, q=q)
        df_soil = df_soil[["soiling"]].resample("1h").ffill()
        df_soil["soilingPercent"] = df_soil["soiling"].div(100)
        df_soil = df_soil[["soilingPercent"]]
        df_soil = df_soil.reindex(dfPOA.index)
        qprint("Loaded Fracsun soiling data.")
    else:
        df_soil = pd.DataFrame(index=dfPOA.index, data={"soilingPercent": np.nan})

    df4xl = (
        dfPOA.join(
            df_acmods.join(
                df_soil.join(
                    df_inv.join(
                        dfpvlib_inv.join(df_avail),
                        how="left",
                        rsuffix="_avail",
                    ),
                ),
            ),
        )
        .rename_axis("Timestamp")
        .reset_index(drop=False)
    )

    # variable for number of inverters
    numInv = df_inv.shape[1]

    # check merged dataframe before continuing
    if df4xl.shape[1] != 6 + 3 * numInv:  # add 6 for tstamp, poa, poaBad, modTemp, mods, soiling
        qprint("\n\n!! problem merging dataframes !!\n")
        qprint(f"No report created for {sitename}.\nExiting function.")
        return

    """check if CAISO site/data exists"""
    df_caiso = df_caiso2 = None  # init
    all_caiso_sites = [
        "Adams East",
        "Alamo",
        "CID",
        "CW-Corcoran",
        "CW-Goose Lake",
        "Camelot",
        "Catalina II",
        "Columbia II",
        "Kansas",
        "Kent South",
        "Maricopa West",
        "Old River One",
        "West Antelope",
    ]
    if sitename in all_caiso_sites:
        caiso_fp = get_caiso_filepath(year, month)
        df_caiso = load_caiso_data(sitename, year, month)
        if caiso_fp is None:
            qprint("No caiso file found.")
        elif df_caiso is None:
            qprint(f"No data found in CAISO file for {sitename}. (file: {caiso_fp.name})")
        else:
            qprint(f'Loaded CAISO file ------ "{caiso_fp.name}"')
            if df_caiso.columns[0] == "Resource ID":
                df_caiso.insert(loc=0, column="Unnamed: 0", value=0)  # for excel formula

            # load pi meter data
            dfm = site.load_query_file(year, month, asset_group="Meter")

            df_caiso2 = create_curtailment_table_from_caiso_data(sitename, df_caiso, dfm, dfpvlib)

    # for sites with utility meter totals but no interval data
    utility_meter_total = None  # default
    if sitename in ["FL1", "FL4"]:
        hist_totals = get_meter_totals_from_historian(year, month)
        if hist_totals is not None:
            utility_meter_total = hist_totals.get(sitename, None)
            if utility_meter_total is not None:
                meterfile = True
                qprint("Loaded meter total from historian.")

    # get filename and generate savepath for report
    has_meterdata = meterfile
    sfolder_ = siteFRpath if not local else str(Path.home().joinpath("Downloads"))

    def filename_exists(filename):
        return os.path.exists(os.path.join(sfolder_, f"{filename}.xlsx"))

    fileID = "FlashReport" if not local else "LocalFlashReport"
    fname = f"{sitename}_{fileID}_{yyyymm}"
    if filename_exists(fname) and has_meterdata:
        n = 1
        while filename_exists(f"{fname}_Rev{n}"):
            n += 1
        rptfilename = f"{fname}_Rev{n}"
    elif not has_meterdata:
        rptfilename = f"{fname}_Rev0"
        if filename_exists(rptfilename):
            n = 0
            while filename_exists(f"{rptfilename}-{n}"):
                n += 1
            rptfilename = f"{rptfilename}-{n}"
    else:
        rptfilename = fname

    savepath = f"{sfolder_}\\{rptfilename}.xlsx"

    """OPENPYXL REPORT GENERATION"""
    create_monthly_report(
        sitename=sitename,
        capacity=site_capacity,
        capacity_DC=site_capacity_DC,
        mod_Tcoeff=mod_Tcoeff,
        dtn_insolation=dtn_insolation,
        missinginvdata=missing_invdata,
        peakkWh_dict=peakkWh_dict,
        df4xl=df4xl,
        dfm=df_meter,
        dfc=dfc,
        fig=msno_fig,
        savepath=savepath,
        q=q,
        ext_curtailment=ext_curtailment,
        df_caiso=df_caiso,
        df_caiso2=df_caiso2,
        utility_meter_total=utility_meter_total,
    )

    if not q:
        print(f"|| ~~~ END MONTHLY REPORT GENERATION - {sitename.upper()} ~~~ ||\n")

    if return_df_and_fpath:
        return df4xl, Path(savepath)
    elif return_df_and_log_info:
        source_fpath_dict.update({"report": Path(savepath)})
        return df4xl, source_fpath_dict

    return
