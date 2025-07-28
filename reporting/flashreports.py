import os
import pandas as pd
import missingno as msno
from pathlib import Path
import matplotlib.pyplot as plt

from ..utils import oepaths, oemeta
from ..reporting.tools import get_ppa_rate
from ..reporting.openpyxl_monthly import create_monthly_report
from ..datatools.meteoqc import transposed_POA_from_DTN_GHI
from ..datatools.meterhistorian import load_meter_historian, METER_HISTORIAN_FILEPATH


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


def get_Comanche_curtailment(year, month):
    frpath = oepaths.frpath(year, month, ext="Solar")
    sourcepath = Path(frpath, "Comanche")
    if not sourcepath.exists():
        return None
    validfiles = list(sourcepath.glob("*Curtailment*Report*"))
    if not validfiles:
        return None

    vfpath = max((fp.stat().st_ctime, fp) for fp in validfiles)[1]

    # read file into pandas dataframe
    dfc = pd.read_excel(vfpath, engine="openpyxl")

    # find column with curtailment value (look for col where prev. col is 'Sum')
    target_cols = [c for i, c in enumerate(list(dfc.columns[1:])) if dfc.columns[i] == "Sum"]
    if not target_cols:
        target_cols = [c for i, c in enumerate(list(dfc.columns[1:])) if dfc.columns[i] == "DATE"]
    target_column = target_cols[0]

    # find row of last non-null value in selected column
    target_row = dfc[target_column].last_valid_index()

    # get value from target cell
    total_curtailment = dfc.at[target_row, target_column]

    return total_curtailment


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
    return [s for s in all_sheets if site_keyword in s].pop()


def load_caiso_data(site, year, month):
    caiso_fpath = get_caiso_filepath(year, month)
    if not caiso_fpath:
        return
    sheet_name = _get_caiso_sheetname(site, caiso_fpath)
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
    timeRangeM = pd.date_range(t_start, t_end, freq="min")[:-1]

    # check if data exists for selected site & times
    if not os.path.exists(siteFRpath):
        qprint("No flash report folder found for selected site.\nExiting function.")
        return

    # relevant data files in flash report folder
    inv_fileids = ["PIQuery", "Inverters", "csv"]
    ppc_fileids = ["PIQuery", "PPC", "csv"]
    pvlib_fileids = [f"PVLib_InvAC", sitename, "csv"]  # prev. ['PVLib','InvAC','Hourly','csv']

    def matchingfiles(path, ids_):
        files = [f for f in os.listdir(path) if all(i in f for i in ids_)]
        cleanfiles = [f for f in files if "CLEANED" in f]
        if cleanfiles:
            files = cleanfiles.copy()
        if len(files) > 1:
            fpaths = [Path(path, f) for f in files]
            sorted_fptups = list(sorted([(fp.stat().st_ctime, fp) for fp in fpaths], reverse=True))
            files = [tup[1].name for tup in sorted_fptups]
        return files

    # create dictionary for log info output (source filepaths)
    source_fpath_dict = {}  # init

    # pvlib file -- REQUIRED
    pvlib_files = matchingfiles(siteFRpath, pvlib_fileids)
    if not pvlib_files:
        qprint("!! No PVLib file found !!\nExiting...")
        return
    pvlibfile = pvlib_files[0]

    # inverter file
    inv_files = matchingfiles(siteFRpath, inv_fileids)
    invfile = inv_files[0] if inv_files else None

    # ppc curtailment file
    ppc_files = matchingfiles(siteFRpath, ppc_fileids)
    ppcfile = ppc_files[0] if ppc_files else None

    """ GET DTN POA INSOLATION """
    frpath = oepaths.frpath(year, month, ext="solar", site=sitename)
    met_fpaths = list(frpath.glob("PIQuery_MetStations*PROCESSED*.csv"))
    met_fp = oepaths.latest_file(met_fpaths)
    if not met_fp:
        dtn_start = pd.Timestamp(year, month, 1)
        dtn_end = dtn_start + pd.DateOffset(months=1)
        df_ext = transposed_POA_from_DTN_GHI(
            site=sitename,
            start=dtn_start,
            end=dtn_end,
            freq="1h",
            q=q,
            keep_tzinfo=False,
        )
        dtn_insolation = df_ext["poa_global"].sum() / 1e3
    else:
        df_ext = pd.read_csv(met_fp, index_col=0, parse_dates=True)
        dtn_insolation = df_ext["poa_global"].sum() / 1e3 / 60  # minute-level

    """LOAD PVLIB DATA"""
    pvlibpath = f"{siteFRpath}\\{pvlibfile}"
    dfpvlib = load_PIdatafile(pvlibpath, t_start, t_end)

    # confirm minute-level data
    delta = dfpvlib.index[1] - dfpvlib.index[0]

    # temp: determine timerange based on freq
    pvl_timerange = timeRangeM if delta.seconds == 60 else timeRangeH
    df_timerng = pd.DataFrame({"Timestamp": pvl_timerange})

    # reindex if shape differs (sometimes extra timestamp)
    if dfpvlib.shape[0] != df_timerng.shape[0]:
        if any(dfpvlib.index.duplicated()):  # fall dst extra timestamp
            dfpvlib = dfpvlib[~dfpvlib.index.duplicated(keep="first")]
        dfpvlib = dfpvlib.reindex(pvl_timerange)

    qprint(f'Loaded PVLib file ------ "{pvlibfile}"')
    source_fpath_dict["pvlib"] = Path(pvlibpath)

    """GET METER DATA"""
    # utility meter data (if exists)
    if df_util is None:
        dfum = load_meter_historian(year=year, month=month, dropna=True)
    else:
        dfum = df_util.copy()

    nometerdata = dfum.shape[0] == 0  # check if any data/rows exist for reporting month

    dfm_ = dfum.dropna(
        axis=1, how="all"
    ).copy()  # drop columns with no data for selected time range
    sitenotexist = sitename not in dfm_.columns  # check for site in filtered columns

    if sitenotexist:
        pimtrfpaths = list(Path(siteFRpath).glob("PIQuery*Meter*.csv"))
        if len(pimtrfpaths) > 0:
            ctime, pimeterpath = max((fp.stat().st_ctime, fp) for fp in pimtrfpaths)
            df_meter = load_PIdatafile(str(pimeterpath), t_start, t_end)
            df_meter = df_meter.resample("h").mean()
            df_meter = df_meter.rename(columns={df_meter.columns[0]: sitename})
            qprint(f'No utility meter found; Loaded PI Meter file ------ "{pimeterpath.name}"')
            meterpath = pimeterpath
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
        qprint('Loaded Meter file ------ "Meter_Generation_Historian.xlsm"')

    source_fpath_dict["meter"] = None if meterpath is None else Path(meterpath)

    """CREATE DICTIONARY FOR MISSING DATA FILES (EXISTENCE) (arg. for report gen. function)"""
    meterfile = False if (nometerdata or sitenotexist) else True
    filelist = [invfile, ppcfile, meterfile]

    def ismissing(file):
        return True if not file else False

    ismissinglist = list(map(ismissing, filelist))
    keylist = ["Inverters", "PPC", "Meter"]
    dict_missingfiles = dict(zip(keylist, ismissinglist))

    """LOAD INVERTER DATA (if available)"""
    if invfile:
        invpath = f"{siteFRpath}\\{invfile}"
        dfi = load_PIdatafile(invpath, t_start, t_end)
        if "cleaned" in invfile.casefold():
            inv_cols = [c for c in dfi.columns if "power" in c.casefold()]
            dfi = dfi[inv_cols].copy()

        expected_idx = pd.date_range(t_start, t_end, freq="min")[:-1]
        if dfi.shape[0] != expected_idx.shape[0]:
            dfi = dfi.reindex(expected_idx)

        # check magnitude of power values (looking for kW)
        inv_maxP = dfi.max().max()
        if inv_maxP < 100 and inv_maxP * 1000 < 1000:
            dfi = dfi * 1000

        qprint(f'Loaded Inverter file --- "{invfile}"')
    else:
        invpath = None
        qprint("!! No inverter data found. !! - replacing with df of zeros.")
        dfi = pd.DataFrame(index=timeRangeH)  # blank dataframe
        inv_cols = [
            c for c in dfpvlib.columns if any(id_ in c.casefold() for id_ in ["inv", "poss"])
        ]
        invcols = [col.split("_")[0] for col in inv_cols]  # using pvlib inv names
        for inv in invcols:
            dfi[inv] = 0

    source_fpath_dict["inverters"] = None if invpath is None else Path(invpath)

    """LOAD PPC CURTAILMENT DATA (if available)"""
    dfc = pd.DataFrame({"Timestamp": timeRangeH, "Curt_SP": site_capacity})

    # get separate curtailment value for Comanche
    if sitename == "Comanche":
        comanchePPCval = get_Comanche_curtailment(year, month)
        if not comanchePPCval:
            qprint("!! No Comanche PPC file found. !! - check source path")
    else:
        comanchePPCval = None

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
    if invfile:
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
    df4xl = (
        dfPOA.join(df_inv.join(dfpvlib_inv.join(df_avail), how="left", rsuffix="_avail"))
        .rename_axis("Timestamp")
        .reset_index(drop=False)
    )

    # variable for number of inverters
    numInv = df_inv.shape[1]

    # check merged dataframe before continuing
    if df4xl.shape[1] != 4 + 3 * numInv:  # add 4 for tstamp, poa, poaBad, and modTemp
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
            pimeterfile = [f for f in os.listdir(siteFRpath) if "PIQuery_Meter" in f][0]
            pimeterpath = os.path.join(siteFRpath, pimeterfile)
            dfm = load_PIdatafile(pimeterpath, t_start, t_end)

            df_caiso2 = create_curtailment_table_from_caiso_data(sitename, df_caiso, dfm, dfpvlib)

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
        comanchePPCval=comanchePPCval,
        missingfiles=dict_missingfiles,
        df_caiso=df_caiso,
        df_caiso2=df_caiso2,
    )

    if not q:
        print(f"|| ~~~ END MONTHLY REPORT GENERATION - {sitename.upper()} ~~~ ||\n")

    if return_df_and_fpath:
        return df4xl, Path(savepath)
    elif return_df_and_log_info:
        source_fpath_dict.update({"report": Path(savepath)})
        return df4xl, source_fpath_dict

    return
