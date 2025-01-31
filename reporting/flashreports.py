import os
import openpyxl
import numpy as np
import pandas as pd
import datetime as dt
import missingno as msno
from pathlib import Path
import matplotlib.pyplot as plt

from oetoolbox.utils import oepaths, oemeta, oeplots
from oetoolbox.reporting.tools import get_ppa_rate, latest_file, site_frpath
from oetoolbox.reporting.openpyxl_monthly import create_monthly_report
from oetoolbox.datatools.meteoqc import transposed_POA_from_DTN_GHI
import oetoolbox.reporting.fr_filestatuschecks as fs


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


"""NEED TO CHANGE THE SOURCE OF CURTAILMENT VALUE"""
# >> getting from Comanche Invoices\2023\092023\2023_09 ComancheSolarPV_PSCCo_Invoice w Curtailment.xlsx"


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


def generate_monthlyFlashReport(
    sitename, year, month, q=True, local=False, return_df_and_fpath=False, df_util=None
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
    if not q:
        print(f"\n|| ~~~ BEGIN MONTHLY REPORT GENERATION - {sitename.upper()} ~~~ ||")
    # check if PI data exists for site name
    af_dict = oemeta.data
    allsolarsites = list(af_dict["AF_Solar_V3"])
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
        print("No flash report folder found for selected site.\nExiting function.")
        return

    # relevant data files in flash report folder
    siteFRfiles = [f for f in os.listdir(siteFRpath) if not os.path.isdir(f"{siteFRpath}\\{f}")]
    inv_fileids = ["PIQuery", "Inverters", "csv"]
    ppc_fileids = ["PIQuery", "PPC", "csv"]
    pvlib_fileids = [f"PVLib_InvAC_{sitename}", "csv"]  # prev. ['PVLib','InvAC','Hourly','csv']

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

    # pvlib file -- REQUIRED
    pvlib_files = matchingfiles(siteFRpath, pvlib_fileids)
    if not pvlib_files:
        if not q:
            print("!! No PVLib file found !!\nExiting...")
        return
    pvlibfile = pvlib_files[0]

    # inverter file
    inv_files = matchingfiles(siteFRpath, inv_fileids)
    invfile = inv_files[0] if inv_files else None

    # ppc curtailment file
    ppc_files = matchingfiles(siteFRpath, ppc_fileids)
    ppcfile = ppc_files[0] if ppc_files else None

    """ GET DTN POA INSOLATION """
    frpath = site_frpath(sitename, year, month)
    met_fp = latest_file(frpath.glob("PIQuery_MetStations*PROCESSED*.csv"))
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

    if not q:
        print(f'Loaded PVLib file ------ "{pvlibfile}"')

    """GET METER DATA"""
    # utility meter data (if exists)
    if df_util is None:
        dfum = rt.load_meter_historian(year=year, month=month)
    else:
        dfum = df_util.copy()

    nometerdata = dfum.shape[0] == 0  # check if any data/rows exist for reporting month

    dfm_ = dfum.dropna(
        axis=1, how="all"
    ).copy()  # drop columns with no data for selected time range
    sitenotexist = sitename not in dfm_.columns  # check for site in filtered columns

    # if not dfm_.empty:
    #     if not pd.infer_freq(dfm_.index):
    #         dfm_ = dfm_[~dfm_.index.duplicated(keep='first')]

    # if sitenotexist, check for PI meter data file
    if sitenotexist:
        pimtrfpaths = list(Path(siteFRpath).glob("PIQuery*Meter*.csv"))
        if len(pimtrfpaths) > 0:
            ctime, pimeterpath = max((fp.stat().st_ctime, fp) for fp in pimtrfpaths)
            df_meter = load_PIdatafile(str(pimeterpath), t_start, t_end)
            df_meter = df_meter.resample("h").mean()
            df_meter = df_meter.rename(columns={df_meter.columns[0]: sitename})
            if not q:
                print(f'No utility meter found; Loaded PI Meter file ------ "{pimeterpath.name}"')
        else:
            df_meter = pd.DataFrame({"Timestamp": timeRangeH, sitename: 0})
            if not q:
                print(f"!! No PI meter data found for {sitename} !! - replacing with df of zeros.")
    else:
        dfm_ = dfm_.reset_index(drop=True)  # for row numbers when writing to excel
        df_meter = dfm_.copy()
        if not q:
            print('Loaded Meter file ------ "Meter_Generation_Historian.xlsm"')

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

        if not q:
            print(f'Loaded Inverter file --- "{invfile}"')
    else:
        if not q:
            print("!! No inverter data found. !! - replacing with df of zeros.")
        dfi = pd.DataFrame(index=timeRangeH)  # blank dataframe
        inv_cols = [
            c for c in dfpvlib.columns if any(id_ in c.casefold() for id_ in ["inv", "poss"])
        ]
        invcols = [col.split("_")[0] for col in inv_cols]  # using pvlib inv names
        for inv in invcols:
            dfi[inv] = 0

    """LOAD PPC CURTAILMENT DATA (if available)"""
    # if ppcfile and sitename!='Comanche':
    #     ppcpath = f'{siteFRpath}\\{ppcfile}'
    #     dfc = load_PIdatafile(ppcpath, t_start, t_end)
    #     dfc['Curt_SP'] = dfc['xcel_MW_setpoint']
    #     dfc['Curt_SP'] = dfc['Curt_SP'].mask(dfc['xcel_MW_cmd_request']==0, site_capacity)   #remove
    #     dfc = dfc.resample('h').mean()
    #     if not q: print(f'Loaded PPC file -------- "{ppcfile}"')
    # else:
    #     dfc = pd.DataFrame({'Timestamp': timeRangeH, 'Curt_SP': site_capacity})

    dfc = pd.DataFrame({"Timestamp": timeRangeH, "Curt_SP": site_capacity})

    # get separate curtailment value for Comanche
    if sitename == "Comanche":
        comanchePPCval = get_Comanche_curtailment(year, month)
        if not comanchePPCval and not q:
            print("!! No Comanche PPC file found. !! - check source path")
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
        print("\n\n!! Check dataframe dimensions before continuing !!\n")
        print(f"\t{df_inv.shape = }\n\t{dfpvlib_inv.shape = }\n\t{df_avail.shape = }\n")
        print(f"\nNo report created for {sitename}.\nExiting function.")
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
        print("\n\n!! problem merging dataframes !!\n")
        print(f"No report created for {sitename}.\nExiting function.")
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
        # find caiso file (check frpath & parent)
        frpath = oepaths.frpath(year, month, ext="Solar")
        ympath = oepaths.frpath(year, month)
        cstr_ = "*CAISO*.xlsx"

        c_fps = list(frpath.glob(cstr_))
        if not c_fps:
            c_fps = list(ympath.glob(cstr_))
        caiso_fp = max((fp.stat().st_ctime, fp) for fp in c_fps)[1] if c_fps else None
        if caiso_fp:
            cfp_ = max((fp.stat().st_ctime, fp) for fp in c_fps)[1]
            csheets_ = list(pd.read_excel(cfp_, sheet_name=None, nrows=0).keys())
            site_keyword = sorted(sitename.replace("-", " ").split(), key=len)[
                -1
            ]  # longest word in sitename
            matching_sheets = [s for s in csheets_ if site_keyword in s]
            if matching_sheets:
                df_caiso = pd.read_excel(cfp_, sheet_name=matching_sheets[0], engine="openpyxl")
                if not q:
                    print(f'Loaded CAISO file ------ "{cfp_.name}"')

                if df_caiso.columns[0] == "Resource ID":
                    df_caiso.insert(loc=0, column="Unnamed: 0", value=0)
                dfcaiso = df_caiso[["Interval Start Time", "DOT", "SUPP"]].copy()
                dfcaiso = dfcaiso.set_index("Interval Start Time")
                dfcaiso = dfcaiso[~dfcaiso.index.duplicated(keep="first")].copy()

                """df (freq=5min) for new caiso curtailment sheet"""
                # reindex/resample to 5-minute data (reindex fills in missing timestamps)
                idx_5m = pd.date_range(t_start, t_end, freq="5min")[:-1]
                df_caiso5 = dfcaiso.reindex(idx_5m).copy()
                df_caiso5["SUPP"] = df_caiso5["SUPP"].fillna(0)

                # pi meter data, resampled
                pimeterfile = [f for f in os.listdir(siteFRpath) if "PIQuery_Meter" in f][0]
                pimeterpath = os.path.join(siteFRpath, pimeterfile)
                df_mtr = load_PIdatafile(pimeterpath, t_start, t_end)
                df_meter5 = df_mtr.resample("5min").mean()

                # resample pvlib data
                pcols = [poacols[0]] + list(dfpvlib_inv.columns)
                df_pvlib5 = dfpvlib[pcols].copy().resample("5min").mean()

                # create new df for new caiso sheet (only California sites)
                df_caiso2 = pd.DataFrame(index=idx_5m)
                df_caiso2["Measured Insolation (Wh/m2)"] = df_pvlib5[pcols[0]].copy()
                df_caiso2["PVLib (MW)"] = df_pvlib5[pcols[1:]].sum(axis=1) / 1000
                df_caiso2["PVLib (MW)"] = df_caiso2["PVLib (MW)"].mask(
                    df_caiso2["PVLib (MW)"] > site_capacity, site_capacity
                )
                df_caiso2["OE.MeterMW"] = df_meter5[df_meter5.columns[0]].copy()
                df_caiso2["Curtailment Rate ($/kWh)"] = get_ppa_rate(sitename, year)

                # change this to an equation linked to "CAISO" sheet columns
                df_caiso2["CAISO_MW_Setpoint"] = df_caiso5["DOT"].mask(
                    df_caiso5["SUPP"] == 0, site_capacity
                )

                df_caiso2["SUPP"] = df_caiso5["SUPP"].copy()

            else:
                if not q:
                    print(f"No matching sheet found in CAISO file for {sitename}.")
        else:
            if not q:
                print(f"!! No CAISO file found !!")

    # get filename and generate savepath for report
    has_meterdata = fs.data_in_meterhistorian(sitename, year, month)  # check for meter data
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

    return


def automatic_monthlyReportGenerator(
    year, month, projects=None, q=True, local=False, onlyCAISO=False, excludeCAISO=False
):
    projectlist = projects if projects else fs.getFRprojectlist(year, month)

    if onlyCAISO or excludeCAISO:
        caiso_sites = fs.allCAISOsites
        CAISOsheets = fs.checkCAISOprojects(year, month)  # returns empty list if file not exist
        if onlyCAISO and not CAISOsheets:
            print(f"No CAISO data found for {year = }, {month = }.\nExiting.")
            return
        elif excludeCAISO:
            projectlist = [p for p in projectlist if p not in caiso_sites]
            if not q:
                print(f">> removed the following CAISO sites from list:\n{caiso_sites}\n")
        else:
            k_word = lambda site: sorted(site.replace("-", " ").split(), key=len)[-1]
            existing_caiso_sites = [
                s for s in caiso_sites if any(k_word(s) in sht for sht in CAISOsheets)
            ]
            removedsites = [s for s in caiso_sites if s not in existing_caiso_sites]
            projectlist = existing_caiso_sites.copy()
            if not q:
                print(
                    f">> filtered list to the following sites with CAISO data:\n{existing_caiso_sites}\n"
                )
                if removedsites:
                    print(f">> removed CAISO sites from list b/c no data found:\n{removedsites}\n")

    if not q:
        print(f"Generating Flash Reports for the following sites:\n{projectlist}\n")
    for project in projectlist:
        generate_monthlyFlashReport(project, year, month, q=q, local=local)

    print("\nEnd of automatic report generation.")
    return
