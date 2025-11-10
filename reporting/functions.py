import time
import calendar
import openpyxl
from openpyxl.styles import PatternFill
import numpy as np
import pandas as pd
import itertools
import tempfile, shutil, pythoncom
import xlwings as xw
from pathlib import Path

from ..utils import oepaths, oemeta
from ..datatools.meterhistorian import load_meter_historian
from ..dataquery import pireference as ref
from ..dataquery.external import query_DTN


ALL_PI_SITES = {
    "gas": list(oemeta.data["AF_Gas_V3"].keys()),
    "solar": list(oemeta.data["AF_Solar_V3"].keys()),
    "wind": list(oemeta.data["AF_Wind_V3"].keys()),
}

CAISO_SITES = [
    "Adams East",
    "Alamo",
    "Camelot",
    "Catalina II",
    "CID",
    "Columbia II",
    "CW-Corcoran",
    "CW-Goose Lake",
    "Kansas",
    "Kent South",
    "Maricopa West",
    "Old River One",
    "West Antelope",
]


def load_kpi_tracker(sheet="ALL PROJECTS"):
    kwargs = dict(sheet_name=sheet, engine="calamine")
    header_row = {"Ref Tables": 1, "Site List": 1, "ALL PROJECTS": 0}.get(sheet)
    if header_row is not None:
        kwargs.update(dict(header=header_row))

    df = pd.read_excel(oepaths.kpi_tracker_rev1, **kwargs)
    if not header_row:
        return df

    if sheet != "Ref Tables":
        df = df[[c for c in df.columns if "Unnamed" not in c]]
        if sheet == "ALL PROJECTS":
            end_row = df.Project.last_valid_index() + 1
            df = df.iloc[:end_row, :].copy()
            for col in ["Year", "Month"]:
                df[col] = df[col].astype(int)
        return df

    df.columns = df.iloc[0].astype(str).str.replace(".0", "")
    ppc_column_indexes = [
        index
        for index, col in enumerate(df.columns)
        if not pd.isna(pd.to_datetime(col, errors="coerce"))
    ]
    ppc_column_indexes.insert(0, ppc_column_indexes[0] - 1)  # for site column
    df = df.iloc[1:, ppc_column_indexes].reset_index(drop=True).copy()
    df.columns = ["Site", *list(df.columns[1:])]
    end_row = df.loc[df.Site.isna()].index[0]
    df = df.iloc[:end_row, :].set_index("Site").copy()
    df.columns = df.columns.map(lambda c: pd.to_datetime(c).strftime("%Y-%m"))
    return df


# function to get ppa rate from kpi tracker (used in flashreport script)
def get_ppa_rate(site, year, month):
    dfk = load_kpi_tracker(sheet="Ref Tables")
    try:
        ppa = dfk.at[site, f"{year}-{month:02d}"]
    except:
        ppa = None
    return ppa


# function to load kpi tracker data
def get_kpis_from_tracker(sitelist, yearmonth_list):
    dfk = load_kpi_tracker()
    df_list = []
    ym_list = list(sorted(yearmonth_list, reverse=True))
    for site, ym_tup in itertools.product(sitelist, ym_list):
        year, month = ym_tup
        conditions = dfk.Project.eq(site) & dfk.Year.eq(year) & dfk.Month.eq(month)
        df_ = dfk[conditions].copy()
        df_list.append(df_)

    dfKPI = pd.concat(df_list, axis=0, ignore_index=True)
    return dfKPI


def get_solar_budget_values(year, month):
    dfk = load_kpi_tracker()
    budget_cols = [c for c in dfk.columns if "budget" in c.casefold()]
    conditions_ = dfk.Year.eq(year) & dfk.Month.eq(month)
    target_cols = ["Project"] + budget_cols
    df_budget = dfk.loc[conditions_, target_cols].reset_index(drop=True).copy()
    df_budget["sortcol"] = df_budget.Project.str.casefold().copy()
    df_budget = df_budget.sort_values(by=["sortcol"])
    df_budget = df_budget.drop(columns=["sortcol"])
    return df_budget


# functions to read from Excel file, with ability to run calculations if file has not been opened
def read_and_calculate_excel_file(filepath, sheet_name, use_tempdir=True):
    original_fpath = Path(filepath)
    parent_dir = tempfile.TemporaryDirectory() if use_tempdir else original_fpath.parent
    with parent_dir as folder:
        excel_fpath = Path(folder, original_fpath.name)
        if use_tempdir:
            shutil.copy2(original_fpath, excel_fpath)

        with xw.App(visible=False) as app:
            pythoncom.CoInitialize()
            wb = app.books.open(excel_fpath)
            app.calculate()
            wb.save()
            wb.close()
            pythoncom.CoUninitialize()

        df = pd.read_excel(excel_fpath, sheet_name=sheet_name, engine="calamine")

    return df


def load_solar_flashreport(filepath, use_tempdir=True):
    not_calculated = lambda df: df.iloc[6:9, 1:3].isna().all().all()
    kwargs = dict(sheet_name="FlashReport", engine="calamine")
    if not_calculated(pd.read_excel(filepath, nrows=10, **kwargs)):
        df = read_and_calculate_excel_file(filepath, kwargs["sheet_name"], use_tempdir=use_tempdir)
    else:
        df = pd.read_excel(filepath, **kwargs)
    return df


def get_flashreport_summary_table(df, inverter_level=False):
    if "Site Totals" in df.loc[0].values:
        df.columns = df.loc[0].values
        df = df.iloc[1:, :].reset_index(drop=True).copy()

    # site-level summary
    if not inverter_level:
        site = df.columns[0]
        end_row = df.loc[df[site].astype(str).str.contains("Tcoeff")].index[0]
        df = df.iloc[1:end_row, :3].reset_index(drop=True).copy()
        df.columns = [site, "%", "Value"]

        # copy values from % to Value column for site capacity & insolation
        copy_idx_start = df.loc[df[site].str.contains("Meter Gen")].index[0] + 1
        for r in range(copy_idx_start, df.shape[0]):
            df.at[r, "Value"] = df.at[r, "%"]
            df.at[r, "%"] = np.nan

        df = df.set_index(site)
        df.index = df.index.str.strip()
        df = df[["Value", "%"]]

    else:
        # get "Inverter Totals" summary table
        col_has_value = lambda idx_, n_rows, str_: (str_ in df.iloc[:n_rows, idx_].values)
        start_col = 0  # init
        while (start_col < df.shape[1]) and (not col_has_value(start_col, 5, "Inverter Totals")):
            start_col += 1

        end_col = 0  # init
        while (end_col < df.shape[1]) and (not col_has_value(end_col, 20, "calcCol")):
            end_col += 1

        # get start/end rows
        col0_vals = [v.strip() for v in list(df.iloc[:15, start_col].values)]
        start_row = col0_vals.index("Inverter Totals")
        end_row = col0_vals.index("Possible Generation") + 1

        # filter and format dataframe
        df = df.iloc[start_row:end_row, start_col:end_col].copy()
        df.columns = df.loc[0].values
        df = df.iloc[1:, :].set_index("Inverter Totals").rename_axis(None)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.index = [i.strip() for i in df.index.values]
        for col in df.columns:
            df.loc[df[col].lt(0), col] = 0.0

    return df


def get_monthly_ghi_total(site, year, month, force_dtn=False):
    # check for met file first
    report_dir = oepaths.frpath(year, month, "solar", site=site)
    met_fpath = oepaths.latest_file(list(report_dir.glob("PIQuery*MetStations*PROCESSED*.csv")))
    if isinstance(met_fpath, Path) and not force_dtn:
        df = pd.read_csv(met_fpath, index_col=0, parse_dates=True)
        ghi_total = df["Processed_GHI"].sum() / 1e3 / 60  # minute-level data
    else:
        # query DTN ghi
        tz = oemeta.data["TZ"].get(site)
        lat, lon = oemeta.data["LatLong"].get(site)
        start = pd.Timestamp(year, month, 1)
        end = start + pd.DateOffset(months=1)
        interval = "hour"
        args = [lat, lon, start, end, interval]
        df = query_DTN(*args, fields=["shortWaveRadiation"], tz=tz)
        df.columns = ["DTN_GHI"]
        dtn_ghi_adj = oemeta.data["DTN_GHI_adj"].get(site)
        if dtn_ghi_adj:
            df["DTN_GHI"] = df["DTN_GHI"].mul(dtn_ghi_adj)
        ghi_total = df["DTN_GHI"].sum()
    return ghi_total


KPI_COLUMN_DICT = {
    "Site": "Project",
    "DTN Insolation [kWh/m^2]": "DTN POA Insolation (kWh/m2)",
    "Insolation [kWh/m^2]": "POA Insolation (kWh/m2)",
    "Possible Generation": "Possible Generation (MWh)",
    "Actual Generation": "Inverter Generation (MWh)",
    "Meter Generation": "Meter Generation (MWh)",
    "DC / System Health": "DC/System Health Loss (MWh)",
    "Downtime": "Downtime Loss (MWh)",
    "Curtailment": "Curtailment - Non_Compensable (MWh)",
    "Inverter Availability": "Inverter Uptime Availability (%)",
}

KPI_DATA_COLUMNS = [
    "DTN POA Insolation (kWh/m2)",
    "POA Insolation (kWh/m2)",
    "GHI Insolation (kWh/m2)",
    "Possible Generation (MWh)",
    "Inverter Generation (MWh)",
    "Meter Generation (MWh)",
    "Meter Generation - ADJUSTED (MWh)",
    "AC Module Loss (MWh)",
    "DC/System Health Loss (MWh)",
    "Snow Derate Loss (MWh)",
    "Downtime Loss (MWh)",
    "Curtailment - Compensable (MWh)",
    "Curtailment - Non_Compensable (MWh)",
    "Curtailment - Total (MWh)",
    "Insurance BI Adjustment (MWh)",
    "Inverter Uptime Availability (%)",
]


def get_kpis_from_flashreport(site, year, month):
    report_dir = oepaths.frpath(year, month, "solar", site=site)
    filepath = oepaths.latest_file(list(report_dir.glob("*FlashReport*.xlsx")))
    if filepath is None:
        return

    # load file and get summary table
    df_flashreport = load_solar_flashreport(filepath)
    df = get_flashreport_summary_table(df_flashreport)

    # filter columns, move avail % to Value col, create entry for sitename
    df = df.loc[df.index.isin(KPI_COLUMN_DICT)].copy()
    df.at["Inverter Availability", "Value"] = df.at["Inverter Availability", "%"]
    df.at["Site", "Value"] = site

    # transpose, reorder/rename columns
    df = df[["Value"]].rename_axis(None).T.reset_index(drop=True).copy()
    reordered_cols = [c for c in KPI_COLUMN_DICT if c in df.columns]
    df = df[reordered_cols].rename(columns=KPI_COLUMN_DICT)

    # add columns to match tracker
    for i, col in enumerate(KPI_DATA_COLUMNS):
        if col not in df.columns:
            df.insert(i + 1, col, np.nan)

    # get ghi total (either from processed met file, or dtn data)
    df["GHI Insolation (kWh/m2)"] = get_monthly_ghi_total(site, year, month)

    return df


def load_kpis_from_flashreport(filepath):
    # load flashreport
    df = load_solar_flashreport(filepath)

    # ensure column names include header of site totals table
    check_str = "Site Totals"
    if check_str not in df.columns:
        hdr_row = None
        for i, tup in enumerate(df.iloc[:5, :5].itertuples()):
            row_dict = tup._asdict()
            if check_str in row_dict.values():
                hdr_row = i
                break
        if hdr_row is None:
            raise Exception("Unexpected formatting in report file.")
        df.columns = df.iloc[hdr_row].values
        df = df.iloc[(hdr_row + 1) :, :].reset_index(drop=True).copy()

    # locate site-level summary table (normally top left)
    col_start = df.columns.get_loc(check_str) - 1
    col_end = col_start + 3

    # find end row
    row_check_list = df.iloc[:25, col_start].to_list()
    if "Timestamp" not in row_check_list:
        print("error")
    row_end = row_check_list.index("Timestamp")

    # extract kpi table
    dft = df.iloc[:row_end, col_start:col_end].copy()

    # move first row to columns
    dft.columns = dft.loc[0].values
    dft = dft.iloc[1:].reset_index(drop=True)

    # convert to numeric (drops text)
    dft["MWh"] = pd.to_numeric(dft["MWh"], errors="coerce")

    # fill in missing 'MWh' values with those from '%' column
    dft.loc[dft["MWh"].isna(), "MWh"] = dft["%"]
    dft = dft.drop(columns=["%"])
    dft.columns = ["Metric", "Value"]

    # check for ac module loss column
    modloss_col = "AC Module Loss [MWh]"
    total_mod_loss = 0
    data_table_cols = df.loc[row_end].to_list()[:25]
    if modloss_col in data_table_cols:
        acmod_col = data_table_cols.index(modloss_col)
        # find row with total mod loss
        checkvals = df.iloc[:row_end, acmod_col - 1].to_list()
        matching_ = [
            v for v in checkvals if all(i in str(v) for i in ["Total", "Mod", "Loss", "MWh"])
        ]
        if len(matching_) == 1:
            mod_total_row = checkvals.index(matching_[0])
            total_mod_loss = df.iat[mod_total_row, acmod_col]

    dft.loc[len(dft)] = [modloss_col, total_mod_loss]

    dft = dft.dropna(axis=0, how="all").reset_index(drop=True)

    return dft
