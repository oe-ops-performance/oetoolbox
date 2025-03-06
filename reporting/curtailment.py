import calendar
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

from ..utils import oepaths

## function to load most recently-created file in fpath list
get_file = lambda fp_list: max((fp.stat().st_ctime, fp) for fp in fp_list)[1] if fp_list else None


## COMANCHE CURTAILMENT REPORT
## function to load supporting documents for curtailment report
def load_curtailment_report_data(year, month, q=True, pvlib_scaling_factor=1):
    qprint = lambda msg: None if q else print(msg)
    frpath_ = Path(oepaths.frpath(year, month, ext="Solar"), "Comanche")
    globstr_list = [f"PVLib_InvAC_Comanche_*.csv", "PIQuery_Meter_*.csv", "PIQuery_PPC_*.csv"]
    glob_fplists = list(map(lambda str_: list(frpath_.glob(str_)), globstr_list))
    reqd_fpaths = list(map(get_file, glob_fplists))
    if any((fp is None) for fp in reqd_fpaths):
        print("Supporting file requirements not met.")
        return pd.DataFrame()

    ## load files
    load_file = lambda fp: pd.read_csv(fp, index_col=0, parse_dates=True)
    df_pvl, df_mtr, df_ppc = map(load_file, reqd_fpaths)

    qprint("Loaded the following files:")
    for fp in reqd_fpaths:
        qprint(f"    {fp.name}")
    qprint("")

    ## check for processed col in mtr
    if df_mtr.shape[1] > 1:
        df_mtr = df_mtr.iloc[:, :1]

    ## get pvlib data
    poacol = "POA" if "POA" in df_pvl.columns else "POA_DTN"
    if poacol not in df_pvl.columns:
        print("!! ERROR !!")  # shouldn't happen; exit function (if not in notebook)
        return pd.DataFrame()
    invcols = [c for c in df_pvl.columns if "Possible_Power" in c]
    df_pvl["inv_sum"] = df_pvl[invcols].sum(axis=1).div(1e3)  # kW to MW
    df_pvl.loc[df_pvl["inv_sum"].gt(120), "inv_sum"] = 120.0  # limit to site capacity

    ## find ppc setpoint col
    spcols_ = [c for c in df_ppc.columns if "setpoint" in c]
    if not spcols_:
        print("!! ERROR !!")  # shouldn't happen; exit function (if not in notebook)
        return pd.DataFrame()
    spcol = spcols_[0]

    ## create new dataframe
    df = df_pvl[[poacol, "inv_sum"]].join(df_mtr).join(df_ppc[[spcol]])
    df.columns = ["poa", "pvlib", "meter", spcol.replace("_PPC", "")]
    if pvlib_scaling_factor != 1:
        df["pvlib"] = df["pvlib"].mul(pvlib_scaling_factor)

    ## calculate lost energy from curtailment using conditions below
    df.loc[:, "lost_nrg"] = float(0)  # init
    c1 = df["xcel_MW_setpoint"].lt(120)
    c2 = df["xcel_MW_setpoint"].lt(df["pvlib"])
    c3 = df["pvlib"].gt(df["meter"])
    df.loc[(c1 & c2 & c3), "lost_nrg"] = df["pvlib"].sub(df["meter"]).div(60).mul(1e3)

    ## add curtailment rate column (fixed/constant)
    df.loc[:, "curt_rate"] = 0.0625

    ## add lost revenue column using ppc rate
    df.loc[:, "lost_revenue"] = df["lost_nrg"].mul(df["curt_rate"])

    ## add number of curtailments column (for curt. hours in summary table)
    df.loc[:, "n_curtailments"] = int(0)
    df.loc[df["lost_nrg"].gt(0), "n_curtailments"] = 1

    return df


## function to get summary table from req'd files (uses df from above function as input)
def curtailment_summary_table(df, all_columns=False):
    if df.empty:
        return pd.DataFrame()
    y, m = df.index[0].year, df.index[0].month
    summary_cols = ["lost_nrg", "n_curtailments", "lost_revenue"]
    if all_columns:
        summary_cols.extend(["meter", "pvlib"])
    dfs = df[summary_cols].rename(columns={"n_curtailments": "curt_hours"}).copy()
    dfs = dfs.groupby(dfs.index.day).sum().copy()
    dfs["curt_hours"] = dfs["curt_hours"].div(60)
    if all_columns:
        dfs["meter"] = dfs["meter"].div(60).mul(1e3)
        dfs["pvlib"] = dfs["pvlib"].div(60).mul(1e3)
    dfs.index = pd.to_datetime(
        f"{y}-{m:02d}-" + dfs.index.astype(str).str.zfill(2), format="%Y-%m-%d"
    )
    return dfs


def comanche_reporting_timeseries(year, month):
    start_ = pd.Timestamp(year, month, 1)
    end_ = start_ + pd.DateOffset(months=1)
    start_date, end_date = map(lambda ts: ts.strftime("%Y-%m-%d"), [start_, end_])

    ## create engine for connection to SQL server, then use pandas to get query results
    engine = create_engine(
        "mssql+pyodbc://CORP-OPSSQL/Comanche?driver=ODBC+Driver+17+for+SQL+Server"
    )
    with engine.connect() as conn, conn.begin():
        df_sql = pd.read_sql_query(
            """
            DECLARE @StartDate date;
            DECLARE @EndDate date;

            SET @StartDate = ?;
            SET @EndDate = ?;

            WITH CTE_POI AS
            (
            SELECT POIData.Timestamp_UTC
                   ,POIData.Meter_KW/60.0 as [Meter_KWh]
                   ,POIData.Park_Potential_KW/60.0 as [Expected_KWh]
                   ,POIData.Power_Limit_SP as [PowerSP]
                   ,CASE
                       WHEN (POIData.Park_Potential_KW > POIData.Meter_KW)
                           AND (POIData.Power_Limit_SP < 120000)
                           AND (POIData.Park_Potential_KW > POIData.Power_Limit_SP)
                           THEN (POIData.Park_Potential_KW/60.0) - (POIData.Meter_KW/60.0)
                       WHEN (POIData.Park_Potential_KW <= POIData.Meter_KW)
                           OR (POIData.Park_Potential_KW < 0)
                           OR (POIData.Power_Limit_SP = 120000)
                           OR (POIData.Park_Potential_KW < POIData.Power_Limit_SP)
                           THEN 0
                   END AS [Curtailed_KWh]
            FROM [Comanche].dbo.POIData
            )
            SELECT DATEPART(YEAR, DATEADD(HOUR, -6, CTE_POI.Timestamp_UTC)) as [Year]
                   ,DATEPART(MONTH, DATEADD(HOUR, -6, CTE_POI.Timestamp_UTC)) as [Mon]
                   ,DATEPART(DAY, DATEADD(HOUR, -6, CTE_POI.Timestamp_UTC)) as [Day]
                   ,DATEPART(HOUR, DATEADD(HOUR, -6, CTE_POI.Timestamp_UTC)) as [Hour]
                   ,DATEPART(MINUTE, DATEADD(HOUR, -6, CTE_POI.Timestamp_UTC)) as [Minute]
                   ,Avg([PowerSP]) as Avg_PowerSP
                   ,Avg([Curtailed_KWh]) as Avg_CurtKWh
                   ,Avg([Meter_KWh]) as Avg_MeterKWh
                   ,Avg([Expected_KWh]) as Avg_ExpKWH
            FROM CTE_POI
            WHERE (DATEADD(HOUR, -6, CTE_POI.Timestamp_UTC) >= @StartDate)
            AND (DATEADD(HOUR, -6, CTE_POI.Timestamp_UTC) < @EndDate)
            GROUP BY DATEPART(YEAR, DATEADD(HOUR, -6, CTE_POI.Timestamp_UTC))
                     ,DATEPART(MONTH, DATEADD(HOUR, -6, CTE_POI.Timestamp_UTC))
                     ,DATEPART(DAY, DATEADD(HOUR, -6, CTE_POI.Timestamp_UTC))
                     ,DATEPART(HOUR, DATEADD(HOUR, -6, CTE_POI.Timestamp_UTC))
                     ,DATEPART(MINUTE, DATEADD(HOUR, -6, CTE_POI.Timestamp_UTC))
            ORDER BY Year, Mon, Day, Hour, Minute
            """,
            con=conn,
            params=(start_date, end_date),
        )
    return df_sql


## function to query Comanche SQL data for comparison to calculated values in summary table
def sql_curtailment_summary(year, month, q=True):
    start_ = pd.Timestamp(year, month, 1)
    end_ = start_ + pd.DateOffset(months=1)
    start_date, end_date = map(lambda ts: ts.strftime("%Y-%m-%d"), [start_, end_])

    ## create engine for connection to SQL server, then use pandas to get query results
    engine = create_engine(
        "mssql+pyodbc://CORP-OPSSQL/Comanche?driver=ODBC+Driver+17+for+SQL+Server"
    )
    with engine.connect() as conn, conn.begin():
        df_sql = pd.read_sql_query(
            """
            DECLARE @StartDate date;
            DECLARE @EndDate date;

            SET @StartDate = ?;
            SET @EndDate = ?;

            WITH CTE_POI AS
                (
                SELECT
                    Timestamp_UTC
                   ,Meter_KW/60.0 AS Meter_KWh
                   ,Park_Potential_KW/60.0 AS Expected_KWh
                   ,Power_Limit_SP AS Power_SP
                   ,CASE
                       WHEN (Park_Potential_KW > Meter_KW) AND (Power_Limit_SP < 120000) AND (Park_Potential_KW > Power_Limit_SP) THEN (Park_Potential_KW/60.0) - (Meter_KW/60.0)
                       WHEN (Park_Potential_KW <= Meter_KW) OR (Park_Potential_KW < 0) OR (Power_Limit_SP = 120000) OR (Park_Potential_KW < Power_Limit_SP) THEN 0
                    END AS Curtailed_KWh
                FROM [Comanche].dbo.POIData
                )
            SELECT
                DATEPART(YEAR, DATEADD(HOUR, -6, Timestamp_UTC)) AS Year
               ,DATEPART(MONTH, DATEADD(HOUR, -6, Timestamp_UTC)) AS Month
               ,DATEPART(DAY, DATEADD(HOUR, -6, Timestamp_UTC)) AS Day
               ,Sum(Meter_KWh) AS Sum_MeterKWh
               ,Sum(Curtailed_KWh) AS Sum_CurtKWh
               ,Sum(Expected_KWh) AS Sum_ExpKWh
            FROM CTE_POI
            WHERE DATEADD(HOUR, -6, Timestamp_UTC) >= @StartDate AND DATEADD(HOUR, -6, Timestamp_UTC) < @EndDate
            GROUP BY
                 DATEPART(YEAR, DATEADD(HOUR, -6, Timestamp_UTC))
                ,DATEPART(MONTH, DATEADD(HOUR, -6, Timestamp_UTC))
                ,DATEPART(DAY, DATEADD(HOUR, -6, Timestamp_UTC))
            ORDER BY Year, Month, Day
            """,
            con=conn,
            params=(start_date, end_date),
        )
    return df_sql


## main function to generate Comanche monthly curtailment report
# TODO: create two separate sheets for PI data & SQL data, then can remove whichever sheet & re-save as "_XCEL" copy
def generate_curtailment_report(year, month, local=False, q=True):
    qprint = lambda msg, end="\n": None if q else print(msg, end=end)

    df = load_curtailment_report_data(year, month, q=q)
    dfs = curtailment_summary_table(df)
    if dfs.empty:
        print("Error generating report.\nExiting..")
        return

    qprint("Querying SQL data... ", end="")
    dfsql = sql_curtailment_summary(year, month)
    qprint("done!\n")

    ## reset index of df and dfs (for dataframe_to_rows function)
    df = df.reset_index(drop=False)
    dfs = dfs.reset_index(drop=False)

    ## get path of curtailment report template
    tpath_ = Path(
        oepaths.solar, "Data", "Comanche", "Templates", "Curtailment_Report_TEMPLATE.xlsx"
    )

    ## establish cell number format strings
    fmt_datetime = "yyyy-mm-dd hh:mm;@"
    fmt_date = "yyyy-mm-dd;@"
    fmt_4z = "0.0000"
    fmt_2z = "0.00"
    fmt_rev = "#,##0.00"
    fmt_int = "0"

    ## OPEN WORKBOOK (template)
    qprint("Generating report... ", end="")
    wb = openpyxl.load_workbook(tpath_)
    ws = wb["Curtailment_ONW"]

    ## MAIN DATA TABLE (from df)
    ## cols. A-I:     [Timestamp,  poa,  pvlib,  meter, xcel_sp, lost_nrg,  rate, lost_rev, n_curt]
    rng1_formats = [fmt_datetime, fmt_4z, fmt_4z, fmt_4z, fmt_2z, fmt_4z, fmt_4z, fmt_rev, fmt_int]
    for r, row in enumerate(dataframe_to_rows(df, index=False, header=False)):
        for c, val in enumerate(row):
            ws.cell(r + 2, c + 1).value = val
            ws.cell(r + 2, c + 1).number_format = rng1_formats[c]

    ## SUMMARY TABLE (from dfs)
    ## cols. L-O: [Timestamp, lost_nrg, curt_hrs, lost_rev]
    rng2_formats = [fmt_date, fmt_2z, fmt_2z, fmt_rev]
    rng2_offset = 12  # starts at col idx 12
    for r, row in enumerate(dataframe_to_rows(dfs, index=False, header=False)):
        for c, val in enumerate(row):
            cell = ws.cell(r + 2, c + rng2_offset)
            cell.value = val
            cell.number_format = rng2_formats[c]

    totals_row = 33  # hard-coded (to accommodate months with diff # of days)
    totals_cols = ["lost_nrg", "curt_hours", "lost_revenue"]
    totals_formats = [fmt_rev, fmt_2z, fmt_rev]
    for c, col in enumerate(totals_cols):
        col_idx = c + rng2_offset + 1
        cell = ws.cell(totals_row, col_idx)
        cell.value = dfs[col].sum()
        cell.number_format = totals_formats[c]
        if col_idx == 13:
            ws.cell(totals_row + 1, col_idx).value = dfs[col].sum() / 1000
            ws.cell(totals_row + 1, col_idx).number_format = totals_formats[c]

    ## SQL SUMMARY TABLE (from dfsql)
    ## cols. R-W:  [Year, Month, Day, Sum_Meter, Sum_Curt, Sum_Expected]
    rng3_formats = [fmt_int] * 3 + [fmt_2z] * 3
    rng3_offset = 18  # starts at col idx 18
    for r, row in enumerate(dataframe_to_rows(dfsql, index=False, header=False)):
        for c, val in enumerate(row):
            cell = ws.cell(r + 2, c + rng3_offset)
            cell.value = val
            cell.number_format = rng3_formats[c]

    sql_curt_col = rng3_offset + 4
    sql_curt_kwh = dfsql["Sum_CurtKWh"].sum()
    ws.cell(totals_row, sql_curt_col).value = sql_curt_kwh
    ws.cell(totals_row, sql_curt_col).number_format = fmt_rev
    ws.cell(totals_row + 1, sql_curt_col).value = sql_curt_kwh / 1000
    ws.cell(totals_row + 1, sql_curt_col).number_format = fmt_rev

    ref_kwh_lost = dfs["lost_nrg"].sum()  # from first summary table (for comparison)
    ws.cell(totals_row + 2, sql_curt_col).value = (sql_curt_kwh - ref_kwh_lost) / ref_kwh_lost
    ws.cell(totals_row + 2, sql_curt_col).number_format = "0.00%"

    ## SAVE FILE TO FLASHREPORT FOLDER
    spath_ = Path(oepaths.frpath(year, month, ext="Solar"), "Comanche")
    if local:
        spath_ = Path.home().joinpath("Downloads")
    stem_ = f"Comanche_Curtailment_Report_{calendar.month_abbr[month]}-{year}"
    n_, fname_ = 1, f"{stem_}.xlsx"
    while Path(spath_, fname_).exists():
        fname_ = f"{stem_} ({n_}).xlsx"
        n_ += 1

    wb.save(Path(spath_, fname_))
    wb.close()
    qprint(f'done!\n    >> saved file: "{fname_}"')
    return
