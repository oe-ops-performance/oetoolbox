import calendar
import pandas as pd
import datetime as dt
from pathlib import Path

from ..utils.assets import SolarSite
from ..utils.oepaths import latest_file, shared_fp


SOLAR_OPERATIONS_DIR = latest_file(shared_fp.parent.glob("*Solar*Operations*"))
path_parts = ["SitePages", "Managers-ONLY"]
if "Site Assets" not in SOLAR_OPERATIONS_DIR.name:
    path_parts.insert(0, "Site Assets")
REFERENCE_DIR = Path(SOLAR_OPERATIONS_DIR, *path_parts)

SOLAR_PERFORMANCE_REPORT_FILEPATH = Path(REFERENCE_DIR, "Weekly-Solar-Performance-Report.xlsx")


def load_performance_report_data():
    return {
        key: pd.read_excel(SOLAR_PERFORMANCE_REPORT_FILEPATH, sheet_name=sheet, engine="calamine")
        for key, sheet in zip(["site", "inverter"], ["Site Outages", "Inverter Outages"])
    }


def process_claim_dates(dataframe: pd.DataFrame, site: str, start_date: str, end_date: str):
    # filter dataframe for relevant site and date range
    start, end = map(pd.Timestamp, [start_date, end_date])
    target_dates = pd.date_range(
        start, end, inclusive="left"
    )  # dates for which to determine insurance adjustment
    c1 = dataframe["Site Name"] == site
    c2 = (dataframe["Date Restored (Est)"] >= start) | dataframe["Date Restored (Est)"].isna()
    c3 = dataframe["Date Offline"] < start
    df = dataframe.loc[(c1 & c2 & c3)].reset_index(drop=True).copy()

    df["Date Offline"] = df["Date Offline"].dt.floor("D")
    df["Date Restored (Est)"] = df["Date Restored (Est)"].dt.ceil("D")

    n_claim_days_list = []
    n_inv_list = []
    for i in range(len(df)):
        # can claim 1 month after outage start
        claim_begin = df.at[i, "Date Offline"] + pd.DateOffset(months=1)
        claim_end = df.at[i, "Date Restored (Est)"]
        if pd.isna(claim_end):
            claim_end = target_dates[-1]
        claim_range = pd.date_range(claim_begin, claim_end)
        relevant_days = [d for d in target_dates if d in claim_range]  # overlapping days
        n_claim_days_list.append(len(relevant_days))

        n_inverters = 1 if "Inverter ID" in df.columns else df.at[i, "# of Inv Offline"]
        n_inv_list.append(n_inverters)

    idx = df.columns.get_loc("Category") - 1
    start_cols = list(df.columns)[:idx]
    end_cols = list(df.columns)[idx:]

    df["n_claim_days"] = n_claim_days_list
    df["n_inv"] = n_inv_list
    df["inverter_outage_days"] = df["n_claim_days"] * df["n_inv"]
    for c in ["n_claim_days", "n_inv", "inverter_outage_days"]:
        df[c] = df[c].astype(int)

    ordered_columns = [*start_cols, "n_claim_days", "n_inv", "inverter_outage_days", *end_cols]
    return df[ordered_columns]


def get_inverter_outage_days(site, start_date, end_date):
    keepcols = [
        "Date Offline",
        "Date Restored (Est)",
        "n_claim_days",
        "n_inv",
        "inverter_outage_days",
    ]
    df_list = []
    df_dict = load_performance_report_data()
    for df_ in df_dict.values():
        dff = process_claim_dates(df_, site, start_date, end_date)
        df_list.append(dff[keepcols].copy())
    df = pd.concat(df_list, axis=0, ignore_index=True)
    df = df.sort_values(by=["n_inv", "Date Offline"], ascending=False).reset_index(drop=True)
    return df


# def get_total_inverter_days(site, start_date, end_date):
