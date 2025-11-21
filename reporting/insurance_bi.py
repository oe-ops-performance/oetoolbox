import calendar
import pandas as pd
import datetime as dt
from pathlib import Path

from ..utils.assets import SolarSite
from ..utils.oemeta import PI_SITES_BY_FLEET
from ..utils.oepaths import latest_file, shared_fp


SOLAR_OPERATIONS_DIR = latest_file(shared_fp.parent.glob("*Solar*Operations*"))
path_parts = ["SitePages", "Managers-ONLY"]
if "Site Assets" not in SOLAR_OPERATIONS_DIR.name:
    path_parts.insert(0, "Site Assets")
REFERENCE_DIR = Path(SOLAR_OPERATIONS_DIR, *path_parts)

SOLAR_PERFORMANCE_REPORT_FILEPATH = Path(REFERENCE_DIR, "Weekly-Solar-Performance-Report.xlsx")


def load_performance_report_data():
    df_dict = {}
    for key, sheet in zip(["site", "inverter"], ["Site Outages", "Inverter Outages"]):
        df = pd.read_excel(SOLAR_PERFORMANCE_REPORT_FILEPATH, sheet_name=sheet, engine="calamine")
        for date_col in ("Date Offline", "Date Restored (Est)"):
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                cond = df[date_col].astype(str).str.contains("/")
                try:
                    df.loc[cond, date_col] = pd.to_datetime(
                        df.loc[cond, date_col].str[:10], format="%m/%d/%Y"
                    )
                    df[date_col] = pd.to_datetime(df[date_col])
                except pd.errors.DateParseError as e:
                    raise e
        df_dict.update({key: df})
    return df_dict


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
    inv_id_list = []
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
        if "Inverter ID" in df.columns:
            inv_id_list.append(str(df.at[i, "Inverter ID"]).replace("\r\n", ", "))
        else:
            inv_id_list.append("")

    idx = df.columns.get_loc("Category") - 1
    start_cols = list(df.columns)[:idx]
    end_cols = list(df.columns)[idx:]

    df["n_claim_days"] = n_claim_days_list
    df["n_inv"] = n_inv_list
    df["inv_names"] = inv_id_list
    df["inverter_outage_days"] = df["n_claim_days"] * df["n_inv"]
    for c in ["n_claim_days", "n_inv", "inverter_outage_days"]:
        df[c] = df[c].astype(int)

    ordered_columns = [
        *start_cols,
        "n_claim_days",
        "n_inv",
        "inv_names",
        "inverter_outage_days",
        *end_cols,
    ]
    return df[ordered_columns]


def get_inverter_outage_days(site, start_date, end_date):
    keepcols = [
        "Date Offline",
        "Date Restored (Est)",
        "n_claim_days",
        "n_inv",
        "inv_names",
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


def get_start_and_end_date(year, month):
    start = pd.Timestamp(year, month, 1)
    end = start + pd.DateOffset(months=1)
    return map(lambda x: x.strftime("%Y-%m-%d"), [start, end])


def calculate_insurance_bi_adjustment(site_name, year, month, return_claim_and_df=False, q=True):
    qprint = lambda msg, end="\n": None if q else print(msg, end=end)
    site = SolarSite(site_name)
    start_date, end_date = get_start_and_end_date(year, month)
    qprint(f"\nsite = {site_name}\n{start_date = }\n{end_date = }")

    df_outages = get_inverter_outage_days(site.name, start_date, end_date)
    if df_outages.empty:
        qprint("\nNO OUTAGES FOUND!\n")
        if return_claim_and_df:
            return {"total_claim_mwh": 0, "df_outages": df_outages}
        return
    qprint(f"\nFound {len(df_outages)} outage records:\n{df_outages}\n")

    # function to get total possible generation for date range
    df_pvlib = site.load_data_file(year, month, file_type="pvlib")
    if df_pvlib is None:
        raise ValueError("No pvlib file found.")
    total_possible_mwh = df_pvlib.filter(like="Possible_Power").div(1e3).sum(axis=1).sum()
    if df_pvlib.index.to_series().diff().max() == pd.Timedelta(minutes=1):
        total_possible_mwh = total_possible_mwh / 60

    # get inverter days for calculation
    inv_names = site.asset_names_by_group.get("Inverters")
    if inv_names is None:
        raise ValueError("Unexpected - couldn't find inverter names.")
    n_inverters = len(inv_names)
    n_month_days = calendar.monthrange(year, month)[-1]
    total_inverter_days = n_inverters * n_month_days

    # using possible power, get the equivalent mwh per inverter day
    mwh_per_inverter_day = total_possible_mwh / total_inverter_days

    # get total inverter outage days & use to calculate total claim mwh
    n_inverter_outage_days = df_outages["inverter_outage_days"].sum()
    total_claim_mwh = mwh_per_inverter_day * n_inverter_outage_days

    qprint(
        f"{n_inverter_outage_days = }"
        f"\n{n_inverters = }"
        f"\n{n_month_days = }"
        f"\n{total_inverter_days = }"
        f"\n{mwh_per_inverter_day = :.3f}"
        f"\n\n{total_claim_mwh = :.2f}"
    )
    if return_claim_and_df:
        return {"total_claim_mwh": total_claim_mwh, "df_outages": df_outages}
    return total_claim_mwh


def get_insurance_bi_claims(year: int, month: int, sitelist: list[str] = [], q: bool = True):
    """Calculated insurance BI adjustments for specified sites; used in KPI tracker."""
    # get target sites (default all solar sites)
    target_sites = PI_SITES_BY_FLEET["solar"]
    if len(sitelist) > 0:
        valid_sites = [s for s in target_sites if s in sitelist]
        if not valid_sites:
            raise ValueError("No valid site names found in sitelist.")
        target_sites = valid_sites

    claim_list = []
    df_outages_list = []
    for site in target_sites:
        output = calculate_insurance_bi_adjustment(site, year, month, return_claim_and_df=True, q=q)
        claimed_mwh = 0  # init
        if output is not None:
            claimed_mwh = output["total_claim_mwh"]
            # get outages and add to list
            df_ = output["df_outages"].copy()
            df_.insert(0, "Site", site)
            df_outages_list.append(df_)

        # add total claim to claim list
        claim_list.append({"Site": site, "insuranceBI": claimed_mwh})

    df_outages = pd.concat(df_outages_list, axis=0, ignore_index=True)
    df_claims = pd.DataFrame(claim_list)

    return {"claims": df_claims, "outages": df_outages}
