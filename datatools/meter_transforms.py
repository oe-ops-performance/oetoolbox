import datetime as dt
import pandas as pd
from pathlib import Path
import xmltodict

from ..utils import oemeta

get_timezone = lambda site: oemeta.data["TZ"].get(site)

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


def formatted_dataframe(df, freq="h"):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Need datetime index to create Day/Hour columns.")

    sitecols = [c for c in df.columns if c in VARSITY_STLMTUI_SITE_IDS]
    keepcols = ["MWh"] if not sitecols else sitecols
    if not any(col in df.columns for col in keepcols):
        raise ValueError("Invalid column names - missing MWh column or stlmtui site columns.")

    df = df[keepcols].copy()

    if freq != "h":
        df = df.resample("h").sum()  # already in MWh, so sum instead of mean

    df["Day"] = df.index.strftime("%Y-%m-%d")
    df["Hour"] = df.index.hour + 1
    new_cols = ["Day", "Hour"] + keepcols
    df = df[new_cols].reset_index(drop=True).copy()

    if not pd.api.types.is_datetime64_any_dtype(df["Day"]):
        df["Day"] = pd.to_datetime(df["Day"])
    if not pd.api.types.is_integer_dtype(df["Hour"]):
        df["Hour"] = df["Hour"].astype(int)

    for col in keepcols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col])  # , errors="coerce")
        df[col] = df[col].fillna(0)

    return df


def validated_datetime_index(df, site, localized=True, freq="h"):
    """returns dataframe with continuous datetime index"""
    if not isinstance(df.index, pd.DatetimeIndex):
        if not all(c in df.columns for c in ["Day", "Hour"]):
            raise ValueError("dataframe must have datetime index")
        df.index = pd.to_datetime(
            df["Day"].astype(str) + " " + df["Hour"].sub(1).astype(str).str.zfill(2) + ":00:00",
            format="%Y-%m-%d %H:%M:%S",
        )
    start, end = df.index.min(), df.index.max()
    tz = oemeta.data["TZ"].get(site)
    idx_kwargs = dict(start=start, end=end, freq=freq)
    expected_index = pd.date_range(**idx_kwargs)
    expected_local_index = pd.date_range(**idx_kwargs, tz=tz)
    n_original, n_expected = df.shape[0], len(expected_local_index)

    if localized:
        if df.index.tz is not None:
            if str(df.index.tz) != tz:
                raise ValueError("index timezone does not match specified site tz")
            if n_original != n_expected:
                df = df.reindex(expected_local_index)
            return df
        elif n_original != n_expected and n_original != len(expected_index):
            df = df.reindex(expected_index)

        n_actual = df.shape[0]

        if n_actual == n_expected:
            df = df.tz_localize(tz=tz, ambiguous="NaT", nonexistent="NaT")
            df.index = expected_local_index
            return df

        if abs(n_actual - n_expected) == 1:
            df = df.tz_localize(tz=tz, ambiguous="NaT", nonexistent="NaT")
            if n_actual > n_expected:
                df = df.rename_axis("tstamp").reset_index(drop=False)
                missing_index = df.loc[df["tstamp"].isna()].index[0]
                df.at[missing_index - 1, "MWh"] = df.at[missing_index, "MWh"]
                df = df.set_index("tstamp").rename_axis(None)
            else:  # fall dst, only applies to solar (wind data includes extra hour)
                if "MWh" in df.columns:
                    df["MWh"] = df["MWh"].fillna(0)
                else:
                    for c in df.columns:  # stlmtui
                        df[c] = df[c].fillna(0)
            df = df.reindex(expected_local_index)  # leaves "NaT" values in index
            df.index = expected_local_index  # overwrites NaT index values
            return df

        df["local_index"] = df.index.tz_localize(tz=tz, ambiguous="NaT", nonexistent="NaT")
        df = df.reset_index(drop=True)
        repeated_tstamp_indexes = df.loc[df["local_index"].isna()].index
        df = pd.concat(
            [
                df.iloc[: repeated_tstamp_indexes[-1] + 1, :].copy(),
                df.iloc[repeated_tstamp_indexes[0] :, :].copy(),
            ],
            axis=0,
            ignore_index=True,
        )
        df.index = expected_local_index
        return df

    else:
        if df.index.tz is not None:
            df = df.tz_localize(None)
        if df.index.duplicated().any():
            df = df.loc[~df.index.duplicated(keep="first")].copy()
        if df.shape[0] < len(expected_index):
            df = df.reindex(expected_index)

    return df


def expected_datetime_index(year, month, freq, tz=None):
    start = pd.Timestamp(year, month, 1)
    end = start + pd.DateOffset(months=1)
    return pd.date_range(start, end, freq=freq, tz=tz)[:-1]


def date_cols_to_index(df, drop=False):
    if not all(c in df.columns for c in ["Day", "Hour"]):
        raise KeyError("Invalid columns.")
    df.index = pd.to_datetime(
        df["Day"].astype(str) + " " + df["Hour"].sub(1).astype(str).str.zfill(2) + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
    )
    if drop:
        df = df.drop(columns=["Day", "Hour"])
    return df


# all functions return dataframe with "Day", "Hour", "MWh" columns
# fall DST - extra hour is included for all sites
# spring DST - missing hour is added back to df with value = 0

wind_date_columns = {
    "Bingham": "ALL OTHER DAYS",
    "Hancock": "Day",
    "Oakfield": "Day",
    "Palouse": "Day",
    "Route 66": "Flowday",
    "South Plains II": "Flowday",
}


def load_wind_data_1(filepath, site, localized=True):
    tz = get_timezone(site)
    datecol = wind_date_columns[site]

    if "xls" in Path(filepath).suffix:
        df_ = pd.read_excel(filepath, engine="calamine")
    else:
        df_ = pd.read_csv(filepath)
    header_index = df_[df_.isin([datecol]).any(axis=1)].index[0]
    df_.columns = df_.loc[header_index].values

    # filter rows for valid dates
    valid_rows = pd.notna(pd.to_datetime(df_[datecol], errors="coerce", format="%Y-%m-%d"))
    df_ = df_.loc[valid_rows].copy()
    df_ = df_.set_index(pd.to_datetime(df_[datecol], format="%Y-%m-%d"))

    # filter for columns with hour numbers
    get_digits = lambda c: "".join(filter(str.isdigit, c)) if isinstance(c, str) else str(int(c))
    valid_col = lambda c: not pd.isna(pd.to_numeric(get_digits(c), errors="coerce"))
    valid_columns = list(filter(valid_col, df_.columns))
    df_ = df_[valid_columns].apply(pd.to_numeric, errors="coerce").copy()
    df_.columns = df_.columns.map(lambda c: int(get_digits(c)))

    df = pd.melt(df_.T, ignore_index=False, var_name="Day", value_name="MWh")
    non_dst_hours = df.index.isin(range(1, 25))
    valid_dst_hours = (df.index == 25) & df["MWh"].gt(0)
    df = df.loc[non_dst_hours | valid_dst_hours].copy()

    df = df.rename_axis("Hour").reset_index(drop=False)

    if valid_dst_hours.any():
        start = df.Day.min()
        end = start + pd.DateOffset(months=1) - pd.Timedelta(hours=1)
        df.index = pd.date_range(start, end, freq="h", tz=tz)
        df = df.drop(columns="Hour")
        df["Hour"] = df.index.hour + 1

    else:
        df.index = pd.to_datetime(
            df["Day"].astype(str) + " " + df["Hour"].sub(1).astype(str).str.zfill(2) + ":00:00",
            format="%Y-%m-%d %H:%M:%S",
        )

    df = validated_datetime_index(df, site, localized)

    return df


def load_bingham(filepath, fmt=True, localized=True):
    df = load_wind_data_1(filepath, site="Bingham", localized=localized)
    return formatted_dataframe(df) if fmt else df


def load_hancock(filepath, fmt=True, localized=True):
    df = load_wind_data_1(filepath, site="Hancock", localized=localized)
    return formatted_dataframe(df) if fmt else df


def load_oakfield(filepath, fmt=True, localized=True):
    df = load_wind_data_1(filepath, site="Oakfield", localized=localized)
    return formatted_dataframe(df) if fmt else df


def load_palouse(filepath, fmt=True, localized=True):
    df = load_wind_data_1(filepath, site="Palouse", localized=localized)
    return formatted_dataframe(df) if fmt else df


site_keys = {
    "Route 66": ["rt66", "route66"],
    "South Plains II": ["splain", "southplain"],
}


def load_wind_data_2(filepath, site, localized=True):
    tz = get_timezone(site)
    datecol = wind_date_columns[site]

    df_ = pd.read_excel(filepath, engine="calamine")
    header_index = df_[df_.isin([datecol]).any(axis=1)].index[0]
    df_.columns = df_.loc[header_index].values

    # check row above header_index to find site cols (in case multiple sites exist in file)
    checkrow = list(df_.loc[header_index - 1].values)
    keys = site_keys[site]
    start_col = [
        i for i, c in enumerate(checkrow) if any(k in str(c).casefold() for k in keys)
    ].pop(0)
    col_index_list = [0, 1] + list(range(start_col, df_.shape[1]))  # last col is $, don't need

    df_ = df_.iloc[header_index + 1 :, col_index_list].copy()
    df_[datecol] = pd.to_datetime(df_[datecol], errors="coerce")

    df_ = df_.loc[df_[datecol].notna()].reset_index(drop=True).copy()

    is_gen_col = lambda col: all(x in col.casefold() for x in ["quantity", "mwh"])
    mwh_col_index = [i for i, c in enumerate(df_.columns) if is_gen_col(c)].pop()

    df = df_.iloc[:, [0, 1, mwh_col_index]].copy()
    df.columns = ["Day", "IE", "MWh"]
    df["IE"] = df["IE"].astype(str).shift(1, fill_value="00:00").str.replace("24", "00")

    year, month = df["Day"][0].year, df["Day"][0].month
    expected_local_index = expected_datetime_index(year, month, freq="15min", tz=tz)
    if len(expected_local_index) == df.shape[0]:
        df = df[["MWh"]].set_index(expected_local_index).copy()
    else:
        df.index = pd.to_datetime(
            df["Day"].astype(str) + " " + df["IE"] + ":00",
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
        df = df[["MWh"]].copy()

    df = df.resample("h").sum()

    df = validated_datetime_index(df, site, localized)

    return df


def load_route_66(filepath, fmt=True, localized=True):
    df = load_wind_data_2(filepath, site="Route 66", localized=localized)
    return formatted_dataframe(df) if fmt else df


def load_south_plains_ii(filepath, fmt=True, localized=True):
    df = load_wind_data_2(filepath, site="South Plains II", localized=localized)
    return formatted_dataframe(df) if fmt else df


def load_sunflower(filepath, fmt=True, localized=True):
    tz = get_timezone("Sunflower")
    df_ = pd.read_table(filepath)
    if df_.shape[1] > 1:
        df_.columns = df_.columns.map(lambda c: c.strip().replace(" ", ""))
    else:
        df_ = df_.dropna()
        cols_ = df_.columns[0].split(",")
        columns = ["".join(filter(str.isalpha, col)) for col in cols_]
        df_ = df_[df_.columns[0]].str.split(",", expand=True).copy()
        df_.columns = [c.strip().replace(" ", "") for c in columns]

    df_["DATE"] = df_["DATE"].astype(str).str.zfill(6)
    df_["HOUR"] = df_["HOUR"].astype(str).str.zfill(4)

    df = df_.iloc[:, 1:].copy()
    uniquecols = []
    for c in df.columns:
        n, col = 1, c
        while col in uniquecols:
            col = f"{c}.{n}"
            n += 1
        uniquecols.append(col)

    df.columns = uniquecols
    kw_col = [c for c in uniquecols if all(i in c for i in ["KW", "1"])].pop(0)
    df["MWh"] = pd.to_numeric(df[kw_col]).div(1e3)
    df["Hour"] = pd.to_numeric(df["HOUR"]).div(100).astype(int)
    df["Day"] = pd.to_datetime(df["DATE"], format="%m%d%y")

    year, month = df["Day"][0].year, df["Day"][0].month
    expected_local_index = expected_datetime_index(year, month, freq="h", tz=tz)
    if len(expected_local_index) == df.shape[0]:
        df = df.set_index(expected_local_index)

    df = validated_datetime_index(df, site="Sunflower", localized=localized)

    return formatted_dataframe(df) if fmt else df


# solar


def load_pi_meter_data(fpath):
    df = pd.read_csv(fpath, index_col=0, parse_dates=True)
    df = df.rename(columns={df.columns[0]: "MWh"})
    df = df.resample("h").mean().copy()
    return df


def load_cw_marin(fpath, fmt=True):
    site = "CW-Marin"
    df = load_pi_meter_data(fpath)
    df = validated_datetime_index(df, site)
    return formatted_dataframe(df) if fmt else df


def load_indy_i(fpath, fmt=True):
    site = "Indy I"
    df = load_pi_meter_data(fpath)
    df = validated_datetime_index(df, site)
    return formatted_dataframe(df) if fmt else df


def load_indy_ii(fpath, fmt=True):
    site = "Indy II"
    df = load_pi_meter_data(fpath)
    df = validated_datetime_index(df, site)
    return formatted_dataframe(df) if fmt else df


def load_indy_iii(fpath, fmt=True):
    site = "Indy III"
    df = load_pi_meter_data(fpath)
    df = validated_datetime_index(df, site)
    return formatted_dataframe(df) if fmt else df


def load_comanche(fpath, fmt=True):
    site = "Comanche"
    is_csv = Path(fpath).suffix == ".csv"
    pandas_read_function = pd.read_csv if is_csv else pd.read_excel
    date_col = [c for c in pandas_read_function(fpath, nrows=0).columns if "DATE" in c].pop(0)
    kwargs = dict(dtype={date_col: str})
    if not is_csv:
        kwargs.update(dict(engine="calamine"))
    df = pandas_read_function(fpath, **kwargs)
    hour_col = [c for c in df.columns if "HOUR" in c].pop(0)
    df = df.rename(columns={date_col: "Day", hour_col: "Hour"})

    end_idx = df["Day"].last_valid_index() + 1
    df = df.iloc[:end_idx, :].copy()

    gen_col = [c for c in df.columns if all(i in c for i in ["KW", "1"])].pop(0)
    df["MWh"] = pd.to_numeric(df[gen_col]).div(1e3)

    df = df[["Day", "Hour", "MWh"]].copy()
    df["Day"] = pd.to_datetime(df["Day"].str.zfill(6), format="%m%d%y")
    df["Hour"] = pd.to_numeric(df["Hour"]).div(100).astype(int)
    df.index = pd.to_datetime(
        df["Day"].astype(str) + " " + df["Hour"].sub(1).astype(str).str.zfill(2) + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
    )
    df = validated_datetime_index(df, site)
    return formatted_dataframe(df) if fmt else df


def load_az1(fpath, fmt=True):
    site = "AZ1"
    df_dict = pd.read_excel(fpath, engine="calamine", sheet_name=None)
    target_sheet = [s for s, df_ in df_dict.items() if (df_.shape[0] > 20)].pop(0)
    df = df_dict[target_sheet]
    hdr = df[df.isin(["End Date"]).any(axis=1)].index[0]  # find header row
    df.columns = df.iloc[hdr].values
    keepcols = [c for c in df.columns if any(i in c for i in ["Date", "HE"])]
    df = df[keepcols].iloc[hdr + 1 :, :].reset_index(drop=True).copy()
    df["End Date"] = pd.to_datetime(df["End Date"])
    end_idx = df["End Date"].last_valid_index()
    df = df.iloc[: end_idx + 1, :].set_index("End Date").copy()
    df = df.T.reset_index(drop=True).rename_axis(None, axis=1).copy()
    df = df.melt(var_name="Date", value_name="MWh", ignore_index=False).copy()
    df.index = pd.to_datetime(
        df["Date"].astype(str) + " " + df.index.astype(str).str.zfill(2) + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
    )
    df = validated_datetime_index(df, "CID")  # note: using CID b/c AZ1 does not follow DST
    return formatted_dataframe(df) if fmt else df


def load_ga(fpath, site, fmt=True):
    df = pd.read_excel(fpath, engine="calamine", header=5)
    df.columns = ["Timestamp", "MWh"]
    df = df.set_index("Timestamp")
    df.index = pd.to_datetime(df.index)
    df = validated_datetime_index(df, site)
    return formatted_dataframe(df) if fmt else df


def load_ga3(fpath, fmt=True):
    return load_ga(fpath, site="GA3", fmt=fmt)


def load_ga4(fpath, fmt=True):
    return load_ga(fpath, site="GA4", fmt=fmt)


def load_grand_view(fpath, site, fmt=True):
    meter_id = "GVSP" if "East" in site else "GVSR"
    df_dict = pd.read_excel(fpath, engine="calamine", sheet_name=None, nrows=4)
    matching_sheets = [
        sheet
        for sheet, df_ in df_dict.items()
        if any(meter_id in v for v in df_.iloc[:, 0].astype(str).values)
    ]
    if not matching_sheets:
        print("Sheet not found.")
        return
    sheet_name = matching_sheets[0]
    if meter_id not in sheet_name:
        if "Nov 24" not in fpath.name:  # known exception
            raise ValueError("Note: sheet names in file do not match the data they contain")

    df_ = pd.read_excel(fpath, engine="calamine", sheet_name=sheet_name)
    hdr_idx = df_[df_.isin(["Date"]).any(axis=1)].index[0]
    df_.columns = df_.iloc[hdr_idx, :].astype(str).values
    end_row = df_["Date"].last_valid_index() + 1
    end_col = list(df_.columns).index("Total")
    df_ = df_.iloc[hdr_idx + 1 : end_row, :end_col].reset_index(drop=True).copy()
    df_.index = pd.to_datetime(df_["Date"])
    df_ = df_.drop(columns="Date")
    dfT = df_.transpose().reset_index(drop=True).rename_axis(None, axis=1)
    df = dfT.melt(var_name="Date", value_name="MWh", ignore_index=False)
    df["MWh"] = df["MWh"].div(1000)
    df.index = pd.to_datetime(
        df["Date"].astype(str) + " " + df.index.astype(str).str.zfill(2) + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
    )
    df = validated_datetime_index(df, site)
    return formatted_dataframe(df) if fmt else df


def load_grand_view_east(fpath, fmt=True):
    df = load_grand_view(fpath, site="Grand View East", fmt=fmt)
    return df


def load_grand_view_west(fpath, fmt=True):
    df = load_grand_view(fpath, site="Grand View West", fmt=fmt)
    return df


def load_ms3(fpath, fmt=True):
    site = "MS3"
    # note: the "astype(str)" prevents error when column names are formatted as datetime
    dcol = [
        c
        for c in pd.read_excel(fpath, engine="calamine", nrows=0).columns.astype(str)
        if "DATE" in c
    ][0]
    df_ = pd.read_excel(fpath, engine="calamine", dtype={dcol: str})
    df_[dcol] = df_[dcol].str.zfill(6)
    df_[dcol] = pd.to_datetime(df_[dcol], format="%m%d%y")
    df_ = df_.set_index(dcol)
    is_hour_col = lambda c: (type(c) in [dt.datetime, dt.time]) or str(c).strip().endswith(
        "0:00:00"
    )
    hour_cols = list(filter(is_hour_col, df_.columns))
    df_ = df_[hour_cols].copy()
    df_.columns = [c.hour for c in hour_cols[:-1]] + [24]
    endrow_ = df_.reset_index().last_valid_index()
    df_ = df_.iloc[: endrow_ + 1, :].copy()
    df_ = df_.transpose().rename_axis(None, axis=1)
    df_ = df_.reset_index(drop=True)
    df = df_.melt(var_name="Date", value_name="MWh", ignore_index=False).copy()
    df["Timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df.index.astype(str).str.zfill(2) + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
    )
    df = df.set_index("Timestamp")
    df = validated_datetime_index(df, site)
    return formatted_dataframe(df) if fmt else df


def load_maplewood_1(fpath, fmt=True):
    site = "Maplewood 1"
    df = pd.read_excel(fpath, engine="calamine")
    datecol = [c for c in df.columns if "flowday" in c.casefold()][0]
    keepcols_ = [datecol, "Interval", "GEN_MWH_INT"]
    df = df[keepcols_].copy()
    df["Interval"] = df["Interval"].replace("24:00", "00:00")
    df["Interval"] = df["Interval"].shift(1, fill_value="00:00")
    if df.Interval.str.len().gt(5).any():
        df = df.loc[df.Interval.str.len().eq(5)].reset_index(drop=True)
    last_row = df[datecol].last_valid_index()
    df = df.iloc[: last_row + 1, :].copy()
    df.index = pd.to_datetime(df[datecol].astype(str) + " " + df.Interval)
    df = df.rename(columns={"GEN_MWH_INT": "MWh"})

    df = validated_datetime_index(df, site, freq="15min")
    return formatted_dataframe(df, freq="15min") if fmt else df


def load_maplewood_2(fpath, fmt=True):
    site = "Maplewood 2"
    df = pd.read_excel(fpath, engine="calamine")
    datecol = [c for c in df.columns if "flowday" in c.casefold()][0]
    keepcols_ = [datecol, "Interval", "RTEI_QTY"]
    df = df[keepcols_].copy()
    df["Interval"] = df["Interval"].replace("24:00", "00:00")
    df["Interval"] = df["Interval"].shift(1, fill_value="00:00")
    if df.Interval.str.len().gt(5).any():
        df = df.loc[df.Interval.str.len().eq(5)].reset_index(drop=True)
    last_row = df[datecol].last_valid_index()
    df = df.iloc[: last_row + 1, :].copy()
    df.index = pd.to_datetime(df[datecol].astype(str) + " " + df.Interval)
    df = df.rename(columns={"RTEI_QTY": "MWh"})
    df["MWh"] = df["MWh"].mul(-1)

    df = validated_datetime_index(df, site, freq="15min")
    return formatted_dataframe(df, freq="15min") if fmt else df


def load_sweetwater(fpath, fmt=True):
    site = "Sweetwater"
    df = pd.read_excel(fpath, engine="calamine", sheet_name="HourlyData")
    df.index = pd.to_datetime(
        df["TradingDate"].astype(str)
        + " "
        + df["TradingHour"].sub(1).astype(str).str.zfill(2)
        + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
    )
    df["MWh"] = df["Meter_KW"].div(1000).copy()
    df = validated_datetime_index(df, site)
    return formatted_dataframe(df) if fmt else df


def load_three_peaks(fpath, fmt=True):
    site = "Three Peaks"
    df = pd.read_excel(fpath, engine="calamine", sheet_name="HourlyData")
    df.index = pd.to_datetime(
        df["TradingDate"].astype(str)
        + " "
        + df["TradingHour"].sub(1).astype(str).str.zfill(2)
        + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
    )
    df["MWh"] = df["Meter_KW"].div(1000).copy()
    df = validated_datetime_index(df, site)
    return formatted_dataframe(df) if fmt else df


# varsity (individual files)


def load_azalea(fpath, fmt=True):
    site = "Azalea"
    df = pd.read_excel(fpath, engine="calamine")
    df = df.rename(columns={"CT Date": "Day", "CT Hour": "Hour", "Scheduled": "MWh"})
    df = df.loc[df.Hour.isin(range(1, 25))].reset_index(drop=True)
    dtFormat = "%m/%d/%Y %H:%M:%S" if ("/" in str(df.at[0, "Day"])) else "%Y-%m-%d %H:%M:%S"
    df.index = pd.to_datetime(
        df["Day"].astype(str) + " " + df["Hour"].sub(1).astype(str).str.zfill(2) + ":00:00",
        format=dtFormat,
    )
    df["MWh"] = df["MWh"].shift(
        periods=1, axis=0, fill_value=0
    )  # added to shift meter data by 1 hour
    df = validated_datetime_index(df, site)
    return formatted_dataframe(df) if fmt else df


def load_imperial_valley(fpath, fmt=True):
    site = "Imperial Valley"
    df_ = pd.read_excel(fpath, engine="calamine")
    hdr_idx = df_[df_.isin(["NAME"]).any(axis=1)].index[0]
    df_.columns = df_.iloc[hdr_idx, :].values
    matching_cols = lambda c: any(i in c for i in ["Date", "HE"]) and ("*" not in c)
    keepcols = list(filter(matching_cols, df_.columns))
    date_col = keepcols[0]
    end_idx = df_[date_col].last_valid_index()
    dfT_ = df_[keepcols].set_index(date_col).iloc[hdr_idx + 1 : end_idx + 1, :]
    dfT_.index = pd.to_datetime(dfT_.index)
    dfT = dfT_.transpose().reset_index(drop=True).rename_axis(None, axis=1)
    df = dfT.melt(var_name="Date", value_name="MWh", ignore_index=False)
    df.index = pd.to_datetime(
        df["Date"].astype(str) + " " + df.index.astype(str).str.zfill(2) + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
    )
    df = validated_datetime_index(df, site)
    return formatted_dataframe(df) if fmt else df


def load_mulberry(fpath, fmt=True):
    site = "Mulberry"
    df = pd.read_excel(fpath, engine="calamine", header=5)
    df = df.loc[df.Hour.isin(range(1, 25))]
    df.index = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Hour"].sub(1).astype(str).str.zfill(2) + ":00:00",
        format="%m/%d/%Y %H:%M:%S",
    )
    df["MWh"] = df["kWh"].div(1000).copy()
    df = validated_datetime_index(df, site)
    return formatted_dataframe(df) if fmt else df


def load_pavant(fpath, fmt=True):
    site = "Pavant"
    df = pd.read_excel(fpath, engine="calamine", sheet_name="HourlyData")
    df.index = pd.to_datetime(
        df["TradingDate"].astype(str)
        + " "
        + df["TradingHour"].sub(1).astype(str).str.zfill(2)
        + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
    )
    df["MWh"] = df["Meter_KW"].div(1000).copy()
    df = validated_datetime_index(df, site)
    return formatted_dataframe(df) if fmt else df


def load_richland(fpath, fmt=True):
    site = "Richland"
    df = pd.read_excel(fpath, engine="calamine")
    hdr_idx = df[df.isin(["HourBegin"]).any(axis=1)].index[0]
    df.columns = df.iloc[hdr_idx, :].values
    df = df.rename(columns={"Generation_MW": "MWh"})
    df = df.iloc[hdr_idx + 1 :, :].reset_index(drop=True)
    if df["HourBegin"].last_valid_index() != df.shape[0]:  # catches last line ("total")
        df = df.iloc[:-1, :].copy()
    df = df.set_index("HourBegin")
    df.index = pd.to_datetime(df.index)
    df = validated_datetime_index(df, site)
    return formatted_dataframe(df) if fmt else df


def load_selmer(fpath, fmt=True):
    site = "Selmer"
    df = pd.read_excel(fpath, engine="calamine", header=5)
    df = df.loc[df.Hour.isin(range(1, 25))]
    df.index = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Hour"].sub(1).astype(str).str.zfill(2) + ":00:00",
        format="%m/%d/%Y %H:%M:%S",
    )
    df["MWh"] = df["kWh"].div(1000).copy()
    df = validated_datetime_index(df, site)
    return formatted_dataframe(df) if fmt else df


def load_somers(fpath, fmt=True):
    site = "Somers"
    df = pd.read_csv(
        fpath, header=6
    )  # NOTE: setting header to anything below 6 causes parsing error
    if "DATE" not in df.columns:  # checking if header was 7
        hdr = df[df.isin(["DATE"]).any(axis=1)].index[0]
        df.columns = df.iloc[hdr].values
        df = df.iloc[hdr + 1 :, :].reset_index(drop=True)
    df = df.dropna(axis=1, how="all")
    uniquecols = []
    for c in df.columns:
        n, col = 1, c
        while col in uniquecols:
            col = f"{c}.{n}"
        uniquecols.append(col)
    df.columns = uniquecols

    df = df.shift(1)  # file is always missing first timestamp
    df.loc[0, "DATE"] = df.loc[1, "DATE"]
    df.loc[0, "TIME"] = "00:00"
    df.index = pd.to_datetime(
        df["DATE"].astype(str) + " " + df["TIME"] + ":00",
        format="%m/%d/%Y %H:%M:%S",
    )
    df["kWh.1"] = pd.to_numeric(df["kWh.1"])
    df["MWh"] = df["kWh.1"].div(1000).copy()
    df = validated_datetime_index(df, site)
    return formatted_dataframe(df) if fmt else df


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


def load_stlmtui_excel_file(fpath):
    sheets_ = list(pd.read_excel(fpath, engine="calamine", sheet_name=None, nrows=0).keys())
    sheet_name = [s for s in sheets_ if "Meter Data" in s].pop(0)
    return pd.read_excel(fpath, engine="calamine", header=1, sheet_name=sheet_name)


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
    return df


def load_stlmtui(filepath, fmt=True):
    try:
        df = parse_stlmtui_file(filepath)
    except:
        df = load_stlmtui_excel_file(filepath)
    df = format_stlmtui_data(df)
    df = validated_datetime_index(df, "Alamo")  # all same tz
    return formatted_dataframe(df) if fmt else df
