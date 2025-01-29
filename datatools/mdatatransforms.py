import pandas as pd
import datetime as dt


# function to get expected datetime index based off of df index min/max (for DST comparisons)
def expected_index(df, freq="h"):
    # note: df is expected to contain monthly data @hourly intervals unless otherwise specified
    year = df.index.min().year
    month = df.index.min().month
    t0 = pd.Timestamp(year=year, month=month, day=1)
    t1 = t0 + pd.DateOffset(months=1)
    return pd.date_range(t0, t1, freq=freq)[:-1]


# function to format dataframe for meter gen historian
def format_meter_dataframe(df_, freq="h"):
    """input df should have datetime index & a "MWh" column"""
    df = df_[["MWh"]].copy()
    expected_idx = expected_index(df, freq=freq)
    if df.index.duplicated().any():
        df = df.loc[~df.index.duplicated(keep="first")]
    if df.shape[0] != expected_idx.shape[0]:
        df = df.reindex(expected_idx)
    if df["MWh"].dtype != float:
        df["MWh"] = pd.to_numeric(df["MWh"]).astype(float)
    if freq != "h":
        df = df.resample("h").sum()
    df["Day"] = pd.to_datetime(df.index.date)
    df["Hour"] = df.index.hour + 1
    df = df[["Day", "Hour", "MWh"]]
    df = df.reset_index(drop=True)

    ## new - requested 11/5/24 - replace NaN values with zeros
    df["MWh"] = df["MWh"].fillna(0)

    return df


"""
WIND
"""


def load_bingham(fpath):
    df_ = pd.read_excel(fpath)
    idx_ = df_[df_.isin(["ALL OTHER DAYS"]).any(axis=1)].index[0]
    df_.columns = df_.iloc[idx_].values
    df_ = df_.iloc[idx_ + 1 :, :]
    df_["ALL OTHER DAYS"] = pd.to_datetime(df_["ALL OTHER DAYS"])
    df_ = df_.set_index("ALL OTHER DAYS").rename_axis(None)
    df_ = df_.iloc[:, :24].apply(pd.to_numeric)  # note: this drops the extra DST hour in Fall
    df_.columns = [str(c).replace(".0", "") for c in df_.columns]

    # transpose/melt
    dfT = df_.transpose().copy()
    df = pd.melt(dfT, ignore_index=False, var_name="Day", value_name="MWh")
    df = df.rename_axis("Hour").reset_index(drop=False)
    df["Hour"] = df["Hour"].astype(int)
    df.index = pd.to_datetime(
        df["Day"].astype(str) + " " + df["Hour"].sub(1).astype(str).str.zfill(2) + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
    )
    df = format_meter_dataframe(df)
    return df


def load_hancock(fpath):
    df_ = pd.read_excel(fpath)
    idx_ = df_[df_.isin(["Day"]).any(axis=1)].index[0]
    df_.columns = df_.iloc[idx_].values
    df_ = df_.iloc[idx_ + 1 : -1, :]  # remove last row "Totals"
    df_["Day"] = pd.to_datetime(df_["Day"])
    df_ = df_.set_index("Day").rename_axis(None)
    df_ = df_.iloc[:, :24]  # note: this drops the extra DST hour in Fall
    df_.columns = [c.replace("Hour ", "") for c in df_.columns]

    # transpose/melt
    dfT = df_.transpose().copy()
    df = pd.melt(dfT, ignore_index=False, var_name="Day", value_name="MWh")
    df = df.rename_axis("Hour").reset_index(drop=False)
    df["Hour"] = df["Hour"].astype(int)
    df.index = pd.to_datetime(
        df["Day"].astype(str) + " " + df["Hour"].sub(1).astype(str).str.zfill(2) + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
    )
    df = format_meter_dataframe(df)
    return df


def load_oakfield(fpath):
    return load_hancock(fpath)


def load_palouse(fpath):
    df_ = pd.read_excel(fpath)
    idx_ = df_[df_.isin(["Day"]).any(axis=1)].index[0]
    df_.columns = df_.iloc[idx_].values
    keepcols = [c for c in df_.columns if any(i in c for i in ["Day", "He"])]
    lastrow = df_["Day"].last_valid_index()
    df_ = df_[keepcols].iloc[idx_ + 1 : lastrow + 1, :].reset_index(drop=True)
    df_["Day"] = pd.to_datetime(df_["Day"])
    df_ = df_.set_index("Day").rename_axis(None)
    df_ = df_.iloc[:, :24]
    df_.columns = [c.replace("He", "").strip() for c in df_.columns]
    # transpose/melt
    dfT = df_.transpose().copy()
    df = pd.melt(dfT, ignore_index=False, var_name="Day", value_name="MWh")
    df = df.rename_axis("Hour").reset_index(drop=False)
    df["Hour"] = df["Hour"].astype(int)
    df.index = pd.to_datetime(
        df["Day"].astype(str) + " " + df["Hour"].sub(1).astype(str).str.zfill(2) + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
    )
    df = format_meter_dataframe(df)
    return df


def load_rt66_or_sp2(fpath, s_keys, return_localized=False):
    df_ = pd.read_excel(fpath)
    idx_ = df_[df_.isin(["Flowday"]).any(axis=1)].index[0]
    ##check row above idx_ to find site cols (in case multiple sites exist in file)
    checkrow = list(df_.loc[idx_ - 1].values)
    matching_ = [
        (i, val) for i, val in enumerate(checkrow) if any(k in str(val).casefold() for k in s_keys)
    ]
    df_.columns = df_.iloc[idx_].values
    df_ = df_.iloc[idx_ + 1 :, :].reset_index(drop=True)
    df_["Flowday"] = pd.to_datetime(df_["Flowday"], errors="coerce")
    rw_start = df_["Flowday"].first_valid_index()
    rw_end = df_["Flowday"].last_valid_index() + 1
    col_start = matching_[0][0]
    col_idx_list = [0, 1] + [i for i in range(col_start, df_.shape[1])]

    df = df_.iloc[rw_start:rw_end, col_idx_list].reset_index(drop=True).copy()

    ##find target cols & convert dtypes
    datecols = list(df.columns)[:2]
    isgencol = lambda c: all(k in c.casefold() for k in ["quantity", "mwh"])
    gen_cols = [c for c in df.columns if isgencol(c)]
    keepcols = datecols + [gen_cols[0]]
    df = df[keepcols]
    df.columns = ["Day", "IE", "MWh"]
    df["MWh"] = pd.to_numeric(df["MWh"])

    if return_localized:
        start_, end_ = df.Day.min(), df.Day.max() + pd.DateOffset(days=1)
        local_start, local_end = map(lambda t: t.tz_localize(tz="US/Central"), [start_, end_])
        expected_local_index = pd.date_range(local_start, local_end, freq="15min")[:-1]
        if df.shape[0] != expected_local_index.shape[0]:
            print("Error: check datetime index")
            return None
        df = df.set_index(expected_local_index)
        df = df[["MWh"]].resample("h").sum()
    else:
        df["IE"] = df["IE"].astype(str).shift(1, fill_value="00:00").str.replace("24", "00")
        df.index = pd.to_datetime(
            df["Day"].astype(str) + " " + df["IE"] + ":00",
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
        df = df.loc[~df.index.isna()]  # drops extra dst hour for fall
        df = format_meter_dataframe(df, freq="15min")
    return df


def load_route_66(fpath, localized=False):
    keys_ = ["rt66", "route66"]
    kwargs_ = dict(s_keys=keys_, return_localized=localized)
    return load_rt66_or_sp2(fpath, **kwargs_)


def load_south_plains_ii(fpath, localized=False):
    keys_ = ["splain", "southplain"]
    kwargs_ = dict(s_keys=keys_, return_localized=localized)
    return load_rt66_or_sp2(fpath, **kwargs_)


def load_sunflower(fpath):
    df = pd.read_table(fpath)
    if df.shape[1] == 1:
        df = df.dropna()
        cols_ = df.columns[0].split(",")
        columns = ["".join([q for q in col if q.isalpha()]) for col in cols_]
        df = df[df.columns[0]].str.split(",", expand=True)
        df.columns = [c.strip().replace(" ", "") for c in columns]
    else:
        df.columns = [c.strip().replace(" ", "") for c in df.columns]
        df["DATE"] = df["DATE"].astype(str).str.zfill(6)
        df["HOUR"] = df["HOUR"].astype(str).str.zfill(4)

    df = df.iloc[:, 1:].copy()
    uniquecols = []
    for c in df.columns:
        n, col = 1, c
        while col in uniquecols:
            col = f"{c}.{n}"
        uniquecols.append(col)
    uniquecols
    df.columns = uniquecols
    kwcol = [c for c in df.columns if all(i in c for i in ["KW", "1"])][0]
    df["MWh"] = pd.to_numeric(df[kwcol]).div(1000)
    df["HOUR"] = pd.to_numeric(df["HOUR"]).div(100).astype(int)
    df.index = pd.to_datetime(
        df["DATE"].astype(str).str.zfill(6)
        + " "
        + df["HOUR"].sub(1).astype(str).str.zfill(2)
        + ":00:00",
        format="%m%d%y %H:%M:%S",
    )
    df = format_meter_dataframe(df)
    return df


"""
SOLAR - COMANCHE
"""


def load_comanche(fpath):
    if fpath.suffix == ".xlsx":
        dcol = [c for c in pd.read_excel(fpath, nrows=0).columns if "DATE" in c][0]
        df = pd.read_excel(fpath, dtype={dcol: str})
    else:
        dcol = [c for c in pd.read_csv(fpath, nrows=0).columns if "DATE" in c][0]
        df = pd.read_csv(fpath, dtype={dcol: str})
    df[dcol] = df[dcol].str.zfill(6)  # pads month with zero if single digit
    df = df.iloc[:-1, :].copy()
    mcols = [c for c in df.columns if any(i in c.casefold() for i in ["kw", "kvar", "kwh"])]
    mcol = mcols[1]  # using the second col with power data (eg. ' KW    .1')
    df["MWh"] = pd.to_numeric(df[mcol]).div(1000)
    df = df.rename(columns={" DATE": "Day", " HOUR": "Hour"})
    df["Day"] = pd.to_datetime(df["Day"], format="%m%d%y").dt.date
    df["Hour"] = pd.to_numeric(df["Hour"]).div(100).astype(int)
    df.index = pd.to_datetime(
        df["Day"].astype(str) + " " + df["Hour"].sub(1).astype(str).str.zfill(2) + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
    )
    df = format_meter_dataframe(df)
    return df


"""
SOLAR - PI SITES
"""


def load_pi_meter_data(fpath):
    df = pd.read_csv(fpath, index_col=0, parse_dates=True)
    df = df.rename(columns={df.columns[0]: "MWh"})
    df = format_meter_dataframe(df)
    return df


def load_cw_marin(fpath):
    return load_pi_meter_data(fpath)


def load_indy_i(fpath):
    return load_pi_meter_data(fpath)


def load_indy_ii(fpath):
    return load_pi_meter_data(fpath)


def load_indy_iii(fpath):
    return load_pi_meter_data(fpath)


"""
SOLAR - ATLAS
"""


def load_az1(fpath):
    df_dict = pd.read_excel(
        fpath, engine="openpyxl", sheet_name=None
    )  # dict of dfs for each sheet in workbook
    target_sheet = [s for s, dff in df_dict.items() if (dff.shape[0] > 20)][0]
    df_ = df_dict[target_sheet]
    hdr = df_[df_.isin(["End Date"]).any(axis=1)].index[
        0
    ]  # find header row; contains value "End Date"
    df_.columns = df_.iloc[hdr].values
    keepcols = [c for c in df_.columns if any(i in c for i in ["Date", "HE"])]
    dfT = df_[keepcols].iloc[hdr + 1 :, :].reset_index(drop=True).copy()
    dfT["End Date"] = pd.to_datetime(dfT["End Date"])
    end_idx = dfT["End Date"].last_valid_index()
    dfT = dfT.iloc[: end_idx + 1, :].set_index("End Date").copy()
    dfT_ = dfT.transpose().reset_index(drop=True).rename_axis(None, axis=1)
    df = dfT_.melt(var_name="Date", value_name="MWh", ignore_index=False)
    df.index = pd.to_datetime(
        df["Date"].astype(str) + " " + df.index.astype(str).str.zfill(2) + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
    )
    df = format_meter_dataframe(df)
    return df


def load_ga3(fpath):
    df = pd.read_excel(fpath, header=5)
    df.columns = ["Timestamp", "MWh"]
    df = df.set_index("Timestamp")
    df.index = pd.to_datetime(df.index)
    df = format_meter_dataframe(df)
    return df


def load_ga4(fpath):
    return load_ga3(fpath)


def load_grand_view(fpath, meter_id, q=True):
    # note: GV-East='GVSP', GV-West='GVSR'
    df_dict = pd.read_excel(fpath, engine="openpyxl", sheet_name=None, nrows=4)
    matching_sheets = [
        sheet
        for sheet, df_ in df_dict.items()
        if any(meter_id in v for v in df_.iloc[:, 0].astype(str).values)
    ]
    if not matching_sheets:
        print("Sheet not found.")
        return None
    sheet_name = matching_sheets[0]
    if (not q) and (meter_id not in sheet_name):
        print("Note: sheet names in file do not match the data they contain")

    df_ = pd.read_excel(fpath, engine="openpyxl", sheet_name=sheet_name)
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
    df = format_meter_dataframe(df)
    return df


def load_grand_view_east(fpath):
    df = load_grand_view(fpath, meter_id="GVSP")
    return df


def load_grand_view_west(fpath):
    df = load_grand_view(fpath, meter_id="GVSR")
    return df


def load_ms3(fpath):
    # note: the "astype(str)" prevents error when column names are formatted as datetime
    dcol = [
        c
        for c in pd.read_excel(fpath, engine="openpyxl", nrows=0).columns.astype(str)
        if "DATE" in c
    ][0]
    df_ = pd.read_excel(fpath, engine="openpyxl", dtype={dcol: str})
    df_[dcol] = df_[dcol].str.zfill(6)
    df_[dcol] = pd.to_datetime(df_[dcol], format="%m%d%y")
    df_ = df_.set_index(dcol)
    hour_cols = [c for c in df_.columns if (type(c) in [dt.datetime, dt.time])]  # hrs ending 1-24
    renamecols_ = {c: c.time() for c in hour_cols if isinstance(c, dt.datetime)}
    df_ = df_[hour_cols].rename(columns=renamecols_).copy()
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
    df = format_meter_dataframe(df)
    return df


def load_maplewood_1(fpath):
    df = pd.read_excel(fpath, engine="openpyxl")
    datecol = [c for c in df.columns if "flowday" in c.casefold()][0]
    keepcols_ = [datecol, "Interval", "GEN_MWH_INT"]
    df = df[keepcols_].copy()
    df["Interval"] = df["Interval"].replace("24:00", "00:00")
    df["Interval"] = df["Interval"].shift(1, fill_value="00:00")
    if df.Interval.str.len().gt(5).any():
        df = df.loc[df.Interval.str.len().eq(5)].reset_index(drop=True)
    df.index = pd.to_datetime(df[datecol].astype(str) + " " + df.Interval)
    df = df.rename(columns={"GEN_MWH_INT": "MWh"})
    df = format_meter_dataframe(df, freq="15min")
    return df


def load_maplewood_2(fpath):
    df = pd.read_excel(fpath, engine="openpyxl")
    datecol = [c for c in df.columns if "flowday" in c.casefold()][0]
    keepcols_ = [datecol, "Interval", "RTEI_QTY"]
    df = df[keepcols_].copy()
    df["Interval"] = df["Interval"].replace("24:00", "00:00")
    df["Interval"] = df["Interval"].shift(1, fill_value="00:00")
    if df.Interval.str.len().gt(5).any():
        df = df.loc[df.Interval.str.len().eq(5)].reset_index(drop=True)
    df.index = pd.to_datetime(df[datecol].astype(str) + " " + df.Interval)
    df = df.rename(columns={"RTEI_QTY": "MWh"})
    df["MWh"] = df["MWh"].mul(-1)
    df = format_meter_dataframe(df, freq="15min")
    return df


def load_sweetwater(fpath):
    df = pd.read_excel(fpath, engine="openpyxl", sheet_name="HourlyData")
    df.index = pd.to_datetime(
        df["TradingDate"].astype(str)
        + " "
        + df["TradingHour"].sub(1).astype(str).str.zfill(2)
        + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
    )
    df["MWh"] = df["Meter_KW"].div(1000).copy()
    df = format_meter_dataframe(df)
    return df


def load_three_peaks(fpath):
    return load_sweetwater(fpath)


"""
SOLAR - VARSITY (non-stlmtui)
"""


def load_azalea(fpath):
    df = pd.read_excel(fpath)
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
    df = format_meter_dataframe(df)
    return df


def load_imperial_valley(fpath):
    df_ = pd.read_excel(fpath)
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
    df = format_meter_dataframe(df)
    return df


def load_marin_carport(filepath):
    df = pd.read_excel(filepath)
    df = df.rename(columns={"ReadTime": "Day", "kWTotal": "MWh"})
    df = df.drop(columns=["POA_Wm2"])
    df["MWh"] = df["MWh"].div(1000).div(4)
    df["Day"] = pd.to_datetime(df["Day"])
    df["Hour"] = pd.to_numeric(df["Day"].dt.hour)
    df["Day"] = df["Day"].dt.date
    df = df.groupby(["Day", "Hour"], as_index=False).sum()
    df = df[["Day", "Hour", "MWh"]]
    return df


def load_mulberry(fpath):
    df = pd.read_excel(fpath, header=5)
    df = df.loc[df.Hour.isin(range(1, 25))]
    df.index = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Hour"].sub(1).astype(str).str.zfill(2) + ":00:00",
        format="%m/%d/%Y %H:%M:%S",
    )
    df["MWh"] = df["kWh"].div(1000).copy()
    df = format_meter_dataframe(df)
    return df


def load_pavant(fpath):
    df = pd.read_excel(fpath, sheet_name="HourlyData")
    df.index = pd.to_datetime(
        df["TradingDate"].astype(str)
        + " "
        + df["TradingHour"].sub(1).astype(str).str.zfill(2)
        + ":00:00",
        format="%Y-%m-%d %H:%M:%S",
    )
    df["MWh"] = df["Meter_KW"].div(1000).copy()
    df = format_meter_dataframe(df)
    return df


def load_richland(fpath):
    df = pd.read_excel(fpath)
    hdr_idx = df[df.isin(["HourBegin"]).any(axis=1)].index[0]
    df.columns = df.iloc[hdr_idx, :].values
    df = df.rename(columns={"Generation_MW": "MWh"})
    df = df.iloc[hdr_idx + 1 :, :].reset_index(drop=True)
    if df["HourBegin"].last_valid_index() != df.shape[0]:  # catches last line ("total")
        df = df.iloc[:-1, :].copy()
    df = df.set_index("HourBegin")
    df.index = pd.to_datetime(df.index)
    df = format_meter_dataframe(df)
    return df


def load_selmer(fpath):
    df = pd.read_excel(fpath, header=5)
    df = df.loc[df.Hour.isin(range(1, 25))]
    df.index = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Hour"].sub(1).astype(str).str.zfill(2) + ":00:00",
        format="%m/%d/%Y %H:%M:%S",
    )
    df["MWh"] = df["kWh"].div(1000).copy()
    df = format_meter_dataframe(df)
    return df


def load_somers(fpath):
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
    df = format_meter_dataframe(df)
    return df
