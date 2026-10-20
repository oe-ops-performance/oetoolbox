import pandas as pd
from timezonefinder import TimezoneFinderL

from . import oemeta

_tf = TimezoneFinderL()

IANA_MAPPING = {
    "America/New_York": "US/Eastern",
    "America/Indiana/Indianapolis": "US/Eastern",
    "America/Chicago": "US/Central",
    "America/Boise": "US/Mountain",
    "America/Denver": "US/Mountain",
    "America/Phoenix": "US/Arizona",
    "America/Los_Angeles": "US/Pacific",
}


def infer_timezone(latitude: float, longitude: float, legacy=True) -> str:
    """Return IANA or Legacy tz name (e.g. 'America/Chicago' or 'US/Central') for a coordinate.

    IANA format -> Continent/City (avoids issues if countries or regional names change)
    Legacy format -> Country/Region (from earlier days of the time zone database)

    Longitude alone is ambiguous, so latitude is required. Handles Arizona
    (America/Phoenix), Alaska, Hawaii, and irregular zone borders.
    Raises ValueError if no timezone is found (e.g. a point far offshore).
    """
    iana_tz = _tf.timezone_at(lat=latitude, lng=longitude)
    if iana_tz is None:
        raise ValueError(f"No timezone found for ({latitude}, {longitude})")
    if legacy is False:
        return iana_tz
    legacy_tz = IANA_MAPPING.get(iana_tz, None)
    if not legacy_tz:
        raise ValueError(f"No legacy timezone mapping found for {iana_tz = }.")
    return legacy_tz


def print_datetime_index_info(df: pd.DataFrame):
    if type(df.index) is not pd.DatetimeIndex:
        print("Invalid index type.")
        return
    d0, d1 = map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"), [df.index[0], df.index[-1]])
    freq_mins = int(infer_freq(df.index).total_seconds() / 60)
    print(f"range: ['{d0}', '{d1}']")
    print(f"( freq = {freq_mins}m | n_rows = {df.shape[0]} | timezone = {df.index.tzinfo} )")
    return


def infer_freq_timedelta(df: pd.DataFrame):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Error: dataframe must have datetime index")
    freq_timedelta = df.index.diff().min()
    if freq_timedelta != df.index.diff().max():
        raise ValueError("Non-standard index detected. Must have uniform intervals.")
    return freq_timedelta


def localize_naive_datetimeindex(
    dataframe: pd.DataFrame, site: str = "", tz: str = "", q: bool = True
):
    if site == tz == "":
        raise ValueError('Error: must specify either "site" or timezone "tz"')
    elif not isinstance(dataframe.index, pd.DatetimeIndex):
        raise ValueError("Error: dataframe must have datetime index")

    if tz == "":
        tz = oemeta.data["TZ"].get(site)

    df = dataframe.copy()
    try:
        df = df.tz_localize(tz)
        return df
    except:
        if not q:
            print("dst condition detected; continuing")
        pass

    freq = pd.infer_freq(df.index)
    if not freq:
        freq = pd.infer_freq(df.index[:100])
        if not freq:
            raise ValueError("Error: could not infer frequency from index")

    # get expected local tz index (includes DST times; i.e. extra or missing hour)
    expected_local_index = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz=tz)

    # get equivalent utc index, then localize
    utc_offset = pd.Timestamp(df.index.min(), tz=tz).utcoffset()
    offset_hours = int(utc_offset.total_seconds() / 3600)
    utc_index = (df.index - pd.Timedelta(hours=offset_hours)).tz_localize("utc")
    localized_index = utc_index.tz_convert(tz=tz)

    # assign localized index, then reindex to add/remove DST changes
    df.index = localized_index
    df = df.reindex(expected_local_index)
    return df


def remove_tzinfo_and_standardize_index(dataframe: pd.DataFrame):
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        raise ValueError("Error: dataframe must have datetime index")

    freq = pd.infer_freq(dataframe.index)
    if not freq:
        freq = pd.infer_freq(dataframe.index[:100])
        if not freq:
            raise ValueError("Error: could not infer frequency from index")

    df = dataframe.tz_localize(None).copy()
    ref_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)

    n_remove = df.index.duplicated().sum()
    if n_remove > 0:
        df = df.iloc[:-n_remove, :]
        df.index = ref_index

    if df.shape[0] != ref_index.shape[0]:
        df = df.reindex(ref_index)  # dst, spring

    df = df.rename_axis("Timestamp")
    return df


def create_naive_index(start, end, freq):
    """Generates a timezone-naive date range (for use as datetime index)"""
    return pd.date_range(start, end, freq=freq)[:-1]


def create_localized_index(site, start, end, freq):
    """Generates a timezone-aware date range localized to a given site"""
    df_ = pd.DataFrame(index=create_naive_index(start, end, freq))
    return localize_naive_datetimeindex(dataframe=df_, site=site).index


def segmented_date_ranges(start: pd.Timestamp, end: pd.Timestamp, n_days: int):
    """Returns a list of sub date ranges with length n_days"""
    t0, t1 = start, start + pd.DateOffset(days=n_days)
    if t1 > end:
        return [(start, end)]
    date_range_list = []  # init
    while t1 <= end:
        date_range_list.append((t0, t1))
        if t1 == end:
            break
        t0 = t1
        t1 = t1 + pd.DateOffset(days=n_days)
        if t1 > end:
            t1 = end
    return date_range_list


def year_month_list(start_date: str, end_date: str):
    """Returns list of (year, month) tuples that exist in range between given start and end dates.

    Parameters
    ----------
    start_date : str
        Start date for target range, format: "%Y-%m-%d"
    end_date : str
        End date for target range, format: "%Y-%m-%d"

    Returns
    -------
    list of tuple
        A list of year/month tuples corresponding to the input dates.
    """
    # validate input dates
    start, end = map(pd.Timestamp, [start_date, end_date])
    fmt = lambda tstamp: tstamp.strftime("%Y-%m-%d")
    if fmt(start) != start_date or fmt(end) != end_date:
        raise ValueError("Input start/end dates must have format '%Y-%m-%d'")
    if start > end:
        start_date, end_date = end_date, start_date

    # generate year/month list
    unique_year_month = pd.date_range(start_date, end_date).strftime("%Y_%m").unique().to_list()
    tuple_from_string = lambda ym_str: tuple(map(lambda x: int(x), ym_str.split("_")))
    return list(map(tuple_from_string, unique_year_month))


def get_dates(year: int, month: int, as_string: bool = False, tz: str = None):
    start = pd.Timestamp(year, month, 1, tz=tz)
    end = start + pd.DateOffset(months=1)
    if as_string is False:
        return [start, end]
    return list(map(lambda x: x.strftime("%Y-%m-%d"), [start, end]))


def infer_freq(datetime_index: pd.DatetimeIndex):
    diffs = datetime_index.to_series().diff().dropna()
    mode_diffs = diffs.mode()
    return mode_diffs.iloc[0] if not mode_diffs.empty else diffs.min()


def upsample(df, new_freq, agg_type="interpolate"):
    if type(df.index) is not pd.DatetimeIndex:
        raise TypeError("Dataframe must have a DatetimeIndex.")
    freq_td = infer_freq(df.index)
    if pd.Timedelta(new_freq) > freq_td:
        raise ValueError("New frequency must be smaller than native data interval.")
    shift_td = freq_td / 2
    if agg_type == "interpolate":
        df_r = df.resample(new_freq).interpolate()
    elif agg_type in ("ffill", "mean", "max", "min"):
        df_r = df.resample(new_freq).agg(agg_type)
    else:
        raise KeyError("Invalid agg_type specified.")
    df_r = df_r.shift(periods=1, freq=shift_td)
    new_index = pd.date_range(
        df.index.min(), df.index.max() + freq_td, freq=new_freq, inclusive="left"
    )
    return df_r.reindex(new_index)


def get_continuous_ranges(df_segmented: pd.DataFrame, native_freq: pd.Timedelta):
    if type(df_segmented.index) is not pd.DatetimeIndex:
        raise TypeError("Dataframe must have a datetime index.")
    if df_segmented.empty:
        return []
    df = df_segmented.copy()
    df["time_diff"] = df.index.diff()
    if len(df) > 1:
        df.iloc[0, -1] = df.iloc[1, -1]

    step = native_freq
    conditions = df.time_diff.eq(step)
    breaks = df.time_diff.gt(step).cumsum()

    return [x.index for _, x in df[conditions].groupby(breaks)]
