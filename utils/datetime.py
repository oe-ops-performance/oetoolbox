import pandas as pd

from . import oemeta


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
