import pandas as pd

from ..utils import oemeta


def localize_naive_datetimeindex(dataframe: pd.DataFrame, site: str = "", tz: str = ""):
    if site == tz == "":
        print('Error: must specify either "site" or timezone "tz"')
        return
    elif not isinstance(dataframe.index, pd.DatetimeIndex):
        print("Error: dataframe must have datetime index")
        return
    if tz == "":
        tz = oemeta.data["TZ"].get(site)

    df = dataframe.copy()
    try:
        df = df.tz_localize(tz)
        return df
    except:
        print("dst condition detected")
        pass

    freq = pd.infer_freq(df.index)
    if not freq:
        freq = pd.infer_freq(df.index[:100])
        if not freq:
            print("Error: could not infer frequency from index")
            return

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
        print("Error: dataframe must have datetime index")
        return
    freq = pd.infer_freq(dataframe.index)
    if not freq:
        freq = pd.infer_freq(dataframe.index[:100])
        if not freq:
            print("Error: could not infer frequency from index")
            return
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
