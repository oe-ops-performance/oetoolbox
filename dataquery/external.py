import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)  # temp, for meteostat

import json
import meteostat
import pandas as pd
import requests

from ..utils import oemeta
from ..utils.config import DTN_CREDENTIALS


def segmented_date_ranges(start, end, n_days):
    t0, t1 = start, start + pd.DateOffset(days=n_days)
    if t1 > end:
        return [(start, end)]
    daterange_list = []  # init
    while t1 <= end:
        daterange_list.append((t0, t1))
        if t1 == end:
            break
        t0 = t1
        t1 = t1 + pd.DateOffset(days=n_days)
        if t1 > end:
            t1 = end
    return daterange_list


def request_access_token():
    api_token_url = DTN_CREDENTIALS["api_token_url"]
    post_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    request_body = {
        "grant_type": "client_credentials",
        "client_id": DTN_CREDENTIALS["client_id"],
        "client_secret": DTN_CREDENTIALS["client_secret"],
        "audience": "https://weather.api.dtn.com/conditions",
    }
    response = requests.post(api_token_url, data=json.dumps(request_body), headers=post_headers)
    access_token = response.json()["data"]["access_token"]
    return access_token


# note: 10-day maximum -- this function is called in the main query function "query_DTN" (below)
def query_DTN_weather_data(latitude, longitude, start, end, interval, fields):
    """
    known fields:
        airTemp
        iceAccPeriod
        liquidAccPeriod
        precipAccPeriod
        precipAccAdjusted
        precipAccRaw
        shortWaveRadiation
        snowAccPeriod
        windSpeed
    """
    token_ = request_access_token()
    headers = {
        "Accept-Encoding": "gzip",
        "Accept": "application/json",
        "Authorization": f"Bearer {token_}",
    }
    query_specs = {
        "lat": latitude,
        "lon": longitude,
        "startTime": start,
        "endTime": end,
        "interval": interval,
        "parameters": ",".join(fields),
    }
    api_request_url = "https://weather.api.dtn.com/v1/conditions"
    response = requests.request("GET", api_request_url, headers=headers, params=query_specs)
    response_dict = json.loads(response.text)
    if "content" not in response_dict:
        print("!! ERROR !!")
        df = pd.DataFrame()
        errors_ = response_dict.get("errors")
        if errors_:
            error_dict = errors_[0]
            [print(f"{key}: {val}") for key, val in error_dict.items()]
        else:
            print(response_dict)
    else:
        data_dict = response_dict["content"]["items"][0].get("parameters").copy()
        df = pd.DataFrame(data_dict)
    return df


def query_DTN(lat, lon, t_start, t_end, interval, fields, tz=None, q=True):
    qprint = lambda msg, end="\n": None if q else print(msg, end=end)
    fmt_tstamp = (
        lambda t: pd.Timestamp(t, tz=tz).tz_convert(tz="UTC").tz_localize(None).isoformat() + "Z"
    )

    # split into 9-day segments (if 10-day & fall dst, end up losing timestamp)
    daterange_list = segmented_date_ranges(t_start, t_end, n_days=9)

    qprint(f"\nQuerying DTN weather data...")
    df_list = []
    for i, rng_ in enumerate(daterange_list):
        start_date, end_date = rng_
        start_, end_ = map(fmt_tstamp, [start_date, end_date])
        df_ = query_DTN_weather_data(lat, lon, start_, end_, interval, fields)
        qprint(
            f">> sub-range {i+1} of {len(daterange_list)}",
            end="\r" if ((i + 1) < len(daterange_list)) else "\n",
        )
        df_list.append(df_)
    qprint("\nDone!")

    df = pd.concat(df_list)
    # if any(df.index.duplicated()):
    #     df = df[~df.index.duplicated(keep='first')]
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(tz=tz)

    return df


def load_noaa_weather_data(site, start, end, freq="h", q=True):
    """Loads NOAA ambient temperature and wind speed data"""
    qprint = lambda msg, end="\n": None if q else print(msg, end=end)
    start_date = pd.Timestamp(start).floor("D")
    end_date = pd.Timestamp(end).ceil("D")

    # get nearest ten stations from meteostat library
    lat, lon = oemeta.data["LatLong"].get(site)
    df_stations = meteostat.Stations().nearby(lat, lon).fetch(10)

    # check most recent available date
    target_end_date = pd.Timestamp(end_date)
    latest_date = df_stations["hourly_end"].max()
    n_missing_days = (end_date - latest_date).days
    if n_missing_days > 0:
        qprint(f"NOTE: hourly data is only available until {latest_date} ({n_missing_days = })")
        target_end_date = latest_date

    # select nearest station with data, then get data
    station = df_stations[(df_stations.hourly_end >= target_end_date)].iloc[0]
    tz = oemeta.data["TZ"].get(site)
    df = meteostat.Hourly(station.name, start_date, target_end_date, timezone=tz).fetch()

    # convert wind speed units, then filter/rename columns
    df["wspd"] = df["wspd"].mul(5 / 18)  # Convert km/h to m/s
    df = df[["temp", "wspd"]].rename(columns={"temp": "NOAA_AmbTemp", "wspd": "NOAA_WindSpeed"})
    return df
