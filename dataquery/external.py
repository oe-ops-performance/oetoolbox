import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)  # temp, for meteostat

import json
import meteostat
import pandas as pd
import requests

from ..utils import oemeta
from ..utils.config import DTN_CREDENTIALS
from ..utils.helpers import quiet_print_function, with_retries


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
    if DTN_CREDENTIALS is None:
        raise Exception("No DTN credentials found in secrets/secrets.json file.")
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
@with_retries(n_max=5)
def query_DTN_weather_data(latitude, longitude, start, end, interval, fields, q=True):
    """
    all parameters/fields available for hourly historical data
    https://devportal.dtn.com/catalog/Weather/dtn-weather-conditions-api/documentation#tag--Parameters
        airTemp - C, F
            Air temperature at two meters above ground level. Units depend on unitcode setting.
        cloudCover - %
            Cloud cover data. Cloud cover refers to the percentage of the sky covered by clouds.
        dewPoint - C, F
            Dew point temperature at two meters above ground level. Dew point temperature is
            defined as the temperature to which the air must be cooled for saturation and
            condensation to occur.
        iceAccPeriod - mm, in
            Hourly ice accumulation data.
        liquidAccPeriod - mm, in
            Hourly liquid accumulation data.
        longWaveRadiation - W/m^2
            Downwelling longwave radiation flux data. Longwave radiation is the energy emitted from
            non-solar radiation sources.
        precipAccPeriod - mm, in
            Hourly liquid-equivalent precipitation accumulation data.
        precipAccAdjusted - mm, in
            Liquid-equivalent precipitation accumulation, fundamentally derived from the raw
            precipitation product, but then adjusted to more closely match available ground truth
            observations. Due to the delays in receiving these ground truth data, accumulation
            adjustment typically lags real-time by a day or more.
        precipAccRaw - mm, in
            Liquid-equivalent precipitation accumulation, estimated from multiple sources of data
            that may include any or all of the following: weather radar, satellite, computer
            models, and surface observation data.
        relativeHumidity - %
            Relative humidity at two meters above ground level. Relative humidity is the ratio of
            the actual amount of water vapor in the air to the maximum amount that can physically
            exist at a given air temperature.
        shortWaveRadiation - W/m^2
            Downwelling shortwave radiation flux data. Shortwave radiation is the high-energy solar
            radiation that reaches Earth’s surface.
        snowAccPeriod - mm, in
            Hourly snow accumulation data.
        surfacePressure - hPa, mb
            This parameter represents the pressure that the air exerts on the surface of the Earth.
        visibility - km, mi
            Visibility data. Visibility is a measure of the lateral distance one can see before
            one’s line of sight is obstructed due to weather conditions.
        windDirection - degrees
            Wind direction at ten meters above ground level, measured with respect to true north.
            A wind direction from true north corresponds to a value of zero degrees, which
            increases to 360 degrees with corresponding clockwise shifts in wind direction. Returns
            "n/a" if wind speed is less than one mph.
        windGust - km/h, mph
            Wind gust at ten meters above ground level. Returns "n/a" if wind speed is less than
            five mph, or if the difference between wind gust and wind speed is less than five mph.
        windSpeed - m/s, mph
            Wind speed at ten meters above ground level. Units dependent on the unitcode setting.

    reference: https://weather.api.dtn.com/v1/docs/conditions/#section/Weather-Parameters
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
        # "interval": interval,
        "parameters": ",".join(fields),
    }
    api_request_url = "https://weather.api.dtn.com/v2/conditions"
    response = requests.request("GET", api_request_url, headers=headers, params=query_specs)
    response_dict = json.loads(response.text)
    if "features" not in response_dict:
        raise KeyError(
            f"Error retrieving data from DTN; 'features' not in response; {response_dict}"
        )
    if not q:
        print("Received valid response from DTN API request.")
    data_dict = response_dict["features"][0].get("properties").copy()
    df = pd.DataFrame(data_dict).T
    return df


def query_DTN(lat, lon, t_start, t_end, interval, fields, tz=None, q=True):
    """Returns a timezone-aware dataframe with data queried from DTN"""
    qprint = quiet_print_function(q=q)
    fmt_tstamp = (
        lambda t: pd.Timestamp(t, tz=tz).tz_convert(tz="UTC").tz_localize(None).isoformat() + "Z"
    )

    # split into 9-day segments (if 10-day & fall dst, end up losing timestamp)
    daterange_list = segmented_date_ranges(t_start, t_end, n_days=9)

    qprint(f"\nQuerying DTN weather data...")
    df_list = []
    for i, rng_ in enumerate(daterange_list):
        qprint(f">> sub-range {i+1} of {len(daterange_list)}")
        start_date, end_date = rng_
        start_, end_ = map(fmt_tstamp, [start_date, end_date])
        df_ = query_DTN_weather_data(lat, lon, start_, end_, interval, fields, q=q)
        df_list.append(df_)
    qprint("DTN query completed successfully.")

    df = pd.concat(df_list)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(tz=tz)
    df = df[~df.index.duplicated(keep="first")]

    expected_index = pd.date_range(t_start, t_end, freq="h", tz=tz, inclusive="left")
    if len(df.index) != len(expected_index) or len(df.index.difference(expected_index)) > 0:
        # dtn output usually includes last timestamp
        df = df.reindex(expected_index)

    return df


def load_noaa_weather_data(site, start, end, freq="h", q=True):
    """Loads NOAA ambient temperature and wind speed data"""
    qprint = quiet_print_function(q=q)
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


@with_retries(n_max=5)
def query_fracsun_daily_soiling(api_key: str, device_id: str, start_date: str, end_date: str):
    """
    Query daily soiling and insolation values via Fracsun API.
    Dates must be in YYYY-MM-DD format.
    """
    url = f"https://admin.fracsun.com/api/device/soiling/{device_id}/"
    params = {"apiKey": api_key, "startDate": start_date, "endDate": end_date}
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()  # Raise error if request failed
    return pd.DataFrame(response.json())


def _get_fracsun_credentials(site: str) -> list[dict]:
    """Returns a list of dictionaries with keys 'api_key' and 'device_id' for specified site."""
    return [
        credentials
        for key, credentials in oemeta.data["fracsun_api_credentials"].items()
        if key.split("_")[0] == site
    ]


def get_fracsun_sites():
    return list(
        sorted(
            set(x.split("_")[0] for x in oemeta.data["fracsun_api_credentials"]),
            key=lambda site: site.lower(),
        )
    )


def _format_fracsun_data(df_response: pd.DataFrame) -> pd.DataFrame:
    df = df_response.drop(columns=["calculation_time", "utc_calcTime", "device"]).copy()
    df["day"] = pd.to_datetime(df["day"])
    return df


def query_fracsun(site: str, start_date: str, end_date: str, average: bool = True, q: bool = True):
    """Runs query for fracsun daily soiling using credentials for specified site."""
    qprint = quiet_print_function(q=q)
    credentials_list = _get_fracsun_credentials(site)
    if not credentials_list:
        raise Exception(f"No Fracsun credentials found for {site = }.")
    n_devices = len(credentials_list)
    qprint(f"[{site}] - Querying data for {n_devices} device(s).")
    df_list = []
    for credentials in credentials_list:
        api_key = credentials["api_key"]
        device_id = credentials["device_id"]
        dff = query_fracsun_daily_soiling(api_key, device_id, start_date, end_date)
        dff = _format_fracsun_data(df_response=dff)
        df_list.append(dff)
    qprint("Done.")

    df = df_list[0].copy()
    datacols = list(sorted([c for c in df.columns if c != "day"]))
    ordered_cols = ["day", *datacols]
    df = df[ordered_cols]
    if len(df_list) > 1:
        for i, df_ in enumerate(df_list[1:], start=2):
            suffixes = (None, "_2") if i == 2 else (None, f"_{i}")
            df = pd.merge(df, df_, on="day", suffixes=suffixes)
        df = df.rename(columns={c: f"{c}_1" for c in datacols})
    df = df.set_index("day").apply(pd.to_numeric, errors="coerce")

    if n_devices > 1 and average is True:
        df = average_across_fracsun_devices(df, q=q)

    return df


def average_across_fracsun_devices(df: pd.DataFrame, q: bool = True):
    qprint = quiet_print_function(q=q)
    if max(len(c.split("_")) for c in df.columns) == 1:
        qprint("The provided Fracsun data does not have multiple devices.")
        return df
    n_devices = max(int(c.split("_")[-1]) for c in df.columns)
    column_names = list(sorted(set(c.split("_")[0] for c in df.columns)))
    df_avg = pd.DataFrame(index=df.index.copy())
    for col in column_names:
        related_cols = [c for c in df.columns if c.startswith(col)]
        df_avg[col] = df[related_cols].mean(axis=1).copy()
    qprint(f"Averaged data across {n_devices} devices.")
    return df_avg
