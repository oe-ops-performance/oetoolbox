from astral import LocationInfo
from astral.location import Location
from astral.sun import sun
import datetime as dt
from json import JSONDecodeError
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pvlib
from sklearn.metrics import r2_score

from ..dataquery.external import query_DTN
from ..datatools.pvlib import get_surface_parameters
from ..utils import oemeta, oepaths
from ..utils.datetime import localize_naive_datetimeindex, remove_tzinfo_and_standardize_index
from ..utils.helpers import quiet_print_function


SENSOR_IDENTIFIERS = {
    "amb_temp": ["amb", "air_temp", "temp_air", "temp_c_skid"],
    "mod_temp": ["mod", "mdl", "mts", "mst", "pnltemp", "pnltmp", "bom"],
    "wind": ["wind", "wnd", "speed", "ws_ms"],
    "ghi": ["ghi", "global", "hor"],
    "poa": ["poa", "plane", "irr"],  # and not ghi ids
}

SENSOR_LIMITS = {
    "poa": [0, 1500],
    "ghi": [0, 1500],
    "amb_temp": [-35, 150],
    "mod_temp": [-35, 150],
    "wind": [0, 75],
}

BACKFILL_COLUMNS = {
    "poa": "poa_global",
    "ghi": "dtn_ghi",
    "amb_temp": "dtn_temp_air",
    "mod_temp": "ModTemp_Monthly_Profile",
    "wind": "dtn_wind_speed",
}


def get_meteo_fpath(site, year, month, version):
    """returns filepath to most recent cleaned meteo file in flashreport folder"""
    dir = oepaths.frpath(year, month, ext="solar", site=site)
    if not dir.exists():
        raise ValueError("No flashreport folder found for given inputs.")

    if version not in ["raw", "cleaned", "processed"]:
        raise ValueError("Version must be one of: [raw, cleaned, processed]")

    file_pattern = {
        "raw": "PIQuery*MetStations*.csv",
        "cleaned": "PIQuery*MetStations*CLEANED*.csv",
        "processed": "PIQuery*MetStations*CLEANED*PROCESSED*.csv",
    }.get(version)

    exclude_ids = {
        "raw": ["CLEANED", "PROCESSED"],
        "cleaned": ["PROCESSED"],
        "processed": [],
    }.get(version)

    matching_fpaths = [
        fp for fp in list(dir.glob(file_pattern)) if not any(i in fp.name for i in exclude_ids)
    ]

    return oepaths.latest_file(matching_fpaths)


def load_clean_meteo_file(site, filepath, q=True):
    """loads most recent clean met file & localizes to site tz"""
    qprint = quiet_print_function(q=q)
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        # this can happen if .csv already has tz info & an error prevents parsing (e.g. dst)
        df.index = pd.to_datetime(
            df.index,
            format="%Y-%m-%d %H:%M:%S%z",
            utc=True,
        ).tz_convert(tz=oemeta.data["TZ"].get(site))
    elif df.index.tz is None:
        qprint("localizing dataframe to site timezone")
        df = localize_naive_datetimeindex(df, site=site)
    return df


def get_sensor_type(column_name):
    """Returns matching key from sensor identifiers dict"""
    for sensor_type, identifiers in SENSOR_IDENTIFIERS.items():
        if any(i in column_name.lower() for i in identifiers):
            return sensor_type
    return


def grouped_meteo_columns(sensor_columns):
    grouped_cols = {
        sensor_type: [c for c in sensor_columns if get_sensor_type(c) == sensor_type]
        for sensor_type in SENSOR_IDENTIFIERS
    }
    return {key: cols for key, cols in grouped_cols.items() if len(cols) > 0}


def site_location_metadata(site):
    alt = oemeta.data["Altitude"].get(site)
    lat, lon = oemeta.data["LatLong"].get(site)
    tz = oemeta.data["TZ"].get(site)
    return alt, lat, lon, tz


def sun_location_info(tz, lat, lon, name="", region=""):
    """returns astral.LocationInfo object for site"""
    kwargs = dict(timezone=tz, latitude=lat, longitude=lon)
    return LocationInfo(name=name, region=region, **kwargs)


def site_location(site):
    """creates astral location object for given site"""
    _, lat, lon, tz = site_location_metadata(site)
    location_info = sun_location_info(tz, lat, lon, name=site)
    return Location(location_info)


getsun = lambda tz, lat, lon, date: sun(sun_location_info(tz, lat, lon).observer, date=date)


def get_suntimes(tz, lat, lon, start, end):
    suntimes = []
    args = [tz, lat, lon]
    localized = lambda date: pd.Timestamp(date).tz_convert(tz)
    for i, date in enumerate(pd.date_range(start, end)):
        args = [tz, lat, lon]
        try:
            sun = getsun(*args, date=date)
            dawn, dusk = sun["dawn"], sun["dusk"]
            noon = sun["noon"]
        except:
            delta = 1 if i == 0 else -1
            date_offset = dt.timedelta(days=delta)
            sun = getsun(*args, date=date + date_offset)
            dawn, dusk = map(lambda d: d - date_offset, [sun["dawn"], sun["dusk"]])
            noon = sun["noon"] - date_offset

        sun_dict = {
            "dawn": localized(dawn).floor("min"),
            "noon": localized(noon).round("min"),
            "dusk": localized(dusk).ceil("min") + dt.timedelta(days=1),
        }
        suntimes.append(sun_dict)

    return suntimes


def get_site_suntimes(site, start, end):
    _, lat, lon, tz = site_location_metadata(site)
    return get_suntimes(tz, lat, lon, start, end)


def get_daylight_timestamps(site, start, end, freq="1min"):
    _, lat, lon, tz = site_location_metadata(site)
    tstamps = []
    for dict_ in get_suntimes(tz, lat, lon, start, end):
        dawn_ = dict_["dawn"].floor(freq)
        dusk_ = dict_["dusk"].ceil(freq)
        tstamps.extend(list(pd.date_range(dawn_, dusk_, freq=freq)))
    return tstamps


def segmented_daylight_timestamps(site, start, end, freq="1min"):
    """returns dictionary with str(day) as keys and [am_tstamps, pm_tstamps] as values"""

    daylight_tstamps = get_daylight_timestamps(site, start, end, freq=freq)


def get_pvlib_solar_position(site, datetime_index):
    """returns dataframe with azimuth, zenith, etc."""
    alt, lat, lon, tz = site_location_metadata(site)
    location = pvlib.location.Location(lat, lon, tz, alt, site)
    idx = datetime_index + pd.Timedelta(minutes=30)
    solar_position = location.get_solarposition(idx)
    solar_position.index = datetime_index
    return solar_position


def get_dtn_data(site, start_date, end_date):
    """queries dtn weather data and adds solar position data

    Parameters
    ----------
    site : str
        Name of solar site
    start_date : str
        Start timestamp formatted '%Y-%m-%d', tz-naive
    end_date : str
        End timestamp formatted '%Y-%m-%d', tz-naive
    """
    _, lat, lon, tz = site_location_metadata(site)
    dtn_columns = {
        "airTemp": "dtn_temp_air",
        "shortWaveRadiation": "dtn_ghi",
        "windSpeed": "dtn_wind_speed",
    }
    fields = list(dtn_columns.keys())
    interval = "hour"
    dtn_start = pd.Timestamp(start_date).floor("D")
    dtn_end = pd.Timestamp(end_date).ceil("D") + pd.Timedelta(
        hours=1
    )  # add hour for upsample to min

    n_retries = 5
    args = [lat, lon, dtn_start, dtn_end, interval, fields]
    errors = []
    while n_retries > 0:
        try:
            df = query_DTN(*args, tz=tz)
            break
        except Exception as e:
            errors.append(e)
            n_retries -= 1

    if n_retries == 0:
        print("Note: Could not successfully retrieve DTN data after 5 retries.")
        return errors
    elif n_retries < 5:
        print(f"Note: DTN data was retrieved after {5 - n_retries} additional attempt(s).")

    # rename columns
    df = df.rename(columns=dtn_columns)

    # add solar position info
    solar_position = get_pvlib_solar_position(site, df.index)
    df = df.join(solar_position)

    df_dates = pd.DataFrame(index=pd.date_range(df.index[0], df.index[-1].ceil("D")))
    df_dates["tz_offset"] = df_dates.index.strftime("%z")
    unique_offsets = df_dates["tz_offset"].unique()
    if 3 in df.index.month.unique() and len(unique_offsets) > 1:
        shift_date = df_dates.loc[df_dates["tz_offset"].eq(unique_offsets[0])].index[-1]
        shifted = df.index >= shift_date
        for col in df.columns:
            df.loc[shifted, col] = df.loc[shifted, col].shift(1)

    return df


def transpose_poa_from_dtn_data(df, site):
    """uses DTN and pvlib clearsky data to transpose POA values and add to input df"""
    dirint = pvlib.irradiance.dirint(
        df["dtn_ghi"],
        df["apparent_zenith"].values,
        df.index,
        max_zenith=85,
    )
    dirint = dirint.rename("dni_dirint")
    disc_dni = dirint.copy()

    df_disc = pvlib.irradiance.complete_irradiance(
        solar_zenith=df["apparent_zenith"],
        ghi=df["dtn_ghi"],
        dhi=None,
        dni=disc_dni,
    )
    df_disc = df_disc.rename(columns={"dni": "dtn_dni", "dhi": "dtn_dhi"})
    df_disc = df_disc.drop(columns=["ghi"])
    df_disc["dni_extra"] = pvlib.irradiance.get_extra_radiation(
        datetime_or_doy=df.index,
        method="nrel",
        epoch_year=df.index.year[0],
    )

    # add columns to df: dtn_dhi, dtn_dni, dni_extra
    df = df.join(df_disc).copy()

    # get surface parameters for transposition
    surface_tilt, surface_azimuth = get_surface_parameters(site, df)

    df_irr = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=df["apparent_zenith"],
        solar_azimuth=df["azimuth"],
        dni=df["dtn_dni"],
        ghi=df["dtn_ghi"],
        dhi=df["dtn_dhi"],
        dni_extra=df["dni_extra"],
        model="perez",
    )
    df_irr = df_irr.fillna(0)

    # add columns: poa_global, poa_direct, poa_diffuse, poa_sky_diffuse, poa_ground_diffuse
    df = df.join(df_irr).copy()
    return df


def get_pvlib_clearsky(site, datetime_index):
    """returns df with columns: ghi_clearsky, dni_clearsky, dhi_clearsky"""
    alt, lat, lon, tz = site_location_metadata(site)
    location = pvlib.location.Location(lat, lon, tz, alt, site)
    return location.get_clearsky(datetime_index).add_suffix("_clearsky")


def load_supporting_data(site, start_date, end_date):
    """queries dtn data for comparing with sensors & determining calibration issues"""
    df_dtn = get_dtn_data(site, start_date, end_date)
    df = transpose_poa_from_dtn_data(df_dtn, site)
    return df


def get_supporting_data(site, year, month, freq="1h", q=True):
    """loads dtn data from files in flashreport dtn folder if exists, otherwise queries/saves
    -> IMPORTANT: returns dataframe with first timestamp of next month (for upsampling using ffill)
    """
    qprint = quiet_print_function(q=q)
    if not oepaths.frpath(year, month).exists():
        raise ValueError("No FlashReport folder exists for selected year/month")

    start_ = pd.Timestamp(year, month, 1)
    end_ = start_ + pd.DateOffset(months=1)
    start_date, end_date = map(lambda d: d.strftime("%Y-%m-%d"), [start_, end_])
    expected_index = pd.date_range(start_date, end_date, freq=freq, inclusive="left")

    dtn_dir = Path(oepaths.frpath(year, month), "DTN")
    if not dtn_dir.exists():
        dtn_dir.mkdir()

    # check for site dtn fpath in flashreports dir
    dtn_fpath = Path(dtn_dir, f"dtn_data_{site}_{year}-{month:02d}.csv")
    if not dtn_fpath.exists():
        qprint("querying dtn data")

        df = load_supporting_data(site, start_date, end_date)
        df.to_csv(dtn_fpath)
        dtn_fpath_str = "\\".join(dtn_fpath.parts[-3:])
        qprint(f"saved file: {dtn_fpath_str}")
    else:
        df = pd.read_csv(dtn_fpath, index_col=0, parse_dates=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(
                df.index,
                format="%Y-%m-%d %H:%M:%S%z",
                utc=True,
            ).tz_convert(tz=oemeta.data["TZ"].get(site))
        qprint(f"loaded from file: {dtn_fpath.name}")

    if pd.Timedelta(freq) < pd.Timedelta(hours=1):
        df = df.resample(freq).ffill()

    start = df.index.min()
    end = start + pd.DateOffset(months=1)
    expected_index = pd.date_range(start, end, freq=freq, inclusive="left")
    df = df.reindex(expected_index)

    return df


def identify_clearsky_days_using_dtn_ghi(site, df, return_all=False):
    """returns list of tuples (day, r2_score) indicating r2 comparing dtn to clearsky (range=0.9-1.1), sorted greatest to least"""
    if "dtn_ghi" not in df.columns:
        raise ValueError("Input dataframe must have 'dtn_ghi' column.")
    ref_col, data_col = "ghi_clearsky", "dtn_ghi"
    df_clear = get_pvlib_clearsky(site, df.index)
    df[ref_col] = df_clear[ref_col].copy()

    clearsky_r2_vals = []
    for date in list(sorted(set(df.index.date))):
        df_ = df.loc[df.index.date == date, [ref_col, data_col]].dropna().copy()
        r2 = r2_score(df_[ref_col].values, df_[data_col].values)
        if not return_all:
            if abs(r2 - 1) >= 0.1:
                continue
        clearsky_r2_vals.append((date, r2))

    return list(sorted(clearsky_r2_vals, key=lambda item: item[1], reverse=True))


# TODO: modify to use dataframe for input instead of year/month (i.e. for dynamic start/end dates)
def evaluate_sensor_data(site, year, month, n_clearsky_days=7, q=True):
    """loads clean met file, queries external data, then compares with poa/ghi columns"""
    qprint = quiet_print_function(q=q)
    met_fpath = get_meteo_fpath(site, year, month, version="cleaned")
    df_met = load_clean_meteo_file(site, met_fpath)  # note: tz-aware
    grouped_sensor_cols = grouped_meteo_columns(df_met.columns)

    df_ext = get_supporting_data(site, year, month)
    clearsky_day_list = identify_clearsky_days_using_dtn_ghi(site, df_ext)
    if not clearsky_day_list:
        qprint("No clearsky days identified in dataset; using all days in range.")
        clearsky_day_list = identify_clearsky_days_using_dtn_ghi(site, df_ext, return_all=True)

    top_clearsky_dates = [item[0] for item in clearsky_day_list[:n_clearsky_days]]

    REFERENCE_COLS = {"poa": "poa_global", "ghi": "dtn_ghi"}
    sensor_summaries = {}
    for sensor_type in REFERENCE_COLS:
        if sensor_type not in grouped_sensor_cols:
            continue
        # remove outliers
        ll_, ul_ = SENSOR_LIMITS[sensor_type]
        cols = grouped_sensor_cols[sensor_type]
        df_met[cols] = df_met[cols].clip(lower=ll_, upper=ul_)

        df = df_met[cols].resample("h").mean().copy()
        comparison_col = REFERENCE_COLS[sensor_type]
        df["REF"] = df_ext[comparison_col].copy()

        r2_scores = {}
        for col in cols:
            df_ = df.loc[df.index.date.isin(top_clearsky_dates), [col, "REF"]].dropna().copy()
            r2_scores[col] = r2_score(df_["REF"].values, df_[col].values)
        dfr2 = pd.DataFrame({"r2_score": r2_scores.values()}, index=r2_scores.keys())
        avg_r2 = dfr2["r2_score"].mean()
        dfr2["%_delta_from_mean"] = dfr2["r2_score"].sub(avg_r2).div(avg_r2).mul(100)
        sensor_summaries[sensor_type] = dfr2.copy()

    return sensor_summaries


def get_sensor_summaries(df_met, df_ext, top_clearsky_dates):
    grouped_sensor_cols = grouped_meteo_columns(df_met.columns)
    REFERENCE_COLS = {"poa": "poa_global", "ghi": "dtn_ghi"}
    sensor_summaries = {}
    for sensor_type in REFERENCE_COLS:
        if sensor_type not in grouped_sensor_cols:
            continue
        # remove outliers
        ll_, ul_ = SENSOR_LIMITS[sensor_type]
        cols = grouped_sensor_cols[sensor_type]
        df_met[cols] = df_met[cols].clip(lower=ll_, upper=ul_)

        df = df_met[cols].resample("h").mean().copy()
        comparison_col = REFERENCE_COLS[sensor_type]
        df["REF"] = df_ext[comparison_col].copy()

        r2_scores = {}
        for col in cols:
            is_clearsky = pd.to_datetime(df.index.date).isin(top_clearsky_dates)
            df_ = df.loc[is_clearsky, [col, "REF"]].dropna().copy()
            if not df_.empty:
                r2_scores[col] = r2_score(df_["REF"].values, df_[col].values)
            else:
                r2_scores[col] = -1000  # no data on clearsky days (flag sensor as bad)
        dfr2 = pd.DataFrame({"r2_score": r2_scores.values()}, index=r2_scores.keys())
        avg_r2 = dfr2["r2_score"].mean()
        dfr2["%_delta_from_mean"] = dfr2["r2_score"].sub(avg_r2).div(avg_r2).mul(100)
        sensor_summaries[sensor_type] = dfr2.copy()

    return sensor_summaries


def bad_cols_from_sensor_summaries(sensor_summaries, r2_diff=0.1):
    output = {}
    for key, df in sensor_summaries.items():
        # by default, r2_score considered good if between 0.9 and 1.1 (i.e. diff = 0.1)
        bad_cols = df.loc[df["r2_score"].lt(1 - r2_diff)].index.to_list()
        output[key] = {"bad_columns": bad_cols, "n_bad": len(bad_cols), "n_all": df.shape[0]}
    return output


def validate_meteo_filepath(fpath, site):
    if not fpath.exists():
        raise ValueError("Referenced filepath does not exist.")
    elif f"PIQuery_MetStations_{site}" not in fpath.name:
        raise ValueError(
            "File must be from FlashReports directory in standard PIQuery output format"
        )
    # load file and identify sensor columns
    df = pd.read_csv(fpath, index_col=0, nrows=0)
    grouped_sensor_cols = grouped_meteo_columns(df.columns)
    if not grouped_sensor_cols:
        raise ValueError("No valid meteo sensor data found in file.")
    return


# functions for outlier detection
# for certain sensor types, ensure there are no negative values
def replace_negative_values(series):
    """for certain sensor types, overwrite any negative values with zero"""
    is_negative = series.lt(0)
    n_negative = len(series.loc[is_negative])
    series.loc[is_negative] = 0.0
    return n_negative


def clip_sensor_outliers(series, sensor_group):
    """removes outliers from series; note: updates existing series object; does not return copy"""
    if sensor_group not in SENSOR_LIMITS:
        raise KeyError("Invalid sensor group")
    LOWER_BOUND, UPPER_BOUND = SENSOR_LIMITS[sensor_group]
    is_below = series.lt(LOWER_BOUND)
    is_above = series.gt(UPPER_BOUND)
    n_clipped_dict = {"above": len(series.loc[is_above]), "below": len(series.loc[is_below])}
    series = series.clip(lower=LOWER_BOUND, upper=UPPER_BOUND)
    return n_clipped_dict


# function to combine above (to be used in main script)
def process_sensor_outliers(series, sensor_group):
    output = {}
    if sensor_group in ["poa", "ghi", "wind"]:
        n_negative = replace_negative_values(series)
        if n_negative > 0:
            output["n_negative"] = n_negative
    n_clipped_dict = clip_sensor_outliers(series, sensor_group)
    if any(x > 0 for x in n_clipped_dict.values()):
        output["n_clipped"] = n_clipped_dict
    return output


def process_and_backfill_meteo_data(filepath, site, n_clearsky=5, r2_diff=0.1, q=True):
    """checks sensor data, detects and removes bad columns/data, then backfills using external data

    Parameters
    ----------
    filepath : str | pathlib.Path
        A filepath to a .csv file containing met station data for a particular solar site.
        If referenced path is to a standard "PIQuery_MetStations" file, bypasses DTN query when
        file exists for site TZ in flashreport dtn folder.
    site : str
        Name of solar site associated with met data.
    q : bool
        Quiet parameter; when False, enables status printouts. Defaults to True.

    Returns
    -------
    pandas.DataFrame
        A dataframe including the original met station data in addition to the processed/
        backfilled data and other related columns (e.g. solar position, DTN data, etc.)
    """
    qprint = quiet_print_function(q=q)
    alt, lat, lon, tz = site_location_metadata(site)

    fpath = Path(filepath)
    validate_meteo_filepath(fpath, site)  # raises ValueError if invalid

    # load file and identify sensor columns
    df_met = load_clean_meteo_file(site, fpath)  # localizes to site tz
    native_freq = df_met.index.to_series().diff().min()
    grouped_sensor_cols = grouped_meteo_columns(df_met.columns)
    sensor_cols = list(itertools.chain.from_iterable(grouped_sensor_cols.values()))
    df_met = df_met[sensor_cols].copy()

    # get start and end dates
    start_ = df_met.index.min().floor("D")
    end_ = df_met.index.max().ceil("D")
    start_date, end_date = map(lambda d: d.strftime("%Y-%m-%d"), [start_, end_])

    # load external data
    df_ext_backfill = None
    if "FlashReports" in fpath.parts:
        year, month = start_.year, start_.month
        if f"{year}-{month:02d}" not in fpath.name:
            raise ValueError("Problem validating Flashreport-related filepath.")
        # get external data either from existing file, or by querying (handled in function)
        df_ext = get_supporting_data(site, year, month)
        if native_freq == pd.Timedelta(minutes=1):
            df_ext_backfill = get_supporting_data(site, year, month, freq="1min", q=q)
    else:
        df_ext = load_supporting_data(site, start_date, end_date)

    # identify top N clearsky days
    clearsky_date_list = identify_clearsky_days_using_dtn_ghi(site, df_ext)
    if not clearsky_date_list:
        qprint(f"WARNING -- no clearsky days identified; using best {n_clearsky} days available")
        clearsky_date_list = identify_clearsky_days_using_dtn_ghi(site, df_ext, return_all=True)
    top_clearsky_dates = [item[0] for item in clearsky_date_list[:n_clearsky]]

    # create dictionary to track changes
    changes_dict = {}
    df = df_met.copy()  # create copy of original data

    ########################################
    # PART 1 - DETECT AND REMOVE BAD SENSORS
    # -> for POA and GHI via DTN comparison
    ########################################
    sensor_summaries = get_sensor_summaries(df, df_ext, top_clearsky_dates)
    bad_col_dict = bad_cols_from_sensor_summaries(sensor_summaries, r2_diff)
    all_bad_columns = []
    for key, dict_ in bad_col_dict.items():
        bad_cols, n_bad, n_all = dict_.values()
        pct_bad = n_bad / n_all
        qprint(f"\n{key} - {n_bad} of {n_all} ({pct_bad:.1%}) of sensors are bad - {bad_cols}")
        qprint(sensor_summaries[key][["r2_score"]].to_string())
        all_bad_columns.extend(bad_cols)

    if len(all_bad_columns) == len(sensor_cols):
        qprint("WARNING -- all existing sensor columns have been flagged as bad & removed.")
        return pd.DataFrame()

    # remove bad columns
    if all_bad_columns:
        qprint(
            f"\n\n>> REMOVING THE FOLLOWING ({len(all_bad_columns)}) COLUMNS  ->  {all_bad_columns}"
        )
        df = df.drop(columns=all_bad_columns)
        changes_dict["removed_columns"] = all_bad_columns
        grouped_sensor_cols = grouped_meteo_columns(df.columns)

    #########################################
    # PART 2 - OUTLIER DETECTION
    # -> outside normal range for sensor type
    #########################################
    removed_outliers = {}
    for sensor_group, sensor_columns in grouped_sensor_cols.items():
        group_outliers = {}
        for col in sensor_columns:
            series = df[col].copy()
            col_outliers = process_sensor_outliers(series, sensor_group)
            clean_col, bad_col = f"Cleaned_{col}", f"Bad_Data_{col}"
            df[clean_col] = series.copy()
            changed_idx = df[col].compare(df[clean_col]).index
            df[bad_col] = np.where(df.index.isin(changed_idx), 1, 0)
            df.loc[df[clean_col].isna(), bad_col] = 1
            if col_outliers:
                group_outliers[col] = col_outliers.copy()

        if group_outliers:
            removed_outliers[sensor_group] = group_outliers.copy()

    if removed_outliers:
        changes_dict["removed_outliers"] = removed_outliers

    # TODO: bad sensor half days? flag for backfill by half day to identify tracker issues

    # average across sensor columns by group to get avg. daily profile for each group
    grp_id_dict = {
        "poa": "POA",
        "ghi": "GHI",
        "mod_temp": "ModTemp",
        "amb_temp": "AmbTemp",
        "wind": "Wind",
    }
    profile_cols = []
    for sensor_group, sensor_columns in grouped_sensor_cols.items():
        cleaned_cols = [f"Cleaned_{c}" for c in sensor_columns]
        grp_id = grp_id_dict[sensor_group]
        df[f"Average_Across_{grp_id}"] = df[cleaned_cols].mean(axis=1).copy()
        profile_cols.append(f"{grp_id}_Monthly_Profile")

    avg_cols = list(filter(lambda c: c.startswith("Average_Across"), df.columns))
    avg_prof = df[avg_cols].groupby([df.index.month, df.index.hour]).mean()
    avg_prof.columns = profile_cols
    avg_prof = avg_prof.rename_axis(("Month", "Hour")).reset_index()

    # add average hourly profiles to df
    dff = pd.DataFrame(index=df.index)
    dff["Month"] = dff.index.month
    dff["Hour"] = dff.index.hour
    # df = df.merge(avg_prof, how="left", on=["Month", "Hour"], sort=True)

    df_avg_profile = dff.reset_index().merge(
        avg_prof,
        how="left",
        on=["Month", "Hour"],
        sort=True,
    )
    df_avg_profile = df_avg_profile.set_index("Timestamp")

    # add average hourly profiles to df
    for c in profile_cols:
        df[c] = df_avg_profile[c].copy()

    # determine ranges for backfill using "Bad_Data" columns
    for sensor_group, sensor_columns in grouped_sensor_cols.items():
        bad_sensor_cols = [f"Bad_Data_{c}" for c in sensor_columns]
        grp_id = grp_id_dict[sensor_group]
        df[f"{grp_id}_all_bad"] = df[bad_sensor_cols].prod(axis=1)

    # backfill data (NOTE: includes all groups in BACKFILL_COLUMNS)
    if df_ext_backfill is None:
        df_backfill = df_ext.copy()
        if native_freq < pd.Timedelta(hours=1):
            df_backfill = df_backfill.resample(native_freq).ffill()
    else:
        df_backfill = df_ext_backfill.copy()

    if len(df.index.difference(df_backfill.index)) == 0:
        if df_backfill.index.tz != df.index.tz:
            df_backfill = df_backfill.tz_convert(df.index.tz)  # otherwise, problem with join

    df = df.join(df_backfill)

    backfilled_data = {}
    for grp, fill_col in BACKFILL_COLUMNS.items():
        grp_id = grp_id_dict[grp]
        avg_col, processed_col = f"Average_Across_{grp_id}", f"Processed_{grp_id}"
        fill_data = df[fill_col].copy() if fill_col in df.columns else np.nan
        if avg_col in df.columns:
            df[processed_col] = np.where(df[f"{grp_id}_all_bad"].eq(0), df[avg_col], fill_data)
            changed_index = df[avg_col].compare(df[processed_col]).index
            n_filled = df.loc[df.index.isin(changed_index)].shape[0]
        else:
            df[processed_col] = fill_data
            n_filled = df.shape[0]

        if n_filled > 0 and grp in grouped_sensor_cols:
            backfilled_data[grp] = {"n_filled": n_filled}

    if backfilled_data:
        changes_dict["backfilled_data"] = backfilled_data

    return df, changes_dict
