from astral import LocationInfo
from astral.location import Location
from astral.sun import sun
import datetime as dt
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


def load_clean_meteo_file(site, filepath):
    """loads most recent clean met file & localizes to site tz"""
    df_ = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return localize_naive_datetimeindex(df_, site=site)


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
    alt, lat, lon, tz = site_location_metadata(site)
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

    df = query_DTN(lat, lon, dtn_start, dtn_end, interval, fields, tz=tz)
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


def get_supporting_data(site, year, month, q=True):
    """loads dtn data from files in flashreport dtn folder if exists, otherwise queries/saves"""
    if not oepaths.frpath(year, month).exists():
        raise ValueError("No FlashReport folder exists for selected year/month")

    dtn_dir = Path(oepaths.frpath(year, month), "DTN")
    if not dtn_dir.exists():
        dtn_dir.mkdir()

    tz = oemeta.data["TZ"].get(site)
    tz_id = "".join(filter(str.isalpha, tz))
    dtn_fpath = Path(dtn_dir, f"dtn_data_tz-{tz_id}.csv")
    if not dtn_fpath.exists():
        if not q:
            print("querying dtn data")
        start_ = pd.Timestamp(year, month, 1)
        end_ = start_ + pd.DateOffset(months=1)
        start_date, end_date = map(lambda d: d.strftime("%Y-%m-%d"), [start_, end_])
        df = load_supporting_data(site, start_date, end_date)
        df.to_csv(dtn_fpath)
        if not q:
            fpath_str = "\\".join(dtn_fpath.parts[-3:])
            print(f"saved file: {fpath_str}")
    else:
        df = pd.read_csv(dtn_fpath, index_col=0, parse_dates=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(
                df.index,
                format="%Y-%m-%d %H:%M:%S%z",
                utc=True,
            ).tz_convert(tz=tz)
        if not q:
            print("loaded from file")

    return df


def identify_clearsky_days_using_dtn_ghi(site, df):
    """returns list of tuples (day, r2_score) indicating r2 comparing dtn to clearsky, sorted greatest to least"""
    if "dtn_ghi" not in df.columns:
        raise ValueError("Input dataframe must have 'dtn_ghi' column.")
    ref_col, data_col = "ghi_clearsky", "dtn_ghi"
    df_clear = get_pvlib_clearsky(site, df.index)
    df[ref_col] = df_clear[ref_col].copy()

    clearsky_r2_vals = []
    for day in df.index.day.unique():
        df_ = df.loc[df.index.day == day, [ref_col, data_col]].dropna().copy()
        r2 = r2_score(df_[ref_col].values, df_[data_col].values)
        clearsky_r2_vals.append((day, r2))

    return list(sorted(clearsky_r2_vals, key=lambda item: item[1], reverse=True))


# TODO: modify to use dataframe for input instead of year/month (i.e. for dynamic start/end dates)
def evaluate_sensor_data(site, year, month):
    """loads clean met file, queries external data, then compares with poa/ghi columns"""
    met_fpath = get_meteo_fpath(site, year, month, version="cleaned")
    df_met = load_clean_meteo_file(site, met_fpath)  # note: tz-aware
    grouped_sensor_cols = grouped_meteo_columns(df_met.columns)

    df_ext = get_supporting_data(site, year, month)
    clearsky_day_list = identify_clearsky_days_using_dtn_ghi(site, df_ext)
    top_10_clearsky_days = [item[0] for item in clearsky_day_list[:10]]

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
            df_ = df.loc[df.index.day.isin(top_10_clearsky_days), [col, "REF"]].dropna().copy()
            r2_scores[col] = r2_score(df_["REF"].values, df_[col].values)
        dfr2 = pd.DataFrame({"r2_score": r2_scores.values()}, index=r2_scores.keys())
        avg_r2 = dfr2["r2_score"].mean()
        dfr2["%_delta_from_mean"] = dfr2["r2_score"].sub(avg_r2).div(avg_r2).mul(100)
        sensor_summaries[sensor_type] = dfr2.copy()

    return sensor_summaries


def get_sensor_summaries(df_met, df_ext, top_clearsky_days):
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
            df_ = df.loc[df.index.day.isin(top_clearsky_days), [col, "REF"]].dropna().copy()
            r2_scores[col] = r2_score(df_["REF"].values, df_[col].values)
        dfr2 = pd.DataFrame({"r2_score": r2_scores.values()}, index=r2_scores.keys())
        avg_r2 = dfr2["r2_score"].mean()
        dfr2["%_delta_from_mean"] = dfr2["r2_score"].sub(avg_r2).div(avg_r2).mul(100)
        sensor_summaries[sensor_type] = dfr2.copy()

    return sensor_summaries


def bad_cols_from_sensor_summaries(sensor_summaries):
    output = {}
    for key, df in sensor_summaries.items():
        bad_cols = df.loc[df["r2_score"].abs().lt(0.90)].index.to_list()
        output[key] = {"bad_columns": bad_cols, "n_bad": len(bad_cols), "n_all": df.shape[0]}
    return output


def process_and_backfill_meteo_data(filepath, site, q=True):
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

    if not fpath.exists():
        raise ValueError("Referenced filepath does not exist.")
    elif fpath.suffix != ".csv":
        raise ValueError("File must be CSV format.")

    # load file and identify sensor columns
    df = pd.read_csv(fpath, index_col=0, parse_dates=True)
    grouped_sensor_cols = grouped_meteo_columns(df.columns)
    if not grouped_sensor_cols:
        raise ValueError("No valid meteo sensor data found in file.")

    if not isinstance(df.index, pd.DatetimeIndex):
        # this can happen if .csv already has tz info & an error prevents parsing (e.g. dst)
        df.index = pd.to_datetime(
            df.index,
            format="%Y-%m-%d %H:%M:%S%z",
            utc=True,
        ).tz_convert(tz=tz)
    elif df.index.tz is None:
        qprint("localizing dataframe to site timezone")
        df = localize_naive_datetimeindex(df, site=site)

    # get start and end dates
    start_ = df.index.min().floor("D")
    end_ = df.index.max().ceil("D")
    start_date, end_date = map(lambda d: d.strftime("%Y-%m-%d"), [start_, end_])

    # load external data (either via query, or from existing file in flashreport directory)
    if "FlashReports" in fpath.parts:
        ym_folder = fpath.parts[fpath.parts.index("FlashReports") + 1]
        year, month = int(ym_folder[:4]), int(ym_folder[4:])
        if f"PIQuery_MetStations_{site}_{year}-{month:02d}" not in fpath.name:
            raise ValueError("Problem validating Flashreport-related filepath.")
        # get external data either from existing file, or by querying (handled in function)
        df_ext = get_supporting_data(site, year, month)
    else:
        df_ext = load_supporting_data(site, start_date, end_date)

    # determine number of clearsky days to use as reference for comparing sensor trends/magnitudes
    n_data_days = len(df.index.day.unique())
    n_ref_days = 10 if "FlashReports" in fpath.parts else int(np.ceil(n_data_days * 0.25))

    # identify top N clearsky days
    clearsky_day_list = identify_clearsky_days_using_dtn_ghi(site, df_ext)
    top_n_clearsky_days = [item[0] for item in clearsky_day_list[:n_ref_days]]

    # create dictionary to track changes
    changes_dict = {}
    dfx = df.copy()

    ############################
    # PART 1 - OUTLIER DETECTION
    ############################

    removed_outliers = {}
    for sensor_group, sensor_columns in grouped_sensor_cols.items():
        ll_, ul_ = SENSOR_LIMITS[sensor_group]

        # identify and record/log outliers
        outliers = pd.DataFrame(dfx[sensor_columns].lt(ll_).sum(), columns=["below"])
        outliers["above"] = dfx[sensor_columns].gt(ul_).sum()
        removed_outliers[sensor_group] = [(c, outliers.loc[c].to_dict()) for c in outliers.index]

        # remove outliers
        dfx[sensor_columns] = dfx[sensor_columns].clip(lower=ll_, upper=ul_)

    # fmt: {group_A: [(col_1, {'below': , 'above': }), (col_2, {'below': , 'above': }), ...], group_B: ...}
    changes_dict["removed_outliers"] = removed_outliers

    ###############################
    # PART 2 - IDENTIFY BAD SENSORS
    ###############################
    sensor_summaries = get_sensor_summaries(
        df_met=df,
        df_ext=df_ext,
        top_clearsky_days=top_n_clearsky_days,
    )
    for key, df_ in sensor_summaries.items():
        qprint(f"\n{key} sensors\n{df_.to_string()}")

    bad_col_dict = bad_cols_from_sensor_summaries(sensor_summaries)
    for key, dict_ in bad_col_dict.items():
        bad_cols, n_bad, n_all = dict_.values()
        pct_bad = n_bad / n_all
        qprint(f"{key} - {n_bad} of {n_all} ({pct_bad:.1%}) of sensors are bad - {bad_cols}")

    all_bad_columns = list(
        itertools.chain.from_iterable(dict_["bad_columns"] for dict_ in bad_col_dict.values())
    )
    changes_dict["removed_columns"] = bad_col_dict

    # remove bad sensors from grouped col dictionary (will not be used going forward)
    grouped_sensor_cols = {
        grp: list(filter(lambda c: c not in all_bad_columns, cols))
        for grp, cols in grouped_sensor_cols.items()
    }

    # bad sensor half days?
