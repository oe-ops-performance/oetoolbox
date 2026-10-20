from astral import LocationInfo
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

# from ..dataquery.external import query_DTN, query_fracsun
from ..datatools.qc import get_sun, get_daylight_timestamps
from ..utils import oemeta, oepaths
from ..utils.datetime import infer_freq, localize_naive_datetimeindex, upsample
from ..utils.helpers import quiet_print_function
from ..utils.solar import SolarDataset

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
    df = df.rename_axis("Timestamp")
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


# def get_daylight_timestamps(site, start, end, freq="1min"):
#     _, lat, lon, tz = site_location_metadata(site)
#     tstamps = []
#     for dict_ in get_suntimes(tz, lat, lon, start, end):
#         dawn_ = dict_["dawn"].floor(freq)
#         dusk_ = dict_["dusk"].ceil(freq)
#         tstamps.extend(list(pd.date_range(dawn_, dusk_, freq=freq)))
#     return tstamps


# def segmented_daylight_timestamps(site, start, end, freq="1min"):
#     """returns dictionary with str(day) as keys and [am_tstamps, pm_tstamps] as values"""

#     daylight_tstamps = get_daylight_timestamps(site, start, end, freq=freq)


def get_surface_parameters(site, df_met):
    """Returns the tilt/azimuth parameters (used for POA transposition from GHI data)"""
    design_db = oemeta.data["Project_Design_Database"].get(site)
    dict_inv = list(design_db.values())[0]
    racking_tilt = int(dict_inv["Tilt Angle (Deg)"])

    if dict_inv.get("Racking Type") != "Tracker":  # same for all inv
        return racking_tilt, 180

    tracking_profile = pvlib.tracking.singleaxis(
        df_met["apparent_zenith"],
        df_met["azimuth"],
        max_angle=racking_tilt,
        backtrack=True,
        gcr=dict_inv["Racking GCR"],
    )
    return tracking_profile["surface_tilt"], tracking_profile["surface_azimuth"]


def get_pvlib_clearsky(site, datetime_index):
    """returns df with columns: ghi_clearsky, dni_clearsky, dhi_clearsky"""
    alt, lat, lon, tz = site_location_metadata(site)
    location = pvlib.location.Location(lat, lon, tz, alt, site)
    return location.get_clearsky(datetime_index).add_suffix("_clearsky")


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


# def get_fracsun_soiling_data(site, start_date, end_date, with_tz=True):
#     """returns fracsun soiling data in dataframe formatted for use in evaluate_sensor_data"""


# TODO: modify to use dataframe for input instead of year/month (i.e. for dynamic start/end dates)
def evaluate_sensor_data(site, year, month, n_clearsky_days=7, q=True):
    """loads clean met file, queries external data, then compares with poa/ghi columns"""
    qprint = quiet_print_function(q=q)
    met_fpath = get_meteo_fpath(site, year, month, version="cleaned")
    df_met = load_clean_meteo_file(site, met_fpath)  # note: tz-aware
    grouped_sensor_cols = grouped_meteo_columns(df_met.columns)

    # df_ext = get_supporting_data(site, year, month)
    df_ext = SolarDataset.get_supporting_data(site, year, month, q=q)
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
            df_ = (
                df.loc[pd.Series(df.index.date).isin(top_clearsky_dates).values, [col, "REF"]]
                .dropna()
                .copy()
            )
            r2_scores[col] = r2_score(df_["REF"].values, df_[col].values)
        dfr2 = pd.DataFrame({"r2_score": r2_scores.values()}, index=r2_scores.keys())
        avg_r2 = dfr2["r2_score"].mean()
        dfr2["%_delta_from_mean"] = dfr2["r2_score"].sub(avg_r2).div(avg_r2).mul(100)
        sensor_summaries[sensor_type] = dfr2.copy()

    return sensor_summaries


def get_sensor_summaries(df_met, df_ext, top_clearsky_dates, df_soiling=None):
    """NOTE: df_ext should be hourly data"""
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


def flag_bad_sensor_days(
    df,
    df_ext,
    grouped_sensor_cols,
    clear_dates,
    cloudy_dates,
    dtn_r2_diff=0.1,
    peer_r2_diff=0.1,
    min_points=6,
    min_peer_sensors=2,
    q=True,
):
    """Per-day QC for POA/GHI sensors. Mutates df in place (Cleaned_/Bad_Data_ columns).

    Instead of dropping a sensor for the whole month, this drops individual (sensor, day)
    pairs where the sensor deviates too far from a trustworthy reference:

    Clear days : drop a sensor-day if its daily r2 vs the DTN reference
                 (poa_global / dtn_ghi) is < 1 - dtn_r2_diff.
    Cloudy days: for groups with >= min_peer_sensors sensors, drop a sensor-day if its daily
                 r2 vs the median of the OTHER sensors in the group is < 1 - peer_r2_diff.
                 (DTN is unreliable under clouds, so same-site peers are used instead.)

    A dropped sensor-day is NaN'd in Cleaned_<col> and flagged in Bad_Data_<col>, so the
    downstream group average excludes it that day and the DTN backfill only kicks in where
    every sensor in the group is bad.

    Returns {"dtn": {col: [dates]}, "peer": {col: [dates]}}.
    """
    qprint = quiet_print_function(q=q)
    REFERENCE_COLS = {"poa": "poa_global", "ghi": "dtn_ghi"}
    dropped = {"dtn": {}, "peer": {}}

    def _drop_day(col, date, bucket):
        rows = df.index.date == date
        df.loc[rows, f"Cleaned_{col}"] = np.nan
        df.loc[rows, f"Bad_Data_{col}"] = 1
        dropped[bucket].setdefault(col, []).append(date)

    for sensor_type, ref_col in REFERENCE_COLS.items():
        if sensor_type not in grouped_sensor_cols:
            continue
        cols = grouped_sensor_cols[sensor_type]
        # hourly snapshot of the group's cleaned data (taken before this group's drops so
        # peer medians are order-independent and unaffected by same-loop mutations)
        dfh = df[[f"Cleaned_{c}" for c in cols]].resample("h").mean()
        dfh.columns = cols

        # --- clear days: compare each sensor to the DTN reference ---
        if clear_dates and ref_col in df_ext.columns:
            dfr = dfh.copy()
            dfr["REF"] = df_ext[ref_col]  # aligns on hourly index (as in get_sensor_summaries)
            for date in clear_dates:
                block = dfr.loc[dfr.index.date == date]
                for col in cols:
                    sub = block[[col, "REF"]].dropna()
                    if len(sub) < min_points:
                        continue
                    if r2_score(sub["REF"].values, sub[col].values) < (1 - dtn_r2_diff):
                        _drop_day(col, date, "dtn")

        # --- cloudy days: compare each sensor to the median of its peers ---
        if cloudy_dates and len(cols) >= min_peer_sensors:
            for date in cloudy_dates:
                block = dfh.loc[dfh.index.date == date]
                for col in cols:
                    others = [c for c in cols if c != col]
                    ref = block[others].median(axis=1)
                    sub = pd.concat([block[col], ref], axis=1, keys=["s", "ref"]).dropna()
                    if len(sub) < min_points:
                        continue
                    if r2_score(sub["ref"].values, sub["s"].values) < (1 - peer_r2_diff):
                        _drop_day(col, date, "peer")

    for bucket, label in (("dtn", "clear/DTN"), ("peer", "cloudy/peer")):
        for col, dates in dropped[bucket].items():
            qprint(f"  per-day drop [{label}]: {col} -> {len(dates)} day(s)")
    return dropped


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


def backfill_meteo_data(df_met, df_ext, site, n_clearsky=5, r2_diff=0.1, q=True):
    qprint = quiet_print_function(q=q)
    if df_met.index.tzinfo is None:
        raise ValueError("Input df index must have tzinfo.")
    native_freq = df_met.index.to_series().diff().min()
    grouped_sensor_cols = grouped_meteo_columns(df_met.columns)
    sensor_cols = list(itertools.chain.from_iterable(grouped_sensor_cols.values()))
    df_met = df_met[sensor_cols]

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
    bad_col_dict = bad_cols_from_sensor_summaries(sensor_summaries, r2_diff=r2_diff)
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

    original_cols = list(df.columns)

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
    df_backfill = df_ext.copy()
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

    processed_cols = list(df.filter(like="Processed_").columns)
    avgcol = lambda pcol: "Average_Across_" + pcol.replace("Processed_", "")
    valid_processed_cols = [c for c in processed_cols if avgcol(c) in df.columns]
    df = df[original_cols + valid_processed_cols]
    return df, changes_dict


def process_and_backfill_meteo_data(
    filepath,
    site,
    n_clearsky=5,
    r2_diff=0.1,
    per_day_qc=True,
    daily_r2_diff=0.1,
    peer_r2_diff=0.1,
    min_daily_points=6,
    min_peer_sensors=2,
    exclude_sensors=None,
    q=True,
):
    """checks sensor data, detects and removes bad columns/data, then backfills using external data

    Parameters
    ----------
    filepath : str | pathlib.Path
        A filepath to a .csv file containing met station data for a particular solar site.
        If referenced path is to a standard "PIQuery_MetStations" file, bypasses DTN query when
        file exists for site TZ in flashreport dtn folder.
    site : str
        Name of solar site associated with met data.
    n_clearsky : int
        Number of clearest-sky days used for the whole-month (PART 1) sensor r2 check.
    r2_diff : float
        Threshold for the whole-month (PART 1) drop; a sensor's aggregate clearsky r2 below
        (1 - r2_diff) removes the entire sensor column for the month.
    per_day_qc : bool
        When True, additionally drops individual (sensor, day) pairs that deviate too far from
        a reference (DTN on clear days, same-type peers on cloudy days). See flag_bad_sensor_days.
    daily_r2_diff : float
        Per-day threshold for the clear-day DTN comparison (drop day if r2 < 1 - daily_r2_diff).
    peer_r2_diff : float
        Per-day threshold for the cloudy-day peer-median comparison (drop day if r2 < 1 - peer_r2_diff).
    min_daily_points : int
        Minimum hourly points required in a day before its per-day r2 is evaluated.
    min_peer_sensors : int
        Minimum sensors in a group before the cloudy-day peer check runs (3 avoids the symmetric
        2-sensor case, where a disagreement flags both sensors and the day falls back to DTN).
    exclude_sensors : list[str] | None
        List of exact met-station column names to force-drop for the whole month before QC
        (any sensor type). Names that match no column are ignored with a printed note.
    q : bool
        Quiet parameter; when False, enables status printouts. Defaults to True.

    Returns
    -------
    pandas.DataFrame
        A dataframe including the original met station data in addition to the processed/
        backfilled data and other related columns (e.g. solar position, DTN data, etc.)
    """
    qprint = quiet_print_function(q=q)
    # alt, lat, lon, tz = site_location_metadata(site)

    fpath = Path(filepath)
    validate_meteo_filepath(fpath, site)  # raises ValueError if invalid

    # load file and identify sensor columns
    df_met = load_clean_meteo_file(site, fpath)  # localizes to site tz
    native_freq = df_met.index.to_series().diff().min()
    grouped_sensor_cols = grouped_meteo_columns(df_met.columns)
    sensor_cols = list(itertools.chain.from_iterable(grouped_sensor_cols.values()))

    # manual override: force-drop user-specified sensor columns (whole month, any type) before QC
    excluded_columns = [c for c in (exclude_sensors or []) if c in sensor_cols]  # exact match
    if exclude_sensors:
        missing = [c for c in exclude_sensors if c not in sensor_cols]
        if missing:
            qprint(f"Note: exclude_sensors not found (ignored): {missing}")
        if excluded_columns:
            qprint(f"Excluding user-specified sensors: {excluded_columns}")
            sensor_cols = [c for c in sensor_cols if c not in excluded_columns]
            grouped_sensor_cols = grouped_meteo_columns(sensor_cols)

    df_met = df_met[sensor_cols].copy()

    # get start and end dates
    start_ = df_met.index.min().floor("D")
    end_ = df_met.index.max().ceil("D")
    start_date, end_date = map(lambda d: d.strftime("%Y-%m-%d"), [start_, end_])

    # load external data
    if "FlashReports" in fpath.parts:
        year, month = start_.year, start_.month
        if f"{year}-{month:02d}" not in fpath.name:
            raise ValueError("Problem validating Flashreport-related filepath.")
        # get external data either from existing file, or by querying (handled in function)
        df_ext = SolarDataset.get_supporting_data(site, year, month, q=q)
    else:
        # df_ext = load_supporting_data(site, start_date, end_date)
        df_ext = SolarDataset.from_dtn_files(site, start_date=start_date, end_date=end_date)

    if native_freq == pd.Timedelta(minutes=1):
        df_backfill = df_ext.resample("1min").ffill().copy()
        df_backfill = df_backfill.reindex(df_met.index)
    else:
        df_backfill = df_ext.copy()

    # identify top N clearsky days
    clearsky_date_list = identify_clearsky_days_using_dtn_ghi(site, df_ext)
    # capture genuinely-clear days (before any fallback) for the per-day DTN check; if none,
    # clear_dates stays empty and every day is treated as cloudy (peer check still applies)
    clear_dates = [d for d, _ in clearsky_date_list]
    if not clearsky_date_list:
        qprint(f"WARNING -- no clearsky days identified; using best {n_clearsky} days available")
        clearsky_date_list = identify_clearsky_days_using_dtn_ghi(site, df_ext, return_all=True)
    top_clearsky_dates = [item[0] for item in clearsky_date_list[:n_clearsky]]

    # create dictionary to track changes
    changes_dict = {}
    if excluded_columns:
        changes_dict["excluded_columns"] = excluded_columns
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

    #############################################
    # PART 2B - PER-DAY SENSOR QC (POA/GHI)
    # -> drop specific bad (sensor, day) pairs
    #    (DTN ref on clear days, peers on cloudy)
    #############################################
    if per_day_qc:
        all_dates = sorted(set(df.index.date))
        clear_date_set = set(clear_dates)
        cloudy_dates = [d for d in all_dates if d not in clear_date_set]
        dropped = flag_bad_sensor_days(
            df,
            df_ext,
            grouped_sensor_cols,
            clear_dates,
            cloudy_dates,
            dtn_r2_diff=daily_r2_diff,
            peer_r2_diff=peer_r2_diff,
            min_points=min_daily_points,
            min_peer_sensors=min_peer_sensors,
            q=q,
        )
        fmt = lambda dd: {c: [str(x) for x in ds] for c, ds in dd.items()}
        if dropped["dtn"]:
            changes_dict["removed_sensor_days_dtn"] = fmt(dropped["dtn"])
        if dropped["peer"]:
            changes_dict["removed_sensor_days_peer"] = fmt(dropped["peer"])

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
    # df_backfill = df_ext.copy()
    # if df_ext_backfill is None:
    #     df_backfill = df_ext.copy()
    #     if native_freq < pd.Timedelta(hours=1):
    #         df_backfill = df_backfill.resample(native_freq).ffill()
    # else:
    #     df_backfill = df_ext_backfill.copy()

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


def r2_by_day(df: pd.DataFrame, data_col: str, ref_col: str) -> pd.DataFrame:
    data_list = []  # cols: date, r2_score
    for date in sorted(set(df.index.date)):
        df_ = df[df.index.date == date].dropna(axis=0, how="any").copy()
        if df_[[data_col]].empty:
            r2_ = None
        else:
            r2_ = r2_score(y_true=df_[ref_col].values, y_pred=df_[data_col].values)
        data_list.append([date, r2_])
    df = pd.DataFrame(data_list, columns=["date", "r2_score"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def r2_by_half_day(
    df: pd.DataFrame, data_col: str, ref_col: str, location_kwargs: dict
) -> pd.DataFrame:
    freq_td = infer_freq(df.index)
    is_daylight = _daylight_filter(df, location_kwargs)
    data_list = []  # cols: date, start (0/noon), period (am/pm), r2_score
    for date in sorted(set(df.index.date)):
        sun_ = get_sun(date=date, **location_kwargs)
        noon_ = sun_["noon"]
        noon_str = noon_.strftime("%Y-%m-%d %H:%M:%S")  # avoid potential tz issues
        matching_date = is_daylight & (df.index.date == date)
        is_am = matching_date & (df.index < noon_str)
        is_pm = matching_date & (df.index >= noon_str)
        for period, filter_cond in zip(["am", "pm"], [is_am, is_pm]):
            df_ = df[filter_cond].copy()
            notna = df_[[data_col, ref_col]].notna().all(axis=1)
            df_ = df_[notna]
            if df_[[data_col]].empty:
                masd_ = 1000.0
                r2_ = None
            else:
                mcond = df_.index.hour.isin(range(10, 15))
                masd_ = df_.loc[mcond, data_col].diff().abs().mean()
                r2_ = r2_score(y_true=df_[ref_col].values, y_pred=df_[data_col].values)
            start_ = pd.Timestamp(date) if period == "am" else noon_.round(freq_td)
            data_list.append([date, start_, period, r2_, masd_])
    df = pd.DataFrame(data_list, columns=["date", "start", "period", "r2_score", "MASD"])
    df["date"] = pd.to_datetime(df["date"])
    df["start"] = pd.to_datetime(df["start"])
    return df.set_index("date")


def formatted_r2_series(
    df_r2: pd.DataFrame, ref_index: pd.DatetimeIndex, stat_col="r2_score"
) -> pd.Series:
    if all(c in df_r2.columns for c in ["period", "start"]):
        # need to resample, otherwise index.map misses the 'pm' values
        freq_td = infer_freq(ref_index)
        df = df_r2.set_index("start")[[stat_col]].copy()
        df.loc[df.index[-1].ceil("D")] = df.iloc[-1, :].values
        df = df.resample(freq_td).ffill().iloc[:-1, :].copy()
        return ref_index.tz_localize(None).map(df[stat_col])
    return ref_index.tz_localize(None).normalize().map(df_r2[stat_col])


BACKFILL_COL_MAPPING = {
    "OE.Ambient_Temp": "dtn_temp_air",
    "OE.POA": "poa_global",
    "OE.Wind_Speed": "dtn_wind_speed",
}


def _grouped_sensor_cols(columns) -> dict:
    output = {}
    matching_cols_ = lambda key_: [
        c
        for c in columns
        if c.startswith(key_) and not any(x in c.lower() for x in ["_r2_", "average", "masd"])
    ]
    for key in BACKFILL_COL_MAPPING.keys():
        cols = matching_cols_(key)
        if len(cols) > 0:
            output.update({key: cols})
    return output


def _process_backfill_input_data(df_sensors, df_dtn, meter_data=None, q=True) -> tuple:
    """Returns grouped sensor_cols dict and combined dataframe."""
    qprint = quiet_print_function(q=q)

    if any(df.index.tzinfo is None for df in [df_sensors, df_dtn]):
        raise ValueError("All data must be timezone-aware.")
    if meter_data is not None:
        if meter_data.index.tzinfo is None:
            raise ValueError("All data must be timezone-aware.")

    check_dfs = [df_sensors, df_dtn]
    if meter_data is not None:
        check_dfs.append(meter_data)
    if len(set(x.index[0].utcoffset() for x in check_dfs)) != 1:
        raise ValueError("All data must be in the same timezone.")

    sensor_cols = _grouped_sensor_cols(df_sensors.columns)
    # if not sensor_cols:
    #     raise KeyError("No valid sensor columns identified in dataset.")

    for key, dtn_col in BACKFILL_COL_MAPPING.items():
        if key in sensor_cols.keys() and dtn_col not in df_dtn.columns:
            qprint(f"Note: missing reference DTN column '{dtn_col}'; excluding group '{key}'")
            del sensor_cols[key]
    # if not sensor_cols:
    #     raise KeyError("No valid sensor columns identified in dataset.")

    df_dtn = df_dtn[list(BACKFILL_COL_MAPPING.values())]

    if "OE.POA" in sensor_cols.keys() and meter_data is None:
        qprint("Note: optional 'meter_data' not provided - no secondary r2 check for POA sensors.")

    df = _add_dtn_to_sensor_data(df_sensors, df_dtn, meter_data)

    return df, sensor_cols


def _daylight_filter(df, location_kwargs):
    start_date = df.index[0].strftime("%Y-%m-%d")
    end_date = df.index[-1].ceil("D").strftime("%Y-%m-%d")
    freq = infer_freq(df.index)
    daylight_tstamps = get_daylight_timestamps(
        start_date,
        end_date,
        freq=freq,
        **location_kwargs,
    )
    return df.index.tz_localize(None).isin(daylight_tstamps)


def _generate_r2_scores(df_combined: pd.DataFrame, sensor_cols: dict, location_kwargs: dict):
    df = df_combined.copy()
    for stype, cols in sensor_cols.items():
        dtn_col = BACKFILL_COL_MAPPING[stype]
        secondary_r2 = "POA" in stype and "site_meter" in df.columns
        for data_col in cols:
            if df[data_col].isna().sum() / len(df) > 0.95:
                df[f"{data_col}_r2_dtn"] = 0.0
                continue

            df_r2 = r2_by_half_day(df, data_col, ref_col=dtn_col, location_kwargs=location_kwargs)
            df[f"{data_col}_r2_dtn"] = formatted_r2_series(df_r2=df_r2, ref_index=df.index)
            if secondary_r2 is True:
                dfn = df[[data_col, "site_meter"]].copy()
                dfn = (dfn - dfn.min()) / (dfn.max() - dfn.min())  # normalized: 0 to 1
                norm_r2 = r2_by_half_day(
                    dfn, data_col, ref_col="site_meter", location_kwargs=location_kwargs
                )
                df[f"{data_col}_r2_meter"] = formatted_r2_series(df_r2=norm_r2, ref_index=df.index)
                df[f"{data_col}_r2_meter"] = df[f"{data_col}_r2_meter"].clip(lower=0.0)
            df[f"{data_col}_MASD"] = formatted_r2_series(
                df_r2=df_r2, ref_index=df.index, stat_col="MASD"
            )
    return df


def _flagged_ranges(df, flag_condition, min_length=None):
    dfx = df[flag_condition].copy()
    dfx["time_diff"] = dfx.index.diff()
    if dfx.shape[0] > 1:
        dfx.iloc[0, -1] = dfx.index[1] - dfx.index[0]

    step = infer_freq(df.index)
    conditions = dfx["time_diff"].eq(step)
    breaks = dfx["time_diff"].gt(step).cumsum()
    length_condition = lambda df_: True if min_length is None else len(df_) >= min_length
    return [df_.index for _, df_ in dfx[conditions].groupby(breaks) if length_condition(df_)]


def _generate_ranges_to_remove_by_col(df_all, sensor_cols, location_kwargs, r2_threshold, q=True):
    qprint = quiet_print_function(q=q)
    output = {}
    df = _generate_r2_scores(df_all.resample("h").mean(), sensor_cols, location_kwargs)
    for pfx, columns in sensor_cols.items():
        for col in columns:
            no_data = df[col].isna()
            off_from_dtn = df[f"{col}_r2_dtn"].lt(r2_threshold)
            if f"{col}_r2_meter" in df.columns:
                similar_to_meter = df[f"{col}_r2_meter"].gt(r2_threshold)
                cloudy_conditions = df[f"{col}_MASD"].gt(120.0)  # threshold for hourly data
                close_enough_to_keep = similar_to_meter & cloudy_conditions
            else:
                close_enough_to_keep = df.index.notna()  # should be False everywhere

            backfill_condition = no_data | (off_from_dtn & ~close_enough_to_keep)
            if not backfill_condition.any():
                continue

            ranges_to_remove = _flagged_ranges(df, backfill_condition, min_length=2)
            output[col] = ranges_to_remove

    return output


def backfill_weather_sensor_data(
    df_sensors: pd.DataFrame,
    df_dtn: pd.DataFrame,
    location_kwargs: dict,
    meter_data: pd.Series = None,
    r2_threshold: float = 0.75,
    q: bool = True,
):
    qprint = quiet_print_function(q=q)
    qprint("\nBegin weather data backfill.")
    df, sensor_cols = _process_backfill_input_data(df_sensors, df_dtn, meter_data, q=q)
    ranges_to_remove_by_col = _generate_ranges_to_remove_by_col(
        df, sensor_cols, location_kwargs, r2_threshold, q=q
    )
    for pfx, columns in sensor_cols.items():
        qprint(f"\nSensor group: {pfx}")
        for col in columns:
            ranges_to_remove = ranges_to_remove_by_col.get(col, [])
            if not ranges_to_remove:
                qprint(f" -> {col = }; no data removed.")
                continue
            n_removed = 0
            for idx in ranges_to_remove:
                in_range = (df.index >= idx.min()) & (df.index < idx.max())
                df.loc[in_range, col] = np.nan
                n_removed += in_range.sum()
            if n_removed == len(df):
                qprint(f" -> removed {col = }.")
            else:
                qprint(f" -> {col = }; removed {n_removed} values.")

        df[f"{pfx}_average"] = df[columns].mean(axis=1).copy()
        fill_condition = df[f"{pfx}_average"].isna()
        if not fill_condition.any():
            df[pfx] = df[f"{pfx}_average"].copy()
            qprint(f">>> no backfill needed; using average across sensor data.")
            continue
        n_fill = fill_condition.sum()
        ref_col = BACKFILL_COL_MAPPING[pfx]
        df[pfx] = np.where(fill_condition, df[ref_col], df[f"{pfx}_average"])
        qprint(f">>> backfilled {n_fill} values with DTN data ('{ref_col}')")

    for pfx, ref_col in BACKFILL_COL_MAPPING.items():
        if pfx not in df.columns:
            qprint(f"\n>>> added column '{pfx}' using DTN data ('{ref_col}')")
            df[pfx] = df[ref_col].copy()
    processed_cols = list(BACKFILL_COL_MAPPING.keys())
    qprint("\nEnd backfill.")
    return {"df": df.copy(), "backfilled": df[processed_cols].copy()}


def _add_dtn_to_sensor_data(
    df_sensors: pd.DataFrame, df_dtn: pd.DataFrame, meter_data: pd.Series = None
) -> pd.DataFrame:
    # all input data is tz-aware
    sensor_freq = infer_freq(df_sensors.index)
    dtn_freq = infer_freq(df_dtn.index)  # should be hourly
    if sensor_freq > dtn_freq:
        raise ValueError("Unsupported datetime index for sensor data.")

    # resolve potential timezone mismatches (i.e. named tz vs utc-offset)
    ref_tzinfo = df_sensors.index.tzinfo
    ref_offset = df_sensors.index[0].utcoffset()
    different_tz = lambda df: df.index.tzinfo != ref_tzinfo
    same_offset = lambda df: df.index[0].utcoffset() == ref_offset
    need_to_convert = lambda df: different_tz(df) and same_offset(df)
    if need_to_convert(df_dtn):
        df_dtn = df_dtn.tz_convert(ref_tzinfo)

    # upsample DTN data to match sensor data (if needed)
    if dtn_freq > sensor_freq:
        # df_dtn = df_dtn.resample(sensor_freq).interpolate().reindex(df_sensors.index)
        df_dtn = upsample(df=df_dtn, new_freq=sensor_freq, agg_type="interpolate")
        df_dtn["poa_global"] = df_dtn["poa_global"].fillna(0.0)

    df_list = [df_sensors, df_dtn]

    # similar steps for meter data (if provided)
    if meter_data is not None:
        if need_to_convert(meter_data):
            meter_data = meter_data.tz_convert(ref_tzinfo)
        mfreq = infer_freq(meter_data.index)
        if mfreq > sensor_freq:
            meter_data = upsample(meter_data.to_frame(), sensor_freq, agg_type="interpolate").iloc[
                :, 0
            ]
            meter_data = meter_data.fillna(0.0)
        elif mfreq < sensor_freq:
            meter_data = meter_data.resample(sensor_freq).mean()
        df_meter = meter_data.to_frame(name="site_meter")
        df_list.append(df_meter)

    return pd.concat(df_list, axis=1)


abbr_mapping = {"OE.Ambient_Temp": "AmbTemp", "OE.POA": "POA", "OE.Wind_Speed": "Wind"}


def format_backfill_data_to_match_legacy(df_result):
    # df is from output of backfill_weather_sensor_data
    df = df_result.copy()
    for col in ["OE.Ambient_Temp", "OE.POA", "OE.Wind_Speed"]:
        sensor_abbr = abbr_mapping[col]
        avgcol = f"{col}_average"
        if avgcol in df.columns:
            df[f"Average_Across_{sensor_abbr}"] = df[avgcol].copy()
        df[f"Processed_{sensor_abbr}"] = df[col].copy()
    return df


def _get_backfilled_ranges_by_sensor_type(df_formatted) -> dict[str, list[pd.DatetimeIndex]]:
    # df_formatted is output from format_backfill_data_to_match_legacy
    output = {}
    df = df_formatted.copy()
    for oe_sensor, abbr in abbr_mapping.items():
        pcol = f"Processed_{abbr}"
        if pcol not in df.columns:
            continue
        dtn_col = BACKFILL_COL_MAPPING[oe_sensor]
        dfx = df[df[pcol] == df[dtn_col]].copy()
        dfx["time_diff"] = dfx.index.diff()
        if dfx.shape[0] > 1:
            dfx.iloc[0, -1] = dfx.index[1] - dfx.index[0]

        step = infer_freq(df.index)
        conditions = dfx["time_diff"].eq(step)
        breaks = dfx["time_diff"].gt(step).cumsum()

        output[oe_sensor] = [df_.index for _, df_ in dfx[conditions].groupby(breaks)]
    return output


def backfill_subplots(df, resample=True):
    sensor_cols = _grouped_sensor_cols(df.columns)
    n_rows = len(sensor_cols)
    s_titles = [f"<b>{grp}</b>" for grp in sensor_cols.keys()]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=s_titles,
        vertical_spacing=0.2 / n_rows,
    )
    htemplate = (
        "<b>%{fullData.name}</b><br>Value: %{y:.2f}<br>" "<i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
    )

    backfilled_ranges = _get_backfilled_ranges_by_sensor_type(df)

    if resample is True and infer_freq(df.index) < pd.Timedelta(minutes=15):
        df = df.resample("15min").mean().copy()
    freq_td = infer_freq(df.index)

    shapes = []
    for row, item in enumerate(BACKFILL_COL_MAPPING.items(), start=1):
        grp, dtn_col = item
        if grp not in sensor_cols:
            continue
        grp_kwargs = dict(legendgroup=grp, legendgrouptitle_text=f"<b>{grp}</b>", legendrank=row)

        dtn_trace = go.Scattergl(
            x=df.index,
            y=df[dtn_col],
            name=dtn_col,
            mode="lines",
            line=dict(color="#777777"),
            hovertemplate=htemplate,
            **grp_kwargs,
        )

        sensor_traces = []
        for col in sensor_cols[grp]:
            sensor_traces.append(
                go.Scattergl(
                    x=df.index,
                    y=df[col],
                    name=col,
                    mode="lines",
                    opacity=0.5,
                    **grp_kwargs,
                )
            )

        processed_col = "Processed_" + abbr_mapping[grp]

        data_trace = go.Scattergl(
            x=df.index,
            y=df[processed_col],
            mode="lines",
            name=processed_col,
            line=dict(color="#000000", dash="dot"),
            **grp_kwargs,
        )

        fig.add_traces([dtn_trace, *sensor_traces, data_trace], rows=row, cols=1)

        for idx in backfilled_ranges[grp]:
            shapes.append(
                dict(
                    type="rect",
                    x0=idx.min().floor(freq_td),
                    x1=idx.max().ceil(freq_td),
                    y0=0,
                    y1=1,
                    yref=f"y{row} domain".replace("y1", "y"),
                    fillcolor="lightsalmon",
                    opacity=0.4,
                    line_width=0,
                    layer="below",
                )
            )

    fig.update_xaxes(tickformat="%m/%d", tick0=df.index[0], dtick=str(86400000 * 7))
    fig.update_layout(
        font_size=9,
        paper_bgcolor="#fbfbfb",
        margin=dict(t=30, r=20, b=10, l=20),
        legend=dict(
            x=1.01,
            xanchor="left",
            groupclick="toggleitem",
            grouptitlefont=dict(size=11),
            tracegroupgap=16,
        ),
        shapes=shapes,
    )

    # format subplot titles (are actually annotations in subplot figures)
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].update(font=dict(size=12), x=0, xanchor="left")

    return fig


def _backfill_key(data_column: str) -> str:
    matching_keys = [k for k in BACKFILL_COL_MAPPING.keys() if k in data_column]
    if len(matching_keys) != 1:
        raise KeyError(f"Could not find match for {data_column = }.")
    return matching_keys[0]


def meteo_backfill_subplots(df_, resample=False):
    df = df_.copy()
    freq = None  # init
    if resample is True:
        freq = "1h"
    elif resample in ["h", "1h", "15min"]:
        freq = resample if any(x.isdigit() for x in resample) else f"1{resample}"
    if freq is not None:
        inferred_freq = pd.infer_freq(df.index)
        if not inferred_freq:
            print("Error: could not resample dataframe")
            return
        if not any(x.isdigit() for x in inferred_freq):
            inferred_freq = f"1{inferred_freq}"
        if pd.Timedelta(inferred_freq) < pd.Timedelta(freq):
            df = df.resample(freq).mean().copy()
    dfcols = list(df.columns)
    if "Hour" in dfcols:
        sensor_cols = dfcols[: dfcols.index("Hour")]
    else:
        cln_cols = [(i, col) for i, col in enumerate(df.columns) if col.startswith("Cleaned")]
        sensor_cols = list(df.columns)[: cln_cols[0][0]]
    grouped_sensor_cols = grouped_meteo_columns(sensor_cols)
    grp_ID_dict = {
        "poa": "POA",
        "ghi": "GHI",
        "mod_temp": "ModTemp",
        "amb_temp": "AmbTemp",
        "wind": "Wind",
    }
    grp_IDs = [grp_ID_dict[grp] for grp in grouped_sensor_cols]

    n_rows = len(grouped_sensor_cols)
    s_titles = [f"<b>{grpID} sensor data</b>" for grpID in grp_IDs]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=s_titles,
        vertical_spacing=0.2 / n_rows,
    )

    htemplate = (
        "<b>%{fullData.name}</b><br>Value: %{y:.2f}<br>" "<i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
    )
    group_external_reference_dict = {
        "poa": ["poa_global"],
        "ghi": ["dtn_ghi"],
        "amb_temp": ["dtn_temp_air"],
        "wind": ["dtn_wind_speed"],
    }
    kwargs = dict(x=df.index, mode="lines", hovertemplate=htemplate)

    for grp, grp_cols in grouped_sensor_cols.items():
        grpID = grp_ID_dict[grp]
        row_ = grp_IDs.index(grpID) + 1
        grp_kwargs = dict(
            legendgroup=grpID, legendgrouptitle_text=f"<b>{grpID}</b>", legendrank=row_
        )

        proc_col = f"Processed_{grpID}"
        bad_col = f"{grpID}_all_bad"

        # add trace with shading to show backfilled sections (i.e. where all sensors had bad/missing data)
        yvals = df[proc_col].mask(df[bad_col].eq(0), 0)
        fig.add_trace(
            go.Scattergl(
                **kwargs,
                **grp_kwargs,
                y=yvals,
                name=f"Backfilled {grpID}",
                line_width=0,
                fill="tozeroy",
                fillcolor="rgba(255,191,0,0.3)",
            ),
            row=row_,
            col=1,
        )

        ## NEW ## -- add reference trace(s) for comparison to external data
        reference_cols = group_external_reference_dict.get(grp)
        if reference_cols:
            dfr = df.resample("h").mean().copy()
            for col in reference_cols:
                fig.add_trace(
                    go.Scattergl(
                        **grp_kwargs,
                        x=dfr.index,
                        y=dfr[col],
                        name=col,
                        mode="lines",
                        hovertemplate=htemplate,
                        line=dict(
                            color="rgba(0,0,0,0.3)",
                            width=3.5 if grp != "wind" else 2.5,
                        ),
                    ),
                    row=row_,
                    col=1,
                )

        # add traces with original sensor data
        for col in grp_cols:
            fig.add_trace(
                go.Scattergl(**kwargs, **grp_kwargs, y=df[col], name=col), row=row_, col=1
            )

        # add processed trace (backfilled)
        fig.add_trace(
            go.Scattergl(
                **kwargs,
                **grp_kwargs,
                y=df[proc_col],
                name=proc_col,
                line_dash="dot",
                line_color="#000000",
                line_width=1.5,
            ),
            row=row_,
            col=1,
        )

    fig.update_xaxes(tickformat="%m/%d", tick0=df.index[0], dtick=str(86400000 * 7))
    fig.update_layout(
        font_size=9,
        paper_bgcolor="#fbfbfb",
        margin=dict(t=30, r=20, b=10, l=20),
        legend=dict(
            x=1.01,
            xanchor="left",
            groupclick="toggleitem",
            grouptitlefont=dict(size=11),
            tracegroupgap=16,
        ),
    )
    # format subplot titles (are actually annotations in subplot figures)
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].update(font=dict(size=12), x=0, xanchor="left")
    return fig
