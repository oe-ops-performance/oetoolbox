import warnings

# temp for pandas (b/c of meteostat FutureWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import numpy as np
import pandas as pd
from pathlib import Path
import meteostat
import astral
from astral.location import Location
import pvlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils import oemeta, oepaths
from ..dataquery.external import query_DTN


hasID_ = lambda col, ids_: any(i in col.casefold() for i in ids_)
isGHI = lambda c: hasID_(c, ["ghi", "global", "hor"])
isPOA = lambda c: hasID_(c, ["poa", "plane", "irr"]) and (not isGHI(c))
isAMBTEMP = lambda c: hasID_(
    c, ["amb", "air_temp", "temp_air", "temp_c_skid"]
)  # added last one for IVSC
isMODTEMP = lambda c: hasID_(c, ["mod", "mdl", "mts", "mst", "pnltemp", "pnltmp", "bom"])
isWIND = lambda c: hasID_(c, ["wind", "wnd", "speed", "ws_ms"])


def grouped_met_columns(column_names):
    groups_ = {
        "poa": list(filter(isPOA, column_names)),
        "ghi": list(filter(isGHI, column_names)),
        "ambtemp": list(filter(isAMBTEMP, column_names)),
        "modtemp": list(filter(isMODTEMP, column_names)),
        "wind": list(filter(isWIND, column_names)),
    }
    col_groups = {grp: cols for grp, cols in groups_.items() if cols}
    return col_groups


group_external_reference_dict = {
    "poa": ["poa_global"],
    "ghi": ["dtn_ghi"],
    "ambtemp": ["dtn_temp_air"],
    "wind": ["dtn_wind_speed"],
}

# function to generate time series subplots
htemplate = (
    "<b>%{fullData.name}</b><br>Value: %{y:.2f}<br><i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
)
subplot_legend = dict(
    x=1.01,
    xanchor="left",
    groupclick="toggleitem",
    grouptitlefont=dict(size=11),
    tracegroupgap=16,
)


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
    grouped_sensor_cols = grouped_met_columns(sensor_cols)
    grp_ID_dict = {
        "poa": "POA",
        "ghi": "GHI",
        "modtemp": "ModTemp",
        "ambtemp": "AmbTemp",
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
        legend=subplot_legend,
    )
    # format subplot titles (are actually annotations in subplot figures)
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].update(font=dict(size=12), x=0, xanchor="left")
    return fig


## function to get astral Location object for a given site
def site_location(site):
    tz = oemeta.data["TZ"].get(site)
    lat, lon = oemeta.data["LatLong"].get(site)
    kwargs = dict(timezone=tz, latitude=lat, longitude=lon)
    location_info = astral.LocationInfo(name=site, region=site, **kwargs)
    return Location(location_info)


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


def backfill_meteo_data(df_met, site, force_avg_profile=[], q=True):
    """
    note: force_avg_profile = list of sensor groups for which to backfill with average profile data instead of DTN data
    """
    qprint = lambda msg, end="\n": None if q else print(msg, end=end)

    ## GET SITE METADATA (location info)
    tz = oemeta.data["TZ"].get(site)
    lat, lon = oemeta.data["LatLong"].get(site)
    alt = oemeta.data["Altitude"].get(site)

    grouped_sensor_cols = grouped_met_columns(list(df_met.columns))

    ## create new dataframe for QC
    dfQC = df_met.copy()
    dfQC["Hour"] = dfQC.index.hour
    dfQC["Day"] = dfQC.index.day
    dfQC["Month"] = dfQC.index.month
    dfQC["Year"] = dfQC.index.year

    ## QC PART 1 - IDENTIFY SENSOR COLUMNS & REMOVE OUTLIERS ABOVE/BELOW RANGE IN DEFINED LIMITS
    group_limits = dict(
        poa=[-10, 1500], ghi=[-10, 1500], ambtemp=[-35, 150], modtemp=[-35, 150], wind=[0, 75]
    )
    sensor_params = {}
    all_sensor_cols = []

    qprint("\nSensors:")
    for grp, cols in grouped_sensor_cols.items():
        sensor_params[grp] = {"columns": cols, "limits": group_limits[grp]}
        all_sensor_cols.extend(cols)
        n_dashes = 9 - len(grp)
        qprint(f'    {grp.upper()} {"-"*n_dashes} {cols}')

    if site == "Imperial Valley":
        atmp_cols = grouped_sensor_cols.get("ambtemp")
        dfQC[atmp_cols] = (dfQC[atmp_cols] - 32).div(1.8)

        wnd_cols = grouped_sensor_cols.get("wind")
        dfQC[wnd_cols] = dfQC[wnd_cols].mul(0.44704)

        poa_cols = grouped_sensor_cols.get("poa")
        dfQC[poa_cols] = dfQC[poa_cols].round(5)

    ## get dawn/dusk times for use in poa qc  (assumes timezone-naive df_met index)
    location = site_location(site)
    local_tstamp = lambda date: pd.Timestamp(date).floor("min").tz_localize(None)
    get_dawn = lambda d: local_tstamp(location.dawn(date=d))
    get_dusk = lambda d: local_tstamp(location.dusk(date=d))
    start_date = df_met.index.min()
    end_date = df_met.index.max()
    date_range = pd.date_range(start_date, end_date)
    dawn_list = list(map(get_dawn, date_range))
    dusk_list = list(map(get_dusk, date_range))
    dfs = pd.DataFrame(dict(dawn=dawn_list, dusk=dusk_list), index=date_range)
    dfs = dfs.resample("min").ffill()

    # add to qc dataframe
    dfQC = dfQC.join(dfs)

    ## BEGIN QC
    qprint("\nStatus:")

    ## QC PART 0 - OVERWRITE NEGATIVE VALUES WITH ZERO
    qprint("    >> overwriting negative values...", end=" ")
    nonnegative_groups = ["poa", "ghi", "wind"]
    for grp, cols in grouped_sensor_cols.items():
        if grp not in nonnegative_groups:
            continue
        qprint(grp, end="(")
        n_negatives = 0  # init
        for col in cols:
            n_neg = dfQC[col].lt(0).sum()
            if n_neg > 0:
                dfQC.loc[dfQC[col].lt(0), col] = 0
                n_negatives += n_neg
                qprint(str(int(n_neg)), end=",")
        qprint(")", end=", ")
    qprint("done!")

    ## QC PART 1 - REMOVE OUTLIERS ABOVE/BELOW DEFINED LIMITS
    qprint("    >> removing outliers...", end=" ")
    for key_, dict_ in sensor_params.items():
        cols_, limits_ = dict_.values()
        for col in cols_:
            if any(i in key_ for i in ["poa", "ghi"]):
                dfQC[f"Bad_Data_{col}"] = (
                    (dfQC.index > dfQC["dawn"]) & (dfQC.index < dfQC["dusk"]) & (dfQC[col] <= 0)
                )

            dfQC[f"Bad_Data_{col}"] = (dfQC[col] > max(limits_)) | (dfQC[col] < min(limits_))
            dfQC.loc[dfQC[col].isnull(), f"Bad_Data_{col}"] = True

            dfQC[f"Cleaned_{col}"] = dfQC[col].copy()
            dfQC.loc[dfQC[f"Bad_Data_{col}"] == True, f"Cleaned_{col}"] = np.nan
    qprint("done!")

    ## add columns for average across good sensors by row
    qprint("    >> averaging sensor data...", end=" ")
    grp_ID_dict = {
        "poa": "POA",
        "ghi": "GHI",
        "modtemp": "ModTemp",
        "ambtemp": "AmbTemp",
        "wind": "Wind",
    }  # temp; in case there are dependencies on col titles elsewhere
    grp_IDs = [grp_ID_dict[grp] for grp in grouped_sensor_cols]
    for grp, grpcols in grouped_sensor_cols.items():
        grpID = grp_ID_dict[grp]
        avg_grpcol = f"Average_Across_{grpID}"
        cleaned_cols = [f"Cleaned_{c}" for c in grpcols]
        dfQC[avg_grpcol] = dfQC[cleaned_cols].mean(axis=1).copy()
    qprint("done!")

    ## add columns for average monthly profile across sensor groups
    avgcols_ = [f"Average_Across_{grp}" for grp in grp_IDs] + ["Month", "Hour"]
    df_monthly_profile = dfQC[avgcols_].groupby([dfQC.index.month, dfQC.index.hour]).mean()
    df_monthly_profile = df_monthly_profile.rename(
        columns={f"Average_Across_{grp}": f"{grp}_Monthly_Profile" for grp in grp_IDs},
    )
    df_monthly_profile = df_monthly_profile.reset_index(drop=True)
    df_monthly = (
        dfQC[["Month", "Hour"]]
        .reset_index()
        .merge(df_monthly_profile, how="left", on=["Month", "Hour"], sort=True)
    )
    df_monthly = df_monthly.drop(columns=["Month", "Hour"]).set_index("Timestamp")

    # add to qc dataframe
    dfQC = dfQC.join(df_monthly)

    ## Add columns for count of good sensor data for each timestamp (0 == all bad sensors)
    for grp, grp_cols in grouped_sensor_cols.items():
        grpID = grp_ID_dict[grp]
        badcols = [f"Bad_Data_{col}" for col in grp_cols]
        dfQC[f"{grpID}_all_bad"] = dfQC[badcols].prod(axis=1)

    ## QC PART 2 - GET EXTERNAL DATA & USE TO BACKFILL SECTIONS OF "ALL_BAD_DATA"
    qprint("    >> getting data: ", end="[")

    # create localized index to use with pvlib functions (developed for dst)
    naive_index = df_met.index.copy()
    localized_index = localize_naive_datetimeindex(df_met, tz=tz).index

    # PVLib local clearsky data (minute level)
    location = pvlib.location.Location(lat, lon, tz, alt, site)
    solarpos = location.get_solarposition(localized_index)
    clearsky = location.get_clearsky(localized_index)
    clearsky.columns = [f"{c}_clearsky" for c in clearsky.columns]
    dfSUN_ = solarpos.join(clearsky)
    dfSUN = remove_tzinfo_and_standardize_index(
        dfSUN_
    )  # remove timezone info from index (for dst purposes)
    dfSUN = dfSUN.reindex(naive_index)
    qprint("PVLib", end=", ")

    # NOAA AmbTemp & WindSpeed
    meteostat_end_date = end_date.ceil("D")
    stns_ = meteostat.Stations().nearby(lat, lon).fetch(10)
    delta_bfill_date = (end_date.ceil("D") - stns_["hourly_end"].max()).days
    if delta_bfill_date == 1:
        meteostat_end_date = stns_["hourly_end"].max()
    elif delta_bfill_date > 1:
        qprint("Insufficient external data available for backfill.\nExiting.")
        return None
    station = stns_[(stns_.hourly_end >= meteostat_end_date)].iloc[0]  # nearest distance
    dfn_ = meteostat.Hourly(station.name, start_date, meteostat_end_date, timezone=tz).fetch()
    dfn_["wspd"] = dfn_["wspd"].mul(5 / 18)  # Convert km/h to m/s
    dfn_ = dfn_[["temp", "wspd"]].rename(columns={"temp": "NOAA_AmbTemp", "wspd": "NOAA_WindSpeed"})
    dfn = remove_tzinfo_and_standardize_index(
        dfn_
    )  # remove timezone info from index (for dst purposes)
    dfNOAA = dfn.resample("min").ffill().copy()  # .interpolate().copy()   #upsample to minute-level
    dfNOAA = dfNOAA.reindex(naive_index)
    qprint("NOAA", end=", ")

    # get local DTN weather data (hourly)
    interval = "hour"
    fields = ["airTemp", "shortWaveRadiation", "windSpeed"]
    dtn_start = start_date.floor("D")
    dtn_end = end_date.ceil("D") + pd.Timedelta(hours=1)  # add hour for upsampling from 1h to 1min
    dfd_ = query_DTN(lat, lon, dtn_start, dtn_end, interval, fields, tz=tz)
    dfd_ = dfd_[["airTemp", "windSpeed", "shortWaveRadiation"]]
    dfd_.columns = ["DTN_temp_air", "DTN_wind_speed", "DTN_ghi"]
    dtn_ghi_adj = oemeta.data["DTN_GHI_adj"].get(site)
    dfd_["DTN_ghi"] = dfd_["DTN_ghi"].mul(dtn_ghi_adj)
    dfd = remove_tzinfo_and_standardize_index(dfd_)
    dfDTN = dfd.resample("min").ffill().copy()  # .interpolate().copy()
    dfDTN = dfDTN.reindex(naive_index)
    qprint("DTN", end="]\n")

    # create combined dataframe of external data
    dfEXT_naive = dfSUN.join(dfNOAA).join(dfDTN)

    # add back timezone info (for pvlib transposition)   <<<<<<<<<<<<check
    dfEXT = localize_naive_datetimeindex(dfEXT_naive, tz=tz)

    # transposition models (DTN GHI to POA)
    qprint("    >> transposing DTN GHI to POA...", end=" ")
    disc = pvlib.irradiance.disc(
        dfEXT["DTN_ghi"],
        dfEXT["zenith"].values,
        dfEXT.index,
        max_zenith=90,
        max_airmass=2,
    )
    df_disc = pvlib.irradiance.complete_irradiance(
        solar_zenith=dfEXT["apparent_zenith"],
        ghi=dfEXT["DTN_ghi"],
        dni=disc["dni"],
        dhi=None,
        dni_clear=dfEXT["dni_clearsky"],
    )
    df_disc = df_disc.rename(columns={"dni": "DTN_dni", "dhi": "DTN_dhi"})
    df_disc = df_disc.drop(columns=["ghi"])
    df_disc["dni_extra"] = pvlib.irradiance.get_extra_radiation(
        datetime_or_doy=df_disc.index,
        method="nrel",
        epoch_year=df_disc.index.year[0],
    )
    dfEXT = dfEXT.join(df_disc)

    equip_dict = oemeta.data["Equip"].copy()
    racking_keys = ["Type", "Tilt", "GCR"]
    site_racking = {key_: equip_dict[f"Racking_{key_}"].get(site) for key_ in racking_keys}
    has_trackers = site_racking["Type"] == "Tracker"
    if has_trackers:
        tracking_profile = pvlib.tracking.singleaxis(
            dfEXT["apparent_zenith"],
            dfEXT["azimuth"],
            max_angle=site_racking["Tilt"],
            backtrack=True,
            gcr=site_racking["GCR"],
        )
        dfEXT = dfEXT.join(tracking_profile)

    surface_tilt = dfEXT["surface_tilt"] if has_trackers else site_racking["Tilt"]
    surface_azimuth = dfEXT["surface_azimuth"] if has_trackers else 180

    POA_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dni=dfEXT["DTN_dni"],
        ghi=dfEXT["DTN_ghi"],
        dhi=dfEXT["DTN_dhi"],
        dni_extra=dfEXT["dni_extra"],
        solar_zenith=dfEXT["apparent_zenith"],
        solar_azimuth=dfEXT["azimuth"],
        model="perez",
    )
    POA_irradiance = POA_irradiance.fillna(value=0)
    dfEXT = dfEXT.join(POA_irradiance)
    qprint("done!")

    # remove tz info again before merging                   ###########################
    dfEXT = remove_tzinfo_and_standardize_index(dfEXT)

    # add to qc dataframe
    dfQC = dfQC.join(dfEXT)

    ## backfilling
    fill_col_dict = {
        "poa": "poa_global",
        "ghi": "DTN_ghi",
        "ambtemp": "DTN_temp_air",
        "modtemp": "ModTemp_Monthly_Profile",
        "wind": "DTN_wind_speed",
    }
    qprint("    >> backfilling bad/missing data...", end=" ")
    for grp, fill_col in fill_col_dict.items():
        grpID = grp_ID_dict[grp]
        if grp in force_avg_profile:
            fill_col = f"{grpID}_Monthly_Profile"

        processed_col = f"Processed_{grpID}"
        if grp in grouped_sensor_cols:
            allbad_col = f"{grpID}_all_bad"
            avggrp_col = f"Average_Across_{grpID}"
            dfQC[processed_col] = np.where(dfQC[allbad_col] == 0, dfQC[avggrp_col], dfQC[fill_col])
        else:
            dfQC[processed_col] = dfQC[fill_col].copy() if fill_col in dfQC.columns else np.nan

    qprint("done!")
    return dfQC


def run_meteo_backfill(
    site,
    year,
    month,
    savefile=True,
    saveplot=True,
    overwrite=False,
    localsave=False,
    displayplot=False,
    return_df=False,
    return_df_and_fpath=False,
    force_avg_profile=[],
    q=True,
):
    """
    args:
        site (dtype: string) = name of solar site
        year (dtype: int) = year number, 4-digit
        month (dtype: int) = month number
        savefile (dtype: bool) = save output PROCESSED data to csv file
        saveplot (dtype: bool) = save/output plot figure to html file
        overwrite (dtype: bool) = overwrite existing output file(s); note: clears all existing PROCESSED files
        displayplot (dtype: bool) = show plot figure(s) in output; use when running in jupyter notebook
        localsave (dtype: bool) = save output file to local downloads folder (defaults to FlashReports folder)
        return_df (dtype: bool) = returns dataframe of PROCESSED data
        q (dtype: bool) = show/hide printouts
    """
    qprint = lambda msg, end="\n": None if q else print(msg, end=end)

    ## GET SITE FLASHREPORT PATH
    frpath = oepaths.frpath(year, month, ext="solar", site=site)
    if not frpath.exists():
        qprint(f'FlashReport path not found: "{str(frpath)}"')
        return

    savepath = frpath if not localsave else Path(Path.home(), "Downloads")

    ## GET CLEANED MET FILE
    mfpaths_ = [
        fp for fp in frpath.glob("PIQuery*MetStations*CLEANED*.csv") if "PROCESSED" not in fp.name
    ]
    if not mfpaths_:
        qprint(f"No CLEANED MetStation files found.")
        return
    mfp_ = max((fp.stat().st_ctime, fp) for fp in mfpaths_)[1]

    start_message = f"\n!!! BEGIN METEO PROCESSING - {site.upper()} !!!"
    qprint(f'{start_message}\n\nFile: "{mfp_.name}"', end=" ")
    if len(mfpaths_) > 1:
        qprint(f"note: using most recent file; all cleaned files = {[fp.name for fp in mfpaths_]}")

    ## LOAD MET FILE
    df_met = pd.read_csv(mfp_, index_col="Timestamp", parse_dates=True)
    expected_index = pd.date_range(df_met.index.min(), df_met.index.max(), freq="min")
    if df_met.shape[0] != expected_index.shape[0]:
        df_met = df_met.reindex(expected_index)
        qprint(f"(reindexed)", end="")

    ## run qc script
    dfQC = backfill_meteo_data(df_met, site, force_avg_profile=force_avg_profile, q=q)

    if saveplot:
        qprint("    >> generating meteo subplot figure...", end=" ")
        dfQC_15 = dfQC.resample("15min").mean().copy()
        fig = meteo_backfill_subplots(dfQC_15)
        qprint("done!")

        # stem = f'{mfp_.stem}_PROCESSED'
        stem = f"meteo_backfill_plots_{site}_{year}-{month:02d}"
        n, fname = 1, f"{stem}.html"

        if overwrite:
            # find/remove all existing processed html files
            existingplot_fps = list(frpath.glob(stem))
            for epfp_ in existingplot_fps:
                os.remove(str(epfp_))
                qprint(f'    !! overwrite=True; removed file "{epfp_.name}" !!')

        while Path(savepath, fname).exists():
            fname = f"{stem} ({n}).html"
            n += 1
        fig.write_html(Path(savepath, fname))
        qprint(f'    >> saved html file: "{fname}"')

    if savefile:
        stem = f"{mfp_.stem}_PROCESSED"
        n, fname = 1, f"{stem}.csv"

        if overwrite:
            # find/remove all existing processed csv files
            existing_fps = list(frpath.glob("PIQuery*MetStations*PROCESSED*.csv"))
            for efp_ in existing_fps:
                os.remove(str(efp_))
                qprint(f'    !! overwrite=True; removed file "{efp_.name}" !!')

        while Path(savepath, fname).exists():
            fname = f"{stem} ({n}).csv"
            n += 1
        dfQC.to_csv(Path(savepath, fname))
        qprint(f'    >> saved data file: "{fname}"')

    qprint(f"\n!!! END METEO PROCESSING - {site.upper()} !!!\n")

    if displayplot:
        dfQC_h = dfQC.resample("h").mean().copy()
        fig = meteo_backfill_subplots(dfQC_h)
        grouped_sensor_cols = grouped_met_columns(list(dfQC_h.columns))
        fig.update_layout(height=150 * len(grouped_sensor_cols))
        fig.show()

    # return dataframe if specified, otherwise exit function
    if return_df:
        return dfQC
    elif return_df_and_fpath:
        return [dfQC, Path(savepath, fname)]
    return


"""
function to get poa sensor data & compare with DTN data (transposed poa)
"""


def poa_comparison_df(site, year, month, q=True):
    qprint = lambda msg, end="\n": None if q else print(msg, end=end)

    ## GET SITE FLASHREPORT PATH
    frpath = oepaths.frpath(year, month, ext="solar", site=site)

    ## GET SITE METADATA (location info)
    tz = oemeta.data["TZ"].get(site)
    lat, lon = oemeta.data["LatLong"].get(site)
    alt = oemeta.data["Altitude"].get(site)

    mfpaths_ = [
        fp for fp in frpath.glob("PIQuery*MetStations*CLEANED*.csv") if "PROCESSED" not in fp.name
    ]
    if not mfpaths_:
        qprint("no met file found.\nexiting..")
        return
    mfp_ = max((fp.stat().st_ctime, fp) for fp in mfpaths_)[1]

    qprint(f"Loaded file: {mfp_.name}")
    if len(mfpaths_) > 1:
        qprint(f"note: using most recent file; all cleaned files = {[fp.name for fp in mfpaths_]}")

    ## LOAD MET FILE
    df_met = pd.read_csv(mfp_, index_col="Timestamp", parse_dates=True)
    start, end = df_met.index.min(), df_met.index.max()
    expected_index = pd.date_range(start, end, freq="min")
    if df_met.shape[0] != expected_index.shape[0]:
        df_met = df_met.reindex(expected_index)
        qprint(f">> reindexed")

    grouped_sensor_cols = grouped_met_columns(list(df_met.columns))
    poacols = grouped_sensor_cols.get("poa")
    if not poacols:
        qprint("no POA data found in met file.\nexiting..")
        return

    df = df_met[poacols].copy()
    df["Hour"] = df.index.hour
    df["OE.POA_average"] = df_met[poacols].mean(axis=1).copy()

    ## create average profile for the month (by hour)
    df_prof = df[["OE.POA_average"]].groupby(df.index.hour).mean().rename_axis("Hour").copy()
    df_prof.columns = ["monthly_profile"]

    ## add monthly profile to dataframe
    df = df.join(df_prof, how="left", on="Hour")
    df = df.drop(columns="Hour")

    ## get external data (dfx) for pvlib POA transposition models

    # get local pvlib clearsky data   (minute level)
    location = pvlib.location.Location(lat, lon, tz, alt, site)
    local_idx = df_met.index.tz_localize(tz, nonexistent="NaT").dropna()
    clearsky = location.get_clearsky(local_idx)
    clearsky.columns = [f"{c}_clearsky" for c in clearsky.columns]
    clearsky = clearsky.tz_localize(None).reindex(df_met.index)
    solarpos = location.get_solarposition(local_idx)
    solarpos = solarpos.tz_localize(None).reindex(df_met.index)
    dfSUN = solarpos.join(clearsky)

    ## get DTN GHI data
    dtn_start, dtn_end = start, (end + pd.Timedelta(hours=1))
    interval = "hour"
    fields = ["shortWaveRadiation"]
    dfDTN = query_DTN(lat, lon, dtn_start, dtn_end, interval, fields, tz=tz, q=q)
    dfDTN.columns = ["DTN_ghi"]
    dfDTN = dfDTN.resample("1min").ffill()
    dfDTN = dfDTN.tz_localize(None).reindex(df_met.index)

    ## combine
    dropcols_ = ["apparent_elevation", "elevation", "equation_of_time"]
    dfx = dfSUN.drop(columns=dropcols_).join(dfDTN).copy()

    ## transpose to POA
    ## disc transposition model
    disc = pvlib.irradiance.disc(
        ghi=dfx["DTN_ghi"],
        solar_zenith=dfx["zenith"],
        datetime_or_doy=dfx.index,
        max_zenith=90,
        max_airmass=2,
    )
    df_disc = pvlib.irradiance.complete_irradiance(
        solar_zenith=dfx["apparent_zenith"],
        ghi=dfx["DTN_ghi"],
        dni=disc["dni"],
        dhi=None,
        dni_clear=dfx["dni_clearsky"],
    )
    df_disc = df_disc.rename(columns={"dni": "DTN_dni", "dhi": "DTN_dhi"})
    df_disc = df_disc.drop(columns=["ghi"])
    df_disc["dni_extra"] = pvlib.irradiance.get_extra_radiation(
        datetime_or_doy=df_disc.index,
        method="nrel",
        epoch_year=df_disc.index.year[0],
    )
    dfx = dfx.join(df_disc)

    ## racking info
    equip_dict = oemeta.data["Equip"].copy()
    racking_keys = ["Type", "Tilt", "GCR"]
    site_racking = {key_: equip_dict[f"Racking_{key_}"].get(site) for key_ in racking_keys}
    has_trackers = site_racking["Type"] == "Tracker"

    if has_trackers:
        tracking_profile = pvlib.tracking.singleaxis(
            dfx["apparent_zenith"],
            dfx["azimuth"],
            max_angle=site_racking["Tilt"],
            backtrack=True,
            gcr=site_racking["GCR"],
        )
        surface_tilt = tracking_profile["surface_tilt"]
        surface_azimuth = tracking_profile["surface_azimuth"]
    else:
        surface_tilt = site_racking["Tilt"]
        surface_azimuth = 180

    ## get transposed poa data
    POA_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dni=dfx["DTN_dni"],
        ghi=dfx["DTN_ghi"],
        dhi=dfx["DTN_dhi"],
        dni_extra=dfx["dni_extra"],
        solar_zenith=dfx["apparent_zenith"],
        solar_azimuth=dfx["azimuth"],
        model="perez",
    )
    POA_irradiance = POA_irradiance.fillna(value=0)
    dfx = dfx.join(POA_irradiance)

    ## return df with poa data for comparison
    df = df.join(dfx[["poa_global", "poa_direct"]])
    return df


## new functions to replace/simplify portions of above funcs


def location_metadata(site):
    tz = oemeta.data["TZ"].get(site)
    lat, lon = oemeta.data["LatLong"].get(site)
    alt = oemeta.data["Altitude"].get(site)
    return {"timezone": tz, "latitude": lat, "longitude": lon, "altitude": alt}


def create_localized_index(site, start, end, freq):
    df_ = pd.DataFrame(index=pd.date_range(start, end, freq=freq)[:-1])
    localized_index = localize_naive_datetimeindex(dataframe=df_, site=site)
    return localized_index


def load_pvlib_clearsky(site, start, end, freq="1min"):
    tz, lat, lon, alt = location_metadata(site).values()
    naive_index = pd.date_range(start, end, freq=freq)[:-1]
    localized_index = localize_naive_datetimeindex(pd.DataFrame(index=naive_index), site=site).index

    location = pvlib.location.Location(lat, lon, tz, alt, site)
    solarpos = location.get_solarposition(localized_index)
    clearsky = location.get_clearsky(localized_index)
    clearsky.columns = [f"{c}_clearsky" for c in clearsky.columns]
    dfSUN = solarpos.join(clearsky)
    dfSUN = remove_tzinfo_and_standardize_index(
        dfSUN
    )  # remove timezone info from index (for dst purposes)
    dfSUN = dfSUN.reindex(naive_index)
    return dfSUN


def load_noaa_weather_data(site, start, end, q=True, upsample=True):
    tz, lat, lon, alt = location_metadata(site).values()
    naive_index = pd.date_range(start, end, freq="1min")[
        :-1
    ]  # noaa data is hourly; freq=1min is for upsampling
    start_date = naive_index.min()
    end_date = naive_index.max()

    df_stations = meteostat.Stations().nearby(lat, lon).fetch(10)  # nearest 10

    target_end_date = end_date.ceil("D")
    latest_date = df_stations["hourly_end"].max()
    n_missing_days = (target_end_date - latest_date).days
    if n_missing_days > 0:
        if not q:
            print(
                f"NOTE: hourly NOAA data is only available until {latest_date} ({n_missing_days = })"
            )
        target_end_date = latest_date

    # select nearest station with data
    station = df_stations[(df_stations.hourly_end >= target_end_date)].iloc[0]  # nearest distance

    # get hourly data
    dfn_ = meteostat.Hourly(station.name, start_date, target_end_date, timezone=tz).fetch()
    dfn_["wspd"] = dfn_["wspd"].mul(5 / 18)  # Convert km/h to m/s
    dfn_ = dfn_[["temp", "wspd"]].rename(columns={"temp": "NOAA_AmbTemp", "wspd": "NOAA_WindSpeed"})
    dfNOAA = remove_tzinfo_and_standardize_index(
        dfn_
    )  # remove timezone info from index (for dst purposes)
    if upsample:
        dfNOAA = dfNOAA.resample("1min").ffill()  # .interpolate()   #upsample to minute-level
        dfNOAA = dfNOAA.reindex(naive_index)
    return dfNOAA


def load_dtn_weather_data(site, start, end, upsample=True):
    tz, lat, lon, alt = location_metadata(site).values()
    naive_index = pd.date_range(start, end, freq="1min")[
        :-1
    ]  # noaa data is hourly; freq=1min is for upsampling
    start_date = naive_index.min()
    end_date = naive_index.max()

    interval = "hour"
    fields = ["airTemp", "shortWaveRadiation", "windSpeed"]
    dtn_start = start_date.floor("D")
    dtn_end = end_date.ceil("D") + pd.Timedelta(hours=1)  # add hour for upsampling from 1h to 1min
    dfd_ = query_DTN(lat, lon, dtn_start, dtn_end, interval, fields, tz=tz)
    dfd_ = dfd_[["airTemp", "windSpeed", "shortWaveRadiation"]]
    dfd_.columns = ["DTN_temp_air", "DTN_wind_speed", "DTN_ghi"]
    dtn_ghi_adj = oemeta.data["DTN_GHI_adj"].get(site)
    if dtn_ghi_adj is not None:
        dfd_["DTN_ghi"] = dfd_["DTN_ghi"].mul(dtn_ghi_adj)
    dfDTN = remove_tzinfo_and_standardize_index(dfd_)
    if upsample:
        dfDTN = dfDTN.resample("1min").ffill()  # .interpolate()   #upsample to minute-level
        dfDTN = dfDTN.reindex(naive_index)
    return dfDTN


def transposed_POA_from_DTN_GHI(site, start, end, freq="1min", q=True, keep_tzinfo=False):
    qprint = lambda msg, end="\n": None if q else print(msg, end=end)

    kwargs = dict(site=site, start=start, end=end)
    df_sun = load_pvlib_clearsky(**kwargs)

    upsample = freq == "1min"
    df_dtn = load_dtn_weather_data(**kwargs, upsample=upsample)

    # combine to single dataframe and add tz info (for pvlib transposition)
    tz = oemeta.data["TZ"].get(site)

    dropcols_ = ["apparent_elevation", "elevation", "equation_of_time"]
    df_naive = df_sun.drop(columns=dropcols_).join(df_dtn)
    # df_naive = df_sun.join(df_dtn)
    df = localize_naive_datetimeindex(df_naive, tz=tz)

    # transposition models (DTN GHI to POA)
    qprint("Transposing DTN GHI to POA...", end=" ")
    disc = pvlib.irradiance.disc(
        ghi=df["DTN_ghi"],
        solar_zenith=df["zenith"],
        datetime_or_doy=df.index,
        max_zenith=90,
        max_airmass=2,
    )

    df_disc = pvlib.irradiance.complete_irradiance(
        solar_zenith=df["apparent_zenith"],
        ghi=df["DTN_ghi"],
        dni=disc["dni"],
        dhi=None,
        dni_clear=df["dni_clearsky"],
    )
    df_disc = df_disc.rename(columns={"dni": "DTN_dni", "dhi": "DTN_dhi"})
    df_disc = df_disc.drop(columns=["ghi"])
    df_disc["dni_extra"] = pvlib.irradiance.get_extra_radiation(
        datetime_or_doy=df_disc.index,
        method="nrel",
        epoch_year=df_disc.index.year[0],
    )

    # add to main dataframe
    df = df.join(df_disc)

    # check racking details & create parameters to use for transposition
    equip_dict = oemeta.data["Equip"].copy()
    racking_keys = ["Type", "Tilt", "GCR"]
    site_racking = {k: equip_dict[f"Racking_{k}"].get(site) for k in racking_keys}
    has_trackers = site_racking["Type"] == "Tracker"
    if has_trackers:
        tracking_profile = pvlib.tracking.singleaxis(
            apparent_zenith=df["apparent_zenith"],
            apparent_azimuth=df["azimuth"],
            max_angle=site_racking["Tilt"],
            backtrack=True,
            gcr=site_racking["GCR"],
        )
        df = df.join(tracking_profile)

    surface_tilt = df["surface_tilt"] if has_trackers else site_racking["Tilt"]
    surface_azimuth = df["surface_azimuth"] if has_trackers else 180

    df_poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dni=df["DTN_dni"],
        ghi=df["DTN_ghi"],
        dhi=df["DTN_dhi"],
        dni_extra=df["dni_extra"],
        solar_zenith=df["apparent_zenith"],
        solar_azimuth=df["azimuth"],
        model="perez",
    )
    df_poa = df_poa.fillna(value=0)
    df = df.join(df_poa)
    qprint("done!")

    if not keep_tzinfo:
        df = remove_tzinfo_and_standardize_index(df)
        qprint("Note: removed timezone information from datetime index.")

    return df
