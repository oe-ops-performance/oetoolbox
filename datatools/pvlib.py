import datetime as dt
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pvlib.irradiance import complete_irradiance, dirint, get_extra_radiation, get_total_irradiance
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import Array, FixedMount, PVSystem, retrieve_sam, SingleAxisTrackerMount
from pvlib.solarposition import get_solarposition
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvlib.tracking import singleaxis

from ..datatools.backfill import get_supporting_data
from ..dataquery.external import load_noaa_weather_data, query_DTN
from ..utils import oemeta, oepaths
from ..utils.datetime import (
    create_localized_index,
    create_naive_index,
    localize_naive_datetimeindex,
    remove_tzinfo_and_standardize_index,
)
from ..utils.helpers import quiet_print_function

# define constants/reference parameters
CEC_INVERTERS = retrieve_sam("cecinverter")
CEC_MODULES = retrieve_sam("CECmod")
TEMPERATURE_PARAMETERS = TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"]

# TODO: move parameters to metadata file
Paco_adjustments = {
    "Alamo": 935000.0,
    "Adams East": 1482000.0,
    "Catalina II": 784000.0,
    "CID": 840000.0,
    "CW-Corcoran": 840000.0,
    "CW-Goose Lake": 875000.0,
    "GA3": 3150000.0,
    "GA4": 3333000.0,
    "Indy I": 714000.0,
    "Indy II": 714000.0,
    "Indy III": 720000.0,
    "Kent South": 1452000.0,
    "Old River One": 1460000.0,
    "Mulberry": 880000.0,
    "Maricopa West": 2200000.0,
    "Pavant": 2198000.0,
    "Richland": 1834000.0,
    "Selmer": 880000.0,
}


def site_location_metadata(site):
    alt = oemeta.data["Altitude"].get(site)
    lat, lon = oemeta.data["LatLong"].get(site)
    tz = oemeta.data["TZ"].get(site)
    return alt, lat, lon, tz


def get_site_model_losses(site, q=True):
    """Returns losses to be applied in pvlib model."""
    current_year = dt.date.today().year
    site_cod = oemeta.data["COD"].get(site)
    losses = oemeta.data["Losses"].get(site)
    degradation = oemeta.data["Losses"]["Degradation"].get(site)

    system_loss_degradation = round((100 - ((current_year - site_cod) * degradation)) / 100, 2)
    system_loss_shading = (100 - losses["shading"]) / 100
    system_loss_soiling = (100 - losses["soiling"]) / 100
    system_loss_module_quality = (100 - losses["quality"]) / 100
    system_loss_module_mismatch = (100 - losses["mismatch"]) / 100
    system_loss_LID = (100 - losses["lid"]) / 100
    system_losses_Array = np.prod(
        [
            system_loss_degradation,
            system_loss_shading,
            system_loss_soiling,
            system_loss_module_quality,
            system_loss_module_mismatch,
            system_loss_LID,
        ]
    )

    dc_adjustment = oemeta.data["Losses"]["DC_Adjustment"].get(site)
    ac_adjustment = oemeta.data["Losses"]["AC_Adjustment"].get(site)

    system_loss_dc_ohmic = (100 - losses["wiring"]) / 100
    system_losses_DC = system_loss_dc_ohmic * dc_adjustment
    system_losses_AC = ac_adjustment

    if not q:
        print("\nSystem losses:")
        param_printouts = {
            "Degradation": system_loss_degradation,
            "Shading": system_loss_shading,
            "Soiling": system_loss_soiling,
            "DC Ohmic": system_loss_dc_ohmic,
            "Module Quality": system_loss_module_quality,
            "Module Mismatch": system_loss_module_mismatch,
            "LID": system_loss_LID,
            "DC Adj": dc_adjustment,
            "AC Adj": ac_adjustment,
        }
        for desc, val in param_printouts.items():
            print(f"{desc} = ".rjust(22) + f"{val:0.3f}")
        print("")
        totals_ = {
            "Total Array": system_losses_Array,
            "Total DC": system_losses_DC,
            "Total AC": system_losses_AC,
        }
        for ttl, val in totals_.items():
            print(f"{ttl.upper()} LOSSES: {val:0.3f}")
        print("")

    return system_losses_Array, system_losses_DC, system_losses_AC


def design_database(site: str):
    """Retrieves the equipment design metadata for a given solar site"""
    return oemeta.data["Project_Design_Database"].get(site)


def get_site_location(site: str):
    """Returns the pvlib Location object for a given solar site"""
    tz = oemeta.data["TZ"].get(site)
    lat, lon = oemeta.data["LatLong"].get(site)
    alt = oemeta.data["Altitude"].get(site)
    return Location(lat, lon, tz, alt, site)


def get_pvlib_solar_position(site, datetime_index):
    """returns dataframe with azimuth, zenith, etc."""
    lat, lon = oemeta.data["LatLong"].get(site)
    idx = datetime_index + pd.Timedelta(minutes=30)
    solar_position = get_solarposition(idx, lat, lon)
    solar_position.index = datetime_index
    return solar_position


def get_inverter_names(site):
    """Returns a list of inverter names for a given site"""
    af_dict = oemeta.data["AF_Solar_V3"].get(site)
    inv_dict = af_dict.get("Inverters")
    if inv_dict:
        return [*inv_dict["Inverters_Assets"]]
    return list(design_database(site).keys())


def get_combiner_list(dict_inv):
    """Returns a list of combiner names for a given inverter

    Parameters
    ----------
    dict_inv : dict
        inverter parameter dictionary (from design database)
    """
    is_cmb_key = lambda key: not any(x in key for x in ["Inverter", "Racking", "Tilt", "Azimuth"])
    return list(filter(is_cmb_key, [*dict_inv]))


def get_inverter_parameters(site, inv_type):
    """Retrieves the inverter parameters for a given CEC inverter type

    Parameters
    ----------
    site : str
        name of solar site
    inv_type : str
        CEC inverter type (from design database)
    """
    inv_params = CEC_INVERTERS[inv_type]
    if site in Paco_adjustments:
        inv_params["Paco"] = Paco_adjustments[site]
    return inv_params


def get_racking_mount(dict_inv):
    """Returns pvlib racking mount object for a given inverter

    Parameters
    ----------
    dict_inv : dict
        inverter parameter dictionary (from design database)
    """
    racking_type = dict_inv.get("Racking Type")
    tilt, azimuth = map(lambda k: int(dict_inv[k]), ["Tilt Angle (Deg)", "Azimuth Angle (Deg)"])
    if racking_type == "Tracker":
        mount = SingleAxisTrackerMount(
            axis_azimuth=azimuth,
            axis_tilt=0,
            max_angle=tilt,
            backtrack=True,
            gcr=dict_inv["Racking GCR"],
            racking_model="open_rack",
        )
    elif racking_type == "Fixed":
        mount = FixedMount(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            racking_model="open_rack",
        )
    return mount


def get_surface_parameters(site, df_met):
    """Returns the tilt/azimuth parameters (used for POA transposition from GHI data)"""
    design_db = design_database(site)
    dict_inv = list(design_db.values())[0]
    racking_tilt = int(dict_inv["Tilt Angle (Deg)"])

    if dict_inv.get("Racking Type") != "Tracker":  # same for all inv
        return racking_tilt, 180

    tracking_profile = singleaxis(
        df_met["apparent_zenith"],
        df_met["azimuth"],
        max_angle=racking_tilt,
        backtrack=True,
        gcr=dict_inv["Racking GCR"],
    )
    return tracking_profile["surface_tilt"], tracking_profile["surface_azimuth"]


def get_inverter_configs(site, inv_names=None):
    """Generates a list of PVSystem objects"""
    design_db = design_database(site)
    if inv_names is None:
        inv_names = [*design_db]
    inv_configs = []
    for inv, dict_inv in design_db.items():
        if inv not in inv_names:
            continue
        inv_arrays = []  # init
        inv_type = dict_inv["Inverter Type (CEC)"]
        inv_params = get_inverter_parameters(site, inv_type)
        racking_mount = get_racking_mount(dict_inv)
        combiner_list = get_combiner_list(dict_inv)

        for cmb, dict_cmb in dict_inv.items():
            cmb_arrays = []  # init
            if cmb not in combiner_list:
                continue
            mod_type = dict_cmb["Module Type (CEC)"]
            mod_wattage_keys = [k for k in dict_cmb if "Wattage" in k]
            for i, key in enumerate(mod_wattage_keys):  # should always be 6
                if not (dict_cmb[key] > 0):
                    continue
                array = Array(
                    mount=racking_mount,
                    module_type="glass_polymer",
                    module_parameters=CEC_MODULES[mod_type],
                    temperature_model_parameters=TEMPERATURE_PARAMETERS,
                    modules_per_string=dict_cmb[f"String Length {i+1}"],
                    strings=dict_cmb[f"String Count {i+1}"],
                )
                cmb_arrays.append(array)

            inv_arrays.extend(cmb_arrays)

        inv_configs.append(PVSystem(arrays=inv_arrays, inverter_parameters=inv_params, name=inv))

    return inv_configs


def load_pvlib_clearsky(site, start, end, freq="1min"):
    """Loads pvlib clearsky data"""
    location = get_site_location(site)
    localized_index = create_localized_index(site, start, end, freq)
    solarpos = location.get_solarposition(localized_index)
    clearsky = location.get_clearsky(localized_index)
    clearsky.columns = [f"{c}_clearsky" for c in clearsky.columns]
    dfSUN = solarpos.join(clearsky)
    dfSUN = remove_tzinfo_and_standardize_index(dfSUN)
    naive_index = create_naive_index(start, end, freq)
    dfSUN = dfSUN.reindex(naive_index)
    return dfSUN


def dtn_data_requires_dst_shift(df):
    """checks for march dst condition in df to determine whether shift is needed"""
    if 3 not in df.index.month:
        return False
    return len(df.index.strftime("%z").unique()) > 1


def apply_dst_shift_to_dtn_data(df_dtn):
    """applies shift for march dst data queried from dtn"""
    df = df_dtn.copy()
    df_dates = pd.DataFrame(index=pd.date_range(df.index[0], df.index[-1].ceil("D")))
    df_dates["offset"] = df_dates.index.strftime("%z")
    tz_offsets = list(df_dates["offset"].unique())
    shift_date = df_dates.loc[df_dates["offset"].eq(tz_offsets[0])].index[-1]
    shifted = df.index >= shift_date
    for col in df.columns:
        df.loc[shifted, col] = df.loc[shifted, col].shift(1)
    return df


def query_dtn_meteo_data(
    site: str,
    start_date: str,
    end_date: str,
    keep_tz: bool = True,
    q: bool = True,
):
    """Queries meteo data from DTN for use in pvlib model (when df_met not provided).

    Parameters
    ----------
    site : str
        name of solar site
    start_date : str
        start date for query; format = '%Y-%m-%d'
    end_date : str
        end date for query; format = '%Y-%m-%d'
    keep_tz : bool, optional
        whether to keep or remove site timezone information from index
    q : bool, optional
        quiet parameter, by default True (i.e. no printouts)

    Returns
    -------
    pandas.DataFrame
        A dataframe with datetime index and columns for DTN data and solar position.
    """
    # run dtn query
    _, lat, lon, tz = site_location_metadata(site)
    interval = "hour"
    fields = ["airTemp", "shortWaveRadiation", "windSpeed"]
    start = pd.Timestamp(start_date).floor("D")
    end = pd.Timestamp(end_date).ceil("D") + pd.Timedelta(hours=1)  # add hour for upsample to min
    df = query_DTN(lat, lon, start, end, interval, fields, tz=tz, q=q)
    df = df.rename(columns=dict(zip(fields, ["dtn_air_temp", "dtn_ghi", "dtn_wind_speed"])))

    # add solar position data
    solar_position = get_pvlib_solar_position(site, df.index.copy())
    df = df.join(solar_position)

    # use dirint method to get decomposed irradiance (ghi, dhi, dni)
    df_dirint = complete_irradiance(
        solar_zenith=df["apparent_zenith"],
        ghi=df["dtn_ghi"],
        dni=dirint(df["dtn_ghi"], df["apparent_zenith"], df.index, max_zenith=85),
        dhi=None,
    )
    df[["dhi_raw", "dni_raw"]] = df_dirint[["dhi", "dni"]].copy()
    df["dni_extra"] = get_extra_radiation(df.index)
    df = df.rename(columns={"dtn_ghi": "ghi_raw"})

    # check for march dst condition & apply shift if necessary
    if dtn_data_requires_dst_shift(df):
        df = apply_dst_shift_to_dtn_data(df)

    if not keep_tz:
        df = remove_tzinfo_and_standardize_index(df)

    data_tz = tz if keep_tz else None
    expected_index = pd.date_range(start_date, end_date, freq="h", inclusive="left", tz=data_tz)
    df = df.reindex(expected_index)
    return df


def get_poa_from_ghi(site, df):
    """Calculates irradiance values using the 'perez' sky diffuse model

    Parameters
    ----------
    site : str
        Name of site.
    df : pandas.DataFrame
        Dataframe with DTN data and solar position data.

    Returns
    -------
    pandas.DataFrame
        A dataframe with columns: 'poa_global', 'poa_direct', 'poa_diffuse',
        'poa_sky_diffuse', 'poa_ground_diffuse'
    """
    # get racking parameters (only uses df for sites with trackers)
    surface_tilt, surface_azimuth = get_surface_parameters(site, df)
    total_irrad = get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dni=df["dni_raw"],
        ghi=df["ghi_raw"],
        dhi=df["dhi_raw"],
        dni_extra=df["dni_extra"],
        solar_zenith=df["apparent_zenith"],
        solar_azimuth=df["azimuth"],
        model="perez",
    )
    total_irrad = total_irrad.fillna(value=0)
    return total_irrad


def format_meteo_data_for_pvlib(df_met, site):
    # check for POA data from sensors (use dtn from file if not exist)
    POA_COL = "POA" if "Average_Across_POA" in df_met.columns else "POA_DTN"
    freq_timedelta = df_met.index.to_series().diff().max()
    if "DTN" in POA_COL and freq_timedelta != pd.Timedelta(hours=1):
        df_met = df_met.resample("1h").mean()
        freq_timedelta = pd.Timedelta(hours=1)

    # note: processed POA column = poa_global when no good sensor data
    rename_cols = {
        "Processed_AmbTemp": "ambient_temperature",
        "Processed_Wind": "wind_speed",
        "Processed_ModTemp": "module_temperature",
        "Processed_POA": "effective_irradiance",
    }
    keep_cols = [c for c in rename_cols if c in df_met.columns]
    df = df_met[keep_cols].rename(columns=rename_cols).copy()

    # make a copy of the POA data (losses only applied to effective_irr)
    df[POA_COL] = df["effective_irradiance"].copy()
    if "POA_all_bad" in df_met.columns:
        df["POA_all_bad"] = df_met["POA_all_bad"].copy()

    if "Average_Across_ModTemp" in df_met.columns:
        if df_met["Average_Across_ModTemp"].isna().all():
            df_met = df_met.drop(columns=["Average_Across_ModTemp"])

    if "Average_Across_ModTemp" not in df_met.columns:
        if not all(c in df.columns for c in ["ambient_temperature", "wind_speed"]):
            raise ValueError("must have either module temp, or amb temp and wind speed.")
        if df["ambient_temperature"].sum() == 0:
            raise ValueError("must have valid ambient temp data if no module temp.")

        param_a = TEMPERATURE_PARAMETERS["a"]
        param_b = TEMPERATURE_PARAMETERS["b"]
        scaling_factor = np.exp(param_a + (param_b * df["wind_speed"]))
        df["module_temperature"] = df[POA_COL].mul(scaling_factor).add(df["ambient_temperature"])

    # get solar position data for model
    df_sun = get_pvlib_solar_position(site, df.index)
    df = df_sun.join(df)
    return df


def run_model_chain(system, location, weather_data):
    """Runs model chain for a single PVSystem"""
    data = [weather_data] * len(system.arrays)
    mc = ModelChain(system, location, aoi_model="physical", spectral_model="no_loss")
    mc.run_model_from_effective_irradiance(data=data)
    return mc.results.ac.div(1e3).rename(f"{system.name}_Possible_Power")


def run_parallel_pvlib_models(inverter_configs, location, weather_data, q=True):
    """Runs multiple model chains simultaneously for list of PVSystem objects
    -> n_jobs = -2 tells joblib to use all available CPU cores except for 1
    """
    results = Parallel(n_jobs=-2, verbose=0 if q else 100)(
        delayed(run_model_chain)(system, location, weather_data) for system in inverter_configs
    )
    return results


def run_pvlib_model(
    site: str,
    datetime_index: pd.DatetimeIndex,
    poa_data: pd.Series = None,
    inverter_names: list = None,
    q: bool = True,
):
    """Runs pvlib model using parameters for the given site.
        -> uses poa_data if provided, otherwise queries dtn ghi and transposes to poa
        -> if poa_data is provided, the index must match the datetime_index
        -> if inverter_names not provided, model will run for all site inverters

    Parameters
    ----------
    site : str
        The name of the solar site. Used to determine model parameters.
    datetime_index : pd.DatetimeIndex
        The index to use; determines the start/end dates and frequency of the output data.
    poa_data : pd.Series, optional
        The POA data to use for the effective_irradiance in the model. If not provided,
        will query DTN data for site location and use POA transposed from GHI. Defaults to None.
        Series must be named either "POA" or "POA_DTN", depending on the source of the data.
    inverter_names : list, optional
        The names of the inverters for which to run the pvlib model. If not provided,
        the model will run for all site inverters. Defaults to None.
    q : bool, optional
        Quiet parameter; when False, enables status printouts. Defaults to True.

    Returns
    -------
    pd.DataFrame
        A dataframe with a datetime index, the POA data used in the model, and the modeled
        generation data for the requested inverters. The POA column will be "POA" or "POA_DTN".
    """
    qprint = quiet_print_function(q=q)
    if datetime_index.inferred_freq is None:
        raise ValueError("Could not infer frequency from provided datetime index.")

    if poa_data is not None:
        if not poa_data.index.difference(datetime_index).empty:
            raise ValueError("Index of provided POA data does not match datetime_index.")

    freq_timedelta = datetime_index.to_series().diff().max()
    start_datetime = datetime_index.min()
    end_datetime = datetime_index.max()

    if poa_data is None:  # use DTN data (overrides frequency to hourly)
        POA_COL = "POA_DTN"
        date_str = lambda d: d.strftime("%Y-%m-%d")
        start_date = date_str(start_datetime.floor("D"))
        end_date = date_str(end_datetime.ceil("D"))
        keep_tz = start_datetime.tzinfo is not None

        # query DTN data (output includes solar position data), then run POA transposition with GHI
        qprint("Querying DTN weather data")
        df_meteo = query_dtn_meteo_data(site, start_date, end_date, keep_tz=keep_tz)
        df_poa = get_poa_from_ghi(site, df_meteo)
        df_meteo[POA_COL] = df_poa["poa_global"].copy()

    else:
        POA_COL = poa_data.name
        if "DTN" in POA_COL and freq_timedelta != pd.Timedelta(hours=1):
            qprint("Detected resampled DTN data - reverting to native hourly frequency.")
            poa_data = poa_data.resample("h").mean()  # could happen from processed met file
            freq_timedelta = pd.Timedelta(hours=1)
            datetime_index = poa_data.index.copy()

        # get solar position data
        df_meteo = get_pvlib_solar_position(site, datetime_index)
        df_meteo[POA_COL] = poa_data.copy()

    # add effective_irradiance column for use in model
    array_losses, dc_losses, _ = get_site_model_losses(site)
    df_meteo["effective_irradiance"] = df_meteo[POA_COL].mul(array_losses).mul(dc_losses).copy()

    # get inverter names and associated configurations
    inverter_configs = get_inverter_configs(site, inverter_names)
    location = get_site_location(site)

    # run model chain
    qprint(f"Running models for {len(inverter_configs)} inverter configurations:")
    mc_results = run_parallel_pvlib_models(inverter_configs, location, weather_data=df_meteo, q=q)
    df_model = pd.concat(mc_results, axis=1)
    df_model.insert(0, POA_COL, df_meteo[POA_COL].copy())
    qprint(f"Done. (note: {POA_COL = })")

    return df_model


def run_flashreport_pvlib_model(site, year, month, localized=False, force_dtn=False, q=True):
    """Runs pvlib model for selected reporting period using site data or DTN
    -> note: uses joblib Parallel to improve speed by running multiple configs in parallel
    """
    qprint = quiet_print_function(q=q)
    qprint(f"\nBegin PVLib script. ({site = })")

    # check for pi query file in flashreport folder
    met_fpath = None
    dir = oepaths.frpath(year, month, ext="solar", site=site)
    if dir.exists():
        met_fpaths = list(dir.glob("PIQuery*MetStations*CLEANED*PROCESSED*.csv"))
        met_fpath = oepaths.latest_file(met_fpaths)

    if met_fpath is None:
        force_dtn = True

    if force_dtn:
        qprint("Using external weather data from DTN.")
        POA_COL = "POA_DTN"
        df = get_supporting_data(site, year, month)  # tz-aware
        df[POA_COL] = df["poa_global"].copy()
        df["effective_irradiance"] = df[POA_COL].copy()
    else:
        qprint("Using weather data from PI query file.")
        df = pd.read_csv(met_fpath, index_col=0, parse_dates=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            # this can happen if .csv already has tz info & an error prevents parsing (e.g. dst)
            df.index = pd.to_datetime(
                df.index,
                format="%Y-%m-%d %H:%M:%S%z",
                utc=True,
            ).tz_convert(tz=oemeta.data["TZ"].get(site))
        elif df.index.tz is None:
            df = localize_naive_datetimeindex(df, site=site)
        df = format_meteo_data_for_pvlib(df, site)
        POA_COL = "POA" if "POA" in df.columns else "POA_DTN"

    # apply losses to effective_irradiance column for use in model
    array_losses, dc_losses, _ = get_site_model_losses(site)
    df["effective_irradiance"] = df["effective_irradiance"].mul(array_losses).mul(dc_losses)

    # get site location & system/inverter configurations
    inverter_configs = get_inverter_configs(site)
    location = get_site_location(site)

    # run model chain
    qprint(f"Running models for {len(inverter_configs)} inverter configurations:")
    mc_results = run_parallel_pvlib_models(inverter_configs, location, weather_data=df, q=q)
    df_model = pd.concat(mc_results, axis=1)
    df_model.insert(0, POA_COL, df[POA_COL].copy())

    if not localized:
        df_model = remove_tzinfo_and_standardize_index(df_model)

    qprint("Done.")
    return df_model
