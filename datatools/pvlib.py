import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
import pvlib
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import Array, FixedMount, PVSystem, SingleAxisTrackerMount
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

from ..dataquery.external import load_noaa_weather_data, query_DTN
from ..utils import oemeta, oepaths
from ..utils.datetime import (
    create_localized_index,
    create_naive_index,
    localize_naive_datetimeindex,
    remove_tzinfo_and_standardize_index,
)

# define constants/reference parameters
CEC_INVERTERS = pvlib.pvsystem.retrieve_sam("cecinverter")
CEC_MODULES = pvlib.pvsystem.retrieve_sam("CECmod")
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


def get_site_model_losses(site, q=True):
    losses = oemeta.data["Losses"].get(site)

    current_year = dt.date.today().year
    site_cod = oemeta.data["COD"].get(site)
    degradation = oemeta.data["Losses"]["Degradation"].get(site)

    system_loss_degradation = round((100 - ((current_year - site_cod) * degradation)) / 100, 2)
    system_loss_shading = (100 - losses["shading"]) / 100
    system_loss_soiling = (100 - losses["soiling"]) / 100
    system_loss_module_quality = (100 - losses["quality"]) / 100
    system_loss_module_mismatch = (100 - losses["mismatch"]) / 100
    system_loss_LID = (100 - losses["lid"]) / 100
    system_losses_Array = np.product(
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


def get_inverter_names(site):
    """Returns a list of inverter names for a given site"""
    af_dict = oemeta.data["AF_Solar_V3"].get(site)
    inv_dict = af_dict.get("Inverters")
    return [*inv_dict["Inverters_Assets"]] if inv_dict else []


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

    tracking_profile = pvlib.tracking.singleaxis(
        df_met["apparent_zenith"],
        df_met["azimuth"],
        max_angle=racking_tilt,
        backtrack=True,
        gcr=dict_inv["Racking GCR"],
    )
    return tracking_profile["surface_tilt"], tracking_profile["surface_azimuth"]


def get_inverter_configs(site):
    design_db = design_database(site)
    inv_configs = []
    for dict_inv in design_db.values():
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

        inv_configs.append(PVSystem(arrays=inv_arrays, inverter_parameters=inv_params))

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


def load_pvlib_noaa_data(site, start, end, freq="1min", q=True):
    """Loads NOAA weather data (hourly)"""
    df_ = load_noaa_weather_data(site, start, end, freq=freq, q=q)
    dfNOAA = remove_tzinfo_and_standardize_index(df_)  # hourly

    if freq == "1min":
        naive_index = create_naive_index(start, end, freq)
        dfNOAA = dfNOAA.resample(freq).ffill()
        dfNOAA = dfNOAA.reindex(naive_index)

    return dfNOAA


def query_dtn_meteo_data(site, start, end, freq="1min", q=True):
    """Queries meteo data from DTN for use in pvlib model (when df_met not provided)

    Parameters
    ----------
    site : str
        name of solar site
    start : str
        start date for query; format = '%Y-%m-%d'
    end : str
        end date for query; format = '%Y-%m-%d'
    q : bool, optional
        quiet parameter, by default True (i.e. no printouts)
    """
    # get location metadata
    tz = oemeta.data["TZ"].get(site)
    lat, lon = oemeta.data["LatLong"].get(site)

    # run dtn query
    interval = "hour"
    fields = ["airTemp", "shortWaveRadiation", "windSpeed"]
    start_date = pd.Timestamp(start).floor("D")
    end_date = pd.Timestamp(end).ceil("D") + pd.Timedelta(hours=1)  # add hour for upsample to min
    df = query_DTN(lat, lon, start_date, end_date, interval, fields, tz=tz, q=q)
    # new_columns = ["DTN_temp_air", "DTN_ghi", "DTN_wind_speed"]
    # df = df[fields].rename(columns=dict(zip(fields, new_columns)))

    # apply ghi adjustment factor (if exists)
    # GHI_ADJ = oemeta.data["DTN_GHI_adj"].get(site)
    # if GHI_ADJ is not None:
    #     df["DTN_ghi"] = df["DTN_ghi"].mul(GHI_ADJ)

    # TODO: determine whether the following lines are necessary
    # calculate solar position for shifted timestamps
    idx = df.index + pd.Timedelta(30, unit="min")
    solar_position = pvlib.solarposition.get_solarposition(idx, lat, lon)
    solar_position.index = df.index  # but still report the values with original timestamps
    df = pd.merge(df, solar_position, left_index=True, right_index=True)

    # transposition
    dirint = pvlib.irradiance.dirint(
        df["shortWaveRadiation"],
        df["apparent_zenith"].values,
        df.index,
        max_zenith=85,
    )
    dirint = dirint.rename("dni_dirint")
    df_disc = pvlib.irradiance.complete_irradiance(
        solar_zenith=df["apparent_zenith"],
        ghi=df["shortWaveRadiation"],
        dni=dirint,
        dhi=None,
    )

    df_extra_dni = pvlib.irradiance.get_extra_radiation(df_disc.index)
    df_extra_dni = df_extra_dni.rename("dni_extra")

    dfDTN = pd.concat([df_disc, df, df_extra_dni], axis=1)
    dfDTN = dfDTN.drop(columns=["ghi"])
    dfDTN = dfDTN.rename(
        columns={
            "dni": "dni_raw",
            "dhi": "dhi_raw",
            "airTemp": "temp_air",
            "windSpeed": "wind_speed",
            "shortWaveRadiation": "ghi_raw",
        }
    )

    # apply losses
    losses_array, dc_losses, _ = get_site_model_losses(site)
    for col in ["ghi", "dni", "dhi"]:
        dfDTN[col] = dfDTN[f"{col}_raw"].mul(losses_array).mul(dc_losses)

    dfDTN = remove_tzinfo_and_standardize_index(dfDTN)
    if freq == "1min":
        naive_index = create_naive_index(start, end, freq)
        dfDTN = dfDTN.resample(freq).ffill()
        dfDTN = dfDTN.reindex(naive_index)

    return dfDTN


def get_poa_from_dtn_ghi(site, start, end, freq="1min", keep_tzinfo=False, q=True):
    """Generates POA time series data via a transposition model using external data"""
    qprint = lambda msg, end="\n": None if q else print(msg, end=end)

    # get external weather data
    args = [site, start, end]
    df_dtn = query_dtn_meteo_data(*args, freq=freq, q=q)
    df_sun = load_pvlib_clearsky(*args, freq=freq)

    meteo = df_dtn.copy()

    # combine and add timezone info
    tz = oemeta.data["TZ"].get(site)
    dropcols = ["apparent_elevation", "elevation", "equation_of_time"]
    df_naive = df_sun.drop(columns=dropcols).join(df_dtn)
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

    # get racking parameters (only uses df for sites with trackers)
    surface_tilt, surface_azimuth = get_surface_parameters(site, df)

    # transpose to POA
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


"""LEFT OFF HERE 02-19"""


def run_pvlib_model(site, df_met=None, start_date=None, end_date=None, q=True):
    qprint = lambda msg, end="\n": None if q else print(msg, end=end)
    if all(x is None for x in [df_met, start_date, end_date]):
        qprint("Error: must provide either df_met, or start/end dates")
        return
    if df_met is None:
        if not all(isinstance(d, str) for d in [start_date, end_date]):
            qprint("Error: start/end times must be strings of format %Y-%m-%d")
            return
        try:
            start, end = map(pd.Timestamp, [start_date, end_date])
        except Exception as e:
            qprint(f"Error with start/end dates: {e}")
            return

        qprint("Querying DTN data")
        df_met = run_pvlib_DTN_query(site, start, end)

        # transpose poa from ghi
        surface_tilt, surface_azimuth = get_surface_parameters(site, df_met)
        df_poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            dni=df_met["dni"],
            ghi=df_met["ghi"],
            dhi=df_met["dhi"],
            dni_extra=df_met["dni_extra"],
            solar_zenith=df_met["apparent_zenith"],
            solar_azimuth=df_met["azimuth"],
            model="perez",
        )
        df_met["POA_DTN"] = df_poa["poa_global"].copy()
        df_met["effective_irradiance"] = df_poa["poa_global"]  # .mul(losses_array)

    else:
        # verify/format df_met
        using_dtn = False
        df_met = pd.DataFrame()  # placeholder

    if df_met.empty:
        qprint("Error: no valid meteo data (df_met is empty)")
        return

    # get metadata
    location = get_site_location(site)

    # run models
    inv_names = get_inverter_names(site)
    inv_configs = get_inverter_configs(site)

    df_list = []
    for inv, config in zip(inv_names, inv_configs):
        meteo_mc = [df_met] * len(config.arrays)
        mc = ModelChain(config, location, aoi_model="physical", spectral_model="no_loss")
        mc.run_model_from_effective_irradiance(meteo_mc)

        # extract ac results
        ac_results = mc.results.ac / 1e3
        df_ = pd.DataFrame({f"{inv}_Possible_Power": ac_results.values}, index=ac_results.index)
        df_list.append(df_.copy())

    df = pd.concat(df_list, axis=1, ignore_index=False)
    return df
