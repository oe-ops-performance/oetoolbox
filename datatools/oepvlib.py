import pvlib
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pathlib import Path
import datetime as dt
import pandas as pd
import numpy as np
import requests
import os, json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils import oemeta, oepaths


mdata_keys = ["SystemSize", "COD", "TZ", "LatLong", "Equip", "Altitude", "Losses"]
(dict_SystemSize, dict_COD, dict_TZ, dict_LatLong, dict_Equip, dict_Altitude, dict_Losses) = [
    oemeta.data[k] for k in mdata_keys
]


def get_site_model_losses(sitename, q=True):
    system_loss_df = pd.DataFrame()
    system_loss_degradation = round(
        (100 - ((dt.date.today().year - dict_COD[sitename]) * dict_Losses["Degradation"][sitename]))
        / 100,
        2,
    )

    system_loss_shading = (100 - dict_Losses[sitename]["shading"]) / 100
    system_loss_soiling = (100 - dict_Losses[sitename]["soiling"]) / 100

    system_loss_module_quality = (100 - dict_Losses[sitename]["quality"]) / 100
    system_loss_module_mismatch = (100 - dict_Losses[sitename]["mismatch"]) / 100
    system_loss_LID = (100 - dict_Losses[sitename]["lid"]) / 100
    system_loss_dc_ohmic = (100 - dict_Losses[sitename]["wiring"]) / 100

    system_loss_DC_adj = dict_Losses["DC_Adjustment"][sitename]
    system_loss_AC_adj = dict_Losses["AC_Adjustment"][sitename]

    system_losses_Array = (
        system_loss_degradation
        * system_loss_shading
        * system_loss_soiling
        * system_loss_module_quality
        * system_loss_module_mismatch
        * system_loss_LID
    )
    system_losses_DC = system_loss_dc_ohmic * system_loss_DC_adj
    system_losses_AC = system_loss_AC_adj

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
            "DC Adj": system_loss_DC_adj,
            "AC Adj": system_loss_AC_adj,
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


def request_access_token():
    ### Request Access Token ###
    api_token_url = "https://api.auth.dtn.com/v1/tokens/authorize"
    post_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    request_body = {
        "grant_type": "client_credentials",
        "client_id": "ee7jYBGim2s8HZa363d2u8AtqFcAy1qA",
        "client_secret": "t1dXBPELo4t-G2umw6OFY7bJgQbk9VO8v45JoqpZtFgOomB26NLl7RwNEz2HS-tG",
        "audience": "https://weather.api.dtn.com/conditions",
    }
    response = requests.post(api_token_url, data=json.dumps(request_body), headers=post_headers)
    access_token = response.json()["data"]["access_token"]
    return access_token


def split_timedelta(start_date, end_date):
    """
    Splits the timedelta between two dates into 10-day chunks.

    Parameters:
    start_date (str): start date in 'YYYY-MM-DD' format.
    end_date (str): end date in 'YYYY-MM-DD' format.

    Returns:
    List of tuples representing 10-day date ranges.
    """
    # Calculate timedelta between dates
    delta = end_date - start_date
    if delta.days <= 10:
        return [(start_date, end_date)]

    # Split timedelta into chunks of 10 days
    date_ranges = []
    current_date = start_date
    while (end_date - current_date).days >= 10:
        next_date = current_date + dt.timedelta(days=10)
        date_ranges.append((current_date, next_date))
        current_date = next_date

    # Add final date range if necessary
    if current_date < end_date:
        date_ranges.append((current_date, end_date))

    return date_ranges


def query_DTN(sitename, startdate=None, enddate=None, printouts=False):
    if printouts:
        print("!! USING DTN !!")
    ### Request API access token ###
    access_token = request_access_token()

    ### Load Project System Losses ###
    system_losses_Array, system_losses_DC, system_losses_AC = get_site_model_losses(
        sitename, q=(not printouts)
    )

    ### Set start/end dates for forecast ###
    if not (startdate and enddate):
        print("Problem with start or end date(s)... Aborting...")
        return
    tz = dict_TZ.get(sitename)
    start_date = pd.Timestamp(startdate, tz=tz).date() - pd.Timedelta(days=1)
    end_date = pd.Timestamp(enddate, tz=tz).date() + pd.Timedelta(days=1)

    # print("start_date: " + str(start_date))
    # print("end_date: " + str(end_date))

    interval = "hour"
    fields = "airTemp,shortWaveRadiation,windSpeed"
    headers = {
        "Accept-Encoding": "gzip",
        "Accept": "application/json",
        "Authorization": "Bearer " + access_token,
    }
    api_request_url = "https://weather.api.dtn.com/v1/conditions"

    Ten_day_ranges = split_timedelta(start_date, end_date)
    Lat, Long = dict_LatLong[sitename]

    df_project = pd.DataFrame()
    for date_range in Ten_day_ranges:
        start = str(date_range[0]) + "T00:00:00Z"
        end = str(date_range[1]) + "T00:00:00Z"
        querystring = {
            "interval": interval,
            "lat": Lat,
            "lon": Long,
            "parameters": fields,
            "startTime": start,
            "endTime": end,
        }
        response = requests.request("GET", api_request_url, headers=headers, params=querystring)
        response_dict = json.loads(response.text)
        data = response_dict["content"]["items"][0]["parameters"]
        df_temp = pd.DataFrame.from_dict(data)
        df_project = pd.concat([df_project, df_temp])

    df_project.index = pd.to_datetime(df_project.index)
    df_project_localized = df_project.tz_convert(dict_TZ[sitename])
    meteo = df_project_localized

    meteo = meteo[
        ~meteo.index.duplicated(keep="first")
    ]  # remove extra tstamps from fall dst adjustment

    # calculate solar position for shifted timestamps:
    idx = meteo.index + pd.Timedelta(30, unit="min")
    solar_position = pvlib.solarposition.get_solarposition(idx, Lat, Long)

    # but still report the values with the original timestamps:
    solar_position.index = meteo.index
    meteo = pd.merge(meteo, solar_position, left_index=True, right_index=True)

    # transposition model
    dirint = pvlib.irradiance.dirint(
        meteo["shortWaveRadiation"], meteo["apparent_zenith"].values, meteo.index, max_zenith=85
    )
    dirint = dirint.rename("dni_dirint")

    df_disc = pvlib.irradiance.complete_irradiance(
        solar_zenith=meteo["apparent_zenith"], ghi=meteo["shortWaveRadiation"], dni=dirint, dhi=None
    )

    df_extra_dni = pvlib.irradiance.get_extra_radiation(df_disc.index)
    df_extra_dni = df_extra_dni.rename("dni_extra")

    meteo = pd.concat([df_disc, meteo, df_extra_dni], axis=1)
    meteo = meteo.drop(columns=["ghi"])
    meteo = meteo.rename(
        columns={
            "dni": "dni_raw",
            "dhi": "dhi_raw",
            "airTemp": "temp_air",
            "windSpeed": "wind_speed",
            "shortWaveRadiation": "ghi_raw",
        }
    )

    ## Apply IRR Losses ##
    meteo["ghi"] = meteo["ghi_raw"].mul(system_losses_Array)  # .mul(system_losses_DC)
    meteo["dni"] = meteo["dni_raw"].mul(system_losses_Array)  # .mul(system_losses_DC)
    meteo["dhi"] = meteo["dhi_raw"].mul(system_losses_Array)  # .mul(system_losses_DC)
    # print(meteo.describe())
    return meteo


def prepare_project_meteo_DF(sitename, DF_Met, printouts):
    if printouts:
        print("\nAdjusting meteo dataframe for model")
    meteo = DF_Met.copy()
    if not isinstance(meteo.index, pd.DatetimeIndex):
        meteo["Timestamp"] = pd.to_datetime(meteo["Timestamp"])
        meteo.set_index("Timestamp", inplace=True, drop=True)
    meteo = meteo.dropna(axis=1, how="all")

    matching_cols = lambda cols, id_list: [
        c for c in cols if "Processed" in c and any(i.casefold() in c.casefold() for i in id_list)
    ]
    print_rename = lambda old, new: (
        print(f'    >> renamed "{old}" column to "{new}"') if printouts else None
    )

    poa_cols = matching_cols(meteo.columns, ["POA", "Plane", "Irradiance"])
    if poa_cols:
        meteo["effective_irradiance"] = meteo[poa_cols[0]].copy()
        meteo.rename(columns={poa_cols[0]: "POA"}, inplace=True)
        print_rename(poa_cols[0], "POA")

    modtemp_cols = matching_cols(meteo.columns, ["Mod", "MDL", "MTS"])
    if modtemp_cols:
        meteo.rename(columns={modtemp_cols[0]: "module_temperature"}, inplace=True)
        print_rename(modtemp_cols[0], "module_temperature")

    ambtemp_cols = matching_cols(meteo.columns, ["Amb", "TEMP_C", "AIR_TEMP", "temp_air"])
    if ambtemp_cols:
        meteo.rename(columns={ambtemp_cols[0]: "ambient_temperature"}, inplace=True)
        print_rename(ambtemp_cols[0], "ambient_temperature")

    windspd_cols = matching_cols(meteo.columns, ["Wind", "Speed"])
    if windspd_cols:
        meteo.rename(columns={windspd_cols[0]: "wind_speed"}, inplace=True)
        print_rename(windspd_cols[0], "wind_speed")

    meteo["Hour"] = meteo.index.hour
    meteo["Day"] = meteo.index.day
    meteo["Month"] = meteo.index.month
    meteo["Year"] = meteo.index.year

    ### Load Project System Losses ###
    system_losses_Array, system_losses_DC, system_losses_AC = get_site_model_losses(
        sitename, q=(not printouts)
    )
    meteo["effective_irradiance"] = meteo["effective_irradiance"].copy().mul(system_losses_Array)
    meteo["effective_irradiance"] = meteo["effective_irradiance"].copy().mul(system_losses_DC)

    meteo = meteo[
        ~meteo.index.duplicated(keep="first")
    ]  # remove extra tstamps from fall dst adjustment

    all_cols = list(meteo.columns)
    no_modtemp = "module_temperature" not in all_cols

    c1 = all(i in all_cols for i in ["ambient_temperature", "wind_speed"])
    c2 = (meteo["ambient_temperature"].sum() != 0) if c1 else True
    alternate_cols = c1 and c2

    if no_modtemp:
        if not alternate_cols:
            print("Missing Required Inputs")
            return
        temperature_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
            "open_rack_glass_polymer"
        ]
        meteo["module_temperature"] = (
            meteo["POA"].copy()
            * np.exp(
                temperature_params["a"] + (temperature_params["b"] * meteo["wind_speed"].copy())
            )
            + meteo["ambient_temperature"].copy()
        )

    keep_cols = [
        "Year",
        "Month",
        "Day",
        "Hour",
        "module_temperature",
        "effective_irradiance",
        "POA",
    ]
    if "POA_all_bad" in meteo.columns:
        keep_cols.append("POA_all_bad")
    meteo = meteo[keep_cols].copy()

    Lat, Long = dict_LatLong[sitename]
    tz, Altitude = dict_TZ[sitename], dict_Altitude[sitename]
    location = pvlib.location.Location(Lat, Long, tz, Altitude, sitename)
    solar_position = location.get_solarposition(meteo.index)

    meteo = pd.merge(solar_position, meteo, left_index=True, right_index=True)

    return meteo


Paco_adjustments = {
    "Alamo": 935000.0,
    "Adams East": 1482000.0,
    "Catalina II": 782600.0,
    "CID": 840000.0,
    "CW-Corcoran": 840000.0,
    "CW-Goose Lake": 840000.0,
    "GA3": 3330000.0,
    "GA4": 3330000.0,
    "Indy I": 714000.0,
    "Indy II": 714000.0,
    "Indy III": 720000.0,
    "Kent South": 1429000.0,
    "Old River One": 1460000.0,
    "Mulberry": 880000.0,
    "Maricopa West": 2200000.0,
    "Pavant": 2174000.0,
    "Richland": 1833000.0,
    "Selmer": 880000.0,
}
pvlib_output_folder = os.path.join(oepaths.released, "PVLib", "Output")


def run_pvlib_model(
    sitename,
    from_FRfiles=False,
    FR_yearmonth=None,
    start_date=None,
    end_date=None,
    DF_Met=None,
    DF_Inv=None,
    save_files=True,
    jupyter_plots=False,
    printouts=False,
    q=True,
    force_DTN=False,
    overwrite=False,
    localsave=False,
    return_df_and_fpath=False,
    custom_savepath=None,
):
    qprint = lambda msg, end_="\n": print(msg, end=end_) if not q else None
    c1 = from_FRfiles and FR_yearmonth
    c2 = start_date and end_date
    c3 = isinstance(DF_Met, pd.DataFrame) and isinstance(DF_Inv, pd.DataFrame)
    if not any([c1, c2, c3]):
        print("\n!! not all input requirements met; check parameters !!\n")
        return None

    qprint("\n" + "#" * 100)
    qprint(f"\nGenerating PVLib model for: {sitename}")

    ### load metadata ###
    Lat, Long = dict_LatLong[sitename]
    Timezone, Altitude = dict_TZ[sitename], dict_Altitude[sitename]

    equip_keys = [
        "Inv_Count",
        "Inv_Type",
        "Inv_Config",
        "Mod_Type",
        "String_Length",
        "String_Count_Per_Inv",
        "Racking_Type",
        "Racking_Tilt",
        "Racking_GCR",
    ]
    (
        inverter_count,
        inverter_type,
        inverter_config,
        module_type,
        string_length,
        strings_per_inv,
        racking_type,
        racking_tilt,
        racking_gcr,
    ) = [dict_Equip[key].get(sitename) for key in equip_keys]

    # pull equipment data
    CEC_modules = pvlib.pvsystem.retrieve_sam("CECmod")
    CEC_inverters = pvlib.pvsystem.retrieve_sam("cecinverter")

    temperature_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
        "open_rack_glass_polymer"
    ]
    module_params = (
        [CEC_modules[mod] for mod in module_type]
        if isinstance(module_type, list)
        else CEC_modules[module_type]
    )
    inverter_params = (
        [CEC_inverters[inv] for inv in inverter_type]
        if isinstance(inverter_type, list)
        else CEC_inverters[inverter_type]
    )

    # invparams_dict = {inv: CEC_inverters[inv] for inv in inverter_type}   #test

    if (sitename in Paco_adjustments) and isinstance(inverter_type, list):
        for param in inverter_params:
            param["Paco"] = Paco_adjustments[sitename]

    savepath = pvlib_output_folder  # default (if not for flashreport files)
    # DF_Met = DF_Inv = dfi = None   #init/default
    dfi = None  # init/default
    if from_FRfiles:
        fr_folder = os.path.join(oepaths.flashreports, FR_yearmonth, "Solar", sitename)
        filtered_files = lambda keys: [
            f for f in os.listdir(fr_folder) if all(i in f for i in keys)
        ]
        fpath_ = lambda file: os.path.join(fr_folder, file)
        if os.path.exists(fr_folder):
            disppath = fr_folder.split("Internal Documents")[-1]
            qprint(f'\nFound FlashReport folder: "{disppath}"')

            frp = Path(fr_folder)
            mfpaths = list(frp.glob("PIQuery*MetStations*CLEANED*PROCESSED*.csv"))
            if mfpaths:
                mfp = max((fp.stat().st_ctime, fp) for fp in mfpaths)[1]
                DF_Met = pd.read_csv(mfp)
                qprint(f'    >> loaded meteo file: "{mfp.name}"')

            i_files = filtered_files(["Inverters", "PIQuery", ".csv"])
            if i_files:
                cln_i_files = [f for f in i_files if "CLEANED" in f]
                ifiname = cln_i_files[0] if cln_i_files else i_files[0]
                ifilepath = fpath_(ifiname)
                DF_Inv = pd.read_csv(ifilepath)
                dfi = DF_Inv.copy()
                dfi["Timestamp"] = pd.to_datetime(dfi["Timestamp"])
                dfi = dfi.set_index("Timestamp")
                qprint(f'    >> loaded inverter file: "{ifiname}"')
            savepath = fr_folder
            qprint("")
        else:
            qprint(
                f"\nThe following FlashReport path does not exist:\n{fr_folder}\n>>saving to local downloads folder."
            )
            savepath = str(Path.home() / "Downloads")
    elif custom_savepath is not None:
        savepath = custom_savepath
        qprint(">> saving to user defined path")
    else:
        qprint(f"{localsave = }") if localsave else qprint(f"{from_FRfiles = }")
        qprint(">> saving to local downloads folder.")
        savepath = str(Path.home() / "Downloads")

    ### Prepare Meteo DF ###
    poafromdtn = None  # init
    if (not force_DTN) and isinstance(DF_Met, pd.DataFrame):
        qprint("\nDF_Met provided!")
        meteo = prepare_project_meteo_DF(sitename, DF_Met, printouts=printouts)
        Start_Date = meteo.index.min().date()
        End_Date = meteo.index.max().date() + dt.timedelta(days=1)
        if not from_FRfiles:
            qprint(f"\nStart_Date = {str(Start_Date)}\nEnd_Date = {str(End_Date)}")
        poafromdtn = "Average_Across_POA" not in DF_Met.columns
        POA_Col_name = "POA_DTN" if poafromdtn else "POA"

    else:
        # query DTN data
        if from_FRfiles:
            qprint("\nNo DF_Met found!") if (not force_DTN) else qprint("force_DTN=True")
            year, month = int(FR_yearmonth[:4]), int(FR_yearmonth[4:])
            Start_Date = dt.datetime(year=year, month=month, day=1).date()
            end_tstamp = pd.Timestamp(Start_Date) + pd.DateOffset(months=1)
            End_Date = end_tstamp.date()
        else:
            Start_Date, End_Date = map(lambda d: pd.Timestamp(d).date(), [start_date, end_date])

        qprint("\nQuerying DTN weather data...")
        meteo = query_DTN(sitename, Start_Date, End_Date, printouts=(not q))
        POA_Col_name = "POA_DTN"

    coordinates = [(Lat, Long, sitename, Altitude, Timezone)]
    location = pvlib.location.Location(Lat, Long, Timezone, Altitude, sitename)

    ### Define racking parameters ###
    if racking_type == "Tracker":
        mount = pvlib.pvsystem.SingleAxisTrackerMount(
            axis_azimuth=0,
            axis_tilt=0,
            max_angle=racking_tilt,
            backtrack=True,
            gcr=racking_gcr,
            racking_model="open_rack",
        )
        tracking_profile = pvlib.tracking.singleaxis(
            meteo["apparent_zenith"],
            meteo["azimuth"],
            max_angle=racking_tilt,
            backtrack=True,
            gcr=racking_gcr,
        )
    elif racking_type == "Fixed":
        mount = pvlib.pvsystem.FixedMount(
            surface_tilt=racking_tilt, surface_azimuth=180, racking_model="open_rack"
        )

    configs = []
    array_list = []  # temp
    for n, inverter in enumerate(inverter_params):
        print(
            "this is module_params[n]"
        )  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print(module_params[n])
        array = pvlib.pvsystem.Array(
            mount=mount,
            module_type="glass_polymer",
            module_parameters=module_params[n],
            temperature_model_parameters=temperature_params,
            modules_per_string=string_length[n],
            strings=strings_per_inv[n],
        )
        # print(f"ARRAY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!:{array}")
        # array_list.append(array)   #temp
        # print('TEST inv_params') #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print(inverter)

        config = pvlib.pvsystem.PVSystem(arrays=[array], inverter_parameters=inverter)
        configs.append(config)

    # pvsystem = pvlib.pvsystem.PVSystem(arrays=array_list, inverter_parameters=dict(inverter_params))   #temp test

    dict_ac_results_by_config = {}
    DF_final_ac_results = pd.DataFrame()

    system_losses_Array, system_losses_DC, system_losses_AC = get_site_model_losses(
        sitename, q=True
    )

    # mc = ModelChain(configs, location, aoi_model='physical', spectral_model="no_loss")
    # mc = ModelChain(pvsystem, location, aoi_model='physical', spectral_model="no_loss")   #test
    # mc.run_model_from_effective_irradiance(meteo)

    config_count = len(configs)
    for n, config in enumerate(configs):
        ## Run ModelChain for each Inv config ##
        # print('in loop')                                                                                       #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print(type(config))
        # print(f'\n[config]\n\t{config}')

        mc = ModelChain(config, location, aoi_model="physical", spectral_model="no_loss")
        if "POA" in meteo.columns:
            msx_ = "" if (not poafromdtn) else " (transposed from DTN GHI)"
            qprint(
                f"Running model with POA{msx_} - part {n+1} / {config_count}",
                end_="\r" if (n + 1) < config_count else "\n",
            )
            mc.run_model_from_effective_irradiance(meteo)
        elif "ghi" in meteo.columns:
            qprint("Running model with GHI transposed to POA - config #" + str(n + 1))
            POA_irradiance = pvlib.irradiance.get_total_irradiance(
                surface_tilt=(
                    tracking_profile["surface_tilt"] if racking_type == "Tracker" else racking_tilt
                ),
                surface_azimuth=(
                    tracking_profile["surface_azimuth"] if racking_type == "Tracker" else 180
                ),
                dni=meteo["dni_raw"],
                ghi=meteo["ghi_raw"],
                dhi=meteo["dhi_raw"],
                dni_extra=meteo["dni_extra"],
                solar_zenith=meteo["apparent_zenith"],
                solar_azimuth=meteo["azimuth"],
                model="perez",
            )
            meteo[POA_Col_name] = POA_irradiance["poa_global"]
            meteo["effective_irradiance"] = POA_irradiance["poa_global"].mul(system_losses_Array)
            mc.run_model_from_effective_irradiance(meteo)

        else:
            qprint("\n\n!! ERROR !! No POA or GHI columns selected\n\n")

        ## Pull AC Power from modeled Inv output and apply AC System loss Adj ##
        Possible_power_ac_series = mc.results.ac / 1000  # *system_losses_AC
        Possible_power_ac_df = pd.DataFrame(
            {
                "Timestamp": Possible_power_ac_series.index,
                "Possible Power KW": Possible_power_ac_series.values,
            }
        )
        dict_ac_results_by_config[n] = Possible_power_ac_df

    # print(f'\n{inverter_config = }\n')

    for inv in inverter_config:
        ## Add AC Possible Power by Inv into DF_final_results ##
        DF_inv_results = pd.DataFrame(dict_ac_results_by_config[inverter_config[inv]].copy())
        DF_inv_results["Inv ID"] = "".join([inv, "_Possible_Power"])
        DF_final_ac_results = pd.concat([DF_final_ac_results, DF_inv_results])

    # print(f'\n{DF_final_ac_results.columns = }\n')
    # return DF_final_ac_results

    today_date = dt.datetime.now().strftime("%Y-%m-%d")
    f_suffix = f"{sitename}_{Start_Date}_to_{End_Date}_created_{today_date}"

    ## Pivot AC Possible Power by Inv, merge with Meteo and save Output to csv ##
    DF_final_ac_results["Possible Power KW"] = DF_final_ac_results["Possible Power KW"].where(
        DF_final_ac_results["Possible Power KW"] >= 0, 0
    )
    DF_final_ac_results = DF_final_ac_results.pivot_table(
        index="Timestamp", values="Possible Power KW", columns="Inv ID", aggfunc="mean"
    )
    DF_PVLib_AC_results = pd.merge(meteo, DF_final_ac_results, left_index=True, right_index=True)
    DF_PVLib_AC_results.index.name = "Timestamp"
    drop_cols = [
        "ghi",
        "dni",
        "dhi",
        "dni_raw",
        "dhi_raw",
        "dni_extra",
        "apparent_zenith",
        "zenith",
        "apparent_elevation",
        "elevation",
        "azimuth",
        "equation_of_time",
        "effective_irradiance",
    ]
    try:
        DF_PVLib_AC_results = DF_PVLib_AC_results.drop(columns=drop_cols)
        DF_PVLib_AC_results = DF_PVLib_AC_results.rename(columns={"ghi_raw": "ghi"})
    except:
        None

    DF_PVLib_AC_results = DF_PVLib_AC_results.tz_localize(None)
    DF_PVLib_AC_results = DF_PVLib_AC_results.loc[Start_Date : pd.Timestamp(End_Date)].copy()

    DF_PVLib_AC_hourly = DF_PVLib_AC_results.resample("h").mean()
    if "POA_all_bad" in DF_PVLib_AC_hourly.columns:
        DF_PVLib_AC_hourly["POA_minutes_filled"] = DF_PVLib_AC_hourly["POA_all_bad"].mul(60)
        POA_Minutes_Col = DF_PVLib_AC_hourly.pop("POA_minutes_filled")
        DF_PVLib_AC_hourly.insert(10, "POA_minutes_filled", POA_Minutes_Col)

    # qprint('\nDF_PVLIB_AC_RESULTS:')

    if isinstance(DF_Inv, pd.DataFrame):
        # qprint('\nDF_Inv provided!')
        DF_Inv = DF_Inv.replace("[-11059] No Good Data For Calculation", np.nan)
        keepcols = [c for c in DF_Inv if "PROCESSED" not in c]
        DF_Inv = DF_Inv[keepcols].copy()
        time_col = list(DF_Inv.filter(regex="Read|Time|TIME").columns)[0]
        power_col = list(DF_Inv.filter(regex="Active|Power|POWER|kW|TOTW").columns)[0]
        id_cols = list(DF_Inv.filter(regex="Inverters_ID|AssetTitle").columns)
        ID_col = id_cols[0] if id_cols else None
        DF_Inv[power_col] = DF_Inv[power_col].astype(float)
        if ID_col:
            DF_Inv[time_col] = pd.to_datetime(DF_Inv[time_col])
            DF_Inv = pd.pivot_table(DF_Inv, index=time_col, columns=ID_col, values=power_col)
            DF_Inv[time_col] = DF_Inv.index

        DF_Inv[time_col] = pd.to_datetime(DF_Inv[time_col])
        DF_Inv = DF_Inv.set_index(time_col, drop=True)
        # DF_Inv = DF_Inv.dropna(how='all', axis=1)                   ###############################
        DF_Inv = DF_Inv.astype(float)
        DF_Inv = DF_Inv.add_suffix("_Actual_Power")
        DF_Inv_Hourly = DF_Inv.resample("h").mean().copy()

        ## Merge Possible Inv Power with Actual Inv Power ##
        DF_Poss_Actual = DF_PVLib_AC_results.merge(DF_Inv, left_index=True, right_index=True).copy()
        DF_Poss_Actual_Hourly = DF_Poss_Actual.resample("h").mean().copy()

        Poss_Power_Cols = list(DF_Poss_Actual.filter(regex="Possible_Power").columns)
        Actual_Power_Cols = list(DF_Poss_Actual.filter(regex="Actual_Power").columns)

        Actual_InvIDs = [
            inv
            for inv in list(DF_Inv_Hourly.columns)
            if any(x in inv for x in ["Inv", "INV", "Actual_Power"])
        ]
        PVLib_InvIDs = [
            inv
            for inv in list(DF_PVLib_AC_hourly.columns)
            if any(x in inv for x in ["Inv", "INV", "Possible_Power"])
        ]

        inv_names = [
            i.replace("OE.ActivePower_", "").replace("_Actual_Power", "") for i in Actual_Power_Cols
        ]

        if jupyter_plots or save_files:
            # qprint('\n\nBEGIN PLOT OUTPUTS\n')
            default_margin = dict(t=20, r=20, b=20, l=20)

            if jupyter_plots:
                qprint("\n|| SCATTER PLOT - POWER VS. POA\n  >> Actual vs. PVLib (site-level)")
            scattertemplate = (
                "<b>%{fullData.name}</b><br>"
                "Power: %{y:.2f} kW<br>POA: %{x:.2f} kW/m2<br>"
                "<i>%{customdata|%Y-%m-%d %H:%M:%S}</i><extra></extra>"
            )
            DF_Poss_Actual_Hourly["Total Actual"] = DF_Poss_Actual_Hourly[Actual_InvIDs].sum(axis=1)
            DF_Poss_Actual_Hourly["Total Possible"] = DF_Poss_Actual_Hourly[PVLib_InvIDs].sum(
                axis=1
            )

            fig_scatter = go.Figure()
            dfsc = DF_Poss_Actual_Hourly.copy()

            for col in ["Total Possible", "Total Actual"]:
                fig_scatter.add_trace(
                    go.Scatter(
                        x=dfsc["POA"],
                        y=dfsc[col],
                        name=col,
                        mode="markers",
                        customdata=dfsc.index,
                        hovertemplate=scattertemplate,
                    )
                )

            fig_scatter.update_layout(
                margin=default_margin,
                width=800,
                height=300,
                font_size=10,
                yaxis_title="Power (kW)",
                xaxis_title="POA (kW/m2)",
            )
            if jupyter_plots:
                fig_scatter.show()

            pwr_hovtemplate = (
                "<b>%{fullData.name}</b>: %{y:.2f} kW<br>"
                "<i>%{x|%Y-%m-%d %H:%M:%S}</i><extra></extra>"
            )
            irrad_hovtemplate = (
                "<b>%{fullData.name}</b>: %{y:.2f} kW/m2<br>"
                "<i>%{x|%Y-%m-%d %H:%M:%S}</i><extra></extra>"
            )

            fig_timeseries = make_subplots(specs=[[{"secondary_y": True}]])

            xy1 = dict(
                x=DF_PVLib_AC_hourly.index, y=DF_PVLib_AC_hourly[Poss_Power_Cols].mean(axis=1)
            )
            kwargs1 = dict(
                name="Avg_Possible", mode="lines", fill="tozeroy", hovertemplate=pwr_hovtemplate
            )
            fig_timeseries.add_trace(go.Scatter(**xy1, **kwargs1), secondary_y=False)

            xy2 = dict(x=DF_PVLib_AC_hourly.index, y=DF_PVLib_AC_hourly["POA"])
            kwargs2 = dict(name="POA", mode="lines", hovertemplate=irrad_hovtemplate)
            fig_timeseries.add_trace(go.Scatter(**xy2, **kwargs2), secondary_y=True)

            for i, inv in enumerate(Actual_InvIDs):
                df_temp = DF_Inv_Hourly[[inv]].copy()
                kwargs = dict(
                    name=f"{inv_names[i]}_Actual", mode="lines", hovertemplate=pwr_hovtemplate
                )
                fig_timeseries.add_trace(
                    go.Scatter(x=df_temp.index, y=df_temp[inv], **kwargs), secondary_y=False
                )

            fig_timeseries.update_layout(
                margin=default_margin,
                font_size=10,
                width=800,
                height=300,
                yaxis_title="Power (kW)",
            )
            if jupyter_plots:
                fig_timeseries.show()

            # inv/pvlib comparison plots
            dfi_h = dfi.copy().resample("h").mean()
            fig_subplots = solarplots.inv_pvlib_subplots(
                sitename, dfinv=dfi_h, dfpvl=DF_PVLib_AC_hourly
            )
            if jupyter_plots:
                fig_subplots.show()

    def next_available_path(path_, fstem_, ext_):
        n, fname_ = 1, f"{fstem_}{ext_}"
        while Path(path_, fname_).exists():
            fname_ = f"{fstem_}({n}){ext_}"
            n += 1
        return Path(path_, fname_)

    if save_files:
        if custom_savepath is not None:
            DF_PVLib_AC_results.to_csv(custom_savepath)
            qprint(f'\n>> saved file: "{Path(custom_savepath).name}"\n{str(custom_savepath)}\nEnd.')
            return DF_PVLib_AC_results

        spath_ = Path(savepath)

        if overwrite:
            # find/remove all existing processed html files
            existing_pvlfps = list(spath_.glob("PVLib*"))
            for fp_ in existing_pvlfps:
                os.remove(str(fp_))
                qprint(f'!! overwrite=True; removed file "{fp_.name}" !!')

        ac_savepath = next_available_path(spath_, f"PVLib_InvAC_{f_suffix}", ".csv")
        DF_PVLib_AC_results.to_csv(ac_savepath)
        qprint(f'\n>> saved file: "{ac_savepath.name}"\n{str(ac_savepath)}')

        # ac_hourly_savepath = next_available_path(spath_, f'PVLib_InvAC_Hourly_{f_suffix}', '.csv')
        # DF_PVLib_AC_hourly.to_csv(ac_hourly_savepath)
        # qprint(f'>> saved file: "{ac_hourly_savepath.name}"')

        if DF_Inv is not None:
            html_savepath = next_available_path(
                spath_, f"PVLib_vs_Actual_vs_POA_Graph_{f_suffix}", ".html"
            )
            fig_timeseries.update_layout(height=600)
            fig_timeseries.write_html(html_savepath)
            qprint(f'>> saved file: "{html_savepath.name}"\n{str(html_savepath)}')

            tstamp_ = dt.datetime.now().strftime("%Y%m%d")
            subplots_savepath = next_available_path(
                spath_, f"PVLIB_v_ACTUAL_comparisonSubplots{tstamp_}", ".html"
            )
            fig_subplots.write_html(subplots_savepath)
            qprint(f'>> saved file: "{subplots_savepath.name}"\n{str(subplots_savepath)}')

    qprint("\n\n!!! DONE !!!")
    if return_df_and_fpath:
        return DF_PVLib_AC_results, ac_savepath

    return DF_PVLib_AC_results


def run_flashreport_pvlib_model(
    site,
    year,
    month,
    savefile=True,
    overwrite=False,
    localsave=False,
    displayplot=False,
    q=True,
    return_df=False,
):
    kwargs_ = dict(
        sitename=site,
        from_FRfiles=True,
        FR_yearmonth=f"{year}{month:02d}",
        save_files=savefile,
        jupyter_plots=displayplot,
        q=q,
        overwrite=overwrite,
        localsave=localsave,
    )
    output_data = run_pvlib_model(**kwargs_)
    if return_df:
        return output_data
    return
