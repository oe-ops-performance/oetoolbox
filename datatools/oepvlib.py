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

from oetoolbox.utils import oemeta, oepaths
from oetoolbox.dataquery.external import query_DTN


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


def run_pvlib_DTN_query(site, start, end, q=True):
    # get location metadata
    tz = oemeta.data["TZ"].get(site)
    lat, lon = oemeta.data["LatLong"].get(site)

    # run dtn query
    interval = "hour"
    fields = ["airTemp", "shortWaveRadiation", "windSpeed"]
    start_date = pd.Timestamp(start).floor("D")
    end_date = pd.Timestamp(end).ceil("D") + pd.Timedelta(hours=1)  # add hour for upsample to min
    df_dtn = query_DTN(lat, lon, start_date, end_date, interval, fields, tz=tz)

    meteo = df_dtn.copy()

    # calculate solar position for shifted timestamps:
    idx = meteo.index + pd.Timedelta(30, unit="min")
    solar_position = pvlib.solarposition.get_solarposition(idx, lat, lon)

    # but still report the values with the original timestamps:
    solar_position.index = meteo.index
    meteo = pd.merge(meteo, solar_position, left_index=True, right_index=True)

    # transposition model
    dirint = pvlib.irradiance.dirint(
        meteo["shortWaveRadiation"],
        meteo["apparent_zenith"].values,
        meteo.index,
        max_zenith=85,
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
    losses_array, dc_losses, _ = get_site_model_losses(site)
    meteo["ghi"] = meteo["ghi_raw"].mul(losses_array).mul(dc_losses)
    meteo["dni"] = meteo["dni_raw"].mul(losses_array).mul(dc_losses)
    meteo["dhi"] = meteo["dhi_raw"].mul(losses_array).mul(dc_losses)

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

    losses_array, losses_dc, _ = get_site_model_losses(sitename)
    meteo["effective_irradiance"] = meteo["effective_irradiance"].copy().mul(losses_array)
    meteo["effective_irradiance"] = meteo["effective_irradiance"].copy().mul(losses_dc)

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

    tz = oemeta.data["TZ"].get(sitename)
    lat, lon = oemeta.data["LatLong"].get(sitename)
    alt = oemeta.data["Altitude"].get(sitename)
    location = pvlib.location.Location(lat, lon, tz, alt, sitename)
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

    # pull equipment data
    CEC_modules = pvlib.pvsystem.retrieve_sam("CECmod")
    CEC_inverters = pvlib.pvsystem.retrieve_sam("cecinverter")

    temperature_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
        "open_rack_glass_polymer"
    ]

    savepath = pvlib_output_folder  # default (if not for flashreport files)
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
        meteo = run_pvlib_DTN_query(sitename, Start_Date, End_Date, q=q)
        POA_Col_name = "POA_DTN"

    # get location metadata
    tz = oemeta.data["TZ"].get(sitename)
    lat, lon = oemeta.data["LatLong"].get(sitename)
    alt = oemeta.data["Altitude"].get(sitename)
    location = pvlib.location.Location(lat, lon, tz, alt, sitename)

    site_GCR = oemeta.data["Equip"]["Racking_GCR"].get(sitename)
    design_db = oemeta.data["Project_Design_Database"].get(sitename)
    inv_list = list(design_db.keys())

    inv_configs = []
    for inv, dict_inv in design_db.items():
        inv_arrays = []
        inv_type = dict_inv["Inverter Type (CEC)"]
        racking_tilt = int(dict_inv["Tilt Angle (Deg)"])
        racking_azimuth = int(dict_inv["Azimuth Angle (Deg)"])
        racking_type = dict_inv["Racking Type"]
        inv_params = CEC_inverters[inv_type]

        if racking_type == "Tracker":
            mount = pvlib.pvsystem.SingleAxisTrackerMount(
                axis_azimuth=racking_azimuth,
                axis_tilt=0,
                max_angle=racking_tilt,
                backtrack=True,
                gcr=site_GCR,
                racking_model="open_rack",
            )
            tracking_profile = pvlib.tracking.singleaxis(
                meteo["apparent_zenith"],
                meteo["azimuth"],
                max_angle=racking_tilt,
                backtrack=True,
                gcr=site_GCR,
            )
        elif racking_type == "Fixed":
            mount = pvlib.pvsystem.FixedMount(
                surface_tilt=racking_tilt,
                surface_azimuth=racking_azimuth,
                racking_model="open_rack",
            )

        cmb_keys = list(dict_inv.keys())
        combiner_list = cmb_keys[10:]

        for combiner in combiner_list:
            cmb_arrays = []
            dict_cmb = dict_inv[combiner]
            combiner_Mod_Type = dict_cmb["Module Type (CEC)"]
            mod_wattage_count_list = []
            total_combiner_mod_count = 0

            for x in range(1, 6):
                mod_watt_string = f"Module Wattage {x}"
                if dict_cmb[mod_watt_string] > 0:
                    mod_wattage_count_list.append(x)

            for x in mod_wattage_count_list:
                combiner_Mod_Count = dict_cmb[f"String Count {x}"]
                combiner_Mod_Length = dict_cmb[f"String Length {x}"]
                mod_count = combiner_Mod_Length * combiner_Mod_Count
                total_combiner_mod_count = total_combiner_mod_count + mod_count

                array = pvlib.pvsystem.Array(
                    mount=mount,
                    module_type="glass_polymer",
                    module_parameters=CEC_modules[combiner_Mod_Type],
                    temperature_model_parameters=temperature_params,
                    modules_per_string=combiner_Mod_Length,
                    strings=combiner_Mod_Count,
                )

                cmb_arrays.append(array)

            inv_arrays.append(cmb_arrays)

        inv_arrays = [array for cmb_array in inv_arrays for array in cmb_array]
        inv_config = pvlib.pvsystem.PVSystem(arrays=inv_arrays, inverter_parameters=inv_params)
        inv_configs.append(inv_config)

    dict_ac_results_by_config = {}
    DF_final_ac_results = pd.DataFrame()

    losses_array, _, _ = get_site_model_losses(sitename)

    config_count = len(inv_configs)
    for n, config in enumerate(inv_configs):
        config_length = len(config.arrays)
        meteo_mc = [meteo] * config_length
        mc = ModelChain(config, location, aoi_model="physical", spectral_model="no_loss")

        if "POA" in meteo.columns:
            msx_ = "" if (not poafromdtn) else " (transposed from DTN GHI)"
            qprint(
                f"Running model with POA{msx_} - part {n+1} / {config_count}",
                end_="\r" if (n + 1) < config_count else "\n",
            )
            mc.run_model_from_effective_irradiance(meteo_mc)

        elif "ghi" in meteo.columns:
            qprint("Running model with GHI transposed to POA - config #" + str(n + 1))
            POA_irradiance = pvlib.irradiance.get_total_irradiance(
                surface_tilt=(
                    tracking_profile["surface_tilt"] if racking_type == "Tracker" else racking_tilt
                ),
                surface_azimuth=(
                    tracking_profile["surface_azimuth"] if racking_type == "Tracker" else 180
                ),
                dni=meteo["dni"],
                ghi=meteo["ghi"],
                dhi=meteo["dhi"],
                dni_extra=meteo["dni_extra"],
                solar_zenith=meteo["apparent_zenith"],
                solar_azimuth=meteo["azimuth"],
                model="perez",
            )
            meteo[POA_Col_name] = POA_irradiance["poa_global"]
            meteo["effective_irradiance"] = POA_irradiance["poa_global"]  # .mul(losses_array)
            mc.run_model_from_effective_irradiance(meteo_mc)

        else:
            qprint("\n\n!! ERROR !! No POA or GHI columns selected\n\n")

        ## Pull AC Power from modeled Inv output ##
        Possible_power_ac_series = mc.results.ac / 1000
        Possible_power_ac_df = pd.DataFrame(
            {
                "Timestamp": Possible_power_ac_series.index,
                "Possible Power KW": Possible_power_ac_series.values,
            }
        )
        dict_ac_results_by_config[str(n)] = Possible_power_ac_df

    for n, inv in enumerate(inv_list):
        ## Add AC Possible Power by Inv into DF_final_results ##
        DF_inv_results = pd.DataFrame(dict_ac_results_by_config[str(n)].copy())
        DF_inv_results["Inv ID"] = "".join([inv, "_Possible_Power"])
        DF_final_ac_results = pd.concat([DF_final_ac_results, DF_inv_results])

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

    if isinstance(DF_Inv, pd.DataFrame):
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
            # dfi_h = dfi.copy().resample("h").mean()
            # fig_subplots = solarplots.inv_pvlib_subplots(
            #     sitename, dfinv=dfi_h, dfpvl=DF_PVLib_AC_hourly
            # )
            # if jupyter_plots:
            #     fig_subplots.show()

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

        if DF_Inv is not None:
            html_savepath = next_available_path(
                spath_, f"PVLib_vs_Actual_vs_POA_Graph_{f_suffix}", ".html"
            )
            fig_timeseries.update_layout(height=600)
            fig_timeseries.write_html(html_savepath)
            qprint(f'>> saved file: "{html_savepath.name}"\n{str(html_savepath)}')

            # tstamp_ = dt.datetime.now().strftime("%Y%m%d")
            # subplots_savepath = next_available_path(
            #     spath_, f"PVLIB_v_ACTUAL_comparisonSubplots{tstamp_}", ".html"
            # )
            # fig_subplots.write_html(subplots_savepath)
            # qprint(f'>> saved file: "{subplots_savepath.name}"\n{str(subplots_savepath)}')

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
