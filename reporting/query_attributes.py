import itertools
from ..utils import oemeta


RENEWABLE_FLEET_AF_PATH = "\\\\CORP-PISQLAF\\Onward Energy\\Renewable Fleet"
SOLAR_AF_PATH = "\\".join([RENEWABLE_FLEET_AF_PATH, "Solar Assets"])


# reference dictionary of met station assets/attributes (by site) for use in monthly query
# OE.GHI, OE.POA, OE.AmbientTemp, OE.WindSpeed (tbd), -- OE.ModuleTemp1, OE.ModuleTemp2, ..
MET_STATION_ATTRIBUTE_DICT = {
    "Adams East": {
        "P003": [f"MST00{n}.Sts.Temp_C" for n in range(1, 5)],
        "P006": ["OE.POA", "WS001.Sts.WindSpdAvg_mps", "WS001.Sts.GHIrr_Wm2"],
        "P013": (
            ["OE.POA", "WS001.Sts.WindSpdAvg_mps", "WS001.Sts.GHIrr_Wm2"]
            + [f"MST00{n}.Sts.Temp_C" for n in range(2, 5)]
        ),
    },
    "Alamo": [
        "OE.POA",
        "OE.Wind_Speed",
        "MODULETEMPERATURE1",
        "MODULETEMPERATURE2",
        "IRRADIANCE_GHI",
    ],
    "AZ1": ["OE.POA", "WS_1.BOM_1.Temp", "WS_1.BOM_2.Temp", "WS_1.WS_1.Wind_Speed"],
    "Azalea": {
        "Met1": ["OE.Wind_Speed", "GHI1_3S"],
        "Met2": ["OE.POA", "OE.Wind_Speed", "GHI1_3S"],
        "Z02": [f"PLC_MTMP{n}_3S" for n in range(1, 7)],
    },
    "Camelot": {
        f"WS0{n}": ["OE.POA", "OE.Wind_Speed", "MODBPTEMP1_C", "MODBPTEMP2_C", "IRRADGLOBALHOR"]
        for n in [2, 3]
    },
    "Catalina II": ["OE.POA", "PNLTMP1", "PNLTMP2", "PYRIRR02_GHI", "PYRIRR03_GHI"],
    "CID": {
        "METB1P3": ["OE.POA", "PnlTemp_CC", "OE.Wind_Speed", "GHIrr_Wm2"],
        "METB1P7": ["PnlTemp_CC", "GHIrr_Wm2"],
    },
    "Columbia II": {
        "WS02": ["OE.POA", "OE.Wind_Speed", "MODBPTEMP1_C", "IRRADGLOBALHOR"],
        "WS03": ["OE.POA", "MODBPTEMP1_C", "MODBPTEMP2_C", "IRRADGLOBALHOR"],
    },
    "Comanche": {
        "ENV06": ["OE.POA", "OE.Wind_Speed", "TMPR_MDL1", "TMPR_MDL2", "TMPR_MDL3", "GHI_1"],
        "ENV16": ["OE.POA", "TMPR_MDL1", "TMPR_MDL2", "TMPR_MDL3", "GHI_1"],
        "ENV27": ["OE.POA"],
        "ENV43": ["OE.POA", "TMPR_MDL1", "TMPR_MDL2", "TMPR_MDL3"],
        "ENV50": ["OE.POA", "TMPR_MDL1", "TMPR_MDL2", "TMPR_MDL3", "GHI_1"],
        "ENV69": ["ch_2_Irradiance"],
        "ENV72": ["OE.POA"],
    },
    "CW-Corcoran": {
        "P002": ["WS001.Sts.WindSpdAvg_mps"] + [f"MST00{n}.Sts.Temp_C" for n in range(1, 5)],
        "P006": ["OE.POA", "WS001.Sts.WindSpdAvg_mps", "WS001.Sts.GHIrr_Wm2", "MST003.Sts.Temp_C"],
    },
    "CW-Goose Lake": ["OE.POA", "PnlTemp_C", "WindSpdAvg_mps", "GHIrr_Wm2"],
    "CW-Marin": ["OE.POA", "PnlTemp2_C", "AirTemp_C", "GHIrr_Wm2", "WindSpdAvg_mps"],
    "FL1": ["OE.POA", "OE.ModuleTemp"],
    "FL4": ["OE.POA", "PoA_2.Irr", "GHI_1.Irr", "WS_1.Wind_Speed"],
    "GA3": {
        "WS1": ["BOM_2_Temp", "BOM_3_Temp"],
        "WS2": ["OE.POA", "GHI_1_Irr"],
        "WS3": ["OE.POA", "BOM_1_Temp", "BOM_2_Temp", "BOM_3_Temp", "GHI_1_Irr"],
        "WS4": ["OE.POA", "BOM_1_Temp", "BOM_3_Temp", "GHI_1_Irr"],
    },
    "GA4": {
        "B1P03": ["OE.POA", "MSTemp001_C", "GHIrr_Wm2", "WindSpd_mps"],
        "B2P06": ["OE.POA", "MSTemp001_C", "GHIrr_Wm2", "WindSpd_mps"],
        "B3P03": ["OE.POA", "MSTemp001_C", "GHIrr_Wm2", "WindSpd_mps"],
        "B5P10": ["OE.POA", "GHIrr_Wm2", "WindSpd_mps"],
        "B6P02": ["OE.POA", "GHIrr_Wm2", "WindSpd_mps"],
    },
    "Grand View East": ["OE.POA", "GHI_1", "Wind_Speed"] + [f"BOM_Temp_{n}" for n in range(1, 4)],
    "Grand View West": {
        "B106": ["OE.POA", "GHI_1", "Wind_Speed"] + [f"BOM{n}" for n in range(1, 4)],
        "B204": ["OE.POA", "GHI_1", "Wind_Speed"] + [f"BOM{n}" for n in range(1, 4)],
        "B210": ["OE.POA"],
    },
    "Imperial Valley": ["OE.Wind_Speed", "IRRADIANCE_1_POA", "TEMP_C"],
    "Indy I": {
        "Met01": [f"OE.Module_Temp{n}" for n in range(1, 6)]
        + ["POA_3_Sec_Sample_POA1", "GHI_3_Sec_Sample_GHI1"],
        "Met02": ["POA_3_Sec_Sample_POA1", "GHI_3_Sec_Sample_GHI1"],
    },
    "Indy II": {
        "MET01": ["POA_3_Sec_Sample_POA1", "GHI_3_Sec_Sample_GHI1"],
        "MET02": (
            [f"OE.Module_Temp{n}" for n in range(2, 5)]
            + ["POA_3_Sec_Sample_POA1", "Wind_Speed_3_Sec_Sample", "GHI_3_Sec_Sample_GHI1"]
        ),
    },
    "Indy III": {
        "MET01": [f"OE.Module_Temp{n}" for n in range(1, 5)] + ["POA_3_Sec_Sample_POA1"],
        "MET02": ["POA_3_Sec_Sample_POA1", "GHI_3_Sec_Sample_GHI1"],
    },
    "Kansas": {
        "Met13": ["OE.POA", "PnlTemp_C", "WindSpdAvg_mps", "GHIrr_Wm2"],
        "Met4": ["PnlTemp_C", "WindSpdAvg_mps", "GHIrr_Wm2"],
    },
    "Kent South": {
        "Met3": ["OE.POA", "PnlTemp_C", "WindSpdAvg_mps", "GHIrr_Wm2"],
    },
    "Maplewood 1": ["OE.POA", "PO_BOMAvg_degC", "FI_Wthr1WS_ms"],
    "Maplewood 2": ["OE.POA", "FI_Wthr1WS_ms"],
    "Maricopa West": {
        "Met1": ["OE.POA", "OE.Module_Temp", "OE.Wind_Speed", "Global Horizontal Irradiance"],
        "Met2": ["OE.POA", "OE.Module_Temp", "OE.Wind_Speed", "Global Horizontal Irradiance"],
    },
    "MS3": {
        "Met1": ["OE.POA", "GHI"],
        "Met3": ["OE.POA", "GHI"],
    },
    "Mulberry": {
        "A4": ["OE.POA", "000003_MC_GENERIC_TEMPERATURE_MODULE_C"],
        "A6": ["OE.POA", "000003_MC_GENERIC_TEMPERATURE_MODULE_C"],
        "B2": ["OE.POA", "000003_MC_GENERIC_TEMPERATURE_MODULE_C"],
        "B6": ["OE.POA", "000003_MC_GENERIC_TEMPERATURE_MODULE_C"],
    },
    "Old River One": {
        "P002": ["MST001.Sts.Temp_C", "MST002.Sts.Temp_C"],
        "P006": (
            ["OE.POA", "WS001.Sts.WindSpdAvg_mps", "WS001.Sts.GHIrr_Wm2"]
            + [f"MST00{n}.Sts.Temp_C" for n in range(1, 5)]
        ),
    },
    "Pavant": {
        "Met04": ["OE.POA", "MTS1"],
        "Met13": ["OE.POA"],
        "Met18": ["OE.POA"],
    },
    "Richland": {
        "WS01": ["AIR_TEMP_ACT_C", "WIND_MS_ACT"],
        "WS02": ["OE.POA", "AIR_TEMP_ACT_C", "WIND_MS_ACT"],
    },
    "Sweetwater": ["OE.POA", "GHI_1", "Wind_Speed", "BOM_Temp_1", "BOM_Temp_2", "BOM_Temp_3"],
    "Three Peaks": ["OE.POA", "Wind_Speed", "GHI_1"],
    "West Antelope": {
        "Met1": ["OE.POA", "OE.Module_Temp", "OE.Wind_Speed", "Irradiance GHI"],
        "Met2": ["OE.POA", "OE.Wind_Speed", "Irradiance GHI"],
    },
}

# minute-level data
INV_MODULE_ATTRIBUTE_DICT = {
    "Adams East": [f"M{n}.Sts.Running" for n in range(1, 5)],
    "FL4": ["Mod_Num_Run"],  # should be number from 0 to 6 (g.t. 0)
    "GA3": ["Module_1_AC_Power_Active", "Module_2_AC_Power_Active"],  #  (g.t. 0)
    "GA4": [f"Module 00{n}.Sts.P_kW" for n in range(1, 7)],  #  (g.t. 0)
    # "Grand View East": ["OE.OnlineModules"],  # inverter-level attribute
    # "Grand View West": ["OE.OnlineModules"],  # inverter-level attribute
    "Imperial Valley": [f"MOD{n}_P_3PHASE" for n in range(1, 5)],  # (g.t. 0)
    "Kansas": [f"M{n}.Sts.Running" for n in range(1, 5)],
    "Kent South": [f"M{n}.Sts.Running" for n in range(1, 5)],
    "Maplewood 1": [f"INV_FI_Mod{n}_P_kW" for n in range(1, 7)],
    "Maplewood 2": [f"INV_FI_Mod{n}_P_kW" for n in range(1, 7)],
    "Old River One": [f"M{n}.Sts.Running" for n in range(1, 5)],
    "Sweetwater": ["U1.Active_power", "U2.Active_power"],
}


PPC_ATTRIBUTE_DICT = {
    "Comanche": ["xcel_MW_cmd_request", "xcel_MW_setpoint"],
    "Grand View East": [
        "Gens_Faulted",
        "Gens_Gross_KW",
        "Gens_Offline",
        "Meter_KW",
        "Power_Limit_SP",
    ],
    "Grand View West": [
        "Gens_Faulted",
        "Gens_Gross_KW",
        "Gens_Offline",
        "Meter_KW",
        "Power_Limit_SP",
    ],
}


def af_key(fleet):
    return f"AF_{fleet}_V3"


def sites_by_fleet():
    """returns dictionary with fleets as keys and list of sites as values"""
    return {fleet: [*oemeta.data[af_key(fleet)]] for fleet in ["Solar", "Wind"]}


def get_fleet(site: str):
    matching_fleets = [f for f, sitelist in sites_by_fleet().items() if site in sitelist]
    if not matching_fleets:
        raise KeyError("Invalid site")
    return matching_fleets[0]


def get_af_path(site: str):
    fleet = get_fleet(site)
    return "\\".join([RENEWABLE_FLEET_AF_PATH, f"{fleet} Assets", site])


def attribute_path(site: str, asset_hierarchy: list, attribute: str):
    """asset_hierarchy: [group, asset, subasset, ... ] or [] if site-level attribute"""
    return "\\".join([get_af_path(site), *asset_hierarchy]) + "|" + attribute


def get_af_dict(site: str, asset_group: str = None, asset: str = None) -> dict:
    """if asset is specified, returns f'{asset} Attributes' from asset dict"""
    fleet = get_fleet(site)
    af_dict = oemeta.data[af_key(fleet)].get(site)
    if not af_dict:
        return {}
    if asset_group is None:
        return af_dict  # returns dict with asset groups as keys
    if asset_group not in af_dict:
        return {}
    group_key = "_".join([asset_group, "Assets"])
    if asset is None:
        return af_dict[asset_group].get(group_key)  # returns dict with group asset names as keys
    attributes_key = "_".join([asset, "Attributes"])
    return af_dict[asset_group][group_key][asset].get(attributes_key)


def get_reporting_attribute_paths(site, asset_group) -> list:
    """returns list of attribute paths if exist, otherwise empty list"""
    # group- or site-level attributes (i.e. no assets)
    if asset_group == "PPC" and site in PPC_ATTRIBUTE_DICT:
        return [
            attribute_path(site, asset_hierarchy=[asset_group], attribute=att)
            for att in PPC_ATTRIBUTE_DICT[site]
        ]

    elif asset_group == "Meter":
        return [attribute_path(site, asset_hierarchy=[], attribute="OE.MeterMW")]  # site-level

    elif asset_group == "Modules" and site in INV_MODULE_ATTRIBUTE_DICT:
        group_af_dict = get_af_dict(site, asset_group="Inverters")
        asset_names = list(group_af_dict.keys())
        return [
            attribute_path(site, asset_hierarchy=["Inverters", inv], attribute=att)
            for inv, att in itertools.product(asset_names, INV_MODULE_ATTRIBUTE_DICT[site])
        ]

    # get group asset names from af dict (if exist)
    group_af_dict = get_af_dict(site, asset_group=asset_group)
    if not group_af_dict:
        return []
    asset_names = list(group_af_dict.keys())

    if asset_group == "Inverters":
        return [
            attribute_path(site, asset_hierarchy=[asset_group, asset], attribute="OE.ActivePower")
            for asset in asset_names
        ]

    elif asset_group == "Met Stations" and site in MET_STATION_ATTRIBUTE_DICT:
        atts = MET_STATION_ATTRIBUTE_DICT[site]
        att_path = lambda asset, att: attribute_path(site, [asset_group, asset], att)
        asset_att_dict = {a: atts for a in asset_names} if type(atts) is list else atts.copy()
        return list(
            itertools.chain.from_iterable(
                [att_path(asset, att) for att in att_list]
                for asset, att_list in asset_att_dict.items()
            )
        )

    # asset group not supported for site
    return []


def monthly_query_attribute_paths(site) -> dict:
    """returns dictionary with asset groups as keys and lists of attribute paths as values
    -> only groups with existing attributes will be included in output
    """
    query_groups = list(get_af_dict(site).keys())
    if "Meter" not in query_groups:
        query_groups.append("Meter")

    # check for inverter module attributes
    inv_mod_atts = INV_MODULE_ATTRIBUTE_DICT.get(site)
    if inv_mod_atts:
        query_groups.append("Modules")

    output = {
        asset_group: get_reporting_attribute_paths(site, asset_group)
        for asset_group in query_groups
    }
    return {key: val for key, val in output.items() if val}
