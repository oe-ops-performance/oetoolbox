import itertools
from ..utils import oemeta
from ..utils.config import PI_DATABASE_PATH


RENEWABLE_FLEET_AF_PATH = f"{PI_DATABASE_PATH}\\Renewable Fleet"
SOLAR_AF_PATH = "\\".join([RENEWABLE_FLEET_AF_PATH, "Solar Assets"])


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


OE_SOLAR_ATTRIBUTES = {
    "Inverters": ["OE.ActivePower"],
    "Met Stations": ["OE.Ambient_Temp", "OE.Module_Temp", "OE.POA", "OE.Wind_Speed"],
    "Meter": ["OE_MeterMW"],
    "PPC": ["xcel_MW_cmd_request", "xcel_MW_setpoint"],  # TEMP: only for Comanche
    "Modules": ["OE.ModulesOffline", "OE.ModulesOffline_Percent"],  # ac module related
}


def _build_solar_attpath_list(site, group=None, asset_names=[], attribute_names=[], validated=True):
    if get_fleet(site) != "Solar":
        raise ValueError("This function currently only supports solar sites.")

    # validation current only applies for asset-level attributes
    if not attribute_names:
        raise ValueError("Must provide attribute names.")

    af_path = f"{SOLAR_AF_PATH}\\{site}"
    if group is not None:
        af_path += f"\\{group}"

    if not asset_names:
        return [f"{af_path}|{att}" for att in attribute_names]

    attpaths = []

    is_valid = lambda asset_, att_: True  # init/default
    if validated is True:
        existing_atts = oemeta.data["solar_attributes_by_group_by_asset"][site].get(group, {})
        is_valid = lambda asset_, att_: (
            True if not existing_atts else (att_ in existing_atts.get(asset_, []))
        )
    for asset in asset_names:
        attpaths.extend(
            [f"{af_path}\\{asset}|{att}" for att in attribute_names if is_valid(asset, att)]
        )
    return attpaths


def get_solar_attributes(site, validated=True) -> dict:
    """returns dictionary with asset groups as keys and lists of attribute paths as values"""
    af_struct = oemeta.data["solar_af_structure"].get(site, None)
    if af_struct is None:
        raise KeyError(f"No solar af structure data found for {site = }")

    attpath_dict = {}

    for group_id, attribute_names in OE_SOLAR_ATTRIBUTES.items():

        if group_id == "PPC" and site != "Comanche":
            continue
        if group_id == "Modules" and "Inverters" not in af_struct.keys():
            continue

        # validate asset group, get asset_names if appl.
        if group_id not in af_struct.keys() and group_id != "Modules":
            continue

        asset_names = []  # init
        if group_id not in ("Meter", "Modules"):
            asset_names = af_struct[group_id]
            if isinstance(asset_names, dict):
                asset_names = [*asset_names]

        asset_group = group_id if group_id not in ("Modules") else None

        attpath_dict[group_id] = _build_solar_attpath_list(
            site=site,
            group=asset_group,
            asset_names=asset_names,
            attribute_names=attribute_names,
            validated=validated,
        )

    cmb_atts_by_inv = oemeta.data["combiner_attributes"].get(site, None)
    if cmb_atts_by_inv is not None:
        cmb_attpaths = []
        for inv, cmb_atts in cmb_atts_by_inv.items():
            cmb_attpaths.extend(
                _build_solar_attpath_list(
                    site=site,
                    group="Inverters",
                    asset_names=[inv],
                    attribute_names=cmb_atts,
                    validated=validated,
                )
            )
        attpath_dict["Combiners"] = cmb_attpaths

    return attpath_dict
