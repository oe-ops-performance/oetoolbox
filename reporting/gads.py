import itertools
import pandas as pd
from pathlib import Path

from ..utils import oemeta, oepaths
from ..utils.assets import PISite


GADS_DIR = Path(oepaths.operations, "Compliance", "GADS")
RENEWABLE_AF_PATH = "\\\\CORP-PISQLAF\\Onward Energy\\Renewable Fleet"
RENEWABLE_SITES = {
    "solar": list(oemeta.data["AF_Solar_V3"].keys()),
    "wind": list(oemeta.data["AF_Wind_V3"].keys()),
}
GADS_CAPACITY_LIMITS = {
    "solar": {"2024": 100, "2025": 20},
    "wind": {"2024": 100, "2025": 75},
}
GADS_ATTRIBUTES = {
    "solar": {
        "site": ["OE.MeterMW", "OE.InvertersOnline", "OE.AvgPOA", "OE.AvgAmbientTemp"],
        "inv": [
            "OE.ActivePower",
        ],
    },  # placeholder
    "wind": {
        "site": ["OE.MeterMW", "OE.CountOnline", "OE.AvgWindSpeed"],
        "wtg": [
            "Ambient.Temperature",
            "Ambient.WindSpeed",
            "Cogent.IsIced",
            "Grid.Power",
            "Grid.PossiblePower",
            "Grid.InternalDerateState",
            "OE.AlarmCode",
        ],
    },
}


def gads_metadata():
    """
    original method of retrieval from PI table
        from ..dataquery.pitools import load_PI_table
        df_meta = load_PI_table(DB, 'Wind Site Metadata')
        keepcols = {
            'Site': 'site', 'Capacity': 'capacity',
            '# Of Turbines': 'n_turbines', 'WTG Rating': 'wtg_rating',
        }
        df_meta = df_meta[[*keepcols]].rename(columns=keepcols)
        df_meta = df_meta.set_index('site').rename_axis(None)
        df_meta['min_wtg_offline'] = (20 / df_meta['wtg_rating']).apply(lambda x: np.ceil(x)).astype(int)
    """
    meta_index = ["capacity", "n_turbines", "wtg_rating", "min_wtg_offline"]
    meta_data = {
        "Maplewood 1": [222, 77, 3.15, 7],
        "GA4": [200, 68, 3.0, 7],
        "Comanche": [120, 75, 1.667, 13],
        "Catalina II": [19.76, 23, 0.7826, 23],
        "Three Peaks": [150.0, 36, 2.5, 5],
        "Bingham": [184.8, 56, 3.3, 7],
        "Hancock": [51.0, 17, 3.0, 7],
        "Oakfield": [147.6, 48, 3.1, 7],
        "Palouse": [105.3, 58, 1.8, 12],
        "Route 66": [150.0, 75, 2.0, 10],
        "South Plains II": [300.3, 91, 3.3, 7],
        "Sunflower": [104.0, 52, 2.0, 10],
        "High Sheldon": [112.5, 75, 1.5, 14],
        "Turkey Track": [169.5, 113, 1.5, 14],
        "Willow Creek": [72.0, 48, 1.5, 14],
    }
    return pd.DataFrame(meta_data, index=meta_index).T


class GADSSite:

    def __init__(self, site_name: str):
        try:
            site = PISite(name=site_name)
        except ValueError as e:
            raise e
        if site.fleet == "gas":
            raise ValueError("Gas sites are currently not supported.")
        self.name = site_name
        self.site = site
        self.fleet = site.fleet

    @property
    def ac_capacity(self) -> int:
        df_meta = gads_metadata()
        if self.name in df_meta.index:
            return gads_metadata().at[self.name, "capacity"]
        return oemeta.data["SystemSize"]["MWAC"].get(self.name)  # returns None if note in json

    @property
    def query_attributes(self) -> list:
        fleet_att_dict = GADS_ATTRIBUTES[self.fleet]
        if self.fleet == "solar":
            pass
        elif self.fleet == "wind":
            wtg_ids = self.site.asset_names_by_group["WTG"]
            wtg_atts = [
                "Ambient.Temperature",
                "Ambient.WindSpeed",
                "Cogent.IsIced",
                "Grid.Power",
                "Grid.PossiblePower",
                "Grid.InternalDerateState",
                "OE.AlarmCode",
            ]
            wtg_attribute_paths = [
                "\\".join([self.site.af_path, "WTG", f"{wtg}|{att}"])
                for wtg, att in itertools.product(wtg_ids, wtg_atts)
            ]
            site_atts = ["OE.MeterMW", "OE.InvertersOnline", "OE.AvgPOA", "OE.AvgAmbientTemp"]
            site_attribute_paths = ["|".join([self.site.af_path, att]) for att in site_atts]
            return wtg_attribute_paths + site_attribute_paths

    def quarterly_gads_path(self, year: int, quarter: int):
        fleet_folder = self.fleet.capitalize()
        savepath = Path(GADS_DIR, fleet_folder, str(year), f"Q{quarter}")
        if not savepath.exists():
            savepath.mkdir(parents=True)
        return savepath

    def is_eligible(self, year: int) -> bool:
        if self.ac_capacity is None:
            return False
        gads_minimum = GADS_CAPACITY_LIMITS[self.fleet].get(str(year))
        return self.ac_capacity >= gads_minimum  # returns None if minimum not defined for year
