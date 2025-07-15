import itertools
import pandas as pd
from pathlib import Path

from ..reporting.query_attributes import attribute_path
from ..utils import oemeta, oepaths
from ..utils.assets import PISite
from ..utils.pi import PIDataset


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
        "asset": {
            "Inverters": [
                "OE.ActivePower",
            ],
        },  # TODO
    },
    "wind": {
        "site": ["OE.MeterMW", "OE.CountOnline", "OE.AvgWindSpeed"],
        "asset": {
            "WTG": [
                "Ambient.Temperature",
                "Ambient.WindSpeed",
                "Cogent.IsIced",
                "Grid.Power",
                "Grid.PossiblePower",
                "Grid.InternalDerateState",
                "OE.AlarmCode",
            ],
        },
    },
}

GADS_METADATA = pd.DataFrame.from_dict(
    data={
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
    },
    orient="index",
    columns=["capacity", "n_turbines", "wtg_rating", "min_wtg_offline"],
)


def eligible_gads_sites(year: int) -> list:
    """Returns list of sites at or above the GADS capacity limit for selected year."""
    eligible_sites = []
    for fleet in ["solar", "wind"]:
        min_capacity = GADS_CAPACITY_LIMITS[fleet].get(str(year))
        if not min_capacity:
            raise KeyError(f"The capacity limit for {year=} is not defined.")
        sitelist = GADS_METADATA.loc[GADS_METADATA["capacity"].ge(min_capacity)].index.to_list()
        eligible_sites.extend(list(sorted(sitelist)))
    return eligible_sites


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


class GADSSite(PISite):
    """Class for sites subject to GADS reporting. Inherits from PISite.

    Attributes (PISite)
    ----------
    name : str
    fleet : str
    fleet_key : str

    Properties (PISite)
    ----------
    af_dict : dict
    asset_heirarchy : dict
    asset_groups : list
    asset_names_by_group : dict
    timezone : str
    coordinates : tuple (lat, lon)
    altitude : float
    af_path : str

    """

    def __init__(self, name: str):
        super().__init__(name)  # raises ValueError if site does not exist in AF structure
        if self.fleet == "gas":
            raise ValueError("Gas sites are currently not supported.")
        # self.site = site
        self.gads_attributes = GADS_ATTRIBUTES[self.fleet]
        self.asset_group = list(self.gads_attributes["asset"].keys())[0]  # temp

    def __str__(self):
        fmt_row = lambda item, val: f"{item.rjust(26)}| {val}"
        n_assets = len(self.asset_names_by_group[self.asset_group])
        asset_abbr = self.asset_group[:3].lower()
        return "\n".join(
            [
                "-> Instance of oetoolbox.reporting.gads.GADSSite",
                fmt_row("name", self.name),
                fmt_row("fleet", self.fleet),
                fmt_row("ac_capacity", f"{self.ac_capacity} MW"),
                fmt_row("", ""),
                fmt_row("n_attributes (site-level)", len(self.gads_attributes["site"])),
                fmt_row("n_attributes (asset-level)", len(self.gads_attributes["asset"])),
                fmt_row("", f"(n_{asset_abbr} = {n_assets})"),
                fmt_row("n_attributes (total)", len(self.query_attributes)),
            ]
        )

    @property
    def ac_capacity(self) -> int:
        if self.name in GADS_METADATA.index:
            return GADS_METADATA.at[self.name, "capacity"]
        return oemeta.data["SystemSize"]["MWAC"].get(self.name)  # returns None if not in json

    @property
    def query_attributes(self) -> list:

        # get site level attribute paths
        att_paths = [attribute_path(self.name, [], att) for att in fleet_atts["site"]]

        # add asset-level attribute paths
        asset_attributes = fleet_atts[self.asset_group]
        asset_names = self.site.subassets_by_asset(asset_group=self.asset_group)
        if isinstance(asset_names, dict):
            asset_names = [*asset_names]
        att_paths.extend(
            [
                attribute_path(
                    site=self.name, asset_heirarchy=[self.asset_group, asset], attribute=att
                )
                for asset, att in itertools.product(asset_names, asset_attributes)
            ]
        )
        return att_paths

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

    def query_pi_data(self, start_date: str, end_date: str, q: bool = True):
        """Queries defined GADs attributes for specified date range.

        Parameters
        ----------
        start_date : str
            Start date for query range, format -> "%Y-%m-%d"
        end_date : str
            End date for query range, format -> "%Y-%m-%d"
        q : bool
            Quiet parameter; when False, enables status printouts. Defaults to True.

        -> note: for date range, inclusive="left" (to end_date, not through)
        """
        dataset = PIDataset.from_attribute_paths(
            site_name=self.name,
            attribute_paths=self.query_attributes,
            start_date=start_date,
            end_date=end_date,
            freq="1h",
            keep_tzinfo=False,
            q=q,
        )
        return dataset.data

    # def _formatted_query_output(self, df_out):

    def _get_query_filename(self, start_date, end_date):
        date_0, date_1 = map(lambda t: pd.Timestamp(t).strftime("%Y%m%d"), [start_date, end_date])
        return f"{self.name}_GADS_PIdata_{date_0}_to_{date_1}.csv"
