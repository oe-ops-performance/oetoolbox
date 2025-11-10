import itertools
import pandas as pd
import numpy as np
from pathlib import Path

from ..reporting.query_attributes import attribute_path
from ..utils import oemeta, oepaths
from ..utils.assets import PISite
from ..utils.helpers import quiet_print_function
from ..utils.pi import PIDataset
from ..utils.solar import SolarDataset


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
        "site": [
            "OE.MeterMW",
            "OE.InvertersOnline",
        ],  # "OE.MeterMW", "OE.AvgPOA", "OE.AvgAmbientTemp"] -> from files
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
    columns=["capacity", "n_assets", "asset_rating", "min_offline"],
)


def eligible_gads_sites(year: int, by_fleet=True) -> list:
    """Returns list of sites at or above the GADS capacity limit for selected year."""
    eligible_sites = {}
    for fleet in ["solar", "wind"]:
        min_capacity = GADS_CAPACITY_LIMITS[fleet].get(str(year))
        if not min_capacity:
            raise KeyError(f"The capacity limit for {year=} is not defined.")
        sitelist = GADS_METADATA.loc[GADS_METADATA["capacity"].ge(min_capacity)].index.to_list()
        eligible_sites[fleet] = list(
            sorted([s for s in sitelist if s in oemeta.PI_SITES_BY_FLEET[fleet]])
        )
    if by_fleet:
        return eligible_sites
    return list(itertools.chain.from_iterable(eligible_sites.values()))


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
    meta_index = ["capacity", "n_assets", "asset_rating", "min_offline"]
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
        self.gads_attributes = GADS_ATTRIBUTES[self.fleet]
        try:
            self.metadata = GADS_METADATA.loc[self.name].to_dict()
        except KeyError:
            self.metadata = {}
        self.site_level_attributes = self.gads_attributes["site"]
        self.asset_group = list(self.gads_attributes["asset"].keys())[0]  # temp
        self.asset_names = self.asset_names_by_group[self.asset_group]
        self.asset_level_attributes = self.gads_attributes["asset"][self.asset_group]

    def __str__(self):
        fmt_row = lambda item, val: "  ".join([item.rjust(29), "-->", str(val)])
        asset_abbr = self.asset_group[:3].lower()
        n_site_atts = len(self.site_level_attributes)
        n_atts_per_asset = len(self.asset_level_attributes)
        n_asset_atts = n_atts_per_asset * len(self.asset_names)
        n_expected = n_asset_atts + n_site_atts
        n_valid = len(self.query_attributes)
        row_list = [
            "oetoolbox.reporting.gads.GADSSite",
            " ",
            fmt_row("fleet", self.fleet.capitalize()),
            fmt_row("site_name", self.name),
            fmt_row("ac_capacity", f"{self.ac_capacity} MW"),
            fmt_row("n_assets", len(self.asset_names)),
        ]
        if self.metadata:
            for key in ["asset_rating", "min_offline"]:
                val = self.metadata[key]
                if "offline" in key:
                    val = int(val)
                row_list.append(fmt_row(key, val))
        row_list.extend(
            [
                " ",
                fmt_row("n_attributes / asset", n_atts_per_asset),
                fmt_row("n_attributes (asset-level)", n_asset_atts),
                fmt_row("n_attributes (site-level)", n_site_atts),
                fmt_row("n_total (expected)", n_expected),
                fmt_row("n_total (valid)", n_valid),
                fmt_row("n_missing", n_expected - n_valid),
            ]
        )
        return "\n".join(row_list)

    @property
    def ac_capacity(self) -> int:
        if self.name in GADS_METADATA.index:
            return GADS_METADATA.at[self.name, "capacity"]
        return oemeta.data["SystemSize"]["MWAC"].get(self.name)  # returns None if not in json

    @property
    def site_level_attribute_paths(self) -> list:
        return self.get_attribute_paths(attributes=self.site_level_attributes)

    @property
    def asset_level_attribute_paths(self) -> list:
        return self.get_attribute_paths(
            asset_group=self.asset_group,
            asset_names=self.asset_names,
            attributes=self.asset_level_attributes,
        )

    @property
    def query_attributes(self) -> list:
        return self.site_level_attributes + self.asset_level_attributes

    def gads_folder(self, year: int) -> Path:
        folder = Path(GADS_DIR, self.fleet.capitalize(), str(year))
        if not folder.exists():
            folder.mkdir(parents=True)
        return folder

    def monthly_gads_path(self, year: int, month: int):
        ym_path = self.gads_folder(year).joinpath(str(month))
        if not ym_path.exists():
            ym_path.mkdir()
        return ym_path

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
        df_list = []
        kwargs = dict(
            site_name=self.name, start_date=start_date, end_date=end_date, keep_tzinfo=False, q=q
        )
        for attpaths in [self.site_level_attribute_paths, self.asset_level_attribute_paths]:
            q_kwargs = {**kwargs, "freq": "1h"}
            if len(attpaths) > 400:
                idx1 = int(round((len(attpaths) / 3), 0))
                idx2 = idx1 * 2
                paths_1 = attpaths[:idx1]
                paths_2 = attpaths[idx1:idx2]
                paths_3 = attpaths[idx2:]
                attpaths_list = [paths_1, paths_2, paths_3]
                q_kwargs.update({"n_segment": 5})
            else:
                attpaths_list = [attpaths]
            # note: query non-inverter attributes at minute-level (grab inv data from files) and then resample to hourly with average aggregation & use .floor to round down
            for atts in attpaths_list:
                dataset = PIDataset.from_attribute_paths(**q_kwargs, attribute_paths=atts)
                df_list.append(dataset.data)

        df = pd.concat(df_list, axis=1)
        fmt_col = lambda c: c if "_" not in c else "_".join(c.split("_")[:-1])
        df.columns = df.columns.map(fmt_col)  # removes asset group
        if self.fleet == "solar":
            # add pvlib data from file
            dfp = self._get_pvlib_data(start_date, end_date)
            keepcols = list(filter(lambda c: "possible" in c, dfp.columns))
            dfp = dfp.resample("h").mean()
            df = df.join(dfp)
            if "OE.AvgPOA" in df.columns:
                df = df.drop(columns=["OE.AvgPOA"])
            df = df.rename(columns={"poa": "OE.AvgPOA"})
        return self._formatted_query_output(df)

    def query_monthly_pi_data(self, year, month, q=True):
        start_ = pd.Timestamp(year, month, 1)
        end_ = start_ + pd.DateOffset(months=1)
        start_date, end_date = map(lambda t: t.strftime("%Y-%m-%d"), [start_, end_])
        return self.query_pi_data(start_date=start_date, end_date=end_date, q=q)

    def get_monthly_data(
        self, year: int, month: int, save: bool = True, q: bool = True
    ) -> dict[str, pd.DataFrame]:
        """Queries data from PI, then generates events and saves both files to GADS directory."""
        qprint = quiet_print_function(q=q)
        df_data = self.query_monthly_pi_data(year, month, q=q)
        df_events = self.generate_gads_events(df_data)
        if save:
            data_fp = self._get_savepath(year, month, file_type="data")
            df_data.to_csv(data_fp)
            qprint("  >> saved: ..\\" + "\\".join(data_fp.parts[-7:]))

            events_fp = self._get_savepath(year, month, file_type="events")
            df_events.to_csv(events_fp, index=False)
            qprint("  >> saved: ..\\" + "\\".join(events_fp.parts[-7:]))
        return dict(data=df_data, events=df_events)

    def _get_pvlib_data(self, start_date, end_date):
        """Loads pvlib data from file (should always exist)"""
        pvlib_dataset = SolarDataset.from_existing_data_files(
            site_name=self.name,
            asset_group="pvlib",
            start_date=start_date,
            end_date=end_date,
        )
        df_pvlib = pvlib_dataset.data.copy()
        inv_names = self.asset_names_by_group["Inverters"]
        col_mapping = {inv.lower(): inv for inv in inv_names}
        new_pvl_cols = []
        for col in df_pvlib.columns:
            if not col.endswith("possible_power"):
                new_pvl_cols.append(col)
                continue
            lower_id = col.split("_")[0]
            new_col = col_mapping[lower_id] + "_possible_power"
            new_pvl_cols.append(new_col)

        df_pvlib.columns = new_pvl_cols
        return df_pvlib

    def _formatted_query_output(self, df_out):
        """processes pi data file from query_pi_data function to add/normalize columns"""
        # rename meter column
        df = df_out.rename(columns={"OE.MeterMW": "Site_Meter_MW"}).copy()

        # get column names for actual & possible generation
        filtered_cols = lambda x: [c for c in df.columns if x in c]
        col_keys = ["Grid.Power", "Grid.PossiblePower"]
        if self.fleet == "wind":
            col_keys = ["Grid.Power", "Grid.PossiblePower"]
        else:
            col_keys = ["OE.ActivePower", "possible_power"]
        actual_cols, possible_cols = map(filtered_cols, col_keys)

        # add columns for site-level generation & estimated loss
        df["Site_Actual_MW"] = df[actual_cols].sum(axis=1).div(1e3)
        df["Site_Possible_MW"] = df[possible_cols].sum(axis=1).div(1e3)
        df["lost_MW"] = df["Site_Possible_MW"].sub(df["Site_Actual_MW"])
        df.loc[df["lost_MW"].lt(0), "lost_MW"] = 0

        # create n_offline column (inverse)
        n_assets = self.metadata["n_assets"]
        online_col = "OE.CountOnline" if self.fleet == "wind" else "OE.InvertersOnline"
        df["n_offline"] = np.floor(n_assets - df[online_col])
        if self.fleet == "solar":
            c1 = df["OE.AvgPOA"] < 250  # increased from 200
            c2 = df["Site_Possible_MW"].le(0)
            df.loc[(c1 | c2), "n_offline"] = 0

        return df

    def _process_gads_offline_events(self, df) -> list[pd.DataFrame]:
        """Uses formatted df from query function"""
        offline_threshold = self.metadata["min_offline"]
        conditions_ = df["n_offline"].ge(offline_threshold)
        breaks_ = (df["n_offline"].lt(offline_threshold)).cumsum()
        groupby_obj = df[conditions_].groupby(breaks_)
        grouped_df_list = [dfg for _, dfg in groupby_obj]
        return grouped_df_list

    def generate_gads_events(self, df) -> pd.DataFrame:
        """Uses formatted df from query function"""
        grouped_df_list = self._process_gads_offline_events(df)
        dfcols = ["event_start", "event_end", "event_hours", "n_offline", "lost_MWh"]
        data_list = []
        for dfg in grouped_df_list:
            event_start = dfg.index.min()
            event_end = dfg.index.max() + pd.Timedelta(hours=1)
            event_hours = (event_end - event_start).seconds / 3600
            avg_offline = dfg["n_offline"].mean()
            lost_energy = dfg["lost_MW"].sum()
            data_list.append([event_start, event_end, event_hours, avg_offline, lost_energy])
        return pd.DataFrame(data_list, columns=dfcols)

    def _get_query_filename(self, start_date, end_date):
        date_0, date_1 = map(lambda t: pd.Timestamp(t).strftime("%Y%m%d"), [start_date, end_date])
        return f"{self.name}_GADS_PIdata_{date_0}_to_{date_1}.csv"

    def _get_data_filename(self, year: int, month: int, file_type: str):
        if file_type not in ["data", "events", "plot"]:
            raise ValueError(f"Invalid {file_type = }.")
        start_date = pd.Timestamp(year, month, 1)
        end_date = start_date + pd.DateOffset(months=1)
        start_ym, end_ym = map(lambda t: t.strftime("%Y%m"), [start_date, end_date])
        ext = ".csv" if file_type != "plot" else ".html"
        return f"{self.name}_GADS-{file_type.upper()}_{start_ym}_to_{end_ym}" + ext

    def _get_savepath(self, year, month, file_type):
        """Generates new (unique) filepath for saving data. Creates folder is not exist."""
        save_dir = self.monthly_gads_path(year, month)
        folder = Path(save_dir, file_type)
        if not folder.exists():
            folder.mkdir()
        filename = self._get_data_filename(year, month, file_type)
        return oepaths.validated_savepath(Path(folder, filename))

    def _get_data_filepath(self, year: int, month: int, file_type: str):
        """Note: returns None if no files exist. Used for loading data."""
        if file_type not in ["data", "events"]:
            raise ValueError("Loading html files is currently not supported.")
        gads_path = self.monthly_gads_path(year, month)
        data_folder = Path(gads_path, file_type)
        if not data_folder.exists():
            return
        filestem = self._get_data_filename(year, month, file_type).split(".csv")[0]
        data_fpaths = list(data_folder.glob(f"{filestem}*csv"))
        if not data_fpaths:
            return
        return oepaths.latest_file(data_fpaths)

    def load_data_from_file(self, year: int, month: int, file_type: str, q: bool = True):
        """Loads data or events file from GADS folder for specified year/month"""
        if file_type not in ["data", "events"]:
            raise ValueError("Loading html files is outside the scope of this function.")

        qprint = quiet_print_function(q=q)
        data_fpath = self._get_data_filepath(year, month, file_type)
        if not data_fpath:
            qprint("No file found.")
            return

        if file_type == "data":
            df = pd.read_csv(data_fpath, index_col=0, parse_dates=True)
        else:
            df = pd.read_csv(data_fpath)
            df["event_start"] = pd.to_datetime(df["event_start"])
            df["event_end"] = pd.to_datetime(df["event_end"])
            df["event_hours"] = df["event_hours"].astype(int)
        fpath_str = '"..\\' + "\\".join(data_fpath.parts[-8:]) + '"'
        qprint(f"Loaded {file_type} file ->   {fpath_str}")
        return df
