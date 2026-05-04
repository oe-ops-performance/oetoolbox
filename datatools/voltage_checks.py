import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils.pi import PIDataset
from ..dataquery.pitools import pipoint_list, query_pi_recordedvalues

VOLTAGE_BAND = 4  # kV

VOLTAGE_PIPOINT_DICT = {
    "Route 66": [f"RT66.RTAC.Tenska_ERCOT_VOLT_{x}" for x in ("Target", "POI")],
    "South Plains II": [f"SP2.RTAC.Tenaska_ERCOT_VOLT_{x}" for x in ("Target", "POI")],
}

VOLTAGE_PIPOINTS = {
    "Route 66": {
        "target": "RT66.RTAC.Tenska_ERCOT_VOLT_Target",
        "poi": "RT66.RTAC.Tenska_ERCOT_VOLT_POI",
    },
    "South Plains II": {
        "target": "SP2.RTAC.Tenaska_ERCOT_VOLT_Target",
        "poi": "SP2.RTAC.Tenaska_ERCOT_VOLT_POI",
    },
}


def _validate_filename(filepath):
    fpath = Path(filepath)
    reqd_keys = ("voltage", "analysis", ".xlsx")
    if not all(k in fpath.name for k in reqd_keys):
        raise ValueError("Invalid filename.")
    if not all(
        s in [*pd.read_excel(fpath, engine="calamine", nrows=0, sheet_name=None)]
        for s in ["Data", "Events"]
    ):
        raise KeyError("One or more required sheets missing in file.")
    site, ym_str = fpath.name.split("_")[:2]
    year, month = int(ym_str[:4]), int(ym_str[4:])
    return site, year, month


class VoltageAnalysis:
    """Class to perform voltage analysis for specified sites and time periods."""

    def __init__(self, site, year, month, threshold=VOLTAGE_BAND):
        if site not in VOLTAGE_PIPOINTS.keys():
            raise ValueError(f"Invalid site specified: {site}")
        self.site = site
        self.year = year
        self.month = month
        self.threshold = threshold

        self._assign_dates()
        self._assign_pipoints()

        self.df_raw = None
        self.df = None
        self.event_list = None
        self.df_events = None
        self.excursion_plot = None

    def _assign_dates(self):
        self.start = pd.Timestamp(year=self.year, month=self.month, day=1)
        self.end = self.start + pd.DateOffset(months=1)
        self.start_date, self.end_date = map(
            lambda t: t.strftime("%Y-%m-%d"), [self.start, self.end]
        )

    def _assign_pipoints(self):
        pipoint_dict = VOLTAGE_PIPOINTS.get(self.site)
        self.target_pipt = pipoint_dict["target"]
        self.poi_pipt = pipoint_dict["poi"]
        self.rename_cols = {
            self.target_pipt: "volt_target",
            self.poi_pipt: "volt_poi",
        }

    def _load_analysis_data_from_file(self, filepath):
        # assigns self.df
        self.df = pd.read_excel(filepath, engine="calamine", sheet_name="Data", index_col=0)
        # self.df_events = pd.read_excel(filepath, engine="calamine", sheet_name="Events")

    @classmethod
    def from_existing_file(cls, filepath):
        site, year, month = _validate_filename(filepath)
        instance = cls(site, year, month)
        instance._load_analysis_data_from_file(filepath)
        instance.identify_events()
        instance.generate_excursion_plot()
        return instance

    def query_data(self, q=True):
        if self.df is not None:
            print("Found existing data; delete/archive file to re-query and save new files.")
            return
        dataset = PIDataset.from_pipoints(
            site_name=self.site,
            pipoint_names=[self.target_pipt, self.poi_pipt],
            start_date=self.start_date,
            end_date=self.end_date,
            method="recorded_values",
            keep_tzinfo=True,
            q=q,
        )
        df = dataset.data.copy()
        df["PIPoint"] = df["PIPoint"].astype(str)

        df_poi = df.loc[df["PIPoint"].eq(self.poi_pipt)].copy()
        df_poi = df_poi[~df_poi.index.duplicated(keep="first")].copy()

        df_target = df.loc[df["PIPoint"].eq(self.target_pipt)].copy()
        df_target = df_target[~df_target.index.duplicated(keep="first")].copy()
        df_target = df_target.resample("1s").ffill()

        dfp = df_poi[["Value"]].rename(columns={"Value": self.poi_pipt}).copy()
        dft = df_target[["Value"]].rename(columns={"Value": self.target_pipt}).copy()

        df_raw = dfp.join(dft, how="left")
        df_raw = df_raw[~df_raw.index.duplicated()].copy()
        self.df_raw = df_raw
        self.df = self._format_data(df_raw)

    def _format_data(self, df_raw):
        df = df_raw.rename(columns=self.rename_cols).copy()
        for col in ["volt_poi", "volt_target"]:
            if df[col].max() > 1e3:
                df[col] = df[col].div(10)
        df["delta"] = df["volt_poi"].sub(df["volt_target"])
        return df

    def identify_events(self):
        if self.df is None:
            raise ValueError("Data must be queried before identifying events.")
        # resample recorded values and forward fill to ensure all timestamps are present for event detection
        df = self.df.resample("1s").ffill().copy()
        condition = df["delta"].abs().gt(self.threshold)
        breaks = (~condition).cumsum()
        self.event_list = [group for _, group in df[condition].groupby(breaks)]
        self.df_events = self._create_excursion_event_table(self.event_list)

    def _create_excursion_event_table(self, event_df_list) -> pd.DataFrame:
        columns = [
            "Start Time",
            "Start Setpoint (kV)",
            "End Time",
            "End Setpoint (kV)",
            "Duration (mins)",
            "Max Delta (kV)",
            "Min Delta (kV)",
        ]
        data = []
        for df in event_df_list:
            start, end = df.index[0], df.index[-1]
            start_sp = df.at[start, "volt_target"]
            end_sp = df.at[end, "volt_target"]
            duration = (end - start).total_seconds() / 60
            max_delta = df["delta"].max()
            min_delta = df["delta"].min()
            data.append([start, start_sp, end, end_sp, duration, max_delta, min_delta])
        return pd.DataFrame(data, columns=columns)

    def generate_excursion_plot(self) -> go.Figure:
        df = self.df.copy()
        df["lower_limit"] = -self.threshold
        df["upper_limit"] = self.threshold

        fig = go.Figure()
        for col in ["volt_target", "volt_poi"]:
            fig.add_trace(go.Scattergl(x=df.index, y=df[col], mode="lines", name=col))

        title_text = f"{self.site} - Voltage Profile - {df.index[0].strftime('%b-%Y')}"
        fig.update_layout(
            height=500,
            width=1050,
            title=dict(text=title_text, pad_b=15, yanchor="bottom", y=1, yref="paper"),
            legend_x=1.05,
            margin=dict(t=50, b=20, l=70, r=20),
            yaxis_title=dict(text="Voltage (kV)", standoff=10),
            yaxis2=dict(
                title=dict(text="Voltage Delta (kV)", standoff=10),
                overlaying="y",
                side="right",
                showgrid=False,
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["delta"],
                mode="lines",
                line=dict(color="black", width=1),
                yaxis="y2",
                name="delta",
            )
        )
        for limit_col in ["lower_limit", "upper_limit"]:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[limit_col],
                    mode="lines",
                    name=limit_col,
                    line=dict(color="black", width=1, dash="dash"),
                    yaxis="y2",
                )
            )

        for dfg in self.event_list:
            x0, x1 = dfg.index.min(), dfg.index.max()
            timedelta = x1 - x0
            if len(self.event_list) > 50:
                if timedelta < pd.Timedelta("15min"):
                    continue
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="lightsalmon",
                opacity=0.4,
                layer="below",
                line_width=0,
            )

        self.excursion_plot = fig

    def run_analysis(self, q=True):
        self.query_data(q=q)
        self.identify_events()
        self.generate_excursion_plot()
