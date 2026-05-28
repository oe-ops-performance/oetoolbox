import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils.pi import PIDataset

ERCOT_VOLTAGE_BAND = 4  # kV

VOLTAGE_TARGET_PIPOINTS = {  # ERCOT sites
    "Route 66": "RT66.RTAC.Tenska_ERCOT_VOLT_Target",
    "South Plains II": "SP2.RTAC.Tenaska_ERCOT_VOLT_Target",
    "Maplewood 1": "MPLWD.MWSP_QRA.PO.VSP_kV",  # same both MW1/2
    "Maplewood 2": "MPLWD.MWSP_QRA.PO.VSP_kV",  # same both MW1/2
}

VOLTAGE_POI_PIPOINTS = {
    "Adams East": "ADMES.AES1_PPC1_SL.AES1_PPC1.Plant.Der.VPOI_kV",
    "AZ1": "ARZO1.Sub_1.PPC_1_HSM.AGC.Average kV L-L",
    "Camelot": "CMLOT.CAM_RIG_VLL_AVG",
    "Columbia II": "COLM2.C2_RIG_VLL_AVG",
    "FL4": "TALLA.Sub_1.PPC_1_HSM.AGC.Average kV L-L",
    # "GA3": "",  # TODO -- add reference to new PI AFAttribute -> OE.Avg_LL_Kv (exists now)
    "Grand View West": "GRDVW.meters.DNP.owner.VOLTSAVG_L_L",
    "Hancock": "HAN.RTAC.735_T2_HIGH_VOLT_3PH_AVG",
    # "Imperial Valley": "",  # TODO -- add reference to new PI AFAttribute -> OE.Avg_LL_Kv
    "Kansas": "KNSAS.KNS1_S001.PMCAISOPri.Sts.VLLAvg_kV",
    "Kent South": "KENTS.KSS1_S001_SL.KSS1_S001.PMCAISOPri.Sts.VLLAvg_kV",
    "Maplewood 1": "MPLWD.MWSP_MTR001.PO.VllAvg_kV",  # substation, not poi
    "Maplewood 2": "MPLWD.MWSP_MTR002.PO.VllAvg_kV",  # substation, not poi
    "Maricopa West": "MARWE.CAISO Main Meter Bus Voltage-Avg",
    "MS3": "SUMS3.MS3.PLC1.0S_REG01.V",
    # "Mulberry": "",  # TODO -- add reference to new PI AFAttribute -> OE.Avg_LL_Kv
    "Old River One": "OLRVO.ORS1_S001.PMCAISOPri.Sts.VLLAvg_kV",
    "Route 66": "RT66.RTAC.Tenska_ERCOT_VOLT_POI",
    # "Selmer": "",  # TODO -- add reference to new PI AFAttribute -> OE.Avg_LL_Kv
    "South Plains II": "SP2.RTAC.Tenaska_ERCOT_VOLT_POI",
    "West Antelope": "TCAAW.Antelope West DNP Client.DNP Client.CAISO Meter kVLL-Avg",
}

STATIC_VOLTAGES = {  # [target, high, low]
    "Adams East": [71.0, 71.71, 70.29],
    "AZ1": [117.0, 120.8, 109.3],
    "Camelot": [66.05, 67.6, 64.5],  # need to confirm (eyeballed)
    "Columbia II": [66.0, 67.6, 64.4],  # need to confirm (eyeballed)
    "FL4": [230.0, 241.5, 218.5],
    "Grand View West": [143.0, 144.9, 138.0],
    "Hancock": [120.0, 122.0, 118.0],
    "Imperial Valley": [91.95, 96.6, 87.3],  # need to confirm (eyeballed)
    "Kansas": [np.nan, np.nan, np.nan],  # TODO - time-of-day logic
    "Kent South": [72.0, 72.72, 71.28],
    "Maricopa West": [71.0, 71.71, 70.29],
    "MS3": [69.0, 71.0, 69.0],
    "Old River One": [71.0, 71.71, 70.29],
    "West Antelope": [66.0, 69.3, 66.0],
}

VOLTAGE_SITES = [
    s for s in VOLTAGE_POI_PIPOINTS if not any(pd.isna(x) for x in STATIC_VOLTAGES.get(s, []))
]


def _get_data_folder(analysis_dir, year, month):
    data_folder = Path(analysis_dir, f"{year}-{month:02d}")
    if allow_create and not data_folder.exists():
        abbr_fpath = lambda fp: "..\\" + "\\".join(fp.parts[-2:])
        print(f"Creating folder: {abbr_fpath(data_folder)}")
        data_folder.mkdir()
    return data_folder


class VoltageMetadata:
    def __init__(self, site: str):

        self.poi_pipt = VOLTAGE_POI_PIPOINTS.get(site, None)
        if self.poi_pipt is None:
            raise KeyError(f"No voltage POI pipoint(s) defined for {site=}.")

        self.static_target = None  # init
        self.upper_limit = None  # init
        self.lower_limit = None  # init
        self.upper_band = ERCOT_VOLTAGE_BAND
        self.lower_band = ERCOT_VOLTAGE_BAND

        self.target_pipt = VOLTAGE_TARGET_PIPOINTS.get(site, None)
        if self.target_pipt is None:
            voltages_ = STATIC_VOLTAGES.get(site, None)
            if voltages_ is None:
                raise KeyError(f"Incomplete voltage metadata found for {site=}.")
            target, high, low = voltages_
            self.static_target = target
            self.upper_limit = high
            self.lower_limit = low
            self.upper_band = high - target
            self.lower_band = target - low

        self.rename_cols = {
            pt: f"volt_{lbl}"
            for pt, lbl in zip([self.poi_pipt, self.target_pipt], ["poi", "target"])
            if pt is not None
        }
        self.pipoint_names = [*self.rename_cols]


def _validate_filename(filepath):
    fpath = Path(filepath)
    reqd_keys = ("voltage", "data", ".parquet")
    if not all(k in fpath.name for k in reqd_keys):
        raise ValueError("Invalid filename.")
    site, ym_str = fpath.name.split("_")[:2]
    year, month = int(ym_str[:4]), int(ym_str[4:])
    return site, year, month


class VoltageAnalysis:
    """Class to perform voltage analysis for specified sites and time periods."""

    def __init__(self, site: str, year: int, month: int):
        self.meta = VoltageMetadata(site=site)
        self.static = self.meta.static_target is not None

        self.site = site
        self.year = year
        self.month = month
        self._assign_dates()

        self.df_raw = None
        self.df = None
        self.event_list = None
        self.df_events = None
        self.excursion_plot = None

    def _assign_dates(self) -> None:
        self.start = pd.Timestamp(year=self.year, month=self.month, day=1)
        self.end = self.start + pd.DateOffset(months=1)
        self.start_date, self.end_date = map(
            lambda t: t.strftime("%Y-%m-%d"), [self.start, self.end]
        )

    def _load_analysis_data_from_file(self, filepath) -> None:
        self.df_query = pd.read_parquet(filepath)
        self.df_raw = self._pivot_recvals_to_wide_format(df=self.df_query)
        self._format_raw_data()  # assigns self.df

    @classmethod
    def from_existing_file(cls, filepath):
        site, year, month = _validate_filename(filepath)
        instance = cls(site, year, month)
        instance._load_analysis_data_from_file(filepath)  # assigns self.df_raw
        instance.identify_events()
        instance.generate_excursion_plot()
        return instance

    @classmethod
    def from_monthly_file(cls, site: str, year: int, month: int, analysis_dir: Path):
        dir_ = _get_data_folder(analysis_dir, year, month)
        if not dir_.is_dir():
            raise ValueError("Invalid path.")
        filepattern = f"*{site}*voltage*data*.parquet"
        fpath = oepaths.latest_file(list(data_folder.glob(filepattern)))
        if fpath is None:
            print("No file found.")
            return
        return cls.from_existing_file(filepath=fpath)

    def query_data(self, q=True) -> None:
        if self.df is not None:
            print("Found existing data; delete/archive file to re-query and save new files.")
            return
        dataset = PIDataset.from_pipoints(
            site_name=self.site,
            pipoint_names=self.meta.pipoint_names,
            start_date=self.start_date,
            end_date=self.end_date,
            method="recorded_values",
            data_format="long",
            freq=None,
            keep_tzinfo=True,
            q=q,
        )
        self.df_query = dataset.data.copy()
        self.df_raw = self._pivot_recvals_to_wide_format(df=dataset.data)
        self._format_raw_data()  # assigns self.df

    def _pivot_recvals_to_wide_format(self, df) -> pd.DataFrame:
        # long-format df as input
        df_list = []
        for pipt in df["PIPoint"].unique():
            df_list.append(
                df.loc[df["PIPoint"].eq(pipt), ["Value"]]
                .rename(columns={"Value": pipt})
                .reset_index(drop=False)
                .drop_duplicates(subset=["Timestamp"], keep="first")
                .set_index("Timestamp")
                .resample("1s")
                .ffill()
            )
        return pd.concat(df_list, axis=1)

    def _format_raw_data(self) -> None:
        df = self.df_raw.rename(columns=self.meta.rename_cols).copy()

        if self.site in ("Route 66", "South Plains II", "Camelot", "Columbia II"):
            # known sites where measured poi voltage is off by factor of 10
            for col in df.columns:
                df[col] = df[col].div(10)
        elif self.site in ("Grand View West"):
            # known sites where measured poi voltage is off by factor of 1000
            for col in df.columns:
                df[col] = df[col].div(1e3)

        if self.static is True:
            df["volt_target"] = self.meta.static_target

        if "volt_target" not in df.columns:
            raise ValueError("Could not find 'volt_target' column in data.")

        df["delta"] = df["volt_poi"].sub(df["volt_target"])
        self.df = df

    def excursion_condition(self) -> pd.Series:
        df = self.df.copy()
        if not self.static:
            is_excursion = df["delta"].abs().gt(ERCOT_VOLTAGE_BAND)
        else:
            above_limit = df["delta"].gt(0.0) & df["volt_poi"].gt(self.meta.upper_limit)
            below_limit = df["delta"].lt(0.0) & df["volt_poi"].lt(self.meta.lower_limit)
            is_excursion = above_limit | below_limit
        return is_excursion

    def identify_events(self):
        if self.df is None:
            raise ValueError("Data must be queried before identifying events.")
        # resample recorded values and forward fill to ensure all timestamps are present for event detection
        df = self.df.resample("1s").ffill().copy()
        condition = self.excursion_condition()
        breaks = (~condition).cumsum()
        self.event_list = [group for _, group in df[condition].groupby(breaks)]
        self.df_events = self._create_excursion_event_table(self.event_list)

    def _create_excursion_event_table(self, event_df_list) -> pd.DataFrame:
        columns = [
            "Start Time",
            "Start Setpoint (kV)",
            "Start POI (kV)",
            "End Time",
            "End Setpoint (kV)",
            "End POI (kV)",
            "Duration (mins)",
            "Max Delta (kV)",
            "Min Delta (kV)",
            "Lower Static Limit",
            "Upper Static Limit",
        ]
        data = []
        for df in event_df_list:
            start, end = df.index[0], df.index[-1]
            start_sp = df.at[start, "volt_target"]
            start_poi = df.at[start, "volt_poi"]
            end_sp = df.at[end, "volt_target"]
            end_poi = df.at[end, "volt_poi"]
            duration = (end - start).total_seconds() / 60
            max_delta = df.loc[df["delta"].abs().eq(df["delta"].abs().max()), "delta"].mode()[0]
            min_delta = df.loc[df["delta"].abs().eq(df["delta"].abs().min()), "delta"].mode()[0]
            data.append(
                [
                    start,
                    start_sp,
                    start_poi,
                    end,
                    end_sp,
                    end_poi,
                    duration,
                    max_delta,
                    min_delta,
                    self.meta.lower_limit,
                    self.meta.upper_limit,
                ]
            )
        return pd.DataFrame(data, columns=columns)

    def _build_subplot_figure(
        self,
        df: pd.DataFrame,
        events_to_shade: list,
        title: str,
        height: int = 500,
    ) -> go.Figure:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.65, 0.35],
        )

        for col, name in [("volt_target", "Setpoint"), ("volt_poi", "POI")]:
            fig.add_trace(
                go.Scattergl(x=df.index, y=df[col], mode="lines", name=name),
                row=1,
                col=1,
            )

        for val in (self.meta.lower_limit, self.meta.upper_limit):
            if val is not None:
                fig.add_hline(
                    y=val,
                    line=dict(color="black", width=1, dash="dash"),
                    row=1,
                    col=1,
                )

        fig.add_trace(
            go.Scattergl(
                x=df.index,
                y=df["delta"],
                mode="lines",
                name="Delta",
                line=dict(color="black", width=1),
            ),
            row=2,
            col=1,
        )
        for val in (-self.meta.lower_band, self.meta.upper_band):
            fig.add_hline(
                y=val,
                line=dict(color="black", width=1, dash="dash"),
                row=2,
                col=1,
            )

        shapes = []
        line_width = 0 if "Event" not in title else 0.5
        for dfg in events_to_shade:
            x0 = dfg.index.min()
            x1 = dfg.index.max()
            shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "paper",
                    "x0": x0,
                    "x1": x1,
                    "y0": 0,
                    "y1": 1,
                    "fillcolor": None if (x1 - x0) < pd.Timedelta(minutes=5) else "lightsalmon",
                    "layer": "above",
                    "line": {"width": line_width},
                    "opacity": 0.3,
                }
            )

        annot_kwargs = dict(
            x=0,
            xanchor="right",
            xref="x domain",
            xshift=-35,
            y=0.5,
            yanchor="middle",
            yshift=0,
            showarrow=False,
            textangle=0,
        )
        annotations = [
            dict(text="Voltage (kV)", yref="y domain", **annot_kwargs),
            dict(text="Delta (kV)", yref="y2 domain", **annot_kwargs),
        ]

        fig.update_layout(
            height=height,
            width=1050,
            title=dict(text=title, pad_b=15, yanchor="bottom", y=1, yref="paper", x=0.5),
            legend=dict(x=1.02),
            margin=dict(t=45, b=20, l=135, r=20),
            shapes=shapes,
            annotations=annotations,
        )

        return fig

    def generate_excursion_plot(self, plot_freq="5min") -> go.Figure:
        if self.df is None or self.event_list is None:
            raise ValueError("Data must be queried and events identified first.")
        if plot_freq is None:
            df = self.df.copy()
        else:
            df = self.df.resample(plot_freq).apply(
                lambda x: x.min() if x["volt_poi"].mean() < x["volt_target"].mean() else x.max()
            )
        title = f"{self.site} - Voltage Profile - {df.index[0].strftime('%b-%Y')}"
        fig = self._build_subplot_figure(df=df, events_to_shade=self.event_list, title=title)
        self.excursion_plot = fig
        return fig

    def generate_event_plot(self, event_idx: int, height: int = 300) -> go.Figure:
        if self.event_list is None:
            raise ValueError("Events must be identified before generating event plots.")
        if event_idx >= len(self.event_list):
            raise IndexError(
                f"Event index {event_idx} out of range ({len(self.event_list)} events)."
            )

        event = self.event_list[event_idx]

        # determine x-range around event for plot
        t0_, t1_ = event.index.min(), event.index.max()
        delta_minutes = (t1_ - t0_).total_seconds() / 60
        if delta_minutes < 0.5:
            window = pd.Timedelta(minutes=1)
        else:
            window_minutes = math.ceil(delta_minutes / 5) * 5  # up to nearest 5
            window = pd.Timedelta(minutes=window_minutes)

        t0, t1 = t0_ - window, t1_ + window
        df = self.df[(self.df.index >= t0) & (self.df.index < t1)].copy()
        title = (
            f"{self.site} - Event {event_idx + 1} of {len(self.event_list)}"
            f" - {event.index[0].strftime('%Y-%m-%d %H:%M')}"
        )
        return self._build_subplot_figure(df, events_to_shade=[event], title=title, height=height)

    def run_analysis(self, plot_freq=None, q=True):
        self.query_data(q=q)
        self.identify_events()
        self.generate_excursion_plot(plot_freq=plot_freq)

    def _format_events_table(self) -> pd.DataFrame:
        df = self.df_events.copy()
        for col in ("Start Time", "End Time"):
            df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
        for col in (
            "Start Setpoint (kV)",
            "Start POI (kV)",
            "End Setpoint (kV)",
            "End POI (kV)",
            "Duration (mins)",
            "Max Delta (kV)",
            "Min Delta (kV)",
        ):
            df[col] = df[col].map(lambda x: f"{x:.2f}")
        return df


# ------------------------------------------------------------------ #
# Module-level report functions
# ------------------------------------------------------------------ #


def _report_title(analyses: list, separator: str = " – ") -> str:
    months = {a.start.strftime("%B %Y") for a in analyses}
    month_str = next(iter(months)) if len(months) == 1 else "Multiple Months"
    return f"Voltage Deviation Analysis{separator}{month_str}"


def _report_subtitle(analyses: list) -> str:
    sites = [a.site for a in analyses]
    if len(sites) == 1:
        sites_str = sites[0]
    elif len(sites) == 2:
        sites_str = " & ".join(sites)
    else:
        sites_str = ", ".join(sites[:-1]) + ", & " + sites[-1]
    return f"Included Sites: {sites_str}"


def generate_email_summary(analyses: "VoltageAnalysis | list[VoltageAnalysis]") -> str:
    """Return a plain-text email summary for one or more completed voltage analyses.

    Parameters
    ----------
    analyses : VoltageAnalysis or list of VoltageAnalysis
        Completed analysis instances (``run_analysis`` or ``from_existing_file``
        must have been called first).
    """
    if isinstance(analyses, VoltageAnalysis):
        analyses = [analyses]

    title = _report_title(analyses)
    subtitle = _report_subtitle(analyses)

    lines = [title, subtitle, ""]

    for a in analyses:
        n = len(a.event_list)
        lines.append(a.site)
        if n == 0:
            lines.append("No voltage deviation events were detected during the analysis period.")
        else:
            lines.append(
                f"{n} voltage deviation event(s) were detected during the analysis period."
            )
            for i, row in a.df_events.iterrows():
                start = pd.to_datetime(row["Start Time"]).strftime("%Y-%m-%d %H:%M")
                duration = row["Duration (mins)"]
                max_abs_delta = max(abs(row["Max Delta (kV)"]), abs(row["Min Delta (kV)"]))
                lines.append(
                    f"  Event {i + 1}: {start}  |  Duration: {duration:.1f} min"
                    f"  |  Max Δ: {max_abs_delta:.2f} kV"
                )
        lines.append("")

    lines.append("Please see the attached report for full details.")
    return "\n".join(lines)


def _fig_to_html_div(fig: go.Figure, include_plotlyjs) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def _build_voltage_report_html(analyses: list) -> str:
    title = _report_title(analyses)
    subtitle = _report_subtitle(analyses)
    first_fig = True
    sections = []
    first_site = True

    for a in analyses:
        n = len(a.event_list)
        if n == 0:
            summary = "No deviation events detected."
        elif n == 1:
            summary = f"{n} deviation event detected."
        else:
            summary = f"{n} deviation events detected."

        table_html = ""
        if n > 0:
            table_html = "<h3>Events</h3>\n" + a._format_events_table().to_html(
                index=False, border=0, classes="evt-table"
            )

        overview_fig = a.excursion_plot or a.generate_excursion_plot()
        include_js = "cdn" if first_fig else False
        first_fig = False
        overview_div = _fig_to_html_div(overview_fig, include_js)

        event_divs = ""
        for i in range(n):
            fig = a.generate_event_plot(i)
            event_divs += (
                f'<div class="keep-together">\n'
                f"<h4>Event {i + 1} of {n}</h4>\n" + _fig_to_html_div(fig, False) + "\n</div>\n"
            )

        event_section = f"<h3>Event Detail</h3>\n{event_divs}" if n > 0 else ""
        section_class = "" if first_site else ' class="page-break"'
        first_site = False
        sections.append(
            f"<section{section_class}>\n"
            f"  <h2>{a.site}</h2>\n"
            f'  <div class="summary">{summary}</div>\n'
            f"  {table_html}\n"
            f"  <h3>Monthly Overview</h3>\n"
            f"  {overview_div}\n"
            f"  {event_section}\n"
            f"</section>"
        )

    sections_html = "\n".join(sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body  {{ font-family: Arial, sans-serif; margin: 40px; color: #333; font-size: 13px; }}
    h1    {{ font-size: 18px; margin-bottom: 4px; }}
    .subtitle {{ font-size: 13px; color: #666; margin-bottom: 16px; }}
    h2    {{ font-size: 15px; color: #333; margin-top: 32px; margin-bottom: 4px;
             border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
    h3    {{ font-size: 13px; color: #444; margin-top: 20px; margin-bottom: 6px; break-after: avoid; }}
    h4    {{ font-size: 12px; color: #555; margin-top: 16px; margin-bottom: 4px; }}
    .page-break   {{ break-before: page; }}
    .keep-together {{ break-inside: avoid; }}
    .summary   {{ margin: 8px 0 16px 0; }}
    .js-plotly-plot {{ border: 1px solid #ccc; }}
    .evt-table {{ border-collapse: collapse; font-size: 11px; margin-top: 6px; }}
    .evt-table th {{ background: #f0f0f0; padding: 5px 10px; border: 1px solid #ccc;
                     text-align: left; white-space: nowrap; }}
    .evt-table td {{ padding: 5px 10px; border: 1px solid #ccc; white-space: nowrap; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="subtitle">{subtitle}</div>
  {sections_html}
</body>
</html>"""


def _save_voltage_report_pdf(analyses: list, filepath: Path) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image,
            KeepTogether,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        raise ImportError("PDF reports require reportlab: pip install reportlab")
    try:
        import kaleido  # noqa: F401
    except ImportError:
        raise ImportError("PDF reports require kaleido for figure export: pip install kaleido")

    import os
    import tempfile

    title = _report_title(analyses)
    subtitle = _report_subtitle(analyses)

    styles = getSampleStyleSheet()
    s_title = ParagraphStyle("rpt_title", parent=styles["Heading1"], fontSize=16, spaceAfter=2)
    s_subtitle = ParagraphStyle(
        "rpt_subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#666666"),
        spaceAfter=10,
    )
    s_site = ParagraphStyle(
        "rpt_site", parent=styles["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4
    )
    s_section = ParagraphStyle(
        "rpt_section", parent=styles["Heading3"], fontSize=11, spaceBefore=8, spaceAfter=3
    )
    s_event = ParagraphStyle(
        "rpt_event", parent=styles["Heading4"], fontSize=10, spaceBefore=6, spaceAfter=2
    )
    s_body = styles["Normal"]

    tbl_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
    )
    fig_border_style = TableStyle(
        [
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]
    )

    margin = 0.75 * inch
    page_w = letter[0] - 2 * margin

    with tempfile.TemporaryDirectory() as tmp:

        def _export(fig, path):
            fw = fig.layout.width or 1050
            fh = fig.layout.height or 600
            fig.write_image(path, width=fw, height=fh, scale=1.5)
            return (path, fh / fw)

        site_data = []
        for i, a in enumerate(analyses):
            overview_fig = a.excursion_plot or a.generate_excursion_plot()
            overview_img, overview_aspect = _export(
                overview_fig, os.path.join(tmp, f"overview_{i}.png")
            )

            n = len(a.event_list)
            event_imgs = []
            for j in range(n):
                event_imgs.append(
                    _export(a.generate_event_plot(j), os.path.join(tmp, f"event_{i}_{j}.png"))
                )

            site_data.append(
                {
                    "a": a,
                    "n": n,
                    "overview": overview_img,
                    "overview_aspect": overview_aspect,
                    "events": event_imgs,
                }
            )

        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            leftMargin=margin,
            rightMargin=margin,
            topMargin=margin,
            bottomMargin=margin,
        )

        story = [Paragraph(title, s_title), Paragraph(subtitle, s_subtitle), Spacer(1, 0.1 * inch)]

        for idx, sd in enumerate(site_data):
            a, n = sd["a"], sd["n"]

            if idx > 0:
                story.append(PageBreak())

            story.append(Paragraph(a.site, s_site))
            summary = (
                "No deviation events detected." if n == 0 else f"{n} deviation event(s) detected."
            )
            story.append(Paragraph(summary, s_body))
            story.append(Spacer(1, 0.1 * inch))

            if n > 0:
                story.append(Paragraph("Events", s_section))
                df_fmt = a._format_events_table()
                cols = list(df_fmt.columns)
                col_w = page_w / len(cols)
                tbl_data = [cols] + [[str(row[c]) for c in cols] for _, row in df_fmt.iterrows()]
                story.append(Table(tbl_data, colWidths=[col_w] * len(cols), style=tbl_style))
                story.append(Spacer(1, 0.15 * inch))

            story.append(
                KeepTogether(
                    [
                        Paragraph("Monthly Overview", s_section),
                        Table(
                            [
                                [
                                    Image(
                                        sd["overview"],
                                        width=page_w,
                                        height=page_w * sd["overview_aspect"],
                                    )
                                ]
                            ],
                            colWidths=[page_w],
                            style=fig_border_style,
                        ),
                    ]
                )
            )
            story.append(Spacer(1, 0.1 * inch))

            if n > 0:
                for j, (img_path, aspect) in enumerate(sd["events"]):
                    header = [Paragraph("Event Detail", s_section)] if j == 0 else []
                    story.append(
                        KeepTogether(
                            header
                            + [
                                Paragraph(f"Event {j + 1} of {n}", s_event),
                                Table(
                                    [[Image(img_path, width=page_w, height=page_w * aspect)]],
                                    colWidths=[page_w],
                                    style=fig_border_style,
                                ),
                            ]
                        )
                    )
                    story.append(Spacer(1, 0.1 * inch))

        doc.build(story)


def save_voltage_report(
    analyses: "VoltageAnalysis | list[VoltageAnalysis]",
    filepath,
    fmt: str = "html",
) -> None:
    """Generate and save a voltage analysis report for one or more sites.

    Parameters
    ----------
    analyses : VoltageAnalysis or list of VoltageAnalysis
        One or more completed analysis instances (``run_analysis`` or
        ``from_existing_file`` must have been called first).
    filepath : str or Path
        Output file path.
    fmt : {"html", "pdf"}
        Report format.  PDF requires ``fpdf2`` and ``kaleido``.
    """
    if isinstance(analyses, VoltageAnalysis):
        analyses = [analyses]
    if not analyses:
        raise ValueError("No analyses provided.")
    for a in analyses:
        if a.df is None or a.event_list is None:
            raise ValueError(f"Analysis for '{a.site}' has not been run.")

    fpath = Path(filepath)
    if fmt == "html":
        fpath.write_text(_build_voltage_report_html(analyses), encoding="utf-8")
    elif fmt == "pdf":
        _save_voltage_report_pdf(analyses, fpath)
    else:
        raise ValueError(f"Unsupported format {fmt!r}. Use 'html' or 'pdf'.")
