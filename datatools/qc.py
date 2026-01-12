from astral.sun import sun
from astral import LocationInfo
import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils.dataset import Dataset
from ..utils.helpers import quiet_print_function


def get_sun(
    date: str, tz: str, lat: float, lon: float, name: str = "", region: str = ""
) -> dict[str, pd.Timestamp]:
    """Returns dictionary with times of the sun on given date for dawn, sunrise, noon, sunset, and dusk.

    Parameters
    ----------
    date : str | pd.Timestamp
        The date for which to calculate sun times.
    tz : str
        The timezone of the desired location.
    lat : float
        The latitude of the desired location.
    lon : float
        The longitude of the desired location.
    name : str, optional
        Location name. Does not affect calculations.
    region : str, optional
        Region location is in. Does not affect calculations.

    Returns
    -------
    dict[str, pd.Timestamp]
        A dictionary with keys = dawn, sunrise, noon, sunset, dusk and corresponding pd.Timestamp values
        with the calculated sun times at the specified location.
    """
    location = LocationInfo(name=name, region=region, timezone=tz, latitude=lat, longitude=lon)
    observer = location.observer
    target_date = pd.Timestamp(date).floor("D")
    output = {}
    for key, dtime in sun(observer, date=target_date).items():
        local_tstamp = pd.Timestamp(dtime).tz_convert(tz).tz_localize(None)
        if local_tstamp.date() != dtime.date():
            local_tstamp = pd.Timestamp(dt.datetime.combine(dtime.date(), local_tstamp.time()))
        tstamp = local_tstamp.round("min")  # round to nearest minute
        if pd.Timestamp(date).tzinfo is not None:
            tstamp = tstamp.tz_localize(tz=tz)
        output.update({key: tstamp})
    return output


def get_sun_times(
    start_date: str,
    end_date: str,
    tz: str,
    lat: float,
    lon: float,
    name: str = "",
    region: str = "",
) -> list[dict[str, pd.Timestamp]]:
    """Returns list of dictionaries with dawn and dusk times for each day in specified date range."""
    location_kwargs = dict(tz=tz, lat=lat, lon=lon, name=name, region=region)
    sun_times = []
    for date in pd.date_range(start_date, end_date):
        sun_dict = get_sun(date, **location_kwargs)
        sun_times.append({"dawn": sun_dict["dawn"], "dusk": sun_dict["dusk"]})
    return sun_times


def get_daylight_timestamps(
    start_date: str,
    end_date: str,
    tz: str,
    lat: float,
    lon: float,
    name: str = "",
    region: str = "",
    freq="1min",
):
    location_kwargs = dict(tz=tz, lat=lat, lon=lon, name=name, region=region)
    timestamp_list = []
    for dict_ in get_sun_times(start_date, end_date, **location_kwargs):
        daylight_range = pd.date_range(dict_["dawn"], dict_["dusk"], freq=freq, inclusive="left")
        timestamp_list.extend(list(daylight_range))
    return list(filter(lambda t: t < end_date, timestamp_list))


def comparison_figure(
    df_inv_0, df_inv_1, resample=True, ignore_no_changes=False, q=True, **location_kwargs
):
    qprint = quiet_print_function(q=q)

    common_cols = [c for c in df_inv_0.columns if c in df_inv_1.columns]
    df0 = df_inv_0[common_cols].copy()
    df1 = df_inv_1[common_cols].copy()

    freq = pd.infer_freq(df0.index)
    if not freq:
        raise ValueError("Dataframe does not have a uniform frequency/interval.")
    freq_timedelta = df0.index.diff().max()

    compare_columns = list(
        set(df0.round(4).compare(df1.round(4)).columns.get_level_values(level=0).to_list())
    )

    if len(compare_columns) == 0:
        print("No changes detected.")
        if ignore_no_changes is False:
            return
        compare_columns = ["all"]

    # resample to 15-min for plot
    if resample is True:
        df0_plot = df0.resample("15min").mean().copy()
        df1_plot = df1.resample("15min").mean().copy()
    else:
        df0_plot, df1_plot = df0.copy(), df1.copy()

    plot_freq = pd.infer_freq(df1_plot.index)
    n_rows = len(compare_columns)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=compare_columns,
        vertical_spacing=0.275 / n_rows,
        specs=[[{"secondary_y": True}]] * n_rows,
    )
    hvtemp = (
        "<b>%{fullData.name}</b><br>Value: %{y:.2f}<br><i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
    )
    kwargs = dict(x=df1_plot.index.copy(), mode="lines", showlegend=False, hovertemplate=hvtemp)

    # add daylight trace
    if all(k in location_kwargs for k in ["tz, lat, lon"]):
        loc_kwargs = {
            k: val for k, val in location_kwargs.items() if k in ["tz, lat, lon", "name", "region"]
        }
        start = df0.index.min().floor("D")
        end = df0.index.max().ceil("D")
        daylight_tstamps = get_daylight_timestamps(start, end, **loc_kwargs)
        df_daylight = pd.DataFrame(
            index=df0.index, data={"is_daytime": df0.index.isin(daylight_tstamps).astype(int)}
        )
        fig.add_trace(
            go.Scattergl(
                x=df_daylight.index,
                y=df_daylight["is_daytime"],
                name="daytime",
                line=dict(color="#999", dash="dot", width=1),
                showlegend=False,
            ),
            secondary_y=True,
            row="all",
            col=1,
        )
    if compare_columns[0] == "all":
        for c in common_cols:
            fig.add_trace(
                go.Scattergl(**kwargs, y=df0_plot[c], name=c), secondary_y=False, row=1, col=1
            )
    else:
        for i, col in enumerate(compare_columns, start=1):
            trace0 = go.Scattergl(
                **kwargs,
                y=df0_plot[col],
                name=f"{col}_ORIGINAL",
                line=dict(color="#333", dash="dot", width=1),
            )
            fig.add_trace(trace0, row=i, col=1, secondary_y=False)
            trace1 = go.Scattergl(**kwargs, y=df1_plot[col], name=f"{col}_NEW")
            fig.add_trace(trace1, row=i, col=1, secondary_y=False)

            # add shaded regions to subplot to highlight removed data
            df_cmp = df0[[col]].round(4).compare(df1[[col]].round(4))
            df_cmp.columns = df_cmp.columns.droplevel(0)  # drop level of multiindex columns
            df_cmp["time_diff"] = df_cmp.index.diff()
            if df_cmp.shape[0] > 1:
                df_cmp.iloc[0, -1] = df_cmp.index[1] - df_cmp.index[0]

            # group rows with continuous date ranges
            data_interval = freq_timedelta
            grp_conditions = df_cmp["time_diff"].eq(data_interval)

            # separate/break groups using cumulative sum of condition: [timedelta > 1min]
            breaks_ = df_cmp["time_diff"].gt(data_interval).cumsum()

            # use groupby function to separate df_cmp into continuous ranges according to the above conditions
            grouped_df_list = [dfg for _, dfg in df_cmp[grp_conditions].groupby(breaks_)]

            vrect_kwargs = dict(
                row=i, line_width=0, opacity=0.4, layer="below", fillcolor="lightsalmon"
            )
            n_ranges = 0
            for dfg in grouped_df_list:
                x0_, x1_ = dfg.index.min(), dfg.index.max()
                if (x1_ - x0_) < pd.Timedelta(hours=2):
                    continue
                x0_plot = x0_.floor(plot_freq)
                x1_plot = x1_.floor(plot_freq)
                fig.add_vrect(x0=x0_plot, x1=x1_plot, **vrect_kwargs)
                n_ranges += 1
            qprint(f"{col = }, {n_ranges = }")

    # set alignment for subplot titles (annotations in subplot figure)
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].update(font_size=12, x=1, xref="paper", xanchor="right")

    fig.update_layout(
        font_size=9,
        height=max(400, 120 * n_rows),
        margin=dict(t=40, b=20, l=0, r=20),
        title=dict(font_size=14, text=f"<b>QC Comparison</b>", x=0.015),
        hoverlabel_font_size=11,
    )
    fig.update_yaxes(showline=True, linewidth=1, linecolor="#777", secondary_y=False)
    fig.update_yaxes(secondary_y=True, side="right", rangemode="tozero")
    fig.update_xaxes(showline=True, linewidth=1, linecolor="#777")

    if all([("OE.ActivePower" in c) for c in compare_columns]) or compare_columns[0] == "all":
        ymax_ = df1.max().max() * 1.05
        fig.update_yaxes(range=[0, ymax_], secondary_y=False)
    else:
        for i, col in enumerate(compare_columns, start=1):
            ymin_ = df1[col].min() * 0.95
            ymax_ = df1[col].max() * 1.05
            fig.update_yaxes(row=i, range=[ymin_, ymax_], secondary_y=False)

    return fig


# outlier detection method: interquartile range (IQR)
def get_iqr_bounds(series, iqr_multiplier=1.5):
    """Returns lower/upper bounds for IQR with specified multiplier."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    iqr = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * iqr
    upper_bound = Q3 + iqr_multiplier * iqr
    return lower_bound, upper_bound


def iqr_outlier_condition(series, iqr_multiplier=1.5):
    """Returns condition for use in outlier removal."""
    lower_bound, upper_bound = get_iqr_bounds(series, iqr_multiplier)
    return series.lt(lower_bound) | series.gt(upper_bound)


# outlier detection method: z_score (for rolling window)
def z_score_outlier_condition(series: pd.Series, z_threshold: int = 3):
    """condition indicates values that exceed z score threshold (# of std. devs. from mean)
    -> window will be determined based on data interval as follows:
        freq=1min -> window=30
        freq=5min -> window=6
        freq=15min -> window=4
        freq=1h -> window=2 (TODO: determine whether this should be supported)
    """
    freq_mins = int(series.index.diff.max().total_seconds() / 60)
    window = {1: 30, 5: 6, 15: 4, 60: 2}.get(freq_mins, None)
    if window is None:
        raise KeyError("Unsupported data interval/frequency detected.")
    rolling_mean = series.rolling(window=window).mean().copy()
    rolling_std = series.rolling(window=window).std().copy()
    rolling_z = (series - rolling_mean) / rolling_std
    return rolling_z.abs().gt(z_threshold)


def remove_outliers(series, method, **kwargs) -> pd.Series:
    """removes outliers from series using selected method; returns number of points removed
    -> NOTE: updates the existing series object; returns separate series with removed outliers.
    """
    if method not in ("iqr", "z_score"):
        raise ValueError(f"Unsupported {method = }.")

    if method == "iqr":
        iqr_multiplier = kwargs.get("iqr_multiplier", 1.5)
        is_outlier = iqr_outlier_condition(series=series, iqr_multiplier=iqr_multiplier)
    elif method == "z_score":
        z_threshold = kwargs.get("z_threshold", 3)
        is_outlier = z_score_outlier_condition(series=series, z_threshold=z_threshold)

    series_outliers = series.loc[is_outlier].copy()
    series.loc[is_outlier] = np.nan
    return series_outliers


class QCDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        self.data = dataframe.select_dtypes(include="number").copy()
        self.columns = list(self.data.columns)
        self._validate_dataframe()  # assigns other properties/attributes

    def _validate_dataframe(self) -> None:
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise TypeError("Dataframe must have a datetime index.")

        # validate index (use smallest detected frequency/interval)
        freq_timedelta = self.data.index.to_series().diff().min()
        start_datetime = self.data.index.min()
        end_datetime = self.data.index.max()
        continuous_index = pd.date_range(start_datetime, end_datetime, freq=freq_timedelta)
        if len(self.data.index) != len(continuous_index):
            raise ValueError("Dataframe index must be continuous; found missing timestamp(s).")

        self.start = start_datetime
        self.end = end_datetime
        self.freq_timedelta = freq_timedelta

    def remove_outliers(
        self, method: str, iqr_multiplier=1.5, z_threshold=3
    ) -> dict[str, pd.DataFrame]:
        """Removes outliers using either z-score or interquartile range method.

        Parameters
        ----------
        method : str
            Outlier detection method; either "z_score" or "iqr".

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary of removed outliers by column. Updates self.data
        """
        kwargs = dict(method=method)
        if method == "iqr":
            kwargs.update(dict(iqr_multiplier=iqr_multiplier))
        else:
            kwargs.update(dict(z_threshold=z_threshold))

        outlier_dict = {}
        series_list = []
        for col in self.columns:
            series = self.data[col].copy()
            outliers = remove_outliers(series, **kwargs)
            series_list.append(series)
            if outliers.empty:
                continue
            outlier_dict.update({col: outliers})

        self.data = pd.concat(series_list, axis=1)
        return outlier_dict
