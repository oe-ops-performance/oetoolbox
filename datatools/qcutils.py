import os, sys, math
import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
import itertools, calendar

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from astral.sun import sun
from astral import LocationInfo
from astral.location import Location
import meteostat
import pvlib
from pvlib.location import Location

from ..utilities import oemeta, oepaths
from ..dataquery.external import query_DTN
import oetoolbox.reporting.tools as rt


hasID_ = lambda col, ids_: any(i in col.casefold() for i in ids_)
# isGHI = lambda c: hasID_(c, ['ghi', 'global', 'hor'])
isGHI = lambda c: hasID_(c, ["ghi", "hor"])
isPOA = lambda c: hasID_(c, ["poa", "plane", "irr"]) and (not isGHI(c))
isINV = lambda c: ("OE.ActivePower" in c)  # or ('MeterMW' in c)
isMETER = lambda c: ("MeterMW" in c)
isTEMP = lambda c: any(
    i in c.casefold() for i in ["temp", "tmp", "bom", "mts", "mst", "mdl", "mod"]
)
isWIND = lambda c: any(i in c.casefold() for i in ["wind", "wnd", "speed", "ws_ms"])
is_daylight_dependent = lambda c: any([isGHI(c), isPOA(c), isINV(c), isMETER(c)])


def group_met_columns(column_names, q=True):
    groups_ = {
        "POA": list(filter(isPOA, column_names)),
        "GHI": list(filter(isGHI, column_names)),
        "TEMP": list(filter(isTEMP, column_names)),
        "WIND": list(filter(isWIND, column_names)),
    }
    col_groups = {grp: cols for grp, cols in groups_.items() if cols}
    if not q:
        grouped_cols = [col for list_ in col_groups.values() for col in list_]
        n_missing = len(column_names) - len(grouped_cols)
        print(f"grouped {len(grouped_cols)} of {len(column_names)} columns")
        print(f"sensor types: {list(col_groups.keys())}")
        if n_missing > 0:
            print(f"\n>> excluded columns: {[c for c in column_names if c not in grouped_cols]}\n")
    return col_groups


## for plotting
def group_inv_columns(column_names, n_groups=None):
    invcols_ = column_names
    if n_groups is None:
        n_groups = 3
    n_cols = len(invcols_)
    cols_per_grp = math.ceil(n_cols / n_groups)

    col_groups = {}
    for i, j in enumerate(range(0, n_cols, cols_per_grp)):
        col_groups[f"subgroup_{i+1}"] = invcols_[j : j + cols_per_grp]
    return col_groups


def grouped_columns(column_list, met_data=False, n_groups=None):
    if len(column_list) == 1:
        return [[c] for c in column_list]
    if met_data:
        col_groups = group_met_columns(column_list)
        ungrouped_cols = [c for c in column_list if c not in col_groups.values()]
        if len(ungrouped_cols) > 0:
            col_groups["Ungrouped"] = ungrouped_cols
    else:
        col_groups = group_inv_columns(column_list, n_groups)
    return col_groups


getsun = lambda tz, lat, lon, date: sun(LocationInfo("", "", tz, lat, lon).observer, date=date)
suntime_utc = lambda key_, tz, lat, lon, date: pd.Timestamp(getsun(tz, lat, lon, date).get(key_))
getDawn = (
    lambda tz, lat, lon, date: suntime_utc("dawn", tz, lat, lon, date)
    .tz_convert(tz)
    .tz_localize(None)
    .floor("min")
)
getDusk = (
    lambda tz, lat, lon, date: suntime_utc("dusk", tz, lat, lon, date)
    .tz_convert(tz)
    .tz_localize(None)
    .ceil("min")
)


def get_daylight_dateranges(tz, lat, lon, start, end):
    date_range = pd.date_range(start, end)  # freq: day
    args_ = [tz, lat, lon]
    dawn_list = [getDawn(*args_, date) for date in date_range]
    dusk_list = [getDusk(*args_, date + dt.timedelta(days=1)) for date in date_range]
    return [pd.date_range(dawn_, dusk_, freq="1min") for dawn_, dusk_ in zip(dawn_list, dusk_list)]


def get_all_daylight_timestamps(tz, lat, lon, start, end):
    daylight_dateranges = get_daylight_dateranges(tz, lat, lon, start, end)
    return [t for rng in daylight_dateranges for t in rng]


getfile = lambda fplist: None if (not fplist) else max((fp.stat().st_ctime, fp) for fp in fplist)[1]


def get_qc_fpaths(site, year, month, type_=None):
    site_fp = Path(oepaths.frpath(year, month, ext="Solar"), site)
    glob_dict = {
        "met": "PIQuery_MetStations*.csv",
        "inv": "PIQuery_Inverters*.csv",
        "meter": "PIQuery_Meter*.csv",
    }

    if type_ is None:
        qctypes_ = ["met", "inv", "meter"]
    elif isinstance(type_, str):
        if type_.casefold() not in glob_dict:
            return None
        qctypes_ = [type_]
    else:
        ##unsupported type arg
        return None

    fpath_dict = {}
    for type_ in qctypes_:
        fplist = list(site_fp.glob(glob_dict[type_]))
        raw_fps = [fp for fp in fplist if "CLEANED" not in fp.name]
        clean_fps = [fp for fp in fplist if ("CLEANED" in fp.name) and ("PROCESSED" not in fp.name)]
        qc_fpaths = list(map(getfile, [raw_fps, clean_fps]))
        fpath_dict[type_] = qc_fpaths

    if len(qctypes_) == 1:
        return qc_fpaths
    return fpath_dict


grouped_lists = lambda list_, n_: [list_[i : i + n_] for i in range(0, len(list_), n_)]


def show_qc_status(year, month):

    ##check meter files
    all_frsite_fps = rt.solar_frpaths(year, month, all_fps=True)
    hasmeterfile = lambda site_fp: len(list(site_fp.glob("PIQuery_Meter*.csv"))) > 0
    hascleanmeterfile = lambda site_fp: len(list(site_fp.glob("PIQuery_Meter*CLEANED*.csv"))) > 0
    site_fps_with_meter = [fp for fp in all_frsite_fps if hasmeterfile(fp)]
    needs_meter = [fp.name for fp in site_fps_with_meter if (not hascleanmeterfile(fp))]
    meter_qc_complete = [fp.name for fp in site_fps_with_meter if hascleanmeterfile(fp)]

    frstatus_dict = rt.get_solar_FR_status(year, month, output=True, all_fps=True)
    completed_ = frstatus_dict["qc complete"]
    completed_sites = [s for s in completed_ if s in meter_qc_complete]
    if completed_sites:
        print("Sites with existing clean/qc'd data:")
    for list_ in grouped_lists(sorted(completed_sites), 4):
        print(" " * 4, end="")
        for i, site in enumerate(list_):
            print(site.ljust(19), end="" if (i + 1) < len(list_) else "\n")

    sites_in_need = list(sorted(set(frstatus_dict["needs manual qc"] + needs_meter)))
    needs_met, needs_inv = (frstatus_dict[f"needs {k_} qc"] for k_ in ["met", "inv"])

    output_dict = {}
    if sites_in_need:
        print("\nSites that require qc:")
    for site in sites_in_need:
        types_ = []
        if site in needs_met:
            types_.append("met")
        if site in needs_inv:
            types_.append("inv")
        if site in needs_meter:
            types_.append("meter")
        print(f'    {site.ljust(20)}({"+".join(types_)})')
        output_dict[site] = types_
    return output_dict


loadfile = lambda fp: (
    pd.DataFrame() if (fp is None) else pd.read_csv(fp, index_col=0, parse_dates=True)
)


def verified_filepath(fpath):
    parent_ = fpath.parent
    n_, stem_ = 0, fpath.stem
    ext_ = fpath.suffix
    getpath = lambda n: (
        Path(parent_, stem_ + ext_) if (n < 1) else Path(parent_, f"{stem_}({n}){ext_}")
    )
    while getpath(n_).exists():
        n_ += 1
    return getpath(n_)


def run_auto_qc(df_raw, site, q=True, q_xtra=True):
    """Script to automatically clean raw PI query data file."""
    qprint = lambda msg_, end_="\n": None if q else print(msg_, end=end_)
    xprint = lambda msg_, end_="\n": (
        None if (q or q_xtra) else print(msg_, end=end_)
    )  # for printout of extra details

    if not isinstance(df_raw.index, pd.DatetimeIndex):
        qprint("Error: dataframe must have datetime index.")
        return None

    df_raw = df_raw.select_dtypes(include="number")
    if df_raw.empty:
        qprint("Error: no numeric columns found in dataset.")
        return None

    # get timezone and location info
    tz = oemeta.data["TZ"].get(site)
    lat, lon = oemeta.data["LatLong"].get(site)

    # get date range
    getdate_ = lambda tstamp_: tstamp_.floor("D")
    start_day, end_day = map(getdate_, [df_raw.index.min(), df_raw.index.max()])

    # get daylight metadata
    daylight_tstamps = get_all_daylight_timestamps(tz, lat, lon, start_day, end_day)

    """
    AUTO-DETECTION [PART 0] -- empty columns
    """
    isinvfile = any("OE.ActivePower" in c for c in df_raw.columns)
    if not isinvfile and any(df_raw.isna().all()):
        qprint("\n" + "_" * 35 + "[[ PART 0 ]]" + "_" * 35 + "\n")
        df = df_raw.dropna(axis=1, how="all").copy()
        rmvd_cols = [c for c in df_raw.columns if c not in df.columns]
        qprint(f">> removed empty data columns;\n{rmvd_cols}")
    else:
        df = df_raw.copy()

    data_cols = list(df.columns)
    jlen = max([len(c) for c in data_cols]) + 3
    df["PROCESSED"] = 0

    """
    AUTO-DETECTION [PART 1a] -- bad columns (low quality data)
              >> developed for Sweetwater modtemp sensor "BOM_Temp_2_B2.10" (June-2024 data)
    """
    n_rows = df.shape[0]
    modtemp_cols = [c for c in data_cols if isTEMP(c)]
    if len(modtemp_cols) > 1:
        n_badvals = lambda c: df[(df[c].lt(-30) | df[c].gt(90))].shape[0]
        pct_bad = lambda c: (n_badvals(c) / n_rows)
        pct_badvals_dict = {col: pct_bad(col) for col in modtemp_cols if pct_bad(col) > 0.70}
        badval_pct_list = [(pct, col) for col, pct in pct_badvals_dict.items()]
        badcols_to_remove = [col for pct, col in sorted(badval_pct_list, reverse=True)]
        if len(badcols_to_remove) == len(modtemp_cols):
            badcols_to_remove = badcols_to_remove[:-1]  # leave the least-bad column
        if badcols_to_remove:
            qprint("\n" + "_" * 35 + "[[ PART 1a ]]" + "_" * 35 + "\n")
            df = df.drop(columns=badcols_to_remove)
            qprint(f">> removed bad-quality mod temp columns;\n{badcols_to_remove}")
            data_cols = [c for c in data_cols if c not in badcols_to_remove]

    """
    AUTO-DETECTION [PART 1] -- outliers
    """
    qprint("\n" + "_" * 35 + "[[ PART 1 ]]" + "_" * 35 + "\n")
    n_outliers = 0  # init
    col_outlier_dict = {}
    for col in data_cols:
        cstats = df[col].describe(percentiles=[0.95])
        p95_ = cstats.loc["95%"]
        max_ = cstats.loc["max"]
        min_ = cstats.loc["min"]

        delta_ = ((max_ - p95_) / p95_) if p95_ > 0 else 0
        # primarily for transient spikes (like in wind data, or inverter data -e.g.FL4)
        if (isINV(col) or isWIND(col)) and (delta_ < 5):
            continue

        if isWIND(col):
            cnd_ = (
                df[col].abs().gt(24)
            )  # 24 m/s is ranked 10 of 12 on the Beaufort Wind Scale (Whole Gale)
        elif isTEMP(col):
            cnd_ = df[col].lt(-30) | df[col].gt(
                90
            )  # based on historical ranges of mod temp sensors across all sites
        else:
            cnd_ = (df[col].shift(-1) - df[col]).abs().rolling(4, center=True).mean().gt(1000)
        n_pts = df[cnd_].shape[0]
        col_outlier_dict[col] = n_pts
        n_outliers += n_pts
        if n_pts > 0:
            qprint(f"{col.ljust(jlen)} >> found {n_pts} outliers")

        ## remove pts (overwrite with NaN value)
        df.loc[cnd_, col] = np.nan
        df.loc[cnd_, "PROCESSED"] = 1

    if n_outliers > 0:
        qprint(f"\nTotal of {n_outliers} outliers removed.\n")
    else:
        qprint("\nNo outliers found.\n")

    """
    AUTO-DETECTION [PART 2] -- interpolated data
    """
    qprint("\n" + "_" * 35 + "[[ PART 2 ]]" + "_" * 35 + "\n")
    n_interpolated = 0  # init
    col_interpolation_dict = {}

    for col in data_cols:
        n_interp_col = 0
        d1_col = f"{col}_d1"
        d2_col = f"{col}_d2"

        qprint(col)
        dfg = df[[col]].copy()
        dfg[d1_col] = dfg[col].shift(-1) - dfg[col]
        dfg[d2_col] = dfg[d1_col].shift(-1) - dfg[d1_col]

        ## get interpolation threshold (min. duration to flag)
        threshold_ = 120  # 2 hours (default)
        if isTEMP(col) or isMETER(col):
            threshold_ = 240  # 4 hours
        elif isINV(col) and ("_Pad_" in col):
            threshold_ = 300  # 5 hours   (FL4)
        elif isWIND(col):
            threshold_ = 1440  # 24 hours

        ## define groupby conditions & break points
        flag_1 = dfg[d2_col].rolling(15, center=True).mean().round(4)

        conditions_ = flag_1.eq(0)
        if isINV(col):
            conditions_ = flag_1.eq(0) & dfg[col].gt(1)
        breaks_ = (flag_1.abs().gt(0)).cumsum()
        groupby_obj = dfg[conditions_].groupby(breaks_)
        grouped_df_list = [df_grp for g, df_grp in groupby_obj]

        n_grps = len(grouped_df_list)
        n_above_threshold = len(
            [df_grp for df_grp in grouped_df_list if df_grp.shape[0] >= threshold_]
        )

        for grp, df_grp in groupby_obj:
            grp_length = df_grp.shape[0]
            if grp_length < threshold_:
                continue

            grp_timedelta = (
                df_grp.index.max() - df_grp.index.min()
            )  # update thresholds to use timedeltas instead of hardcoded numbers

            if is_daylight_dependent(col) or isTEMP(col):
                df_grp_daytime = df_grp[df_grp.index.isin(daylight_tstamps)].copy()
                df_night = df_grp[~df_grp.index.isin(daylight_tstamps)].copy()

                pctnight = (grp_length - df_grp_daytime.shape[0]) / grp_length
                pct_limit = 0.7  # prev 0.75 (changed when reviewing mulberry)

                if isTEMP(col) and (pctnight > pct_limit):
                    xprint(f"  ~~ nighttime group; ({pctnight:.2%} night); skipping ~~")
                    continue

                ## ac module outage condition
                lmt_ = 25 if not isMETER(col) else 0.02
                if isINV(col) and (pctnight < 0.05) and (df_grp[col].mean() > lmt_):
                    # if isMETER(col) and (df_grp[d1_col].abs().mean() < .01)
                    # if (df_grp[d1_col].abs().mean() < 1):
                    xprint(
                        f"  ~~ daylight group; ({1-pctnight:.2%} day); AC mod condition (or semi-linear), skipping ~~ {df_grp.index.max()}, {df_grp[d2_col].abs().max()}"
                    )
                    continue

                # if (isGHI(col) or isPOA(col)) and (pctnight < 0.05) and (grp_timedelta < pd.Timedelta(hours=2.5)) and (df_grp[col].mean() > 25) and (df_grp[d1_col].abs().mean() > 1):
                if (
                    (pctnight < 0.05)
                    and (grp_timedelta < pd.Timedelta(hours=2.5))
                    and (df_grp[col].mean() > 25)
                    and (df_grp[d1_col].abs().mean() > 1)
                ):
                    xprint(
                        f"  ~|~|~ daylight group, semi-linear; ({1-pctnight:.2%} day); skipping..  {grp_timedelta}, {df_grp[col].mean():.2f}, {df_grp[d1_col].abs().mean():.2f}"
                    )
                    continue

                if pctnight > pct_limit:
                    if (df_night[col].mean() < 5) and (df_night[col].max() < 5):
                        xprint(
                            f"  ~~ nighttime group, flatlined at zero; ({pctnight:.2%} night); skipping ~~"
                        )
                        continue

                if (pctnight > 0.5) and (grp_timedelta < pd.Timedelta(hours=8)):
                    limit_ = 25 if not isMETER(col) else 0.02
                    if (df_night[col].mean() < 5) and (df_grp[col].max() < limit_):
                        xprint(
                            f"  ~`~`~ mostly night, flatlined at zero; ({pctnight:.2%} night); skipping ~~ {df_grp.index.min()}, {df_grp[col].mean():.2f}, {df_grp[d1_col].abs().mean():.2f}"
                        )
                        continue

                if (pctnight > 0.2) and (grp_timedelta < pd.Timedelta(hours=2)):
                    xprint(
                        f"  ~`~`~`````` some night, super short duration, near zero; ({pctnight:.2%} night); skipping ~~"
                    )
                    continue

                if isMETER(col) and (df_grp[col].mean().round(1) == 0.0):
                    xprint(
                        f"  ~~ meter outage (flatline at zero) - ({pctnight:.2%} night); skipping ~~"
                    )
                    continue

                xprint(f"({pctnight:.2%} night)")

            ## remove data from df
            ll_, ul_ = df_grp.index.min(), df_grp.index.max()
            rng_ = pd.date_range(ll_, ul_, freq="1min")
            cond_ = df.index.isin(rng_)

            df.loc[cond_, col] = np.nan
            df.loc[cond_, "PROCESSED"] = 1
            qprint(
                f'  >> removed {grp_length} interpolated values; ["{str(ll_)[5:-3]}", "{str(ul_)[5:-3]}"]  ({grp_timedelta})'
            )
            n_interp_col += grp_length

        col_interpolation_dict[col] = n_interp_col
        n_interpolated += n_interp_col

    qprint(f"\nTotal of {n_interpolated} interpolated values removed.\n")

    """
    AUTO-DETECTION [PART 3] -- daylight-dependent cols > 0 during night hours
    """
    qprint("\n" + "_" * 35 + "[[ PART 3 ]]" + "_" * 35 + "\n")
    affected_cols = [c for c in data_cols if is_daylight_dependent(c)]

    n_positive_night = 0
    col_pos_night_dict = {}
    for col in affected_cols:
        x = 15  # 5 if isINV(col) else 15   #threshold of 5 for inverter data & 15 for POA/GHI data
        notday_ = ~df.index.isin(daylight_tstamps)
        cnds_ = df[col].gt(x) & notday_
        if df.loc[cnds_, col].empty:
            continue

        n_overwrite = df.loc[cnds_, col].shape[0]
        df.loc[cnds_, col] = 0
        df.loc[cnds_, "PROCESSED"] = 1
        qprint(f"{col.ljust(jlen)} >> changed {n_overwrite} values to zero")
        n_positive_night += n_overwrite
        col_pos_night_dict[col] = n_overwrite

    qprint(f"\nTotal of {n_positive_night} positive night values changed to zero.\n")

    """
    AUTO-DETECTION [PART 4] -- negative values for POA or GHI cols
    """
    qprint("\n" + "_" * 35 + "[[ PART 4 ]]" + "_" * 35 + "\n")
    affected_cols = [c for c in data_cols if (isPOA(c) or isGHI(c))]

    n_negative_vals = 0
    col_negative_val_dict = {}
    for col in affected_cols:
        lowerlimit_ = -5  # prev 0
        if df.loc[df[col].lt(lowerlimit_), col].empty:
            continue

        n_negative = df.loc[df[col].lt(lowerlimit_), col].shape[0]
        df.loc[df[col].lt(lowerlimit_), col] = 0
        df.loc[df[col].lt(lowerlimit_), "PROCESSED"] = 1
        qprint(f"{col.ljust(jlen)} >> changed {n_negative} negative values to zeros")
        n_negative_vals += n_negative
        col_negative_val_dict[col] = n_negative

    qprint(f"\nTotal of {n_negative_vals} negative values changed to zero.")

    return df


def solar_reporting_autoqc(
    site, year, month, type_, save_file=False, overwrite=False, q=True, q_xtra=True
):
    qprint = lambda msg_, end_="\n": None if q else print(msg_, end=end_)

    if type_ not in ["met", "inv", "meter"]:
        qprint("error: type_ must be one of ['met', 'inv', 'meter']\nexiting..")
        return

    raw_fp, clean_fp = get_qc_fpaths(site, year, month, type_)
    if raw_fp is None:
        qprint(f"error: no raw data file found.\nexiting..")
        return

    ## get savepath (if save_file)
    site_fp = raw_fp.parent  # site flashreport path for year/month

    file_prefix = raw_fp.name.split(str(year))[0] + f"{year}-{month:02d}"

    if save_file:
        clean_filename = f"{file_prefix}_CLEANED" + raw_fp.suffix
        savepath_ = Path(site_fp, clean_filename)
        if savepath_.exists():
            qprint('>> note: "CLEANED" file already exists!')
            if not overwrite:
                savepath_ = verified_filepath(savepath_)

        msg_ = "overwriting" if overwrite else "saving new"
        qprint(f'>> ({save_file=}, {overwrite=})\n>> {msg_} file: "{savepath_.name}"')
    else:
        qprint(f">> note: {save_file = }")

    ## load raw data file
    df_raw = pd.read_csv(raw_fp, index_col=0, parse_dates=True)

    ## run auto qc script to clean data
    df = run_auto_qc(df_raw, site, q=q, q_xtra=q_xtra)

    qprint("\nQC complete!")
    if save_file:
        df.to_csv(savepath_)
        qprint(f'>> saved file: "{savepath_.name}"')

    return df


"""
PLOTS / FIGURES
"""
hvtemp = (
    "<b>%{fullData.name}</b><br>Value: %{y:.2f}" "<br><i>%{x|%Y-%m-%d %H:%M}</i><extra></extra>"
)


def qc_compare_fig(df0, df1, resample_=True, title_=None, hide_if_same=False, q=False):
    qprint = lambda msg, end_="\n": None if q else print(msg, end=end_)
    original_cols = list(df0.columns)
    clean_cols = [c for c in df1.columns if c != "PROCESSED"]
    removed_cols = [c for c in original_cols if c not in clean_cols]
    if removed_cols:
        qprint(f"Found column(s) removed during QC: {removed_cols = }")

    ## check for changes in existing columns
    commoncols = [c for c in original_cols if c in clean_cols]

    df00 = df0[commoncols].round(4).copy()
    df11 = df1[commoncols].round(4).copy()
    identical_files = False
    if df00.compare(df11).empty:
        if not removed_cols:
            identical_files = True
            qprint("NOTE: data files are identical; no changes made during QC.")
            if hide_if_same:
                qprint(">> exiting.")
                return
        else:
            qprint(f"NOTE: no changes made to data during QC.")

    qprint("")

    ## resample to 15-min for plot
    if resample_:
        df0_plot = df0.resample("15min").mean().copy()
        df1_plot = df1.resample("15min").mean().copy()
    else:
        df0_plot, df1_plot = df0.copy(), df1.copy()

    plot_freq = pd.infer_freq(df1_plot.index)

    n_rows = len(original_cols)
    strikethru = lambda c: f'<span style="text-decoration: line-through;">{c}</span>'
    s_titles = [c if c not in removed_cols else strikethru(c) for c in original_cols]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=s_titles,
        vertical_spacing=0.275 / n_rows,
    )
    ## common plot trace kwargs
    plot_kwargs = dict(
        x=df1_plot.index.copy(), mode="lines", showlegend=False, hovertemplate=hvtemp
    )

    ## compare/plot dfs and add shading for timestamps/ranges with removed data
    df0_ = df0.round(4).copy()  # used for comparison for ranges in shaded regions
    df1_ = df1[clean_cols].round(4).copy()  # used for comparison for ranges in shaded regions

    for i, col in enumerate(original_cols):
        ## add traces for original & new data (if exists)
        trace0 = go.Scatter(
            **plot_kwargs,
            y=df0_plot[col],
            name=f"{col}_ORIGINAL",
            line=dict(color="#333", dash="dot", width=1),
        )
        trace1 = go.Scatter(
            **plot_kwargs,
            y=df1_plot[col] if (col in clean_cols) else pd.Series(0),
            name=f"{col}_NEW",
        )

        if identical_files:
            fig.add_trace(trace1, row=i + 1, col=1)
            continue

        qprint(f"\n{col}")
        fig.add_trace(trace0, row=i + 1, col=1)
        fig.add_trace(trace1, row=i + 1, col=1)

        if col not in clean_cols:
            continue

        ## check for changes/removed data & add associated shaded regions to subplot
        ## using df0_ & df1_ (defined previously)
        df_cmp = df0_[[col]].compare(df1_[[col]])
        df_cmp.columns = df_cmp.columns.droplevel(0)  # drop level of multiindex columns
        if df_cmp.empty:
            continue

        ## add new column "time_diff" using diff() of index
        ##     >> creates a series where each value is a timedelta = (index[row] - index[row - 1])
        df_cmp["time_diff"] = df_cmp.index.diff()
        if df_cmp.shape[0] > 1:
            df_cmp.iloc[0, -1] = (
                df_cmp.index[1] - df_cmp.index[0]
            )  # time_diff for first row; value=NaT due to shift from diff() calc

        ## group rows using condition: [timedelta = 1min]  >>assumes standard query data file w/ freq '1min'; needs further testing for other freqs
        _1_MINUTE_ = pd.Timedelta(minutes=1)
        grp_conditions = df_cmp["time_diff"].eq(_1_MINUTE_)

        ## separate/break groups using cumulative sum of condition: [timedelta > 1min]
        breaks_ = df_cmp["time_diff"].gt(_1_MINUTE_).cumsum()

        ## use groupby function to separate df_cmp into continuous ranges according to the above conditions
        vrect_kwargs = dict(row=i + 1, line_width=0, opacity=0.4, layer="below")
        n_removed_pts = 0
        n_changed_pts = 0
        for g, dfg in df_cmp[grp_conditions].groupby(breaks_):
            grp_length = dfg.shape[0]
            if dfg["other"].isna().all():
                qctype_ = "REMOVED data range"
                vrect_color = "lightsalmon"
                n_removed_pts += grp_length
            elif dfg["other"].isna().any():
                qctype_ = "REMOVED/MODIFIED data"
                vrect_color = "navajowhite"
                n_removed_pts += grp_length
            else:
                qctype_ = "MODIFIED data range"
                vrect_color = "lightgreen"
                n_changed_pts += grp_length

                if grp_length < 240:  # exclude ranges below 4 hours
                    continue

            x0_, x1_ = dfg.index.min(), dfg.index.max()
            x0_plot = x0_.floor(plot_freq)
            x1_plot = x1_.ceil(plot_freq)
            fig.add_vrect(x0=x0_plot, x1=x1_plot, **vrect_kwargs, fillcolor=vrect_color)

            t_str = lambda dt_: str(dt_)[5:-3]
            qprint(f"  >> found {qctype_} (n={grp_length});".ljust(44), end_="")
            qprint(f'range=["{t_str(x0_)}", "{t_str(x1_)}"]  ({(x1_ - x0_)})')

    qprint("")

    ## set alignment for subplot titles (annotations in subplot figure)
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].update(font_size=12, x=1, xref="paper", xanchor="right")

    ## update layout props
    if n_rows <= 14:
        hgt_x = 100
    elif n_rows <= 27:
        hgt_x = 80
    else:
        hgt_x = 60

    fig.update_layout(
        font_size=9,
        height=hgt_x * n_rows,
        margin=dict(t=40, b=20, l=0, r=20),
        title=dict(
            font_size=14,
            text=f"<b>QC Comparison</b>" if (title_ is None) else f"<b>{title_}</b>",
            x=0.015,
        ),
        hoverlabel_font_size=11,
    )
    fig.update_yaxes(showline=True, linewidth=1, linecolor="#777")
    fig.update_xaxes(showline=True, linewidth=1, linecolor="#777")

    is_inv_data = all([("OE.ActivePower" in c) for c in original_cols])
    if is_inv_data:
        ymax_ = df1[original_cols].max().max() * 1.05
        fig.update_yaxes(range=[0, ymax_])
    else:
        for i, col in enumerate(original_cols):
            if col not in df1.columns:
                continue
            ymin_ = df1[col].min() * 0.95
            ymax_ = df1[col].max() * 1.05
            fig.update_yaxes(row=i + 1, range=[ymin_, ymax_])

    return fig


valid_index = lambda idx_: isinstance(idx_, pd.DatetimeIndex)


def df_comparison_subplots(df0_, df1_, resample_=True, freq_="15min", round_=None, q=True):
    qprint = lambda msg_: None if q else print(msg_)

    cols_to_compare = list(sorted([c for c in df0_.columns if c in df1_.columns]))
    if not cols_to_compare:
        qprint("no identical column names found to compare")
        return
    valid_indexes = valid_index(df0_.index) and valid_index(df1_.index)
    if not valid_indexes:
        qprint("incompatible index type(s)")
        return

    common_idx = df0_.loc[df0_.index.isin(df1_.index)].index
    if common_idx.empty:
        qprint("no shared timestamps found for comparison")
        return

    if pd.infer_freq(df0_.index) != pd.infer_freq(df1_.index):
        qprint("dataframes cannot have different frequencies.")
        return

    ## make indexes uniform & remove other cols (if exist)
    df0 = df0_.loc[df0_.index.isin(common_idx), cols_to_compare].copy()
    df1 = df1_.loc[df1_.index.isin(common_idx), cols_to_compare].copy()

    if isinstance(round_, int):
        df0 = df0.round(round_)
        df1 = df1.round(round_)

    ## get frequency info & resample (if specified)
    orig_freq = pd.infer_freq(df0.index)
    if not any(s.isnumeric() for s in orig_freq):
        orig_freq = "1" + orig_freq

    timedelta_ = pd.Timedelta(orig_freq)
    freq_mins = int(timedelta_.seconds / 60)
    pd_freq = f"{freq_mins}min"

    plot_freq = orig_freq
    df0_plot = df0.copy()
    df1_plot = df1.copy()
    if resample_:
        if not (freq_.endswith("min") or freq_.endswith("h")):
            plot_freq = "15min"
        else:
            plot_freq = freq_
        df0_plot = df0.resample(plot_freq).mean().copy()
        df1_plot = df1.resample(plot_freq).mean().copy()

    ## create subplot figure
    n_rows = len(cols_to_compare)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=cols_to_compare,
        vertical_spacing=0.275 / n_rows,
    )

    ## create comparison dataframe
    df_compare = df0.compare(df1)
    df_compare.columns = df_compare.columns.map("_".join)
    cols_with_diffs = list(
        sorted(set(c.replace("_self", "").replace("_other", "") for c in df_compare.columns))
    )

    ## common plot trace kwargs
    idx_ = df0_plot.index
    plot_kwargs = dict(x=idx_, showlegend=False, hovertemplate=hvtemp)
    for i, col in enumerate(cols_to_compare):
        trace_color = colors2_[i % len(colors2_)]

        ## set initial/default marker properties for new traces (using df)
        marker_1 = dict(
            line_color=pd.Series(trace_color, index=idx_),
            symbol=pd.Series("circle", index=idx_),
            size=pd.Series(0, index=idx_),
        )

        ## check for changes/removed data & add associated shaded regions to subplot
        if col in cols_with_diffs:  # if cols exist in comparison, have differences
            ccols_ = [f"{col}_self", f"{col}_other"]
            dfcc = df_compare[ccols_].dropna(how="all").copy()
            dfcc["time_diff"] = dfcc.index.diff()
            dfcc.iloc[0, -1] = dfcc.index[1] - dfcc.index[0]  # first row is na by default

            conditions_ = dfcc.time_diff.eq(timedelta_)
            breaks_ = dfcc.time_diff.gt(timedelta_).cumsum()
            groupby_object = dfcc.loc[conditions_].groupby(breaks_)
            grouped_df_list = [dfg for g, dfg in groupby_object]
            for dfg in grouped_df_list:
                x0_, x1_ = dfg.index.min(), dfg.index.max()
                x0_plot = x0_.floor(plot_freq)
                x1_plot = x1_.ceil(plot_freq)
                diffs_exist = idx_.isin(pd.date_range(x0_plot, x1_plot, freq=pd_freq))
                marker_1["line_color"].loc[diffs_exist] = "black"
                marker_1["symbol"].loc[diffs_exist] = "x-thin"
                marker_1["size"].loc[diffs_exist] = 6

        ## add traces for original & new data (if exists)
        trace0 = go.Scatter(
            **plot_kwargs,
            y=df0_plot[col],
            name=f"{col}_0",
            mode="lines",
            line=dict(color="#333", dash="dot", width=1),
        )
        trace1 = go.Scatter(
            **plot_kwargs,
            y=df1_plot[col],
            name=f"{col}_1",
            mode="markers+lines",
            line_color=trace_color,
            marker=marker_1,
        )
        fig.add_trace(trace0, row=i + 1, col=1)
        fig.add_trace(trace1, row=i + 1, col=1)

    ## set alignment for subplot titles (annotations in subplot figure)
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].update(font_size=12, x=1, xref="paper", xanchor="right")

    ## update layout props
    if n_rows <= 14:
        hgt_x = 100
    elif n_rows <= 27:
        hgt_x = 80
    else:
        hgt_x = 60

    fig.update_layout(
        font_size=9,
        height=hgt_x * n_rows,
        margin=dict(t=40, b=20, l=0, r=20),
        title=dict(font_size=14, text=f"<b>dataframe comparison</b>", x=0.015),
        hoverlabel_font_size=11,
    )
    fig.update_yaxes(showline=True, linewidth=1, linecolor="#777")
    fig.update_xaxes(showline=True, linewidth=1, linecolor="#777")

    return fig


# define common x-axes properties
x_rangeselector = dict(
    buttons=list(
        [
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(count=3, label="3d", step="day", stepmode="backward"),
            dict(count=7, label="1w", step="day", stepmode="backward"),
            dict(step="all"),
        ]
    ),
    activecolor="#D1DDEB",
    xanchor="right",
    x=1.003,
    yanchor="bottom",
    y=1.004,
)
x_tickfmt = [
    dict(dtickrange=[None, 7200000], value="%m/%d %H:%M:%S"),
    dict(dtickrange=[7200000, 43200000], value="%m/%d %H:%M"),
    dict(dtickrange=[43200000, None], value="%b-%d"),
]
x_minor = dict(
    ticklen=1,
    griddash="dot",
    gridcolor="white",
)


## figure for qc subplots (originally from </dashoeapp/utils/qctools.py>)
def qcplot_timeseries(df_raw, freq, is_rawdata=False):
    dfcols = list(df_raw.columns)
    is_invdata = is_metdata = False
    if is_rawdata:
        invcols = [c for c in dfcols if "OE.ActivePower" in c]
        is_invdata = len(invcols) > 0
        is_metdata = len(group_met_columns(dfcols)) > 0

    df = df_raw.copy()

    # resample to selected freq
    if pd.infer_freq(df.index) != freq:
        df = df.resample(freq).mean()

    # default number of rows is 2 (cleaned & raw/original)
    n_groups = 1
    dfcols = list(df.columns)
    column_row_dict = {c: 1 for c in dfcols}  # default, not met data
    plot_titles = ["Cleaned data", "Raw data (original)"]

    # if met station data, group columns by sensor type
    if is_metdata:
        met_cols = list(df.columns)
        col_groups = group_met_columns(met_cols)
        all_grouped_cols = [
            c for grp in col_groups.values() for c in grp
        ]  # flattened list of lists (dict vals)
        if len(met_cols) == len(all_grouped_cols):  # if all cols were successfully grouped
            plot_groups = list(col_groups.keys())
            plot_titles = [f"{grp} data" for grp in plot_groups] + ["Raw data (original)"]
            n_groups = len(plot_groups)
            column_row_dict = {}
            for i, cols in enumerate(col_groups.values()):
                for c in cols:
                    column_row_dict[c] = i + 1
            col_group_dict = {c: plot_groups[rw - 1] for c, rw in column_row_dict.items()}
        else:
            is_metdata = False

    # if inverter data, separate into 4 groups
    if is_invdata:
        inv_cols = list(df.columns)
        col_groups = group_inv_columns(inv_cols)
        plot_groups = list(col_groups.keys())
        n_groups = len(plot_groups)
        plot_titles = [f"Inverters ({grp})" for grp in plot_groups] + ["Raw data (original)"]
        column_row_dict = {}
        for i, cols in enumerate(col_groups.values()):
            for c in cols:
                column_row_dict[c] = i + 1
        col_group_dict = {c: plot_groups[rw - 1] for c, rw in column_row_dict.items()}

    row_heights = [0.925 / n_groups] * n_groups + [0.075]
    n_rows = n_groups + 1

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.04,
        subplot_titles=plot_titles,
    )

    # get plotly default colors & use to ensure subplot trace colors match
    pltcolors = plotly.colors.qualitative.Dark24
    c_num = len(pltcolors)
    dfcols = list(df.columns)
    colorlist = [pltcolors[n % c_num] for n in range(len(dfcols))]
    colcolors = dict(zip(dfcols, colorlist))

    qc_kwargs = dict(
        mode="markers+lines",
        marker=dict(size=1, opacity=0, line_width=1),
        selected=dict(marker=dict(size=7, opacity=1)),
        hovertemplate="%{x|%m/%d %H:%M:%S}<br>Value: %{y:.2f}",
    )

    for c in df.columns:
        col_kwargs = dict(x=df.index, y=df[c], name=c, line_color=colcolors[c])
        qckwgs_ = col_kwargs | qc_kwargs
        if is_metdata:
            c_grp = col_group_dict[c]
            qckwgs_ = qckwgs_ | dict(
                legendgroup=c_grp, legendgrouptitle_text=c_grp, legendrank=column_row_dict[c]
            )

        if is_invdata:
            c_grp = col_group_dict[c]
            qckwgs_ = qckwgs_ | dict(
                legendgroup=c_grp, legendgrouptitle_text=c_grp, legendrank=column_row_dict[c]
            )

        qc_trace = go.Scattergl(**qckwgs_)
        fig.add_trace(qc_trace, row=column_row_dict[c], col=1)

        raw_trace = go.Scattergl(
            **col_kwargs, mode="lines", line_width=1, hoverinfo="skip", showlegend=False
        )
        fig.add_trace(raw_trace, row=n_rows, col=1)

    # format subplot titles (are actually annotations in subplot figures)
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].update(font=dict(size=14), x=0, xanchor="left")

    fig.update_xaxes(
        ticks="outside",
        ticklen=2,
        tickformatstops=x_tickfmt,
        minor=x_minor,
        rangeselector=x_rangeselector,
        # range=[df.index.min(), df.index.max()],
        # type='date',
    )

    fig.update_xaxes(
        rangeslider=dict(
            visible=True,
            thickness=0.02,
        ),
        row=n_rows,
        col=1,
    )

    fig.update_layout(
        font_size=9,
        dragmode="zoom",
        margin=dict(t=60, b=15, l=20, r=20),
        legend=dict(
            x=1.01,
            xanchor="left",
            groupclick="toggleitem",
            grouptitlefont=dict(size=11),
            tracegroupgap=16,
        ),
        plot_bgcolor="#F0F4FA",
    )

    return fig
