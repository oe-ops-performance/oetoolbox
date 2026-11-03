"""Tests for the POA sensor assessment engine (synthetic data; no site database needed)."""

import numpy as np
import pandas as pd
import pytest
from pvlib.location import Location

from oetoolbox.datatools import solarqc2 as sq

LOCATION = Location(36.14, -119.56, "US/Pacific", 120, "synthetic")
TRACKER_DESIGN = dict(has_trackers=True, tilt=45, azimuth=180.0, gcr=0.34)
FIXED_DESIGN = dict(has_trackers=False, tilt=25, azimuth=180.0, gcr=None)
CFG = sq.POAQCConfig()
STEP = pd.Timedelta(CFG.analysis_freq)
GAIN = 1.05


@pytest.fixture(scope="module")
def world():
    """8 days of 1-minute truth: clear days alternate with cloudy days."""
    index = pd.date_range("2026-09-01", "2026-09-08 23:59", freq="1min", tz="Etc/GMT+7")
    grid = index.floor(STEP).unique()
    geoms, solpos = sq.build_geometries(LOCATION, grid, TRACKER_DESIGN, STEP, CFG.stow_rotation_step)
    rng = np.random.default_rng(0)
    csi = pd.Series(1.0, index=grid)
    for day in range(8):
        if day % 2:  # cloudy day: smooth random walk between 0.2 and 1
            idx = csi.index[csi.index.date == pd.Timestamp("2026-09-01").date() + pd.Timedelta(days=day)]
            walk = np.clip(0.6 + np.cumsum(rng.normal(0, 0.12, len(idx))), 0.2, 1.0)
            csi.loc[idx] = walk
    to_min = lambda s: s.reindex(index, method="ffill")
    truth = {g: to_min(geoms[g] * csi * GAIN) for g in geoms.columns}
    meter = to_min(np.minimum(geoms[sq.TRACKER] * csi, 800.0) / 1000.0)  # clips like a real plant
    # DTN: hourly mean of the tracker POA (hour-beginning)
    dtn = (geoms[sq.TRACKER] * csi).resample("1h").mean().rename("poa_global")
    return dict(index=index, truth=truth, meter=meter, dtn=dtn, geoms=geoms)


def assess(sensors, world, **kwargs):
    kwargs.setdefault("ref_meter", world["meter"])
    kwargs.setdefault("config", CFG)
    return sq.assess_poa_sensors(
        pd.DataFrame(sensors), world["dtn"], LOCATION, TRACKER_DESIGN, **kwargs
    )


def daylight_share(result, col, flag, days):
    a = result["analysis"]
    lab = result["sensors"][col]["label"]
    sel = a["use"] & np.isin(a.index.date, [pd.Timestamp(f"2026-09-0{d}").date() for d in days])
    return lab[sel].eq(flag).mean()


def test_healthy_sensor_is_not_flagged_on_clear_or_cloudy_days(world):
    res = assess({"OE.POA_A": world["truth"][sq.TRACKER]}, world)
    assert daylight_share(res, "OE.POA_A", "ok", range(1, 9)) > 0.93
    assert res["sensors"]["OE.POA_A"]["gain"] == pytest.approx(GAIN, abs=0.05)


CLEAR_DAYS = (1, 3, 5, 7)  # the synthetic cloudy days are Sep 2, 4, 6, 8


def on_days(index, days):
    return np.isin(index.date, [pd.Timestamp(f"2026-09-0{d}").date() for d in days])


def test_sensor_stowed_flat_is_flagged_stow_on_those_days_only(world):
    stowed = world["truth"][sq.TRACKER].copy()
    on = on_days(stowed.index, (3, 5))
    stowed[on] = world["truth"]["rot+0"][on]  # flat: differs from tracking morning and evening
    res = assess({"OE.POA_A": world["truth"][sq.TRACKER], "OE.POA_B": stowed}, world)
    assert daylight_share(res, "OE.POA_B", "stow", (3, 5)) > 0.85
    assert daylight_share(res, "OE.POA_B", "stow", (1, 7)) < 0.05
    assert daylight_share(res, "OE.POA_A", "stow", range(1, 9)) < 0.02


def test_stuck_sensor_is_stow_while_it_differs_from_tracking(world):
    """Stuck at the end-of-day angle: wrong in the morning, matches tracking by mid-afternoon."""
    s = world["truth"][sq.TRACKER].copy()
    on = on_days(s.index, (3, 5))
    s[on] = world["truth"]["rot-45"][on]  # faces west: reads very low in the morning
    res = assess({"OE.POA_B": s}, world)
    lab, a = res["sensors"]["OE.POA_B"]["label"], res["analysis"]
    morning = a["use"] & (a.index.hour < 12) & on_days(a.index, (3, 5))
    assert lab[morning].eq("stow").mean() > 0.85
    assert daylight_share(res, "OE.POA_B", "stow", (1, 7)) < 0.05


def test_morning_only_stow_is_flagged_only_in_the_morning(world):
    s = world["truth"][sq.TRACKER].copy()
    morning = on_days(s.index, (1,)) & (s.index.hour < 12)
    s[morning] = world["truth"]["rot-45"][morning]
    res = assess({"OE.POA_A": s}, world)
    lab, a = res["sensors"]["OE.POA_A"]["label"], res["analysis"]
    day1 = on_days(a.index, (1,))
    assert lab[day1 & (a.index.hour < 10) & a["use"]].eq("stow").mean() > 0.8
    assert lab[day1 & (a.index.hour >= 14) & a["use"]].eq("stow").mean() < 0.1


def test_overcast_day_geometry_is_not_judged(world):
    """Under thick cloud light is diffuse, so a stowed sensor reads about right -- no flag."""
    s = world["truth"][sq.TRACKER].copy()
    on = on_days(s.index, (4,))
    s[on] = world["truth"]["rot+0"][on]
    res = assess({"OE.POA_A": s}, world)
    assert daylight_share(res, "OE.POA_A", "bad", (4,)) < 0.1


def test_meter_outage_does_not_condemn_a_healthy_sensor(world):
    meter = world["meter"].copy()
    meter[meter.index.date == pd.Timestamp("2026-09-03").date()] = 0.0  # meter reads dead all day
    res = assess({"OE.POA_A": world["truth"][sq.TRACKER]}, world, ref_meter=meter)
    assert daylight_share(res, "OE.POA_A", "ok", range(1, 9)) > 0.9
    assert daylight_share(res, "OE.POA_A", "bad", (3,)) < 0.05


def test_works_without_a_meter(world):
    res = assess({"OE.POA_A": world["truth"][sq.TRACKER]}, world, ref_meter=None)
    assert daylight_share(res, "OE.POA_A", "ok", CLEAR_DAYS) > 0.9


def test_sensor_reading_far_below_everything_is_bad_not_ok(world):
    s = world["truth"][sq.TRACKER] * 0.3  # fits no geometry (and not a stow rotation)
    res = assess({"OE.POA_A": s}, world)
    flagged = 1 - daylight_share(res, "OE.POA_A", "ok", CLEAR_DAYS)
    assert flagged > 0.7  # stow or bad; either way it is not representative of the array


def test_steady_fraction_of_expected_is_not_a_stow(world):
    """30% of expected all day matches *different* rotations at different hours -> not one stow."""
    res = assess({"OE.POA_A": world["truth"][sq.TRACKER] * 0.3}, world)
    assert daylight_share(res, "OE.POA_A", "stow", CLEAR_DAYS) < 0.02
    assert daylight_share(res, "OE.POA_A", "bad", CLEAR_DAYS) > 0.7


def test_stow_run_reports_one_geometry(world):
    s = world["truth"][sq.TRACKER].copy()
    on = on_days(s.index, (3,))
    s[on] = world["truth"]["rot+0"][on]
    res = assess({"OE.POA_A": s}, world)["sensors"]["OE.POA_A"]
    assert set(res["geometry"].dropna().unique()) == {"rot+0"}


def test_bad_afternoon_cannot_hide_behind_good_morning():
    idx = pd.date_range("2026-09-01 08:00", "2026-09-01 17:55", freq="5min", tz="Etc/GMT+7")
    pm = pd.Series(idx >= pd.Timestamp("2026-09-01 13:00", tz="Etc/GMT+7"), index=idx)
    score = pd.DataFrame({sq.TRACKER: 5.0, "rot+0": np.where(pm, 1.6, 0.2)}, index=idx)
    lab = pd.Series("stow", index=idx, dtype=object)
    # whole-run mean is 0.8 (would pass); judged per half the afternoon fails
    assert score["rot+0"].mean() <= 1.0
    out, geom = sq._enforce_single_geometry(lab, score, 1.0, pm, min_bins=6)
    assert out[~pm].eq("stow").all() and geom[~pm].eq("rot+0").all()
    assert out[pm].eq("bad").all()


def test_short_half_follows_the_other_half():
    idx = pd.date_range("2026-09-01 08:00", "2026-09-01 13:10", freq="5min", tz="Etc/GMT+7")
    pm = pd.Series(idx >= pd.Timestamp("2026-09-01 13:00", tz="Etc/GMT+7"), index=idx)  # 3 pm bins
    score = pd.DataFrame({sq.TRACKER: 5.0, "rot+0": np.where(pm, 3.0, 0.2)}, index=idx)
    lab = pd.Series("stow", index=idx, dtype=object)
    out, geom = sq._enforce_single_geometry(lab, score, 1.0, pm, min_bins=6)
    assert out.eq("stow").all()  # the 15-min tail is not judged on its own noise


def test_backfill_has_no_dip_when_sensor_converges_after_stow(world):
    """Morning stow removed, afternoon sensor still reading ~12% low as it converges onto tracking:
    that biased seam data must not drag the backfilled midday level down."""
    from types import SimpleNamespace

    truth = world["truth"][sq.TRACKER]
    s = truth.copy()
    day = on_days(s.index, (3,))
    morning = day & (s.index.hour < 13)
    s[morning] = world["truth"]["rot-45"][morning]
    converge = day & (s.index.hour >= 13) & (s.index.hour < 16)
    ramp = 0.88 + 0.12 * ((s.index.hour[converge] - 13) * 60 + s.index.minute[converge]) / 180
    s[converge] = s[converge] * ramp

    res = assess({"OE.POA_A": s}, world)
    a, lab = res["analysis"], res["sensors"]["OE.POA_A"]["label"]
    kept = a["OE.POA_A"].where(~lab.isin(["stow", "bad"]))
    stub = SimpleNamespace(
        config=CFG, analysis_freq=STEP, analysis=a, gains={"OE.POA_A": res["sensors"]["OE.POA_A"]["gain"]}
    )
    model = sq.POADataset._anchored_model(stub, kept)

    truth_bins = truth.resample(STEP).mean().reindex(a.index)
    noon = on_days(a.index, (3,)) & (a.index.hour >= 11) & (a.index.hour < 14)
    err = (model[noon] / truth_bins[noon] - 1).abs()
    assert err.max() < 0.08


def _labels(spec):
    idx = pd.date_range("2026-09-01 08:00", periods=len(spec), freq="5min", tz="Etc/GMT+7")
    return pd.Series(list(spec), index=idx, dtype=object), idx


def test_trim_extends_stow_while_sensor_reads_low_after_a_run():
    spec = ["stow"] * 6 + ["ok"] * 20
    lab, idx = _labels(spec)
    geom = pd.Series(np.where(lab.eq("stow"), "rot-45", None), index=idx, dtype=object)
    low = pd.Series([False] * 6 + [True] * 5 + [False] * 15, index=idx)  # converges after 5 bins
    out, g = sq._trim_stow_transitions(lab, geom, low, max_bins=24)
    assert out.eq("stow").sum() == 11 and out.iloc[10] == "stow" and out.iloc[11] == "ok"
    assert g.iloc[10] == "rot-45"


def test_trim_extends_backward_into_stow_and_respects_the_cap():
    lab, idx = _labels(["ok"] * 20 + ["stow"] * 6)
    geom = pd.Series(np.where(lab.eq("stow"), "rot+0", None), index=idx, dtype=object)
    low = pd.Series([True] * 26, index=idx)  # never converges: only the cap stops it
    out, _ = sq._trim_stow_transitions(lab, geom, low, max_bins=4)
    assert out.eq("stow").sum() == 6 + 4 and out.iloc[16:].eq("stow").all()


def test_trim_never_touches_bins_that_are_not_ok():
    lab, idx = _labels(["stow"] * 4 + ["bad"] * 2 + ["ok"] * 6)
    geom = pd.Series(np.where(lab.eq("stow"), "rot+0", None), index=idx, dtype=object)
    out, _ = sq._trim_stow_transitions(lab, geom, pd.Series(True, index=idx), max_bins=12)
    assert out.eq("bad").sum() == 2 and out.iloc[6:].eq("ok").all()


def test_sensor_converging_after_stow_is_trimmed_but_reading_high_is_not(world):
    truth = world["truth"][sq.TRACKER]

    def run(start_factor):
        s = truth.copy()
        day = on_days(s.index, (3,))
        morning = day & (s.index.hour < 13)
        s[morning] = world["truth"]["rot-45"][morning]
        conv = day & (s.index.hour >= 13) & (s.index.hour < 16)
        mins = (s.index.hour[conv] - 13) * 60 + s.index.minute[conv]
        s[conv] = s[conv] * (start_factor + (1 - start_factor) * mins / 180)
        res = assess({"OE.POA_A": s}, world)
        lab, a = res["sensors"]["OE.POA_A"]["label"], res["analysis"]
        return lab, a

    lab, a = run(0.85)  # reads 15% low at 13:00, recovers by 16:00
    early = on_days(a.index, (3,)) & (a.index.hour == 13)
    late = on_days(a.index, (3,)) & (a.index.hour >= 15) & (a.index.hour < 17)
    assert lab[early].eq("stow").mean() > 0.8  # transition belongs to the stow
    assert lab[late].eq("stow").mean() < 0.1  # converged: kept

    lab_hi, a = run(1.15)  # reads high instead: reference error, not a transition
    early = on_days(a.index, (3,)) & (a.index.hour == 13)
    assert lab_hi[early].eq("stow").mean() < 0.3


def test_ambiguous_window_where_tracking_fits_needs_stow_on_both_sides():
    lab, idx = _labels(["ok"] * 4 + ["amb"] * 4 + ["stow"] * 4)
    fits = pd.Series(True, index=idx)
    out = sq._fill_ambiguous(lab, pd.Timedelta("3h"), fits)
    assert out.iloc[4:8].eq("ok").all()  # a later stow must not leak backward over a good sensor

    lab2, _ = _labels(["stow"] * 4 + ["amb"] * 4 + ["stow"] * 4)
    assert sq._fill_ambiguous(lab2, pd.Timedelta("3h"), fits).iloc[4:8].eq("stow").all()  # bridged

    out3 = sq._fill_ambiguous(lab, pd.Timedelta("3h"), ~fits)  # tracking fails: nearest known label
    assert out3.iloc[4:8].tolist() == ["ok", "ok", "stow", "stow"]  # split at the midpoint


def test_low_sun_excess_is_not_flagged_but_midday_fault_still_is(world):
    """A sensor reading well above both references near sunset is model/plane detail, not a fault."""
    truth = world["truth"][sq.TRACKER]
    clear = world["geoms"][sq.TRACKER].reindex(truth.index, method="ffill")
    evening = (truth.index.hour >= 12) & (clear < 0.45 * clear.groupby(truth.index.date).transform("max"))
    s = truth.copy()
    s[evening] = s[evening] * 1.6
    res = assess({"OE.POA_A": s}, world)
    lab, a = res["sensors"]["OE.POA_A"]["label"], res["analysis"]
    pm_low = a["use"] & (a.index.hour >= 12) & (a["cs_tracker"] < 0.45 * a["cs_tracker"].groupby(a.index.date).transform("max"))
    assert lab[pm_low & on_days(a.index, CLEAR_DAYS)].eq("bad").mean() < 0.05

    dead_midday = truth.copy()
    midday = (dead_midday.index.hour >= 10) & (dead_midday.index.hour < 14)
    dead_midday[midday] = dead_midday[midday] * 0.3
    res2 = assess({"OE.POA_A": dead_midday}, world)
    lab2, a2 = res2["sensors"]["OE.POA_A"]["label"], res2["analysis"]
    mid = a2["use"] & (a2.index.hour >= 11) & (a2.index.hour < 13) & on_days(a2.index, CLEAR_DAYS)
    assert (~lab2[mid].isin(["ok"])).mean() > 0.7  # still caught (bad or stow)


def weather_frame(world, wind=2.0, sunshine=58.0, precip=0.0):
    idx = world["dtn"].index
    return pd.DataFrame(
        {"dtn_wind_speed": wind, "sunshineDuration": sunshine, "precipAmount": precip, "unused": 1.0},
        index=idx,
    )


def plant_stowed_world(world, day=3):
    """The meter itself follows a flat (stowed) plant for one clear day; DTN still says clear sky."""
    w = dict(world)
    meter = world["meter"].copy()
    on = on_days(meter.index, (day,))
    flat = world["truth"]["rot+0"] / GAIN
    meter[on] = (np.minimum(flat, 800.0) / 1000.0)[on]
    w["meter"] = meter
    return w


def meter_stow_share(res, day=3):
    a = res["analysis"]
    sel = a["use"] & on_days(a.index, (day,))
    return a["meter_stow"][sel].mean()


def test_jagged_collapse_is_clouds_not_a_stow(world):
    """Output that follows a fixed geometry but swings minute to minute is sky, not a stowed plant."""
    w = plant_stowed_world(world)
    meter = w["meter"].copy()
    day = on_days(meter.index, (3,))
    rng = np.random.default_rng(1)
    meter[day] = meter[day] * np.exp(rng.normal(0, 0.35, day.sum()))  # cloud-like 1-minute swings
    w["meter"] = meter
    res = assess({"OE.POA_A": world["truth"][sq.TRACKER]}, w, ref_weather=weather_frame(world, wind=2.0))
    assert meter_stow_share(res) == 0.0


def test_roughness_limit_is_configurable(world):
    from dataclasses import replace

    w = plant_stowed_world(world)
    meter = w["meter"].copy()
    day = on_days(meter.index, (3,))
    meter[day] = meter[day] * np.exp(np.random.default_rng(1).normal(0, 0.05, day.sum()))
    w["meter"] = meter
    s = {"OE.POA_A": world["truth"][sq.TRACKER]}
    assert meter_stow_share(assess(s, w)) == 0.0  # 5% minute noise is already above a stow's smoothness
    lax = assess(s, w, config=replace(CFG, stow_max_roughness=0.5))
    assert meter_stow_share(lax) > 0.3


def test_robust_roughness_separates_smooth_from_jagged():
    idx = pd.date_range("2026-09-03 10:00", periods=240, freq="1min", tz="Etc/GMT+7")
    smooth = pd.Series(800 + 0.1 * np.arange(240), index=idx)
    jagged = smooth * np.exp(np.random.default_rng(0).normal(0, 0.3, 240))
    assert sq._robust_roughness(smooth) < 0.001 < 0.1 < sq._robust_roughness(jagged)
    assert np.isnan(sq._robust_roughness(pd.Series(0.0, index=idx)))


def test_plant_wide_stow_is_detected_without_weather_data(world):
    w = plant_stowed_world(world)
    res = assess({"OE.POA_A": world["truth"][sq.TRACKER]}, w)
    assert meter_stow_share(res) > 0.5


def test_plant_stow_in_calm_fair_weather_is_still_a_stow_by_default(world):
    """Wind is advisory: a meter that follows a fixed geometry in fair weather is unusable either way."""
    w = plant_stowed_world(world)
    s = {"OE.POA_A": world["truth"][sq.TRACKER]}
    calm = assess(s, w, ref_weather=weather_frame(world, wind=2.0))
    assert meter_stow_share(calm) > 0.5
    assert not calm["analysis"].loc[calm["analysis"]["meter_stow"], "windy"].any()  # reported as not windy


def test_stow_requires_wind_when_configured(world):
    from dataclasses import replace

    w = plant_stowed_world(world)
    s = {"OE.POA_A": world["truth"][sq.TRACKER]}
    cfg = replace(CFG, stow_requires_wind=True)
    calm = assess(s, w, ref_weather=weather_frame(world, wind=2.0), config=cfg)
    windy = assess(s, w, ref_weather=weather_frame(world, wind=12.0), config=cfg)
    assert meter_stow_share(calm) == 0.0
    assert meter_stow_share(windy) > 0.5


def test_meter_stow_is_excluded_from_the_cloud_index_used_for_backfill(world):
    """The model must not inherit a stowed meter: its cloud index falls back to DTN there."""
    w = plant_stowed_world(world)
    res = assess({"OE.POA_A": world["truth"][sq.TRACKER]}, w, ref_weather=weather_frame(world, wind=2.0))
    a = res["analysis"]
    stowed = a["meter_stow"] & a["use"] & (a["cs_tracker"] > 400)
    assert stowed.sum() > 20
    assert (a.loc[stowed, "csi_primary"] - a.loc[stowed, "csi_dtn"]).abs().max() < 1e-9
    assert a.loc[stowed, "csi_primary"].median() > 0.9  # ~clear sky, not the stowed meter's lower index


@pytest.mark.parametrize("storm", [dict(sunshine=15.0), dict(precip=0.8)])
def test_storm_is_not_a_stow_even_when_windy(world, storm):
    w = plant_stowed_world(world)
    s = {"OE.POA_A": world["truth"][sq.TRACKER]}
    res = assess(s, w, ref_weather=weather_frame(world, wind=12.0, **storm))
    assert meter_stow_share(res) == 0.0


def test_weather_without_wind_column_still_vetoes_storms(world):
    w = plant_stowed_world(world)
    s = {"OE.POA_A": world["truth"][sq.TRACKER]}
    wx = weather_frame(world, sunshine=15.0).drop(columns="dtn_wind_speed")
    assert meter_stow_share(assess(s, w, ref_weather=wx)) == 0.0
    clear = weather_frame(world).drop(columns="dtn_wind_speed")
    assert meter_stow_share(assess(s, w, ref_weather=clear)) > 0.5


def test_missing_weather_values_never_veto(world):
    w = plant_stowed_world(world)
    s = {"OE.POA_A": world["truth"][sq.TRACKER]}
    wx = weather_frame(world, wind=np.nan, sunshine=np.nan, precip=np.nan)
    assert meter_stow_share(assess(s, w, ref_weather=wx)) > 0.5


def test_weather_frame_with_no_known_columns_is_rejected(world):
    s = {"OE.POA_A": world["truth"][sq.TRACKER]}
    with pytest.raises(ValueError):
        assess(s, world, ref_weather=pd.DataFrame({"nope": 1.0}, index=world["dtn"].index))


def test_sensor_level_stow_is_not_gated_by_weather(world):
    """A stuck tracker is mechanical: it is flagged even in calm, sunny weather."""
    s = world["truth"][sq.TRACKER].copy()
    on = on_days(s.index, (3, 5))
    s[on] = world["truth"]["rot+0"][on]
    res = assess({"OE.POA_B": s}, world, ref_weather=weather_frame(world, wind=1.0))
    assert daylight_share(res, "OE.POA_B", "stow", (3, 5)) > 0.85


def _anchor_stub(cfg, anchor_ratio=1.1):
    """Two clean days (sensor/model = 1.0), then a day whose sensor is missing until 15:00 and reads
    ``anchor_ratio`` x the model afterwards. Returns (modelled POA, base, day-3 index mask)."""
    from types import SimpleNamespace

    idx = pd.date_range("2026-09-01", "2026-09-03 23:55", freq="5min", tz="Etc/GMT+7")
    hour = idx.hour + idx.minute / 60
    base = pd.Series(np.where((hour >= 8) & (hour < 18), 900.0, 0.0), index=idx)
    analysis = pd.DataFrame({"cs_tracker": base, "csi_primary": 1.0, "use": base > 0})
    kept = base.copy()
    day3 = idx.date == pd.Timestamp("2026-09-03").date()
    kept[day3 & (hour < 15)] = np.nan
    kept[day3 & (hour >= 15)] = base[day3 & (hour >= 15)] * anchor_ratio
    stub = SimpleNamespace(config=cfg, analysis_freq=STEP, analysis=analysis, gains={"S": 1.0})
    return sq.POADataset._anchored_model(stub, kept), base, day3 & (hour >= 8) & (hour < 18)


def test_backfill_gain_ramps_to_the_anchor_instead_of_stepping():
    """The old anchor was held for a fixed reach then dropped to the default, leaving a step (and so a
    hump in the filled trace) where it ended."""
    from dataclasses import replace

    cfg = replace(CFG, anchor_max_adjust=0.2)
    model, base, day = _anchor_stub(cfg)
    gain = (model / base.where(base > 0))[day]
    assert gain.diff().abs().max() < 0.012  # 10% spread over the 1 h fade, 5-min bins
    gap = gain[(gain.index.hour + gain.index.minute / 60) < 14]
    assert (gap - 1.0).abs().max() < 1e-9  # far from the data: the default gain
    assert gain.iloc[-1] == pytest.approx(1.1, abs=0.02)  # at the data: anchored to the sensor


def test_an_abrupt_anchor_cutoff_is_what_makes_the_step():
    from dataclasses import replace

    abrupt = replace(CFG, anchor_max_adjust=0.2, anchor_reach="5min")
    model, base, day = _anchor_stub(abrupt)
    gain = (model / base.where(base > 0))[day]
    assert gain.diff().abs().max() > 0.05


def test_anchor_uses_peak_sun_data_only():
    """A low-sun ratio (here 1.4) must not become the backfill gain."""
    from dataclasses import replace

    cfg = replace(CFG, anchor_max_adjust=0.5)
    idx = pd.date_range("2026-09-01", "2026-09-02 23:55", freq="5min", tz="Etc/GMT+7")
    hour = idx.hour + idx.minute / 60
    base = pd.Series(np.where((hour >= 8) & (hour < 18), np.where((hour >= 8) & (hour < 16.5), 900.0, 300.0), 0.0), index=idx)
    analysis = pd.DataFrame({"cs_tracker": base, "csi_primary": 1.0, "use": base > 0})
    kept = base.copy()
    day2 = idx.date == pd.Timestamp("2026-09-02").date()
    kept[day2 & (hour < 14)] = np.nan
    kept[day2 & (hour >= 16.5)] *= 1.4  # low-sun excess (base 300 = a third of peak)
    from types import SimpleNamespace

    stub = SimpleNamespace(config=cfg, analysis_freq=STEP, analysis=analysis, gains={"S": 1.0})
    model = sq.POADataset._anchored_model(stub, kept)
    gap = day2 & (hour >= 9) & (hour < 13)
    assert (model[gap] / base[gap]).max() == pytest.approx(1.0, abs=1e-6)


def _stowed_flat_sensor(world, days=(3,)):
    s = world["truth"][sq.TRACKER].copy()
    on = on_days(s.index, days)
    s[on] = world["truth"]["rot+0"][on]
    return s


def test_shape_finds_a_stow_that_the_dtn_reference_hides(world):
    """DTN's hourly index is uncertain by about the stow-vs-tracking difference. Here it even bends
    to the stowed sensor's bell, so tracking 'fits'; the sensor's own shape still gives it away."""
    from dataclasses import replace

    s = _stowed_flat_sensor(world)
    hidden = dict(world)
    hidden["dtn"] = (s / GAIN).resample("1h").mean().rename("poa_global")
    on = assess({"OE.POA_A": s}, hidden, ref_meter=None)
    off = assess({"OE.POA_A": s}, hidden, ref_meter=None, config=replace(CFG, shape_max_rmse=0.0))
    assert daylight_share(on, "OE.POA_A", "stow", (3,)) > 0.85
    assert daylight_share(off, "OE.POA_A", "stow", (3,)) < 0.2  # the references alone are fooled


def _solar_noon(solpos):
    el = solpos["apparent_elevation"]
    return el.groupby(el.index.date).transform(lambda e: e.idxmax())  # time of peak sun, per day


def _shape_halves(world, sensor):
    res = assess({"S": sensor}, world, ref_meter=None)
    a = res["analysis"]
    geoms = a[[c for c in a.columns if c.startswith("cs_")]].rename(columns=lambda c: c[3:])
    noon = _solar_noon(res["solpos"])
    afternoon = pd.Series(a.index >= noon.to_numpy(), index=a.index)
    flag, geom = sq._shape_stow_halves(a["S"], geoms, a["use"], CFG, afternoon)
    return a, flag, geom, noon


def test_shape_test_fires_on_smooth_stowed_days_only(world):
    a, flag, geom, _ = _shape_halves(world, _stowed_flat_sensor(world, (3, 4)))
    use = a["use"]
    assert flag[use & on_days(a.index, (3,))].all()  # clear day, flat stow
    assert set(geom[use & on_days(a.index, (3,))].dropna()) == {"rot+0"}
    assert not flag[use & on_days(a.index, (4,))].any()  # cloudy day: shape fit is meaningless
    assert not flag[use & on_days(a.index, (1, 5, 7))].any()  # tracking days


def test_shape_test_judges_morning_and_afternoon_separately(world):
    # solar noon on the analysis grid (peak sun elevation), from the same call the engine makes
    probe = assess({"S": world["truth"][sq.TRACKER]}, world, ref_meter=None)
    noon = _solar_noon(probe["solpos"])[pd.Timestamp("2026-09-03 12:00", tz="Etc/GMT+7")]
    s = world["truth"][sq.TRACKER].copy()
    am = on_days(s.index, (3,)) & (s.index < noon)
    s[am] = world["truth"]["rot-45"][am]  # stuck until solar noon, tracking after
    a, flag, _, _ = _shape_halves(world, s)
    day = a["use"] & on_days(a.index, (3,))
    assert flag[day & (a.index < noon)].all()
    assert not flag[day & (a.index >= noon)].any()


def test_without_a_meter_cloudy_windows_are_not_judged_against_hourly_dtn(world):
    """A healthy tracker under mild minute-scale cloud jaggedness, hourly DTN with its own large error.
    (Heavier jaggedness already widens the sensor's own tolerance, so it is the mild case that is
    misjudged.)"""
    from dataclasses import replace

    rng = np.random.default_rng(5)
    truth = world["truth"][sq.TRACKER]
    cloudy = on_days(truth.index, (2, 4, 6, 8))
    blocks = pd.Series(np.exp(rng.normal(0, 0.05, len(truth.index[::5]))), index=truth.index[::5])
    jag = blocks.reindex(truth.index, method="ffill")
    sensor = truth.where(~cloudy, truth * jag)  # jagged on cloudy days, untouched on clear days
    noisy = dict(world)
    d = world["dtn"].copy()
    hourly_cloudy = on_days(d.index, (2, 4, 6, 8))
    d[hourly_cloudy] = d[hourly_cloudy] * np.exp(rng.normal(0, 0.45, hourly_cloudy.sum()))
    noisy["dtn"] = d
    s = {"OE.POA_A": sensor}

    def flagged(cfg, days):
        res = assess(s, noisy, ref_meter=None, config=cfg)
        return sum(daylight_share(res, "OE.POA_A", f, days) for f in ("stow", "bad"))

    unguarded = flagged(replace(CFG, dtn_only_max_curvature=None), (2, 4, 6, 8))
    assert unguarded > 0.12  # without the guard, hourly DTN wrongly condemns a healthy sensor
    assert flagged(CFG, (2, 4, 6, 8)) < 0.4 * unguarded
    assert flagged(CFG, (1, 3, 5, 7)) == 0.0  # clear days unaffected


def _gain_stub(cfg, with_meter, sensor_gain=0.98):
    """Only jagged, cloudy windows survive QC and there the sensor reads 0.7x the (hourly-DTN) model."""
    from types import SimpleNamespace

    idx = pd.date_range("2026-09-01", "2026-09-02 23:55", freq="5min", tz="Etc/GMT+7")
    hour = idx.hour + idx.minute / 60
    base = pd.Series(np.where((hour >= 8) & (hour < 18), 900.0, 0.0), index=idx)
    analysis = pd.DataFrame({"cs_tracker": base, "csi_primary": 1.0, "use": base > 0})
    analysis["csi_meter"] = 1.0 if with_meter else np.nan
    jag = np.where(np.arange(len(idx)) % 2 == 0, 0.5, 0.9)  # 5-min cloud swings
    kept = (base * 0.7 * jag / 0.7).where(analysis["use"])  # mean ratio ~0.7 with large curvature
    stub = SimpleNamespace(config=cfg, analysis_freq=STEP, analysis=analysis, gains={"S": sensor_gain})
    model = sq.POADataset._anchored_model(stub, kept)
    mid = analysis["use"] & (hour >= 10) & (hour < 16)
    return (model[mid] / base[mid]).median()


def test_cloudy_windows_without_a_meter_cannot_set_the_backfill_gain():
    """QC removes the smooth stowed days, so cloudy days may be all that is kept; against hourly DTN
    the sensor reads low there (it cannot see the clouds). That bias must not become the gain."""
    assert _gain_stub(CFG, with_meter=False) == pytest.approx(0.98, abs=0.01)  # sensors' own gain


def test_with_a_meter_cloudy_windows_do_inform_the_gain():
    """The 1-minute meter does resolve the clouds, so those windows are valid calibration data."""
    assert _gain_stub(CFG, with_meter=True) < 0.8


def _hourly_model_inputs(meter_scale=1.0, dip=False):
    """One day on the 5-min grid: flat clear-sky, hourly DTN index 0.9, meter index = scale x (0.9, with
    an optional dip in the first half of 12:00-13:00)."""
    idx = pd.date_range("2026-09-01", "2026-09-01 23:55", freq="5min", tz="Etc/GMT+7")
    hour = idx.hour + idx.minute / 60
    cs = pd.Series(np.where((hour >= 8) & (hour < 18), 900.0, 0.0), index=idx)
    csi_dtn = pd.Series(0.9, index=idx)
    meter = pd.Series(0.9 * meter_scale, index=idx)
    if dip:
        meter[(hour >= 12) & (hour < 12.5)] *= 0.4
    return cs, csi_dtn, meter


def test_model_cloud_index_ignores_the_meters_magnitude():
    cs, csi_dtn, meter = _hourly_model_inputs(meter_scale=0.55)  # e.g. a stowed plant
    model = sq.model_cloud_index(csi_dtn, meter, cs, pd.Timedelta("1h"), STEP)
    assert model[cs > 0].to_numpy() == pytest.approx(0.9)


def test_model_cloud_index_takes_the_within_hour_shape_from_the_meter_and_keeps_dtns_energy():
    cs, csi_dtn, meter = _hourly_model_inputs(dip=True)
    model = sq.model_cloud_index(csi_dtn, meter, cs, pd.Timedelta("1h"), STEP)
    hour = model.index.hour + model.index.minute / 60
    dip, rest = (hour >= 12) & (hour < 12.5), (hour >= 12.5) & (hour < 13)
    assert model[dip].mean() < 0.5 * model[rest].mean()  # the meter's passing cloud survives
    hour_mean = (model * cs)[(hour >= 12) & (hour < 13)].sum() / cs[(hour >= 12) & (hour < 13)].sum()
    assert hour_mean == pytest.approx(0.9)  # ...but the hour still carries DTN's energy
    assert model[(hour >= 9) & (hour < 12)].to_numpy() == pytest.approx(0.9)  # hours without a dip are untouched


def test_a_slow_meter_slope_is_not_turned_into_an_hourly_sawtooth():
    """A stowed plant's index declines steadily through the afternoon: re-normalising each hour to
    DTN's mean would saw-tooth it. Only cloud-scale structure may survive."""
    cs, csi_dtn, meter = _hourly_model_inputs()
    hour = cs.index.hour + cs.index.minute / 60
    meter = pd.Series(np.interp(hour, [8, 18], [0.95, 0.5]), index=cs.index)
    trend = pd.Timedelta("2h")
    saw = sq.model_cloud_index(csi_dtn, meter, cs, pd.Timedelta("1h"), STEP)
    flat = sq.model_cloud_index(csi_dtn, meter, cs, pd.Timedelta("1h"), STEP, trend=trend)
    day = (cs > 0).to_numpy() & (hour > 9) & (hour < 17)
    assert saw[day].diff().abs().max() > 0.04  # without detrending: a step at every hour
    assert flat[day].diff().abs().max() < 0.01
    assert flat[day].to_numpy() == pytest.approx(0.9, abs=0.02)


def test_a_cloud_dip_survives_the_trend_removal():
    cs, csi_dtn, meter = _hourly_model_inputs(dip=True)
    model = sq.model_cloud_index(csi_dtn, meter, cs, pd.Timedelta("1h"), STEP, trend=pd.Timedelta("2h"))
    hour = model.index.hour + model.index.minute / 60
    assert model[(hour >= 12) & (hour < 12.5)].mean() < 0.6 * model[(hour >= 12.5) & (hour < 13)].mean()


def test_model_cloud_index_is_flat_dtn_where_the_meter_is_untrusted():
    cs, csi_dtn, meter = _hourly_model_inputs(dip=True)
    model = sq.model_cloud_index(csi_dtn, meter.where(meter.index.hour != 12), cs, pd.Timedelta("1h"), STEP)
    assert model[cs > 0].to_numpy() == pytest.approx(0.9)


def test_model_cloud_index_needs_no_shaping_when_the_reference_is_not_coarser():
    cs, csi_dtn, meter = _hourly_model_inputs(dip=True)
    assert sq.model_cloud_index(csi_dtn, meter, cs, STEP, STEP) is csi_dtn


def test_sensor_level_reads_the_median_of_sunny_days_only():
    idx = pd.date_range("2026-09-01", periods=8 * 288, freq="5min", tz="Etc/GMT+7")
    hour = idx.hour + idx.minute / 60
    cs = pd.Series(np.where((hour >= 8) & (hour < 18), 900.0, 0.0), index=idx)
    ref = cs * 0.9
    csi = pd.Series(1.0, index=idx)
    x = ref * 0.85
    assert sq.sensor_level(x, ref, csi, cs, cs > 0) == pytest.approx(0.85)
    assert np.isnan(sq.sensor_level(x, ref, csi * 0.5, cs, cs > 0))  # overcast: not judged
    assert np.isnan(sq.sensor_level(x, ref, csi, cs, cs > 0, min_days=9))  # too few days


def test_level_check_flags_a_sensor_reading_low_against_dtn_only_when_asked(world):
    from dataclasses import replace

    ok, low = world["truth"][sq.TRACKER], world["truth"][sq.TRACKER] * 0.8
    res = assess({"ok": ok, "low": low}, world)
    assert res["sensors"]["ok"]["level"] == pytest.approx(GAIN, abs=0.05)
    assert res["sensors"]["low"]["level"] == pytest.approx(0.8 * GAIN, abs=0.05)
    assert not res["sensors"]["low"]["label"].eq("calibration").any()  # off by default

    res = assess({"ok": ok, "low": low}, world, config=replace(CFG, level_bounds=(0.95, 1.15)))
    assert res["sensors"]["low"]["label"].eq("calibration").any()
    assert not res["sensors"]["ok"]["label"].eq("calibration").any()


def _drop_stub(removed_fraction, cfg=CFG):
    from types import SimpleNamespace

    idx = pd.date_range("2026-09-01", periods=288, freq="5min", tz="Etc/GMT+7")
    use = pd.Series((idx.hour >= 8) & (idx.hour < 18), index=idx)
    final = pd.DataFrame(
        {c: np.where(use, 500.0, 0.0) for c in removed_fraction}, index=idx
    )
    for c, frac in removed_fraction.items():
        final.loc[final.index[use.to_numpy()][: int(frac * use.sum())], c] = np.nan
    return SimpleNamespace(
        config=cfg, analysis=pd.DataFrame({"use": use}), analysis_freq=STEP,
        df_final=final, poa_columns=list(removed_fraction),
    )


def test_a_sensor_that_is_mostly_removed_is_dropped():
    stub = _drop_stub({"good": 0.1, "stowed": 0.9})
    assert list(sq.POADataset._mostly_removed_sensors(stub)) == ["stowed"]


def test_the_last_sensor_is_never_dropped_and_dropping_can_be_disabled():
    assert sq.POADataset._mostly_removed_sensors(_drop_stub({"only": 0.95})) == {}
    assert sq.POADataset._mostly_removed_sensors(_drop_stub({"a": 0.9, "b": 0.95})) == {}
    from dataclasses import replace

    off = replace(CFG, drop_removed_fraction=None)
    assert sq.POADataset._mostly_removed_sensors(_drop_stub({"good": 0.1, "stowed": 0.9}, off)) == {}


def _gappy_series():
    idx = pd.date_range("2026-09-01 10:00", periods=60, freq="1min", tz="Etc/GMT+7")
    s = pd.Series(np.linspace(100.0, 160.0, 60), index=idx)
    s.iloc[5:8] = np.nan  # 3 min
    s.iloc[20:35] = np.nan  # 15 min: at the limit
    s.iloc[40:41] = np.nan  # 1 min
    s.iloc[45:55] = np.nan  # 10 min but followed by data
    return s


def test_short_gaps_are_bridged_from_the_data_itself():
    s = _gappy_series()
    out, filled = sq._fill_short_gaps(s, "15min", pd.Timedelta("1min"))
    assert out.notna().all() and filled.sum() == 3 + 15 + 1 + 10
    assert out.iloc[6] == pytest.approx(np.linspace(100.0, 160.0, 60)[6])  # linear data -> exact


def test_gaps_longer_than_the_limit_are_left_alone():
    s = _gappy_series()
    out, filled = sq._fill_short_gaps(s, "10min", pd.Timedelta("1min"))
    assert out.iloc[20:35].isna().all() and not filled.iloc[20:35].any()  # the 15-min gap
    assert out.iloc[5:8].notna().all() and out.iloc[45:55].notna().all()


def test_gaps_at_the_edges_are_never_filled():
    s = _gappy_series()
    s.iloc[:4] = np.nan  # leading gap: nothing before it to interpolate from
    s.iloc[-3:] = np.nan
    out, filled = sq._fill_short_gaps(s, "15min", pd.Timedelta("1min"))
    assert out.iloc[:4].isna().all() and out.iloc[-3:].isna().all() and not filled.iloc[:4].any()


def test_gap_filling_can_be_disabled():
    s = _gappy_series()
    for off in (None, "0min"):
        out, filled = sq._fill_short_gaps(s, off, pd.Timedelta("1min"))
        assert out.isna().sum() == s.isna().sum() and not filled.any()


def test_fill_max_gap_is_validated():
    with pytest.raises(ValueError):
        sq.POAQCConfig(fill_max_gap="not a duration")
    assert sq.POAQCConfig(fill_max_gap=None).fill_max_gap is None


def test_sensor_reading_near_zero_is_bad_not_stow(world):
    s = world["truth"][sq.TRACKER] * 0.03  # dead/covered: no geometry reads this little
    res = assess({"OE.POA_A": s}, world)
    assert daylight_share(res, "OE.POA_A", "stow", CLEAR_DAYS) < 0.02
    assert daylight_share(res, "OE.POA_A", "bad", CLEAR_DAYS) > 0.7


def test_stow_floor_is_configurable(world):
    stowed = world["truth"]["rot-45"].copy()  # west-facing all day: very low mornings
    loose = sq.assess_poa_sensors(pd.DataFrame({"S": stowed}), world["dtn"], LOCATION, TRACKER_DESIGN,
                                  ref_meter=world["meter"], config=sq.POAQCConfig(min_stow_ratio=0.0))
    strict = sq.assess_poa_sensors(pd.DataFrame({"S": stowed}), world["dtn"], LOCATION, TRACKER_DESIGN,
                                   ref_meter=world["meter"], config=sq.POAQCConfig(min_stow_ratio=0.9))
    n = lambda r: r["sensors"]["S"]["label"].eq("stow").sum()
    assert n(loose) > n(strict)


def test_fixed_tilt_design_has_only_the_array_geometry():
    index = pd.date_range("2026-09-01", periods=288, freq="5min", tz="Etc/GMT+7")
    geoms, _ = sq.build_geometries(LOCATION, index, FIXED_DESIGN, STEP, 15)
    assert list(geoms.columns) == [sq.TRACKER]


def test_stow_rotations_are_symmetric_and_include_flat_and_max():
    rots = sq._stow_rotations(52, 15)
    assert rots == sorted(rots) and 0 in rots and 52 in rots and -52 in rots
    assert all(-r in rots for r in rots)


def test_naive_index_is_rejected(world):
    naive = world["truth"][sq.TRACKER].tz_localize(None)
    with pytest.raises(TypeError):
        assess({"OE.POA_A": naive}, world)


def test_config_validation():
    with pytest.raises(ValueError):
        sq.POAQCConfig(tol=0)
    with pytest.raises(ValueError):
        sq.POAQCConfig(gain_bounds=(1.2, 0.8))
    with pytest.raises(ValueError):
        sq.POAQCConfig(analysis_freq="30s")
