"""Tests for backfill3 (synthetic data; POADataset is stubbed, so no site database is needed)."""

import numpy as np
import pandas as pd
import pytest

from oetoolbox.datatools import backfill3 as b3

TZ = "Etc/GMT+7"
DAYS = 10
TEMP, WIND = "OE.Ambient_Temp", "OE.Wind_Speed"


@pytest.fixture(scope="module")
def world():
    """10 days: 1-minute index, hourly DTN (deg C, m/s), and 1-minute truth on the minute grid."""
    index = pd.date_range("2026-09-01", periods=DAYS * 1440, freq="1min", tz=TZ)
    hourly = pd.date_range(index[0], periods=DAYS * 24, freq="1h")
    rng = np.random.default_rng(1)
    hours = np.arange(len(hourly))
    temp_c = 20 + 8 * np.sin(2 * np.pi * (hours - 9) / 24) + rng.normal(0, 0.3, len(hourly))
    wind = 5 + 2 * np.sin(2 * np.pi * (hours - 14) / 24) + rng.normal(0, 0.3, len(hourly))
    dtn = pd.DataFrame(
        {"dtn_temp_air": temp_c, "dtn_wind_speed": wind, "poa_global": 0.0}, index=hourly
    )
    truth_temp = b3._ref_on_index(dtn["dtn_temp_air"], index)
    truth_wind = b3._ref_on_index(dtn["dtn_wind_speed"], index)
    return dict(index=index, dtn=dtn, temp=truth_temp, wind=truth_wind, rng=rng)


def temp_sensor(world, offset_c=0.5, noise=0.1, seed=0):
    """deg F sensor reading truth + offset."""
    rng = np.random.default_rng(seed)
    c = world["temp"] + offset_c + rng.normal(0, noise, len(world["index"]))
    return c * 1.8 + 32


def wind_sensor(world, gain=0.6, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    return (world["wind"] * gain + rng.normal(0, noise, len(world["index"]))).clip(lower=0)


def run(df, world, **kwargs):
    return b3.backfill_weather_sensor_data("Synthetic", df, world["dtn"], **kwargs)


def day_mask(index, day):
    return (index >= index[0] + pd.Timedelta(days=day)) & (
        index < index[0] + pd.Timedelta(days=day + 1)
    )


# ---- temperature -------------------------------------------------------------------------------
def test_fahrenheit_detected_and_converted_to_celsius(world):
    df = pd.DataFrame(
        {f"{TEMP}_A": temp_sensor(world, seed=1), f"{TEMP}_B": temp_sensor(world, seed=2)}
    )
    r = run(df, world)
    assert r["qc_changes"][TEMP]["native_units"] == "F"
    assert (r["flags"][[f"{TEMP}_A_FLAG", f"{TEMP}_B_FLAG"]] == "").all().all()
    assert (r["source"][TEMP] == "sensor").all()
    as_c = (df - 32) / 1.8
    pd.testing.assert_series_equal(r["df"][TEMP], as_c.mean(axis=1), check_names=False)
    pd.testing.assert_frame_equal(r["df"][df.columns], as_c)  # sensor columns too
    pd.testing.assert_frame_equal(r["raw"][df.columns], as_c)  # and the raw (plot) data
    # whole-record offset is reported in deg C
    assert r["qc_changes"][TEMP]["corrections"][f"{TEMP}_A"] == pytest.approx(0.5, abs=0.2)


def test_celsius_sensors_are_not_converted(world):
    df = pd.DataFrame({f"{TEMP}_A": (temp_sensor(world) - 32) / 1.8})
    r = run(df, world)
    assert r["qc_changes"][TEMP]["native_units"] == "C"
    assert r["df"][TEMP].median() == pytest.approx(df.iloc[:, 0].median(), abs=1e-9)


def test_step_fault_is_bad_and_peers_stay_ok(world):
    df = pd.DataFrame({f"{TEMP}_{k}": temp_sensor(world, seed=i) for i, k in enumerate("ABC")})
    on_day4 = day_mask(df.index, 4)
    df.loc[on_day4, f"{TEMP}_B"] += 8 * 1.8  # +8 deg C for a day
    r = run(df, world)
    flags = r["flags"]
    assert flags.loc[on_day4, f"{TEMP}_B_FLAG"].eq("bad").mean() > 0.8
    assert flags.loc[~on_day4, f"{TEMP}_B_FLAG"].eq("bad").mean() < 0.02
    for k in "AC":
        assert flags[f"{TEMP}_{k}_FLAG"].eq("").all()
    # the bad data is removed, and the average comes from the healthy sensors
    assert r["df"].loc[on_day4, f"{TEMP}_B"].isna().mean() > 0.8
    assert (r["source"][TEMP] == "sensor").all()


def test_peers_excuse_dtn_disagreement(world):
    """Both sensors leave DTN for half a day (e.g. a local front DTN misses): kept, not flagged."""
    df = pd.DataFrame(
        {f"{TEMP}_A": temp_sensor(world, seed=1), f"{TEMP}_B": temp_sensor(world, seed=2)}
    )
    front = (df.index >= df.index[0] + pd.Timedelta(days=5)) & (
        df.index < df.index[0] + pd.Timedelta(days=5, hours=12)
    )
    df.loc[front] -= 7 * 1.8
    r = run(df, world)
    assert (r["flags"] == "").all().all()


def test_short_gap_interpolated_long_gap_modelled_without_step(world):
    df = pd.DataFrame({f"{TEMP}_A": temp_sensor(world, offset_c=2.0, noise=0.0)})
    i0 = df.index[0]
    short = (df.index >= i0 + pd.Timedelta("2D10h")) & (df.index < i0 + pd.Timedelta("2D10h30min"))
    long_ = (df.index >= i0 + pd.Timedelta("6D06h")) & (df.index < i0 + pd.Timedelta("6D14h"))
    df.loc[short | long_] = np.nan
    r = run(df, world)
    src = r["source"][TEMP]
    assert src[short].eq("interpolated").all()
    assert src[long_].eq("model").all()
    assert r["flags"].loc[long_, f"{TEMP}_A_FLAG"].eq("missing").all()
    # the anchored model carries the sensor's +2 C offset into the gap: no step at either edge
    proc = r["df"][TEMP]
    first, last = np.flatnonzero(long_)[[0, -1]]
    assert abs(proc.iloc[first] - proc.iloc[first - 1]) < 0.5
    assert abs(proc.iloc[last + 1] - proc.iloc[last]) < 0.5
    truth_c = world["temp"] + 2.0
    assert (proc[long_] - truth_c[long_]).abs().max() < 1.0


def test_mostly_bad_sensor_is_dropped(world):
    df = pd.DataFrame(
        {f"{TEMP}_A": temp_sensor(world, seed=1), f"{TEMP}_B": temp_sensor(world, seed=2)}
    )
    broken = df.index >= df.index[0] + pd.Timedelta(days=2)
    # from day 2 on, B reads the diurnal cycle upside down (e.g. a heated enclosure)
    df.loc[broken, f"{TEMP}_B"] = (40 - world["temp"][broken]) * 1.8 + 32
    r = run(df, world)
    assert f"{TEMP}_B" in r["qc_changes"][TEMP]["dropped_sensors"]
    assert r["df"][f"{TEMP}_B"].isna().all()
    assert r["df"][TEMP].notna().all()


def test_offset_out_of_bounds_is_calibration(world):
    df = pd.DataFrame({f"{TEMP}_A": temp_sensor(world, offset_c=9.0)})
    r = run(df, world, sensor_configs={TEMP: {"drop_removed_fraction": None}})
    assert r["flags"][f"{TEMP}_A_FLAG"].eq("calibration").mean() > 0.95


def test_unusable_columns_removed_and_dtn_used(world):
    df = pd.DataFrame({f"{TEMP}_A": 0.0, f"{TEMP}_B": 32.0}, index=world["index"])
    r = run(df, world)
    removed = r["qc_changes"][TEMP]["removed_columns"]
    assert removed == {"all_zero": [f"{TEMP}_A"], "constant": [f"{TEMP}_B"]}
    assert (r["source"][TEMP] == "model").all()
    assert r["df"][TEMP].notna().all()


# ---- wind --------------------------------------------------------------------------------------
def test_frozen_anemometers_flagged_even_when_they_agree(world):
    df = pd.DataFrame(
        {f"{WIND}_A": wind_sensor(world, seed=1), f"{WIND}_B": wind_sensor(world, seed=2)}
    )
    frozen = day_mask(df.index, 3)
    df.loc[frozen] = 0.0
    r = run(df, world)
    for k in "AB":
        assert r["flags"].loc[frozen, f"{WIND}_{k}_FLAG"].eq("bad").mean() > 0.9
    assert r["source"].loc[frozen, WIND].eq("model").mean() > 0.9
    # backfill uses the sensors' gain (~0.6), not raw DTN
    ratio = r["df"].loc[frozen, WIND].mean() / world["wind"][frozen].mean()
    assert ratio == pytest.approx(0.6, abs=0.1)


def test_healthy_wind_untouched(world):
    df = pd.DataFrame(
        {f"{WIND}_A": wind_sensor(world, seed=1), f"{WIND}_B": wind_sensor(world, 0.7, seed=2)}
    )
    r = run(df, world)
    assert (r["flags"] == "").all().all()


# ---- group handling / POA ----------------------------------------------------------------------
def test_missing_groups_use_dtn(world):
    df = pd.DataFrame({f"{TEMP}_A": temp_sensor(world)})
    r = run(df, world)
    assert set(r["backfilled"].columns) == {TEMP, "OE.POA", WIND}
    assert (r["source"][WIND] == "model").all()
    np.testing.assert_allclose(r["df"][WIND], r["df"]["dtn_wind_speed"])


class StubPOADataset:
    """Stands in for POADataset: one sensor, half flagged as stow."""

    calls = []

    def __init__(
        self,
        site,
        dataframe,
        ref_poa,
        ref_meter=None,
        config=None,
        meter_valid_mask=None,
        ref_weather=None,
    ):
        StubPOADataset.calls.append(
            dict(site=site, columns=list(dataframe.columns), config=config, ref_weather=ref_weather)
        )
        self.poa_columns = list(dataframe.columns)
        self.df_native = dataframe
        flag = pd.Series("", index=dataframe.index)
        flag.iloc[: len(flag) // 2] = "stow"
        self.df_flags = pd.DataFrame({f"{c}_FLAG": flag for c in self.poa_columns})
        self.qc_changes = {"stub": True}

    def remove_flagged_ranges(self, include_stow=True):
        df = self.df_native.copy()
        stow = self.df_flags.iloc[:, 0].eq("stow")
        if include_stow:
            df.loc[stow] = np.nan
        df["OE.POA_average"] = df[self.poa_columns].mean(axis=1)
        df["Processed_POA"] = df["OE.POA_average"].fillna(-1.0)
        self.poa_source = pd.Series(
            np.where(df["OE.POA_average"].isna(), "model", "sensor"), index=df.index
        )
        return df


def test_poa_group_uses_poadataset(world, monkeypatch):
    monkeypatch.setattr(b3, "POADataset", StubPOADataset)
    StubPOADataset.calls.clear()
    df = pd.DataFrame({"OE.POA_1": 500.0, f"{TEMP}_A": temp_sensor(world)}, index=world["index"])
    r = run(df, world)
    call = StubPOADataset.calls[0]
    assert call["site"] == "Synthetic" and call["columns"] == ["OE.POA_1"]
    assert call["ref_weather"] is not None  # dtn_wind_speed is a recognized weather column
    half = len(df) // 2
    assert r["df"]["OE.POA"].iloc[:half].eq(-1.0).all()
    assert r["df"]["OE.POA"].iloc[half:].eq(500.0).all()
    assert r["df"]["OE.POA_1"].iloc[:half].isna().all()
    assert r["source"]["OE.POA"].iloc[:half].eq("model").all()
    assert "OE.POA_1_FLAG" in r["flags"]
    assert isinstance(r["poa_dataset"], StubPOADataset)


def test_poa_analysis_freq_follows_coarse_sensor_data(world, monkeypatch):
    monkeypatch.setattr(b3, "POADataset", StubPOADataset)
    StubPOADataset.calls.clear()
    idx = pd.date_range(world["index"][0], world["index"][-1], freq="15min")
    df = pd.DataFrame({"OE.POA_1": 500.0}, index=idx)
    run(df, world)
    assert pd.Timedelta(StubPOADataset.calls[0]["config"].analysis_freq) == pd.Timedelta("15min")


def test_poa_falls_back_to_dtn_without_valid_data(world, monkeypatch):
    def raising(*args, **kwargs):
        raise ValueError("No valid data found for 'OE.POA' sensor column(s).\n{}")

    monkeypatch.setattr(b3, "POADataset", raising)
    df = pd.DataFrame({"OE.POA_1": np.nan}, index=world["index"])
    r = run(df, world)
    assert (r["source"]["OE.POA"] == "model").all()
    assert r["poa_dataset"] is None


def test_requires_tz_aware_data(world):
    df = pd.DataFrame({f"{TEMP}_A": temp_sensor(world)}).tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        run(df, world)


def test_unknown_sensor_config_group_rejected(world):
    df = pd.DataFrame({f"{TEMP}_A": temp_sensor(world)})
    with pytest.raises(KeyError):
        run(df, world, sensor_configs={"OE.POA": {}})


def test_plot_and_legacy_format(world):
    from oetoolbox.datatools.backfill2 import format_backfill_data_to_match_legacy

    df = pd.DataFrame({f"{TEMP}_A": temp_sensor(world), f"{WIND}_A": wind_sensor(world)})
    r = run(df, world)
    fig = b3.backfill_subplots(r)
    assert len(fig.data) > 0
    legacy = format_backfill_data_to_match_legacy(r["df"])
    assert {"Processed_AmbTemp", "Processed_POA", "Processed_Wind"} <= set(legacy.columns)
    assert not b3.flag_summary(r).empty or (r["flags"] == "").all().all()
