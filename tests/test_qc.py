"""Tests for oetoolbox.datatools.qc (flag_interpolated_data and QCDataset).

All data is synthetic (generated below), so no data files are needed.
"""

import numpy as np
import pandas as pd
import pytest

from oetoolbox.datatools import qc

LOC = dict(tz="US/Pacific", lat=34.7, lon=-117.3)
DAY = 1440  # samples per day at 1-minute resolution


@pytest.fixture
def index():
    return pd.date_range("2026-06-10", periods=4 * DAY, freq="1min", tz="US/Pacific")


@pytest.fixture
def solar(index):
    """Four days of noisy, full-precision solar-like power (0 at night, ~10 at noon)."""
    rng = np.random.default_rng(0)
    minute = np.arange(len(index)) % DAY
    sun = np.clip(np.sin((minute - 360) / 810 * np.pi), 0, None)
    return pd.Series(10 * sun * (1 + 0.05 * rng.standard_normal(len(index))), index=index)


def flagged(series, **kwargs):
    return qc.flag_interpolated_data(series, location_kwargs=LOC, **kwargs)


def inject_ramp(s, a, b):
    """Linear interpolation between anchors at positions a and b (interior = b - a - 1 points)."""
    s = s.copy()
    s.iloc[a : b + 1] = np.linspace(s.iloc[a], s.iloc[b], b - a + 1)
    return s


# ---- flag_interpolated_data -----------------------------------------------------------------
def test_clean_data_not_flagged(solar):
    assert flagged(solar).empty


def test_linear_run_flagged_exactly(solar):
    a, b = 2040, 2281  # day 2, ~10:00-14:00
    out = flagged(inject_ramp(solar, a, b))
    assert len(out) == 1
    row = out.iloc[0]
    assert row.kind == "linear"
    assert row.n_points == b - a - 1
    assert row.start == solar.index[a + 1] and row.end == solar.index[b - 1]  # anchors excluded


def test_quantized_linear_run_flagged(solar):
    a, b = 2040, 2240
    out = flagged(inject_ramp(solar, a, b).round(3))
    assert len(out) == 1 and out.iloc[0].n_points == b - a - 1


def test_float32_noise_linear_run_flagged(solar):
    s = inject_ramp(solar, 2040, 2240).round(3)
    out = flagged(pd.Series(s.to_numpy().astype("float32").astype(float), index=s.index))
    assert len(out) == 1


def test_flatline_across_night_flagged(solar):
    s = solar.copy()
    s.iloc[1500:2500] = 5.0  # held at 5 from 01:00 to 17:40
    out = flagged(s)
    assert len(out) == 1 and out.iloc[0].kind == "flatline" and out.iloc[0].n_points == 998


def test_short_daytime_plateau_is_exempt(solar):
    s = solar.copy()
    s.iloc[2000:2200] = 5.0  # ~3.3 h daytime clipping/curtailment
    assert flagged(s).empty


def test_long_daytime_plateau_flagged(solar):
    s = solar.copy()
    s.iloc[1700:2300] = 5.0  # 10 h > max_plateau (8 h)
    assert len(flagged(s)) == 1


def test_flat_zero_not_flagged(solar):
    s = solar.copy()
    s.iloc[100:400] = 0.0
    assert flagged(s).empty


def test_low_amplitude_dawn_ramp_not_flagged(solar):
    s = solar.copy()
    s.iloc[1500:1861] = np.linspace(0, 0.3, 361)  # 0 -> 3% of scale over 6 h
    assert flagged(s).empty


def test_run_shorter_than_min_duration_not_flagged(solar):
    assert flagged(inject_ramp(solar, 2040, 2130)).empty  # 89 interior points < 2 h
    assert len(flagged(inject_ramp(solar, 2040, 2130), min_duration="1h")) == 1


def test_missing_timestamps_break_run(solar):
    s = inject_ramp(solar, 2040, 2240).drop(solar.index[2140:2145])
    assert flagged(s).empty  # both halves (~100 min) < 2 h


def test_unsorted_duplicated_and_naive_index_equivalent(solar):
    s = inject_ramp(solar, 2040, 2281)
    messy = pd.concat([s, s.iloc[:10]]).sample(frac=1, random_state=1)
    assert flagged(messy).n_points.tolist() == flagged(s).n_points.tolist()
    naive = qc.flag_interpolated_data(s.tz_localize(None), location_kwargs=LOC)
    assert naive.n_points.tolist() == flagged(s).n_points.tolist()


def test_degenerate_input(solar):
    assert qc.flag_interpolated_data(solar.iloc[:2]).empty
    assert qc.flag_interpolated_data(solar * np.nan).empty


# ---- QCDataset ------------------------------------------------------------------------------
@pytest.fixture
def raw(index):
    rng = np.random.default_rng(1)
    t = np.arange(len(index))
    minute = t % DAY
    sun = np.clip(np.sin((minute - 420) / 660 * np.pi), 0, None)
    return pd.DataFrame(
        {
            "OE_MeterMW": 10 * sun * (1 + 0.05 * rng.standard_normal(len(t))),
            "Ambient_Temp": -5 + 8 * np.sin((minute - 540) / DAY * 2 * np.pi)
            + 0.1 * rng.standard_normal(len(t)),
            "Wind_Speed": np.abs(8 + 3 * rng.standard_normal(len(t))),
            "Empty": np.nan,
        },
        index=index,
    )


def test_instantiates_and_runs(raw):
    q = qc.QCDataset(raw, location_kwargs=LOC).run()
    assert "Empty" not in q.data.columns
    assert q.raw.shape == raw.drop(columns=[]).shape  # raw is untouched
    assert not q.summary().empty


def test_requires_datetime_index_and_numeric_data(raw):
    with pytest.raises(TypeError):
        qc.QCDataset(raw.reset_index(drop=True))
    with pytest.raises(ValueError):
        qc.QCDataset(pd.DataFrame({"a": ["x"] * 3}, index=raw.index[:3]))


def test_regularize_index_logs_duplicates_and_gaps(raw):
    messy = pd.concat([raw, raw.iloc[:5]]).drop(raw.index[4000:4010])
    q = qc.QCDataset(messy, location_kwargs=LOC)
    assert len(q.data) == len(raw)
    log = pd.DataFrame(q._log).set_index("step").n_points
    assert log["duplicate_timestamps_dropped"] == 5
    assert log["missing_timestamps_filled"] == 10


def test_negative_temperature_preserved(raw):
    q = qc.QCDataset(raw, location_kwargs=LOC).handle_negatives()
    assert q.data["Ambient_Temp"].min() == raw["Ambient_Temp"].min() < 0


def test_negatives_small_clipped_large_removed(raw):
    df = raw[["OE_MeterMW"]].rename(columns={"OE_MeterMW": "POA_Irradiance"})
    df.iloc[100, 0] = -0.01  # noise -> 0
    df.iloc[101, 0] = -5.0  # sensor error -> NaN
    q = qc.QCDataset(df, location_kwargs=LOC).handle_negatives()
    assert q.data.iloc[100, 0] == 0.0
    assert np.isnan(q.data.iloc[101, 0])
    assert q.removed("POA_Irradiance").value.tolist() == [-5.0]


def test_spikes_removed_and_recorded(raw):
    raw = raw.copy()
    raw.iloc[3000, 0] = 90.0
    raw.iloc[2500, 2] = 300.0
    q = qc.QCDataset(raw, location_kwargs=LOC).remove_spikes()
    assert np.isnan(q.data.iloc[3000, 0]) and np.isnan(q.data["Wind_Speed"].iloc[2500])
    assert q.removed("OE_MeterMW").value.tolist() == [90.0]
    # natural noise in wind speed must not be mistaken for spikes
    assert q.data["Wind_Speed"].isna().sum() == 1


def test_auto_outliers_keep_real_solar_peaks(raw):
    """Global IQR fences would cut real daytime peaks; the default (hampel) must not."""
    q = qc.QCDataset(raw, location_kwargs=LOC).remove_outliers()
    assert q.data["OE_MeterMW"].isna().sum() < 0.005 * len(raw)
    assert q.data["OE_MeterMW"].max() == raw["OE_MeterMW"].max()


def test_interpolation_runs_first_and_is_recorded(raw):
    raw = raw.copy()
    raw["OE_MeterMW"] = inject_ramp(raw["OE_MeterMW"], 2040, 2281)
    q = qc.QCDataset(raw, location_kwargs=LOC).run()
    ranges = q.interpolated["OE_MeterMW"]
    assert len(ranges) == 1 and ranges.iloc[0].n_points == 240
    assert q.removed("OE_MeterMW").step.eq("interpolated").sum() == 240


def test_non_solar_never_uses_daylight(raw):
    q = qc.QCDataset(raw[["Wind_Speed"]], is_solar=False).run()
    assert not q.is_solar and q.location_kwargs is None


def test_out_of_range_and_fill_gaps(raw):
    q = qc.QCDataset(raw, location_kwargs=LOC)
    q.flag_out_of_range({"temperature": (-50, -10)})
    assert q.data["Ambient_Temp"].max() <= -10
    q.data.iloc[500:510, 0] = np.nan
    q.fill_gaps(max_gap="30min", columns=["OE_MeterMW"])
    assert q.data["OE_MeterMW"].iloc[500:510].notna().all()
