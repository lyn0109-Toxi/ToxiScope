"""Presentation regressions: preserve reported values without new inference."""
import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from core.bioequivalence import DEFAULT_DISSOLUTION_PROFILE
from presentation.be_visuals import (
    build_dissolution_curves,
    build_dissolution_differences,
    build_simulation_quantiles,
)


def _rows(spec):
    return [row for dataset in spec.get("datasets", {}).values() for row in dataset]


def _quantiles(**overrides):
    return {
            "simulation_interval_low": 45.123456,
            "simulation_p05": 46.234567,
            "simulation_median": 51.345678,
            "simulation_p95": 63.456789,
            "simulation_interval_high": 64.567891,
            **overrides,
        }


def test_curve_spec_preserves_both_entered_mean_profiles_and_sampling_times():
    profile = DEFAULT_DISSOLUTION_PROFILE.copy(deep=True)
    before = profile.copy(deep=True)
    spec = build_dissolution_curves(profile).to_dict()
    rows = _rows(spec)
    assert len(rows) == 2 * len(profile)
    for sample, source in [("기준품", "Reference"), ("시험품", "Test")]:
        actual = [(row["시간 (분)"], row["평균 용출 (%)"]) for row in rows if row["시료"] == sample]
        expected = list(zip(profile["Time (min)"], profile[f"{source} Mean (%)"]))
        assert actual == expected
    assert spec["mark"]["type"] == "line"
    assert spec["encoding"]["x"]["title"] == "시간 (분)"
    assert spec["encoding"]["y"]["title"] == "평균 용출 (%)"
    assert "transform" not in spec
    pd.testing.assert_frame_equal(profile, before)


def test_sd_band_keeps_raw_negative_and_above_100_bounds_without_claiming_a_ci():
    profile = DEFAULT_DISSOLUTION_PROFILE.iloc[:3].copy()
    profile["Reference Mean (%)"] = [1, 98, 99]
    profile["Reference SD"] = [5, 10, 10]
    spec = build_dissolution_curves(profile, show_sd=True).to_dict()
    rows = [row for row in _rows(spec) if row["시료"] == "기준품"]
    assert [row["하한"] for row in rows] == [-4, 88, 89]
    assert [row["상한"] for row in rows] == [6, 108, 109]
    assert spec["layer"][0]["mark"]["type"] == "area"
    assert spec["layer"][0]["encoding"]["y"]["field"] == "하한"
    assert spec["layer"][0]["encoding"]["y2"]["field"] == "상한"
    encoded = json.dumps(spec, ensure_ascii=False)
    assert "표본 SD (pp)" in encoded
    assert "domainMax" not in encoded
    assert '"clip": true' not in encoded
    assert "confidence" not in encoded.lower()


def test_difference_bars_keep_signed_percentage_point_differences_and_zero_baseline():
    profile = DEFAULT_DISSOLUTION_PROFILE.iloc[:3].copy()
    profile["Reference Mean (%)"] = [30, 50, 70]
    profile["Test Mean (%)"] = [27, 50, 74]
    spec = build_dissolution_differences(profile).to_dict()
    rows = [row for row in _rows(spec) if "차이 (pp)" in row]
    assert [row["차이 (pp)"] for row in rows] == [-3, 0, 4]
    assert [row["방향"] for row in rows] == ["시험품 낮음", "동일", "시험품 높음"]
    assert spec["layer"][0]["mark"]["type"] == "bar"
    assert spec["layer"][0]["encoding"]["y"]["title"] == "시험품 − 기준품 (pp)"
    assert any(row.get("기준 차이") == 0 for row in _rows(spec))


def test_quantile_glyph_contains_exact_reported_quantiles_and_descriptive_50_line_only():
    result = _quantiles()
    before = copy.deepcopy(result)
    spec = build_simulation_quantiles(result).to_dict()
    quantiles = {row["백분위"]: row["f2"] for row in _rows(spec) if "백분위" in row}
    assert quantiles == {
        "2.5%": 45.123456, "5%": 46.234567, "50%": 51.345678,
        "95%": 63.456789, "97.5%": 64.567891,
    }
    intervals = {row["구간"]: (row["하한"], row["상한"]) for row in _rows(spec) if "구간" in row}
    assert intervals == {
        "2.5–97.5 백분위": (45.123456, 64.567891),
        "5–95 백분위": (46.234567, 63.456789),
    }
    assert [row["서술적 참조"] for row in _rows(spec) if "서술적 참조" in row] == [50]
    assert len(spec["layer"]) == 4
    assert spec["layer"][2]["mark"]["shape"] == "diamond"
    encoded = json.dumps(spec, ensure_ascii=False)
    assert "density" not in encoded and '"bin"' not in encoded
    assert "probability" not in encoded.lower()
    assert "transform" not in encoded
    assert result == before


def test_quantile_glyph_keeps_negative_reported_values_instead_of_clipping():
    spec = build_simulation_quantiles(_quantiles(simulation_interval_low=-8.25)).to_dict()
    assert any(row.get("f2") == -8.25 for row in _rows(spec))
    assert any(row.get("하한") == -8.25 for row in _rows(spec))


@pytest.mark.parametrize("result", [
    None,
    SimpleNamespace(ci_low=10, bootstrap_median=50, ci_high=90),
    _quantiles(simulation_p05=None),
    _quantiles(simulation_interval_low=float("nan")),
    _quantiles(simulation_median=70),
])
def test_missing_legacy_nonfinite_or_unordered_quantiles_never_generate_a_distribution(result):
    assert build_simulation_quantiles(result) is None


def test_incomplete_entered_means_do_not_draw_a_partial_comparison_as_complete():
    profile = DEFAULT_DISSOLUTION_PROFILE.copy()
    profile.loc[1, "Reference Mean (%)"] = float("nan")
    assert build_dissolution_curves(profile) is None
    assert build_dissolution_differences(profile) is None
