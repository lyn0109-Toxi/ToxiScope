"""Numerical/software checks for descriptive f2 and model-based simulation."""
from dataclasses import asdict
import json
import math
import shutil
import subprocess
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core import bioequivalence as be


COLUMNS = (
    "Time (min)", "Reference Mean (%)", "Reference SD", "Reference n",
    "Test Mean (%)", "Test SD", "Test n",
)


def profile(*, reference=(20.0, 50.0, 80.0), test=None, sd=0.0, n=12, times=(5.0, 10.0, 20.0)):
    test = reference if test is None else test
    return pd.DataFrame({
        "Time (min)": list(times),
        "Reference Mean (%)": list(reference), "Reference SD": sd, "Reference n": n,
        "Test Mean (%)": list(test), "Test SD": sd, "Test n": n,
    })


def calculate(data=None, *, runs=100, seed=1729, backend="python"):
    return be.calculate_f2(profile() if data is None else data, simulation_runs=runs, seed=seed, backend=backend)


def assert_simulation_not_be(result):
    assert result.schema == "dissolution-f2-simulation.v2"
    assert result.clinical_be_established is False
    assert result.regulatory_acceptance == "Not established"
    assert result.simulation_assumptions
    assert result.limitations
    assert result.cv_flag != "Acceptable"
    assert result.fda_risk != "Low"


def test_identical_profiles_have_analytic_f2_one_hundred():
    result = calculate()
    assert result.f2_unrounded == 100.0
    assert result.f2 == 100.0
    assert result.simulation_interval_low == 100.0
    assert result.simulation_interval_high == 100.0
    assert result.simulated_fraction_f2_ge_50 == 1.0
    assert_simulation_not_be(result)


def test_constant_ten_percentage_point_difference_matches_analytic_oracle():
    result = calculate(profile(test=(10.0, 40.0, 70.0)))
    expected = 50.0 * math.log10(100.0 / math.sqrt(101.0))
    assert result.f2_unrounded == pytest.approx(expected, abs=1e-12)
    assert result.f2 == 49.89
    assert result.simulation_interval_low == pytest.approx(expected, abs=1e-12)
    assert result.simulation_interval_high == pytest.approx(expected, abs=1e-12)
    assert result.simulated_fraction_f2_ge_50 == 0.0
    assert_simulation_not_be(result)


def test_f2_is_symmetric_and_does_not_mutate_the_input_profile():
    original = profile(test=(21.0, 48.0, 83.0), sd=2.0)
    before = original.copy(deep=True)
    swapped = original.rename(columns={
        "Reference Mean (%)": "Test Mean (%)", "Test Mean (%)": "Reference Mean (%)",
        "Reference SD": "Test SD", "Test SD": "Reference SD",
        "Reference n": "Test n", "Test n": "Reference n",
    })
    left, right = calculate(original), calculate(swapped)
    assert left.f2_unrounded == pytest.approx(right.f2_unrounded, abs=1e-12)
    pd.testing.assert_frame_equal(original, before)


def test_sorting_preserves_all_points_and_records_a_canonical_input_fingerprint():
    data = profile(test=(21.0, 48.0, 83.0), sd=2.0)
    sorted_result = calculate(data, runs=20, seed=109)
    reordered_result = calculate(data.iloc[[2, 0, 1]], runs=20, seed=109)
    assert sorted_result.time_points_used == reordered_result.time_points_used == [5.0, 10.0, 20.0]
    assert sorted_result.profile_fingerprint == reordered_result.profile_fingerprint
    assert sorted_result.simulation_interval_low == reordered_result.simulation_interval_low
    assert len(sorted_result.profile_fingerprint) == 64
    changed = data.copy()
    changed.loc[0, "Reference SD"] = 3.0
    assert calculate(changed, runs=20, seed=109).profile_fingerprint != sorted_result.profile_fingerprint


def test_applicability_tail_warning_does_not_silently_remove_any_supplied_points():
    data = profile(reference=(20.0, 50.0, 80.0, 90.0, 95.0), times=(5.0, 10.0, 15.0, 30.0, 45.0))
    result = calculate(data)
    assert result.time_points_used == [5.0, 10.0, 15.0, 30.0, 45.0]
    assert any("85" in issue for issue in result.applicability_issues)


def test_threshold_uses_unrounded_f2_when_both_display_values_round_to_fifty():
    below_difference = math.sqrt(10.0 ** ((100.0 - 49.999) / 25.0) - 1.0)
    above_difference = math.sqrt(10.0 ** ((100.0 - 50.001) / 25.0) - 1.0)
    below = calculate(profile(test=tuple(x - below_difference for x in (20.0, 50.0, 80.0))))
    above = calculate(profile(test=tuple(x - above_difference for x in (20.0, 50.0, 80.0))))
    assert below.f2 == above.f2 == 50.0
    assert below.f2_unrounded < 50.0 < above.f2_unrounded
    assert below.descriptive_threshold_met is False
    assert above.descriptive_threshold_met is True
    assert below.simulated_fraction_f2_ge_50 == 0.0
    assert above.simulated_fraction_f2_ge_50 == 1.0
    assert below.conclusion != above.conclusion


def test_seeded_simulation_repeats_and_seed_changes_only_the_simulation():
    data = profile(test=(21.0, 48.0, 83.0), sd=4.0)
    first = calculate(data, runs=250, seed=1024)
    repeated = calculate(data, runs=250, seed=1024)
    changed = calculate(data, runs=250, seed=2048)
    assert asdict(first) == asdict(repeated)
    assert first.f2_unrounded == changed.f2_unrounded
    assert (first.simulation_interval_low, first.simulation_interval_high) != (changed.simulation_interval_low, changed.simulation_interval_high)
    assert first.seed == 1024 and changed.seed == 2048
    assert first.backend == "python"
    assert first.backend_version
    assert first.simulation_runs == 250


def test_fixed_seed_small_simulation_matches_independent_numeric_fixture():
    data = profile(test=(18.0, 49.0, 76.0))
    data["Reference SD"] = [1.0, 2.0, 3.0]
    data["Test SD"] = [4.0, 5.0, 6.0]
    data["Test n"] = 18
    stats = be._python_bootstrap_stats(data, bootstrap_runs=5, seed=73)
    # Fixed PCG64(73) scalar-normal fixture, independently evaluated with the
    # f2 formula. Linear interpolation between these five ordered samples is
    # the quantile convention; no observed dissolution units are resampled.
    samples = [
        68.67557838657906, 72.74346566690966, 69.39637861037592,
        87.69577894264673, 69.85148521144461,
    ]
    ordered = sorted(samples)
    assert stats["median"] == pytest.approx(ordered[2], abs=1e-10)
    assert stats["ci_low"] == pytest.approx(ordered[0] + 0.1 * (ordered[1] - ordered[0]), abs=1e-10)
    assert stats["ci_high"] == pytest.approx(ordered[3] + 0.9 * (ordered[4] - ordered[3]), abs=1e-10)
    assert stats["probability_f2_ge_50"] == 100.0


def test_sample_mean_simulation_matches_chi_squared_quantile_oracle():
    # For equal means and SDs, independent arm differences have variance 2*SD²/n.
    # With three timepoints, their mean squared difference is (2*SD²/n)*χ²(3)/3.
    # Quantiles below are fixed χ²(3) table values; f2 decreases with that quantity.
    data = profile(reference=(50.0, 50.0, 50.0), sd=6.0, n=12)
    result = calculate(data, runs=40000, seed=9182)
    variance = 2.0 * 6.0**2 / 12.0
    expected_low = 100.0 - 25.0 * math.log10(1.0 + variance * 9.348403604496148 / 3.0)
    expected_high = 100.0 - 25.0 * math.log10(1.0 + variance * 0.21579528262389785 / 3.0)
    assert result.simulation_interval_low == pytest.approx(expected_low, abs=0.25)
    assert result.simulation_interval_high == pytest.approx(expected_high, abs=0.25)
    assert_simulation_not_be(result)


def test_more_units_reduce_simulated_spread_and_larger_sd_increases_it():
    small_n = calculate(profile(reference=(50.0, 50.0, 50.0), sd=6.0, n=12), runs=5000, seed=32)
    large_n = calculate(profile(reference=(50.0, 50.0, 50.0), sd=6.0, n=48), runs=5000, seed=32)
    large_sd = calculate(profile(reference=(50.0, 50.0, 50.0), sd=12.0, n=12), runs=5000, seed=32)
    assert large_n.simulation_interval_low > small_n.simulation_interval_low > large_sd.simulation_interval_low
    assert large_n.simulation_interval_high > small_n.simulation_interval_high > large_sd.simulation_interval_high


def test_zero_sd_is_degenerate_for_every_seed_and_one_simulation_is_allowed():
    data = profile(test=(10.0, 40.0, 70.0), sd=0.0)
    first = calculate(data, runs=1, seed=0)
    other = calculate(data, runs=100, seed=2**31 - 1)
    assert first.simulation_interval_low == first.simulation_interval_high
    assert other.simulation_interval_low == other.simulation_interval_high
    assert first.simulation_interval_low == pytest.approx(first.f2_unrounded, abs=1e-12)
    assert other.simulation_interval_low == pytest.approx(first.f2_unrounded, abs=1e-12)
    assert first.simulation_runs == 1


def test_normal_simulation_is_not_silently_clipped_to_physical_dissolution_bounds():
    result = calculate(profile(reference=(0.0, 0.0, 0.0), sd=1000.0, n=2), runs=1000, seed=41)
    assert result.f2_unrounded == 100.0
    assert result.simulation_interval_low < 0.0
    assert result.simulated_fraction_f2_ge_50 < 1.0
    assert_simulation_not_be(result)


def test_maximum_documented_simulation_run_count_is_accepted():
    result = calculate(runs=100000)
    assert result.simulation_runs == 100000
    assert result.simulation_interval_low == result.simulation_interval_high == 100.0


def test_legacy_run_count_alias_is_compatible_but_conflicting_counts_fail():
    result = be.calculate_f2(profile(), bootstrap_runs=7)
    assert result.simulation_runs == result.bootstrap_runs == 7
    assert be.calculate_f2(profile(), simulation_runs=7, bootstrap_runs=7).simulation_runs == 7
    with pytest.raises(ValueError):
        be.calculate_f2(profile(), simulation_runs=7, bootstrap_runs=8)


def test_cv_screen_uses_explicit_first_sample_policy_instead_of_clock_time():
    data = profile(reference=(20.0, 40.0, 60.0), times=(20.0, 30.0, 40.0))
    data["Reference SD"] = [3.0, 5.0, 0.0]  # 15% at first sample; 12.5% later.
    result = calculate(data)
    assert "12.5" in result.cv_flag
    assert "15.0" not in result.cv_flag
    assert result.applicability_issues
    assert_simulation_not_be(result)


@pytest.mark.parametrize("data,expected_text", (
    (profile(n=11), "12"),
    (profile(reference=(0.0, 50.0, 80.0)), "undefined"),
    (profile(reference=(20.0, 90.0, 95.0)), "85"),
    (profile(reference=(20.0, 10.0, 80.0)), "monoton"),
))
def test_applicability_concerns_are_returned_as_warnings_not_acceptance(data, expected_text):
    result = calculate(data)
    assert expected_text in " ".join(result.applicability_issues).lower()
    assert_simulation_not_be(result)


@pytest.mark.parametrize("column", COLUMNS)
def test_missing_columns_are_rejected_instead_of_defaulted(column):
    with pytest.raises((ValueError, TypeError)):
        calculate(profile().drop(columns=[column]))


def test_duplicate_column_names_are_rejected():
    data = pd.concat([profile(), profile()[["Reference n"]]], axis=1)
    with pytest.raises(ValueError, match="(?i)duplicate"):
        calculate(data)


@pytest.mark.parametrize("column", COLUMNS)
@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf, "not numeric", True))
def test_nonfinite_nonnumeric_or_boolean_cells_are_rejected(column, bad):
    data = profile().astype(object)
    data.loc[0, column] = bad
    with pytest.raises((ValueError, TypeError)):
        calculate(data)


@pytest.mark.parametrize("column,bad", (
    ("Time (min)", 0.0), ("Time (min)", -1.0),
    ("Reference Mean (%)", -0.1), ("Reference Mean (%)", 100.1),
    ("Test Mean (%)", -0.1), ("Test Mean (%)", 100.1),
    ("Reference SD", -0.1), ("Test SD", -0.1),
    ("Reference n", 0), ("Reference n", 1), ("Reference n", 2.5),
    ("Test n", 0), ("Test n", 1), ("Test n", 2.5),
))
def test_invalid_cell_ranges_are_rejected(column, bad):
    data = profile().astype(object)
    data.loc[0, column] = bad
    with pytest.raises((ValueError, TypeError)):
        calculate(data)


def test_duplicate_times_and_inconsistent_per_arm_unit_counts_are_rejected():
    duplicate = profile(times=(5.0, 5.0, 20.0))
    with pytest.raises(ValueError):
        calculate(duplicate)
    for column in ("Reference n", "Test n"):
        inconsistent = profile()
        inconsistent.loc[1, column] = 13
        with pytest.raises(ValueError):
            calculate(inconsistent)


@pytest.mark.parametrize("data", (pd.DataFrame(), profile().iloc[:2], None, [], "invalid"))
def test_unusable_profile_types_or_too_few_points_are_rejected(data):
    # Call directly so None is not replaced by the test helper's default fixture.
    with pytest.raises((ValueError, TypeError)):
        be.calculate_f2(data, bootstrap_runs=5)


@pytest.mark.parametrize("runs", (0, -1, 100001, 1.5, True, np.nan, np.inf, "10", None))
def test_invalid_simulation_run_counts_are_rejected(runs):
    with pytest.raises((ValueError, TypeError)):
        calculate(runs=runs)


@pytest.mark.parametrize("seed", (-1, 2**31, 1.5, True, np.nan, np.inf, "10", None))
def test_invalid_seeds_are_rejected(seed):
    with pytest.raises((ValueError, TypeError)):
        calculate(seed=seed)


@pytest.mark.parametrize("backend", ("auto", "invalid", "R", None, 1))
def test_unknown_backend_is_rejected(backend):
    with pytest.raises((ValueError, TypeError)):
        calculate(backend=backend)


def test_default_python_backend_does_not_probe_or_invoke_r():
    with patch("core.bioequivalence._run_r_backend", side_effect=AssertionError("R must not run")) as r:
        result = be.calculate_f2(profile(), bootstrap_runs=10, seed=123)
    assert r.call_count == 0
    assert result.backend == "python"
    assert result.r_backend_used is False


def test_explicit_r_unavailability_never_falls_back_to_python():
    with patch("core.bioequivalence.shutil.which", return_value=None), patch(
        "core.bioequivalence._python_bootstrap_stats", side_effect=AssertionError("No fallback permitted")
    ) as python:
        with pytest.raises(be.BackendUnavailableError, match="(?i)Rscript|unavailable|not installed|not found"):
            calculate(backend="r")
    assert python.call_count == 0


def test_explicit_r_execution_failure_never_falls_back_to_python():
    with patch("core.bioequivalence.shutil.which", return_value="/synthetic/Rscript"), patch(
        "core.bioequivalence.subprocess.run", side_effect=subprocess.CalledProcessError(1, ["Rscript"])
    ), patch("core.bioequivalence._python_bootstrap_stats", side_effect=AssertionError("No fallback permitted")) as python:
        with pytest.raises(be.BackendExecutionError):
            calculate(backend="r")
    assert python.call_count == 0


@pytest.mark.parametrize("field,bad", (
    ("f2", float("nan")), ("f2", 100.0),
    ("ci_low", float("nan")), ("ci_high", float("inf")),
    ("ci_low", 101.0), ("median", -500.0),
    ("p05", 101.0), ("p95", float("nan")),
    ("probability_f2_ge_50", 101.0), ("probability_f2_ge_50", -1.0),
))
def test_corrupt_r_numeric_results_are_rejected_without_python_fallback(field, bad):
    data = profile(test=(10.0, 40.0, 70.0))
    expected = 50.0 * math.log10(100.0 / math.sqrt(101.0))
    result = {
        "f2": expected, "ci_low": expected, "ci_high": expected,
        "median": expected, "p05": expected, "p95": expected,
        "probability_f2_ge_50": 0.0, "backend_version": "Synthetic R test output",
    }
    result[field] = bad
    with patch("core.bioequivalence._run_r_backend", return_value=result), patch(
        "core.bioequivalence._python_bootstrap_stats", side_effect=AssertionError("No fallback permitted")
    ) as python:
        with pytest.raises(be.BackendExecutionError):
            calculate(data, backend="r")
    assert python.call_count == 0


@pytest.mark.parametrize("result", (None, {}, {"f2": 100.0}, "invalid"))
def test_incomplete_r_outputs_are_rejected_without_python_fallback(result):
    with patch("core.bioequivalence._run_r_backend", return_value=result), patch(
        "core.bioequivalence._python_bootstrap_stats", side_effect=AssertionError("No fallback permitted")
    ) as python:
        with pytest.raises(be.BackendExecutionError):
            calculate(backend="r")
    assert python.call_count == 0


def test_r_simulation_quantiles_above_mathematical_f2_maximum_are_rejected():
    result = {
        "f2": 100.0, "ci_low": 101.0, "p05": 101.0, "median": 101.0,
        "p95": 101.0, "ci_high": 101.0, "probability_f2_ge_50": 100.0,
        "backend_version": "Synthetic R test output",
    }
    with patch("core.bioequivalence._run_r_backend", return_value=result):
        with pytest.raises(be.BackendExecutionError):
            calculate(backend="r")


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not installed; optional R numerics not exercised")
def test_actual_r_backend_matches_analytic_zero_sd_profile():
    result = calculate(profile(test=(10.0, 40.0, 70.0)), backend="r", runs=20)
    expected = 50.0 * math.log10(100.0 / math.sqrt(101.0))
    assert result.f2_unrounded == pytest.approx(expected, abs=1e-10)
    assert result.simulation_interval_low == pytest.approx(expected, abs=1e-10)
    assert result.simulation_interval_high == pytest.approx(expected, abs=1e-10)
    assert result.backend == "r" and result.r_backend_used
    assert result.backend_version


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not installed; optional R numerics not exercised")
def test_actual_r_backend_seed_is_reproducible_and_point_f2_matches_python():
    data = profile(test=(21.0, 48.0, 83.0), sd=4.0)
    first = calculate(data, backend="r", runs=500, seed=771)
    repeated = calculate(data, backend="r", runs=500, seed=771)
    changed = calculate(data, backend="r", runs=500, seed=772)
    python = calculate(data, backend="python", runs=500, seed=771)
    assert asdict(first) == asdict(repeated)
    assert first.f2_unrounded == pytest.approx(python.f2_unrounded, abs=1e-10)
    assert (first.simulation_interval_low, first.simulation_interval_high) != (changed.simulation_interval_low, changed.simulation_interval_high)


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not installed; optional R numerics not exercised")
def test_actual_r_and_python_sample_mean_distributions_agree_within_monte_carlo_tolerance():
    data = profile(reference=(50.0, 50.0, 50.0), sd=6.0, n=12)
    python = calculate(data, backend="python", runs=20000, seed=910)
    r = calculate(data, backend="r", runs=20000, seed=910)
    # Backends intentionally use their own RNG streams, so compare distribution
    # summaries within finite-simulation tolerance instead of exact random draws.
    assert r.simulation_interval_low == pytest.approx(python.simulation_interval_low, abs=0.5)
    assert r.simulation_interval_high == pytest.approx(python.simulation_interval_high, abs=0.5)


def test_export_is_serializable_and_has_explicit_simulation_limitations():
    result = calculate(profile(sd=3.0), runs=20, seed=213)
    exported = result.to_dict()
    assert json.loads(json.dumps(exported, allow_nan=False)) == exported
    assert_simulation_not_be(result)
    assert result.bootstrap_runs == result.simulation_runs
    assert result.probability_f2_ge_50 == pytest.approx(result.simulated_fraction_f2_ge_50 * 100.0, abs=0.051)
    for legacy in ("ci_low", "ci_high", "bootstrap_median", "bootstrap_p05", "bootstrap_p95", "bootstrap_runs", "probability_f2_ge_50", "fda_risk", "fda_decision", "fda_next_action"):
        assert legacy not in exported


@pytest.mark.parametrize("column", be.REQUIRED_COLUMNS)
def test_complex_numeric_cells_are_rejected_without_discarding_imaginary_parts(column):
    data = profile().astype(object)
    data.loc[0, column] = complex(data.loc[0, column], 500)
    with pytest.raises(be.ProfileValidationError, match="real|complex"):
        calculate(data)


@pytest.mark.parametrize("version", (None, "", "nan", "None", float("nan")))
def test_r_requires_recorded_nonplaceholder_backend_version(version):
    output = {"f2": 100.0, "ci_low": 100.0, "p05": 100.0, "median": 100.0,
              "p95": 100.0, "ci_high": 100.0, "probability_f2_ge_50": 100.0,
              "backend_version": version}
    with patch("core.bioequivalence._run_r_backend", return_value=output):
        with pytest.raises(be.BackendExecutionError):
            calculate(backend="r")


def test_point_formula_rejects_complex_arrays_without_silently_casting():
    with pytest.raises(be.ProfileValidationError, match="real|complex"):
        be._f2_from_means(np.array([20+500j, 40, 60]), [20, 40, 60])
