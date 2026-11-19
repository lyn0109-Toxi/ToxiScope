"""UI regressions for descriptive dissolution f2 and labelled simulation."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core.bioequivalence import DEFAULT_DISSOLUTION_PROFILE


APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def _start(profile=None):
    app = AppTest.from_file(APP)
    if profile is not None:
        app.session_state["be_profile"] = profile
    app.run(timeout=30)
    assert not app.exception
    app.segmented_control(key="analysis_view").set_value("용출 비교").run(timeout=30)
    assert not app.exception
    return app


def _calculate(app):
    next(button for button in app.button if button.label == "Calculate f2 and simulation").click().run(timeout=30)
    assert not app.exception
    return app


def _configured():
    app = _start()
    app.number_input(key="be_simulation_runs").set_value(200)
    app.number_input(key="be_seed").set_value(123)
    app.run(timeout=30)
    return _calculate(app)


def _text(app):
    return "\n".join(str(element.value) for elements in
                     (app.markdown, app.caption, app.warning, app.info, app.error, app.success)
                     for element in elements)


def _assert_cleared(app):
    assert not app.exception
    assert app.session_state["be_result"] is None
    assert app.session_state["be_export"] is None
    assert not any(metric.label == "f2 · 기술 통계" for metric in app.metric)
    assert not any(item.proto.label == "Download dissolution calculation + provenance"
                   for item in app.get("download_button"))
    charts = [*app.get("vega_lite_chart"), *app.get("arrow_vega_lite_chart")]
    assert not any(item.proto.alt.startswith("현재 입력의 정규모형") for item in charts)


def test_simulation_labels_and_export_preserve_inputs_without_claiming_bioequivalence():
    app = _configured()
    result = app.session_state["be_result"]
    assert result.schema == "dissolution-f2-simulation.v2"
    assert result.backend == "python"
    assert result.seed == 123
    assert result.simulation_runs == 200
    assert result.clinical_be_established is False
    assert result.regulatory_acceptance == "Not established"
    labels = {metric.label for metric in app.metric}
    assert {"f2 · 기술 통계", "모형 시뮬레이션 2.5–97.5%", "조건부 시뮬레이션 빈도 · f2 ≥ 50"} <= labels
    assert not any("bootstrap" in label.lower() or "confidence" in label.lower() or "probability" in label.lower()
                   for label in labels)
    text = _text(app)
    assert "설명용 합성 예시" in text
    assert "독립 정규모형 시뮬레이션" in text
    assert "신뢰구간 또는 생물학적 동등성 확률이 아닙니다" in text
    assert "FDA strategy decision" not in text
    assert "FDA submission risk" not in text
    assert not app.success
    assert result.applicability_status == "Review needed"
    assert result.applicability_issues
    assert all(issue in text for issue in result.applicability_issues)
    export = app.session_state["be_export"]
    assert export["schema"] == "toxiscope-dissolution-review.v2"
    assert export["profile"] == app.session_state["be_profile"].to_dict("records")
    assert export["calculation_inputs"] == {"simulation_runs": 200, "seed": 123, "backend": "python"}
    assert export["versions"]["backend_version"]
    assert export["versions"]["numpy"]
    assert export["result"]["profile_fingerprint"]
    assert export["simulation_assumptions"]
    assert export["limitations"]
    assert export["clinical_be_established"] is False
    assert export["regulatory_acceptance"] == "Not established"
    assert not {"ci_low", "ci_high", "probability_f2_ge_50", "bootstrap_runs", "fda_decision"} & export["result"].keys()
    first_result = export["result"]
    _calculate(app)
    assert app.session_state["be_export"]["result"] == first_result


@pytest.mark.parametrize("control,value", [
    ("seed", 124), ("runs", 300), ("backend", "R"), ("project", "SYNTHETIC-OTHER-PROJECT"),
    ("compound", "Synthetic changed compound label"),
])
def test_changed_calculation_controls_or_project_clear_old_result_and_download(control, value):
    app = _configured()
    if control == "seed":
        app.number_input(key="be_seed").set_value(value).run(timeout=30)
    elif control == "runs":
        app.number_input(key="be_simulation_runs").set_value(value).run(timeout=30)
    elif control == "backend":
        app.selectbox(key="be_backend").set_value(value).run(timeout=30)
    elif control == "project":
        next(item for item in app.text_input if item.label == "Project ID").set_value(value).run(timeout=30)
    else:
        next(item for item in app.text_input if item.label == "Compound Name").set_value(value).run(timeout=30)
    _assert_cleared(app)


def test_profile_editor_change_clears_old_result_in_the_same_run():
    app = _configured()
    editor = next(item for item in app.dataframe if item.key and item.key.startswith("be_profile_editor:"))
    app.session_state[editor.key] = {
        "edited_rows": {0: {"Test Mean (%)": 29.0}}, "added_rows": [], "deleted_rows": [],
    }
    app.run(timeout=30)
    _assert_cleared(app)
    assert app.session_state["be_profile"].iloc[0]["Test Mean (%)"] == 29.0


@pytest.mark.parametrize("invalid_kind", ["text_n", "missing_sd", "missing_value", "duplicate_time", "above_100"])
def test_invalid_profiles_display_validation_errors_before_charting(invalid_kind):
    profile = DEFAULT_DISSOLUTION_PROFILE.copy()
    if invalid_kind == "text_n":
        profile["Reference n"] = profile["Reference n"].astype(object)
        profile.loc[1, "Reference n"] = "twelve"
    elif invalid_kind == "missing_sd":
        profile = profile.drop(columns=["Reference SD"])
    elif invalid_kind == "missing_value":
        profile.loc[1, "Test SD"] = np.nan
    elif invalid_kind == "duplicate_time":
        profile.loc[1, "Time (min)"] = profile.loc[0, "Time (min)"]
    else:
        profile.loc[1, "Test Mean (%)"] = 101
    app = _start(profile)
    assert any("Profile validation failed" in item.value for item in app.error)
    assert next(button for button in app.button if button.label == "Calculate f2 and simulation").disabled
    charts = [*app.get("vega_lite_chart"), *app.get("arrow_vega_lite_chart")]
    assert not any(item.proto.alt.startswith(("입력된 기준품과 시험품", "입력된 각 측정 시점", "현재 입력의 정규모형"))
                   for item in charts)
    assert not any(metric.label == "f2 · 기술 통계" for metric in app.metric)


def test_explicit_r_failure_is_visible_and_does_not_produce_a_python_result():
    app = _start()
    app.selectbox(key="be_backend").set_value("R").run(timeout=30)
    with patch("core.bioequivalence.calculate_f2", side_effect=RuntimeError("Synthetic R backend failure")) as calculation:
        _calculate(app)
    assert calculation.call_count == 1
    assert calculation.call_args.kwargs["backend"] == "r"
    assert any("Synthetic R backend failure" in item.value for item in app.error)
    _assert_cleared(app)


def test_legacy_cached_be_output_is_cleared_without_clearing_ntr_evidence():
    app = AppTest.from_file(APP)
    app.session_state["be_result"] = SimpleNamespace(f2=90, cv_flag="Acceptable", fda_risk="Low")
    app.session_state["be_export"] = {"legacy": True}
    app.session_state["ntr_docs"] = ["Preserved NTR source state"]
    app.run(timeout=30)
    _assert_cleared(app)
    assert app.session_state["ntr_docs"] == ["Preserved NTR source state"]
    app.segmented_control(key="analysis_view").set_value("용출 비교").run(timeout=30)
    _assert_cleared(app)
    assert any("이전 용출 결과를 지웠습니다" in item.value for item in app.info)


def test_threshold_message_uses_unrounded_f2_at_the_boundary():
    profile = DEFAULT_DISSOLUTION_PROFILE.copy()
    profile["Reference Mean (%)"] = 30.0
    profile["Test Mean (%)"] = 30.0 + np.sqrt(99) + 0.00001
    app = _calculate(_start(profile))
    assert app.session_state["be_result"].f2 == 50.0
    assert app.session_state["be_result"].descriptive_threshold_met is False
    assert any("기술적 기준: f2 < 50" in item.value for item in app.caption)


def test_duplicate_column_profile_is_rejected_before_the_editor_can_crash():
    profile = DEFAULT_DISSOLUTION_PROFILE.copy()
    profile.columns = ["Time (min)", "Reference Mean (%)", "Reference SD", "Reference n",
                       "Test Mean (%)", "Test SD", "Test SD"]
    app = _start(profile)
    assert any("seven unique dissolution columns" in item.value for item in app.error)
    assert not any(metric.label == "f2 · 기술 통계" for metric in app.metric)
