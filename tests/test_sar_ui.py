"""Regression coverage for the screening labels shown in the local app."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def _assess(smiles):
    app = AppTest.from_file(APP).run(timeout=30)
    assert not app.exception
    next(t for t in app.text_input if t.label == "Compound Name").set_value("Screening regression")
    app.text_area(key="smiles").set_value(smiles)
    next(b for b in app.button if b.label == "구조 신호 분석").click().run(timeout=30)
    assert not app.exception
    return app


def _visible_text(app):
    kinds = (app.markdown, app.caption, app.info, app.warning, app.error, app.success, app.text_area)
    return "\n".join(str(element.value) for elements in kinds for element in elements)


def test_no_alert_ui_requires_review_without_safe_or_negative_calls():
    app = _assess("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert app.session_state["results"]["status"] == "review"
    text = _visible_text(app)
    assert "does not establish safety" in text
    assert "No trained statistical model is implemented" in text
    assert "Validated through Harness" not in text
    assert "Class 5" not in text
    assert "Concordant negative" not in text
    assert not any("No " in item.value and "alert" in item.value for item in app.success)
    assert any(m.label == "Statistical model" and m.value == "Not implemented" for m in app.metric)


def test_rule_match_ui_has_atom_mapping_and_no_probability():
    app = _assess("c1ccc(cc1)[N+](=O)[O-]")
    assert app.session_state["results"]["status"] == "alert"
    assert app.session_state["results"]["expert_alerts"][0]["matched_atoms"]
    text = _visible_text(app)
    assert "Nitro Group" in text
    assert "Probability:" not in text
    assert "estimated probability" not in text
    assert "QSAR Dual-Method" not in text
    assert "No statistical fragment alert identified" not in text


def test_local_historical_dossier_is_visibly_unverified():
    app = _assess("Nc1ccccc1")
    assert app.session_state["results"]["status"] == "review"
    text = _visible_text(app)
    assert "Local historical toxicology dossier (unverified)" in text
    assert "have not been independently verified" in text
    assert "Confirmed in historical toxicology studies" not in text
    assert "Class 2" not in text


def test_invalid_smiles_ui_reports_error_without_crashing():
    app = _assess("not-a-smiles")
    assert app.session_state["results"]["status"] == "error"
    assert any("Invalid SMILES" in item.value for item in app.error)


def test_legacy_cached_screen_is_cleared_without_clearing_other_workspaces():
    app = AppTest.from_file(APP).run(timeout=30)
    app.segmented_control(key="analysis_view").set_value("용출 비교").run(timeout=30)
    next(b for b in app.button if b.label == "Calculate f2 and simulation").click().run(timeout=30)
    assert not app.exception
    be_result = app.session_state["be_result"]
    app.segmented_control(key="analysis_view").set_value("구조 신호").run(timeout=30)
    app.session_state["results"] = {"status": "safe", "class": "ICH M7 Class 5"}
    app.session_state["evidence_package"] = {"assessment": {"old_probability": 0.82}}
    app.session_state["degradants"] = [{"status": "safe"}]
    app.session_state["known_impurities"] = [{"class": 5}]
    app.session_state["ntr_docs"] = ["Existing NTR document state"]
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state["results"] is None
    assert app.session_state["evidence_package"] is None
    assert app.session_state["degradants"] == []
    assert app.session_state["known_impurities"] == []
    assert app.session_state["ntr_docs"] == ["Existing NTR document state"]
    assert app.session_state["be_result"].f2 == be_result.f2
    assert any("Previous structural assessment output was cleared" in item.value for item in app.info)
    app.segmented_control(key="analysis_view").set_value("용출 비교").run(timeout=30)
    assert not app.exception
    assert any(m.label == "f2 · 기술 통계" and float(m.value) == be_result.f2 for m in app.metric)
    assert any(b.label == "Calculate f2 and simulation" for b in app.button)


def test_orphaned_legacy_package_is_cleared_before_ntr_context_export():
    app = AppTest.from_file(APP)
    app.session_state["results"] = None
    app.session_state["evidence_package"] = {"assessment": {"status": "safe", "statistical_alerts": [{"probability": 0.82}]}}
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state["evidence_package"] is None
