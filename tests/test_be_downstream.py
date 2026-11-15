"""Dissolution summaries must not become clinical or regulatory decisions."""
import json
from types import SimpleNamespace

import pytest

from core.bioequivalence import DEFAULT_DISSOLUTION_PROFILE, calculate_f2
from core.ontology import (
    FDA_GUIDANCE_MAP,
    build_strategy_snapshot,
    build_submission_workflow,
    classify_be_risk,
)


@pytest.mark.parametrize("f2", [100.0, 75.0, 50.0, 49.99, 0.0, -5.0])
@pytest.mark.parametrize("cv_flag", ["Acceptable", "Not assessed", "High variability"])
def test_descriptive_f2_and_cv_never_establish_low_or_high_clinical_be_risk(f2, cv_flag):
    result = SimpleNamespace(f2=f2, cv_flag=cv_flag)
    risk, action = classify_be_risk(result)
    assert risk == "Review needed"
    assert "do not establish clinical bioequivalence" in action
    assert "regulatory acceptance" in action
    assert "product-specific review" in action
    snapshot = build_strategy_snapshot(be_result=result)
    assert snapshot["bioequivalence_risk"] == "Review needed"
    assert snapshot["overall_risk"] == "Review needed"
    be_action = next(item for item in snapshot["module_actions"] if item["Module"] == "Dissolution Profile Comparison")
    assert be_action["Risk"] == "Review needed"
    assert "f2 is acceptable" not in be_action["Next action"]


@pytest.mark.parametrize("result", [{}, {"f2": 100.0, "fda_risk": "Low"}, SimpleNamespace(), False])
def test_incomplete_or_legacy_result_cannot_be_promoted_to_clinical_be_clearance(result):
    assert classify_be_risk(result)[0] == "Review needed"
    workflow = build_submission_workflow(be_result=result)
    assert workflow[3]["Status"] == "Review needed"


def test_no_dissolution_result_remains_pending():
    risk, action = classify_be_risk(None)
    assert risk == "Not started"
    assert "descriptive f2" in action
    assert build_submission_workflow()[3]["Status"] == "Pending"


def test_workflow_keeps_completed_calculation_under_review_and_json_preserves_limitations():
    result = SimpleNamespace(f2=100.0, cv_flag="Acceptable")
    workflow = build_submission_workflow(be_result=result)
    be_step = workflow[3]
    assert be_step["Status"] == "Review needed"
    assert "Descriptive f2" in be_step["Output"]
    assert "simulation and assumptions" in be_step["Output"]
    assert "clinical BE and regulatory acceptance are not established" in be_step["Output"]
    exported = json.loads(json.dumps({"strategy": build_strategy_snapshot(be_result=result), "workflow": workflow}))
    assert exported["strategy"]["bioequivalence_risk"] == "Review needed"
    assert exported["workflow"][3]["Status"] == "Review needed"
    assert "BE strategy decision" not in exported["workflow"][3]["Output"]


def test_fda_ich_guidance_is_background_without_implied_validation_or_acceptance():
    be_map = next(row for row in FDA_GUIDANCE_MAP if row["Platform module"] == "Dissolution Profile Comparison")
    assert "Background references" in be_map["Primary basis"]
    assert "do not validate" in be_map["Primary basis"]
    assert "regulatory acceptance are not established" in be_map["Decision use"]
    assert "support BE or biowaiver strategy" not in be_map["Decision use"]


def test_dissolution_result_export_retains_assumptions_and_unestablished_clinical_status():
    result = calculate_f2(DEFAULT_DISSOLUTION_PROFILE.copy(), simulation_runs=128, seed=1729, backend="python")
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert payload["schema"] == "dissolution-f2-simulation.v2"
    assert payload["clinical_be_established"] is False
    assert payload["regulatory_acceptance"] == "Not established"
    assert payload["simulation_assumptions"]
    assert payload["limitations"]
    assert payload["backend_version"]
    assert payload["seed"] == 1729
    assert payload["simulation_runs"] == 128
    assert len(payload["profile_fingerprint"]) == 64
    assert not any("probability" in key.lower() for key in payload)
    assert not {"ci_low", "ci_high", "bootstrap_median", "bootstrap_runs", "fda_decision", "fda_risk"}.intersection(payload)
    assert classify_be_risk(result)[0] == "Review needed"
    assert build_submission_workflow(be_result=result)[3]["Status"] == "Review needed"
