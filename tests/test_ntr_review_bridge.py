"""Synthetic intake/export checks; not scientific or customer validation."""
import copy
import json
from dataclasses import replace

import pytest

from nora.evidence import EvidenceAssertion, extract_document
from nora.evidence_review import build_project_review, application_blockers
from ntr_bridge import empty_input, parse_review_input, reviewed_assessment, build_review_export


def test_empty_and_partial_json_never_restore_negative_or_valid_control_defaults():
    for payload in ({}, {"nam_assay": {"use_nam": True}}, {"ai_model": {"use_ai": True}}):
        inp = parse_review_input(payload, compound_name="Synthetic A")
        assert inp.ai_model.result == ""
        assert inp.nam_assay.result == ""
        assert inp.nam_assay.positive_control == "불명확"
        assert inp.nam_assay.negative_control == "불명확"
        assert inp.ai_model.external_validation == "불명확"
        assert not inp.supporting_evidence.evidence_traceable
        assert not inp.supporting_evidence.version_locked
        if not payload:
            assert not inp.ai_model.use_ai and not inp.nam_assay.use_nam
            _, result = reviewed_assessment(inp, [], [])
            assert result.evidence_role_code == "R0"
            assert result.evidence_stream_count == 0


@pytest.mark.parametrize("payload", [
    [], None, {"unknown": {}}, {"nam_assay": []},
    {"nam_assay": {"use_nam": "false"}}, {"ai_model": {"use_ai": 1}},
    {"ai_model": {"probability_percent": float("nan")}},
    {"ai_model": {"sensitivity_percent": 101}},
    {"ai_model": {"training_sample_size": 1.5}},
    {"ai_model": {"auroc": float("inf")}},
    {"nam_assay": {"cell_types": "Human hepatocytes"}},
    {"context_of_use": {"decision_consequence": 99}},
])
def test_invalid_manual_types_or_numbers_block_legacy_input_without_supplying_defaults(payload):
    with pytest.raises(ValueError):
        parse_review_input(payload)


def test_explicit_manual_entries_are_retained_but_not_source_linked():
    payload = {"nam_assay": {"use_nam": True, "result": "양성"},
               "context_of_use": {"target_endpoint": "초기 간독성"}}
    before = copy.deepcopy(payload)
    inp = parse_review_input(payload)
    assert inp.nam_assay.use_nam
    assert inp.nam_assay.result == "양성"
    assert inp.nam_assay.positive_control == "불명확"
    report = build_project_review([], [], assessment_input=payload)
    nam_row = next(row for row in report["coverage"] if row["field_path"] == "nam_assay.result")
    assert nam_row["review_state"] != "reviewed_source_linked"
    assert payload == before


def _claim(kind="observation", value="positive", scope=None):
    document = extract_document(f"Synthetic assay result: {value}".encode(), "synthetic.txt")
    document.source_kind = "customer_evidence"
    assertion = EvidenceAssertion(
        assertion_id="AST-synthetic", category="synthetic", field_path="nam_assay.result",
        label_ko="synthetic", proposed_value=value, value_type="str",
        source_document_id=document.document_id, source_document_name=document.name,
        source_location=document.segments[0].location, source_excerpt=document.segments[0].text,
        confidence=0.5, evidence_kind=kind, scope=scope or {}, review_status="승인",
    )
    return assertion, document


def test_outcome_without_comparison_subject_endpoint_system_requires_scope_review():
    assertion, document = _claim()
    blockers = application_blockers([assertion], [document])
    assert "outcome_scope_review_required" in blockers[assertion.assertion_id]
    with pytest.raises(ValueError, match="Source review"):
        reviewed_assessment(empty_input(), [assertion], [document])


def test_different_study_scopes_cannot_silently_merge_into_single_legacy_card():
    scope = {"subject": "Synthetic A", "endpoint": "hepatotoxicity", "system": "cells", "study_id": "Study 1"}
    first, document = _claim(scope=scope)
    second = replace(first, assertion_id="AST-second", scope={**scope, "study_id": "Study 2"})
    blockers = application_blockers([first, second], [document])
    assert all("multiple_scopes_require_selection" in blockers[a.assertion_id] for a in [first, second])
    with pytest.raises(ValueError, match="Source review"):
        reviewed_assessment(empty_input(), [first, second], [document])


def test_opposite_prediction_observation_labels_are_kept_as_different_evidence_types():
    scope = {"subject": "Synthetic A", "endpoint": "hepatotoxicity", "system": "cells"}
    observation, document = _claim(scope=scope)
    prediction = replace(observation, assertion_id="AST-model", field_path="ai_model.result",
                         proposed_value="negative", evidence_kind="prediction")
    report = build_project_review([prediction, observation], [document])
    differences = [f for f in report["findings"] if f["kind"] == "prediction_observation_difference"]
    assert differences
    assert {p["evidence_kind"] for p in differences[0]["provenance"]} == {"prediction", "observation"}
    assert report["summary"]["prediction_observation_differences"] == 1
    assert "toxicity_direction" not in report
    assert "different evidence types" in differences[0]["detail"]


def test_candidate_scope_mismatch_never_silently_asserts_identity_equivalence():
    assertion, document = _claim(scope={"subject": "Synthetic B"})
    report = build_project_review([assertion], [document], compound_name="Synthetic A")
    assert any(f["kind"] == "project_scope_mismatch" for f in report["findings"])
    assert report["decision"] == "Expert review required"


def test_disabled_optional_stream_is_not_counted_as_a_missing_evidence_requirement():
    report = build_project_review([], [], assessment_input={"nam_assay": {"use_nam": False}})
    row = next(row for row in report["coverage"] if row["field_path"] == "nam_assay.result")
    assert row["review_state"] == "not_requested"
    assert not any(f["kind"] == "coverage_gap" and f["field_path"] == "nam_assay.result" for f in report["findings"])


def test_review_export_available_with_blocked_legacy_evaluation_and_retains_snapshot_provenance():
    assertion, document = _claim()
    report = build_project_review([assertion], [document], project_id="Synthetic P")
    export = build_review_export(report, [document], [assertion], project_id="Synthetic P",
        evaluation_error="Synthetic conflict: expert review required", raw_manual_input="{invalid",
        package={"assessment": {"screening_method": "Rule-based structural alert screening"}})
    assert export["schema"] == "toxiscope-ntr-review-2"
    assert export["project_review"]["schema"] == "toxiscope-ntr-project-review.v1"
    assert export["result"] is None
    assert export["evaluation_error"]
    assert export["documents"][0]["sha256"] == document.sha256
    assert export["structural_context"]["kind"] == "structural_screen_context_not_observed_toxicity"
    assert json.loads(json.dumps(export)) == export
    assertion.proposed_value = "Changed after export"
    document.segments.clear()
    report["findings"].clear()
    assert export["assertions"][0]["proposed_value"] == "positive"
    assert export["documents"][0]["segments"]
    assert export["project_review"]["findings"]


@pytest.mark.parametrize("kind,path", [("observation", "nam_assay.result"), ("prediction", "ai_model.result")])
def test_single_result_for_other_endpoint_is_flagged_and_cannot_fill_requested_card(kind, path):
    scope = {"subject": "Synthetic A", "endpoint": "Cardiotoxicity", "system": "cardiomyocytes"}
    assertion, document = _claim(kind=kind, scope=scope)
    assertion.field_path = path
    inp = empty_input("Synthetic A")
    before = copy.deepcopy(inp)
    report = build_project_review([assertion], [document], compound_name="Synthetic A", assessment_input=inp)
    assert any(f["kind"] == "project_scope_mismatch" and f["field_path"] == "context_of_use.target_endpoint" for f in report["findings"])
    assert "requested_endpoint_mismatch" in report["application_blockers"][assertion.assertion_id]
    row = next(row for row in report["coverage"] if row["field_path"] == path)
    assert row["review_state"] != "reviewed_source_linked"
    with pytest.raises(ValueError, match="Source review"):
        reviewed_assessment(inp, [assertion], [document])
    assert inp == before


@pytest.mark.parametrize("endpoint", [["Cardiotoxicity"], {"name": "Cardiotoxicity"}, True, 17])
def test_malformed_scope_types_still_produce_serializable_review_findings(endpoint):
    report = build_project_review([], [], assessment_input={"context_of_use": {"target_endpoint": endpoint}})
    assert report["scope"]["target_endpoint"] == "Not specified"
    assert any(f["kind"] == "input_issue" for f in report["findings"])
    assert json.loads(json.dumps(report)) == report


def test_unrepresentable_manual_float_is_rejected_instead_of_crashing():
    with pytest.raises(ValueError):
        parse_review_input({"ai_model": {"sensitivity_percent": 10 ** 400}})


def test_reviewed_context_edit_cannot_rebind_outcome_to_another_endpoint():
    inp = empty_input("Synthetic A")
    scope = {"subject": "Synthetic A", "endpoint": inp.context_of_use.target_endpoint, "system": "cells"}
    outcome, document = _claim(scope=scope)
    context_claim = replace(outcome, assertion_id="AST-context", field_path="context_of_use.target_endpoint",
                            evidence_kind="context", proposed_value="Cardiotoxicity")
    with pytest.raises(ValueError, match="final assessment scope"):
        reviewed_assessment(inp, [outcome, context_claim], [document])
    assert inp.context_of_use.target_endpoint == "초기 간독성"


@pytest.mark.parametrize("path,value,value_type", [
    ("ai_model.sensitivity_percent", "1000", "float"),
    ("ai_model.false_negative_rate_percent", "-5", "float"),
    ("ai_model.auroc", "2", "float"),
    ("ai_model.training_sample_size", "-1", "int"),
    ("context_of_use.model_influence", "6", "int"),
])
def test_impossible_reviewed_metrics_remain_for_correction_without_application(path, value, value_type):
    assertion, document = _claim()
    assertion.field_path, assertion.proposed_value, assertion.value_type = path, value, value_type
    assertion.evidence_kind = "context"
    inp = empty_input()
    report = build_project_review([assertion], [document])
    row = next(row for row in report["coverage"] if row["field_path"] == path)
    assert row["review_state"] == "value_review_required"
    assert "value_out_of_range" in report["application_blockers"][assertion.assertion_id]
    assert report["assertion_inventory"][0]["proposed_value"] == value
    with pytest.raises(ValueError):
        reviewed_assessment(inp, [assertion], [document])


def test_case_sensitive_exposure_units_are_not_normalized_into_matched_scope():
    scope = {"subject": "Synthetic A", "endpoint": "hepatotoxicity", "system": "cells", "exposure": "10 mM"}
    left, document = _claim(scope=scope)
    right = replace(left, assertion_id="AST-unit-difference", proposed_value="negative",
                    scope={**scope, "exposure": "10 mm"})
    report = build_project_review([left, right], [document])
    differences = [f for f in report["findings"] if f["kind"] == "scope_difference"]
    assert differences and "exposure" in differences[0]["differing_scope_keys"]
    assert not any(f["kind"] == "possible_conflict" for f in report["findings"])
