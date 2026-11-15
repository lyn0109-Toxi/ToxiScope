"""Synthetic assertion integrity checks; no claims of scientific validation."""
import copy
import json

import pytest

from nora.assertions import (
    apply_reviewed_assertions,
    assertion_table_rows,
    assertion_value_issues,
    assertions_from_table_rows,
    extract_assertions,
    metadata_issues,
    reviewed_assertion_conflicts,
    validate_assertion_value,
)
from nora.evidence import EVIDENCE_KIND_OPTIONS, SCOPE_KEYS, EvidenceAssertion, SourceSegment, extract_document
from nora.models import AssessmentInput


def claims_for(text):
    document = extract_document(text.encode(), "synthetic.txt")
    document.source_kind = "customer_evidence"
    return document, extract_assertions(document)


def claim(path="ai_model.model_name", value="Synthetic Model", value_type="str"):
    return EvidenceAssertion(
        assertion_id="AST-synthetic", category="Synthetic", field_path=path,
        label_ko="Synthetic", proposed_value=value, value_type=value_type,
        source_document_id="DOC-synthetic", source_document_name="synthetic.txt",
        source_location="line 1", source_excerpt="Synthetic evidence", confidence=0.5,
        review_status="승인", evidence_kind="prediction" if isinstance(path, str) and path.startswith("ai_model.") else "unspecified",
    )


def test_legacy_json_loads_conservatively_without_fabricating_original_value():
    payload = claim().to_dict()
    for key in ("evidence_kind", "scope", "extracted_value", "source_segment_id"):
        payload.pop(key)
    restored = EvidenceAssertion.from_dict(json.loads(json.dumps(payload)))
    assert restored.evidence_kind == "unspecified"
    assert restored.scope == {}
    assert restored.extracted_value == ""
    assert restored.source_segment_id == ""


def test_new_metadata_roundtrips_through_json():
    original = claim()
    original.scope = {key: f"synthetic-{key}" for key in SCOPE_KEYS}
    original.extracted_value = "Synthetic extracted model"
    restored = EvidenceAssertion.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    assert tuple(EVIDENCE_KIND_OPTIONS) == ("unspecified", "observation", "prediction", "context")


def test_extraction_does_not_auto_promote_nam_to_observation():
    _, assertions = claims_for("Product name: Synthetic Candidate\nModel name: Synthetic Model\nNAM assay result: negative")
    by_path = {item.field_path: item for item in assertions}
    assert by_path["product.product_name"].evidence_kind == "context"
    assert by_path["ai_model.model_name"].evidence_kind == "prediction"
    assert by_path["nam_assay.result"].evidence_kind == "unspecified"
    assert "ai_model.result" not in by_path
    assert all(item.scope == {} for item in assertions)
    assert all(item.extracted_value == item.proposed_value for item in assertions)
    assert all(item.source_segment_id for item in assertions)


@pytest.mark.parametrize("text", (
    "NAM assay predicted result: negative",
    "Assay predicted result: positive",
    "NAM assay prediction: negative",
    "Predicted NAM assay result: negative",
    "AI model prediction of NAM assay result: negative",
    "Generic result: negative",
    "NAM assay result is not positive",
    "Model prediction result not negative",
))
def test_prediction_only_or_unattributed_assay_text_is_not_an_observation(text):
    _, assertions = claims_for(text)
    assert not [item for item in assertions if item.field_path in {"ai_model.result", "nam_assay.result"}]


def test_assay_outcome_does_not_absorb_control_outcomes_across_sentences():
    _, assertions = claims_for("Assay result: negative. Positive control valid; negative control valid.")
    results = [item for item in assertions if item.field_path == "nam_assay.result"]
    assert [item.proposed_value for item in results] == ["음성"]
    assert not [item for item in assertions if item.field_path == "ai_model.result"]


def test_explicit_prediction_and_assay_outcomes_remain_distinct():
    _, assertions = claims_for("Model prediction result: positive; NAM assay result: negative")
    results = {item.field_path: item for item in assertions if item.field_path.endswith(".result")}
    assert results["ai_model.result"].proposed_value == "양성 / 위험 신호"
    assert results["ai_model.result"].evidence_kind == "prediction"
    assert results["nam_assay.result"].proposed_value == "음성"
    assert results["nam_assay.result"].evidence_kind == "unspecified"


def test_comma_separated_ai_prediction_does_not_erase_preceding_assay_result():
    _, assertions = claims_for("NAM assay result: negative, Model prediction result: positive")
    results = {item.field_path: item.proposed_value for item in assertions if item.field_path.endswith(".result")}
    assert results == {"nam_assay.result": "음성", "ai_model.result": "양성 / 위험 신호"}


def test_repeated_explicit_values_in_one_segment_remain_visible_for_conflicts():
    document, assertions = claims_for("Sensitivity: 0.95; Sensitivity: 0.25")
    results = [item for item in assertions if item.field_path == "ai_model.sensitivity_percent"]
    assert [item.proposed_value for item in results] == ["0.95", "0.25"]
    assert all(item.source_location == document.segments[0].location for item in results)
    for item in results:
        item.review_status = "승인"
    assert reviewed_assertion_conflicts(results) == {"ai_model.sensitivity_percent": ["0.25", "0.95"]}
    with pytest.raises(ValueError, match="conflicting"):
        apply_reviewed_assertions(AssessmentInput(), results)


def test_identical_explicit_repeats_and_distinct_segment_ids_are_retained():
    document, assertions = claims_for("Model name: Synthetic Model\nModel name: Synthetic Model")
    models = [item for item in assertions if item.field_path == "ai_model.model_name"]
    assert len(models) == 2
    assert len({item.assertion_id for item in models}) == 2
    document.segments = [
        SourceSegment("SEG-A", "page 1", "Model name: Synthetic Model"),
        SourceSegment("SEG-B", "page 1", "Model name: Synthetic Model"),
    ]
    models = [item for item in extract_assertions(document) if item.field_path == "ai_model.model_name"]
    assert {item.source_segment_id for item in models} == {"SEG-A", "SEG-B"}


@pytest.mark.parametrize("path,value,value_type,code", (
    ("ai_model.use_ai", "maybe", "bool", "value_invalid"),
    ("ai_model.use_ai", "nan", "bool", "value_invalid"),
    ("ai_model.auroc", "nan", "float", "value_invalid"),
    ("ai_model.auroc", "inf", "float", "value_invalid"),
    ("ai_model.auroc", "-inf", "float", "value_invalid"),
    ("ai_model.auroc", "1e999", "float", "value_invalid"),
    ("ai_model.auroc", "unknown", "float", "value_invalid"),
    ("ai_model.training_sample_size", "3.5", "int", "value_invalid"),
    ("ai_model.training_sample_size", "nan", "int", "value_invalid"),
    ("ai_model.training_sample_size", "inf", "int", "value_invalid"),
    ("ai_model.model_name", "Synthetic", "float", "value_type_mismatch"),
    ("ai_model.use_ai", "true", "str", "value_type_mismatch"),
    ("ai_model.unknown", "true", "bool", "field_path_invalid"),
    ("unknown.value", "true", "bool", "field_path_invalid"),
    ("ai_model.result.extra", "positive", "str", "field_path_invalid"),
    ("ai_model.auroc", "0.9", "unknown", "value_type_mismatch"),
    ("ai_model.model_name", "", "str", "value_invalid"),
    ("ai_model.model_name", None, "str", "value_invalid"),
    (None, "true", "bool", "field_path_invalid"),
))
def test_invalid_reviewed_values_fail_atomically(path, value, value_type, code):
    inp = AssessmentInput()
    before = copy.deepcopy(inp.to_dict())
    assertion = claim(path, value, value_type)
    assert assertion_value_issues(assertion) == [code]
    with pytest.raises(ValueError):
        apply_reviewed_assertions(inp, [claim(), assertion])
    assert inp.to_dict() == before
    assert not inp.supporting_evidence.assertions_reviewed


@pytest.mark.parametrize("value,expected", (("true", True), ("false", False), ("1", True), ("0", False), ("yes", True), ("no", False)))
def test_recognized_boolean_values_are_typed(value, expected):
    assertion = claim("ai_model.use_ai", value, "bool")
    assert validate_assertion_value(assertion) is expected
    assert apply_reviewed_assertions(AssessmentInput(), [assertion]).ai_model.use_ai is expected


def test_finite_numeric_and_list_values_apply_to_their_declared_types():
    values = (
        claim("ai_model.auroc", "0.75", "float"),
        claim("ai_model.training_sample_size", "100.0", "int"),
        claim("ai_model.domain_modalities", "synthetic one;synthetic two", "list"),
    )
    updated = apply_reviewed_assertions(AssessmentInput(), values)
    assert updated.ai_model.auroc == 0.75
    assert isinstance(updated.ai_model.auroc, float)
    assert updated.ai_model.training_sample_size == 100
    assert isinstance(updated.ai_model.training_sample_size, int)
    assert updated.ai_model.domain_modalities[-2:] == ["synthetic one", "synthetic two"]


def test_large_integer_value_does_not_lose_precision_during_conversion():
    assertion = claim("ai_model.training_sample_size", "9007199254740993", "int")
    assert validate_assertion_value(assertion) == 9007199254740993


@pytest.mark.parametrize("kind,scope,expected", (
    ("observed", {}, "evidence_kind_invalid"),
    ("context", None, "scope_invalid"),
    ("context", {"invented": "value"}, "scope_key_unsupported"),
    ("context", {"subject": 1}, "scope_value_invalid"),
))
def test_invalid_review_metadata_is_reported_and_blocks_application(kind, scope, expected):
    assertion = claim()
    assertion.evidence_kind = kind
    assertion.scope = scope
    assert expected in metadata_issues(assertion)
    with pytest.raises(ValueError, match="metadata"):
        apply_reviewed_assertions(AssessmentInput(), [assertion])


def test_nam_result_requires_explicit_observation_review_even_without_documents():
    assertion = claim("nam_assay.result", "양성")
    with pytest.raises(ValueError, match="observation"):
        apply_reviewed_assertions(AssessmentInput(), [assertion])
    assertion.evidence_kind = "observation"
    assert apply_reviewed_assertions(AssessmentInput(), [assertion]).nam_assay.result == "양성"


def test_ai_result_cannot_be_applied_as_an_observation():
    assertion = claim("ai_model.result", "양성 / 위험 신호")
    assertion.evidence_kind = "observation"
    with pytest.raises(ValueError, match="prediction"):
        apply_reviewed_assertions(AssessmentInput(), [assertion])


def test_normal_source_linked_reviewed_model_name_still_applies():
    document, assertions = claims_for("Model Name: Synthetic Model")
    assertions[0].review_status = "승인"
    updated = apply_reviewed_assertions(AssessmentInput(), assertions, documents=[document])
    assert updated.ai_model.model_name == "Synthetic Model"
    assert updated.supporting_evidence.assertions_reviewed
    assert updated.supporting_evidence.evidence_traceable


def test_table_preserves_immutable_provenance_and_extracted_value():
    _, assertions = claims_for("Model Name: Synthetic Model")
    original = copy.deepcopy(assertions[0])
    row = assertion_table_rows(assertions)[0]
    assert row["Original extracted value"] == "Synthetic Model"
    assert all(f"Scope {key}" in row for key in SCOPE_KEYS)
    row.update({"Field Path": "ai_model.result", "근거 발췌": "invented", "출처 위치": "invented", "Original extracted value": "invented"})
    restored = assertions_from_table_rows([row], assertions)[0]
    assert restored == original
    assert assertions[0] == original


@pytest.mark.parametrize("field,value", (("제안/수정 값", "Edited model"), ("Evidence kind", "context"), ("Scope subject", "Synthetic B")))
def test_semantic_table_edits_clear_approval_until_separate_review(field, value):
    _, assertions = claims_for("Model Name: Synthetic Model")
    assertions[0].review_status = "승인"
    row = assertion_table_rows(assertions)[0]
    row[field] = value
    row["검토 상태"] = "승인"
    edited = assertions_from_table_rows([row], assertions)
    assert edited[0].review_status == "제안됨"
    # An unchanged second action can explicitly review the edited claim.
    reviewed_row = assertion_table_rows(edited)[0]
    reviewed_row["검토 상태"] = "수정"
    assert assertions_from_table_rows([reviewed_row], edited)[0].review_status == "수정"
    assert assertions[0].review_status == "승인"


def test_unchanged_table_roundtrip_preserves_approval_and_empty_scope():
    assertion = claim()
    result = assertions_from_table_rows(assertion_table_rows([assertion]), [assertion])[0]
    assert result.review_status == "승인"
    assert result.scope == {}
