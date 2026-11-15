"""Synthetic review-triage checks; these do not validate toxicology outcomes."""
import copy
import hashlib
import json

import pytest

from nora.assertions import apply_reviewed_assertions, extract_assertions_from_documents
from nora.evidence import DocumentRecord, EvidenceAssertion, SourceSegment
from nora.evidence_review import build_project_review
from nora.models import AssessmentInput


SHARED_SCOPE = {
    "subject": "Synthetic Candidate A",
    "endpoint": "early hepatotoxicity",
    "system": "synthetic human hepatocyte culture",
    "exposure": "10 uM",
    "timepoint": "24 h",
    "study_id": "SYNTHETIC-STUDY-1",
}


def document(identifier, text, *, source_kind="customer_evidence", location="page 1"):
    return DocumentRecord(
        document_id=identifier,
        name=f"{identifier}.txt",
        media_type="text/plain",
        extension=".txt",
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        byte_size=len(text.encode()),
        extracted_text=text,
        segments=[SourceSegment(f"{identifier}-segment-1", location, text)],
        source_kind=source_kind,
    )


def assertion(identifier, source, value, *, review_status="제안됨", evidence_kind="observation", scope=None, field_path="nam_assay.result", location=None, excerpt=None, extracted_value=None):
    return EvidenceAssertion.from_dict({
        "assertion_id": identifier,
        "category": "Synthetic review fixture",
        "field_path": field_path,
        "label_ko": field_path,
        "proposed_value": value,
        "value_type": "str",
        "source_document_id": source.document_id,
        "source_document_name": source.name,
        "source_location": location or source.segments[0].location,
        "source_excerpt": excerpt or source.segments[0].text,
        "confidence": 0.8,
        "review_status": review_status,
        "reviewer_note": "Synthetic review note; scientific claim not independently verified.",
        "evidence_kind": evidence_kind,
        "scope": dict(SHARED_SCOPE if scope is None else scope),
        "extracted_value": value if extracted_value is None else extracted_value,
    })


def findings(report, kind):
    return [item for item in report["findings"] if item["kind"] == kind]


def coverage(report, path):
    return next(row for row in report["coverage"] if row["field_path"] == path)


def conflicting_pair(*, left_scope=None, right_scope=None, review_status="제안됨"):
    left = document("DOC-positive", "Synthetic assay result: Positive")
    right = document("DOC-negative", "Synthetic assay result: Negative")
    return [
        assertion("AST-positive", left, "Positive", scope=left_scope, review_status=review_status),
        assertion("AST-negative", right, "Negative", scope=right_scope, review_status=review_status),
    ], [left, right]


def test_empty_project_is_unknown_with_review_gaps_not_a_negative_outcome():
    report = build_project_review([], [], project_id="SYNTHETIC-P1", compound_name="Synthetic Candidate A")
    assert report["schema"] == "toxiscope-ntr-project-review.v1"
    assert report["decision"] == "Expert review required"
    assert report["summary"]["possible_conflicts"] == 0
    assert report["summary"]["coverage_gaps"] > 0
    assert report["limitations"]
    assert report["scope"]
    assert coverage(report, "nam_assay.result")["review_state"] == "not_extracted"
    assert all(row["review_state"] != "reviewed_source_linked" for row in report["coverage"])
    assert "toxicity_direction" not in report
    assert "evidence_role_code" not in report
    assert findings(report, "coverage_gap")
    assert all(finding["review_state"] == "needs_expert_review" for finding in report["findings"])


def test_proposed_opposite_values_with_shared_scope_are_possible_conflicts_only():
    assertions, documents = conflicting_pair()
    report = build_project_review(assertions, documents)
    conflicts = findings(report, "possible_conflict")
    assert conflicts
    conflict = next(item for item in conflicts if item["field_path"] == "nam_assay.result")
    assert {item["assertion_id"] for item in conflict["provenance"]} == {"AST-positive", "AST-negative"}
    assert conflict["uncertainty"]
    assert conflict["review_state"] == "needs_expert_review"
    assert not findings(report, "confirmed_conflict")
    assert coverage(report, "nam_assay.result")["review_state"] != "reviewed_source_linked"


def test_missing_scope_keeps_disagreement_uncertain_and_never_confirms_contradiction():
    assertions, documents = conflicting_pair(left_scope={}, right_scope={})
    report = build_project_review(assertions, documents)
    conflicts = findings(report, "possible_conflict")
    assert conflicts
    assert all(item["comparability"] == "scope_incomplete" for item in conflicts)
    assert all(item["uncertainty"] for item in conflicts)
    assert not findings(report, "confirmed_conflict")
    assert report["decision"] == "Expert review required"


@pytest.mark.parametrize("scope_key,other_value", [("exposure", "100 uM"), ("timepoint", "72 h"), ("system", "synthetic rat hepatocyte culture"), ("study_id", "SYNTHETIC-STUDY-2")])
def test_explicit_scope_differences_are_not_same_scope_conflicts(scope_key, other_value):
    different_scope = {**SHARED_SCOPE, scope_key: other_value}
    assertions, documents = conflicting_pair(right_scope=different_scope)
    report = build_project_review(assertions, documents)
    assert findings(report, "scope_difference")
    assert not findings(report, "possible_conflict")
    assert report["summary"]["scope_differences"] >= 1


def test_provenance_preserves_reviewed_and_original_values_with_full_source_reference():
    source = document("DOC-corrected", "Synthetic assay result: Positive")
    claim = assertion("AST-corrected", source, "Negative", review_status="수정", extracted_value="Positive")
    other_source = document("DOC-opposite", "Synthetic assay result: Positive")
    other_claim = assertion("AST-opposite", other_source, "Positive", review_status="승인")
    report = build_project_review([claim, other_claim], [source, other_source])
    refs = [ref for item in report["findings"] for ref in item["provenance"] if ref["assertion_id"] == claim.assertion_id]
    assert refs
    ref = next(ref for ref in refs if ref["assertion_id"] == claim.assertion_id)
    assert ref["source_document_id"] == source.document_id
    assert ref["source_document_name"] == source.name
    assert ref["source_sha256"] == source.sha256
    assert ref["source_location"] == "page 1"
    assert ref["source_excerpt"] == source.segments[0].text
    assert ref["proposed_value"] == "Negative"
    assert ref["extracted_value"] == "Positive"
    assert ref["evidence_kind"] == "observation"
    assert ref["review_status"] == "수정"
    assert ref["reviewer_note"] == claim.reviewer_note
    assert ref["source_kind"] == "customer_evidence"
    assert ref["source_issues"] == []
    assert ref["scope"] == SHARED_SCOPE


@pytest.mark.parametrize("source_kind", ["external_reference", "unclassified"])
def test_approval_does_not_turn_external_or_unclassified_sources_into_customer_coverage(source_kind):
    source = document("DOC-reference", "Synthetic assay result: Negative", source_kind=source_kind)
    claim = assertion("AST-reference", source, "Negative", review_status="승인")
    report = build_project_review([claim], [source])
    assert coverage(report, "nam_assay.result")["review_state"] != "reviewed_source_linked"
    assert findings(report, "source_issue")
    assert report["summary"]["source_issues"] > 0


@pytest.mark.parametrize("change", ["excerpt", "location", "name", "id", "hash", "duplicate_document", "ambiguous_location"])
def test_provenance_tampering_remains_a_source_issue(change):
    source = document("DOC-source", "Synthetic assay result: Negative")
    claim = assertion("AST-source", source, "Negative", review_status="승인")
    documents = [source]
    if change == "excerpt":
        claim.source_excerpt = "Fabricated text not in the source"
    elif change == "location":
        claim.source_location = "invented page 999"
    elif change == "name":
        claim.source_document_name = "different-document.txt"
    elif change == "id":
        claim.source_document_id = "DOC-not-in-project"
    elif change == "hash":
        source.sha256 = "malformed-recorded-hash"
    elif change == "duplicate_document":
        documents.append(copy.deepcopy(source))
    else:
        source.segments.append(SourceSegment("SEG-ambiguous", claim.source_location, claim.source_excerpt))
    report = build_project_review([claim], documents)
    assert findings(report, "source_issue")
    assert coverage(report, "nam_assay.result")["review_state"] != "reviewed_source_linked"
    refs = [ref for item in findings(report, "source_issue") for ref in item["provenance"]]
    assert any(ref["assertion_id"] == claim.assertion_id and ref["source_issues"] for ref in refs)


def test_rejected_assertions_do_not_create_conflicts_with_active_candidates():
    assertions, documents = conflicting_pair()
    assertions[1].review_status = "거절"
    report = build_project_review(assertions, documents)
    assert not findings(report, "possible_conflict")
    assert not findings(report, "scope_difference")
    assert report["summary"]["possible_conflicts"] == 0


@pytest.mark.parametrize("evidence_kind", ["prediction", "context", "unspecified"])
def test_nonobservational_claim_targeting_nam_result_never_fills_observation_coverage(evidence_kind):
    source = document("DOC-prediction", "Synthetic model predicts Negative")
    claim = assertion("AST-prediction", source, "Negative", review_status="승인", evidence_kind=evidence_kind)
    report = build_project_review([claim], [source])
    assert coverage(report, "nam_assay.result")["review_state"] != "reviewed_source_linked"
    assert any(item["field_path"] == "nam_assay.result" for item in findings(report, "coverage_gap") + findings(report, "source_issue"))
    assert claim.evidence_kind == evidence_kind
    inp = AssessmentInput()
    before = copy.deepcopy(inp)
    with pytest.raises(ValueError):
        apply_reviewed_assertions(inp, [claim], documents=[source])
    assert inp == before


def test_approved_observation_with_linked_customer_source_counts_as_reviewed_coverage_only():
    source = document("DOC-observation", "Synthetic assay result: Negative")
    claim = assertion("AST-observation", source, "Negative", review_status="승인")
    report = build_project_review([claim], [source])
    row = coverage(report, "nam_assay.result")
    assert row["review_state"] == "reviewed_source_linked"
    assert "AST-observation" in row["assertion_ids"]
    assert report["decision"] == "Expert review required"
    assert "toxicity_direction" not in report


@pytest.mark.parametrize("manual_input", [AssessmentInput(), {"nam_assay": {"use_nam": True, "result": "Negative"}, "ai_model": {"use_ai": True, "result": "Negative"}}])
def test_default_or_manual_values_are_unverified_without_source_coverage(manual_input):
    report = build_project_review([], [], assessment_input=manual_input)
    assert coverage(report, "nam_assay.result")["review_state"] != "reviewed_source_linked"
    assert coverage(report, "ai_model.result")["review_state"] != "reviewed_source_linked"
    assert report["summary"]["coverage_gaps"] > 0
    assert not findings(report, "possible_conflict")
    assert report["decision"] == "Expert review required"
    assert any("manual" in text.lower() and "not source-linked" in text.lower() for text in report["limitations"])


def test_unsupported_endpoint_is_flagged_without_validated_outcome_adjudication():
    report = build_project_review([], [], assessment_input={"context_of_use": {"target_endpoint": "Cardiotoxicity"}})
    assert findings(report, "out_of_scope")
    assert report["decision"] == "Expert review required"
    assert "toxicity_direction" not in report
    assert "evidence_role_code" not in report
    assert report["limitations"]


def test_project_review_is_json_serializable_deterministic_and_does_not_mutate_inputs():
    assertions, documents = conflicting_pair()
    manual = {"nam_assay": {"result": "Negative"}}
    before = copy.deepcopy((assertions, documents, manual))
    report = build_project_review(assertions, documents, project_id="SYNTHETIC-P1", assessment_input=manual)
    again = build_project_review(assertions, documents, project_id="SYNTHETIC-P1", assessment_input=manual)
    assert report == again
    assert json.loads(json.dumps(report)) == report
    assert (assertions, documents, manual) == before
    ids = [item["finding_id"] for item in report["findings"]]
    assert len(ids) == len(set(ids))


def test_same_document_multiple_locations_are_preserved_in_review_provenance():
    source = document("DOC-multi", "Synthetic assay result: Positive")
    source.segments.append(SourceSegment("SEG-second", "page 2", "Synthetic assay result: Negative"))
    source.extracted_text += "\n" + source.segments[1].text
    claims = [
        assertion("AST-page-1", source, "Positive"),
        assertion("AST-page-2", source, "Negative", location="page 2", excerpt=source.segments[1].text),
    ]
    report = build_project_review(claims, [source])
    refs = [ref for item in findings(report, "possible_conflict") for ref in item["provenance"]]
    assert {ref["source_location"] for ref in refs} == {"page 1", "page 2"}
    assert {ref["source_document_id"] for ref in refs} == {source.document_id}
    assert all(ref["source_issues"] == [] for ref in refs)


def test_extraction_preserves_repeated_value_at_distinct_source_locations():
    source = document("DOC-repeated", "Model Name: Synthetic Model")
    source.segments.append(SourceSegment("SEG-second", "page 2", "Model Name: Synthetic Model"))
    claims = extract_assertions_from_documents([source])
    model_claims = [item for item in claims if item.field_path == "ai_model.model_name"]
    assert {item.source_location for item in model_claims} == {"page 1", "page 2"}
