"""Conservative project evidence triage; no toxicity adjudication or validation."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from copy import deepcopy
from collections import Counter
from itertools import combinations
from typing import Any, Iterable

from .evidence import DocumentRecord, EvidenceAssertion

SOURCE_KINDS = ("unclassified", "customer_evidence", "external_reference")
REVIEWED = {"승인", "수정"}
SCOPE_KEYS = ("subject", "endpoint", "system", "exposure", "timepoint", "study_id")
# Internal intake prompts, not a complete or regulatory-required checklist.
INTAKE_FIELDS = (
    "product.product_name", "product.modality", "product.route",
    "product.planned_dose", "product.exposure_pattern",
    "context_of_use.target_endpoint", "context_of_use.question_of_interest",
    "ai_model.model_name", "ai_model.model_version", "ai_model.result",
    "ai_model.external_validation", "nam_assay.nam_type", "nam_assay.result",
    "nam_assay.measured_exposure", "nam_assay.positive_control",
    "nam_assay.negative_control", "supporting_evidence.pk_tk_evidence",
)
LIMITATIONS = [
    "Rule-based intake triage, not validated conflict detection, toxicity prediction, scientific adjudication or a regulatory checklist.",
    "Missing or rejected extraction means coverage is not established from the supplied material; it does not establish that evidence is absent or a toxicity result is negative.",
    "Recorded SHA-256 values identify uploaded file bytes; metadata/excerpt linkage does not authenticate a study, verify its claims, or establish source independence.",
    "Values are compared conservatively by field and stated scope. Units, synonyms, study quality, biological relevance and causality are not resolved automatically.",
    "Unknown subject, endpoint, system, exposure, timepoint or study identifiers limit comparability. Different study conditions or prediction/observation types are not formal contradictions.",
    "Regex extraction is incomplete and can miss or misread claims, including negation, tables and scanned pages. No OCR is performed.",
    "Human review records are user-supplied decisions, not fabricated expert opinions or independently verified validation. Manual inputs and template defaults are not source-linked observations.",
    "Structural SAR screens are context only and are never converted to observed customer toxicity evidence. Claim counts do not establish independent evidence streams.",
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _comparison(text: str) -> str:
    return _normalize(unicodedata.normalize("NFKC", str(text))).casefold()


def _literal_value(text: str) -> str:
    # Preserve case because units and model identifiers can be case-sensitive.
    return _normalize(unicodedata.normalize("NFKC", str(text)))


def source_issues(assertion: EvidenceAssertion, documents: Iterable[DocumentRecord]) -> list[str]:
    """Check recorded source linkage, never scientific truth or authenticity."""
    if not assertion.source_document_id.strip():
        return ["source_document_missing_or_ambiguous"]
    matches = [d for d in documents if d.document_id == assertion.source_document_id]
    if len(matches) != 1:
        return ["source_document_missing_or_ambiguous"]
    document = matches[0]
    issues = []
    if document.source_kind != "customer_evidence":
        issues.append("external_reference" if document.source_kind == "external_reference" else "source_unclassified")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", document.sha256) or document.name != assertion.source_document_name:
        issues.append("source_metadata_mismatch")
    segments = [s for s in document.segments if s.location == assertion.source_location]
    segment_id = getattr(assertion, "source_segment_id", "")
    if segment_id:
        segments = [s for s in segments if s.segment_id == segment_id]
    excerpt = _normalize(assertion.source_excerpt)
    if not assertion.source_location.strip() or len(segments) != 1 or not excerpt or excerpt not in _normalize(segments[0].text):
        issues.append("source_excerpt_not_linked")
    return issues


def _claim_issues(assertion: EvidenceAssertion) -> list[str]:
    # Lazy import avoids the assertions -> evidence_review application cycle.
    from .assertions import metadata_issues, assertion_value_issues
    issues = assertion_value_issues(assertion) + metadata_issues(assertion)
    if assertion.field_path == "nam_assay.result" and assertion.evidence_kind != "observation":
        issues.append("observation_review_required")
    if assertion.field_path == "ai_model.result" and assertion.evidence_kind != "prediction":
        issues.append("prediction_label_required")
    return list(dict.fromkeys(issues))


def _scope(assertion: EvidenceAssertion) -> dict[str, str]:
    return {k: v.strip() for k, v in assertion.scope.items() if k in SCOPE_KEYS and isinstance(v, str) and v.strip()} if isinstance(assertion.scope, dict) else {}


def _input_dict(assessment_input: Any) -> dict[str, Any]:
    data = assessment_input.to_dict() if hasattr(assessment_input, "to_dict") else assessment_input
    return data if isinstance(data, dict) else {}


def application_blockers(assertions: Iterable[EvidenceAssertion], documents: Iterable[DocumentRecord], *, assessment_input: Any = None) -> dict[str, list[str]]:
    """Reviewed claims need linked customer sources and valid, typed values."""
    documents, assertions = list(documents), list(assertions)
    manual = _input_dict(assessment_input)
    context = manual.get("context_of_use", {})
    product = manual.get("product", {})
    endpoint = context.get("target_endpoint") if isinstance(context, dict) else None
    subject = product.get("product_name") if isinstance(product, dict) else None
    counts = Counter(a.assertion_id for a in assertions)
    scopes_by_path: dict[str, dict[str, set[str]]] = {}
    for assertion in assertions:
        if assertion.review_status in REVIEWED:
            for key, value in _scope(assertion).items():
                scopes_by_path.setdefault(assertion.field_path, {}).setdefault(key, set()).add(_literal_value(value))
    blocked = {}
    for assertion in assertions:
        if assertion.review_status not in REVIEWED:
            continue
        issues = source_issues(assertion, documents) + _claim_issues(assertion)
        if not assertion.assertion_id or counts[assertion.assertion_id] != 1:
            issues.append("assertion_id_missing_or_ambiguous")
        if assertion.field_path in {"ai_model.result", "nam_assay.result"} and not all(_scope(assertion).get(k, "").strip() for k in ("subject", "endpoint", "system")):
            issues.append("outcome_scope_review_required")
        if assertion.field_path in {"ai_model.result", "nam_assay.result"}:
            if isinstance(endpoint, str) and endpoint.strip() and _comparison(_scope(assertion).get("endpoint", "")) != _comparison(endpoint):
                issues.append("requested_endpoint_mismatch")
            if isinstance(subject, str) and subject.strip() and _comparison(_scope(assertion).get("subject", "")) != _comparison(subject):
                issues.append("project_subject_mismatch")
        if any(len(values) > 1 for values in scopes_by_path.get(assertion.field_path, {}).values()):
            issues.append("multiple_scopes_require_selection")
        if issues:
            blocked[assertion.assertion_id] = list(dict.fromkeys(issues))
    return blocked


def _provenance(assertion: EvidenceAssertion, documents: list[DocumentRecord]) -> dict[str, Any]:
    matches = [d for d in documents if d.document_id == assertion.source_document_id]
    document = matches[0] if len(matches) == 1 else None
    return {
        "assertion_id": assertion.assertion_id,
        "source_document_id": assertion.source_document_id,
        "source_document_name": assertion.source_document_name,
        "source_sha256": document.sha256 if document else None,
        "source_kind": document.source_kind if document else "unresolved",
        "source_location": assertion.source_location,
        "source_segment_id": getattr(assertion, "source_segment_id", ""),
        "source_excerpt": assertion.source_excerpt,
        "proposed_value": assertion.proposed_value,
        "extracted_value": assertion.extracted_value,
        "evidence_kind": assertion.evidence_kind,
        "review_status": assertion.review_status,
        "reviewer_note": assertion.reviewer_note,
        "scope": deepcopy(assertion.scope),
        "source_issues": source_issues(assertion, documents),
    }


def _finding(kind: str, field_path: str, title: str, detail: str, uncertainty: str,
             claims: list[EvidenceAssertion], documents: list[DocumentRecord], **extra: Any) -> dict[str, Any]:
    refs = [_provenance(a, documents) for a in claims]
    identity = json.dumps([kind, field_path, refs, extra], sort_keys=True, ensure_ascii=False)
    return {
        "finding_id": "NTR-" + hashlib.sha256(identity.encode()).hexdigest()[:16],
        "kind": kind, "field_path": field_path, "title": title, "detail": detail,
        "uncertainty": uncertainty, "review_state": "needs_expert_review",
        "provenance": refs, "scope": {a.assertion_id: _scope(a) for a in claims},
        **extra,
    }


def _value_key(assertion: EvidenceAssertion) -> str:
    from .assertions import validate_assertion_value
    try:
        value = validate_assertion_value(assertion)
    except (ValueError, TypeError):
        value = assertion.proposed_value
    if assertion.field_path in {"ai_model.result", "nam_assay.result"} and _direction(assertion):
        return _direction(assertion)
    if isinstance(value, list):
        return json.dumps(sorted({_literal_value(v) for v in value}))
    return _literal_value(value)


def _scope_comparison(left: EvidenceAssertion, right: EvidenceAssertion) -> tuple[str, list[str]]:
    left_scope, right_scope = _scope(left), _scope(right)
    differences = [key for key in SCOPE_KEYS if left_scope.get(key) and right_scope.get(key)
                   and _literal_value(left_scope[key]) != _literal_value(right_scope[key])]
    if differences:
        return "different_stated_scope", differences
    if left.evidence_kind != right.evidence_kind:
        return "different_evidence_kind", []
    if all(left_scope.get(key) and right_scope.get(key) for key in SCOPE_KEYS):
        return "matched_explicit_scope", []
    return "scope_incomplete", []


def evidence_gap_rows(assertions: Iterable[EvidenceAssertion], documents: Iterable[DocumentRecord], *, assessment_input: Any = None) -> list[dict[str, Any]]:
    """Intake coverage is source/review linkage, never evidence completeness."""
    assertions, documents = list(assertions), list(documents)
    blocked = application_blockers(assertions, documents, assessment_input=assessment_input)
    rows = []
    for path in sorted(set(INTAKE_FIELDS) | {a.field_path for a in assertions}):
        candidates = [a for a in assertions if a.field_path == path and a.review_status != "거절"]
        reviewed = [a for a in candidates if a.review_status in REVIEWED]
        values = {_value_key(a) for a in reviewed if a.value_type != "list"}
        if not candidates:
            state = "all_candidates_rejected" if any(a.field_path == path for a in assertions) else "not_extracted"
        elif any(source_issues(a, documents) for a in reviewed):
            state = "source_review_required"
        elif any(_claim_issues(a) for a in reviewed):
            state = "observation_review_required" if path == "nam_assay.result" else "value_review_required"
        elif len(values) > 1:
            state = "conflicting_reviewed_values"
        elif any(blocked.get(a.assertion_id) for a in reviewed):
            state = "scope_review_required"
        elif reviewed:
            state = "reviewed_source_linked"
        else:
            state = "human_review_pending"
        rows.append({
            "field_path": path, "review_state": state,
            "assertion_ids": ", ".join(a.assertion_id for a in candidates),
            "source_locations": "; ".join(f"{a.source_document_name} · {a.source_location}" for a in candidates),
            "detail": "Reviewed source linkage does not establish scientific validity or complete coverage." if state == "reviewed_source_linked" else "Reviewed source-linked coverage is not established; inspect the supplied material and intended scope.",
        })
    return rows


def _direction(assertion: EvidenceAssertion) -> str | None:
    # Exact vocabulary only; no free-text negation or clinical inference.
    value = _comparison(assertion.proposed_value)
    if value in {"negative", "음성", "음성 / 낮은 위험 예측", "negative / low risk prediction"}:
        return "negative"
    if value in {"positive", "양성", "양성 / 위험 신호", "positive / risk signal"}:
        return "positive"
    return None


def build_project_review(assertions: Iterable[EvidenceAssertion], documents: Iterable[DocumentRecord], *,
                         project_id: str = "", compound_name: str = "", assessment_input: Any = None) -> dict[str, Any]:
    """Return a reviewable inventory of possible differences and coverage needs.

    This function never approves assertions, modifies evidence, chooses a winning
    source, infers independence or evaluates the toxicity of a customer product.
    """
    assertions, documents = list(assertions), list(documents)
    manual = _input_dict(assessment_input)
    context = manual.get("context_of_use", {})
    context = context if isinstance(context, dict) else {}
    scope = {"project_id": project_id, "compound_name": compound_name,
             "target_endpoint": context.get("target_endpoint") if isinstance(context.get("target_endpoint"), str) and context["target_endpoint"].strip() else "Not specified",
             "question_of_interest": context.get("question_of_interest") if isinstance(context.get("question_of_interest"), str) and context["question_of_interest"].strip() else "Not specified",
             "basis": "User-stated project context; not a verified study population or observation"}
    findings = []
    for key in ("target_endpoint", "question_of_interest"):
        if key in context and not isinstance(context[key], str):
            findings.append(_finding("input_issue", "context_of_use." + key, "Requested scope has an invalid value type",
                "The requested scope field must be text. The raw input remains in the export; no endpoint or observation is inferred.",
                "Correct the manual input before optional assessment. Source-linked triage remains available.", [], documents))
    active = [a for a in assertions if a.review_status != "거절"]
    counts = Counter(a.assertion_id for a in assertions)
    for assertion in active:
        issues = source_issues(assertion, documents) + _claim_issues(assertion)
        if not assertion.assertion_id or counts[assertion.assertion_id] != 1:
            issues.append("assertion_id_missing_or_ambiguous")
        if issues:
            findings.append(_finding("source_issue", assertion.field_path, "Source or assertion review required",
                "Recorded linkage, evidence kind or typed value needs review: " + ", ".join(dict.fromkeys(issues)),
                "No scientific truth or authenticity decision is made by these checks.", [assertion], documents,
                issue_codes=list(dict.fromkeys(issues))))
        if assertion.evidence_kind == "unspecified" or not all(_scope(assertion).get(k) for k in SCOPE_KEYS):
            findings.append(_finding("scope_review", assertion.field_path, "Evidence kind or comparison scope is incomplete",
                "Specify observation/prediction/context and relevant subject, endpoint, system, exposure, timepoint and study identifiers before interpreting comparability.",
                "Unspecified scope is not assumed to match another source or the active project.", [assertion], documents))

    grouped: dict[str, list[EvidenceAssertion]] = {}
    for assertion in active:
        grouped.setdefault(assertion.field_path, []).append(assertion)
    comparisons = 0
    comparison_limit = 2000
    comparison_limited = False
    for path, candidates in sorted(grouped.items()):
        for left, right in combinations(candidates, 2):
            if comparisons >= comparison_limit:
                comparison_limited = True
                break
            comparisons += 1
            if _value_key(left) == _value_key(right):
                continue
            comparability, differing_keys = _scope_comparison(left, right)
            if comparability in {"different_stated_scope", "different_evidence_kind"}:
                kind = "scope_difference"
                detail = "Different reported values have different stated scopes or evidence kinds. They are not treated as a direct contradiction."
            elif left.value_type == "list" or right.value_type == "list":
                kind = "possible_value_difference"
                detail = "Reported list contents differ and may be complementary. Review scope before combining them into one assessment card."
            else:
                kind = "possible_conflict"
                detail = "Different scalar values were reported for the same intake field. This is a candidate disagreement, not an adjudicated scientific contradiction."
            findings.append(_finding(kind, path, "Possible evidence difference", detail,
                "Comparability is limited to stated scope metadata; expert review must resolve study context, units, quality and relevance.",
                [left, right], documents, comparability=comparability, differing_scope_keys=differing_keys))
    predictions = [a for a in active if a.field_path == "ai_model.result" and a.evidence_kind == "prediction"]
    observations = [a for a in active if a.field_path == "nam_assay.result" and a.evidence_kind == "observation"]
    for prediction in predictions:
        for observation in observations:
            if comparisons >= comparison_limit:
                comparison_limited = True
                break
            comparisons += 1
            if not _direction(prediction) or not _direction(observation) or _direction(prediction) == _direction(observation):
                continue
            comparability, differing_keys = _scope_comparison(prediction, observation)
            findings.append(_finding("prediction_observation_difference", "ai_model.result / nam_assay.result",
                "Prediction and reported observation differ",
                "A prediction and a source assertion labelled as an observation have opposite explicit result labels. These are different evidence types, not independent observed tests or an adjudicated contradiction.",
                "Expert review must establish endpoint, study/exposure comparability and source credibility. Neither result automatically overrides the other.",
                [prediction, observation], documents, comparability=comparability, differing_scope_keys=differing_keys))
    if comparison_limited:
        findings.append(_finding("review_limit", "", "Pairwise comparison limit reached",
            "Only the first 2000 candidate pairs were compared. Narrow the project material or inspect remaining claims manually.",
            "Further differences may remain undetected. No completeness conclusion is available.", [], documents))

    product = manual.get("product", {})
    product = product if isinstance(product, dict) else {}
    supplied_name = str(product.get("product_name") or "").strip()
    name_claims = [a for a in active if compound_name and (
        (a.field_path == "product.product_name" and _comparison(a.proposed_value) != _comparison(compound_name))
        or (_scope(a).get("subject") and _comparison(_scope(a)["subject"]) != _comparison(compound_name)))]
    if (supplied_name and compound_name and _comparison(supplied_name) != _comparison(compound_name)) or name_claims:
        findings.append(_finding("project_scope_mismatch", "product.product_name", "Candidate identity needs review",
            "The active compound label differs from a manual input or source assertion. Do not silently apply another candidate's evidence.",
            "Names may be aliases or different products; no identity equivalence is inferred.", name_claims, documents,
            active_compound_name=compound_name, manual_compound_name=supplied_name))
    endpoint_claims = [a for a in active if a.field_path in {"ai_model.result", "nam_assay.result"}
                       and _scope(a).get("endpoint") and scope["target_endpoint"] != "Not specified"
                       and _comparison(_scope(a)["endpoint"]) != _comparison(scope["target_endpoint"])]
    if endpoint_claims:
        findings.append(_finding("project_scope_mismatch", "context_of_use.target_endpoint", "Reported endpoint differs from requested review scope",
            "A result assertion has an endpoint label different from the requested endpoint. It cannot silently fill that endpoint's assessment card.",
            "Endpoint names may be aliases or different outcomes; equivalence and biological relevance require explicit expert review.",
            endpoint_claims, documents, requested_endpoint=scope["target_endpoint"]))
    if scope["target_endpoint"] not in {"Not specified", "초기 간독성"}:
        findings.append(_finding("out_of_scope", "context_of_use.target_endpoint", "Endpoint-specific assessment is unavailable",
            "The existing detailed NTR heuristic is limited to early hepatotoxicity. Generic source/coverage triage can still be reviewed for this endpoint.",
            "No endpoint validation, toxicity conclusion or biological conflict adjudication is provided.", [], documents,
            requested_endpoint=scope["target_endpoint"]))

    for document in documents:
        if document.warnings or not any(s.text.strip() for s in document.segments):
            ref = {"assertion_id": None, "source_document_id": document.document_id, "source_document_name": document.name,
                   "source_sha256": document.sha256, "source_kind": document.source_kind,
                   "source_location": "Document-level extraction", "source_excerpt": "",
                   "warnings": list(document.warnings)}
            finding = _finding("extraction_limit", "", "Document extraction requires review",
                "Some content may be unreadable, empty or unsupported. No absence-of-evidence inference is made.",
                "Review the original source; no OCR or extraction completeness check is available.", [], documents,
                document=ref)
            finding["provenance"] = [ref]
            findings.append(finding)

    coverage = evidence_gap_rows(assertions, documents, assessment_input=manual)
    for row in coverage:
        section = row["field_path"].split(".")[0]
        stream_flag = {"ai_model": "use_ai", "nam_assay": "use_nam"}.get(section)
        section_input = manual.get(section, {})
        if stream_flag and isinstance(section_input, dict) and section_input.get(stream_flag) is False and not row["assertion_ids"]:
            row["review_state"] = "not_requested"
            row["detail"] = "This evidence stream is explicitly disabled. The optional intake prompt is not counted as missing project evidence."
        if row["review_state"] not in {"reviewed_source_linked", "not_requested"}:
            claims = [a for a in active if a.field_path == row["field_path"]]
            findings.append(_finding("coverage_gap", row["field_path"], "Intake evidence coverage needs review",
                row["detail"], "This internal prompt is not a mandatory regulatory requirement; evidence may exist outside extracted or supplied material.",
                claims, documents, coverage_state=row["review_state"]))
    return {
        "schema": "toxiscope-ntr-project-review.v1", "method_version": "ntr-project-triage-1",
        "scope": scope, "decision": "Expert review required",
        "comparison_scope": {"pairs_checked": comparisons, "pair_limit": comparison_limit, "limited": comparison_limited},
        "summary": {"possible_conflicts": sum(f["kind"] == "possible_conflict" for f in findings),
                    "scope_differences": sum(f["kind"] in {"scope_difference", "scope_review", "project_scope_mismatch"} for f in findings),
                    "coverage_gaps": sum(f["kind"] == "coverage_gap" for f in findings),
                    "source_issues": sum(f["kind"] in {"source_issue", "extraction_limit"} for f in findings),
                    "prediction_observation_differences": sum(f["kind"] == "prediction_observation_difference" for f in findings),
                    "assertion_count": len(assertions), "document_count": len(documents),
                    "count_basis": "Claim and uploaded-document counts; source independence is not established"},
        "findings": findings, "coverage": coverage,
        "assertion_inventory": [_provenance(a, documents) for a in assertions],
        "application_blockers": application_blockers(assertions, documents, assessment_input=manual),
        "limitations": list(LIMITATIONS),
    }
