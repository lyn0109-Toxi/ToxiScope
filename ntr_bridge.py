"""Local bridge: source-linked triage stays separate from toxicity predictions."""
import copy
import hashlib
import json
import math
import types
from dataclasses import fields
from typing import get_args, get_origin, get_type_hints, Union

from nora.models import AssessmentInput
from nora.assertions import apply_reviewed_assertions, reviewed_assertion_conflicts, validate_numeric_domain
from nora.engine import evaluate

SOURCE_COMMIT = "6d5e75476188cb99b9884483502599d605a4ed09"
REVIEW_METHOD_VERSION = "ntr-project-triage-1"


def empty_input(name=""):
    """Neutral intake defaults: no method outcome or control validity is supplied."""
    inp = AssessmentInput()
    inp.product.product_name = name
    inp.product.modality = ""
    inp.product.route = ""
    inp.product.exposure_pattern = ""
    inp.product.target_organs = ""
    inp.product.distribution_status = "불명확"
    inp.context_of_use.objective = "Evidence gaps and disagreements for expert review"
    inp.context_of_use.intended_evidence_role = "R1 · 가설 생성"
    inp.ai_model.use_ai = False
    inp.ai_model.result = ""
    inp.ai_model.model_type = ""
    inp.ai_model.external_validation = "불명확"
    inp.ai_model.domain_modalities = []
    inp.nam_assay.use_nam = False
    inp.nam_assay.result = ""
    inp.nam_assay.nam_type = ""
    inp.nam_assay.system_origin = ""
    inp.nam_assay.cell_types = []
    inp.nam_assay.metabolic_competence = "불명확"
    inp.nam_assay.exposure_design = ""
    inp.nam_assay.positive_control = "불명확"
    inp.nam_assay.negative_control = "불명확"
    inp.nam_assay.protocol_completeness = "불명확"
    inp.nam_assay.measured_exposure = "불명확"
    inp.nam_assay.reproducibility = "불명확"
    inp.nam_assay.endpoints = []
    inp.supporting_evidence.evidence_traceable = False
    inp.supporting_evidence.version_locked = False
    return inp


def _typed_value(value, annotation):
    origin = get_origin(annotation)
    if origin in {Union, types.UnionType}:
        return any(_typed_value(value, option) for option in get_args(annotation))
    if annotation is type(None):
        return value is None
    if origin is list:
        return isinstance(value, list) and all(_typed_value(item, get_args(annotation)[0]) for item in value)
    if annotation is bool:
        return type(value) is bool
    if annotation is int:
        return type(value) is int
    if annotation is float:
        try:
            return type(value) in {int, float} and math.isfinite(value)
        except OverflowError:
            return False
    return isinstance(value, annotation)


def parse_review_input(payload, *, compound_name=""):
    """Overlay explicitly supplied, typed JSON fields on neutral intake defaults.

    Unlike AssessmentInput.from_dict, omitted sections do not enable methods or
    supply negative outcomes/valid controls. Manual assertions remain unverified.
    """
    if not isinstance(payload, dict):
        raise ValueError("Reviewed input must be a JSON object.")
    inp = empty_input(compound_name)
    allowed_sections = {field.name for field in fields(inp)}
    for section, values in payload.items():
        if section not in allowed_sections or not isinstance(values, dict):
            raise ValueError(f"Unknown section or invalid object: {section}")
        target = getattr(inp, section)
        hints = get_type_hints(type(target))
        for key, value in values.items():
            if key not in hints or not _typed_value(value, hints[key]):
                raise ValueError(f"Invalid field or value type: {section}.{key}")
            validate_numeric_domain(f"{section}.{key}", value)
            setattr(target, key, copy.deepcopy(value))
    return inp


def context_snapshot(project_id, name, package):
    return {"project_id": project_id, "compound_name": name,
            "kind": "structural_screen_context_not_observed_toxicity",
            "package": copy.deepcopy(package)}


def context_key(project_id, name, package):
    return hashlib.sha256(json.dumps(context_snapshot(project_id, name, package),
        sort_keys=True, default=str).encode()).hexdigest()


def reviewed_assessment(inp, assertions, documents):
    if reviewed_assertion_conflicts(assertions):
        raise ValueError("Resolve conflicting reviewed assertions before evaluation.")
    updated = apply_reviewed_assertions(inp, assertions, documents=documents)
    return updated, evaluate(updated)


def build_review_export(project_review, documents, assertions, *, project_id="", compound_name="",
                        package=None, manual_input=None, reviewed_input=None, result=None,
                        evaluation_error=None, raw_manual_input=None):
    """Export triage even when optional legacy evaluation is unavailable/blocked."""
    def snapshot(value):
        return copy.deepcopy(value.to_dict() if hasattr(value, "to_dict") else value)
    return {
        "schema": "toxiscope-ntr-review-2",
        "ntr_source_commit": SOURCE_COMMIT,
        "local_review_method_version": REVIEW_METHOD_VERSION,
        "project_review": snapshot(project_review),
        "structural_context": context_snapshot(project_id, compound_name, package or {}),
        "manual_input": snapshot(manual_input),
        "raw_manual_input": raw_manual_input,
        "reviewed_input": snapshot(reviewed_input),
        "documents": [snapshot(d) for d in documents],
        "assertions": [snapshot(a) for a in assertions],
        "result": snapshot(result),
        "evaluation_error": evaluation_error,
        "legacy_result_scope": "Optional uncalibrated heuristic output; its observation/interpretation labels do not establish measured facts or source independence.",
        "limitations": list(project_review.get("limitations", [])),
        "decision": "Expert review required; no safety or regulatory decision is automated.",
    }
