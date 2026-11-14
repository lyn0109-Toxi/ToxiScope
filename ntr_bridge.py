"""Local bridge: structural screens are context, never observed toxicity facts."""
import copy
import hashlib
import json
from nora.models import AssessmentInput
from nora.assertions import apply_reviewed_assertions, reviewed_assertion_conflicts
from nora.engine import evaluate

SOURCE_COMMIT = "6d5e75476188cb99b9884483502599d605a4ed09"

def empty_input(name=""):
    inp = AssessmentInput()
    inp.product.product_name = name
    inp.product.modality = ""
    inp.product.route = ""
    inp.product.exposure_pattern = ""
    inp.product.target_organs = ""
    inp.context_of_use.objective = "Evidence gaps and disagreements for expert review"
    inp.context_of_use.intended_evidence_role = "R1 · 가설 생성"
    inp.ai_model.use_ai = False
    inp.nam_assay.use_nam = False
    inp.supporting_evidence.evidence_traceable = False
    inp.supporting_evidence.version_locked = False
    return inp

def context_snapshot(project_id, name, package):
    # Copy verbatim provenance; no translation to NTR AI/NAM positive/negative.
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
