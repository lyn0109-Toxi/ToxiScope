import json
from unittest.mock import patch
from ntr_bridge import empty_input, context_key, context_snapshot, reviewed_assessment
from nora.evidence import extract_document
from nora.assertions import extract_assertions_from_documents
from core.identity import get_smiles_from_name

def test_no_network_without_consent():
    with patch("requests.get", side_effect=AssertionError("Network prohibited")) as network:
        assert get_smiles_from_name("CONFIDENTIAL-TEST-912") is None
        assert network.call_count == 0

def test_consent_can_enable_identity_request():
    with patch("requests.get") as get:
        get.return_value.status_code = 404
        get_smiles_from_name("synthetic-public-test", allow_external=True)
        assert get.call_count == 2

def test_structural_context_never_becomes_toxicity_observation():
    inp = empty_input("Synthetic A")
    assert not inp.ai_model.use_ai and not inp.nam_assay.use_nam
    package = {"assessment": {"expert_alerts": [{"alert": "synthetic"}]}}
    snapshot = context_snapshot("P1", "Synthetic A", package)
    package["assessment"]["expert_alerts"].clear()
    assert snapshot["package"]["assessment"]["expert_alerts"]
    updated, result = reviewed_assessment(inp, [], [])
    assert updated.to_dict() == inp.to_dict()
    assert result.evidence_role_code == "R0"

def test_context_changes_invalidate_review():
    assert context_key("P", "A", {}) != context_key("P", "B", {})
    assert context_key("P", "A", {}) != context_key("Q", "A", {})
    assert context_key("P", "A", {}) != context_key("P", "A", {"x": 1})

def test_source_hash_and_unreviewed_assertion_not_applied():
    doc = extract_document(b"Product name: Synthetic Candidate\nHuman Cmax: 25 ng/mL", "synthetic.txt")
    claims = extract_assertions_from_documents([doc])
    assert len(doc.sha256) == 64
    assert claims
    inp = empty_input("Original")
    updated, _ = reviewed_assessment(inp, claims, [doc])
    assert updated.to_dict() == inp.to_dict()
    claims[0].review_status = "승인"
    import pytest
    with pytest.raises(ValueError, match="Source review"):
        reviewed_assessment(inp, claims, [doc])
    doc.source_kind = "external_reference"
    with pytest.raises(ValueError, match="Source review"):
        reviewed_assessment(inp, claims, [doc])
    doc.source_kind = "customer_evidence"
    reviewed_assessment(inp, claims, [doc])

def test_unsupported_endpoint_stays_r0():
    inp = empty_input()
    inp.context_of_use.target_endpoint = "Cardiotoxicity"
    _, result = reviewed_assessment(inp, [], [])
    assert result.evidence_role_code == "R0"

def test_provenance_tampering_and_conflict_blocked():
    import copy
    import pytest
    doc = extract_document(b"Model Name: Synthetic Model", "synthetic.txt")
    doc.source_kind = "customer_evidence"
    claims = extract_assertions_from_documents([doc])
    claims[0].review_status = "승인"
    modified = copy.deepcopy(claims)
    modified[0].source_excerpt = "not present in source"
    with pytest.raises(ValueError, match="Source review"):
        reviewed_assessment(empty_input(), modified, [doc])
    other = copy.deepcopy(claims[0])
    other.assertion_id += "-other"
    other.proposed_value = "Different Model"
    with pytest.raises(ValueError, match="conflicting"):
        reviewed_assessment(empty_input(), claims + [other], [doc])
