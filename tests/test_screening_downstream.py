import json

import pytest

from core.degradation import _known_product_result, predict_degradation_products
from core.compendial import get_pharmacopeia_info, get_regulatory_profile, match_known_impurities
from core.evidence import summarize_concordance
from core.harness import build_worker_report, get_harness_manifest, validate_assessment
from core.ontology import (
    build_strategy_snapshot,
    build_submission_workflow,
    classify_degradation_risk,
    classify_toxicology_risk,
)
from core.qsar import assess_mutagenicity
from core.reporting import generate_regulatory_narrative


ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
NITROBENZENE = "c1ccc(cc1)[N+](=O)[O-]"


@pytest.mark.parametrize("left,right", [([], []), ([{"alert": "Nitro Group"}], [{"alert": "Nitro Group"}]), ([{"alert": "A"}], [])])
def test_rule_overlap_never_establishes_model_concordance(left, right):
    result = summarize_concordance(left, right)
    assert result == "Not assessed (one rule-based screen; no independent model)"


@pytest.mark.parametrize("smiles", [ASPIRIN, NITROBENZENE, "Nc1ccccc1", "invalid SMILES"])
def test_screening_and_error_outputs_never_become_low_toxicology_risk(smiles):
    assessment = assess_mutagenicity(smiles)
    risk, action = classify_toxicology_risk(assessment)
    assert risk == "Review needed"
    assert action
    workflow = build_submission_workflow(assessment)
    assert workflow[1]["Status"] == "Review needed"
    snapshot = build_strategy_snapshot(assessment)
    assert snapshot["toxicology_risk"] == "Review needed"
    assert snapshot["overall_risk"] != "Low"


def test_missing_screening_fields_never_become_low_risk():
    assert classify_toxicology_risk({"status": "review"})[0] == "Review needed"
    assert classify_toxicology_risk({"status": "safe", "class": "ICH M7 Class 5"})[0] == "Review needed"
    assert classify_degradation_risk([{"smiles": "CCO"}])[0] == "Review needed"


@pytest.mark.parametrize("smiles", [ASPIRIN, NITROBENZENE, "Nc1ccccc1"])
def test_narrative_discloses_missing_model_and_unassigned_class(smiles):
    narrative = generate_regulatory_narrative(assess_mutagenicity(smiles), "Synthetic example")
    assert "Classification: Not assigned (expert review required)" in narrative
    assert "Statistical model: Not performed (no statistical model implemented)" in narrative
    assert "concordance: Not assessed" in narrative
    assert "No alert does not establish safety" in narrative
    assert "historical dossier assertions have not been independently verified" in narrative
    for false_claim in ["Concordant positive", "Concordant negative", "Statistical (SAR)", "estimated probability", "0.82", "0.75", "0.79", "0.67"]:
        assert false_claim not in narrative


def test_error_and_missing_output_narratives_do_not_claim_no_alert():
    for assessment in [assess_mutagenicity("invalid SMILES"), {"status": "error"}, {}]:
        narrative = generate_regulatory_narrative(assessment)
        summary = next(line for line in narrative.splitlines() if line.startswith("- Alert summary:"))
        assert "Not performed" in summary
        assert "No alert identified" not in narrative


def test_worker_checks_report_software_completeness_without_certifying_classification():
    assessment = assess_mutagenicity(ASPIRIN)
    package = {"assessment": assessment, "narrative": generate_regulatory_narrative(assessment)}
    report = build_worker_report(package)
    gates = {item["gate"]: item for item in report["validation"]["gates"]}
    assert "Dual QSAR" not in gates
    assert gates["ICH M7 Classification"]["status"] == "review"
    assert gates["Human Review"]["status"] == "review"
    assert "not scientific validation" in report["validation"]["scope"]
    assert "not scientific validation" in get_harness_manifest()["scope"]
    assert report["summary"]["screening_method"] == "Rule-based structural alert screening"
    assert report["summary"]["limitations"]
    assert report["validation"]["review"] >= 2
    assert "Concordant negative" not in json.dumps(report)
    # Presence of legacy field names alone must not pass the disclosure check.
    incomplete = {"qsar_summary": {"expert_call": "Negative", "statistical_call": "Negative"}}
    incomplete_gates = {item["gate"]: item for item in validate_assessment(incomplete)}
    assert incomplete_gates["Screening Transparency"]["status"] == "review"


def test_missing_degradant_structure_cannot_inherit_a_library_ich_class():
    product = _known_product_result({"name": "Synthetic unstructured impurity", "class": 5})
    assert product["class"] == "Not assigned (expert review required)"
    assert product["status"] == "review"
    assert product["limitations"]
    assert classify_degradation_risk([product])[0] == "Review needed"


@pytest.mark.parametrize("smiles", [None, "CCO", NITROBENZENE, "invalid SMILES"])
def test_local_class_three_assertion_never_becomes_high_mutagenic_risk(smiles):
    record = {"name": "Synthetic local class-three entry", "class": 3, "smiles": smiles}
    product = _known_product_result(record)
    assert product["risk"] == "Review needed"
    assert product["risk_basis"]
    assert product["class"] == "Not assigned (expert review required)"
    assert product["known_match"]["class"] == 3
    assert product["known_match"]["source_verified"] is False
    assert product["known_match"]["verification_status"] == "unverified"
    assert "verification_status" not in record  # metadata annotation does not mutate the source library
    assert classify_degradation_risk([product])[0] == "Review needed"
    assert build_strategy_snapshot(degradants=[product])["degradation_risk"] == "Review needed"


def test_predicted_degradants_retain_screening_limitations():
    products = predict_degradation_products("COC(=O)C")
    assert products
    for product in products:
        assert product["class"] == "Not assigned (expert review required)"
        assert product["status"] in {"alert", "review"}
        assert product["screening_method"] == "Rule-based structural alert screening"
        assert product["limitations"]
    assert classify_degradation_risk(products)[0] != "Low"


def test_exported_local_class_values_are_marked_as_unverified_assertions():
    profile = get_pharmacopeia_info("Atorvastatin")
    matches = match_known_impurities("Atorvastatin")
    package_profile = get_regulatory_profile("Atorvastatin")
    for item in profile["impurities"] + matches + package_profile["impurities"]:
        assert item["class_verification_status"] == "unverified"
        assert "not assigned by the screen" in item["class_assignment"]
    source_map = get_regulatory_profile("Synthetic unknown")["regulatory_sources"]
    assert any("does not assign an ICH M7 class" in item["use_in_app"] for item in source_map)


def test_pathway_priority_is_not_reported_as_determined_mutagenic_risk():
    products = predict_degradation_products("COC(=O)C")
    risk, action = classify_degradation_risk(products)
    assert risk == "High"  # Preserve existing hydrolysis pathway review priority.
    assert "heuristic priority does not establish mutagenicity" in action
