"""Deterministic regression checks for the public regulatory API."""

import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.bioequivalence import DEFAULT_DISSOLUTION_PROFILE, calculate_f2
from core.ontology import build_strategy_snapshot, build_submission_workflow
from core.regulatory import (
    assess_genotoxicity,
    build_harnessed_evidence_package,
    get_smiles_from_name,
    predict_degradation_products,
)


ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
NITROBENZENE = "c1ccc(cc1)[N+](=O)[O-]"


class RegulatoryIntegrationTests(unittest.TestCase):
    def test_public_assessment_preserves_evidence_api_without_safety_claim(self):
        result = assess_genotoxicity(ASPIRIN, compound_name="Synthetic test compound")
        self.assertEqual(result["status"], "review")
        self.assertNotIn("ICH M7 Class", result["class"])
        for key in (
            "alerts", "expert_alerts", "statistical_alerts", "qsar_summary",
            "structural_explanation", "evidence_objects", "ttc_info",
            "harness_manifest", "validation_gates",
        ):
            self.assertIn(key, result)
        self.assertTrue(result["evidence_objects"])
        self.assertTrue(result["validation_gates"])

    def test_package_and_worker_report_remain_serializable_without_network(self):
        with patch("requests.get", side_effect=AssertionError("Unexpected network request")):
            package = build_harnessed_evidence_package(
                "Synthetic test compound", NITROBENZENE,
                project_id="TEST-SAR", analyst="Regression test",
            )
        self.assertEqual(package["assessment"]["status"], "alert")
        self.assertEqual(package["assessment"]["statistical_alerts"], [])
        self.assertTrue(package["degradation_products"])
        report = package["worker_report"]
        self.assertEqual(report["schema"], "worker-report.v1")
        self.assertEqual(report["harness"]["project_id"], "TEST-SAR")
        self.assertEqual(report["harness"]["analyst"], "Regression test")
        self.assertTrue(report["validation"]["gates"])
        self.assertIn("review", report["validation"])
        self.assertTrue(json.dumps(package))

    def test_local_identity_lookup_does_not_need_external_requests(self):
        with patch("requests.get", side_effect=AssertionError("Unexpected network request")):
            result = get_smiles_from_name("Atorvastatin")
        self.assertIsNotNone(result)
        self.assertTrue(result["smiles"])
        self.assertTrue(result["evidence"])

    def test_degradation_screen_propagates_review_instead_of_safe(self):
        degradants = predict_degradation_products(NITROBENZENE)
        self.assertTrue(degradants)
        for item in degradants:
            with self.subTest(product=item["smiles"]):
                self.assertNotIn(item["status"], ("safe", "controlled"))
                self.assertNotIn("ICH M7 Class", str(item["class"]))
                self.assertTrue(item["evidence_objects"])

    def test_be_python_fallback_and_strategy_integration_are_preserved(self):
        # Retain the separately calculated BE output while correcting SAR labels.
        with patch("core.bioequivalence._run_r_backend", return_value=None):
            result = calculate_f2(DEFAULT_DISSOLUTION_PROFILE, bootstrap_runs=50, seed=1729)
        expected_f2 = round(50 * math.log10(100 / math.sqrt(1 + 4)), 2)
        self.assertEqual(result.f2, expected_f2)
        self.assertFalse(result.r_backend_used)
        self.assertEqual(result.bootstrap_runs, 50)
        self.assertIsNotNone(result.ci_low)
        self.assertIsNotNone(result.ci_high)
        self.assertLessEqual(result.ci_low, result.ci_high)
        self.assertIsNotNone(result.probability_f2_ge_50)
        assessment = assess_genotoxicity(ASPIRIN)
        snapshot = build_strategy_snapshot(assessment, [], result)
        workflow = build_submission_workflow(assessment, [], result)
        self.assertEqual(snapshot["toxicology_risk"], "Review needed")
        self.assertTrue(snapshot["bioequivalence_risk"])
        self.assertGreaterEqual(len(workflow), 5)


if __name__ == "__main__":
    unittest.main()
