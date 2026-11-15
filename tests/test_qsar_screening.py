"""Rule matches are screening observations, never calibrated toxicity probabilities."""

import builtins
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem

from core.qsar import (
    KNOWN_MUTAGENS,
    assess_mutagenicity,
    get_experimental_detail,
    get_expert_rule_assessment,
    get_statistical_assessment,
)


ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
NITROBENZENE = "c1ccc(cc1)[N+](=O)[O-]"


def assert_no_probability_fields(testcase, value):
    """Check nested evidence/export structures, not just top-level alerts."""
    if isinstance(value, dict):
        for key, item in value.items():
            testcase.assertNotIn("probability", str(key).lower())
            assert_no_probability_fields(testcase, item)
    elif isinstance(value, list):
        for item in value:
            assert_no_probability_fields(testcase, item)


class RuleBasedScreeningTests(unittest.TestCase):
    def test_each_rule_family_reports_atom_matches_without_probability(self):
        examples = (
            ("Nc1ccccc1", "Aromatic Amine"),
            ("Nc1ccc2ccccc2c1", "Fused-Ring Aromatic Amine"),
            (NITROBENZENE, "Nitro Group"),
            ("C1OC1", "Epoxide / Aziridine"),
            ("C1NC1", "Epoxide / Aziridine"),
            ("CCCl", "Alkyl Halide / Acyl Halide"),
            ("CC(=O)Cl", "Alkyl Halide / Acyl Halide"),
        )
        for smiles, expected_alert in examples:
            with self.subTest(smiles=smiles):
                alerts = get_expert_rule_assessment(smiles)
                matching = [a for a in alerts if a["alert"] == expected_alert]
                self.assertTrue(matching, expected_alert)
                atom_count = Chem.MolFromSmiles(smiles).GetNumAtoms()
                for alert in alerts:
                    self.assertIn("rule", alert["method"].lower())
                    self.assertTrue(alert["matched_atoms"])
                    self.assertTrue(all(
                        0 <= index < atom_count
                        for match in alert["matched_atoms"] for index in match
                    ))
                    self.assertNotEqual(alert.get("likelihood"), "Certain")
                assert_no_probability_fields(self, alerts)

    def test_legacy_statistical_entry_point_never_invents_second_model(self):
        for smiles in (NITROBENZENE, "Nc1ccccc1", ASPIRIN, "invalid"):
            with self.subTest(smiles=smiles):
                self.assertEqual(get_statistical_assessment(smiles), [])

    def test_no_rule_match_remains_unknown_and_needs_review(self):
        result = assess_mutagenicity(ASPIRIN)
        self.assertEqual(result["status"], "review")
        self.assertEqual(result["alerts"], [])
        self.assertEqual(result["expert_alerts"], [])
        self.assertEqual(result["statistical_alerts"], [])
        self.assertNotIn("ICH M7 Class", result["class"])
        self.assertIn("rule-based", result["screening_method"].lower())
        self.assertTrue(result["limitations"])
        summary = result["qsar_summary"]
        self.assertNotEqual(summary["expert_call"].lower(), "negative")
        self.assertIn("not performed", summary["statistical_call"].lower())
        self.assertNotIn("concordant", summary["concordance"].lower())
        self.assertIn("no alert", result["structural_explanation"].lower())

    def test_matched_rule_is_screening_alert_not_an_ich_class(self):
        result = assess_mutagenicity(NITROBENZENE)
        self.assertEqual(result["status"], "alert")
        self.assertEqual([a["alert"] for a in result["expert_alerts"]], ["Nitro Group"])
        self.assertEqual(result["statistical_alerts"], [])
        self.assertEqual(result["alerts"], result["expert_alerts"])
        self.assertNotIn("ICH M7 Class", result["class"])
        self.assertIn("not performed", result["qsar_summary"]["statistical_call"].lower())
        self.assertNotIn("concordant", result["qsar_summary"]["concordance"].lower())
        assert_no_probability_fields(self, result)

    def test_shared_drug_substance_motif_never_establishes_control_or_class_four(self):
        # The broad nitro SMARTS matches aromatic and aliphatic environments.
        result = assess_mutagenicity(
            NITROBENZENE, drug_substance_smiles="c1ccccc1CC[N+](=O)[O-]"
        )
        self.assertEqual(result["status"], "alert")
        self.assertNotIn("ICH M7 Class", result["class"])
        self.assertIn("Nitro Group", result["shared_alerts"])
        self.assertIn("review", result["note"].lower())

    def test_invalid_drug_substance_does_not_hide_impurity_alert(self):
        result = assess_mutagenicity(NITROBENZENE, drug_substance_smiles="not a smiles")
        self.assertEqual(result["status"], "alert")
        self.assertTrue(result["expert_alerts"])
        self.assertNotIn("ICH M7 Class", result["class"])
        self.assertEqual(result["shared_alerts"], [])

    def test_invalid_or_empty_structure_is_an_error_not_safe(self):
        for smiles in ("invalid", ""):
            with self.subTest(smiles=smiles):
                result = assess_mutagenicity(smiles)
                self.assertEqual(result["status"], "error")
                self.assertTrue(result["message"])
                self.assertEqual(result["alerts"], [])

    def test_missing_rdkit_is_an_error_not_no_alert(self):
        original_import = builtins.__import__

        def without_rdkit(name, *args, **kwargs):
            if name == "rdkit" or name.startswith("rdkit."):
                raise ImportError("RDKit unavailable in regression test")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_rdkit):
            result = assess_mutagenicity(NITROBENZENE)
        self.assertEqual(result["status"], "error")
        self.assertIn("RDKit", result["message"])
        self.assertEqual(result["alerts"], [])

    def test_local_historical_assertions_are_unverified_not_confirmed_results(self):
        detail = get_experimental_detail("Nc1ccccc1")
        self.assertIsNotNone(detail)
        result = assess_mutagenicity("Nc1ccccc1")
        self.assertEqual(result["status"], "review")
        self.assertNotIn("ICH M7 Class", result["class"])
        text = json.dumps({"detail": detail, "result": result}).lower()
        self.assertTrue("unverified" in text or "not independently verified" in text)
        self.assertNotIn("confirmed in historical toxicology studies", text)
        self.assertNotIn('"result": "known positive"', text)
        historical_evidence = [
            e for e in result["evidence_objects"]
            if "historical" in e["evidence_type"].lower()
            or "assay" in e["evidence_type"].lower()
            or "dossier" in e["evidence_type"].lower()
        ]
        self.assertTrue(historical_evidence)
        for item in historical_evidence:
            with self.subTest(evidence_type=item["evidence_type"]):
                self.assertNotEqual(item["confidence"], "High")
        # Historical claims are evidence for review, not another computed model.
        for alert in result["expert_alerts"]:
            self.assertIn("rule", alert["method"].lower())
        self.assertEqual(result["statistical_alerts"], [])
        assert_no_probability_fields(self, result)

    def test_serialized_results_do_not_retain_fixed_probability_claims(self):
        for smiles in (ASPIRIN, NITROBENZENE, "Nc1ccccc1", "C1OC1", "CCCl"):
            with self.subTest(smiles=smiles):
                result = assess_mutagenicity(smiles)
                assert_no_probability_fields(self, result)
                text = json.dumps(result).lower()
                for misleading in (
                    "estimated probability", "statistical (sar)",
                    "dual-method qsar screen", "concordant positive", "concordant negative",
                ):
                    self.assertNotIn(misleading, text)

    def test_equivalent_smiles_preserve_unverified_local_dossier_leads(self):
        for source_smiles, source_record in KNOWN_MUTAGENS.items():
            canonical = Chem.MolToSmiles(Chem.MolFromSmiles(source_smiles), isomericSmiles=True)
            for smiles in (source_smiles, canonical):
                with self.subTest(record=source_record["name"], smiles=smiles):
                    result = assess_mutagenicity(smiles)
                    self.assertEqual(result["status"], "review")
                    self.assertNotIn("ICH M7 Class", result["class"])
                    self.assertTrue(result["historical_evidence"])
                    record = result["historical_evidence"][0]
                    self.assertEqual(record["name"], source_record["name"])
                    self.assertEqual(record["verification_status"], "unverified")
                    self.assertFalse(record["source_verified"])


if __name__ == "__main__":
    unittest.main()
