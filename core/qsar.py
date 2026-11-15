from __future__ import annotations

from copy import deepcopy
from typing import Any

from .evidence import make_evidence, sort_evidence, summarize_concordance


SCREENING_METHOD = "Rule-based structural alert screening"
UNASSIGNED_CLASS = "Not assigned (expert review required)"
SCREENING_LIMITATIONS = [
    "Limited hand-coded SMARTS screen; no trained statistical model or independent second model is implemented.",
    "No calibrated mutagenicity probabilities, performance validation or validated applicability domain are available.",
    "Absence of a matched rule does not establish a negative mutagenicity result or safety.",
    "A shared rule motif with the drug substance does not establish equivalent reactivity, safety or impurity control.",
    "ICH M7 classification requires qualified review and supporting evidence; this screen does not assign a class.",
    "Local historical dossier entries and their citations have not been independently verified.",
]


ASHBY_ALERTS: dict[str, dict[str, Any]] = {
    "Aromatic Amine": {
        "smarts": "[NX3;H2,H1;!$(NC=O)]-c1ccccc1",
        "likelihood": "Not estimated (rule match only)",
        "mechanism": "Potential metabolic activation to DNA-reactive nitrenium ions; relevance requires expert review.",
        "reference": "ICH M7(R2); Ashby & Tennant structural alert framework",
        "priority": "Critical",
        "expert_comment": "Primary aromatic amines/anilines are key structural alerts for bacterial mutagenicity.",
    },
    "Fused-Ring Aromatic Amine": {
        "smarts": "[NX3;H2,H1;!$(NC=O)]-c1ccc2ccccc2c1",
        "likelihood": "Not estimated (rule match only)",
        "mechanism": "Planar fused rings may enhance DNA interaction; amine activation can form DNA-reactive species.",
        "reference": "IARC monograph context; ICH M7 expert review principle",
        "priority": "Critical",
        "expert_comment": "Naphthylamine-like fused systems warrant conservative expert review.",
    },
    "Nitro Group": {
        "smarts": "[N+](=O)[O-]",
        "likelihood": "Not estimated (rule match only)",
        "mechanism": "Reductive metabolism can form hydroxylamine/nitrenium-like DNA-reactive intermediates.",
        "reference": "ICH M7(R2); Kazius et al. structural alert literature",
        "priority": "High",
        "expert_comment": "Aromatic nitro groups require context-specific review and often confirmatory Ames evidence.",
    },
    "Epoxide / Aziridine": {
        "smarts": "C1[O,N]C1",
        "likelihood": "Not estimated (rule match only)",
        "mechanism": "Strained three-membered rings can directly alkylate nucleophilic DNA sites via ring opening.",
        "reference": "ICH M7(R2) DNA-reactive alert principle; Miller & Miller electrophile framework",
        "priority": "Critical",
        "expert_comment": "Epoxides and aziridines are direct electrophile alerts unless deactivated by local structure.",
    },
    "Alkyl Halide / Acyl Halide": {
        "smarts": "[CX4,CX3][Cl,Br,I]",
        "likelihood": "Not estimated (rule match only)",
        "mechanism": "Electrophilic carbon-halogen centers can alkylate biological nucleophiles depending on stability and exposure.",
        "reference": "ICH M7(R2) expert review principle",
        "priority": "High",
        "expert_comment": "Assess reactivity, purge, and exposure before assigning final class.",
    },
}


EXPERIMENTAL_DB = {
    "Nc1ccccc1": {
        "name": "Aniline",
        "verification_status": "unverified",
        "source_verified": False,
        "assay_data": [
            {"test": "Ames (TA98)", "result": "Positive", "condition": "+S9 activation", "source": "NTP TR-211"},
            {"test": "Ames (TA100)", "result": "Positive", "condition": "+S9 activation", "source": "NTP TR-211"},
            {"test": "In vitro micronucleus", "result": "Positive", "condition": "Human lymphocytes", "source": "IARC Vol. 77"},
        ],
        "overall_conclusion": "The local dossier reports mutagenicity under metabolic activation. Its assay assertions and citations have not been independently verified.",
    }
}


KNOWN_MUTAGENS = {
    "Nc1ccccc1": {
        "class": 2,
        "name": "Aniline",
        "evidence": "Local dossier reports positive Ames evidence and cites NTP; not independently verified",
        "verification_status": "unverified",
        "source_verified": False,
    },
    "c1ccc2c(c1)ccc3c2ccc4c3ccc5c4cccc5": {
        "class": 1,
        "name": "Benzo[a]pyrene",
        "evidence": "Local dossier labels this structure as a carcinogen/mutagen; identity and source assertions not independently verified",
        "verification_status": "unverified",
        "source_verified": False,
    },
}


def canonicalize_smiles(smiles: str | None) -> str | None:
    if not smiles:
        return None
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return smiles


def calculate_ttc_limit(daily_dose_mg: float, duration_days: int = 3650) -> dict[str, Any]:
    if duration_days <= 30:
        limit_ug = 120
    elif duration_days <= 365:
        limit_ug = 20
    elif duration_days <= 3650:
        limit_ug = 10
    else:
        limit_ug = 1.5

    if daily_dose_mg > 0:
        return {"limit_ug_day": limit_ug, "limit_ppm": round(limit_ug / daily_dose_mg, 2), "duration_days": duration_days}
    return {"limit_ug_day": limit_ug, "limit_ppm": "N/A", "duration_days": duration_days}


def _matched_atoms(mol: Any, pattern: Any) -> list[list[int]]:
    return [list(match) for match in mol.GetSubstructMatches(pattern)]


def get_expert_rule_assessment(smiles: str) -> list[dict[str, Any]]:
    """Return matched hand-coded rules, not a mutagenicity prediction.

    An empty list is only a list of matches. It does not establish a negative
    result; use assess_mutagenicity for input/dependency errors and limitations.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return []

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return []

    results = []
    for name, data in ASHBY_ALERTS.items():
        pattern = Chem.MolFromSmarts(data["smarts"])
        if pattern and mol.HasSubstructMatch(pattern):
            atoms = _matched_atoms(mol, pattern)
            results.append(
                {
                    "method": SCREENING_METHOD,
                    "alert": name,
                    "likelihood": data["likelihood"],
                    "mechanism": data["mechanism"],
                    "reference": data["reference"],
                    "priority": data["priority"],
                    "expert_comment": data["expert_comment"],
                    "matched_atoms": atoms,
                    "reasoning": f"{name} SMARTS matched atom index set(s): {atoms}. {data['expert_comment']}",
                }
            )
    return results


def get_statistical_assessment(smiles: str) -> list[dict[str, Any]]:
    """Deprecated compatibility API: no statistical model is implemented.

    Always returns an empty list, meaning no statistical assessment was
    performed, never a negative prediction. Rule matches are available from
    get_expert_rule_assessment; assessment metadata states the missing method.
    """
    return []


def get_experimental_detail(smiles: str | None) -> dict[str, Any] | None:
    can_smiles = canonicalize_smiles(smiles)
    if not can_smiles:
        return None
    record = EXPERIMENTAL_DB.get(can_smiles) or EXPERIMENTAL_DB.get(smiles or "")
    return deepcopy(record) if record else None


def _experimental_evidence(smiles: str, compound: str) -> list[dict[str, Any]]:
    exp = get_experimental_detail(smiles)
    if not exp:
        return []
    evidence = []
    for assay in exp["assay_data"]:
        evidence.append(
            make_evidence(
                compound=exp["name"],
                smiles=smiles,
                evidence_type="unverified local assay assertion",
                endpoint=assay["test"],
                result=f"Local dossier reports {assay['result']} (unverified)",
                source_tier="inference",
                source_name=f"Unverified local dossier citation: {assay['source']}",
                reasoning=f"The local dossier reports {assay['test']}: {assay['result']} under {assay['condition']}. The source and assay assertion have not been independently verified.",
                confidence="Low",
                regulatory_mapping=["Source verification required before ICH M7 interpretation"],
                details={**assay, "verification_status": "unverified", "source_verified": False},
            )
        )
    return evidence


def _screening_summary(expert_alerts: list[dict[str, Any]]) -> dict[str, str]:
    return {
        "expert_call": "Alert identified" if expert_alerts else "No alert identified in implemented rules",
        "statistical_call": "Not performed (no statistical model implemented)",
        "concordance": summarize_concordance(expert_alerts, []),
        "applicability_domain": "Limited SMARTS rule coverage; applicability domain and predictive performance have not been validated.",
    }


def _screening_error(message: str) -> dict[str, Any]:
    summary = _screening_summary([])
    summary["expert_call"] = "Not performed (screening error)"
    return {
        "status": "error",
        "message": message,
        "class": UNASSIGNED_CLASS,
        "screening_method": SCREENING_METHOD,
        "alerts": [],
        "expert_alerts": [],
        "statistical_alerts": [],
        "historical_evidence": [],
        "shared_alerts": [],
        "qsar_summary": summary,
        "evidence_objects": [],
        "limitations": list(SCREENING_LIMITATIONS),
        "note": "Screening was not completed. Expert review is required.",
    }


def assess_mutagenicity(
    smiles: str,
    drug_substance_smiles: str | None = None,
    daily_dose_mg: float = 10,
    compound_name: str = "Submitted compound",
) -> dict[str, Any]:
    """Screen limited structural rules without predicting safety or an ICH class.

    Legacy statistical/qsar keys remain for callers, with an explicit missing
    statistical method. Local dossier assertions are unverified source leads.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return _screening_error("RDKit not installed")

    if not isinstance(smiles, str) or not smiles.strip():
        return _screening_error("Invalid SMILES")
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return _screening_error("Invalid SMILES")

    can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    expert_alerts = get_expert_rule_assessment(can_smiles)
    evidence = []
    for alert in expert_alerts:
        evidence.append(
            make_evidence(
                compound=compound_name,
                smiles=can_smiles,
                evidence_type="structural rule match",
                endpoint="bacterial mutagenicity screening",
                result="Rule matched; mutagenicity not determined",
                source_tier="qsar",
                source_name=SCREENING_METHOD,
                reasoning=alert["reasoning"],
                confidence="Low",
                method=SCREENING_METHOD,
                alert=alert["alert"],
                mechanism=alert["mechanism"],
                regulatory_mapping=["Expert review required; no ICH M7 class assigned"],
                details=alert,
            )
        )

    shared_alerts = []
    ds_note = ""
    if drug_substance_smiles:
        ds_mol = Chem.MolFromSmiles(drug_substance_smiles)
        if ds_mol:
            for alert in expert_alerts:
                pattern = Chem.MolFromSmarts(ASHBY_ALERTS[alert["alert"]]["smarts"])
                if pattern and ds_mol.HasSubstructMatch(pattern):
                    shared_alerts.append(alert["alert"])
            if shared_alerts:
                ds_note = (
                    " Shared rule motif(s) with the drug substance: " + ", ".join(shared_alerts)
                    + ". Motif overlap alone does not establish equivalent reactivity, a negative drug-substance result or impurity control; expert review remains required."
                )
        else:
            ds_note = " Drug-substance SMILES was invalid; shared-motif comparison was not performed."

    historical_evidence = []
    # Keep local assertions as source leads, including their original class hint.
    # Neither the cited assays nor the dossier's identity/class claims are verified.
    known = next(
        (record for key, record in KNOWN_MUTAGENS.items() if canonicalize_smiles(key) == can_smiles),
        None,
    )
    if known:
        historical = {
            "method": "Unverified local dossier",
            "alert": "Local historical record (unverified)",
            "name": known["name"],
            "evidence": known["evidence"],
            "reference": known["evidence"],
            "reported_ich_m7_class": known["class"],
            "verification_status": "unverified",
            "source_verified": False,
            "reasoning": "A local dossier entry matches the submitted structure. Its identity, toxicology and class assertions have not been independently verified; source review is required.",
        }
        historical_evidence.append(historical)
        evidence.append(
            make_evidence(
                compound=known["name"],
                smiles=can_smiles,
                evidence_type="unverified local historical assertion",
                endpoint="reported mutagenicity / carcinogenicity",
                result="Local dossier assertion; unverified",
                source_tier="inference",
                source_name="Unverified local historical dossier",
                reasoning=historical["reasoning"],
                confidence="Low",
                method=historical["method"],
                alert=historical["alert"],
                regulatory_mapping=["Source verification required; no ICH M7 class assigned"],
                details=historical,
            )
        )
    evidence.extend(_experimental_evidence(can_smiles, compound_name))

    if expert_alerts:
        note = "Structural rule match(es) require expert review. Mutagenicity is undetermined and no ICH M7 class is assigned."
        structural_explanation = " | ".join(f"{a['alert']}: {a['mechanism']}" for a in expert_alerts)
    else:
        note = "No alert identified in the limited implemented rules. Mutagenicity remains undetermined; this is not a negative result or a safety conclusion."
        structural_explanation = "No alert identified in implemented SMARTS rules; rule coverage is limited and has not been validated."
        evidence.append(
            make_evidence(
                compound=compound_name,
                smiles=can_smiles,
                evidence_type="structural screening observation",
                endpoint="bacterial mutagenicity screening",
                result="No rule matched; mutagenicity undetermined",
                source_tier="qsar",
                source_name=SCREENING_METHOD,
                reasoning=note,
                confidence="Low",
                method=SCREENING_METHOD,
                regulatory_mapping=["Expert review required; no ICH M7 class assigned"],
                details={"limitations": list(SCREENING_LIMITATIONS)},
            )
        )
    if historical_evidence:
        note += " A local historical dossier entry also requires source verification before use."

    return {
        "status": "review" if historical_evidence or not expert_alerts else "alert",
        "class": UNASSIGNED_CLASS,
        "screening_method": SCREENING_METHOD,
        "alerts": expert_alerts + historical_evidence,
        "expert_alerts": expert_alerts,
        "statistical_alerts": [],
        "historical_evidence": historical_evidence,
        "shared_alerts": shared_alerts,
        "qsar_summary": _screening_summary(expert_alerts),
        "structural_explanation": structural_explanation,
        "evidence_objects": sort_evidence(evidence),
        "ttc_info": calculate_ttc_limit(daily_dose_mg),
        "canonical_smiles": can_smiles,
        "limitations": list(SCREENING_LIMITATIONS),
        "note": note + ds_note,
    }
