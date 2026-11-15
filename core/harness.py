from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


HARNESS_VERSION = "v4.7.0"
ENGINE = "Antigravity v2.5"
POLICY_RANGE = "R01-R13"


POLICY_RULES = [
    {"id": "R01", "name": "Think Before Coding", "check": "Assessment states purpose and scope."},
    {"id": "R02", "name": "Simplicity First", "check": "Output keeps screening logic direct and explainable."},
    {"id": "R03", "name": "Surgical Changes", "check": "Only requested toxicology evidence workflow is affected."},
    {"id": "R04", "name": "Plan Work Review", "check": "Result contains validation gates and worker-report."},
    {"id": "R05", "name": "Regulatory Sync", "check": "Assessment maps to ICH M7 and Q3A/Q3B where relevant."},
    {"id": "R06", "name": "RDKit Availability", "check": "RDKit import status is recorded."},
    {"id": "R07", "name": "Evidence Required", "check": "Every decision has structured evidence objects."},
    {"id": "R08", "name": "Screening Transparency", "check": "Output labels rule-based screening and discloses the absent statistical model."},
    {"id": "R09", "name": "Source Tiering", "check": "Sources are tiered by guideline, compendial, identity, experimental, screening, or inference."},
    {"id": "R10", "name": "Structural Explanation", "check": "Alerting substructure and mechanism are explained."},
    {"id": "R11", "name": "Degradation Link", "check": "Known or predicted degradants are assessed when available."},
    {"id": "R12", "name": "CTD Readiness", "check": "Narrative links decision to submission language."},
    {"id": "R13", "name": "Human Review", "check": "Final result remains decision support pending expert review."},
]


@dataclass
class HarnessGate:
    gate: str
    status: str
    detail: str


def get_harness_manifest(project_id: str = "TXS-2026-001", analyst: str = "Lee Young-nam") -> dict[str, Any]:
    return {
        "status": "Harness Active",
        "version": HARNESS_VERSION,
        "engine": ENGINE,
        "policy": POLICY_RANGE,
        "project_id": project_id,
        "analyst": analyst,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rules": POLICY_RULES,
        "scope": "Software workflow and output completeness checks only; not scientific validation or regulatory acceptance.",
    }


def rdkit_status() -> HarnessGate:
    try:
        from rdkit import rdBase

        return HarnessGate("RDKit", "pass", f"RDKit available: {rdBase.rdkitVersion}")
    except Exception as exc:
        return HarnessGate("RDKit", "fail", f"RDKit unavailable: {exc}")


def validate_assessment(result: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = result.get("evidence_objects", [])
    qsar = result.get("qsar_summary", {})
    alerts = result.get("alerts", [])
    structural = result.get("structural_explanation")

    gates = [
        rdkit_status(),
        HarnessGate(
            "Evidence Object",
            "pass" if evidence else "review",
            f"{len(evidence)} structured evidence item(s) attached.",
        ),
        HarnessGate(
            "Screening Transparency",
            "pass" if result.get("screening_method") and result.get("limitations") and "Not performed" in qsar.get("statistical_call", "") else "review",
            f"Method={result.get('screening_method', 'Not recorded')}; statistical model={qsar.get('statistical_call', 'Not recorded')}. This checks disclosure, not predictive validity.",
        ),
        HarnessGate(
            "Structural Explanation",
            "pass" if structural else "review",
            structural or "No structural explanation generated.",
        ),
        HarnessGate(
            "ICH M7 Classification",
            "review",
            "No final ICH M7 class is established by this rule-based screen. Qualified evidence review is required.",
        ),
        HarnessGate(
            "Alert Traceability",
            "pass" if not alerts or all(a.get("alert") and a.get("method") for a in alerts) else "review",
            f"{len(alerts)} alert(s) with method labels.",
        ),
        HarnessGate(
            "Human Review",
            "review",
            "Decision-support output; final regulatory position requires qualified expert review.",
        ),
    ]
    return [asdict(gate) for gate in gates]


def validate_package(package: dict[str, Any]) -> list[dict[str, Any]]:
    gates = validate_assessment(package.get("assessment", {}))
    known = package.get("known_impurity_matches", [])
    degradants = package.get("degradation_products", [])
    narrative = package.get("narrative", "")
    gates.extend(
        [
            asdict(
                HarnessGate(
                    "Known Impurity Search",
                    "pass" if known else "review",
                    f"{len(known)} known USP/EP/DMF-style impurity match(es).",
                )
            ),
            asdict(
                HarnessGate(
                    "Degradation Assessment",
                    "pass" if degradants else "review",
                    f"{len(degradants)} known or predicted degradation product(s) assessed.",
                )
            ),
            asdict(
                HarnessGate(
                    "Regulatory Narrative",
                    "pass" if len(narrative) > 80 else "review",
                    "Draft narrative generated for expert review." if narrative else "Narrative missing.",
                )
            ),
        ]
    )
    return gates


def build_worker_report(
    package: dict[str, Any],
    project_id: str = "TXS-2026-001",
    analyst: str = "Lee Young-nam",
) -> dict[str, Any]:
    assessment = package.get("assessment", {})
    gates = validate_package(package)
    passed = len([g for g in gates if g["status"] == "pass"])
    review = len([g for g in gates if g["status"] == "review"])
    failed = len([g for g in gates if g["status"] == "fail"])

    return {
        "schema": "worker-report.v1",
        "harness": get_harness_manifest(project_id=project_id, analyst=analyst),
        "summary": {
            "classification": assessment.get("class"),
            "status": assessment.get("status"),
            "qsar_concordance": assessment.get("qsar_summary", {}).get("concordance"),
            "screening_method": assessment.get("screening_method"),
            "limitations": assessment.get("limitations", []),
            "evidence_count": len(assessment.get("evidence_objects", [])),
            "known_impurity_count": len(package.get("known_impurity_matches", [])),
            "degradation_product_count": len(package.get("degradation_products", [])),
        },
        "validation": {
            "scope": "Software workflow and output completeness checks only; not scientific validation or regulatory acceptance.",
            "passed": passed,
            "review": review,
            "failed": failed,
            "gates": gates,
        },
        "decision": {
            "use": "decision support",
            "limitation": "Limited rule-based structural screening only. No trained statistical model, calibrated mutagenicity probability, independent-model concordance, or final ICH M7 class is provided. No-alert and shared-alert findings require review. Local historical dossier assertions require source verification. Workflow checks do not establish scientific validity or regulatory acceptance.",
        },
    }
