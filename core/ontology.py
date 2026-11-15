FDA_GUIDANCE_MAP = [
    {
        "Area": "Toxicology / Mutagenic impurities",
        "Primary basis": "ICH M7(R2)",
        "Platform module": "Structural Alert Screening",
        "Decision use": "Flag implemented structural alerts and identify evidence-review needs; no final ICH M7 class is assigned.",
    },
    {
        "Area": "Impurity / degradant control",
        "Primary basis": "ICH Q3A/Q3B, USP/EP where applicable",
        "Platform module": "Impurity & Degradation",
        "Decision use": "Map known and predicted impurities to control strategy and evidence needs.",
    },
    {
        "Area": "Bioequivalence",
        "Primary basis": "FDA/ICH M13A, FDA IR Dissolution Testing guidance",
        "Platform module": "Bioequivalence Strategy",
        "Decision use": "Compare reference/test dissolution profiles and support BE or biowaiver strategy.",
    },
    {
        "Area": "Submission narrative",
        "Primary basis": "FDA eCTD/CTD quality and nonclinical structure",
        "Platform module": "Report Builder",
        "Decision use": "Convert evidence into draft regulatory language for qualified review.",
    },
]


ROLE_GUIDANCE = {
    "Regulatory Affairs": {
        "focus": "Submission pathway, evidence gap, and FDA/ICH narrative readiness.",
        "next_action": "Confirm application route, target market, product-specific guidance, and required justification package.",
    },
    "CMC / Analytical": {
        "focus": "Impurity control, degradation pathway, dissolution method, and specification setting.",
        "next_action": "Align impurity/degradant controls with method validation, stability data, and dissolution acceptance criteria.",
    },
    "Nonclinical / Toxicology": {
        "focus": "Structural alert screening, evidence gaps, and experimental follow-up for ICH M7 review.",
        "next_action": "Review rule coverage, source verification, and need for additional assessment or experimental evidence.",
    },
    "Executive / Portfolio": {
        "focus": "Development risk, timeline exposure, and go/no-go clarity.",
        "next_action": "Prioritize high-risk evidence gaps that could delay IND/ANDA/NDA decision gates.",
    },
}


def classify_toxicology_risk(results):
    if not results:
        return "Not started", "Run structural alert screening and review supporting evidence."
    if results.get("status") == "error":
        return "Review needed", "Screening did not complete. Resolve the input or dependency error before evidence review."
    if results.get("historical_evidence"):
        return "Review needed", "Local historical dossier assertions require source verification and qualified review."
    if results.get("expert_alerts") or results.get("status") == "alert":
        return "Review needed", "Structural alert detected. Review relevance and additional evidence; no final ICH M7 class or control decision is established."
    return "Review needed", "No alert or missing output from limited rule screening cannot establish low risk or a negative mutagenicity finding."


def classify_be_risk(be_result):
    if not be_result:
        return "Not started", "Enter reference/test dissolution data and calculate f2."
    if be_result.f2 >= 50 and be_result.cv_flag == "Acceptable":
        return "Low", "FDA-style f2 similarity rationale is supported by current dissolution profile."
    if be_result.f2 >= 50:
        return "Medium", "f2 is acceptable, but variability needs justification."
    return "High", "f2 is below 50. Review formulation/process or in vivo BE strategy."


def classify_degradation_risk(degradants):
    if not degradants:
        return "Not started", "Load or predict degradants and map pharmacopeial/FDA evidence."
    high_risk = [
        item
        for item in degradants
        if "Class 1" in str(item.get("class")) or "Class 2" in str(item.get("class")) or item.get("risk") == "High"
    ]
    if high_risk:
        return "High", "One or more degradation entries are flagged for priority review. This heuristic priority does not establish mutagenicity or an ICH M7 class."
    review_items = [
        item for item in degradants
        if not item.get("smiles")
        or item.get("smiles") == "Structure not loaded"
        or item.get("status") in {"review", "error", "alert"}
        or not str(item.get("class", "")).startswith("ICH M7 Class")
    ]
    if review_items:
        return "Review needed", "Degradant screening or structure evidence requires review; absence of a rule alert does not establish low mutagenic risk."
    return "Review needed", "Review degradant source evidence and structural screening limitations before drawing a risk conclusion."


def build_strategy_snapshot(results=None, degradants=None, be_result=None, role="Regulatory Affairs"):
    tox_risk, tox_action = classify_toxicology_risk(results)
    deg_risk, deg_action = classify_degradation_risk(degradants or [])
    be_risk, be_action = classify_be_risk(be_result)
    module_risks = [tox_risk, deg_risk, be_risk]
    if "High" in module_risks:
        overall = "High"
    elif "Medium" in module_risks:
        overall = "Medium"
    elif "Not started" in module_risks or "Review needed" in module_risks:
        overall = "Review needed"
    else:
        overall = "Low"
    role_item = ROLE_GUIDANCE.get(role, ROLE_GUIDANCE["Regulatory Affairs"])
    return {
        "overall_risk": overall,
        "toxicology_risk": tox_risk,
        "degradation_risk": deg_risk,
        "bioequivalence_risk": be_risk,
        "role_focus": role_item["focus"],
        "role_next_action": role_item["next_action"],
        "module_actions": [
            {"Module": "Structural Alert Screening", "Risk": tox_risk, "Next action": tox_action},
            {"Module": "Impurity / Degradation", "Risk": deg_risk, "Next action": deg_action},
            {"Module": "Bioequivalence", "Risk": be_risk, "Next action": be_action},
            {"Module": "Regulatory Evidence", "Risk": "Review needed", "Next action": "Map FDA/ICH/USP sources to the selected submission pathway."},
        ],
    }


def build_submission_workflow(results=None, degradants=None, be_result=None):
    return [
        {"Step": "1. Product profile", "Status": "Ready", "Output": "Project, product, route, dosage form, and submission pathway defined."},
        {"Step": "2. Structural alert screening", "Status": "Review needed" if results else "Pending", "Output": "Limited rule-screen findings and evidence matrix for qualified ICH M7 review; no final class."},
        {"Step": "3. Degradation / impurity", "Status": "Ready" if degradants else "Pending", "Output": "Known/predicted degradants, structure map, source review queue."},
        {"Step": "4. Bioequivalence", "Status": "Ready" if be_result else "Pending", "Output": "FDA f2/bootstrap result and BE strategy decision."},
        {"Step": "5. Source mapping", "Status": "In progress", "Output": "FDA/ICH/USP/EP/DMF evidence basis."},
        {"Step": "6. Submission narrative", "Status": "Draft", "Output": "Regulatory justification for internal review or agency package."},
    ]
