def generate_regulatory_narrative(result: dict, compound_name: str = "the submitted compound") -> str:
    ttc = result.get("ttc_info", {})
    qsar = result.get("qsar_summary", {})
    alerts = result.get("alerts", [])
    if result.get("status") == "error":
        alert_text = "Not performed (screening was not completed)"
    else:
        alert_text = ", ".join(sorted({a.get("alert", "Unknown alert") for a in alerts}))
        alert_text = alert_text or qsar.get("expert_call", "Not performed (screening completion not established)")
    limitations = result.get("limitations", [])
    limitation_text = " ".join(limitations) if isinstance(limitations, list) else str(limitations)

    return f"""
A limited rule-based structural alert screen was requested for {compound_name}. This output supports evidence review; it does not establish an ICH M7 classification.

Assessment result:
- Classification: {result.get('class', 'Not determined')}
- Status: {result.get('status', 'review')}
- Screening method: {result.get('screening_method', 'Rule-based structural alert screening')}
- Rule screening finding: {qsar.get('expert_call', 'Not evaluated')}
- Statistical model: Not performed (no statistical model implemented)
- Independent-model concordance: Not assessed (one rule-based screen; no independent model)
- Alert summary: {alert_text}

Exposure calculation context (not a compound-specific acceptable intake or control decision):
- Generic TTC calculation: {ttc.get('limit_ug_day', 'N/A')} ug/day
- Calculated concentration at entered dose: {ttc.get('limit_ppm', 'N/A')} ppm

Structural interpretation:
{result.get('structural_explanation', 'No structural explanation available.')}

Limitations and review needs:
{limitation_text}
The implemented SMARTS rules have limited coverage and do not provide a trained statistical model or calibrated mutagenicity probabilities. No alert does not establish safety or a negative mutagenicity finding. A shared drug-substance alert does not establish control or ICH M7 Class 4. Local historical dossier assertions have not been independently verified. Final ICH M7 classification requires qualified review of source evidence, applicability, and any additional assessment or experimental evidence.
""".strip()
