import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import urllib.parse
import requests
import hashlib
import json
import platform

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from core.regulatory import (
        build_evidence_package,
        build_harnessed_evidence_package,
        generate_regulatory_narrative,
        get_smiles_from_name,
        assess_genotoxicity,
        predict_degradation_products,
        get_pharmacopeia_info,
        get_experimental_detail,
        match_known_impurities,
    )
    from core.bioequivalence import (
        DEFAULT_DISSOLUTION_PROFILE,
        calculate_f2,
        dissolution_profile_summary,
        FDA_BE_GUIDANCE_SOURCES,
    )
    from core.ontology import (
        FDA_GUIDANCE_MAP,
        ROLE_GUIDANCE,
        build_strategy_snapshot,
        build_submission_workflow,
    )
except ImportError as e:
    st.error(f"Module Import Error: {e}")
    st.stop()

# Try importing RDKit. Keep core chemistry separate from drawing because
# Streamlit Cloud can have drawing-backend differences.
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdDepictor, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError as e:
    RDKIT_IMPORT_ERROR = str(e)
    RDKIT_AVAILABLE = False

try:
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_DRAW_AVAILABLE = True
    RDKIT_DRAW_ERROR = ""
except ImportError as e:
    RDKIT_DRAW_ERROR = str(e)
    RDKIT_DRAW_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="ToxiGuard-Platform | Regulatory Strategy",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS: Premium Design System ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --bg-dark: #0f172a;
    --accent: #0ea5e9;
    --accent-glow: rgba(14, 165, 233, 0.3);
    --glass: rgba(255, 255, 255, 0.03);
    --glass-border: rgba(255, 255, 255, 0.1);
    --text-main: #f1f5f9;
}

.stApp {
    background-color: var(--bg-dark);
    color: var(--text-main);
    font-family: 'Outfit', sans-serif;
}

.glass-card {
    background: var(--glass);
    backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: 24px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: 4.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #fff 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.accent-text {
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.3em;
    font-weight: 700;
    font-size: 0.9rem;
}

.badge {
    padding: 0.4rem 1rem;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 800;
    text-transform: uppercase;
}

.badge-class1 { background: #ef4444; color: white; }
.badge-class3 { background: #f59e0b; color: white; }
.badge-review { background: #f59e0b; color: white; }

</style>
""", unsafe_allow_html=True)

# --- State Management ---
if "smiles" not in st.session_state:
    st.session_state.smiles = ""
if "results" not in st.session_state:
    st.session_state.results = None
if "degradants" not in st.session_state:
    st.session_state.degradants = []
if "identity" not in st.session_state:
    st.session_state.identity = {}
if "known_impurities" not in st.session_state:
    st.session_state.known_impurities = []
if "evidence_package" not in st.session_state:
    st.session_state.evidence_package = None
if "be_profile" not in st.session_state:
    st.session_state.be_profile = DEFAULT_DISSOLUTION_PROFILE.copy()
if st.session_state.get("be_result") is not None and (
    getattr(st.session_state.be_result, "schema", None) != "dissolution-f2-simulation.v2"
    or not st.session_state.get("be_request_fingerprint")
):
    st.session_state.be_result = None
    st.session_state.be_export = None
    st.session_state.be_invalidation_notice = True

# An existing browser session may still hold output from the former fixed-score
# implementation. Require a new screen before showing those derived packages.
legacy_screening_reset = bool(
    (st.session_state.results and "screening_method" not in st.session_state.results)
    or (
        st.session_state.evidence_package
        and "screening_method" not in st.session_state.evidence_package.get("assessment", {})
    )
)
if legacy_screening_reset:
    st.session_state.results = None
    st.session_state.evidence_package = None
    st.session_state.degradants = []
    st.session_state.known_impurities = []

# --- UI Layout ---
with st.sidebar:
    workspace = st.radio("Workspace", ["Strategy Dashboard", "Toxicity Evidence Review"], key="workspace")
    st.caption("NTR integration 2026.11.14")
    if workspace == "Strategy Dashboard":
        st.markdown("<div class='accent-text'>Harness Active</div>", unsafe_allow_html=True)
    st.title("Project Scope")
    project_id = st.text_input("Project ID", value="TXS-2026-001")
    analyst = st.text_input("Expert Analyst", value="Lee Young-nam")
    development_stage = st.selectbox(
        "Development Stage",
        ["Preformulation", "Preclinical", "IND-enabling", "Phase 1", "Phase 2/3", "ANDA / Generic", "NDA / 505(b)(2)", "Post-approval change"],
        index=5,
    )
    submission_path = st.selectbox(
        "Submission Pathway",
        ["ANDA", "NDA", "505(b)(2)", "IND", "DMF", "IMPD / CTA", "Post-approval variation"],
        index=0,
    )
    user_role = st.selectbox("Platform View", list(ROLE_GUIDANCE.keys()), index=0)
    dosage_form = st.selectbox(
        "Dosage Form",
        ["Immediate-release tablet/capsule", "Modified-release oral solid", "Oral solution", "Injectable", "Semisolid", "Other"],
        index=0,
    )
    reference_product = st.text_input("Reference Product / RLD", value="")
    daily_dose_mg = st.number_input("Daily Dose (mg/day)", min_value=0.001, value=10.0, step=1.0)
    st.markdown("---")
    st.markdown("### Screening scope")
    st.checkbox("ICH M7(R2) reference context", value=True, disabled=True)
    st.checkbox("Limited SMARTS structural alert rules", value=True, disabled=True)
    st.checkbox("Proactive Degradation", value=True)
    st.checkbox("Dissolution f2 comparison", value=True)

def run_assessment(compound_name, smiles_value):
    package = build_harnessed_evidence_package(
        compound_name,
        smiles_value,
        daily_dose_mg=daily_dose_mg,
        project_id=project_id,
        analyst=analyst,
    )
    st.session_state.assessment_compound_name = compound_name
    st.session_state.evidence_package = package
    st.session_state.results = package["assessment"]
    st.session_state.degradants = package["degradation_products"]
    st.session_state.known_impurities = package["known_impurity_matches"]

def build_structure_profile(smiles_value):
    if not RDKIT_AVAILABLE or not smiles_value:
        return None
    mol = Chem.MolFromSmiles(smiles_value)
    if not mol:
        return None
    return {
        "mol": mol,
        "canonical_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "ring_count": Lipinski.RingCount(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
    }

def _simple_svg_for_molecule(mol, highlight_atoms=None, width=620, height=420):
    if mol is None:
        return None
    highlight_atoms = set(highlight_atoms or [])
    work_mol = Chem.Mol(mol)
    try:
        rdDepictor.Compute2DCoords(work_mol)
    except Exception:
        return None

    conf = work_mol.GetConformer()
    drawable_atoms = [
        atom.GetIdx()
        for atom in work_mol.GetAtoms()
        if atom.GetDegree() > 0 or atom.GetIdx() in highlight_atoms
    ]
    if not drawable_atoms:
        drawable_atoms = [atom.GetIdx() for atom in work_mol.GetAtoms()]
    counter_ions = []
    for atom in work_mol.GetAtoms():
        if atom.GetIdx() not in drawable_atoms:
            symbol = atom.GetSymbol()
            charge = atom.GetFormalCharge()
            if charge > 0:
                symbol = f"{symbol}+"
            elif charge < 0:
                symbol = f"{symbol}-"
            counter_ions.append(symbol)

    coord_by_atom = {}
    for atom_idx in drawable_atoms:
        pos = conf.GetAtomPosition(atom_idx)
        coord_by_atom[atom_idx] = (pos.x, pos.y)
    if not coord_by_atom:
        return None

    min_x = min(x for x, _ in coord_by_atom.values())
    max_x = max(x for x, _ in coord_by_atom.values())
    min_y = min(y for _, y in coord_by_atom.values())
    max_y = max(y for _, y in coord_by_atom.values())
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    pad = 58
    scale = min((width - 2 * pad) / span_x, (height - 2 * pad) / span_y)
    scale = min(scale * 1.18, 150)

    def xy(atom_idx):
        x, y = coord_by_atom[atom_idx]
        return (
            width / 2 + (x - (min_x + max_x) / 2) * scale,
            height / 2 - (y - (min_y + max_y) / 2) * scale,
        )

    bond_parts = []
    alert_halo_parts = []
    if highlight_atoms:
        for atom_idx in highlight_atoms:
            if atom_idx in coord_by_atom:
                hx, hy = xy(atom_idx)
                alert_halo_parts.append(
                    f"<circle cx='{hx:.1f}' cy='{hy:.1f}' r='24' fill='#fee2e2' stroke='#ef4444' stroke-width='1.5' opacity='0.72' />"
                )
    for bond in work_mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 not in coord_by_atom or a2 not in coord_by_atom:
            continue
        x1, y1 = xy(a1)
        x2, y2 = xy(a2)
        order = bond.GetBondTypeAsDouble()
        color = "#111827"
        stroke_width = 2.4
        if a1 in highlight_atoms or a2 in highlight_atoms:
            color = "#b91c1c"
            stroke_width = 3.2
        bond_parts.append(
            f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{color}' stroke-width='{stroke_width}' stroke-linecap='round' />"
        )
        if order >= 2:
            dx = x2 - x1
            dy = y2 - y1
            length = max((dx * dx + dy * dy) ** 0.5, 1.0)
            off_x = -dy / length * 4
            off_y = dx / length * 4
            bond_parts.append(
                f"<line x1='{x1 + off_x:.1f}' y1='{y1 + off_y:.1f}' x2='{x2 + off_x:.1f}' y2='{y2 + off_y:.1f}' stroke='{color}' stroke-width='1.5' stroke-linecap='round' />"
            )

    atom_parts = []
    element_colors = {
        "N": "#1d4ed8",
        "O": "#dc2626",
        "S": "#ca8a04",
        "P": "#9333ea",
        "F": "#15803d",
        "Cl": "#15803d",
        "Br": "#92400e",
        "I": "#6b21a8",
    }
    for atom in work_mol.GetAtoms():
        idx = atom.GetIdx()
        if idx not in coord_by_atom:
            continue
        x, y = xy(idx)
        symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        total_h = atom.GetTotalNumHs()
        h_label = ""
        if symbol != "C" and total_h > 0:
            h_label = "H" if total_h == 1 else f"H{total_h}"
        if charge > 0:
            charge_label = "+"
        elif charge < 0:
            charge_label = "-"
        else:
            charge_label = ""
        atom_label = f"{symbol}{h_label}{charge_label}"
        is_hetero = symbol not in {"C", "H"}
        is_highlight = idx in highlight_atoms
        if is_highlight or is_hetero:
            text_color = "#b91c1c" if is_highlight else element_colors.get(symbol, "#111827")
            atom_parts.append(
                f"<text x='{x:.1f}' y='{y + 5:.1f}' text-anchor='middle' font-family='Arial, sans-serif' font-size='18' font-weight='700' fill='{text_color}' stroke='#ffffff' stroke-width='4' paint-order='stroke'>{atom_label}</text>"
            )
        elif atom.GetDegree() == 0:
            atom_parts.append(
                f"<text x='{x:.1f}' y='{y + 5:.1f}' text-anchor='middle' font-family='Arial, sans-serif' font-size='16' font-weight='700' fill='#111827'>{atom_label}</text>"
            )

    ion_parts = []
    if counter_ions:
        label = "Ion pair: " + ", ".join(counter_ions)
        ion_parts.append(
            f"<line x1='{width - 242}' y1='34' x2='{width - 212}' y2='34' stroke='#6d28d9' stroke-width='1.8' stroke-dasharray='4 5' />"
        )
        ion_parts.append(
            f"<rect x='{width - 212}' y='15' width='190' height='38' rx='19' fill='#f5f3ff' stroke='#6d28d9' stroke-width='1.4' opacity='0.98' />"
        )
        ion_parts.append(
            f"<text x='{width - 117}' y='39' text-anchor='middle' font-family='Arial, sans-serif' font-size='13' font-weight='700' fill='#4c1d95'>{label}</text>"
        )

    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='100%' height='100%' role='img'>"
        f"<rect width='100%' height='100%' rx='12' fill='#ffffff' />"
        f"<rect x='10' y='10' width='{width - 20}' height='{height - 20}' rx='18' fill='none' stroke='#d1d5db' stroke-width='1.2' />"
        + "".join(alert_halo_parts)
        + "".join(bond_parts)
        + "".join(atom_parts)
        + "".join(ion_parts)
        + "</svg>"
    )

def render_molecule_svg(mol, highlight_atoms=None, width=620, height=420):
    if mol is None:
        return None
    return _simple_svg_for_molecule(mol, highlight_atoms=highlight_atoms, width=width, height=height)

def show_molecule(mol, highlight_atoms=None, width=620, height=420):
    svg = render_molecule_svg(mol, highlight_atoms=highlight_atoms, width=width, height=height)
    if not svg:
        st.warning("Structure rendering failed for this molecule.")
        return
    st.markdown(
        f"""
        <div style="width:100%; height:{height}px; display:flex; justify-content:center; align-items:center; background:#ffffff; border:1px solid #d1d5db; border-radius:12px; overflow:hidden;">
            {svg}
        </div>
        """,
        unsafe_allow_html=True,
    )

def collect_alert_atoms(result):
    atoms = set()
    for alert in result.get("expert_alerts", []):
        for match in alert.get("matched_atoms", []) or []:
            atoms.update(match)
    return sorted(atoms)

def toxic_alerts(result):
    rows = []
    seen = set()
    for alert in result.get("alerts", []) + result.get("expert_alerts", []):
        key = (alert.get("method"), alert.get("alert"), str(alert.get("matched_atoms")))
        if key in seen:
            continue
        seen.add(key)
        rows.append(alert)
    return rows

def genotoxic_alert_rows(result):
    rows = []
    for alert in toxic_alerts(result):
        if not alert.get("matched_atoms") and alert.get("method") != "Expert Rule-based":
            link = "Local historical dossier assertion; source verification required"
            functional_group = "Whole molecule / local dossier record"
            alert_type = "Unverified local record"
        else:
            link = alert.get("mechanism") or alert.get("reasoning") or "Structural alert linked to DNA-reactive mutagenicity review."
            functional_group = alert.get("alert")
            alert_type = "Structural alert"
        rows.append(
            {
                "Type": alert_type,
                "Functional group / structural region": functional_group,
                "Genotoxicity linkage": link,
                "Matched atoms": str(alert.get("matched_atoms") or "N/A"),
                "Evidence": alert.get("reference") or alert.get("evidence") or "N/A",
            }
        )
    return rows

def collect_evidence_alert_atoms(evidence_objects):
    atoms = set()
    for item in evidence_objects or []:
        details = item.get("details") or {}
        for match in details.get("matched_atoms", []) or []:
            atoms.update(match)
    return sorted(atoms)

def evidence_alert_rows(evidence_objects):
    rows = []
    for item in evidence_objects or []:
        if item.get("alert") or item.get("method"):
            details = item.get("details") or {}
            rows.append({
                "Method": item.get("method") or details.get("method"),
                "Alert": item.get("alert") or details.get("alert"),
                "Matched atoms": details.get("matched_atoms"),
                "Mechanism": item.get("mechanism") or details.get("mechanism"),
                "Reasoning": item.get("reasoning"),
            })
    return rows

def risk_badge_html(label, value):
    color = {
        "High": "#ef4444",
        "Medium": "#f59e0b",
        "Low": "#10b981",
        "Not started": "#64748b",
        "Review needed": "#f59e0b",
    }.get(value, "#64748b")
    return f"""
    <div class='glass-card' style='padding:1.2rem; min-height:142px;'>
        <div style='color:#94a3b8; font-size:0.85rem; font-weight:700;'>{label}</div>
        <div style='font-size:2rem; font-weight:900; color:{color}; margin-top:0.4rem;'>{value}</div>
    </div>
    """

def render_strategy_dashboard():
    be_result = current_be_result()
    snapshot = build_strategy_snapshot(
        results=st.session_state.results,
        degradants=st.session_state.degradants,
        be_result=be_result,
        role=user_role,
    )
    workflow = build_submission_workflow(
        results=st.session_state.results,
        degradants=st.session_state.degradants,
        be_result=be_result,
    )

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### Strategy Dashboard")
    meta_cols = st.columns(5)
    meta_cols[0].metric("Stage", development_stage)
    meta_cols[1].metric("Pathway", submission_path)
    meta_cols[2].metric("Dosage form", dosage_form)
    meta_cols[3].metric("Role view", user_role)
    meta_cols[4].metric("RLD / Reference", reference_product or "TBD")
    st.markdown("</div>", unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    r1.markdown(risk_badge_html("Overall Regulatory Risk", snapshot["overall_risk"]), unsafe_allow_html=True)
    r2.markdown(risk_badge_html("Structural alert screening", snapshot["toxicology_risk"]), unsafe_allow_html=True)
    r3.markdown(risk_badge_html("Impurity / Degradation", snapshot["degradation_risk"]), unsafe_allow_html=True)
    r4.markdown(risk_badge_html("Dissolution comparison", snapshot["bioequivalence_risk"]), unsafe_allow_html=True)

    st.info(f"**{user_role} focus**: {snapshot['role_focus']}  \n\n**Recommended next action**: {snapshot['role_next_action']}")

    dash_tab1, dash_tab2, dash_tab3 = st.tabs(["Workflow", "Module Actions", "Guidance Map"])
    with dash_tab1:
        st.dataframe(pd.DataFrame(workflow), use_container_width=True, hide_index=True)
    with dash_tab2:
        st.dataframe(pd.DataFrame(snapshot["module_actions"]), use_container_width=True, hide_index=True)
    with dash_tab3:
        st.dataframe(pd.DataFrame(FDA_GUIDANCE_MAP), use_container_width=True, hide_index=True)

def _be_project_context():
    return {
        "project_id": project_id,
        "reference_product": reference_product,
        "dosage_form": dosage_form,
        "submission_pathway": submission_path,
        "development_stage": development_stage,
        "analyst": analyst,
        "daily_dose_mg": daily_dose_mg,
        "compound_name": st.session_state.get("be_current_compound_name", st.session_state.get("assessment_compound_name", "")),
        "smiles": st.session_state.smiles,
    }


def _be_fingerprint(profile, runs, seed, backend):
    profile_data = {"columns": list(profile.columns), "rows": profile.to_dict("records")} if isinstance(profile, pd.DataFrame) else profile
    payload = {"schema": "dissolution-f2-simulation.v2", "profile": profile_data,
               "simulation_runs": runs, "seed": seed, "backend": backend,
               "project_context": _be_project_context()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def current_be_result():
    """Expose only a result bound to the current profile, controls and context."""
    result = st.session_state.get("be_result")
    if result is None:
        return None
    fingerprint = _be_fingerprint(st.session_state.be_profile,
        st.session_state.get("be_simulation_runs", 2000), st.session_state.get("be_seed", 1729),
        str(st.session_state.get("be_backend", "Python")).lower())
    if (getattr(result, "schema", None) != "dissolution-f2-simulation.v2"
            or st.session_state.get("be_request_fingerprint") != fingerprint):
        st.session_state.be_result = None
        st.session_state.be_export = None
        st.session_state.be_invalidation_notice = True
        return None
    return result


def render_bioequivalence_module():
    # The chemical-name input is rendered immediately before this module. Bind
    # dissolution provenance to its current label even without a SAR rerun.
    st.session_state.be_current_compound_name = input_name
    st.markdown("---")
    st.markdown("<div class='accent-text'>Dissolution profile comparison</div>", unsafe_allow_html=True)
    st.markdown("## Descriptive f2 and parametric simulation")
    st.caption("The starting profile is an illustrative synthetic example, not product evidence. "
               "Enter reference/test means, standard deviations and sample sizes from your own study.")
    st.warning("This summary-data tool calculates descriptive f2 and an independent-normal parametric simulation. "
               "It does not perform an empirical bootstrap or establish clinical bioequivalence, a biowaiver, or regulatory acceptance.")

    be_col1, be_col2 = st.columns([1.15, 0.85])
    summary_df, profile_error = None, None
    with be_col1:
        initial_profile = st.session_state.be_profile
        if not isinstance(initial_profile, pd.DataFrame):
            st.session_state.be_result = None
            st.session_state.be_export = None
            st.error("Profile validation failed: a table with the seven required dissolution columns is required.")
            return
        if initial_profile.columns.duplicated().any():
            st.session_state.be_result = None
            st.session_state.be_export = None
            st.error("Profile validation failed: duplicate column names are not supported. Correct the source table before editing.")
            return
        column_config = {
            "Time (min)": st.column_config.NumberColumn("Time (min)", min_value=0.0, step=5.0),
            "Reference Mean (%)": st.column_config.NumberColumn("Reference Mean (%)", min_value=0.0, max_value=100.0, step=1.0),
            "Reference SD": st.column_config.NumberColumn("Reference SD", min_value=0.0, step=0.1),
            "Reference n": st.column_config.NumberColumn("Reference n", min_value=2, step=1),
            "Test Mean (%)": st.column_config.NumberColumn("Test Mean (%)", min_value=0.0, max_value=100.0, step=1.0),
            "Test SD": st.column_config.NumberColumn("Test SD", min_value=0.0, step=0.1),
            "Test n": st.column_config.NumberColumn("Test n", min_value=2, step=1),
        }
        # A malformed imported/session table must remain correctable rather than
        # fail while a numeric editor is being configured for text cells.
        column_config = {column: config for column, config in column_config.items()
                         if column in initial_profile and pd.api.types.is_numeric_dtype(initial_profile[column])}
        editor_revision = hashlib.sha256(json.dumps(
            initial_profile.to_dict("records") if isinstance(initial_profile, pd.DataFrame) else initial_profile,
            sort_keys=True, default=str,
        ).encode()).hexdigest()[:16]
        edited_profile = st.data_editor(
            initial_profile,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            alt="Dissolution means, standard deviations and sample sizes at each sampling time",
            column_config=column_config,
            key="be_profile_editor:" + editor_revision,
        )
        st.session_state.be_profile = edited_profile
        try:
            summary_df = dissolution_profile_summary(edited_profile)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            profile_error = str(exc)
            st.error("Profile validation failed: " + profile_error)
            st.caption("Correct the supplied cells. Invalid rows and missing variability values are not dropped or replaced.")
        if summary_df is not None and not summary_df.empty:
            plot_df = summary_df.set_index("Time (min)")[["Reference Mean (%)", "Test Mean (%)"]]
            st.line_chart(plot_df, width="stretch", alt="Reference and test mean dissolution profiles at all supplied valid time points")
            st.caption("All supplied valid time points are retained. Time-point selection and dissolution method conditions require expert review.")

    with be_col2:
        st.markdown("#### Calculation controls")
        simulation_runs = st.number_input("Simulation runs", min_value=1, max_value=100000,
                                          value=2000, step=500, key="be_simulation_runs")
        seed = st.number_input("Simulation seed", min_value=0, max_value=2147483647,
                               value=1729, step=1, key="be_seed")
        backend_label = st.selectbox("Simulation backend", ["Python", "R"], key="be_backend")
        backend = backend_label.lower()
        st.caption("The selected backend is explicit. R requires Rscript; an R failure does not switch to Python.")
        fingerprint = _be_fingerprint(edited_profile, simulation_runs, seed, backend)
        result = current_be_result()
        if profile_error and result is not None:
            st.session_state.be_result = None
            st.session_state.be_export = None
            result = None
        if st.session_state.pop("be_invalidation_notice", False):
            st.info("Previous dissolution output was cleared because the calculation inputs, schema or project context changed.")
        if st.button("Calculate f2 and simulation", width="stretch", disabled=profile_error is not None):
            st.session_state.be_result = None
            st.session_state.be_export = None
            st.session_state.be_calculation_error = None
            try:
                result = calculate_f2(edited_profile, simulation_runs=int(simulation_runs), seed=int(seed), backend=backend)
                st.session_state.be_result = result
                st.session_state.be_request_fingerprint = fingerprint
            except (ValueError, TypeError, RuntimeError, OSError) as exc:
                result = None
                st.session_state.be_calculation_error = {"fingerprint": fingerprint, "message": str(exc)}
        calculation_error = st.session_state.get("be_calculation_error")
        if calculation_error and calculation_error.get("fingerprint") == fingerprint:
            st.error("Dissolution calculation blocked: " + calculation_error["message"])
        if result is not None:
            st.metric("Descriptive f2", result.f2)
            st.metric("Simulated f2, 2.5–97.5 percentiles",
                      f"{result.simulation_interval_low:.2f} – {result.simulation_interval_high:.2f}")
            st.metric("Simulated fraction with f2 ≥ 50", f"{result.simulated_fraction_f2_ge_50:.1%}")
            st.caption("The displayed fraction is conditional on the simulation assumptions. "
                       "The interval is a simulated percentile interval, not a confidence interval or a probability of bioequivalence.")
            st.info(f"Descriptive threshold comparison: f2 {'≥' if result.descriptive_threshold_met else '<'} 50 for the supplied mean profiles.")
            st.caption(f"Displayed f2 is rounded. The threshold comparison uses unrounded f2 = {result.f2_unrounded:.12g}.")
            st.warning("Applicability: review needed. Clinical bioequivalence and regulatory acceptance are not established.")
            for issue in result.applicability_issues:
                st.warning(issue)
            st.caption(f"Backend: {result.backend} {result.backend_version} · Seed: {result.seed} · Simulation runs: {result.simulation_runs}")
            st.caption(result.method_note)
            with st.expander("Simulation assumptions and limitations", expanded=True):
                for assumption in result.simulation_assumptions:
                    st.write(assumption)
                for limitation in result.limitations:
                    st.caption(limitation)
                st.write(f"**Variability review policy**: {result.variability_policy}")
            payload = result.to_dict()
            export = {
                "schema": "toxiscope-dissolution-review.v2",
                "method": "Descriptive f2 and independent-normal parametric simulation",
                "project_context": _be_project_context(),
                "profile_columns": list(edited_profile.columns),
                "profile": edited_profile.to_dict("records"),
                "calculation_inputs": {"simulation_runs": int(simulation_runs), "seed": int(seed), "backend": backend},
                "calculation_fingerprint": fingerprint,
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "backend": result.backend, "backend_version": result.backend_version},
                "result": payload,
                "simulation_assumptions": result.simulation_assumptions,
                "limitations": result.limitations,
                "clinical_be_established": False,
                "regulatory_acceptance": "Not established",
            }
            st.session_state.be_export = export
            st.download_button("Download dissolution calculation + provenance",
                json.dumps(export, ensure_ascii=False, indent=2, default=str),
                "toxiscope-dissolution-review.json", "application/json", key="be_download:" + fingerprint)
        else:
            st.info("Validate the profile and calculate descriptive f2 with the selected simulation settings.")

    if summary_df is not None and not summary_df.empty:
        with st.expander("Supplied dissolution summary statistics"):
            st.dataframe(summary_df, width="stretch", hide_index=True,
                         alt="Descriptive dissolution statistics for the supplied profile")
    with st.expander("Background guidance and method sources"):
        st.dataframe(pd.DataFrame(FDA_BE_GUIDANCE_SOURCES), width="stretch", hide_index=True,
                     alt="Background sources for expert review of dissolution method and study context")
        st.caption("These links provide background for expert review. Listing them does not establish that this calculation complies with a guidance or that a product is bioequivalent.")

if workspace == "Toxicity Evidence Review":
    from ntr_panel import render_ntr_panel
    name = st.text_input("Compound name / project label", value=st.session_state.get("assessment_compound_name", ""), key="ntr_compound_name")
    st.caption("Select Strategy Dashboard to run structural screens. They remain separate from observed toxicity evidence.")
    context_package = st.session_state.evidence_package or {}
    if context_package and name != st.session_state.get("assessment_compound_name", ""):
        st.warning("Compound label differs from the last structural assessment. Structural context is excluded.")
        context_package = {}
    render_ntr_panel(project_id, name, context_package)
    st.stop()

st.markdown("<div class='accent-text'>Regulatory Development Strategy Platform</div>", unsafe_allow_html=True)
st.markdown("<h1 class='hero-title'>ToxiGuard-Platform</h1>", unsafe_allow_html=True)
st.caption("Ontology + ToxiGuard-AI + ToxiScope + Bioequivalence strategy are connected through one project-level decision workflow.")
if legacy_screening_reset:
    st.info("Previous structural assessment output was cleared because the screening method changed. Run the rule-based screen again to create a current package.")

render_strategy_dashboard()

col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Chemical Identification")
    
    input_name = st.text_input("Compound Name", placeholder="e.g. Brivaracetam, Aniline...")
    
    allow_external = st.checkbox("Allow this compound name to be sent to PubChem / NCI", value=False,
        key="external_consent:" + input_name)
    if st.button("🔍 Search SMILES from Name", use_container_width=True):
        if input_name:
            with st.spinner(f"Resolving '{input_name}'..."):
                res = get_smiles_from_name(input_name, allow_external=allow_external)
                if res:
                    st.session_state.smiles = res['smiles']
                    st.session_state.identity = res
                    st.success(f"Found via {res['source']}")
                    with st.spinner("Running regulatory assessment..."):
                        run_assessment(input_name, res["smiles"])
                else:
                    st.error("No local match. Enter SMILES manually, or explicitly allow external lookup for this compound name.")

    input_smiles = st.text_area("SMILES String", key="smiles", height=110)
    if input_smiles:
        st.caption("SMILES resolved. You can edit it manually and re-run the assessment.")

    if st.button("🚀 Run Regulatory Assessment", use_container_width=True):
        if input_smiles:
            with st.spinner("Analyzing toxicity and degradation..."):
                run_assessment(input_name, input_smiles)
        else:
            st.warning("Please provide a SMILES string.")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if st.session_state.results:
        res = st.session_state.results
        status_color = "badge-class1" if res.get("status") == "error" else "badge-review"
        st.markdown(f"""
        <div class='glass-card' style='text-align: center;'>
            <div class='accent-text'>Rule-based screening result</div>
            <h2 style='font-size: 2rem; margin-top: 1rem;'>{res.get('class', 'Not assigned')}</h2>
            <div class='badge {status_color}' style='display: inline-block; margin-top: 1rem;'>{res.get('status', 'review')}</div>
            <p style='margin-top: 1.5rem; color: #94a3b8;'>Expert review required. Package checks do not validate toxicology predictions.</p>
        </div>
        """, unsafe_allow_html=True)
        if res.get("status") == "error":
            st.error(res.get("message", "Screening could not be completed."))
        else:
            st.warning(res.get("note", "Limited structural alert screening requires expert review."))
        st.caption(res.get("screening_method", "Rule-based structural alert screening"))
        limitations = res.get("limitations") or [
            "Limited SMARTS rules; absence of a match does not establish safety or a negative mutagenicity result.",
            "No trained statistical model or independent model comparison is implemented.",
        ]
        st.caption(" ".join(limitations) if isinstance(limitations, list) else limitations)

render_bioequivalence_module()

if st.session_state.results and st.session_state.results.get("status") != "error":
    st.markdown("---")
    structure_smiles = st.session_state.smiles or st.session_state.results.get("canonical_smiles")
    profile = build_structure_profile(structure_smiles)
    alert_list = toxic_alerts(st.session_state.results)
    highlighted_atoms = collect_alert_atoms(st.session_state.results)

    st.markdown("<div class='accent-text'>Molecular Structure</div>", unsafe_allow_html=True)
    map_col1, map_col2 = st.columns([1.05, 1.25])
    with map_col1:
        if profile:
            show_molecule(profile["mol"], highlighted_atoms, width=620, height=420)
            st.caption("Wikipedia-style skeletal formula. Pale red halos mark genotoxic structural-alert regions when present.")
        else:
            st.warning("Structure parsing is not available for the submitted SMILES.")
    with map_col2:
        st.markdown("#### Genotoxicity-Linked Functional Groups")
        if alert_list:
            st.dataframe(pd.DataFrame(genotoxic_alert_rows(st.session_state.results)), use_container_width=True, hide_index=True)
            for idx, alert in enumerate(alert_list, 1):
                mechanism = alert.get("mechanism") or alert.get("reasoning")
                is_local_record = not alert.get("matched_atoms") and alert.get("method") != "Expert Rule-based"
                label = "Unverified local dossier record" if is_local_record else "Structural alert rule match"
                message = f"**{idx}. {alert.get('alert')}** ({label})  \n{mechanism}"
                st.warning(message)
                with st.expander(f"Details: {alert.get('alert')}"):
                    st.write(f"**Method**: {alert.get('method', 'N/A')}")
                    st.write("**Interpretation**: This screening item requires source verification and expert review; it does not establish mutagenicity or an ICH M7 class.")
                    st.write(f"**Matched atoms**: {alert.get('matched_atoms', 'N/A')}")
                    st.write(f"**Reference / Evidence**: {alert.get('reference') or alert.get('evidence') or 'N/A'}")
                    st.write(f"**Reasoning**: {alert.get('reasoning', 'N/A')}")
                    if alert.get("expert_comment"):
                        st.write(f"**Expert comment**: {alert.get('expert_comment')}")
        else:
            st.info("No structural alert was mapped by the implemented rules. Mutagenicity remains undetermined; no-alert screening does not establish safety.")
            st.caption("Element labels and ion-pair labels describe structure/physicochemical form only.")

        exp_data = get_experimental_detail(st.session_state.results.get("canonical_smiles") or st.session_state.smiles)
        if exp_data:
            with st.expander("Local historical toxicology dossier (unverified)"):
                st.warning("These are assertions stored in the local dossier. The cited studies and assay details have not been independently verified by this app.")
                st.write(exp_data.get("overall_conclusion", ""))
                st.dataframe(pd.DataFrame(exp_data.get("assay_data", [])), use_container_width=True, hide_index=True)

    ttc = st.session_state.results.get('ttc_info', {})
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1: st.metric("TTC reference (ug/day)", f"{ttc.get('limit_ug_day')} µg")
    with q_col2: st.metric("Reference concentration (ppm)", f"{ttc.get('limit_ppm')} ppm")
    with q_col3: st.metric("ICH M7 classification", st.session_state.results.get('class', 'Not assigned'))
    st.caption("These preserved TTC calculations are generic reference values, not a substance-specific acceptable intake or evidence of screening safety. Applicability and any compound-specific limits require expert review.")

    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧫 Structural Elucidation", "⚖️ Evidence Matrix", "🧬 Degradation Profile", "📚 USP/EP/DMF Ref", "📝 Regulatory Draft", "🧾 Harness Report"])

    with tab0:
        st.markdown("<div class='accent-text'>Encyclopedia-Style Structure View</div>", unsafe_allow_html=True)
        if profile:
            s_col1, s_col2 = st.columns([1, 1.25])
            with s_col1:
                show_molecule(profile["mol"], highlighted_atoms, width=520, height=360)
                st.caption("2D skeletal formula with genotoxic-alert regions highlighted only when mapped.")
            with s_col2:
                st.markdown("#### Identity & Physicochemical Profile")
                st.code(profile["canonical_smiles"], language="text")
                p1, p2, p3 = st.columns(3)
                p1.metric("Formula", profile["formula"])
                p2.metric("MW", profile["molecular_weight"])
                p3.metric("cLogP", profile["logp"])
                p4, p5, p6 = st.columns(3)
                p4.metric("TPSA", profile["tpsa"])
                p5.metric("HBD / HBA", f"{profile['hbd']} / {profile['hba']}")
                p6.metric("Rings", profile["ring_count"])
                st.caption(f"Heavy atoms: {profile['heavy_atoms']} | Rotatable bonds: {profile['rotatable_bonds']}")

            alert_rows = []
            for alert in st.session_state.results.get("expert_alerts", []):
                alert_rows.append({
                    "Method": alert.get("method"),
                    "Structural alert": alert.get("alert"),
                    "Matched atoms": alert.get("matched_atoms"),
                    "Mechanistic interpretation": alert.get("mechanism") or alert.get("reasoning"),
                    "Reference": alert.get("reference"),
                })
            if alert_rows:
                st.markdown("#### Alert Mapping")
                st.dataframe(pd.DataFrame(alert_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No alert was mapped by the implemented rules. This does not establish absence of DNA reactivity.")
            st.markdown("#### Regulatory Interpretation")
            st.info(st.session_state.results.get("structural_explanation", "No structural explanation available."))
        else:
            st.warning("Structure rendering is not available for the submitted SMILES.")
    
    with tab1:
        st.markdown("<div class='accent-text'>ICH M7 Evidence Object Matrix</div>", unsafe_allow_html=True)
        qsar = st.session_state.results.get("qsar_summary", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rule-based screen", qsar.get("expert_call", "Not evaluated"))
        c2.metric("Statistical model", "Not implemented")
        c3.metric("Independent model comparison", "Not available")
        c4.metric("Evidence Items", len(st.session_state.results.get("evidence_objects", [])))

        st.markdown("#### Structural Explanation")
        st.info(st.session_state.results.get("structural_explanation", "No structural explanation available."))
        st.caption(qsar.get("applicability_domain", "Applicability domain not documented."))

        evidence_rows = []
        for item in st.session_state.results.get("evidence_objects", []):
            evidence_rows.append({
                "Tier": item.get("source_tier_label"),
                "Type": item.get("evidence_type"),
                "Endpoint": item.get("endpoint"),
                "Result": item.get("result"),
                "Source": item.get("source_name"),
                "Confidence": item.get("confidence"),
                "Reasoning": item.get("reasoning"),
                "URL": item.get("source_url") or "",
            })

        if evidence_rows:
            st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No structured evidence objects were returned.")

        st.markdown("#### Structural alert screening detail")
        e_col1, e_col2 = st.columns(2)
        with e_col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("##### Implemented SMARTS rules")
            expert_alerts = st.session_state.results.get("expert_alerts", [])
            if expert_alerts:
                for alert in expert_alerts:
                    st.warning(f"**{alert.get('alert')}**")
                    st.write(alert.get("mechanism", ""))
                    st.caption(f"Matched atoms: {alert.get('matched_atoms', 'N/A')} | Ref: {alert.get('reference', 'N/A')}")
            else:
                st.info("No alert identified in the implemented rules. This is an undetermined mutagenicity result.")
            st.markdown("</div>", unsafe_allow_html=True)

        with e_col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("##### Statistical model availability")
            st.info("No trained statistical model is implemented. No mutagenicity probability, second independent prediction, or model concordance is available.")
            st.caption("This limited rule-based screen cannot establish an ICH M7 classification on its own.")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='accent-text'>Compendial / FDA-Anchored Degradation Evidence</div>", unsafe_allow_html=True)
        st.caption("Known pharmacopeial related substances and stability-literature degradation products are shown before purely predicted RDKit degradation products.")
        if st.session_state.degradants:
            st.markdown("#### Degradation Structure Gallery")
            gallery_cols = st.columns(2)
            for idx, d in enumerate(st.session_state.degradants):
                d_evidence = d.get("evidence_objects", [])
                d_profile = build_structure_profile(d.get("smiles"))
                d_alert_atoms = collect_evidence_alert_atoms(d_evidence)
                with gallery_cols[idx % 2]:
                    st.markdown(f"**{d.get('name', 'Degradation product')}**")
                    if d_profile:
                        show_molecule(d_profile["mol"], d_alert_atoms, width=420, height=280)
                        st.caption("Pale red halo marks genotoxic alert region when mapped.")
                    else:
                        st.info("Structure not loaded. Add qualified SMILES/InChI to render this degradation product.")
                    st.write(f"**ICH M7**: {d.get('class')} ({d.get('status')})")
                    st.write(f"**Origin**: {d.get('condition')}")
                    if d.get("source_url"):
                        st.markdown(f"[Source]({d['source_url']})")
            st.markdown("---")

            for d in st.session_state.degradants:
                with st.expander(f"🚩 [{d['pathway']}] {d.get('name', 'Product Identification')}"):
                    d_evidence = d.get("evidence_objects", [])
                    d_profile = build_structure_profile(d.get("smiles"))
                    d_alert_atoms = collect_evidence_alert_atoms(d_evidence)
                    d_col1, d_col2 = st.columns([1, 1])
                    with d_col1:
                        if d_profile:
                            show_molecule(d_profile["mol"], d_alert_atoms, width=440, height=300)
                            st.caption("Degradation/impurity structure with matched structural alert atoms highlighted when available.")
                        else:
                            st.info("No drawable structure is loaded for this degradation product.")
                        st.write(f"**SMILES**: `{d.get('smiles')}`")
                        st.write(f"**ICH M7 Result**: {d.get('class')} ({d.get('status')})")
                        st.write(f"**Condition / Origin**: {d.get('condition')}")
                        st.write(f"**Review priority**: {d.get('risk')}")
                        st.caption(d.get("risk_basis") or "Qualitative degradation-pathway priority; it does not establish mutagenicity or an ICH M7 class.")
                    with d_col2:
                        if d.get("source_name"):
                            st.write(f"**Evidence Source**: {d.get('source_name')}")
                        if d.get("evidence_source_category"):
                            st.write(f"**Source Category**: {d.get('evidence_source_category')}")
                        st.write(f"**Rationale**: {d.get('rationale')}")
                        st.write(f"**Regulatory Significance**: {d.get('significance', 'N/A')}")
                        if d.get("source_url"):
                            st.markdown(f"[Source reference]({d['source_url']})")
                    st.markdown("**Rule-based screening interpretation**")
                    st.info(d.get("structural_explanation") or "No structural explanation available.")
                    d_alert_rows = evidence_alert_rows(d_evidence)
                    if d_alert_rows:
                        st.markdown("**Degradation Product Toxicophore Detail**")
                        st.dataframe(pd.DataFrame(d_alert_rows), use_container_width=True, hide_index=True)
                    if d_evidence:
                        st.dataframe(pd.DataFrame([{
                            "Tier": e.get("source_tier_label"),
                            "Type": e.get("evidence_type"),
                            "Result": e.get("result"),
                            "Source": e.get("source_name"),
                            "URL": e.get("source_url") or "",
                            "Reasoning": e.get("reasoning"),
                        } for e in d_evidence]), use_container_width=True, hide_index=True)
        else:
            st.info("No compound-specific degradation products are loaded or predicted yet. Use the source map below to complete targeted FDA/USP/literature review.")
            source_rows = (st.session_state.evidence_package or {}).get("regulatory_source_map", [])
            if source_rows:
                st.markdown("#### Source Review Queue")
                st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("<div class='accent-text'>Known Impurity / Degradation Product Search</div>", unsafe_allow_html=True)
        pharma_info = (st.session_state.evidence_package or {}).get("regulatory_profile") or get_pharmacopeia_info(input_name)
        st.markdown("#### Parent Compound Reference")
        if pharma_info.get("profile_type") == "curated":
            st.success("Curated pharmacopeial/DMF-style profile loaded.")
        else:
            st.info("No curated compound-specific impurity profile is loaded yet. The app generated a source-review map for this compound.")
        st.write(f"**Monograph / Reference Context**: {pharma_info.get('monograph_ref')}")
        st.write(f"**DMF / Control Summary**: {pharma_info.get('dmf_summary')}")
        source_rows = pharma_info.get("regulatory_sources", [])
        if source_rows:
            st.markdown("#### FDA / Compendial Source Map")
            st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)

        matches = st.session_state.known_impurities or match_known_impurities(input_name, st.session_state.smiles)
        if matches:
            st.caption("Class values below are local reference assertions requiring source verification and expert review; they are not classifications established by this screening run.")
            match_rows = []
            for match in matches:
                match_rows.append({
                    "Parent": match.get("parent", input_name),
                    "Impurity ID": match.get("id"),
                    "Name": match.get("name"),
                    "Origin": match.get("origin"),
                    "Alert": match.get("alert"),
                    "Local class assertion (unverified)": f"Class {match.get('class')}",
                    "CAS": match.get("cas") or "",
                    "Source": match.get("source_name"),
                    "URL": match.get("source_url") or "",
                    "Issue": match.get("issue"),
                })
            st.dataframe(pd.DataFrame(match_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No USP/EP/DMF-style impurity match found in the current curated library.")

    with tab4:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### Regulatory draft for expert review")
        narrative = generate_regulatory_narrative(st.session_state.results, input_name or "the submitted compound")
        st.text_area("Narrative Preview", value=narrative, height=360)
        st.button("📥 Download PDF Report")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab5:
        package = st.session_state.evidence_package or {}
        report = package.get("worker_report", {})
        manifest = package.get("harness_manifest", {})
        validation = report.get("validation", {})

        st.markdown("<div class='accent-text'>worker-report.v1</div>", unsafe_allow_html=True)
        h_col1, h_col2, h_col3, h_col4 = st.columns(4)
        h_col1.metric("Harness", manifest.get("status", "N/A"))
        h_col2.metric("Policy", manifest.get("policy", "N/A"))
        h_col3.metric("Package checks passed", validation.get("passed", 0))
        h_col4.metric("Checks needing review", validation.get("review", 0))
        st.caption("These checks inspect package completeness and traceability. They do not establish model validity, toxicological safety, or regulatory acceptance.")

        gates = validation.get("gates") or package.get("validation_gates") or st.session_state.results.get("validation_gates", [])
        if gates:
            st.markdown("#### Package checks")
            st.dataframe(pd.DataFrame(gates), use_container_width=True, hide_index=True)

        if report:
            st.markdown("#### Harness Summary")
            st.json(report, expanded=False)
        else:
            st.warning("Harness report is not available for this run.")

else:
    st.image("./hero.png", use_container_width=True)
    st.markdown("<div style='text-align: center; color: #64748b;'>Precision regulatory decision support platform.</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption(f"ToxiScope AI v2.0 | Harness: {project_id} | Security Level: R01-R13")
