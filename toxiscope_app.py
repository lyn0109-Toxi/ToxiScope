import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import hashlib
import json
import platform
from presentation.be_visuals import render_dissolution_insights
from presentation.molecule import render_molecule_svg as _standard_molecule_svg
from presentation.workbench import install_brand, brand_header, analysis_flow, empty_signal, render_rule_chart, evidence_mix_chart, render_summary

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from core.regulatory import (
        build_harnessed_evidence_package,
        generate_regulatory_narrative,
        get_smiles_from_name,
        get_experimental_detail,
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
        build_submission_workflow,
    )
except ImportError as e:
    st.error(f"Module Import Error: {e}")
    st.stop()

# Keep structure parsing available if the standard drawing backend is missing.
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError as e:
    RDKIT_IMPORT_ERROR = str(e)
    RDKIT_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="ToxiGuard | Visual evidence workbench",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- State Management ---
if "smiles" not in st.session_state:
    st.session_state.smiles = st.session_state.get("workbench_smiles", "")
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

if "pending_structure" in st.session_state:
    st.session_state.smiles = st.session_state.pop("pending_structure")
    st.session_state.workbench_smiles = st.session_state.smiles

# --- UI Layout ---
with st.sidebar:
    st.markdown("### ToxiGuard")
    st.caption("VISUAL EVIDENCE WORKBENCH")
    workspace = st.radio("Workspace", ["Strategy Dashboard", "Toxicity Evidence Review"], key="workspace", format_func=lambda value: {"Strategy Dashboard": "분석 워크벤치", "Toxicity Evidence Review": "근거 검토"}[value])
    with st.expander("프로젝트 설정", expanded=False):
        project_id = st.text_input("Project ID", value="TXS-2026-001")
        analyst = st.text_input("Expert Analyst", value="Lee Young-nam")
        development_stage = st.selectbox("Development Stage", ["Preformulation", "Preclinical", "IND-enabling", "Phase 1", "Phase 2/3", "ANDA / Generic", "NDA / 505(b)(2)", "Post-approval change"], index=5)
        submission_path = st.selectbox("Submission Pathway", ["ANDA", "NDA", "505(b)(2)", "IND", "DMF", "IMPD / CTA", "Post-approval variation"], index=0)
        user_role = st.selectbox("Platform View", list(ROLE_GUIDANCE.keys()), index=0)
        dosage_form = st.selectbox("Dosage Form", ["Immediate-release tablet/capsule", "Modified-release oral solid", "Oral solution", "Injectable", "Semisolid", "Other"], index=0)
        reference_product = st.text_input("Reference Product / RLD", value="")
        daily_dose_mg = st.number_input("Daily Dose (mg/day)", min_value=0.001, value=10.0, step=1.0)
    st.caption("규칙·출처·시뮬레이션을 구분하는 검토 도구")

def run_assessment(compound_name, smiles_value):
    package = build_harnessed_evidence_package(
        compound_name,
        smiles_value,
        daily_dose_mg=daily_dose_mg,
        project_id=project_id,
        analyst=analyst,
    )
    st.session_state.assessment_compound_name = compound_name
    st.session_state.assessment_project_id = project_id
    st.session_state.evidence_package = package
    st.session_state.results = package["assessment"]
    st.session_state.degradants = package["degradation_products"]
    st.session_state.known_impurities = package["known_impurity_matches"]

def assessment_project_id():
    return st.session_state.get("assessment_project_id") or (st.session_state.evidence_package or {}).get("worker_report", {}).get("harness", {}).get("project_id")

def build_structure_profile(smiles_value):
    if not RDKIT_AVAILABLE or not smiles_value:
        return None
    mol = Chem.MolFromSmiles(smiles_value)
    if not mol:
        return None
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    # Screening atom indices come from reparsing canonical SMILES. Use that
    # same atom order for parent and degradation-product highlight maps.
    mol = Chem.MolFromSmiles(canonical_smiles)
    return {
        "mol": mol,
        "canonical_smiles": canonical_smiles,
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

def render_molecule_svg(mol, highlight_atoms=None, width=620, height=420):
    return _standard_molecule_svg(mol, highlight_atoms=highlight_atoms, width=width, height=height)

def show_molecule(mol, highlight_atoms=None, width=620, height=420):
    svg = render_molecule_svg(mol, highlight_atoms=highlight_atoms, width=width, height=height)
    if not svg:
        st.warning("분자 그림을 표시할 수 없습니다. 구조 정보를 확인해 주세요.")
        if RDKIT_AVAILABLE and mol is not None:
            st.code(Chem.MolToSmiles(mol, isomericSmiles=True), language="text")
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
    saved = st.session_state.get("be_calculation_settings", {"runs": 2000, "seed": 1729, "backend": "Python"})
    fingerprint = _be_fingerprint(st.session_state.be_profile,
        st.session_state.get("be_simulation_runs", saved["runs"]), st.session_state.get("be_seed", saved["seed"]),
        str(st.session_state.get("be_backend", saved["backend"])).lower())
    if (getattr(result, "schema", None) != "dissolution-f2-simulation.v2"
            or st.session_state.get("be_request_fingerprint") != fingerprint):
        st.session_state.be_result = None
        st.session_state.be_export = None
        st.session_state.be_invalidation_notice = True
        return None
    return result


def render_bioequivalence_module():
    st.session_state.be_current_compound_name = input_name
    st.subheader("용출의 차이를 읽다")
    st.caption("평균 곡선 → 시점별 차이 → 모형 시나리오. 기본 표는 설명용 합성 예시입니다.")
    initial_profile = st.session_state.be_profile
    if not isinstance(initial_profile, pd.DataFrame) or initial_profile.columns.duplicated().any():
        st.session_state.be_result = None
        st.session_state.be_export = None
        st.error("Profile validation failed: a table with seven unique dissolution columns is required.")
        return
    with st.expander("입력 데이터 · 평균 / 표본 SD / 용출 단위 수", expanded=False):
        st.caption("SD는 백분율 포인트 단위의 표본 표준편차입니다. n은 임상 대상자가 아닌 용출 단위 수입니다.")
        configs = {
            "Time (min)": st.column_config.NumberColumn("Time (min)", min_value=0.0, step=5.0),
            "Reference Mean (%)": st.column_config.NumberColumn("Reference Mean (%)", min_value=0.0, max_value=100.0, step=1.0),
            "Reference SD": st.column_config.NumberColumn("Reference SD", min_value=0.0, step=0.1),
            "Reference n": st.column_config.NumberColumn("Reference n", min_value=2, step=1),
            "Test Mean (%)": st.column_config.NumberColumn("Test Mean (%)", min_value=0.0, max_value=100.0, step=1.0),
            "Test SD": st.column_config.NumberColumn("Test SD", min_value=0.0, step=0.1),
            "Test n": st.column_config.NumberColumn("Test n", min_value=2, step=1),
        }
        configs = {column: config for column, config in configs.items() if column in initial_profile and pd.api.types.is_numeric_dtype(initial_profile[column])}
        revision = hashlib.sha256(json.dumps(initial_profile.to_dict("records"), sort_keys=True, default=str).encode()).hexdigest()[:16]
        edited_profile = st.data_editor(initial_profile, num_rows="dynamic", width="stretch", hide_index=True,
            alt="시간별 기준품·시험품의 평균 용출률, 표본 표준편차와 단위 수", column_config=configs, key="be_profile_editor:" + revision)
    st.session_state.be_profile = edited_profile
    profile_error, summary_df = None, None
    try:
        summary_df = dissolution_profile_summary(edited_profile)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        profile_error = str(exc)
        st.error("Profile validation failed: " + profile_error)
        st.caption("누락·오류 행을 제거하거나 값을 대신 채우지 않습니다.")
    saved = st.session_state.get("be_calculation_settings", {"runs": 2000, "seed": 1729, "backend": "Python"})
    controls = st.columns([1, 1, 1, 1.5], vertical_alignment="bottom")
    with controls[0]:
        simulation_runs = st.number_input("Simulation runs", min_value=1, max_value=100000, value=saved["runs"], step=500, key="be_simulation_runs")
    with controls[1]:
        seed = st.number_input("Simulation seed", min_value=0, max_value=2147483647, value=saved["seed"], step=1, key="be_seed")
    with controls[2]:
        backend_label = st.selectbox("Simulation backend", ["Python", "R"], index=0 if saved["backend"] == "Python" else 1, key="be_backend")
    with controls[3]:
        calculate = st.button("Calculate f2 and simulation", width="stretch", type="primary", disabled=profile_error is not None)
    st.session_state.be_calculation_settings = {"runs": int(simulation_runs), "seed": int(seed), "backend": backend_label}
    backend = backend_label.lower()
    fingerprint = _be_fingerprint(edited_profile, simulation_runs, seed, backend)
    result = current_be_result()
    if profile_error:
        st.session_state.be_result = None
        st.session_state.be_export = None
        result = None
    if st.session_state.pop("be_invalidation_notice", False):
        st.info("입력·설정 변경 · 이전 용출 결과를 지웠습니다.")
    if calculate:
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
    error = st.session_state.get("be_calculation_error")
    if error and error.get("fingerprint") == fingerprint:
        st.error("Dissolution calculation blocked: " + error["message"])
    if summary_df is not None:
        if result:
            a, b = st.columns([1, 2])
            with a:
                st.metric("f2 · 기술 통계", result.f2, border=True)
            with b:
                st.metric("모형 시뮬레이션 2.5–97.5%", f"{result.simulation_interval_low:.2f} — {result.simulation_interval_high:.2f}", border=True)
            st.caption(f"기술적 기준: f2 {'≥' if result.descriptive_threshold_met else '<'} 50 · 반올림 전 {result.f2_unrounded:.12g} · 임상 동등성 미확정")
        render_dissolution_insights(edited_profile, result)
    st.caption("독립 정규모형 시뮬레이션 · 경험적 부트스트랩/EMA Expected-f2 절차/임상 BE 확률이 아닙니다.")
    if result:
        st.caption(f"{result.backend} · seed {result.seed} · {result.simulation_runs:,}회 · 검토 항목 {len(result.applicability_issues)}개")
        with st.expander("근거·방법·한계 및 계산 상세"):
            st.metric("조건부 시뮬레이션 빈도 · f2 ≥ 50", f"{result.simulated_fraction_f2_ge_50:.1%}")
            st.caption("모형 가정에 따른 반복 빈도입니다. 신뢰구간 또는 생물학적 동등성 확률이 아닙니다.")
            st.write(result.backend_version)
            st.write(result.method_note)
            st.write(result.variability_policy)
            st.markdown("**검토할 조건**")
            for issue in result.applicability_issues:
                st.write("• " + issue)
            st.markdown("**모형 가정**")
            for assumption in result.simulation_assumptions:
                st.write("• " + assumption)
            st.markdown("**해석의 한계**")
            for limitation in result.limitations:
                st.write("• " + limitation)
        export = {
            "schema": "toxiscope-dissolution-review.v2",
            "method": "Descriptive f2 and independent-normal parametric simulation",
            "project_context": _be_project_context(),
            "profile_columns": list(edited_profile.columns), "profile": edited_profile.to_dict("records"),
            "calculation_inputs": {"simulation_runs": int(simulation_runs), "seed": int(seed), "backend": backend},
            "calculation_fingerprint": fingerprint,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "backend": result.backend, "backend_version": result.backend_version},
            "result": result.to_dict(), "simulation_assumptions": result.simulation_assumptions,
            "limitations": result.limitations, "clinical_be_established": False, "regulatory_acceptance": "Not established",
        }
        st.session_state.be_export = export
        st.download_button("Download dissolution calculation + provenance", json.dumps(export, ensure_ascii=False, indent=2, default=str),
                           "toxiscope-dissolution-review.json", "application/json", key="be_download:" + fingerprint)
    with st.expander("입력 통계·참고 지침"):
        if summary_df is not None:
            st.dataframe(summary_df, width="stretch", hide_index=True, alt="입력 용출 자료의 기술 통계")
        st.dataframe(pd.DataFrame(FDA_BE_GUIDANCE_SOURCES), width="stretch", hide_index=True, alt="방법 검토를 위한 참고 자료; 지침 준수 또는 규제 승인을 의미하지 않음")


def render_sar_results(res):
    if res.get("status") == "error":
        st.error(res.get("message", "Screening could not be completed."))
        return
    current = build_structure_profile(st.session_state.smiles)
    if (
        not current
        or current["canonical_smiles"] != res.get("canonical_smiles")
        or assessment_project_id() != project_id
        or st.session_state.get("assessment_compound_name", "") != st.session_state.get("workbench_compound_name", "")
    ):
        st.warning("입력·프로젝트 변경 · 아래는 마지막 분석의 구조 스냅샷입니다. 현재 입력으로 재분석해 주세요.")
    profile = build_structure_profile(res.get("canonical_smiles"))
    alerts = toxic_alerts(res)
    a, b, c = st.columns(3)
    a.metric("규칙 일치", len(res.get("expert_alerts", [])), border=True)
    b.metric("매핑 원자", len(collect_alert_atoms(res)), border=True)
    c.metric("근거 기록", len(res.get("evidence_objects", [])), border=True)
    st.caption("SMARTS 규칙 기반 선별 · 변이원성 미확정 · ICH M7 분류는 전문가 검토 필요")
    map_col, signal_col = st.columns([1.1, 1])
    with map_col:
        with st.container(border=True):
            st.markdown("#### 구조 지도")
            if profile:
                show_molecule(profile["mol"], collect_alert_atoms(res), width=620, height=310)
            st.caption("강조 영역 = 일치 규칙의 원자 · 독성 확률 아님")
    with signal_col:
        with st.container(border=True):
            st.markdown("#### 일치 규칙의 분포")
            render_rule_chart(res)
            if res.get("historical_evidence"):
                st.warning("로컬 문헌 기록 · 출처 미검증")
    tabs = st.tabs(["근거의 구성", "구조·물성", "분해·불순물", "검토·내보내기"])
    with tabs[0]:
        evidence_mix_chart(res.get("evidence_objects", []))
        with st.expander("원문 근거와 메커니즘"):
            for alert in alerts:
                st.markdown(f"**{alert.get('alert')}**")
                st.write(alert.get("mechanism") or alert.get("reasoning"))
                st.caption(alert.get("reference") or alert.get("evidence") or "")
                st.write({"method": alert.get("method"), "matched_atoms": alert.get("matched_atoms"), "reasoning": alert.get("reasoning")})
            evidence_rows = [{key: json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else value for key, value in item.items()} for item in res.get("evidence_objects", [])]
            st.dataframe(pd.DataFrame(evidence_rows), hide_index=True, width="stretch", alt="원래 출처·검토 이유를 보존한 근거 기록")
        detail = get_experimental_detail(res.get("canonical_smiles"))
        if detail:
            with st.expander("로컬 독성 자료 · 미검증"):
                st.warning("Local historical toxicology dossier (unverified). The cited studies and assay details have not been independently verified.")
                st.write(detail.get("overall_conclusion", ""))
                st.dataframe(pd.DataFrame(detail.get("assay_data", [])), hide_index=True, width="stretch", alt="독립 검증되지 않은 로컬 시험 주장")
    with tabs[1]:
        if profile:
            fields = [("분자식", "formula"), ("분자량", "molecular_weight"), ("cLogP", "logp"), ("TPSA", "tpsa")]
            for column, (label, key) in zip(st.columns(4), fields):
                column.metric(label, profile[key])
            st.code(profile["canonical_smiles"], language="text")
        st.dataframe(pd.DataFrame(genotoxic_alert_rows(res)), hide_index=True, width="stretch", alt="규칙 일치 및 미검증 로컬 기록의 구분")
    with tabs[2]:
        products = st.session_state.degradants
        if products:
            st.caption("예측 경로와 로컬 참고 자료 · 독립적으로 확인된 생성물/ICH 분류가 아닙니다.")
            selection = st.selectbox("살펴볼 분해·불순물", range(len(products)), format_func=lambda n: products[n].get("name", products[n].get("pathway", str(n))), key="degradant_detail")
            item = products[selection]
            x, y = st.columns([1, 1.2])
            with x:
                item_profile = build_structure_profile(item.get("smiles"))
                if item_profile:
                    show_molecule(item_profile["mol"], collect_evidence_alert_atoms(item.get("evidence_objects", [])), width=480, height=260)
                else:
                    st.info("구조 자료 미등록 · 평가 미실행")
            with y:
                st.write(item.get("pathway"))
                st.caption(item.get("condition"))
                st.write(item.get("structural_explanation"))
                st.caption(item.get("class"))
                if item.get("source_url"):
                    st.link_button("출처 보기", item["source_url"])
            with st.expander("생성물 원문·근거·한계"):
                st.json(item, expanded=False)
        else:
            st.caption("등록·예측된 생성물 없음 · 생성물 부재를 입증하지 않습니다.")
        package = st.session_state.evidence_package or {}
        matches = package.get("known_impurity_matches", [])
        if matches:
            with st.expander("로컬 불순물 참조 · 분류 주장 미검증"):
                st.dataframe(pd.DataFrame(matches), hide_index=True, width="stretch", alt="출처 확인이 필요한 로컬 불순물·분류 주장")
        with st.expander("참조 자료 지도"):
            st.json(package.get("regulatory_profile", {}), expanded=False)
    with tabs[3]:
        package = st.session_state.evidence_package or {}
        narrative = generate_regulatory_narrative(res, st.session_state.get("assessment_compound_name") or "the submitted compound")
        st.caption("전문가 검토용 초안 · 제출·규제 승인 미확정")
        with st.expander("검토 보고서 초안"):
            st.text_area("Narrative Preview", value=narrative, height=280)
        x, y = st.columns(2)
        x.download_button("근거 패키지 JSON", json.dumps(package, ensure_ascii=False, indent=2, default=str), "toxiguard-evidence.json", "application/json")
        y.download_button("검토 초안 TXT", narrative, "toxiguard-review.txt", "text/plain")
        with st.expander("추적성·패키지 점검"):
            st.caption("Software workflow checks only; not scientific validation or regulatory acceptance.")
            st.json(package.get("worker_report", {}), expanded=False)
    with st.expander("방법·한계 · 선별 결과의 해석"):
        st.write(res.get("note", ""))
        for limitation in res.get("limitations", []):
            st.write("• " + limitation)
        st.metric("Statistical model", "Not implemented")
        st.write("No trained statistical model is implemented. No alert does not establish safety or a negative mutagenicity result.")
        st.caption(res.get("qsar_summary", {}).get("applicability_domain", ""))
        st.json(res.get("qsar_summary", {}), expanded=False)
        ttc = res.get("ttc_info", {})
        st.write({"generic_TTC_ug_day": ttc.get("limit_ug_day"), "reference_ppm": ttc.get("limit_ppm")})
        st.caption("Generic reference calculation only; not a substance-specific acceptable intake or evidence of safety.")


install_brand()
if workspace == "Toxicity Evidence Review":
    from ntr_panel import render_ntr_panel
    brand_header("근거의 빈틈을 찾다.", "출처, 관찰, 예측을 구분하고 검토할 차이를 좁힙니다.")
    name = st.text_input("Compound name / project label", value=st.session_state.get("assessment_compound_name", ""), key="ntr_compound_name")
    context_package = st.session_state.evidence_package or {}
    if context_package and name != st.session_state.get("assessment_compound_name", ""):
        st.warning("Compound label differs from the last structural assessment. Structural context is excluded.")
        context_package = {}
    if context_package and assessment_project_id() != project_id:
        st.warning("Project differs from the structural assessment snapshot. Structural context is excluded.")
        context_package = {}
    render_ntr_panel(project_id, name, context_package)
    st.stop()

brand_header()
if legacy_screening_reset:
    st.info("Previous structural assessment output was cleared because the screening method changed. Run the rule-based screen again to create a current package.")
input_name = st.text_input("Compound Name", value=st.session_state.get("workbench_compound_name", ""), placeholder="화합물 또는 프로젝트의 이름", key="compound_name_input")
st.session_state.workbench_compound_name = input_name
st.session_state.be_current_compound_name = input_name
analysis_view = st.segmented_control("분석 보기", ["구조 신호", "용출 비교", "프로젝트 요약"], default="구조 신호", key="analysis_view") or "구조 신호"
analysis_flow({"구조 신호": 2, "용출 비교": 3, "프로젝트 요약": 4}[analysis_view])
if analysis_view == "용출 비교":
    render_bioequivalence_module()
    st.stop()
if analysis_view == "프로젝트 요약":
    ntr = st.session_state.get("ntr_project_review")
    if not ntr or ntr.get("scope", {}).get("project_id") != project_id or ntr.get("scope", {}).get("compound_name") != input_name:
        ntr = None
    results = st.session_state.results
    if assessment_project_id() != project_id:
        results = None
    if st.session_state.get("assessment_compound_name", "") != input_name:
        results = None
    render_summary(results, st.session_state.degradants, current_be_result(), ntr, project_id)
    with st.expander("분석 흐름·참고 자료"):
        st.dataframe(pd.DataFrame(build_submission_workflow(results, st.session_state.degradants, current_be_result())), hide_index=True, width="stretch", alt="원래 분석 워크플로와 남은 검토 상태")
        st.dataframe(pd.DataFrame(FDA_GUIDANCE_MAP), hide_index=True, width="stretch", alt="방법 검토를 위한 참고 맥락; 승인 또는 적합성 판정이 아님")
    st.stop()

left, right = st.columns([1.15, 1])
has_screening = bool(st.session_state.results and st.session_state.results.get("status") != "error")
with left:
    input_panel = st.expander("구조 입력·다시 분석", expanded=False) if has_screening else st.container(border=True)
    with input_panel:
        if not has_screening:
            st.markdown("### 구조에서 시작")
        input_smiles = st.text_area("SMILES String", key="smiles", height=92, placeholder="분자 구조의 SMILES를 입력하세요.")
        st.session_state.workbench_smiles = input_smiles
        if st.button("구조 신호 분석", type="primary", width="stretch"):
            if input_smiles:
                with st.spinner("규칙 일치와 근거를 연결하는 중..."):
                    run_assessment(input_name, input_smiles)
                st.rerun()
            else:
                st.warning("Please provide a SMILES string.")
        with st.expander("이름으로 구조 찾기"):
            allow_external = st.checkbox("Allow this compound name to be sent to PubChem / NCI", value=False, key="external_consent:" + input_name)
            if st.button("🔍 Search SMILES from Name", width="stretch"):
                if input_name:
                    found = get_smiles_from_name(input_name, allow_external=allow_external)
                    if found:
                        st.session_state.pending_structure = found["smiles"]
                        st.session_state.workbench_smiles = found["smiles"]
                        st.session_state.identity = found
                        run_assessment(input_name, found["smiles"])
                        st.rerun()
                    else:
                        st.error("No local match. Enter SMILES manually, or explicitly allow external lookup for this compound name.")
with right:
    if st.session_state.results:
        res = st.session_state.results
        status = "입력 오류" if res.get("status") == "error" else "일치 규칙 확인" if res.get("expert_alerts") else "일치 규칙 없음"
        st.markdown(f"**규칙 결과 · {status}**")
        st.caption("검토 지원 · 독성/규제 판정 아님")
        snapshot_project = assessment_project_id() or "프로젝트 미연결"
        st.caption(f"분석 스냅샷: {snapshot_project} / {st.session_state.get('assessment_compound_name') or '제출 구조'}")
        if res.get("status") == "error":
            st.error(res.get("message", "Invalid SMILES"))
        elif res.get("historical_evidence"):
            st.warning("로컬 기록 있음 · 독립 검증 필요")
    else:
        empty_signal()
        st.caption("규칙 일치와 실험 사실은 구분해 표시합니다.")
if st.session_state.results and st.session_state.results.get("status") != "error":
    render_sar_results(st.session_state.results)
st.caption(f"ToxiGuard · {project_id} · 출처와 검토 상태를 보존하는 분석 워크벤치")
