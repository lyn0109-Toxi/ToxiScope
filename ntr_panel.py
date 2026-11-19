"""Local, source-linked NTR review; legacy assessment stays optional and separate."""
import hashlib
import json
from typing import Any

import pandas as pd
import streamlit as st

from nora.evidence import extract_document
from nora.assertions import (
    extract_assertions_from_documents,
    assertion_table_rows,
    assertions_from_table_rows,
    REVIEW_STATUS_OPTIONS,
)
from nora.evidence_review import build_project_review
from presentation.ntr_visuals import render_ntr_charts
from ntr_bridge import (
    empty_input,
    context_key,
    context_snapshot,
    reviewed_assessment,
    parse_review_input,
    build_review_export,
)


_SCOPE_KEYS = ("subject", "endpoint", "system", "exposure", "timepoint", "study_id")
_EVIDENCE_KINDS = ("unspecified", "observation", "prediction", "context")


def _workspace_key(project_id, name):
    return hashlib.sha256(json.dumps([project_id, name], ensure_ascii=False).encode()).hexdigest()


def _save_workspace(workspace_key):
    st.session_state.ntr_workspaces[workspace_key] = {
        "docs": st.session_state.ntr_docs,
        "assertions": st.session_state.ntr_assertions,
        "generation": st.session_state.ntr_generation,
        "manual_raw": st.session_state.get("ntr_manual_raw"),
        "evaluation": st.session_state.get("ntr_evaluation_state"),
    }


def _preserve_unbound_legacy(state, previous_context_key):
    """Retain legacy evidence without attaching it to an unproven project."""
    snapshot = {
        "previous_context_key": previous_context_key,
        "documents": [document.to_dict() for document in state.get("docs", [])],
        "assertions": [assertion.to_dict() for assertion in state.get("assertions", [])],
        "raw_manual_input": state.get("manual_raw"),
        "legacy_evaluation": state.get("evaluation"),
    }
    quarantine = st.session_state.setdefault("ntr_unbound_legacy_workspace", {
        "schema": "toxiscope-ntr-unbound-legacy.v1",
        "status": "Unbound legacy evidence; project assignment requires review",
        "workspaces": [],
        "limitations": ["The previous context could not be matched to the active project and compound. "
                        "These sources were retained without assigning them to the active project or applying their assertions."],
    })
    quarantine["workspaces"].append(snapshot)


def _activate_workspace(project_id, name, package):
    """Keep evidence per project/compound, independently of SAR package changes."""
    key = _workspace_key(project_id, name)
    workspaces = st.session_state.setdefault("ntr_workspaces", {})
    previous = st.session_state.get("ntr_active_workspace")
    if previous != key:
        if previous is not None:
            _save_workspace(previous)
        legacy_state = None
        if previous is None:
            candidate = {
                "docs": st.session_state.get("ntr_docs", []),
                "assertions": st.session_state.get("ntr_assertions", []),
                "generation": st.session_state.get("ntr_generation", 0) + 1,
                "manual_raw": st.session_state.get("ntr_manual_raw"),
                "evaluation": st.session_state.get("ntr_evaluation_state"),
            }
            if any(candidate.get(field) for field in ("docs", "assertions", "manual_raw", "evaluation")):
                previous_context_key = st.session_state.get("ntr_context_key")
                if previous_context_key == context_key(project_id, name, package) and key not in workspaces:
                    legacy_state = candidate
                else:
                    _preserve_unbound_legacy(candidate, previous_context_key)
        state = workspaces.get(key)
        if state is None:
            state = legacy_state or {
                "docs": [],
                "assertions": [],
                "generation": st.session_state.get("ntr_generation", 0) + 1,
            }
        st.session_state.ntr_docs = state.get("docs", [])
        st.session_state.ntr_assertions = state.get("assertions", [])
        st.session_state.ntr_generation = state.get("generation", 1)
        st.session_state.ntr_manual_raw = state.get("manual_raw")
        st.session_state.ntr_evaluation_state = state.get("evaluation")
        st.session_state.ntr_active_workspace = key
    return key


def _fingerprint(raw, assertions, documents, structural_context):
    snapshot = {
        "manual_input": raw,
        "assertions": [a.to_dict() for a in assertions],
        "documents": [d.to_dict() for d in documents],
        "structural_context": structural_context,
    }
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=str).encode()).hexdigest()


def _table_rows(rows):
    return [
        {key: json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (list, dict)) else value
         for key, value in row.items()}
        for row in rows
    ]


def _render_project_review(report, project_id, name, manual_input, assertions, docs, package, key_prefix):
    context = manual_input.get("context_of_use", {}) if isinstance(manual_input, dict) else {}
    endpoint = context.get("target_endpoint") if isinstance(context, dict) else None
    endpoint = endpoint if isinstance(endpoint, str) and endpoint.strip() else "미지정"
    st.caption(f"{project_id or '프로젝트 미지정'} / {name or '물질 미지정'} · 요청 범위: {endpoint}")
    summary = report.get("summary", {})
    with st.container(horizontal=True, wrap=True, gap="small"):
        st.metric("자료", len(docs), width=155, border=True,
                  help="업로드 문서 수. 독립 연구 수가 아닙니다.")
        st.metric("주장", len(assertions), width=155, border=True,
                  help="제안·승인·수정·거절 상태를 모두 포함합니다.")
        st.metric("검토된 주장", sum(a.review_status in {"승인", "수정"} for a in assertions),
                  width=155, border=True, help="사용자가 승인·수정한 주장 수. 과학적 검증을 뜻하지 않습니다.")
        st.metric("값 차이 후보", summary.get("possible_conflicts", 0), width=155, border=True,
                  help="동일 필드에서 비교 대상으로 남은 서로 다른 단일 값의 후보 수. 전체 검토 항목 수나 확정된 과학적 충돌 수가 아닙니다.")
        st.metric("미확인 필드", summary.get("coverage_gaps", 0), width=155, border=True,
                  help="현재 추출·검토 자료의 연결이 확인되지 않은 필드 수.")
    render_ntr_charts(report, assertions, docs, key_prefix=key_prefix)

    findings = report.get("findings", [])
    if findings:
        finding_map = {finding["finding_id"]: finding for finding in findings}
        revision = hashlib.sha256(json.dumps(list(finding_map), sort_keys=True).encode()).hexdigest()[:12]
        kind_labels = {"possible_conflict": "값 차이 후보", "source_issue": "출처 확인",
            "scope_review": "범위 확인", "scope_difference": "범위 차이",
            "project_scope_mismatch": "프로젝트 범위", "coverage_gap": "필드 근거",
            "extraction_limit": "추출 한계", "out_of_scope": "지원 범위 밖",
            "input_issue": "입력 확인", "possible_value_difference": "목록 차이"}
        selected = st.selectbox("상세 검토 항목", list(finding_map),
            format_func=lambda item: f"{kind_labels.get(finding_map[item].get('kind'), '검토 항목')} · {finding_map[item].get('field_path') or '문서'}",
            key="ntr_finding:" + key_prefix + ":" + revision)
        finding = finding_map[selected]
        with st.expander("선택 항목 · 근거와 범위", expanded=False):
            st.write(finding.get("detail", ""))
            st.caption(finding.get("uncertainty", "Expert review required."))
            st.json({"finding_id": finding.get("finding_id"), "scope": finding.get("scope", {}),
                     "provenance": finding.get("provenance", []),
                     "review_state": finding.get("review_state", "needs_expert_review")}, expanded=False)
    with st.expander("근거·방법·한계", expanded=False):
        st.caption("전문가 검토 필요 · 자동 추출의 빈칸은 음성 결과가 아닙니다.")
        st.caption("The endpoint is the requested review scope, not an observed toxicity finding. "
                   "Manual input values and template fields do not count as source-linked observations.")
        st.caption("Research prototype: deterministic evidence review, not toxicity prediction or regulatory acceptance. "
                   "Observation, prediction, reference context and unreviewed assertions remain distinct. "
                   "Empty or failed extraction does not establish absence of toxicity or evidence.")
        st.json(report.get("scope", {}), expanded=False)
        if report.get("coverage"):
            st.dataframe(pd.DataFrame(_table_rows(report["coverage"])), hide_index=True, width="stretch",
                         alt="Intake field coverage with assertion identifiers and source locations")
        if findings:
            st.dataframe(pd.DataFrame(_table_rows([
                {key: finding.get(key, "") for key in
                 ("kind", "field_path", "title", "detail", "uncertainty", "review_state")}
                for finding in findings
            ])), hide_index=True, width="stretch", alt="All source-linked findings requiring expert review")
        for limitation in report.get("limitations", []):
            st.caption(limitation)
        st.caption("ToxiScope results stay separate from customer observations. They do not populate NAM/AI outcomes.")
        st.json(context_snapshot(project_id, name, package), expanded=False)
        st.json(report, expanded=False)


def render_ntr_panel(project_id, name, package):
    st.subheader("NTR · 프로젝트 근거 검토")
    st.caption("검토 지원 · 독성/규제 판정 아님")
    download_slot = st.empty()
    key = _activate_workspace(project_id, name, package)
    st.session_state.ntr_context_key = context_key(project_id, name, package)
    quarantine = st.session_state.get("ntr_unbound_legacy_workspace")
    if quarantine:
        st.warning("Legacy evidence could not be matched to this project and compound. "
                   "It remains separate from the active review and is preserved for download; verify its project assignment before reuse.")
        with st.expander("Unbound legacy evidence and provenance"):
            st.json(quarantine, expanded=False)
            st.download_button("Download unbound legacy evidence + provenance",
                json.dumps(quarantine, ensure_ascii=False, indent=2, default=str),
                "toxiscope-ntr-unbound-legacy.json", "application/json", key="ntr_unbound_download")
    overview_slot = st.container()
    prefix = key + ":" + str(st.session_state.ntr_generation)
    docs = st.session_state.ntr_docs
    assertions = st.session_state.ntr_assertions
    with st.expander("자료 추가·출처 분류", expanded=False):
        st.caption("허용된 자료만 업로드하세요. 서버 내 처리 · 세션 내 보관 · 닫기 전 다운로드 · OCR 없음 · 추출 실패는 근거 부재가 아닙니다.")
        source_options = ["unclassified", "customer_evidence", "external_reference"]
        source_kind = st.selectbox("Source type", source_options, key="ntr_source:" + prefix)
        uploads = st.file_uploader("PDF, DOCX, CSV, XLSX, JSON or TXT", accept_multiple_files=True,
            type=["pdf", "docx", "csv", "xlsx", "json", "txt"], key="ntr_upload:" + prefix)
        if st.button("Extract locally", key="ntr_extract:" + prefix):
            for upload in uploads or []:
                if upload.size > 10 * 1024 * 1024:
                    st.error("Maximum document size: 10 MB.")
                    continue
                document = extract_document(upload.getvalue(), upload.name, upload.type)
                document.source_kind = source_kind
                if any(d.sha256 == document.sha256 for d in docs):
                    st.info("Duplicate file skipped: " + document.name)
                    continue
                docs.append(document)
                assertions.extend(extract_assertions_from_documents([document]))
            _save_workspace(key)
            st.rerun()
        for document in docs:
            document.source_kind = st.selectbox(document.name + " · source classification", source_options,
                index=source_options.index(document.source_kind) if document.source_kind in source_options else 0,
                key="kind:" + document.document_id)
            for warning in document.warnings:
                st.warning(warning)
            with st.expander(document.name + " · provenance"):
                st.json(document.to_dict(), expanded=False)
        if st.button("Clear NTR workspace", key="ntr_clear"):
            st.session_state.ntr_docs = []
            st.session_state.ntr_assertions = []
            st.session_state.ntr_generation += 1
            st.session_state.ntr_manual_raw = None
            st.session_state.ntr_evaluation_state = None
            _save_workspace(key)
            st.rerun()
    with st.expander("주장 검토 · 값/유형/범위", expanded=False):
        st.caption("Regex extraction proposes claims only. Approve/edit after reading the cited excerpt. "
                   "Choose evidence kind and scope explicitly; no claim is automatically an observation. "
                   "Changing a value, evidence kind or scope requires renewed approval. "
                   "External references and unclassified sources cannot become customer facts.")
        if assertions:
            rows = assertion_table_rows(assertions)
            # Streamlit retains editor deltas by widget key. Revision the key
            # when stored review rows change so a prior approval delta cannot
            # overwrite a semantic edit's reset to pending on the next rerun.
            editor_revision = hashlib.sha256(json.dumps(
                rows, sort_keys=True, ensure_ascii=False, default=str,
            ).encode()).hexdigest()[:16]
            editable = {"제안/수정 값", "검토 상태", "검토 메모", "Evidence kind"}
            editable.update("Scope " + field for field in _SCOPE_KEYS)
            column_config = {
                "검토 상태": st.column_config.SelectboxColumn(options=REVIEW_STATUS_OPTIONS),
                "Evidence kind": st.column_config.SelectboxColumn(options=list(_EVIDENCE_KINDS)),
            }
            column_config.update({"Scope " + field: st.column_config.TextColumn("Scope " + field)
                                  for field in _SCOPE_KEYS})
            edited = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch",
                alt="Review extracted assertions, evidence kinds and study scopes",
                disabled=[c for c in rows[0] if c not in editable], column_config=column_config,
                key="ntr_review:" + prefix + ":" + ":".join(a.assertion_id for a in assertions) + ":" + editor_revision)
            assertions = assertions_from_table_rows(edited.to_dict("records"), assertions)
            st.session_state.ntr_assertions = assertions
        else:
            st.info("No proposed assertions. You may enter reviewed data below; absent extraction is not a negative finding.")
    with st.expander("고급 입력 · 요청 범위와 수동 JSON", expanded=False):
        st.caption("Manual entries are user-supplied, not independently verified. Default AI/NAM streams are disabled. "
                   "Review every enabled field, source and use context; do not treat template values as observations.")
        initial_raw = st.session_state.get("ntr_manual_raw")
        if initial_raw is None:
            initial_raw = json.dumps(empty_input(name).to_dict(), ensure_ascii=False, indent=2)
        raw = st.text_area("Reviewed NTR input (JSON)", initial_raw, height=300, key="ntr_input:" + prefix)
        st.session_state.ntr_manual_raw = raw
    manual_input, input_error = None, None
    try:
        manual_input = json.loads(raw)
        if not isinstance(manual_input, dict):
            raise ValueError("Reviewed input must be a JSON object.")
    except (ValueError, TypeError) as exc:
        input_error = str(exc)
        manual_input = None
    if manual_input is not None:
        try:
            parse_review_input(manual_input, compound_name=name)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            input_error = str(exc)
    if input_error:
        st.warning("Manual input requires correction. Source-linked project review and export remain available: " + input_error)
    report = build_project_review(assertions, docs, project_id=project_id, compound_name=name,
                                  assessment_input=manual_input)
    st.session_state.ntr_project_review = report
    with overview_slot:
        _render_project_review(report, project_id, name, manual_input, assertions, docs, package, prefix)
    structural_context = context_snapshot(project_id, name, package)
    fingerprint = _fingerprint(raw, assertions, docs, structural_context)
    cached: dict[str, Any] | None = st.session_state.get("ntr_evaluation_state")
    if cached and cached.get("fingerprint") != fingerprint:
        st.session_state.ntr_evaluation_state = None
        cached = None
        st.info("Previous legacy assessment output was cleared because the inputs, evidence review, sources or structural context changed.")
    with st.expander("시나리오 · 기존 규칙 평가 (미보정)", expanded=False):
        st.caption("The original fixed-rule assessment is optional and supports early hepatotoxicity only; "
                   "other endpoints remain unevaluable (R0). R0–R5 are uncalibrated evidence-role labels, not safety scores. "
                   "Its fields labelled observations and interpretations are rule-engine output, not independently observed facts.")
        if st.button("Review evidence gaps", key="ntr_evaluate:" + prefix, type="primary"):
            cached = {"fingerprint": fingerprint, "manual_input": manual_input,
                      "reviewed_input": None, "result": None, "evaluation_error": None}
            try:
                if input_error:
                    raise ValueError(input_error)
                if any(finding.get("kind") == "project_scope_mismatch" for finding in report.get("findings", [])):
                    raise ValueError("Candidate context review required: source or manual input identity differs from the active compound.")
                inp = parse_review_input(manual_input, compound_name=name)
                updated, result = reviewed_assessment(inp, assertions, docs)
                cached.update({"manual_input": inp.to_dict(), "reviewed_input": updated.to_dict(),
                               "result": result.to_dict()})
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                cached["evaluation_error"] = str(exc)
            st.session_state.ntr_evaluation_state = cached
        evaluation_error = cached.get("evaluation_error") if cached else None
        if evaluation_error:
            st.error(f"Evaluation blocked: {evaluation_error}")
        legacy_result = cached.get("result") if cached else None
        if legacy_result:
            a, b, c = st.columns(3)
            a.metric("Legacy evidence role (uncalibrated)", legacy_result.get("evidence_role_code", "Not evaluated"))
            b.metric("Legacy data-gap prompts", len(legacy_result.get("data_gaps", [])))
            c.metric("Legacy enabled evidence flags", legacy_result.get("evidence_stream_count", 0))
            st.warning("This optional legacy result is uncalibrated rule-engine output. It does not establish observed toxicity, safety, or regulatory acceptance.")
            if legacy_result.get("data_gaps"):
                st.dataframe(pd.DataFrame(_table_rows(legacy_result["data_gaps"])), hide_index=True, width="stretch",
                             alt="Optional legacy rule-engine data-gap prompts")
            with st.expander("Legacy rules, uncertainty & full result"):
                st.json(legacy_result, expanded=False)
    export = build_review_export(report, docs, assertions, project_id=project_id, compound_name=name,
        package=package, manual_input=manual_input,
        reviewed_input=cached.get("reviewed_input") if cached else None,
        result=cached.get("result") if cached else None,
        evaluation_error=cached.get("evaluation_error") if cached else input_error,
        raw_manual_input=raw)
    export["review_fingerprint"] = fingerprint
    st.session_state.ntr_review_export = export
    with download_slot:
        st.download_button("Download review + provenance", json.dumps(export, ensure_ascii=False, indent=2, default=str),
                           "toxiscope-ntr-review.json", "application/json", key="ntr_download:" + prefix, width="content")
    _save_workspace(key)
