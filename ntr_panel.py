"""Compact local NTR workspace, sharing ToxiScope context without auto-approval."""
import json
import pandas as pd
import streamlit as st
from nora.models import AssessmentInput
from nora.evidence import extract_document
from nora.assertions import (extract_assertions_from_documents, assertion_table_rows,
    assertions_from_table_rows, REVIEW_STATUS_OPTIONS)
from ntr_bridge import empty_input, context_key, context_snapshot, reviewed_assessment, SOURCE_COMMIT

def render_ntr_panel(project_id, name, package):
    st.subheader("Toxicity Evidence Review")
    st.caption("Project → sources → human review → gaps. Local processing on this app server; no external AI calls.")
    st.warning("Research prototype: fixed rules, not trained toxicity prediction or regulatory acceptance. "
               "Early hepatotoxicity only; other endpoints remain unevaluable (R0). "
               "R0–R5 are uncalibrated evidence-role labels, not safety scores. Indication does not change the rules.")
    key = context_key(project_id, name, package)
    if st.session_state.get("ntr_context_key") != key:
        st.session_state.ntr_context_key = key
        st.session_state.ntr_docs = []
        st.session_state.ntr_assertions = []
        st.session_state.ntr_generation = st.session_state.get("ntr_generation", 0) + 1
    if st.button("Clear NTR workspace", key="ntr_clear"):
        st.session_state.ntr_docs = []
        st.session_state.ntr_assertions = []
        st.session_state.ntr_generation += 1
        st.rerun()
    prefix = key + ":" + str(st.session_state.ntr_generation)
    docs = st.session_state.ntr_docs
    assertions = st.session_state.ntr_assertions
    cols = st.columns(3)
    cols[0].metric("Sources", len(docs))
    cols[1].metric("Assertions", len(assertions))
    cols[2].metric("Reviewed", sum(a.review_status in {"승인", "수정"} for a in assertions))
    with st.expander("Project context & structural evidence", expanded=False):
        st.caption("ToxiScope results stay separate from customer observations. They do not populate NAM/AI outcomes.")
        st.json(context_snapshot(project_id, name, package), expanded=False)
    with st.expander("1 · Customer evidence", expanded=not docs):
        st.caption("Upload only documents permitted on this server. Raw text stays in this session; download before closing. "
                   "No OCR. Failed extraction does not mean evidence is absent.")
        source_kind = st.selectbox("Source type", ["unclassified", "customer_evidence", "external_reference"],
                                  key="ntr_source:" + prefix)
        uploads = st.file_uploader("PDF, DOCX, CSV, XLSX, JSON or TXT", accept_multiple_files=True,
            type=["pdf", "docx", "csv", "xlsx", "json", "txt"], key="ntr_upload:" + prefix)
        if st.button("Extract locally", key="ntr_extract:" + prefix):
            for upload in uploads or []:
                if upload.size > 10 * 1024 * 1024:
                    st.error("Maximum document size: 10 MB."); continue
                document = extract_document(upload.getvalue(), upload.name, upload.type)
                document.source_kind = source_kind
                if any(d.sha256 == document.sha256 for d in docs):
                    st.info("Duplicate file skipped: " + document.name); continue
                docs.append(document)
                assertions.extend(extract_assertions_from_documents([document]))
            st.rerun()
        for document in docs:
            document.source_kind = st.selectbox(document.name + " · source classification",
                ["unclassified", "customer_evidence", "external_reference"],
                index=["unclassified", "customer_evidence", "external_reference"].index(document.source_kind),
                key="kind:" + document.document_id)
            for warning in document.warnings:
                st.warning(warning)
            with st.expander(document.name + " · provenance"):
                st.json(document.to_dict(), expanded=False)
    with st.expander("2 · Review proposed assertions", expanded=bool(assertions)):
        st.caption("Regex extraction proposes claims only. Approve/edit after reading the cited excerpt. "
                   "External references and unclassified sources cannot become customer facts.")
        if assertions:
            rows = assertion_table_rows(assertions)
            editable = {"제안/수정 값", "검토 상태", "검토 메모"}
            edited = st.data_editor(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                disabled=[c for c in rows[0] if c not in editable],
                column_config={"검토 상태": st.column_config.SelectboxColumn(options=REVIEW_STATUS_OPTIONS)},
                key="ntr_review:" + prefix + ":" + ":".join(a.assertion_id for a in assertions))
            assertions = assertions_from_table_rows(edited.to_dict("records"), assertions)
            st.session_state.ntr_assertions = assertions
        else:
            st.info("No proposed assertions. You may enter reviewed data below; absent extraction is not a negative finding.")
    with st.expander("3 · Advanced assessment inputs", expanded=False):
        st.caption("Manual entries are user-supplied, not independently verified. Default AI/NAM streams are disabled. "
                   "Review every enabled field, source and use context; do not treat template values as observations.")
        raw = st.text_area("Reviewed NTR input (JSON)", json.dumps(empty_input(name).to_dict(),
            ensure_ascii=False, indent=2), height=300, key="ntr_input:" + prefix)
    if st.button("Review evidence gaps", key="ntr_evaluate:" + prefix, type="primary"):
        try:
            inp = AssessmentInput.from_dict(json.loads(raw))
            updated, result = reviewed_assessment(inp, assertions, docs)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            st.error("Evaluation blocked: " + str(exc))
            return
        a, b, c = st.columns(3)
        a.metric("Provisional evidence role", result.evidence_role_code)
        b.metric("Data gaps", len(result.data_gaps))
        c.metric("Evidence streams", result.evidence_stream_count)
        if result.data_gaps:
            st.dataframe(pd.DataFrame([g.to_dict() for g in result.data_gaps]),
                         hide_index=True, use_container_width=True)
        with st.expander("Rules, uncertainty & full result"):
            st.json(result.to_dict(), expanded=False)
        export = {"schema": "toxiscope-ntr-review-1", "ntr_source_commit": SOURCE_COMMIT,
                  "structural_context": context_snapshot(project_id, name, package),
                  "manual_input": inp.to_dict(), "reviewed_input": updated.to_dict(),
                  "documents": [d.to_dict() for d in docs],
                  "assertions": [a.to_dict() for a in assertions], "result": result.to_dict(),
                  "limitations": "Uncalibrated rule-based evidence triage; not safety or regulatory approval."}
        st.download_button("Download review + provenance", json.dumps(export, ensure_ascii=False,
            indent=2, default=str), "toxiscope-ntr-review.json", "application/json")
