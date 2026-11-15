"""Local, source-linked NTR review; legacy assessment stays optional and separate."""
import hashlib
import json

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


def _render_project_review(report, project_id, name, manual_input):
    st.subheader("Project evidence gaps and possible conflicts")
    context = manual_input.get("context_of_use", {}) if isinstance(manual_input, dict) else {}
    endpoint = context.get("target_endpoint") if isinstance(context, dict) else None
    st.caption(f"Project: {project_id or 'Not specified'} · Compound: {name or 'Not specified'} · "
               f"Requested endpoint: {endpoint or 'Not specified'}")
    st.caption("The endpoint is the requested review scope, not an observed toxicity finding. "
               "Manual input values and template fields do not count as source-linked observations.")
    summary = report.get("summary", {})
    cols = st.columns(4)
    for col, label, field in zip(cols,
        ("Possible conflicts", "Scope differences", "Coverage gaps", "Source issues"),
        ("possible_conflicts", "scope_differences", "coverage_gaps", "source_issues")):
        col.metric(label, summary.get(field, 0))
    st.warning(report.get("decision", "Expert review required"))
    findings = report.get("findings", [])
    if findings:
        st.dataframe(pd.DataFrame(_table_rows([
            {key: finding.get(key, "") for key in
             ("kind", "field_path", "title", "detail", "uncertainty", "review_state")}
            for finding in findings
        ])), hide_index=True, width="stretch", alt="Source-linked evidence gaps and possible conflicts requiring expert review")
        for finding in findings:
            with st.expander(f"{finding.get('kind', 'Review finding')} · {finding.get('title', finding.get('field_path', 'Source-linked finding'))}"):
                st.write(finding.get("detail", ""))
                st.caption(finding.get("uncertainty", "Expert review required."))
                st.json({"finding_id": finding.get("finding_id"), "scope": finding.get("scope", {}),
                         "provenance": finding.get("provenance", []),
                         "review_state": finding.get("review_state", "needs_expert_review")}, expanded=False)
    else:
        st.info("No possible conflict was detected in the current extracted assertions. "
                "This does not establish consistency, evidence completeness, or safety.")
    with st.expander("Coverage and source locations", expanded=True):
        coverage = report.get("coverage", [])
        if coverage:
            st.dataframe(pd.DataFrame(_table_rows(coverage)), hide_index=True, width="stretch",
                         alt="Intake field coverage with assertion identifiers and source locations")
        else:
            st.info("No coverage rows are available. Missing extraction is not a negative finding.")
    with st.expander("Review scope and limitations"):
        st.json(report.get("scope", {}), expanded=False)
        for limitation in report.get("limitations", []):
            st.caption(limitation)


def render_ntr_panel(project_id, name, package):
    st.subheader("Toxicity Evidence Review")
    st.caption("Project → sources → human review → gaps and possible conflicts. "
               "Local processing on this app server; no external AI calls.")
    st.warning("Research prototype: deterministic evidence review, not toxicity prediction or regulatory acceptance. "
               "Observation, prediction, reference context and unreviewed assertions remain distinct. "
               "Empty or failed extraction does not establish absence of toxicity or evidence.")
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
    if st.button("Clear NTR workspace", key="ntr_clear"):
        st.session_state.ntr_docs = []
        st.session_state.ntr_assertions = []
        st.session_state.ntr_generation += 1
        st.session_state.ntr_manual_raw = None
        st.session_state.ntr_evaluation_state = None
        _save_workspace(key)
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
    with st.expander("1 · Source documents", expanded=not docs):
        st.caption("Upload only documents permitted on this server. Raw text stays in this session; download before closing. "
                   "No OCR. Failed extraction does not mean evidence is absent. Classify each source explicitly.")
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
    with st.expander("2 · Review proposed assertions", expanded=bool(assertions)):
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
    with st.expander("3 · Advanced assessment inputs", expanded=False):
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
    _render_project_review(report, project_id, name, manual_input)
    structural_context = context_snapshot(project_id, name, package)
    fingerprint = _fingerprint(raw, assertions, docs, structural_context)
    cached = st.session_state.get("ntr_evaluation_state")
    if cached and cached.get("fingerprint") != fingerprint:
        st.session_state.ntr_evaluation_state = None
        cached = None
        st.info("Previous legacy assessment output was cleared because the inputs, evidence review, sources or structural context changed.")
    with st.expander("Optional legacy evidence-role assessment (uncalibrated)"):
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
        if cached and cached.get("evaluation_error"):
            st.error("Evaluation blocked: " + cached["evaluation_error"])
        if cached and cached.get("result"):
            result = cached["result"]
            a, b, c = st.columns(3)
            a.metric("Legacy evidence role (uncalibrated)", result.get("evidence_role_code", "Not evaluated"))
            b.metric("Legacy data-gap prompts", len(result.get("data_gaps", [])))
            c.metric("Legacy enabled evidence flags", result.get("evidence_stream_count", 0))
            st.warning("This optional legacy result is uncalibrated rule-engine output. It does not establish observed toxicity, safety, or regulatory acceptance.")
            if result.get("data_gaps"):
                st.dataframe(pd.DataFrame(_table_rows(result["data_gaps"])), hide_index=True, width="stretch",
                             alt="Optional legacy rule-engine data-gap prompts")
            with st.expander("Legacy rules, uncertainty & full result"):
                st.json(result, expanded=False)
    export = build_review_export(report, docs, assertions, project_id=project_id, compound_name=name,
        package=package, manual_input=manual_input,
        reviewed_input=cached.get("reviewed_input") if cached else None,
        result=cached.get("result") if cached else None,
        evaluation_error=cached.get("evaluation_error") if cached else input_error,
        raw_manual_input=raw)
    export["review_fingerprint"] = fingerprint
    st.session_state.ntr_review_export = export
    st.download_button("Download review + provenance", json.dumps(export, ensure_ascii=False, indent=2, default=str),
                       "toxiscope-ntr-review.json", "application/json", key="ntr_download:" + prefix)
    _save_workspace(key)
