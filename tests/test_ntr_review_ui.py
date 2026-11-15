"""Synthetic UI regressions for source-linked project evidence review."""
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from nora.evidence import EvidenceAssertion, extract_document


APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def _start():
    app = AppTest.from_file(APP).run(timeout=30)
    assert not app.exception
    app.radio(key="workspace").set_value("Toxicity Evidence Review").run(timeout=30)
    assert not app.exception
    return app


def _evidence(value, *, source_kind="customer_evidence", evidence_kind="observation", number=1,
              endpoint="초기 간독성"):
    document = extract_document(f"Synthetic assay {number}: result {value}.".encode(), f"assay-{number}.txt")
    document.source_kind = source_kind
    assertion = EvidenceAssertion(
        assertion_id=f"AST-SYNTHETIC-{number}", category="NAM assay", field_path="nam_assay.result",
        label_ko="Synthetic assay result", proposed_value=value, value_type="str",
        source_document_id=document.document_id, source_document_name=document.name,
        source_location=document.segments[0].location, source_excerpt=document.segments[0].text,
        confidence=0.5, review_status="승인", extracted_value=value,
        source_segment_id=document.segments[0].segment_id,
    )
    assertion.evidence_kind = evidence_kind
    assertion.scope = {"subject": "Synthetic A", "endpoint": endpoint, "system": "human hepatocytes",
                       "exposure": "1 µM", "timepoint": "24 hours", "study_id": "SYNTHETIC-1"}
    return document, assertion


def _seed(app, pairs):
    app.session_state["ntr_docs"] = [pair[0] for pair in pairs]
    app.session_state["ntr_assertions"] = [pair[1] for pair in pairs]
    app.run(timeout=30)
    assert not app.exception
    return app


def _evaluate(app):
    next(button for button in app.button if button.label == "Review evidence gaps").click().run(timeout=30)
    assert not app.exception


def _manual_input(app):
    return next(item for item in app.text_area if item.label == "Reviewed NTR input (JSON)")


def _text(app):
    return "\n".join(str(element.value) for elements in
                     (app.markdown, app.caption, app.warning, app.error, app.info) for element in elements)


def test_possible_conflicts_visible_and_exportable_even_when_legacy_evaluation_blocks():
    app = _seed(_start(), [_evidence("positive", number=1), _evidence("negative", number=2)])
    report = app.session_state["ntr_project_review"]
    assert report["summary"]["possible_conflicts"] >= 1
    assert any(metric.label == "Possible conflicts" and int(metric.value) >= 1 for metric in app.metric)
    assert app.get("download_button")
    before = app.session_state["ntr_review_export"]
    assert before["result"] is None
    assert before["schema"] == "toxiscope-ntr-review-2"
    assert before["project_review"]["schema"] == "toxiscope-ntr-project-review.v1"
    _evaluate(app)
    assert any("Evaluation blocked" in item.value for item in app.error)
    export = app.session_state["ntr_review_export"]
    assert export["evaluation_error"]
    assert export["result"] is None
    assert export["project_review"]["findings"]
    assert export["project_review"]["coverage"]
    assert export["limitations"]
    assert app.get("download_button")
    serialized = json.dumps(export, ensure_ascii=False)
    for document in app.session_state["ntr_docs"]:
        assert document.document_id in serialized
        assert document.sha256 in serialized
        assert document.segments[0].location in serialized
        assert document.segments[0].text in serialized
    assert any(finding["provenance"] for finding in export["project_review"]["findings"])


def test_reference_and_prediction_remain_distinct_and_review_columns_are_available():
    app = _seed(_start(), [
        _evidence("positive", source_kind="external_reference", evidence_kind="context", number=1),
        _evidence("negative", evidence_kind="prediction", number=2),
    ])
    export = app.session_state["ntr_review_export"]
    assert {a["evidence_kind"] for a in export["assertions"]} == {"context", "prediction"}
    assert {d["source_kind"] for d in export["documents"]} == {"external_reference", "customer_evidence"}
    editor_frames = [element.value for element in app.dataframe
                     if "Evidence kind" in element.value.columns]
    assert editor_frames
    assert {"Scope " + field for field in ("subject", "endpoint", "system", "exposure", "timepoint", "study_id")} <= set(editor_frames[0].columns)
    _evaluate(app)
    assert any("Evaluation blocked" in item.value for item in app.error)
    assert "not independently observed facts" in _text(app)
    assert "automatically an observation" in _text(app)


def test_empty_and_malformed_input_keep_review_and_download_without_negative_finding():
    app = _start()
    document = extract_document(b"", "empty.txt")
    app.session_state["ntr_docs"] = [document]
    app.run(timeout=30)
    assert not app.exception
    report = app.session_state["ntr_project_review"]
    assert report["decision"] == "Expert review required"
    assert report["summary"]["coverage_gaps"] >= 1
    assert report["summary"]["source_issues"] >= 1
    assert not any(metric.value == "R0" for metric in app.metric)
    _manual_input(app).set_value("{malformed").run(timeout=30)
    assert not app.exception
    _evaluate(app)
    assert any("Evaluation blocked" in item.value for item in app.error)
    export = app.session_state["ntr_review_export"]
    assert export["result"] is None
    assert export["evaluation_error"]
    assert export["documents"][0]["sha256"] == document.sha256
    assert export["raw_manual_input"] == "{malformed"
    assert app.get("download_button")
    assert "not a negative finding" in _text(app)


def test_changed_manual_input_or_source_review_clears_old_result():
    app = _start()
    _evaluate(app)
    assert any(metric.value == "R0" for metric in app.metric)
    manual = json.loads(_manual_input(app).value)
    manual["context_of_use"]["target_endpoint"] = "Synthetic requested endpoint"
    _manual_input(app).set_value(json.dumps(manual)).run(timeout=30)
    assert not app.exception
    assert app.session_state["ntr_review_export"]["result"] is None
    assert not any(metric.value == "R0" for metric in app.metric)
    assert "Synthetic requested endpoint" in _text(app)
    _evaluate(app)
    assert app.session_state["ntr_review_export"]["result"]
    app = _seed(app, [_evidence("positive", number=1, endpoint="Synthetic requested endpoint")])
    assert app.session_state["ntr_review_export"]["result"] is None
    _evaluate(app)
    assert app.session_state["ntr_review_export"]["result"]
    document = app.session_state["ntr_docs"][0]
    app.selectbox(key="kind:" + document.document_id).set_value("external_reference").run(timeout=30)
    assert not app.exception
    assert app.session_state["ntr_review_export"]["result"] is None


def test_structural_package_change_preserves_evidence_and_clears_legacy_result():
    app = _seed(_start(), [_evidence("positive", number=1)])
    _evaluate(app)
    assert app.session_state["ntr_review_export"]["result"]
    document_id = app.session_state["ntr_docs"][0].document_id
    app.session_state["evidence_package"] = {
        "assessment": {"screening_method": "Rule-based structural alert screening"},
        "generated_at": "synthetic changed SAR timestamp",
    }
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state["ntr_docs"][0].document_id == document_id
    assert len(app.session_state["ntr_assertions"]) == 1
    assert app.session_state["ntr_review_export"]["result"] is None


def test_project_compound_workspaces_restore_evidence_and_clear_only_active_workspace():
    app = _seed(_start(), [_evidence("positive", number=1)])
    document_id = app.session_state["ntr_docs"][0].document_id
    original_name = app.text_input(key="ntr_compound_name").value
    app.text_input(key="ntr_compound_name").set_value("Synthetic second compound").run(timeout=30)
    assert not app.exception
    assert app.session_state["ntr_docs"] == []
    app = _seed(app, [_evidence("negative", number=2)])
    next(button for button in app.button if button.label == "Clear NTR workspace").click().run(timeout=30)
    assert app.session_state["ntr_docs"] == []
    app.text_input(key="ntr_compound_name").set_value(original_name).run(timeout=30)
    assert not app.exception
    assert app.session_state["ntr_docs"][0].document_id == document_id
    assert len(app.session_state["ntr_assertions"]) == 1


def test_candidate_mismatch_keeps_findings_and_export_when_evaluation_is_blocked():
    app = _start()
    app.text_input(key="ntr_compound_name").set_value("Synthetic active candidate").run(timeout=30)
    manual = json.loads(_manual_input(app).value)
    manual["product"]["product_name"] = "Synthetic other candidate"
    _manual_input(app).set_value(json.dumps(manual)).run(timeout=30)
    assert not app.exception
    assert any(finding["kind"] == "project_scope_mismatch"
               for finding in app.session_state["ntr_project_review"]["findings"])
    _evaluate(app)
    assert any("Candidate context review required" in item.value for item in app.error)
    export = app.session_state["ntr_review_export"]
    assert export["result"] is None
    assert export["evaluation_error"]
    assert any(finding["kind"] == "project_scope_mismatch" for finding in export["project_review"]["findings"])
    assert app.get("download_button")


def test_legacy_evidence_with_unknown_project_is_preserved_unbound():
    document, assertion = _evidence("positive")
    app = AppTest.from_file(APP)
    app.session_state["workspace"] = "Toxicity Evidence Review"
    app.session_state["ntr_docs"] = [document]
    app.session_state["ntr_assertions"] = [assertion]
    app.session_state["ntr_context_key"] = "unmatched-previous-project-context"
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state["ntr_docs"] == []
    assert app.session_state["ntr_assertions"] == []
    quarantined = app.session_state["ntr_unbound_legacy_workspace"]
    saved = quarantined["workspaces"][0]
    assert saved["documents"][0] == document.to_dict()
    assert saved["assertions"][0] == assertion.to_dict()
    assert saved["previous_context_key"] == "unmatched-previous-project-context"
    assert any("Legacy evidence could not be matched" in item.value for item in app.warning)
    assert any(item.proto.label == "Download unbound legacy evidence + provenance" for item in app.get("download_button"))
    next(button for button in app.button if button.label == "Clear NTR workspace").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["ntr_unbound_legacy_workspace"] == quarantined


def test_legacy_evidence_with_matching_context_is_adopted_without_loss():
    from ntr_bridge import context_key

    document, assertion = _evidence("positive")
    app = AppTest.from_file(APP)
    app.session_state["workspace"] = "Toxicity Evidence Review"
    app.session_state["ntr_docs"] = [document]
    app.session_state["ntr_assertions"] = [assertion]
    app.session_state["ntr_context_key"] = context_key("TXS-2026-001", "", {})
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state["ntr_docs"][0].to_dict() == document.to_dict()
    assert app.session_state["ntr_assertions"][0].to_dict() == assertion.to_dict()
    assert app.session_state["ntr_review_export"]["documents"][0] == document.to_dict()
    assert not any("Legacy evidence could not be matched" in item.value for item in app.warning)


def test_valid_json_with_invalid_field_types_keeps_review_and_export_available():
    app = _start()
    manual = {"context_of_use": {"target_endpoint": ["Cardiotoxicity"]}}
    _manual_input(app).set_value(json.dumps(manual)).run(timeout=30)
    assert not app.exception
    assert any("Manual input requires correction" in item.value for item in app.warning)
    assert app.session_state["ntr_review_export"]["manual_input"] == manual
    assert app.session_state["ntr_project_review"]["decision"] == "Expert review required"
    _evaluate(app)
    assert any("Evaluation blocked" in item.value for item in app.error)
    export = app.session_state["ntr_review_export"]
    assert export["result"] is None
    assert export["evaluation_error"]
    assert export["manual_input"] == manual
    assert app.get("download_button")


def test_editor_delta_cannot_replay_approval_after_a_semantic_edit():
    app = _seed(_start(), [_evidence("positive")])
    _evaluate(app)
    assert app.session_state["ntr_review_export"]["result"]
    editor = next(item for item in app.dataframe if "Assertion ID" in item.value.columns)
    old_key = editor.key
    # AppTest has no data-editor cell interaction API. Seed its actual widget
    # delta to exercise the same server-side state as a frontend edit.
    app.session_state[old_key] = {
        "edited_rows": {0: {"제안/수정 값": "negative", "검토 상태": "승인"}},
        "added_rows": [], "deleted_rows": [],
    }
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state["ntr_assertions"][0].proposed_value == "negative"
    assert app.session_state["ntr_assertions"][0].review_status == "제안됨"
    assert app.session_state["ntr_review_export"]["result"] is None
    app.run(timeout=30)
    assert not app.exception
    editor = next(item for item in app.dataframe if "Assertion ID" in item.value.columns)
    assert editor.key != old_key
    assert editor.value.iloc[0]["검토 상태"] == "제안됨"
    assert app.session_state["ntr_assertions"][0].review_status == "제안됨"
    assert app.session_state["ntr_assertions"][0].extracted_value == "positive"
    assert app.get("download_button")
    # A separate approval on the new revision is still available to the user.
    app.session_state[editor.key] = {
        "edited_rows": {0: {"검토 상태": "승인"}}, "added_rows": [], "deleted_rows": [],
    }
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state["ntr_assertions"][0].review_status == "승인"
