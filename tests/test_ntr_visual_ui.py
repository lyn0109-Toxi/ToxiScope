"""Presentation regressions: real counts and review states, never toxicity scores."""
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
from streamlit.testing.v1 import AppTest

from nora.evidence import EvidenceAssertion, extract_document
from presentation.ntr_visuals import coverage_heatmap


APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def _start():
    app = AppTest.from_file(APP).run(timeout=30)
    assert not app.exception
    app.radio(key="workspace").set_value("Toxicity Evidence Review").run(timeout=30)
    assert not app.exception
    return app


def _pair(number, value, *, field="nam_assay.result", kind="observation", review="승인", source="customer_evidence"):
    document = extract_document(f"Synthetic record {number}: {value}.".encode(), f"synthetic-{number}.txt")
    document.source_kind = source
    assertion = EvidenceAssertion(
        assertion_id=f"AST-VISUAL-{number}", category="Synthetic evidence", field_path=field,
        label_ko="Synthetic field", proposed_value=value, value_type="str",
        source_document_id=document.document_id, source_document_name=document.name,
        source_location=document.segments[0].location, source_excerpt=document.segments[0].text,
        source_segment_id=document.segments[0].segment_id, extracted_value=value,
        confidence=0.5, review_status=review, evidence_kind=kind,
        scope={"subject": "Synthetic A", "endpoint": "초기 간독성", "system": "human hepatocytes",
               "exposure": "1 µM", "timepoint": "24 hours", "study_id": "SYNTHETIC-VISUAL"},
    )
    return document, assertion


def _seed(app, pairs):
    app.session_state["ntr_docs"] = [pair[0] for pair in pairs]
    app.session_state["ntr_assertions"] = [pair[1] for pair in pairs]
    app.run(timeout=30)
    assert not app.exception
    return app


def _chart(app, alt_prefix):
    return next(chart for chart in app.get("vega_lite_chart") if chart.proto.alt.startswith(alt_prefix))


def _chart_data(chart):
    return pa.ipc.open_stream(chart.proto.datasets[0].data.data).read_all().to_pandas()


def test_visual_overview_uses_actual_claim_document_and_finding_counts():
    pairs = [
        _pair(1, "positive"),
        _pair(2, "negative", field="ai_model.result", kind="prediction", review="거절"),
        _pair(3, "intravenous", field="product.route", kind="context", review="제안됨", source="external_reference"),
        _pair(4, "unreported", field="nam_assay.measured_exposure", kind="unspecified", review="제안됨", source="unclassified"),
    ]
    app = _seed(_start(), pairs)
    assert len(app.get("vega_lite_chart")) == 4
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["자료"] == "4"
    assert metrics["주장"] == "4"
    assert metrics["검토된 주장"] == "1"
    report = app.session_state["ntr_project_review"]
    assert int(metrics["값 차이 후보"]) == report["summary"]["possible_conflicts"]
    finding_counts = _chart_data(_chart(app, "현재 보고서의 항목별"))
    assert finding_counts["건수"].sum() == len(report["findings"])
    kinds = _chart_data(_chart(app, "관찰형·예측형"))
    assert dict(zip(kinds["근거 유형"], kinds["건수"])) == {"관찰": 1, "예측": 1, "맥락": 1, "유형 미지정": 1}
    sources = _chart_data(_chart(app, "고객 제공 자료"))
    assert dict(zip(sources["출처 구분"], sources["문서 수"])) == {"고객 제공 자료": 2, "외부 참고자료": 1, "출처 미분류": 1}
    coverage = _chart_data(_chart(app, "영역별 필드"))
    assert coverage["필드수"].sum() == len(report["coverage"])
    assert set(coverage["상태 코드"]) == {row["review_state"] for row in report["coverage"]}
    assert "not_extracted" in set(coverage["상태 코드"])
    assert not app.success


def test_empty_sources_have_explicit_unknown_states_without_fake_zero_bars():
    app = _start()
    assert len(app.get("vega_lite_chart")) == 2
    captions = "\n".join(item.value for item in app.caption)
    assert "문서 없음 · 출처 검토 대기" in captions
    assert "주장 없음 · 근거 유형 미확인" in captions
    assert "검토 지원 · 독성/규제 판정 아님" in captions
    assert "세션 내 보관 · 닫기 전 다운로드" in captions
    assert "출처 연결 ≠ 과학적 검증" in captions
    coverage = _chart_data(_chart(app, "영역별 필드"))
    assert set(coverage["상태 코드"]) == {"not_extracted", "not_requested"}
    assert (coverage["필드수"] > 0).all()
    collapsed = {item.label: item.proto.expanded for item in app.expander}
    assert collapsed["근거·방법·한계"] is False
    assert collapsed["주장 검토 · 값/유형/범위"] is False
    assert collapsed["고급 입력 · 요청 범위와 수동 JSON"] is False
    assert app.session_state["ntr_review_export"]["result"] is None


def test_selected_findings_and_full_provenance_remain_accessible_with_blocked_export():
    app = _seed(_start(), [_pair(1, "positive"), _pair(2, "negative")])
    report = app.session_state["ntr_project_review"]
    conflict = next(finding for finding in report["findings"] if finding["kind"] == "possible_conflict")
    selector = next(item for item in app.selectbox if item.label == "상세 검토 항목")
    selector.set_value(conflict["finding_id"]).run(timeout=30)
    assert not app.exception
    assert any(conflict["finding_id"] in item.value for item in app.json)
    assert next(item for item in app.expander if item.label == "선택 항목 · 근거와 범위").proto.expanded is False
    rendered_json = "\n".join(item.value for item in app.json)
    for document in app.session_state["ntr_docs"]:
        assert document.sha256 in rendered_json
        assert document.segments[0].text in rendered_json
    next(item for item in app.button if item.label == "Review evidence gaps").click().run(timeout=30)
    assert not app.exception
    assert any("Evaluation blocked" in item.value for item in app.error)
    assert app.session_state["ntr_review_export"]["evaluation_error"]
    assert app.session_state["ntr_review_export"]["project_review"]["findings"]
    assert any(item.proto.label == "Download review + provenance" for item in app.get("download_button"))


def test_heatmap_keeps_unknown_unrequested_rejected_and_scope_states_separate():
    codes = ["not_requested", "not_extracted", "all_candidates_rejected", "scope_review_required",
             "prediction_review_required", "unrecognized_one", "unrecognized_two"]
    frame = pd.DataFrame([{"영역": "NAM 자료", "필드": f"field-{index}", "검토 상태": code, "상태 코드": code}
                          for index, code in enumerate(codes)])
    spec = coverage_heatmap(frame).to_dict()
    values = next(iter(spec["datasets"].values()))
    assert sum(row["필드수"] for row in values) == len(codes)
    labels = {row["상태 코드"]: row["상태"] for row in values}
    assert len(set(labels.values())) == len(codes)
    assert labels["not_requested"] == "요청 범위 밖"
    assert labels["not_extracted"] == "미확인"
    assert labels["all_candidates_rejected"] == "모두 거절"
    assert labels["scope_review_required"] == "범위 점검"
    assert labels["prediction_review_required"] == "예측 점검"
    assert not any(row["필드수"] == 0 for row in values)
