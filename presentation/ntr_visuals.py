"""Visual summaries of recorded NTR review states, never toxicity scores."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from .data import coverage_matrix, finding_pareto, evidence_kind_counts, source_state_counts


INK = "#172B3A"
CYAN = "#168C94"
AMBER = "#B77824"
SLATE = "#8796A5"
LIGHT_BG = "#F5F7FA"

_STATE_LABELS = {
    "not_requested": "요청 범위 밖",
    "not_extracted": "미확인",
    "all_candidates_rejected": "모두 거절",
    "human_review_pending": "검토 대기",
    "source_review_required": "출처 점검",
    "value_review_required": "값 점검",
    "observation_review_required": "유형 점검",
    "prediction_review_required": "예측 점검",
    "scope_review_required": "범위 점검",
    "conflicting_reviewed_values": "값 차이",
    "reviewed_source_linked": "출처 연결",
}
_STATE_COLORS = {
    "요청 범위 밖": "#E5EAF0",
    "미확인": "#B6C1CC",
    "모두 거절": "#DDCDB7",
    "검토 대기": "#C5DFE0",
    "출처 점검": "#E8CDA9",
    "값 점검": "#DFC092",
    "유형 점검": "#E8DCCB",
    "예측 점검": "#D5DFD8",
    "범위 점검": "#D6D8E8",
    "값 차이": "#D8AD74",
    "출처 연결": "#75B9BD",
    "상태 미기록": "#CBD2DA",
}


def _style(chart, height):
    return (chart.properties(height=height, background="transparent")
        .configure_view(stroke=None)
        .configure_axis(labelColor=INK, titleColor=INK, labelFont="sans-serif", titleFont="sans-serif",
                        gridColor="#E7ECEF", domain=False, tickColor="#D4DEE4", titleFontWeight=500)
        .configure_legend(labelColor=INK, titleColor=INK, labelFont="sans-serif", titleFont="sans-serif"))


def coverage_heatmap(frame: pd.DataFrame):
    """Count fields by recorded coverage state without filling unobserved cells."""
    if frame.empty:
        return None
    data = frame.copy()
    data["상태"] = data["상태 코드"].map(
        lambda code: "상태 미기록" if pd.isna(code) else _STATE_LABELS.get(code, "기타 · " + str(code)))
    cells = (data.groupby(["영역", "상태", "상태 코드", "검토 상태"], sort=False, dropna=False)
             .agg(필드수=("필드", "size"), 필드목록=("필드", lambda values: " · ".join(map(str, values))))
             .reset_index())
    actual_states = list(dict.fromkeys(cells["상태"].tolist()))
    states = [label for label in _STATE_COLORS if label in actual_states]
    states.extend(label for label in actual_states if label not in states)
    areas = list(dict.fromkeys(cells["영역"].tolist()))
    base = alt.Chart(cells).encode(
        x=alt.X("상태:N", title=None, sort=states, axis=alt.Axis(labelAngle=0, labelLimit=100)),
        y=alt.Y("영역:N", title=None, sort=areas, axis=alt.Axis(labelLimit=130)),
        tooltip=[alt.Tooltip("영역:N"), alt.Tooltip("검토 상태:N", title="기록된 상태"),
                 alt.Tooltip("상태 코드:N"), alt.Tooltip("필드수:Q", title="필드 수", format="d"),
                 alt.Tooltip("필드목록:N", title="포함 필드")],
    )
    blocks = base.mark_rect(cornerRadius=4, stroke="white", strokeWidth=3).encode(
        color=alt.Color("상태:N", scale=alt.Scale(domain=states,
            range=[_STATE_COLORS.get(state, SLATE) for state in states]), legend=None))
    labels = base.mark_text(color=INK, fontSize=14, fontWeight=600).encode(text=alt.Text("필드수:Q", format="d"))
    return _style(blocks + labels, max(135, len(areas) * 32))


def finding_counts_chart(frame: pd.DataFrame):
    if frame.empty:
        return None
    base = alt.Chart(frame).encode(
        x=alt.X("건수:Q", title="검토 항목 수", axis=alt.Axis(tickMinStep=1), scale=alt.Scale(zero=True)),
        y=alt.Y("항목:N", title=None, sort=frame["항목"].tolist(), axis=alt.Axis(labelLimit=160)),
        tooltip=[alt.Tooltip("항목:N"), alt.Tooltip("건수:Q", format="d")],
    )
    bars = base.mark_bar(color=AMBER, cornerRadiusEnd=4, size=17)
    labels = base.mark_text(align="left", dx=5, color=INK, fontWeight=600).encode(text=alt.Text("건수:Q", format="d"))
    return _style(bars + labels, max(135, len(frame) * 29))


def category_counts_chart(frame: pd.DataFrame, label: str, count: str, color: str):
    if frame.empty:
        return None
    base = alt.Chart(frame).encode(
        x=alt.X(f"{count}:Q", title=None, axis=alt.Axis(tickMinStep=1), scale=alt.Scale(zero=True)),
        y=alt.Y(f"{label}:N", title=None, sort=frame[label].tolist(), axis=alt.Axis(labelLimit=180)),
        tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip(f"{count}:Q", format="d")],
    )
    bars = base.mark_bar(color=color, cornerRadiusEnd=4, size=15)
    labels = base.mark_text(align="left", dx=5, color=INK, fontWeight=600).encode(text=alt.Text(f"{count}:Q", format="d"))
    return _style(bars + labels, max(100, len(frame) * 27))


def render_ntr_charts(report, assertions, documents, *, key_prefix):
    quality, differences = st.columns([1.2, 1])
    with quality:
        st.markdown("##### 데이터 품질 · 필드 검토 상태")
        chart = coverage_heatmap(coverage_matrix(report))
        if chart is not None:
            st.altair_chart(chart, width="stretch", theme=None,
                key="ntr_coverage_heatmap:" + key_prefix,
                alt="영역별 필드 검토 상태. 미확인, 요청 범위 밖, 거절, 출처 연결은 서로 다른 상태입니다.")
        else:
            st.caption("검토 상태 자료 없음 · 근거 유무 미확인")
        st.caption("숫자 = 필드 수 · 출처 연결 ≠ 과학적 검증")
    with differences:
        st.markdown("##### 비교·차이 · 검토 항목")
        chart = finding_counts_chart(finding_pareto(report))
        if chart is not None:
            st.altair_chart(chart, width="stretch", theme=None,
                key="ntr_finding_counts:" + key_prefix,
                alt="현재 보고서의 항목별 검토 건수. 심각도나 독성 위험 점수가 아닙니다.")
        else:
            st.caption("추출된 비교 항목 없음 · 일치·안전을 뜻하지 않음")
        st.caption("건수는 검토 목록의 수 · 심각도 순위 아님")
    kinds, sources = st.columns(2)
    with kinds:
        st.markdown("##### 주장 유형")
        chart = category_counts_chart(evidence_kind_counts(assertions), "근거 유형", "건수", CYAN)
        if chart is not None:
            st.altair_chart(chart, width="stretch", theme=None,
                key="ntr_evidence_kind_counts:" + key_prefix,
                alt="관찰형·예측형·맥락형·미지정 주장의 수. 승인되거나 검증된 관찰 수가 아닙니다.")
        else:
            st.caption("주장 없음 · 근거 유형 미확인")
        st.caption("제안·거절 주장을 포함한 분류 수 · 관찰 사실로 자동 승격하지 않음")
    with sources:
        st.markdown("##### 문서 출처 분류")
        chart = category_counts_chart(source_state_counts(documents), "출처 구분", "문서 수", SLATE)
        if chart is not None:
            st.altair_chart(chart, width="stretch", theme=None,
                key="ntr_source_state_counts:" + key_prefix,
                alt="고객 제공 자료·외부 참고자료·미분류 문서의 수. 독립 연구 수를 뜻하지 않습니다.")
        else:
            st.caption("문서 없음 · 출처 검토 대기")
        st.caption("업로드 문서 수 · 출처의 독립성·진위는 미확인")
