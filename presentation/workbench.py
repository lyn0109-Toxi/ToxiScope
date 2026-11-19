"""Visual review shell. Counts describe supplied records, not biomedical risk."""
from __future__ import annotations

from collections import Counter
from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from presentation.data import screening_rule_table, finding_pareto

INK, TEAL, AMBER, SLATE = "#172B3A", "#168C94", "#B77824", "#8796A5"


def install_brand():
    st.html("""<style>
.tg-brand{display:flex;align-items:center;gap:12px;margin-bottom:10px;color:#172B3A}
.tg-mark{display:grid;grid-template-columns:repeat(2,9px);gap:3px;transform:rotate(-12deg)}
.tg-mark i{display:block;width:9px;height:9px;border-radius:3px;background:#168C94}
.tg-mark i:last-child{background:#B77824}
.tg-brand-name{font-weight:750;font-size:20px;letter-spacing:-.8px}
.tg-brand-kicker{font-size:11px;letter-spacing:2px;color:#617383;margin-left:8px}
.tg-heading{font-size:clamp(26px,3vw,40px);font-weight:650;letter-spacing:-1.5px;line-height:1.22;margin:0 0 8px;color:#172B3A}
.tg-subtitle{color:#617383;font-size:14px;margin:0 0 18px;line-height:1.6}
.tg-flow{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:5px;margin:8px 0 18px}
.tg-step{padding:10px 12px;border-bottom:2px solid #DCE4EB;color:#718291;font-size:12px}
.tg-step b{font-size:11px;margin-right:8px;color:#8796A5;font-weight:500}
.tg-step.active{border-color:#168C94;color:#116C74;background:#EAF4F3;border-radius:6px 6px 0 0}
.tg-heading-small{font-size:19px;letter-spacing:-.5px;font-weight:650;color:#172B3A;margin:5px 0}
.tg-chip{display:inline-block;padding:5px 10px;font-size:12px;color:#6A4B1B;background:#FCF2E1;border-radius:5px}
.tg-empty{min-height:165px;display:flex;flex-direction:column;justify-content:center;align-items:center;gap:8px;background:#ECF2F5;border-radius:12px;color:#617383;margin-top:6px}
.tg-empty b{font-size:18px;color:#172B3A;font-weight:550}
@media(max-width:640px){.tg-brand-kicker{display:none}.tg-flow{gap:3px}.tg-step{padding:8px 4px;font-size:11px;word-break:keep-all}.tg-step b{display:block;margin-bottom:2px}.tg-heading{letter-spacing:-1px}}
</style>""")


def brand_header(title="근거를 한눈에.", subtitle="데이터의 상태, 신호의 차이, 다음 검토를 연결합니다."):
    st.html(f"""<div class="tg-brand"><span class="tg-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span><span class="tg-brand-name">ToxiGuard</span><span class="tg-brand-kicker">VISUAL EVIDENCE WORKBENCH</span></div><h1 class="tg-heading">{escape(title)}</h1><p class="tg-subtitle">{escape(subtitle)}</p>""")


def analysis_flow(active=0):
    labels = ["문제 정의", "데이터 품질", "비교·차이", "시나리오", "검토"]
    steps = "".join(f'<div class="tg-step {"active" if n == active else ""}"><b>{n+1:02d}</b>{label}</div>' for n, label in enumerate(labels))
    st.html(f'<div class="tg-flow" aria-label="분석 흐름: 문제 정의, 데이터 품질, 비교와 차이, 시나리오, 검토">{steps}</div>')


def empty_signal():
    st.html('<div class="tg-empty"><b>분자에서 근거까지</b><span>구조를 입력하면 일치 규칙과 근거 지도를 표시합니다.</span></div>')


def render_rule_chart(result):
    data = screening_rule_table(result)
    if data.empty:
        st.info("일치 규칙 없음 · 변이원성 미확정")
        return
    base = alt.Chart(data).encode(y=alt.Y("규칙:N", sort="-x", title=None, axis=alt.Axis(labelLimit=210)),
        x=alt.X("일치 수:Q", title="SMARTS 패턴 일치 수", axis=alt.Axis(tickMinStep=1)),
        tooltip=["규칙:N", "일치 수:Q", "일치 원자 수:Q"])
    bars = base.mark_bar(color=TEAL, cornerRadiusEnd=4, size=18)
    labels = base.mark_text(align="left", dx=7, color=INK).encode(text="일치 수:Q")
    chart = (bars + labels).properties(height=max(100, len(data)*42)).configure_view(stroke=None)
    st.altair_chart(chart, width="stretch", key="sar_rule_chart", alt="입력 분자에서 실제 일치한 SMARTS 규칙별 패턴 수; 독성 확률이 아님")
    st.caption("패턴 일치 수 · 독성 확률이나 독립된 실험 수가 아닙니다.")


def evidence_mix_chart(evidence):
    counts = Counter(str(e.get("evidence_type") or "미분류") for e in evidence or [])
    if not counts:
        st.caption("현재 분석의 근거 항목이 없습니다.")
        return
    frame = pd.DataFrame([{"근거 유형": key, "항목 수": value} for key, value in counts.items()])
    chart = alt.Chart(frame).mark_bar(color=TEAL, cornerRadiusEnd=3).encode(
        x=alt.X("항목 수:Q", title="기록 수", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("근거 유형:N", title=None, sort="-x", axis=alt.Axis(labelLimit=230)),
        tooltip=["근거 유형:N", "항목 수:Q"],
    ).properties(height=max(130,len(frame)*35)).configure_view(stroke=None)
    st.altair_chart(chart, width="stretch", key="sar_evidence_chart", alt="근거 유형별 실제 기록 수; 과학적 타당성이나 독립성의 점수가 아님")


def render_summary(results, degradants, be_result, ntr_report, project_id):
    st.subheader("프로젝트의 다음 질문")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("구조 규칙 일치", len(results.get("expert_alerts", [])) if results and results.get("status") != "error" else "—", border=True)
    with c2:
        st.metric("근거 연결·검토 공백", ntr_report.get("summary", {}).get("coverage_gaps", "—") if ntr_report else "—", border=True)
    with c3:
        st.metric("용출 f2 · 기술 통계", be_result.f2 if be_result else "—", border=True)
    left, right = st.columns([1.05,1])
    with left:
        with st.container(border=True):
            st.markdown("#### 01 / 근거가 어디에 모여 있나")
            evidence_mix_chart(results.get("evidence_objects", []) if results else [])
    with right:
        with st.container(border=True):
            st.markdown("#### 02 / 어떤 검토가 남아 있나")
            pareto = finding_pareto(ntr_report)
            if pareto.empty:
                st.caption("이 프로젝트·화합물의 근거 검토를 실행하면 확인 항목을 표시합니다.")
            else:
                chart = alt.Chart(pareto).mark_bar(color=AMBER,cornerRadiusEnd=3).encode(
                    x=alt.X("건수:Q",title="검토 항목 수",axis=alt.Axis(tickMinStep=1)),
                    y=alt.Y("항목:N",title=None,sort="-x"),tooltip=["항목:N","건수:Q"],
                ).properties(height=max(150,len(pareto)*32)).configure_view(stroke=None)
                st.altair_chart(chart,width="stretch",key="project_finding_chart",alt="활성 프로젝트 근거 검토에서 생성된 유형별 확인 항목 수")
    st.caption("— 미실행·미연결 · 모든 지표는 기술적 요약이며 안전성·규제 승인을 뜻하지 않습니다.")
    with st.expander("검토 루프 · 발견과 다음 확인"):
        st.text_area("관찰한 차이", key=f"learning_observation:{project_id}", placeholder="자료에서 직접 확인한 차이를 기록하세요.")
        st.text_area("다음에 확인할 근거", key=f"learning_next:{project_id}", placeholder="필요한 자료·조건·전문가 검토를 기록하세요.")
        st.caption("사용자 작성 메모 · 전문가의 승인이나 관찰 사실을 자동 생성하지 않습니다.")
