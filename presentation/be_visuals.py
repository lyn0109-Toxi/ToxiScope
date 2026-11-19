"""Visual summaries of supplied dissolution data and actual simulation quantiles.

This module does not estimate a distribution, alter a profile, or decide
bioequivalence. Numerical and input checks stay in the calculation layer;
scientific validity is not established by either layer.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .data import dissolution_comparison, dissolution_differences, quantile_summary


INK = "#172B3A"
CYAN = "#168C94"
AMBER = "#B77824"
SLATE = "#8796A5"
_SERIES = ["기준품", "시험품"]
_PERCENTILES = ["2.5%", "5%", "50%", "95%", "97.5%"]


def _finite(frame: pd.DataFrame, columns: list[str]) -> bool:
    """Avoid drawing a partial profile as if missing input were complete."""
    if frame.empty or any(column not in frame for column in columns):
        return False
    try:
        return bool(np.isfinite(frame[columns].to_numpy(dtype=float)).all())
    except (TypeError, ValueError):
        return False


def _finish(chart, *, height: int):
    return (
        chart.properties(height=height)
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=INK,
            titleColor=INK,
            labelFontSize=11,
            titleFontSize=11,
            titlePadding=10,
            gridColor="#E7EDF0",
            domainColor="#CDD7DD",
            tickColor="#CDD7DD",
        )
        .configure_legend(
            orient="top",
            titleColor=INK,
            labelColor=INK,
            labelFontSize=11,
            titleFontSize=11,
            symbolStrokeWidth=3,
            padding=4,
        )
    )


def build_dissolution_curves(profile_df, *, show_sd: bool = False):
    """Return paired entered means, optionally with raw mean ± sample SD bands."""
    frame = dissolution_comparison(profile_df)
    columns = ["시간 (분)", "평균 용출 (%)"]
    if show_sd:
        columns += ["표준편차", "하한", "상한"]
    if not _finite(frame, columns):
        return None
    color = alt.Color(
        field="시료", type="nominal", title="입력 요약",
        scale=alt.Scale(domain=_SERIES, range=[CYAN, AMBER]),
        sort=_SERIES,
    )
    tooltip = [
        alt.Tooltip(field="시료", type="nominal", title="시료"),
        alt.Tooltip(field="시간 (분)", type="quantitative", title="시간 (분)", format="g"),
        alt.Tooltip(field="평균 용출 (%)", type="quantitative", title="평균 용출 (%)", format=".2f"),
        alt.Tooltip(field="표준편차", type="quantitative", title="표본 SD (pp)", format=".2f"),
    ]
    base = alt.Chart(frame).encode(
        x=alt.X(field="시간 (분)", type="quantitative", title="시간 (분)", scale=alt.Scale(zero=False)),
        color=color,
    )
    line = base.mark_line(strokeWidth=2.8, point=alt.OverlayMarkDef(size=45, filled=True)).encode(
        y=alt.Y(field="평균 용출 (%)", type="quantitative", title="평균 용출 (%)", scale=alt.Scale(zero=True)),
        order=alt.Order(field="시간 (분)", type="quantitative"),
        tooltip=tooltip,
    )
    if show_sd:
        band = base.mark_area(opacity=0.12).encode(
            y=alt.Y(field="하한", type="quantitative", title="평균 용출 (%)", scale=alt.Scale(zero=True), stack=None),
            y2=alt.Y2(field="상한"),
            tooltip=tooltip + [
                alt.Tooltip(field="하한", type="quantitative", title="평균 − SD (%)", format=".2f"),
                alt.Tooltip(field="상한", type="quantitative", title="평균 + SD (%)", format=".2f"),
            ],
        )
        chart = band + line
    else:
        chart = line
    return _finish(chart, height=250)


def build_dissolution_differences(profile_df):
    """Return the signed Test minus Reference difference at actual sample times."""
    frame = dissolution_differences(profile_df)
    if not _finite(frame, ["시간 (분)", "차이 (pp)"]):
        return None
    frame = frame.copy()
    frame["방향"] = np.select(
        [frame["차이 (pp)"] > 0, frame["차이 (pp)"] < 0],
        ["시험품 높음", "시험품 낮음"], default="동일",
    )
    bars = alt.Chart(frame).mark_bar(size=19, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X(field="시간 (분)", type="quantitative", title="시간 (분)", scale=alt.Scale(zero=False)),
        y=alt.Y(field="차이 (pp)", type="quantitative", title="시험품 − 기준품 (pp)", scale=alt.Scale(zero=True)),
        color=alt.Color(
            field="방향", type="nominal", title="입력 평균의 차이",
            scale=alt.Scale(domain=["시험품 높음", "시험품 낮음", "동일"], range=[AMBER, CYAN, SLATE]),
        ),
        tooltip=[
            alt.Tooltip(field="시간 (분)", type="quantitative", title="시간 (분)", format="g"),
            alt.Tooltip(field="차이 (pp)", type="quantitative", title="시험품 − 기준품 (pp)", format="+.2f"),
            alt.Tooltip(field="방향", type="nominal", title="방향"),
        ],
    )
    zero = alt.Chart(pd.DataFrame({"기준 차이": [0]})).mark_rule(color=SLATE, strokeWidth=1).encode(
        y=alt.Y(field="기준 차이", type="quantitative"),
        tooltip=[alt.Tooltip(field="기준 차이", type="quantitative", title="같은 평균 (pp)")],
    )
    return _finish(bars + zero, height=250)


def build_simulation_quantiles(result):
    """Return an interval glyph from all five actual reported quantiles only."""
    frame = quantile_summary(result)
    if not _finite(frame, ["f2"]) or "백분위" not in frame:
        return None
    if len(frame) != 5 or set(frame["백분위"]) != set(_PERCENTILES):
        return None
    values = frame.set_index("백분위").loc[_PERCENTILES, "f2"].to_numpy(dtype=float)
    if np.any(np.diff(values) < 0):
        return None
    intervals = pd.DataFrame({
        "하한": [values[0], values[1]],
        "상한": [values[4], values[3]],
        "구간": ["2.5–97.5 백분위", "5–95 백분위"],
        "순서": [0, 1],
    })
    interval_marks = alt.Chart(intervals).mark_rule(strokeCap="round").encode(
        x=alt.X(field="하한", type="quantitative", title="f₂ · 현재 입력의 정규모형 시뮬레이션", scale=alt.Scale(zero=False)),
        x2=alt.X2(field="상한"),
        y=alt.value(34),
        color=alt.Color(
            field="구간", type="nominal", title="시뮬레이션 요약",
            scale=alt.Scale(domain=["2.5–97.5 백분위", "5–95 백분위"], range=[SLATE, CYAN]),
            sort=["2.5–97.5 백분위", "5–95 백분위"],
        ),
        strokeWidth=alt.StrokeWidth(
            field="구간", type="nominal",
            scale=alt.Scale(domain=["2.5–97.5 백분위", "5–95 백분위"], range=[3, 9]), legend=None,
        ),
        order=alt.Order(field="순서", type="quantitative"),
        tooltip=[
            alt.Tooltip(field="구간", type="nominal", title="요약 구간"),
            alt.Tooltip(field="하한", type="quantitative", title="하한 f₂", format=".4f"),
            alt.Tooltip(field="상한", type="quantitative", title="상한 f₂", format=".4f"),
        ],
    )
    points = alt.Chart(frame).mark_tick(color=INK, orient="vertical", size=17, thickness=1.2).encode(
        x=alt.X(field="f2", type="quantitative", scale=alt.Scale(zero=False)), y=alt.value(34),
        tooltip=[alt.Tooltip(field="백분위", type="nominal", title="백분위"),
                 alt.Tooltip(field="f2", type="quantitative", title="실제 요약 f₂", format=".4f")],
    )
    median = alt.Chart(frame[frame["백분위"] == "50%"]).mark_point(
        color=AMBER, shape="diamond", size=105, filled=True,
    ).encode(
        x=alt.X(field="f2", type="quantitative", scale=alt.Scale(zero=False)), y=alt.value(34),
        tooltip=[alt.Tooltip(field="백분위", type="nominal", title="중앙값 백분위"),
                 alt.Tooltip(field="f2", type="quantitative", title="중앙값 f₂", format=".4f")],
    )
    reference = alt.Chart(pd.DataFrame({"서술적 참조": [50]})).mark_rule(
        color=AMBER, strokeDash=[4, 4], strokeWidth=1.2,
    ).encode(
        x=alt.X(field="서술적 참조", type="quantitative", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip(field="서술적 참조", type="quantitative", title="f₂ 50 · 서술적 참조선")],
    )
    return _finish(interval_marks + points + median + reference, height=88)


def render_dissolution_insights(profile_df, result=None) -> None:
    """Render a compact comparison with no inference beyond supplied summaries."""
    show_sd = st.toggle(
        "표준편차 띠 표시", value=False, key="be_sd_band_toggle",
        help="입력 평균 ± 표본 SD입니다. 신뢰구간이 아니며 0–100%로 잘라내지 않습니다.",
    )
    curves = build_dissolution_curves(profile_df, show_sd=show_sd)
    differences = build_dissolution_differences(profile_df)
    if curves is None or differences is None:
        st.info("비교할 입력 요약을 먼저 확인하세요.")
        return
    left, right = st.columns([1.15, 1], gap="medium")
    with left:
        st.markdown("#### 평균 용출 곡선")
        st.altair_chart(
            curves, key="be_comparison_chart", width="stretch",
            alt="입력된 기준품과 시험품의 시간별 평균 용출률. 선택한 띠는 평균 더하기 빼기 표본 표준편차이며 신뢰구간이 아닙니다.",
        )
    with right:
        st.markdown("#### 시간별 차이")
        st.altair_chart(
            differences, key="be_difference_chart", width="stretch",
            alt="입력된 각 측정 시점에서 시험품 평균에서 기준품 평균을 뺀 차이. 단위는 퍼센트포인트이며 0선은 같은 평균을 뜻합니다.",
        )
    if show_sd:
        st.caption("띠 = 평균 ± 표본 SD · 신뢰구간 아님 · 원래 범위 유지")
    if result is not None:
        quantiles = build_simulation_quantiles(result)
        if quantiles is not None:
            st.markdown("#### 시뮬레이션 범위")
            st.altair_chart(
                quantiles, key="be_simulation_chart", width="stretch",
                alt="현재 입력의 정규모형 시뮬레이션에서 보고된 2.5, 5, 50, 95, 97.5 백분위 f2. 점선 50은 서술적 참조이며 임상 동등성 판정이 아닙니다.",
            )
            st.caption("마름모 = 중앙값 · 점선 50 = 서술적 참조 · 신뢰구간·임상 동등성 확률 아님")
