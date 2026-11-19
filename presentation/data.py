"""Pure tables for evidence and dissolution views.

These helpers summarize supplied records, not independent evidence streams or
biomedical risk. Missing input stays unavailable. They neither run assessment
engines nor infer safety, scientific validity, or regulatory acceptance.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd
import numpy as np


FINDING_LABELS = {
    "possible_conflict": "가능한 불일치",
    "scope_difference": "범위 차이",
    "coverage_gap": "근거 연결·검토 공백",
    "source_issue": "출처·연결 확인",
    "scope_review": "비교 범위 미정",
    "possible_value_difference": "값 차이 검토",
    "prediction_observation_difference": "예측·관찰 차이",
    "project_scope_mismatch": "프로젝트 범위 차이",
    "out_of_scope": "상세 평가 범위 밖",
    "extraction_limit": "추출 한계",
    "review_limit": "비교 한계",
    "input_issue": "입력 형식 확인",
}
COVERAGE_LABELS = {
    "not_requested": "요청 범위 밖",
    "not_extracted": "미추출 · 근거 유무 미확인",
    "unextracted": "미추출 · 근거 유무 미확인",
    "all_candidates_rejected": "후보 모두 거절 · 근거 공백 미확정",
    "human_review_pending": "사람 검토 대기",
    "source_review_required": "출처·연결 검토 필요",
    "observation_review_required": "관찰 유형 검토 필요",
    "prediction_review_required": "예측 유형 검토 필요",
    "value_review_required": "값·형식 검토 필요",
    "scope_review_required": "범위 연결 검토 필요",
    "conflicting_reviewed_values": "검토된 값 차이 · 해소 필요",
    "reviewed_value_difference": "검토된 값 차이 · 해소 필요",
    "manual_unlinked": "수동 입력 · 출처 미연결",
    "reviewed_source_linked": "검토·출처 연결 (과학적 타당성 미확인)",
}
REGION_LABELS = {
    "context_of_use": "문제 정의",
    "product": "제품·후보",
    "ai_model": "AI 모델",
    "nam_assay": "NAM 자료",
    "supporting_evidence": "보조 근거",
}
EVIDENCE_KIND_LABELS = {"observation": "관찰", "prediction": "예측", "context": "맥락", "unspecified": "유형 미지정"}
SOURCE_KIND_LABELS = {"customer_evidence": "고객 제공 자료", "external_reference": "외부 참고자료", "unclassified": "출처 미분류"}


def _mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return None


def _records(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping, pd.DataFrame)):
        return []
    return [record for item in value if (record := _mapping(item)) is not None]


def _code(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value: Any) -> float:
    """Unavailable plot values are NaN, never a replacement zero."""
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def finding_pareto(report: Any) -> pd.DataFrame:
    """Count actual finding records; cumulative proportions are fractions 0..1."""
    columns = ["항목", "건수", "누적 비율"]
    report = _mapping(report)
    records = _records(report.get("findings")) if report is not None else []
    counts: Counter[str] = Counter()
    for record in records:
        kind = _code(record.get("kind"))
        label = FINDING_LABELS.get(kind, "기타 검토 항목" if kind else "유형 미기재")
        counts[label] += 1
    total = sum(counts.values())
    if not total:
        return pd.DataFrame(columns=columns)
    cumulative = 0
    rows = []
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        cumulative += count
        rows.append({"항목": label, "건수": count, "누적 비율": cumulative / total})
    table = pd.DataFrame(rows, columns=columns)
    table.attrs["count_basis"] = "Supplied finding records; not severity, independent sources, or biomedical risk."
    return table


def coverage_matrix(report: Any) -> pd.DataFrame:
    """Translate review states while retaining original field paths and codes."""
    columns = ["영역", "필드", "검토 상태", "상태 코드"]
    report = _mapping(report)
    records = _records(report.get("coverage")) if report is not None else []
    rows = []
    for record in records:
        path, state = _code(record.get("field_path")), _code(record.get("review_state"))
        region = path.partition(".")[0]
        rows.append({
            "영역": REGION_LABELS.get(region, "기타 영역" if region else "영역 미기재"),
            "필드": path or pd.NA,
            "검토 상태": COVERAGE_LABELS.get(state, f"기타 검토 상태 · {state}" if state else "상태 미기재"),
            "상태 코드": state or pd.NA,
        })
    table = pd.DataFrame(rows, columns=columns)
    table.attrs["meaning"] = "Review/source linkage only; not safety, scientific validity, or complete evidence coverage."
    return table


def _category_counts(records: Any, key: str, labels: dict[str, str], default: str, columns: list[str]) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    for record in _records(records):
        code = _code(record.get(key)) or default
        counts[labels.get(code, f"기타 · {code}")] += 1
    return pd.DataFrame(
        [{columns[0]: label, columns[1]: count} for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))],
        columns=columns,
    )


def evidence_kind_counts(assertions: Any) -> pd.DataFrame:
    """Count all supplied assertions, preserving observation/prediction/context."""
    table = _category_counts(assertions, "evidence_kind", EVIDENCE_KIND_LABELS, "unspecified", ["근거 유형", "건수"])
    table.attrs["count_basis"] = "Supplied assertion records, without review-status filtering; not independent evidence streams."
    return table


def source_state_counts(documents: Any) -> pd.DataFrame:
    """Count supplied document records by stated source class, not independence."""
    table = _category_counts(documents, "source_kind", SOURCE_KIND_LABELS, "unclassified", ["출처 구분", "문서 수"])
    table.attrs["count_basis"] = "Supplied document records; not independent evidence streams or authenticated sources."
    return table


def screening_rule_table(result: Any) -> pd.DataFrame:
    """Show unique atom-set matches per implemented rule in the submitted molecule.

    Historical dossier records are excluded. Match count counts distinct atom
    index sets; atom count is the union of those sets, not summed match lengths.
    Missing match data is unavailable rather than a zero-match conclusion.
    """
    columns = ["규칙", "일치 수", "일치 원자 수"]
    result = _mapping(result)
    if result is None:
        return pd.DataFrame(columns=columns)
    candidates = result.get("expert_alerts") if "expert_alerts" in result else result.get("alerts")
    matches_by_rule: dict[str, set[tuple[int, ...]] | None] = {}
    unavailable_rules = set()
    for alert in _records(candidates):
        if _code(alert.get("method")) not in {"Rule-based structural alert screening", "Expert Rule-based"}:
            continue
        name = _code(alert.get("alert"))
        if not name:
            continue
        raw_matches = alert.get("matched_atoms")
        if not isinstance(raw_matches, (list, tuple)):
            unavailable_rules.add(name)
            matches_by_rule.setdefault(name, None)
            continue
        valid_matches = set()
        invalid = False
        for match in raw_matches:
            if not isinstance(match, (list, tuple)) or not match or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in match):
                invalid = True
                break
            valid_matches.add(tuple(sorted(set(match))))
        if invalid:
            unavailable_rules.add(name)
            matches_by_rule.setdefault(name, None)
        else:
            existing = matches_by_rule.get(name)
            if existing is None:
                matches_by_rule[name] = valid_matches
            else:
                existing.update(valid_matches)
    rows = []
    for name, matches in matches_by_rule.items():
        if name in unavailable_rules:
            matches = None
        rows.append({"규칙": name, "일치 수": len(matches) if matches is not None else pd.NA,
                     "일치 원자 수": len({atom for match in matches for atom in match}) if matches is not None else pd.NA})
    return pd.DataFrame(rows, columns=columns)


def dissolution_comparison(profile: Any) -> pd.DataFrame:
    """Reshape entered means ± sample SD; bands are neither clipped nor CIs."""
    columns = ["시간 (분)", "시료", "평균 용출 (%)", "표준편차", "하한", "상한"]
    required = ["Time (min)", "Reference Mean (%)", "Reference SD", "Test Mean (%)", "Test SD"]
    if not isinstance(profile, pd.DataFrame) or profile.empty or not all(column in profile.columns for column in required) or profile.columns.duplicated().any():
        return pd.DataFrame(columns=columns)
    rows = []
    for _, row in profile.iterrows():
        for arm, label in (("Reference", "기준품"), ("Test", "시험품")):
            mean, sd = _number(row[f"{arm} Mean (%)"]), _number(row[f"{arm} SD"])
            rows.append({"시간 (분)": _number(row["Time (min)"]), "시료": label, "평균 용출 (%)": mean,
                         "표준편차": sd, "하한": mean - sd, "상한": mean + sd})
    table = pd.DataFrame(rows, columns=columns)
    table.attrs["band_label"] = "입력 요약: 평균 ± 표본 표준편차 (신뢰구간 아님)"
    table.attrs["input_validation"] = "Presentation only; pass a backend-validated profile. Missing values are not imputed."
    return table


def dissolution_differences(profile: Any) -> pd.DataFrame:
    """Entered summary difference in percentage points: Test minus Reference."""
    columns = ["시간 (분)", "차이 (pp)"]
    required = ["Time (min)", "Reference Mean (%)", "Test Mean (%)"]
    if not isinstance(profile, pd.DataFrame) or profile.empty or not all(column in profile.columns for column in required) or profile.columns.duplicated().any():
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([
        {"시간 (분)": _number(row["Time (min)"]), "차이 (pp)": _number(row["Test Mean (%)"]) - _number(row["Reference Mean (%)"])}
        for _, row in profile.iterrows()
    ], columns=columns)


def quantile_summary(result: Any) -> pd.DataFrame:
    """Use only available computed simulation quantiles; invent no distribution."""
    columns = ["백분위", "f2"]
    result = _mapping(result)
    if result is None:
        return pd.DataFrame(columns=columns)
    fields = (("2.5%", "simulation_interval_low"), ("5%", "simulation_p05"), ("50%", "simulation_median"),
              ("95%", "simulation_p95"), ("97.5%", "simulation_interval_high"))
    rows = []
    for label, key in fields:
        value = _number(result.get(key))
        if math.isfinite(value):
            rows.append({"백분위": label, "f2": value})
    table = pd.DataFrame(rows, columns=columns)
    table.attrs["meaning"] = "Computed simulation percentiles, not an invented histogram or calibrated confidence interval."
    return table
