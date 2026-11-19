"""Chart tables preserve supplied values and uncertainty, without risk scoring."""
import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from core.bioequivalence import DEFAULT_DISSOLUTION_PROFILE, calculate_f2
from core.qsar import assess_mutagenicity
from presentation.data import (
    coverage_matrix,
    dissolution_comparison,
    dissolution_differences,
    evidence_kind_counts,
    finding_pareto,
    quantile_summary,
    screening_rule_table,
    source_state_counts,
)


@pytest.mark.parametrize("helper,columns", [
    (finding_pareto, ["항목", "건수", "누적 비율"]),
    (coverage_matrix, ["영역", "필드", "검토 상태", "상태 코드"]),
    (evidence_kind_counts, ["근거 유형", "건수"]),
    (source_state_counts, ["출처 구분", "문서 수"]),
    (screening_rule_table, ["규칙", "일치 수", "일치 원자 수"]),
    (dissolution_comparison, ["시간 (분)", "시료", "평균 용출 (%)", "표준편차", "하한", "상한"]),
    (dissolution_differences, ["시간 (분)", "차이 (pp)"]),
    (quantile_summary, ["백분위", "f2"]),
])
def test_unavailable_input_has_empty_schema_without_fabricated_zero_rows(helper, columns):
    for missing in (None, {}, []):
        result = helper(missing)
        assert list(result.columns) == columns
        assert result.empty


def test_pareto_uses_actual_finding_counts_not_summary_scores_or_severity():
    report = {"summary": {"possible_conflicts": 999}, "findings": [
        {"kind": "possible_conflict", "severity": 100},
        {"kind": "source_issue"}, {"kind": "source_issue"},
        {"kind": "future_kind"}, {"kind": "another_future_kind"}, {},
    ]}
    original = copy.deepcopy(report)
    result = finding_pareto(report)
    counts = result.set_index("항목")["건수"].to_dict()
    assert counts == {"기타 검토 항목": 2, "출처·연결 확인": 2, "가능한 불일치": 1, "유형 미기재": 1}
    assert result["건수"].sum() == 6
    assert result["건수"].tolist() == sorted(result["건수"], reverse=True)
    np.testing.assert_allclose(result["누적 비율"], result["건수"].cumsum() / 6)
    assert result["누적 비율"].iloc[-1] == 1.0
    assert report == original
    assert_frame_equal(result, finding_pareto(report))


def test_coverage_keeps_not_requested_unextracted_and_source_linked_distinct():
    report = {"coverage": [
        {"field_path": "ai_model.result", "review_state": "not_requested"},
        {"field_path": "nam_assay.result", "review_state": "not_extracted"},
        {"field_path": "supporting_evidence.pk_tk_evidence", "review_state": "reviewed_source_linked"},
        {"field_path": "product.route", "review_state": "all_candidates_rejected"},
        {"field_path": "future.field", "review_state": "future_state"}, {},
    ]}
    result = coverage_matrix(report)
    assert result["필드"].iloc[:5].tolist() == [row["field_path"] for row in report["coverage"][:5]]
    assert result["상태 코드"].iloc[:5].tolist() == [row["review_state"] for row in report["coverage"][:5]]
    assert result["검토 상태"].iloc[0] != result["검토 상태"].iloc[1]
    assert "유무 미확인" in result["검토 상태"].iloc[1]
    assert "타당성 미확인" in result["검토 상태"].iloc[2]
    assert "future_state" in result["검토 상태"].iloc[4]
    assert pd.isna(result["상태 코드"].iloc[5])
    assert pd.isna(result["필드"].iloc[5])
    assert not result["검토 상태"].str.contains("안전|검증 완료|과학적 검증").any()


@dataclass
class AssertionFixture:
    evidence_kind: str
    review_status: str = "제안됨"


@dataclass
class DocumentFixture:
    source_kind: str
    document_id: str = "same-recorded-id"


def test_evidence_kind_counts_include_all_supplied_claims_without_merging_prediction_with_observation():
    claims = [AssertionFixture("observation"), AssertionFixture("prediction", "거절"),
              {"evidence_kind": "context"}, {"evidence_kind": "unspecified"}, {},
              {"evidence_kind": "future_kind"}]
    original = copy.deepcopy(claims)
    result = evidence_kind_counts(claims)
    assert result.set_index("근거 유형")["건수"].to_dict() == {
        "관찰": 1, "예측": 1, "맥락": 1, "유형 미지정": 2, "기타 · future_kind": 1,
    }
    assert result["건수"].sum() == len(claims)
    assert "without review-status filtering" in result.attrs["count_basis"]
    assert claims == original


def test_source_counts_are_document_records_not_inferred_independent_streams():
    docs = [DocumentFixture("customer_evidence"), DocumentFixture("customer_evidence"),
            {"source_kind": "external_reference"}, {}, {"source_kind": "future_source"}]
    result = source_state_counts(docs)
    assert result.set_index("출처 구분")["문서 수"].to_dict() == {
        "고객 제공 자료": 2, "외부 참고자료": 1, "출처 미분류": 1, "기타 · future_source": 1,
    }
    assert result["문서 수"].sum() == 5
    assert "not independent" in result.attrs["count_basis"]


def test_screening_counts_distinct_atom_sets_and_union_without_historical_records():
    result = {"expert_alerts": [
        {"method": "Rule-based structural alert screening", "alert": "Synthetic rule", "matched_atoms": [[0, 1], [1, 0], [1, 2]]},
        {"method": "Rule-based structural alert screening", "alert": "Synthetic rule", "matched_atoms": [[1, 2]]},
        {"method": "Unverified local dossier", "alert": "Local historical record", "matched_atoms": [[7, 8]]},
        {"method": "Statistical (SAR)", "alert": "Old false model", "matched_atoms": [[9]]},
    ]}
    original = copy.deepcopy(result)
    assert screening_rule_table(result).to_dict("records") == [{"규칙": "Synthetic rule", "일치 수": 2, "일치 원자 수": 3}]
    assert result == original


def test_missing_or_malformed_match_data_remains_unavailable_not_zero():
    result = {"expert_alerts": [
        {"method": "Rule-based structural alert screening", "alert": "Missing matches"},
        {"method": "Rule-based structural alert screening", "alert": "Missing matches", "matched_atoms": [[0]]},
        {"method": "Rule-based structural alert screening", "alert": "Malformed matches", "matched_atoms": [[-1, True]]},
        {"method": "Rule-based structural alert screening", "alert": "Explicit empty matches", "matched_atoms": []},
    ]}
    rows = screening_rule_table(result).set_index("규칙")
    assert pd.isna(rows.loc["Missing matches", "일치 수"])
    assert pd.isna(rows.loc["Missing matches", "일치 원자 수"])
    assert pd.isna(rows.loc["Malformed matches", "일치 수"])
    assert rows.loc["Explicit empty matches", "일치 수"] == 0


def test_current_screening_result_keeps_local_dossier_out_of_rule_chart():
    assessment = assess_mutagenicity("Nc1ccccc1")
    result = screening_rule_table(assessment)
    assert result["규칙"].tolist() == ["Aromatic Amine"]
    assert result["일치 수"].tolist() == [1]
    assert result["일치 원자 수"].tolist() == [7]
    assert assessment["historical_evidence"]


def profile():
    return pd.DataFrame({
        "Time (min)": [5.0, 10.0, 20.0],
        "Reference Mean (%)": [2.0, 50.0, 98.0], "Reference SD": [5.0, 7.0, 6.0],
        "Test Mean (%)": [1.0, 55.0, 97.0], "Test SD": [4.0, 8.0, 7.0],
    })


def test_dissolution_chart_is_exact_mean_plus_minus_sample_sd_without_clipping():
    supplied = profile()
    original = supplied.copy(deep=True)
    result = dissolution_comparison(supplied)
    assert result["시료"].tolist() == ["기준품", "시험품"] * 3
    reference = result[result["시료"] == "기준품"]
    np.testing.assert_allclose(reference["평균 용출 (%)"], supplied["Reference Mean (%)"])
    np.testing.assert_allclose(reference["표준편차"], supplied["Reference SD"])
    assert reference["하한"].iloc[0] == -3.0
    assert reference["상한"].iloc[-1] == 104.0
    assert "표본 표준편차" in result.attrs["band_label"]
    assert "신뢰구간 아님" in result.attrs["band_label"]
    assert_frame_equal(supplied, original)


def test_dissolution_difference_is_test_minus_reference_in_percentage_points():
    result = dissolution_differences(profile())
    assert result.to_dict("records") == [
        {"시간 (분)": 5.0, "차이 (pp)": -1.0},
        {"시간 (분)": 10.0, "차이 (pp)": 5.0},
        {"시간 (분)": 20.0, "차이 (pp)": -1.0},
    ]


def test_missing_dissolution_values_are_not_dropped_or_filled_with_zero():
    supplied = profile()
    supplied.loc[1, "Reference SD"] = np.nan
    supplied.loc[2, "Test Mean (%)"] = np.nan
    result = dissolution_comparison(supplied)
    assert len(result) == 6
    row = result[(result["시료"] == "기준품") & (result["시간 (분)"] == 10.0)].iloc[0]
    assert row["평균 용출 (%)"] == 50.0
    assert pd.isna(row["표준편차"]) and pd.isna(row["하한"]) and pd.isna(row["상한"])
    assert pd.isna(dissolution_differences(supplied)["차이 (pp)"].iloc[-1])
    assert dissolution_comparison(supplied.drop(columns="Reference SD")).empty


def test_quantile_summary_uses_only_existing_computed_values_without_legacy_fallback_or_interpolation():
    supplied = {"simulation_interval_low": 61.123456, "simulation_p05": None,
                "simulation_median": 80.987654, "simulation_p95": np.inf,
                "simulation_interval_high": None, "ci_high": 99.0, "bootstrap_p95": 90.0}
    result = quantile_summary(supplied)
    assert result.to_dict("records") == [{"백분위": "2.5%", "f2": 61.123456}, {"백분위": "50%", "f2": 80.987654}]
    assert quantile_summary({"simulation_median": np.bool_(False)}).empty


def test_real_simulation_quantiles_are_plotted_exactly_in_requested_percentile_order():
    result = calculate_f2(DEFAULT_DISSOLUTION_PROFILE, simulation_runs=24, seed=1729, backend="python")
    actual = quantile_summary(result)
    assert actual["백분위"].tolist() == ["2.5%", "5%", "50%", "95%", "97.5%"]
    np.testing.assert_array_equal(actual["f2"], [result.simulation_interval_low, result.simulation_p05,
                                               result.simulation_median, result.simulation_p95, result.simulation_interval_high])
    assert_frame_equal(actual, quantile_summary(result.to_dict()))
