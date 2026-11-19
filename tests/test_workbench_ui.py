"""Visual workbench regressions preserve data, uncertainty and review state."""
from pathlib import Path
import copy
import json

import pandas as pd
import pyarrow as pa
from streamlit.testing.v1 import AppTest


APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def _charts(app):
    # Streamlit changed this element name; cover both supported public releases.
    return [*app.get("vega_lite_chart"), *app.get("arrow_vega_lite_chart")]


def _chart(app, alt_prefix):
    matches = [item for item in _charts(app) if item.proto.alt.startswith(alt_prefix)]
    assert len(matches) == 1
    return matches[0]


def _datasets(chart):
    return {
        item.name: pa.ipc.open_stream(item.data.data).read_all().to_pandas()
        for item in chart.proto.datasets
    }


def _mark_types(spec):
    if isinstance(spec, dict):
        mark = spec.get("mark")
        values = [mark.get("type") if isinstance(mark, dict) else mark] if mark else []
        return values + [item for value in spec.values() for item in _mark_types(value)]
    if isinstance(spec, list):
        return [item for value in spec for item in _mark_types(value)]
    return []


def test_dissolution_visuals_show_supplied_data_and_sd_toggle_preserves_calculation():
    app = AppTest.from_string("""
import streamlit as st
from core.bioequivalence import DEFAULT_DISSOLUTION_PROFILE, calculate_f2
from presentation.be_visuals import render_dissolution_insights
if 'visual_test_profile' not in st.session_state:
    st.session_state.visual_test_profile = DEFAULT_DISSOLUTION_PROFILE.copy()
if 'visual_test_result' not in st.session_state:
    st.session_state.visual_test_result = calculate_f2(st.session_state.visual_test_profile, simulation_runs=20, seed=41)
render_dissolution_insights(st.session_state.visual_test_profile, st.session_state.visual_test_result)
""").run(timeout=30)
    assert not app.exception
    before_profile = app.session_state["visual_test_profile"].copy(deep=True)
    before_result = copy.deepcopy(app.session_state["visual_test_result"].to_dict())
    assert len(_charts(app)) == 3
    curve = _chart(app, "입력된 기준품과 시험품")
    frame = next(frame for frame in _datasets(curve).values() if "평균 용출 (%)" in frame.columns)
    assert frame["평균 용출 (%)"].tolist() == [32.0, 30.0, 51.0, 49.0, 68.0, 66.0, 86.0, 84.0, 93.0, 91.0]
    assert "area" not in _mark_types(json.loads(curve.proto.spec))
    app.toggle(key="be_sd_band_toggle").set_value(True).run(timeout=30)
    assert not app.exception
    curve = _chart(app, "입력된 기준품과 시험품")
    assert "area" in _mark_types(json.loads(curve.proto.spec))
    frame = next(frame for frame in _datasets(curve).values() if "하한" in frame.columns)
    assert frame.iloc[0]["하한"] == 32.0 - 4.5
    assert frame.iloc[0]["상한"] == 32.0 + 4.5
    assert "신뢰구간이 아닙니다" in curve.proto.alt
    assert "임상 동등성 판정이 아닙니다" in _chart(app, "현재 입력의 정규모형").proto.alt
    pd.testing.assert_frame_equal(app.session_state["visual_test_profile"], before_profile)
    assert app.session_state["visual_test_result"].to_dict() == before_result


def test_dissolution_visuals_do_not_invent_simulation_when_no_result_exists():
    app = AppTest.from_string("""
from core.bioequivalence import DEFAULT_DISSOLUTION_PROFILE
from presentation.be_visuals import render_dissolution_insights
render_dissolution_insights(DEFAULT_DISSOLUTION_PROFILE)
""").run(timeout=30)
    assert not app.exception
    assert len(_charts(app)) == 2
    assert not any(item.proto.alt.startswith("현재 입력의 정규모형") for item in _charts(app))
    _chart(app, "입력된 기준품과 시험품")
    differences = _chart(app, "입력된 각 측정 시점")
    frame = next(frame for frame in _datasets(differences).values() if "차이 (pp)" in frame.columns)
    assert frame["차이 (pp)"].tolist() == [-2.0] * 5


def test_ntr_charts_count_recorded_claims_and_sources_without_upgrading_evidence():
    from nora.evidence import EvidenceAssertion, extract_document

    app = AppTest.from_file(APP).run(timeout=30)
    assert not app.exception
    app.radio(key="workspace").set_value("Toxicity Evidence Review").run(timeout=30)
    assert not app.exception
    documents, assertions = [], []
    fixtures = (
        ("customer_evidence", "context", "승인", "product.product_name", "Synthetic Visual Candidate"),
        ("external_reference", "prediction", "거절", "ai_model.result", "양성 / 위험 신호"),
        ("unclassified", "unspecified", "제안됨", "nam_assay.result", "음성"),
    )
    for number, (source_kind, evidence_kind, review_status, path, value) in enumerate(fixtures):
        document = extract_document(f"Synthetic claim {number}: {value}".encode(), f"visual-{number}.txt")
        document.source_kind = source_kind
        documents.append(document)
        assertions.append(EvidenceAssertion(
            assertion_id=f"AST-VISUAL-{number}", category="Synthetic visual fixture",
            field_path=path, label_ko="Synthetic", proposed_value=value, value_type="str",
            source_document_id=document.document_id, source_document_name=document.name,
            source_location=document.segments[0].location, source_excerpt=document.segments[0].text,
            source_segment_id=document.segments[0].segment_id, confidence=0.5,
            review_status=review_status, evidence_kind=evidence_kind, extracted_value=value,
        ))
    app.session_state["ntr_docs"] = documents
    app.session_state["ntr_assertions"] = assertions
    app.run(timeout=30)
    assert not app.exception
    kinds = _chart(app, "관찰형·예측형·맥락형·미지정")
    frame = next(frame for frame in _datasets(kinds).values() if "근거 유형" in frame.columns)
    assert dict(zip(frame["근거 유형"], frame["건수"])) == {"맥락": 1, "예측": 1, "유형 미지정": 1}
    assert "승인되거나 검증된 관찰 수가 아닙니다" in kinds.proto.alt
    sources = _chart(app, "고객 제공 자료·외부 참고자료·미분류")
    frame = next(frame for frame in _datasets(sources).values() if "출처 구분" in frame.columns)
    assert dict(zip(frame["출처 구분"], frame["문서 수"])) == {"고객 제공 자료": 1, "외부 참고자료": 1, "출처 미분류": 1}
    assert "독립 연구 수를 뜻하지 않습니다" in sources.proto.alt
    report = app.session_state["ntr_project_review"]
    coverage = _chart(app, "영역별 필드 검토 상태")
    frame = next(frame for frame in _datasets(coverage).values() if "필드수" in frame.columns)
    assert int(frame["필드수"].sum()) == len(report["coverage"])
    export = app.session_state["ntr_review_export"]
    assert export["result"] is None
    assert [item["review_status"] for item in export["assertions"]] == ["승인", "거절", "제안됨"]
    assert [item["evidence_kind"] for item in export["assertions"]] == ["context", "prediction", "unspecified"]
    assert [item["sha256"] for item in export["documents"]] == [document.sha256 for document in documents]


def _start_workbench():
    app = AppTest.from_file(APP).run(timeout=30)
    assert not app.exception
    return app


def _view(app, label):
    app.segmented_control(key="analysis_view").set_value(label).run(timeout=30)
    assert not app.exception
    return app


def test_default_workbench_separates_analysis_views_and_retains_the_review_flow():
    app = _start_workbench()
    assert app.segmented_control(key="analysis_view").value == "구조 신호"
    assert list(app.segmented_control(key="analysis_view").options) == ["구조 신호", "용출 비교", "프로젝트 요약"]
    assert not any(button.label == "Calculate f2 and simulation" for button in app.button)
    assert app.session_state["results"] is None
    flow = next(str(item.proto) for item in app.get("html") if "문제 정의" in str(item.proto))
    assert [flow.rindex(label) for label in ("문제 정의", "데이터 품질", "비교·차이", "시나리오", "검토")] == sorted(
        flow.rindex(label) for label in ("문제 정의", "데이터 품질", "비교·차이", "시나리오", "검토")
    )
    _view(app, "용출 비교")
    assert any(button.label == "Calculate f2 and simulation" for button in app.button)
    assert not any(item.proto.alt.startswith("현재 입력의 정규모형") for item in _charts(app))
    _view(app, "프로젝트 요약")
    assert any(metric.label == "용출 f2 · 기술 통계" and metric.value == "—" for metric in app.metric)
    assert not any(metric.value in {"Safe", "Low", "Acceptable"} for metric in app.metric)


def test_analysis_view_navigation_preserves_sar_be_inputs_settings_and_exports():
    app = _start_workbench()
    app.text_input(key="compound_name_input").set_value("Synthetic navigation candidate")
    app.text_area(key="smiles").set_value("c1ccc(cc1)[N+](=O)[O-]")
    next(button for button in app.button if button.label == "구조 신호 분석").click().run(timeout=30)
    assert not app.exception
    sar_before = copy.deepcopy(app.session_state["results"])
    package_before = copy.deepcopy(app.session_state["evidence_package"])
    _view(app, "용출 비교")
    app.number_input(key="be_simulation_runs").set_value(200)
    app.number_input(key="be_seed").set_value(271)
    next(button for button in app.button if button.label == "Calculate f2 and simulation").click().run(timeout=30)
    assert not app.exception
    result_before = copy.deepcopy(app.session_state["be_result"].to_dict())
    export_before = copy.deepcopy(app.session_state["be_export"])
    fingerprint_before = app.session_state["be_request_fingerprint"]
    _view(app, "프로젝트 요약")
    assert any(metric.label == "용출 f2 · 기술 통계" and float(metric.value) == result_before["f2"] for metric in app.metric)
    _view(app, "구조 신호")
    assert app.text_area(key="smiles").value == "c1ccc(cc1)[N+](=O)[O-]"
    assert app.text_input(key="compound_name_input").value == "Synthetic navigation candidate"
    assert app.session_state["results"] == sar_before
    assert app.session_state["evidence_package"] == package_before
    _view(app, "용출 비교")
    assert app.number_input(key="be_simulation_runs").value == 200
    assert app.number_input(key="be_seed").value == 271
    assert app.selectbox(key="be_backend").value == "Python"
    assert app.session_state["be_request_fingerprint"] == fingerprint_before
    assert app.session_state["be_result"].to_dict() == result_before
    assert app.session_state["be_export"] == export_before
    assert _chart(app, "현재 입력의 정규모형")
    assert app.session_state["be_export"]["clinical_be_established"] is False
    assert app.session_state["be_export"]["regulatory_acceptance"] == "Not established"
    app.radio(key="workspace").set_value("Toxicity Evidence Review").run(timeout=30)
    assert not app.exception
    assert app.session_state["be_result"].to_dict() == result_before
    app.radio(key="workspace").set_value("Strategy Dashboard").run(timeout=30)
    assert not app.exception
    _view(app, "용출 비교")
    assert app.number_input(key="be_simulation_runs").value == 200
    assert app.number_input(key="be_seed").value == 271
    assert app.session_state["be_result"].to_dict() == result_before
    assert app.session_state["be_export"] == export_before


def _screen_candidate(app, name="Synthetic source candidate"):
    app.text_input(key="compound_name_input").set_value(name)
    app.text_area(key="smiles").set_value("c1ccc(cc1)[N+](=O)[O-]")
    next(button for button in app.button if button.label == "구조 신호 분석").click().run(timeout=30)
    assert not app.exception
    return app


def test_summary_does_not_relabel_a_cached_sar_snapshot_for_another_compound():
    app = _screen_candidate(_start_workbench())
    snapshot = copy.deepcopy(app.session_state["results"])
    package = copy.deepcopy(app.session_state["evidence_package"])
    app.text_input(key="compound_name_input").set_value("Synthetic other candidate").run(timeout=30)
    _view(app, "프로젝트 요약")
    assert any(metric.label == "구조 규칙 일치" and metric.value == "—" for metric in app.metric)
    assert not any(item.proto.alt.startswith("근거 유형별 실제 기록 수") for item in _charts(app))
    assert app.session_state["results"] == snapshot
    assert app.session_state["evidence_package"] == package
    app.text_input(key="compound_name_input").set_value("Synthetic source candidate").run(timeout=30)
    assert any(metric.label == "구조 규칙 일치" and int(metric.value) == len(snapshot["expert_alerts"]) for metric in app.metric)


def test_project_switch_excludes_old_structural_context_and_retains_labeled_snapshot():
    app = _screen_candidate(_start_workbench())
    snapshot = copy.deepcopy(app.session_state["results"])
    package = copy.deepcopy(app.session_state["evidence_package"])
    original_project = app.session_state["assessment_project_id"]
    next(item for item in app.text_input if item.label == "Project ID").set_value("SYNTHETIC-OTHER-PROJECT").run(timeout=30)
    assert not app.exception
    assert any(item.value.startswith("분석 스냅샷:") and original_project in item.value for item in app.caption)
    _view(app, "프로젝트 요약")
    assert any(metric.label == "구조 규칙 일치" and metric.value == "—" for metric in app.metric)
    app.radio(key="workspace").set_value("Toxicity Evidence Review").run(timeout=30)
    assert not app.exception
    export = app.session_state["ntr_review_export"]
    assert export["structural_context"]["package"] == {}
    assert export["structural_context"]["project_id"] == "SYNTHETIC-OTHER-PROJECT"
    assert any("Structural context is excluded" in item.value for item in app.warning)
    assert app.session_state["results"] == snapshot
    assert app.session_state["evidence_package"] == package


def test_local_name_lookup_uses_pending_structure_without_external_requests():
    from unittest.mock import patch
    from core.compendial import get_local_smiles
    from core.qsar import canonicalize_smiles

    app = _start_workbench()
    app.text_input(key="compound_name_input").set_value("Atorvastatin").run(timeout=30)
    assert not app.checkbox(key="external_consent:Atorvastatin").value
    with patch("requests.get", side_effect=AssertionError("Local lookup must not call external services")) as network:
        next(button for button in app.button if button.label == "🔍 Search SMILES from Name").click().run(timeout=30)
    assert not app.exception
    assert network.call_count == 0
    expected = get_local_smiles("Atorvastatin")["smiles"]
    assert app.text_area(key="smiles").value == expected
    assert app.session_state["workbench_smiles"] == expected
    assert app.session_state["assessment_compound_name"] == "Atorvastatin"
    assert app.session_state["results"]["canonical_smiles"] == canonicalize_smiles(expected)
    assert app.session_state["results"]["class"] == "Not assigned (expert review required)"


def _indexed_molecule(mol):
    """Indexed graph identity, rather than canonical graph equivalence alone."""
    atoms = tuple((atom.GetSymbol(), atom.GetIsotope(), atom.GetFormalCharge(),
                   int(atom.GetChiralTag()), atom.GetIsAromatic()) for atom in mol.GetAtoms())
    bonds = tuple((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), str(bond.GetBondType()),
                   int(bond.GetStereo()), tuple(bond.GetStereoAtoms())) for bond in mol.GetBonds())
    return atoms, bonds


def _assert_draws_match_screening(draw, result):
    from rdkit import Chem

    expected = Chem.MolFromSmiles(result["canonical_smiles"])
    expected_atoms = sorted({index for alert in result["expert_alerts"]
                             for match in alert.get("matched_atoms", []) for index in match})
    assert draw.call_args_list
    for call in draw.call_args_list:
        assert _indexed_molecule(call.args[0]) == _indexed_molecule(expected)
        assert list(call.kwargs["highlight_atoms"]) == expected_atoms
    assert any(expected.GetAtomWithIdx(index).GetSymbol() == "Cl" for index in expected_atoms)


def test_molecule_renderer_uses_canonical_atom_indices_and_preserves_edited_input_snapshot():
    import re
    from unittest.mock import patch
    from xml.etree import ElementTree as ET
    from presentation import molecule

    # A spy wraps the real RDKit renderer; the UI receives actual drawing output.
    with patch.object(molecule, "render_molecule_svg", wraps=molecule.render_molecule_svg) as draw:
        app = _start_workbench()
        app.text_input(key="compound_name_input").set_value("Synthetic canonical-index fixture")
        app.text_area(key="smiles").set_value("ClCC")
        next(button for button in app.button if button.label == "구조 신호 분석").click().run(timeout=30)
        assert not app.exception
        result = copy.deepcopy(app.session_state["results"])
        package = copy.deepcopy(app.session_state["evidence_package"])
        assert result["canonical_smiles"] == "CCCl"
        _assert_draws_match_screening(draw, result)
        drawings = [match.group(0) for item in app.markdown
                    for match in re.finditer(r"<svg\b.*?</svg>", str(item.value), re.DOTALL)]
        assert drawings
        root = ET.fromstring(drawings[0])
        description = root.find("{http://www.w3.org/2000/svg}desc")
        assert description is not None
        assert "RDKit two-dimensional depiction" in description.text
        assert "Highlighted submitted atom indices: 1, 2" in description.text
        assert {item.get("class") for item in root.iter() if item.tag.endswith("ellipse")} >= {"atom-1", "atom-2"}
        for changed_structure in ("N#CC", "not-a-smiles"):
            draw.reset_mock()
            app.text_area(key="smiles").set_value(changed_structure).run(timeout=30)
            assert not app.exception
            assert app.session_state["results"] == result
            assert app.session_state["evidence_package"] == package
            assert app.text_area(key="smiles").value == changed_structure
            assert any("마지막 분석의 구조 스냅샷" in item.value for item in app.warning)
            assert any(item.value.startswith("분석 스냅샷:") and "Synthetic canonical-index fixture" in item.value
                       for item in app.caption)
            _assert_draws_match_screening(draw, result)


def test_noncanonical_degradant_drawing_binds_highlights_without_rewriting_raw_provenance():
    from unittest.mock import patch
    from presentation import molecule
    from core.qsar import assess_mutagenicity

    app = _screen_candidate(_start_workbench())
    assessed = assess_mutagenicity("ClCC", compound_name="Synthetic drawing fixture")
    product = {
        "name": "Synthetic noncanonical drawing fixture", "smiles": "ClCC",
        "pathway": "Synthetic rendering fixture", "condition": "Synthetic test state",
        "class": assessed["class"], "status": assessed["status"],
        "structural_explanation": assessed["structural_explanation"],
        "evidence_objects": assessed["evidence_objects"],
    }
    before = copy.deepcopy(product)
    app.session_state["degradants"] = [product]
    with patch.object(molecule, "render_molecule_svg", wraps=molecule.render_molecule_svg) as draw:
        app.run(timeout=30)
    assert not app.exception
    calls = [call for call in draw.call_args_list if call.kwargs.get("width") == 480 and call.kwargs.get("height") == 260]
    assert len(calls) == 1
    from rdkit import Chem
    assert _indexed_molecule(calls[0].args[0]) == _indexed_molecule(Chem.MolFromSmiles(assessed["canonical_smiles"]))
    assert calls[0].kwargs["highlight_atoms"] == sorted({index for alert in assessed["expert_alerts"]
                                                       for match in alert["matched_atoms"] for index in match})
    assert app.session_state["degradants"][0] == before
    assert app.session_state["degradants"][0]["smiles"] == "ClCC"
