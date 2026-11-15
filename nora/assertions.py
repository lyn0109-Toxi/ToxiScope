from __future__ import annotations

import copy
import math
import re
import uuid
from dataclasses import fields, is_dataclass
from typing import Any, Iterable, get_args, get_origin, get_type_hints

from .evidence import EVIDENCE_KIND_OPTIONS, SCOPE_KEYS, DocumentRecord, EvidenceAssertion, SourceSegment
from .models import AssessmentInput


REVIEWED_STATUSES = {"승인", "수정"}
REVIEW_STATUS_OPTIONS = ["제안됨", "승인", "수정", "거절"]


def _new_id() -> str:
    return f"AST-{uuid.uuid4().hex[:12]}"


def _excerpt(text: str, start: int, end: int, radius: int = 230) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return re.sub(r"\s+", " ", text[left:right]).strip()


def _clean_value(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip(" \t\n\r;,.|-")
    return value[:220]


def _add(
    assertions: list[EvidenceAssertion],
    document: DocumentRecord,
    segment: SourceSegment,
    *,
    category: str,
    field_path: str,
    label_ko: str,
    proposed_value: str,
    value_type: str = "str",
    confidence: float = 0.75,
    match_span: tuple[int, int] | None = None,
    evidence_kind: str | None = None,
    scope: dict[str, str] | None = None,
) -> None:
    value = _clean_value(proposed_value)
    if not value:
        return
    start, end = match_span or (0, min(len(segment.text), 320))
    excerpt = _excerpt(segment.text, start, end)
    scope = dict(scope or {})
    kind = evidence_kind or (
        "prediction" if field_path.startswith("ai_model.")
        else "context" if field_path.startswith(("product.", "context_of_use."))
        else "unspecified"
    )
    # Retain explicit scalar repetitions so same-document contradictions remain
    # visible. Merge only identical list candidates from the same cited location.
    key = (field_path, value.lower(), document.document_id, segment.segment_id, segment.location, excerpt, kind, tuple(sorted(scope.items())))
    if value_type == "list" and any(
        (item.field_path, item.proposed_value.lower(), item.source_document_id, item.source_segment_id,
         item.source_location, item.source_excerpt, item.evidence_kind,
         tuple(sorted(item.scope.items()))) == key for item in assertions
    ):
        return
    assertions.append(
        EvidenceAssertion(
            assertion_id=_new_id(),
            category=category,
            field_path=field_path,
            label_ko=label_ko,
            proposed_value=value,
            value_type=value_type,
            source_document_id=document.document_id,
            source_document_name=document.name,
            source_location=segment.location,
            source_excerpt=excerpt,
            confidence=round(max(0.0, min(1.0, confidence)), 2),
            evidence_kind=kind,
            scope=scope,
            extracted_value=value,
            source_segment_id=segment.segment_id,
        )
    )


def _first_regex(
    assertions: list[EvidenceAssertion],
    document: DocumentRecord,
    segment: SourceSegment,
    pattern: str,
    *,
    category: str,
    field_path: str,
    label_ko: str,
    value_type: str = "str",
    confidence: float = 0.92,
    flags: int = re.IGNORECASE,
    transform: Any = None,
) -> None:
    # Keep the historical helper name for callers, but retain every explicit
    # value rather than silently hiding later study values in the same segment.
    for match in re.finditer(pattern, segment.text, flags=flags):
        value = match.group(1) if match.lastindex else match.group(0)
        if transform:
            value = transform(value)
        _add(
            assertions, document, segment,
            category=category, field_path=field_path, label_ko=label_ko,
            proposed_value=str(value), value_type=value_type,
            confidence=confidence, match_span=(match.start(), match.end()),
        )



_NEGATION_RE = re.compile(
    r"(?:\b(?:not|no|without|never|neither|nor|absent|unavailable|missing)\b"
    r"|\b(?:did|does|do|was|were|is|are|has|have|had)\s+not\b"
    r"|not\s+(?:available|measured|included|performed|reported|demonstrated|assessed|validated)"
    r"|없(?:음|다|었)|미(?:측정|포함|수행|보고|평가|검증)|않(?:음|았다|은|는)"
    r"|확인되지|포함되지|수행되지|측정되지|보고되지|제공되지|검증되지)",
    flags=re.IGNORECASE,
)


def _is_negated(text: str, start: int, end: int) -> bool:
    """Detect simple sentence-level negation around a candidate fact.

    The function is intentionally conservative: a negated or missing statement
    is not converted into a positive Evidence Assertion. The source sentence
    remains available for human review in the document workspace.
    """
    left_boundary = max(text.rfind(".", 0, start), text.rfind("\n", 0, start), text.rfind(";", 0, start))
    right_candidates = [pos for pos in (text.find(".", end), text.find("\n", end), text.find(";", end)) if pos >= 0]
    right_boundary = min(right_candidates) if right_candidates else min(len(text), end + 180)
    window = text[max(0, left_boundary + 1, start - 120): min(len(text), right_boundary + 1, end + 180)]
    return bool(_NEGATION_RE.search(window))


def _keyword_assertion(
    assertions: list[EvidenceAssertion],
    document: DocumentRecord,
    segment: SourceSegment,
    *,
    patterns: Iterable[str],
    category: str,
    field_path: str,
    label_ko: str,
    proposed_value: str,
    value_type: str = "str",
    confidence: float = 0.72,
    allow_negated: bool = False,
) -> None:
    for pattern in patterns:
        for match in re.finditer(pattern, segment.text, flags=re.IGNORECASE):
            if not allow_negated and _is_negated(segment.text, match.start(), match.end()):
                continue
            _add(
                assertions,
                document,
                segment,
                category=category,
                field_path=field_path,
                label_ko=label_ko,
                proposed_value=proposed_value,
                value_type=value_type,
                confidence=confidence,
                match_span=(match.start(), match.end()),
            )


def _result_assertions(assertions: list[EvidenceAssertion], document: DocumentRecord, segment: SourceSegment) -> None:
    """Propose explicitly attributed results; leave ambiguous source text for review."""
    role_pattern = re.compile(r"\bAI\b|\bmodel\b|모델|\bNAM\b|\bassay\b|시험", re.IGNORECASE)
    prediction_pattern = re.compile(r"\b(?:prediction|predicted)\b|예측", re.IGNORECASE)
    output_pattern = re.compile(r"\b(?:prediction|predicted|result)\b|예측|결과", re.IGNORECASE)
    outcome_pattern = re.compile(
        r"(?P<negative>\bnegative\b|음성|\blow\s+risk\b|낮은\s*위험)"
        r"|(?P<positive>\bpositive\b|양성|\bhigh\s+risk\b|높은\s*위험)", re.IGNORECASE,
    )
    for clause in re.finditer(r"[^\n;.!?]+", segment.text):
        text = clause.group()
        outcomes = list(outcome_pattern.finditer(text))
        for index, outcome in enumerate(outcomes):
            if re.match(r"\s*(?:control|대조)", text[outcome.end():], re.IGNORECASE):
                continue
            preceding = text[:outcome.start()]
            roles = list(role_pattern.finditer(preceding))
            if roles:
                context = roles[-1]
                role = "nam" if re.fullmatch(r"NAM|assay|시험", context.group(), re.IGNORECASE) else "ai"
                relevant = preceding[context.start():]
            elif prediction_pattern.search(preceding):
                role, relevant = "ai", preceding
            else:
                continue
            if not output_pattern.search(relevant) or re.search(r"\bcontrol\b|대조", relevant, re.IGNORECASE):
                continue
            preceding_result_end = outcomes[index - 1].end() if index else 0
            if role == "nam" and prediction_pattern.search(preceding[preceding_result_end:]):
                continue
            if _is_negated(segment.text, clause.start(), clause.start() + outcome.end()):
                continue
            negative = outcome.group("negative") is not None
            _add(
                assertions, document, segment,
                category="AI 모델" if role == "ai" else "NAM 시험",
                field_path="ai_model.result" if role == "ai" else "nam_assay.result",
                label_ko="AI 예측 결과" if role == "ai" else "NAM 결과",
                proposed_value=("음성 / 낮은 위험 예측" if negative else "양성 / 위험 신호") if role == "ai" else ("음성" if negative else "양성"),
                confidence=0.80 if role == "ai" else 0.78,
                match_span=(clause.start(), clause.start() + outcome.end()),
            )


def extract_assertions(document: DocumentRecord) -> list[EvidenceAssertion]:
    """Create source-linked, reviewable candidate assertions from one document.

    The extractor is deliberately conservative. It proposes structured facts; it does
    not make an Evidence Role decision and does not mark assertions as accepted.
    """
    assertions: list[EvidenceAssertion] = []

    for segment in document.segments:
        text = segment.text

        # Product and development context
        _first_regex(assertions, document, segment, r"(?:제품명|후보물질명|Product\s*Name|Candidate)\s*[:：]\s*([^\n;|]{2,100})", category="제품", field_path="product.product_name", label_ko="후보물질명")
        _first_regex(assertions, document, segment, r"(?:유효성분|활성성분|Active\s*(?:Substance|Ingredient)|Sequence)\s*[:：]\s*([^\n;|]{2,130})", category="제품", field_path="product.active_substance", label_ko="유효성분/서열")
        _first_regex(assertions, document, segment, r"(?:표적|Target|Mechanism(?:\s*of\s*Action)?|작용기전)\s*[:：]\s*([^\n;|]{2,150})", category="제품", field_path="product.target_mechanism", label_ko="표적·작용기전")
        _first_regex(assertions, document, segment, r"(?:전달체|Carrier|Formulation|제형)\s*[:：]\s*([^\n;|]{2,160})", category="제품", field_path="product.carrier_formulation", label_ko="전달체·제형")

        _keyword_assertion(assertions, document, segment, patterns=[r"\bsiRNA\b.*(?:nano|나노)|(?:nano|나노).*\bsiRNA\b"], category="제품", field_path="product.modality", label_ko="제품 Modality", proposed_value="siRNA + 나노의약품", confidence=0.82)
        _keyword_assertion(assertions, document, segment, patterns=[r"\bsiRNA\b"], category="제품", field_path="product.modality", label_ko="제품 Modality", proposed_value="siRNA 치료제", confidence=0.70)
        _keyword_assertion(assertions, document, segment, patterns=[r"nanoparticle|nanomedicine|나노입자|나노의약품"], category="제품", field_path="product.modality", label_ko="제품 Modality", proposed_value="나노의약품", confidence=0.70)
        _keyword_assertion(assertions, document, segment, patterns=[r"small[- ]?molecule|저분자"], category="제품", field_path="product.modality", label_ko="제품 Modality", proposed_value="저분자 NME", confidence=0.70)
        _keyword_assertion(assertions, document, segment, patterns=[r"intravenous|\bIV\b|정맥(?:투여|주사)?"], category="제품", field_path="product.route", label_ko="투여경로", proposed_value="정맥투여")
        _keyword_assertion(assertions, document, segment, patterns=[r"repeat(?:ed)?\s*(?:dose|exposure)|반복(?:투여|노출)"], category="노출", field_path="product.exposure_pattern", label_ko="노출 형태", proposed_value="반복 노출")
        _keyword_assertion(assertions, document, segment, patterns=[r"single\s*(?:dose|exposure)|단회(?:투여|노출)"], category="노출", field_path="product.exposure_pattern", label_ko="노출 형태", proposed_value="단회 노출")
        _keyword_assertion(assertions, document, segment, patterns=[r"quantitative\s+biodistribution|정량적\s*(?:조직)?분포"], category="노출", field_path="product.distribution_status", label_ko="조직분포", proposed_value="정량적 자료")
        _keyword_assertion(assertions, document, segment, patterns=[r"qualitative\s+biodistribution|정성적\s*(?:조직)?분포"], category="노출", field_path="product.distribution_status", label_ko="조직분포", proposed_value="정성적 자료")

        # AI model card
        _first_regex(assertions, document, segment, r"(?:모델명|Model\s*Name)\s*[:：]\s*([^\n;|]{2,120})", category="AI 모델", field_path="ai_model.model_name", label_ko="AI 모델명")
        _first_regex(assertions, document, segment, r"(?:모델\s*)?(?:버전|Version)\s*[:：]?\s*(v?\d+(?:\.\d+){0,3}[A-Za-z0-9._-]*)", category="AI 모델", field_path="ai_model.model_version", label_ko="AI 모델 버전")
        _first_regex(assertions, document, segment, r"Sensitivity\s*[:=]\s*(\d+(?:\.\d+)?)\s*%?", category="AI 모델", field_path="ai_model.sensitivity_percent", label_ko="Sensitivity", value_type="float")
        _first_regex(assertions, document, segment, r"Specificity\s*[:=]\s*(\d+(?:\.\d+)?)\s*%?", category="AI 모델", field_path="ai_model.specificity_percent", label_ko="Specificity", value_type="float")
        _first_regex(assertions, document, segment, r"(?:False[- ]?negative(?:\s*rate)?|거짓\s*음성률)\s*[:=]\s*(\d+(?:\.\d+)?)\s*%?", category="AI 모델", field_path="ai_model.false_negative_rate_percent", label_ko="False-negative rate", value_type="float")
        _first_regex(assertions, document, segment, r"(?:False[- ]?positive(?:\s*rate)?|거짓\s*양성률)\s*[:=]\s*(\d+(?:\.\d+)?)\s*%?", category="AI 모델", field_path="ai_model.false_positive_rate_percent", label_ko="False-positive rate", value_type="float")
        _first_regex(assertions, document, segment, r"(?:PPV|Positive\s+Predictive\s+Value)\s*[:=]\s*(\d+(?:\.\d+)?)\s*%?", category="AI 모델", field_path="ai_model.ppv_percent", label_ko="PPV", value_type="float")
        _first_regex(assertions, document, segment, r"(?:NPV|Negative\s+Predictive\s+Value)\s*[:=]\s*(\d+(?:\.\d+)?)\s*%?", category="AI 모델", field_path="ai_model.npv_percent", label_ko="NPV", value_type="float")
        _first_regex(assertions, document, segment, r"Balanced\s+Accuracy\s*[:=]\s*(\d+(?:\.\d+)?)\s*%?", category="AI 모델", field_path="ai_model.balanced_accuracy_percent", label_ko="Balanced accuracy", value_type="float")
        _first_regex(assertions, document, segment, r"AUROC\s*[:=]\s*(0?\.\d+|1(?:\.0+)?)", category="AI 모델", field_path="ai_model.auroc", label_ko="AUROC", value_type="float")
        _first_regex(assertions, document, segment, r"AUPRC\s*[:=]\s*(0?\.\d+|1(?:\.0+)?)", category="AI 모델", field_path="ai_model.auprc", label_ko="AUPRC", value_type="float")
        _first_regex(assertions, document, segment, r"(?:Decision\s+Threshold|분류\s*임계값|Threshold)\s*[:=]\s*(\d+(?:\.\d+)?)", category="AI 모델", field_path="ai_model.decision_threshold", label_ko="Decision threshold", value_type="float")
        _first_regex(assertions, document, segment, r"(?:Training\s+Sample\s+Size|Training\s+N|학습\s*표본\s*수)\s*[:=]\s*(\d+)", category="AI 데이터", field_path="ai_model.training_sample_size", label_ko="학습표본 수", value_type="int")
        _first_regex(assertions, document, segment, r"(?:Positive\s+Class|양성\s*비율)\s*[:=]\s*(\d+(?:\.\d+)?)\s*%", category="AI 데이터", field_path="ai_model.positive_class_percent", label_ko="독성 양성 비율", value_type="float")
        _first_regex(assertions, document, segment, r"(?:Dataset\s+Version|Data\s+Version|데이터셋\s*버전)\s*[:：]\s*([^\n;|]{1,100})", category="AI 데이터", field_path="ai_model.dataset_version", label_ko="데이터셋 버전")
        _first_regex(assertions, document, segment, r"(?:Dataset\s+Source|Data\s+Source|데이터\s*출처)\s*[:：]\s*([^\n;|]{2,140})", category="AI 데이터", field_path="ai_model.dataset_source", label_ko="데이터 출처")
        _keyword_assertion(assertions, document, segment, patterns=[r"scaffold\s+split|scaffold\s*분할"], category="AI 데이터", field_path="ai_model.split_strategy", label_ko="분할전략", proposed_value="Scaffold 분할", confidence=0.82)
        _keyword_assertion(assertions, document, segment, patterns=[r"temporal\s+split|time[- ]?based\s+split|시간\s*분할"], category="AI 데이터", field_path="ai_model.split_strategy", label_ko="분할전략", proposed_value="시간 분할", confidence=0.82)
        _keyword_assertion(assertions, document, segment, patterns=[r"random\s+split|무작위\s*분할"], category="AI 데이터", field_path="ai_model.split_strategy", label_ko="분할전략", proposed_value="무작위 분할", confidence=0.72)
        _keyword_assertion(assertions, document, segment, patterns=[r"no\s+(?:data\s+)?leakage|leakage\s+(?:assessment\s+)?(?:passed|negative)|누수\s*(?:없음|문제\s*없음)"], category="AI 데이터", field_path="ai_model.leakage_assessment", label_ko="Data leakage 평가", proposed_value="평가 완료 — 문제 없음", confidence=0.82, allow_negated=True)
        _keyword_assertion(assertions, document, segment, patterns=[r"data\s+leakage\s+(?:confirmed|identified)|누수\s*(?:확인|발견)"], category="AI 데이터", field_path="ai_model.leakage_assessment", label_ko="Data leakage 평가", proposed_value="누수 확인", confidence=0.88)
        _keyword_assertion(assertions, document, segment, patterns=[r"calibrated\s+probability|보정된\s*확률"], category="AI 모델", field_path="ai_model.probability_type", label_ko="출력값 의미", proposed_value="보정된 확률", confidence=0.82)
        _keyword_assertion(assertions, document, segment, patterns=[r"raw\s+(?:model\s+)?score|원시\s*모델\s*점수"], category="AI 모델", field_path="ai_model.probability_type", label_ko="출력값 의미", proposed_value="원시 모델점수", confidence=0.82)
        _keyword_assertion(assertions, document, segment, patterns=[r"confidence\s+intervals?\s+(?:reported|provided)|신뢰구간\s*(?:보고|제시)"], category="AI 모델", field_path="ai_model.performance_confidence_intervals", label_ko="성능 신뢰구간", proposed_value="보고됨", confidence=0.78)
        _keyword_assertion(assertions, document, segment, patterns=[r"input\s+(?:structure|sequence|quality)[^\n]{0,60}(?:verified|confirmed)|입력[^\n]{0,60}(?:검증|확인)"], category="AI 예측", field_path="ai_model.input_quality_verified", label_ko="개별 예측 입력품질", proposed_value="true", value_type="bool", confidence=0.75)
        _keyword_assertion(assertions, document, segment, patterns=[r"external\s+validation[^\n]{0,80}(?:partial|limited)|외부\s*(?:독립\s*)?검증[^\n]{0,80}(?:부분|제한)"], category="AI 모델", field_path="ai_model.external_validation", label_ko="외부 독립검증", proposed_value="부분적으로 확인", confidence=0.82)
        _keyword_assertion(assertions, document, segment, patterns=[r"external\s+validation[^\n]{0,80}(?:completed|confirmed|independent|validated)|외부\s*(?:독립\s*)?검증[^\n]{0,80}(?:완료|확인|독립|검증됨)"], category="AI 모델", field_path="ai_model.external_validation", label_ko="외부 독립검증", proposed_value="확인됨", confidence=0.82)
        _keyword_assertion(assertions, document, segment, patterns=[r"calibrat(?:ed|ion).*valid|확률\s*보정.*검증"], category="AI 모델", field_path="ai_model.calibration_status", label_ko="Calibration", proposed_value="검증됨", confidence=0.76)
        _keyword_assertion(assertions, document, segment, patterns=[r"hepatotoxicity|drug[- ]?induced\s+liver\s+injury|\bDILI\b|간독성|약물유발\s*간손상"], category="AI 모델", field_path="ai_model.endpoint", label_ko="AI 예측 Endpoint", proposed_value="초기 간독성", confidence=0.78)
        _keyword_assertion(assertions, document, segment, patterns=[r"training[^\n]{0,90}small[- ]?molecule|학습[^\n]{0,90}저분자"], category="AI 모델", field_path="ai_model.domain_modalities", label_ko="AI 학습 Modality", proposed_value="저분자", value_type="list", confidence=0.78)
        _keyword_assertion(assertions, document, segment, patterns=[r"training[^\n]{0,90}oligonucleotide|학습[^\n]{0,90}올리고"], category="AI 모델", field_path="ai_model.domain_modalities", label_ko="AI 학습 Modality", proposed_value="올리고뉴클레오타이드", value_type="list", confidence=0.78)
        _keyword_assertion(assertions, document, segment, patterns=[r"training[^\n]{0,90}nano|학습[^\n]{0,90}나노"], category="AI 모델", field_path="ai_model.domain_modalities", label_ko="AI 학습 Modality", proposed_value="나노의약품", value_type="list", confidence=0.78)

        _result_assertions(assertions, document, segment)

        # NAM card
        _keyword_assertion(assertions, document, segment, patterns=[r"2D\s*(?:cell|culture)|2차원\s*세포"], category="NAM 시험", field_path="nam_assay.nam_type", label_ko="NAM 유형", proposed_value="2D 세포시험")
        _keyword_assertion(assertions, document, segment, patterns=[r"co[- ]?culture|공배양"], category="NAM 시험", field_path="nam_assay.nam_type", label_ko="NAM 유형", proposed_value="공배양(Coculture)")
        _keyword_assertion(assertions, document, segment, patterns=[r"spheroid"], category="NAM 시험", field_path="nam_assay.nam_type", label_ko="NAM 유형", proposed_value="3D 간 Spheroid")
        _keyword_assertion(assertions, document, segment, patterns=[r"organoid"], category="NAM 시험", field_path="nam_assay.nam_type", label_ko="NAM 유형", proposed_value="간 Organoid")
        _keyword_assertion(assertions, document, segment, patterns=[r"organ[- ]?on[- ]?chip|liver[- ]?on[- ]?chip|microphysiological|\bMPS\b"], category="NAM 시험", field_path="nam_assay.nam_type", label_ko="NAM 유형", proposed_value="Liver-on-chip / MPS")
        _keyword_assertion(assertions, document, segment, patterns=[r"human[- ]?derived|primary\s+human|사람\s*유래"], category="NAM 시험", field_path="nam_assay.system_origin", label_ko="시험계 기원", proposed_value="사람 유래")
        _keyword_assertion(assertions, document, segment, patterns=[r"hepatocyte|간세포"], category="NAM 시험", field_path="nam_assay.cell_types", label_ko="포함 세포", proposed_value="간세포(Hepatocyte)", value_type="list")
        _keyword_assertion(assertions, document, segment, patterns=[r"Kupffer"], category="NAM 시험", field_path="nam_assay.cell_types", label_ko="포함 세포", proposed_value="Kupffer cell", value_type="list")
        _keyword_assertion(assertions, document, segment, patterns=[r"stellate"], category="NAM 시험", field_path="nam_assay.cell_types", label_ko="포함 세포", proposed_value="Stellate cell", value_type="list")
        _keyword_assertion(assertions, document, segment, patterns=[r"positive\s+control[^\n]{0,80}(?:valid|pass)|양성대조군[^\n]{0,80}(?:유효|통과)"], category="NAM 시험", field_path="nam_assay.positive_control", label_ko="양성대조군", proposed_value="유효")
        _keyword_assertion(assertions, document, segment, patterns=[r"negative\s+control[^\n]{0,80}(?:valid|pass)|음성대조군[^\n]{0,80}(?:유효|통과)"], category="NAM 시험", field_path="nam_assay.negative_control", label_ko="음성대조군", proposed_value="유효")
        _keyword_assertion(assertions, document, segment, patterns=[r"carrier[- ]?only\s+control[^\n]{0,60}(?:included|yes)|전달체\s*단독대조군[^\n]{0,60}(?:포함|있음)"], category="NAM 시험", field_path="nam_assay.carrier_only_control", label_ko="Carrier-only 대조군", proposed_value="포함")
        _keyword_assertion(assertions, document, segment, patterns=[r"free\s+concentration|intracellular\s+concentration|measured\s+exposure|실제\s*(?:세포내|유리)\s*농도"], category="노출", field_path="nam_assay.measured_exposure", label_ko="실제 노출", proposed_value="측정됨")
        _keyword_assertion(assertions, document, segment, patterns=[r"\bQIVIVE\b|\bPBPK\b"], category="노출", field_path="nam_assay.qivive_pbpk", label_ko="QIVIVE/PBPK", proposed_value="수행됨")
        _keyword_assertion(assertions, document, segment, patterns=[r"repeat(?:ed)?\s+exposure|반복\s*노출"], category="NAM 시험", field_path="nam_assay.exposure_design", label_ko="NAM 노출설계", proposed_value="반복노출")
        _keyword_assertion(assertions, document, segment, patterns=[r"single\s+(?:exposure|24[- ]?h)|급성\s*단회|단회\s*노출"], category="NAM 시험", field_path="nam_assay.exposure_design", label_ko="NAM 노출설계", proposed_value="단회/급성 노출")
        _keyword_assertion(assertions, document, segment, patterns=[r"CYP(?:450)?|metabolic\s+competence|대사능"], category="NAM 시험", field_path="nam_assay.metabolic_competence", label_ko="대사능", proposed_value="부분 확인", confidence=0.68)
        _keyword_assertion(assertions, document, segment, patterns=[r"inter[- ]?day|inter[- ]?lot|donor\s+variability|재현성"], category="NAM 시험", field_path="nam_assay.reproducibility", label_ko="재현성", proposed_value="일부 확인", confidence=0.68)
        _keyword_assertion(assertions, document, segment, patterns=[r"cell\s+viability|\bATP\b|세포\s*생존"], category="NAM 시험", field_path="nam_assay.endpoints", label_ko="NAM Endpoint", proposed_value="Cell viability / ATP", value_type="list")
        _keyword_assertion(assertions, document, segment, patterns=[r"mitochondrial|미토콘드리아"], category="NAM 시험", field_path="nam_assay.endpoints", label_ko="NAM Endpoint", proposed_value="미토콘드리아 기능", value_type="list")
        _keyword_assertion(assertions, document, segment, patterns=[r"cytokine|IL-6|TNF[- ]?alpha|사이토카인"], category="NAM 시험", field_path="nam_assay.endpoints", label_ko="NAM Endpoint", proposed_value="Cytokine", value_type="list")

        # Supporting evidence flags
        _keyword_assertion(assertions, document, segment, patterns=[r"mechanis(?:m|tic)|기전(?:적| 기반)?"], category="보조 근거", field_path="supporting_evidence.mechanistic_evidence", label_ko="기전 기반 근거", proposed_value="true", value_type="bool", confidence=0.64)
        _keyword_assertion(assertions, document, segment, patterns=[r"class\s+effect|class\s+toxicity|동일\s*계열|계열\s*독성"], category="보조 근거", field_path="supporting_evidence.class_or_clinical_evidence", label_ko="동일계열/임상 Class 근거", proposed_value="true", value_type="bool", confidence=0.70)
        _keyword_assertion(assertions, document, segment, patterns=[r"quantitative\s+biodistribution|정량적\s*(?:조직)?분포"], category="보조 근거", field_path="supporting_evidence.quantitative_biodistribution", label_ko="정량적 Biodistribution", proposed_value="true", value_type="bool", confidence=0.82)
        _keyword_assertion(assertions, document, segment, patterns=[r"toxicokinetic|\bTK\b|pharmacokinetic|\bPK\b|노출[-– ]?반응"], category="보조 근거", field_path="supporting_evidence.pk_tk_evidence", label_ko="PK/TK 근거", proposed_value="true", value_type="bool", confidence=0.68)
        _keyword_assertion(assertions, document, segment, patterns=[r"in\s+vivo|동물\s*시험|반복투여\s*독성"], category="보조 근거", field_path="supporting_evidence.existing_in_vivo_evidence", label_ko="기존 in vivo 근거", proposed_value="true", value_type="bool", confidence=0.66)
        _keyword_assertion(assertions, document, segment, patterns=[r"clinical\s+(?:evidence|safety)|human\s+(?:data|evidence)|임상\s*안전|사람\s*근거"], category="보조 근거", field_path="supporting_evidence.human_evidence", label_ko="사람/임상 근거", proposed_value="true", value_type="bool", confidence=0.72)

    return assertions


def extract_assertions_from_documents(documents: Iterable[DocumentRecord]) -> list[EvidenceAssertion]:
    assertions: list[EvidenceAssertion] = []
    for document in documents:
        assertions.extend(extract_assertions(document))
    return assertions


class AssertionValidationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def metadata_issues(assertion: EvidenceAssertion) -> list[str]:
    """Validate reviewer metadata without inferring its scientific correctness."""
    issues = []
    if assertion.evidence_kind not in EVIDENCE_KIND_OPTIONS:
        issues.append("evidence_kind_invalid")
    if not isinstance(assertion.scope, dict):
        issues.append("scope_invalid")
    else:
        if any(key not in SCOPE_KEYS for key in assertion.scope):
            issues.append("scope_key_unsupported")
        if any(not isinstance(value, str) for value in assertion.scope.values()):
            issues.append("scope_value_invalid")
    return issues


def _target(root: AssessmentInput, path: str) -> tuple[Any, str, str]:
    if not isinstance(path, str):
        raise AssertionValidationError("field_path_invalid", "Field path must be a string")
    parts = path.split(".")
    root_fields = {item.name for item in fields(root)}
    if len(parts) != 2 or parts[0] not in root_fields:
        raise AssertionValidationError("field_path_invalid", f"Unsupported field path: {path}")
    parent_name, attribute = parts
    parent = getattr(root, parent_name)
    if not is_dataclass(parent) or attribute not in {item.name for item in fields(parent)}:
        raise AssertionValidationError("field_path_invalid", f"Unknown target attribute: {path}")
    annotation = get_type_hints(type(parent))[attribute]
    alternatives = get_args(annotation)
    if type(None) in alternatives:
        non_null = [item for item in alternatives if item is not type(None)]
        annotation = non_null[0] if len(non_null) == 1 else annotation
    if annotation in (str, bool, int, float):
        expected = annotation.__name__
    elif get_origin(annotation) is list and get_args(annotation) == (str,):
        expected = "list"
    else:
        raise AssertionValidationError("target_type_unsupported", f"Unsupported target type: {path}")
    return parent, attribute, expected


def _coerce(value: str, value_type: str) -> Any:
    value = str(value).strip()
    if not value:
        raise AssertionValidationError("value_invalid", "Reviewed assertion value must not be empty")
    if value_type == "bool":
        normalized = value.casefold()
        if normalized in {"true", "1", "yes", "y", "예", "있음", "확인"}:
            return True
        if normalized in {"false", "0", "no", "n", "아니오", "없음"}:
            return False
        raise AssertionValidationError("value_invalid", f"Invalid boolean value: {value}")
    if value_type == "int":
        if not re.fullmatch(r"[+-]?\d+(?:\.0+)?", value):
            raise AssertionValidationError("value_invalid", f"Invalid integer value: {value}")
        return int(value.split(".", 1)[0])
    if value_type == "float":
        numeric = value.removesuffix("%").strip()
        try:
            number = float(numeric)
        except (ValueError, OverflowError) as exc:
            raise AssertionValidationError("value_invalid", f"Invalid {value_type} value: {value}") from exc
        if not math.isfinite(number):
            raise AssertionValidationError("value_invalid", f"Invalid finite {value_type} value: {value}")
        return number
    if value_type == "list":
        result = [item.strip() for item in re.split(r"[,;|]", value) if item.strip()]
        if not result:
            raise AssertionValidationError("value_invalid", "Reviewed list value must not be empty")
        return result
    if value_type == "str":
        return value
    raise AssertionValidationError("value_type_invalid", f"Unsupported assertion value type: {value_type}")


def validate_numeric_domain(field_path: str, value: Any) -> None:
    """Check basic numeric domains, not model performance or scientific validity."""
    if type(value) not in {int, float}:
        return
    key = field_path.rsplit(".", 1)[-1]
    bounds = (0, 100) if key.endswith("_percent") else (0, 1) if key in {
        "auroc", "auprc", "decision_threshold", "brier_score"
    } else (1, 5) if key in {"model_influence", "decision_consequence"} else None
    if bounds and not bounds[0] <= value <= bounds[1]:
        raise AssertionValidationError("value_out_of_range", f"Value outside {bounds[0]}–{bounds[1]}: {field_path}")
    if key == "training_sample_size" and value < 0:
        raise AssertionValidationError("value_out_of_range", f"Negative sample size: {field_path}")


def validate_assertion_value(assertion: EvidenceAssertion, inp: AssessmentInput | None = None) -> Any:
    """Return the typed value or raise; never silently skip a reviewed claim."""
    _, _, expected = _target(inp if inp is not None else AssessmentInput(), assertion.field_path)
    if assertion.value_type != expected:
        raise AssertionValidationError(
            "value_type_mismatch",
            f"Value type for {assertion.field_path} must be {expected}, got {assertion.value_type}",
        )
    if not isinstance(assertion.proposed_value, str):
        raise AssertionValidationError("value_invalid", "Reviewed assertion value must be a string for typed conversion")
    value = _coerce(assertion.proposed_value, assertion.value_type)
    validate_numeric_domain(assertion.field_path, value)
    return value


def assertion_value_issues(assertion: EvidenceAssertion) -> list[str]:
    try:
        validate_assertion_value(assertion)
    except AssertionValidationError as exc:
        return [exc.code]
    return []


def _set_path(root: AssessmentInput, assertion: EvidenceAssertion) -> None:
    parent, attribute, expected = _target(root, assertion.field_path)
    coerced = validate_assertion_value(assertion, root)
    if expected == "list":
        merged = list(getattr(parent, attribute) or [])
        for item in coerced:
            if item not in merged:
                merged.append(item)
        setattr(parent, attribute, merged)
    else:
        setattr(parent, attribute, coerced)


def reviewed_assertion_conflicts(assertions: Iterable[EvidenceAssertion]) -> dict[str, list[str]]:
    """Return scalar field paths that have more than one reviewed value."""
    grouped: dict[str, set[str]] = {}
    for item in assertions:
        if item.review_status not in REVIEWED_STATUSES or item.value_type == "list":
            continue
        grouped.setdefault(item.field_path, set()).add(str(item.proposed_value).strip())
    return {path: sorted(values) for path, values in grouped.items() if len(values) > 1}


def apply_reviewed_assertions(inp: AssessmentInput, assertions: Iterable[EvidenceAssertion], *, documents: Iterable[DocumentRecord] | None = None) -> AssessmentInput:
    """Apply reviewed, valid assertions atomically to a copy of AssessmentInput."""
    assertions = list(assertions)
    documents = list(documents) if documents is not None else None
    reviewed = [item for item in assertions if item.review_status in REVIEWED_STATUSES]
    for assertion in reviewed:
        validate_assertion_value(assertion, inp)
        issues = metadata_issues(assertion)
        if issues:
            raise ValueError(f"Assertion metadata invalid for {assertion.assertion_id}: " + ", ".join(issues))
        if assertion.field_path == "ai_model.result" and assertion.evidence_kind != "prediction":
            raise ValueError("AI model results require prediction evidence kind before application")
        if assertion.field_path == "nam_assay.result" and assertion.evidence_kind != "observation":
            raise ValueError("NAM results require explicit observation review before application")
    conflicts = reviewed_assertion_conflicts(reviewed)
    if conflicts:
        raise ValueError("Resolve conflicting reviewed assertions before application: " + ", ".join(conflicts))
    if documents is not None:
        from .evidence_review import application_blockers
        blockers = application_blockers(assertions, documents, assessment_input=inp)
        if blockers:
            raise ValueError("Source review required before application: " + ", ".join(blockers))
    updated = copy.deepcopy(inp)
    for assertion in reviewed:
        _set_path(updated, assertion)
    if documents is not None:
        # Reviewed context fields may also change the target subject/endpoint.
        # Recheck the final card so those edits cannot rebind another study's result.
        blockers = application_blockers(assertions, documents, assessment_input=updated)
        if blockers:
            raise ValueError("Source review required for final assessment scope: " + ", ".join(blockers))
    if reviewed:
        updated.supporting_evidence.assertions_reviewed = True
        updated.supporting_evidence.evidence_traceable = all(
            item.source_document_name and item.source_location for item in reviewed
        )
    return updated


def assertion_table_rows(assertions: Iterable[EvidenceAssertion]) -> list[dict[str, Any]]:
    return [
        {
            "Assertion ID": item.assertion_id,
            "분류": item.category,
            "평가 필드": item.label_ko,
            "Field Path": item.field_path,
            "제안/수정 값": item.proposed_value,
            "Original extracted value": item.extracted_value,
            "Evidence kind": item.evidence_kind,
            **{f"Scope {key}": item.scope.get(key, "") if isinstance(item.scope, dict) else "" for key in SCOPE_KEYS},
            "형식": item.value_type,
            "출처 문서": item.source_document_name,
            "출처 위치": item.source_location,
            "근거 발췌": item.source_excerpt,
            "Extraction heuristic (uncalibrated)": item.confidence,
            "검토 상태": item.review_status,
            "검토 메모": item.reviewer_note,
        }
        for item in assertions
    ]


def assertions_from_table_rows(rows: Iterable[dict[str, Any]], existing: Iterable[EvidenceAssertion]) -> list[EvidenceAssertion]:
    existing_by_id = {item.assertion_id: item for item in existing}
    output: list[EvidenceAssertion] = []
    for row in rows:
        assertion_id = str(row.get("Assertion ID", ""))
        original = existing_by_id.get(assertion_id)
        if not original:
            continue
        # Provenance, field targets and the extracted value always come from the
        # existing record, never from editable table data.
        base = copy.deepcopy(original)
        base.proposed_value = str(row.get("제안/수정 값", base.proposed_value))
        base.evidence_kind = str(row.get("Evidence kind", base.evidence_kind))
        scope = dict(base.scope) if isinstance(base.scope, dict) else {}
        for key in SCOPE_KEYS:
            if f"Scope {key}" in row:
                value = str(row[f"Scope {key}"]).strip()
                if value:
                    scope[key] = value
                else:
                    scope.pop(key, None)
        base.scope = scope
        base.review_status = str(row.get("검토 상태", base.review_status))
        base.reviewer_note = str(row.get("검토 메모", base.reviewer_note))
        if (base.proposed_value, base.evidence_kind, base.scope) != (original.proposed_value, original.evidence_kind, original.scope):
            base.review_status = "제안됨"
        output.append(base)
    return output
