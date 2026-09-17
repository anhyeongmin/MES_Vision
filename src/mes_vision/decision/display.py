from mes_vision.i18n import text_join
from mes_vision.i18n import tr, trf
"""Operator-facing decision explanations shared by present and future UIs."""
LABELS = {
    "POLICY_SCOPE_MISMATCH": tr('실물·모의 판정 설정 불일치'),
    "RUN_IDENTITY_INVALID": tr('검사 실행 식별 오류'), "POLICY_PRODUCT_MISMATCH": tr('품목 설정 불일치 또는 미설정'),
    "POLICY_NOT_VALIDATED": tr('판정 규칙 미검증'), "DETECTOR_MODEL_MISMATCH": tr('물체 검출 모델 불일치 또는 미설정'),
    "INSPECTION_EXECUTION_ERROR": tr('검사 실행 실패'), "INSPECTION_INCOMPLETE": tr('검사 실행 미완료'),
    "FRAME_GEOMETRY_INVALID": tr('원본 영상 좌표 정보 오류'), "FRAME_CHECKS_UNCONFIGURED": tr('촬영 품질·작업 영역 검사 미설정'),
    "FRAME_EVIDENCE_IDENTITY_MISMATCH": tr('다른 촬영의 근거가 연결됨'), "FRAME_CRITERIA_NOT_VALIDATED": tr('촬영 조건 기준 미검증 또는 불일치'),
    "IMAGE_QUALITY_NOT_PASSED": tr('촬영 품질 확인 필요'), "WORKSPACE_NOT_PASSED": tr('작업 영역 확인 필요'),
    "EXPECTED_COUNT_UNCONFIGURED": tr('예상 수량 미설정'), "OBJECT_COUNT_MISMATCH": tr('예상 수량과 검출 수량 불일치'),
    "ZERO_OBJECTS_NOT_CONFIRMED_EMPTY": tr('물체 미검출 · 빈 작업대로 확정할 수 없음'), "DUPLICATE_OBJECT_ID": tr('물체 식별 중복'),
    "RUN_ISSUE": tr('검사 실행 주의 사항'), "OBJECT_IDENTITY_INVALID": tr('물체 식별 연결 오류'), "CROP_IDENTITY_INVALID": tr('물체 이미지 좌표 연결 오류'),
    "DUPLICATE_CHECK_ID": tr('검사 결과 중복'), "CHECK_IDENTITY_MISMATCH": tr('다른 물체·촬영의 검사 결과'),
    "FINDING_COORDINATES_INVALID": tr('불량 근거 좌표 연결 오류'), "UNREGISTERED_CHECK": tr('등록되지 않은 검사 항목'),
    "OBJECT_ISSUE": tr('물체 배치 확인 필요'), "OBJECT_TOUCHES_IMAGE_EDGE": tr('물체가 영상 가장자리에 닿음'),
    "OBJECT_PARTIALLY_OUTSIDE_IMAGE": tr('물체 일부가 영상 밖에 있음'), "OBJECT_OR_CROP_OVERLAP": tr('물체 또는 검사 영역이 겹침'),
    "REQUIRED_CHECK_MISSING": tr('필수 검사 결과 없음'), "OPTIONAL_CHECK_MISSING": tr('선택 검사 결과 없음'),
    "CHECK_ERROR": tr('검사 오류'), "CHECK_NOT_RUN": tr('검사 미실행'), "CHECK_CRITERIA_NOT_VALIDATED": tr('검사 기준 미검증'),
    "CHECK_MODEL_MISMATCH": tr('검사 모델 불일치'), "CHECK_PRODUCT_MISMATCH": tr('검사 품목 불일치'), "CHECK_SCOPE_MISMATCH": tr('실물·모의 검사 구분 불일치'),
    "CANDIDATE_CONTRACT_MISMATCH": tr('불량 후보 결과 형식 불일치'), "CANDIDATE_THRESHOLD_MISMATCH": tr('후보 수집 기준 불일치'),
    "DEFECT_CLASS_MAPPING_MISMATCH": tr('모델 클래스와 불량 코드 연결 불일치'), "INVALID_DEFECT_CANDIDATE": tr('유효하지 않은 불량 후보'),
    "DEFECT_CANDIDATE_IN_REVIEW_BAND": tr('확정 기준 미만의 불량 후보 · 확인 필요'), "CHECK_CRITERIA_VERSION_MISMATCH": tr('검사 기준 버전 불일치'),
    "ANOMALY_CRITERIA_MISMATCH": tr('정상 비교 기준 내용 불일치'), "ANOMALY_CRITERIA_INVALID": tr('정상 비교 기준 오류'),
    "ANOMALY_REFERENCE_MISMATCH": tr('정상 참조·품목·기준 연결 불일치'), "ANOMALY_SCORE_INVALID": tr('이상 점수 오류'),
    "ANOMALY_SCORE_STATUS_CONFLICT": tr('이상 점수와 검사 상태 충돌'), "FAIL_WITHOUT_CONFIRMED_DEFECT_CODE": tr('불합격 근거 코드 없음'),
    "ANOMALY_CANNOT_NAME_DEFECT": tr('이상 점수만으로 결함 종류를 확정할 수 없음'), "CHECK_UNCERTAIN": tr('검사 결과 불확실'),
    "CONFIRMED_DEFECT": tr('검증된 기준의 불량 근거'), "CHECK_PASSED": tr('검사 통과'),
}


def decision_lines(record):
    details = record.get("decision_details")
    if not details: return []
    names = {"known_defects": tr('알려진 불량'), "anomaly": tr('이상 탐지'), "geometry": tr('형상')}
    title = {"OK": tr('정상'), "NG": tr('불량'), "REVIEW": tr('판정 보류')}.get(record.get("final_decision"), tr('미판정'))
    lines = [trf('최종 판정: {v0}', v0=title), tr('판정 규칙: ') + details["policy_version"],
             tr('필수 검사 완료: ') + (tr('예') if details["required_checks_complete"] else tr('아니오'))]
    if details["kind"] == "synthetic": lines.append(tr('모의 판정 · 실물 검사 결과가 아닙니다.'))
    if details["defect_codes"]: lines.append(tr('확정 불량 코드: ') + text_join(', ', details["defect_codes"]))
    for r in details["reasons"]:
        if r["code"] == "CHECK_PASSED": continue
        suffix = " · " + names.get(r["check_id"], r["check_id"]) if r["check_id"] else ""
        lines.append(LABELS.get(r["code"], r["code"]) + suffix)
    return lines + ["", tr('원본 검사 근거:')]
