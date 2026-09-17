"""Offline registration audit. Never creates stores, opens devices or grants motion."""
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3

from mes_vision.training.data import sha256


FIELD_CHECKS = [
    ('image_quality', '전체 위치 검출과 근접 불량 검사에 필요한 화질·조명·검사면'),
    ('capture_timing', '이동 안정 시간과 노출부터 수신까지의 최대 영상 지연'),
    ('coordinates', '촬영점·집기점 독립 검증 오차와 적용 영역'),
    ('motion', '카메라·케이블·집게를 포함한 전체 이동 경로와 정지 동작'),
    ('sorting', '집기 확인 입력·놓기 확인·연속 적재·남은 물체의 위치 유지'),
    ('accuracy', '독립 실물 자료의 불량 통과·오탐·보류 및 미학습 불량 평가'),
    ('throughput', '물체 수별 전체 처리 시간·VLM ON/OFF·장시간 운전'),
    ('laptop', 'RTX5070 노트북의 동일 자료·메모리·처리 시간 검증'),
]


def registration_snapshot(runtime):
    """One SQLite read transaction; no migrations, recovery or preference writes."""
    path = Path(runtime).resolve() / 'operation.sqlite3'
    if not path.is_file():
        return None, []
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=5)) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        equipment = db.execute('SELECT data FROM equipment ORDER BY version DESC LIMIT 1').fetchone()
        products = db.execute('SELECT p.data FROM products p JOIN '
                              '(SELECT id,MAX(version) v FROM products GROUP BY id) n '
                              'ON p.id=n.id AND p.version=n.v ORDER BY p.id').fetchall()
        return json.loads(equipment[0]) if equipment else None, [json.loads(p[0]) for p in products]


def audit(runtime):
    """Only checks registered files/conditions; live start checks remain authoritative."""
    from mes_vision.operation.catalog import readiness
    from .settings import StationSettings
    from .recipe import load_recipe

    root = Path(runtime).resolve()
    checks = []

    def check(identity, title, action):
        try:
            detail = action()
            checks.append(dict(id=identity, title=title, status='passed', detail=detail))
            return detail
        except Exception as exc:
            checks.append(dict(id=identity, title=title, status='needs_attention',
                               detail=f'{type(exc).__name__}: {exc}'))
            return None

    def snapshot():
        e, p = registration_snapshot(root)
        if e is None:
            raise ValueError('장비 설정이 등록되지 않았습니다.')
        if not isinstance(e, dict) or not isinstance(e.get('version'), int):
            raise ValueError('장비 설정 형식 오류')
        if any(not isinstance(v, dict) or not isinstance(v.get('id'), str)
               or not isinstance(v.get('name'), str) for v in p):
            raise ValueError('품목 설정 형식 오류')
        return e, p

    values = check('registration', '등록 자료 읽기', snapshot)
    equipment, products = values if values else (None, [])
    # Avoid retaining the entire configuration in the general check description.
    if values:
        checks[-1]['detail'] = {'equipment_version': equipment['version'], 'products': len(products)}
    if not products:
        checks.append(dict(id='products', title='품목 등록', status='needs_attention', detail='등록된 품목이 없습니다.'))

    def mounted():
        settings = StationSettings(root)
        calibration = settings.calibrated()
        if equipment is None:
            raise ValueError('장비 설정 확인이 먼저 필요합니다.')
        c = equipment['camera']
        calibration.verify_context(dict(calibration_digest=calibration.digest,
            product_id=calibration.product_id, overview_pose=calibration.value['overview_pose'],
            camera_serial=c['serial'], mount_revision=c['mount_revision'],
            image_size=[c['width'], c['height']]), equipment)
        return {'settings_version': settings.value['version'], 'product_id': calibration.product_id,
                'calibration_id': calibration.identity}

    calibration_info = check('capture_setup', '촬영 설정·보정 파일 일치', mounted)
    if equipment is not None:
        def robot():
            from mes_vision.robot import load_profile
            name = equipment['robot']['profile']
            if not name:
                raise ValueError('로봇 운전 설정을 등록하세요.')
            profile = load_profile(Path(name))
            if not profile.validated or profile.kind != 'real':
                raise ValueError('실측 검증된 로봇 운전 설정이 필요합니다.')
            if calibration_info is None or profile.calibration_version != calibration_info['calibration_id']:
                raise ValueError('촬영·집기 보정과 로봇 운전 설정을 확인하세요.')
            if profile.product_id != calibration_info['product_id']:
                raise ValueError('로봇 운전 품목과 보정 품목이 다릅니다.')
            return {'path': str(Path(name).resolve()), 'sha256': sha256(Path(name))}
        check('robot_profile', '로봇 운전 파일·보정 연결', robot)
        for product in products:
            identity = product['id']
            def detail(p=product):
                problems = readiness(p, equipment, verify_files=True)
                if problems:
                    raise ValueError('\n'.join(problems))
                return {'product_id': p['id'], 'version': p['version']}
            check('detail:' + identity, product['name'] + ' · 상세 검사 등록', detail)
            def recipe(p=product):
                value = load_recipe(root, p, equipment)
                return {'product_id': p['id'], 'overview_sha256': value['overview_asset']['sha256']}
            check('recipe:' + identity, product['name'] + ' · 전체·상세 촬영 연결', recipe)
            if calibration_info and identity != calibration_info['product_id']:
                checks.append(dict(id='product_calibration:' + identity, title=product['name'] + ' · 품목 보정',
                    status='needs_attention', detail='현재 보정은 다른 품목에 연결돼 있습니다.'))

    return dict(schema_version=1, time_utc=datetime.now(timezone.utc).isoformat(), runtime=str(root),
        scope='등록 파일 일부와 조건의 오프라인 확인. 장비 연결·추론·실측·저장 쓰기 시험은 수행하지 않습니다.',
        registration_status='checks_passed' if all(c['status'] == 'passed' for c in checks) else 'needs_attention',
        physical_acceptance='not_assessed', motion_authorized=False, checks=checks,
        field_checks=[dict(id=k, title=v, status='not_assessed') for k, v in FIELD_CHECKS])


def markdown(report):
    lines = ['# 전체·상세 검사 사전 점검', '', report['scope'], '',
             '- 점검 시각(UTC): ' + report['time_utc'], '- 운영 폴더: ' + report['runtime'],
             '- 실제 장비 검증: 미평가', '- 이 보고서는 검사 시작·로봇 동작을 승인하지 않습니다.', '',
             '## 등록 자료', '']
    for item in report['checks']:
        label = '파일·조건 확인 통과' if item['status'] == 'passed' else '확인 필요'
        detail = item['detail'] if isinstance(item['detail'], str) else json.dumps(item['detail'], ensure_ascii=False)
        lines += [f"- **{item['title']} — {label}**", '  ' + detail.replace('\n', '\n  ')]
    if report.get('software_tests', {}).get('status') == 'not_run':
        lines += ['', '## 소프트웨어 연결 시험', '', '- 이번 점검에서는 실행하지 않았습니다.']
    elif report.get('software_tests'):
        item = report['software_tests']
        lines += ['', '## 소프트웨어 연결 시험', '',
                  f"- 상태: {item['status']} · 실행 {item.get('tests_run', 0)}개 · 건너뜀 {item.get('skipped', 0)}개",
                  '- 주입 영상·모의 장치 시험이며 실제 AI 정확도나 로봇 성능 결과가 아닙니다.',
                  '- 상세 결과: software-tests.json / software-tests.log']
    if report.get('source_unchanged_during_check') is False:
        lines += ['', '**점검 중 소스가 변경됐습니다. 현재 소스에 대한 시험 통과로 사용할 수 없습니다.**']
    lines += ['', '## 실물 준비 후 확인', '']
    lines += ['- 미평가: ' + item['title'] for item in report['field_checks']]
    return '\n'.join(lines) + '\n'
