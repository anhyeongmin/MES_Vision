"""Whole-cycle archive with verified copies and resumable deletion of inventoried files."""
from pathlib import Path
from copy import deepcopy
import json,os,shutil
from mes_vision.anomaly.features import fingerprint
from mes_vision.training.data import require,sha256
from mes_vision.vlm.snapshots import load_snapshot
from mes_vision.operation.storage import plain_path,within,inventory
from .sequence import ScanJournal,TERMINAL
from .history import assets,relocate,sync
from .worker import read_capture


def managed(source,root):
    source=plain_path(source)
    require(source.parent in {root/'station-captures',root/'station-evidence'},'허용된 전체·상세 원본 폴더가 아닙니다.')
    return within(source,source.parent)


def validate_cycle(cycle):
    if cycle.get('overview'): read_capture(cycle['overview'])
    for target in cycle['targets']:
        if target.get('detail'): read_capture(target['detail'])
        receipt=target.get('result') or {}
        if receipt.get('snapshot_path'): load_snapshot(receipt['snapshot_path'],expected_digest=receipt['snapshot_digest'])


def preview(service,now,policy):
    sync(service.store); candidates=[]; protected=[]; errors=[]
    if not (service.root/'station.sqlite3').exists(): return candidates,protected,errors
    with ScanJournal(service.root).connect() as db: cycles=[json.loads(r[0]) for r in db.execute('SELECT data FROM cycles')]
    with service.store.connect() as db:
        for cycle in cycles:
            run='station:'+cycle['id']; storage=db.execute('SELECT * FROM capture_storage WHERE run_id=?',(run,)).fetchone()
            if storage and storage['archived']: continue
            rows=[dict(r) for r in db.execute('SELECT * FROM station_records WHERE session=?',(cycle['id'],))]; reason=None
            if cycle['state'] not in TERMINAL: reason='진행 중인 전체·상세 검사'
            elif cycle.get('restored_history'): reason='복원 후 별도 보존'
            elif storage and storage['hold_note']: reason='별도 보존 지정'
            elif cycle['state']!='COMPLETED': reason='중단·실패 회차 확인 필요'
            elif not rows: reason='검사 결과 없는 회차 확인 필요'
            elif db.execute('SELECT 1 FROM station_index_cycles WHERE id=? AND error IS NOT NULL',(cycle['id'],)).fetchone(): reason='이전 기록 색인 필요'
            elif any(t['status']=='SORT_UNCONFIRMED' for t in cycle['targets']): reason='이송 확인 필요'
            elif policy['protect_unreviewed'] and any(not r['review'] and (r['decision']=='REVIEW' or 'NG_UNKNOWN' in json.loads(r['codes'])) for r in rows): reason='미검토 보류·미등록 이상'
            elif any(now-r['created']<policy[{'OK':'days_ok','NG':'days_ng','REVIEW':'days_review'}[r['decision']]]*86400 for r in rows): reason='보존 기간 이내'
            if reason: protected.append({'run_id':run,'reason':reason}); continue
            try:
                validate_cycle(cycle); groups=[]; files={}
                for i,path in enumerate(assets(cycle)):
                    source=managed(path,service.root); listing=inventory(source); relative='assets/'+str(i)
                    groups.append({'source':str(source),'relative':relative,'files':listing})
                    files.update({relative+'/'+name:value for name,value in listing.items()})
                require(groups,'전체·상세 원본이 없습니다.')
                candidates.append({'kind':'station','run_id':run,'cycle_id':cycle['id'],'cycle_revision':cycle['revision'],
                    'source':str(service.root/'station-cycles'/cycle['id']),'digest':fingerprint(cycle),'cycle':cycle,
                    'assets':groups,'files':files,'bytes':sum(v['bytes'] for v in files.values())})
            except Exception as exc: errors.append({'path':run,'error':str(exc)})
    return candidates,protected,errors


def finish(service,identity,data):
    root=plain_path(service.root); archive=plain_path(data['archive_root']); destination=within(data['destination'],archive)
    require(destination.parent==archive and not root.is_relative_to(archive) and not archive.is_relative_to(root),'보관 경로가 잘못됐습니다.')
    require(data['source']==str(root/'station-cycles'/data['cycle_id']) and fingerprint(data['cycle'])==data['digest'],'회차 보관 자료가 변경됐습니다.')
    mappings=[]
    for group in data['assets']:
        source=managed(group['source'],root); target=within(destination/group['relative'],destination); mappings.append((source,target))
    require({(g['relative']+'/'+n):v for g in data['assets'] for n,v in g['files'].items()}==data['files'],'원본 목록이 변경됐습니다.')
    require(set(assets(data['cycle']))=={x[0] for x in mappings},'회차와 원본 폴더 목록이 다릅니다.')
    destination.mkdir(parents=True,exist_ok=True)
    present=inventory(destination)
    require(all(k in data['files'] and data['files'][k]==v for k,v in present.items()),'보관 파일이 변경됐습니다.')
    for group,(source,target) in zip(data['assets'],mappings):
        for name,meta in group['files'].items():
            dest=within(target/name,destination)
            if dest.exists(): require(sha256(dest)==meta['sha256'] and dest.stat().st_size==meta['bytes'],'보관 파일 검증 실패'); continue
            original=within(source/name,source)
            require(sha256(original)==meta['sha256'] and original.stat().st_size==meta['bytes'],'복사 전 원본 변경')
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(original,dest)
    require(inventory(destination)==data['files'],'보관 원본의 무결성을 확인할 수 없습니다.')
    moved=relocate(data['cycle'],mappings); moved['coordinate_valid']=False; moved['pending']=None
    moved['revision']=data['cycle_revision']+1; moved['archive_path']=str(destination)
    validate_cycle(moved); service._intent(identity,'COPIED',data)
    journal=ScanJournal(root); current=journal.read(data['cycle_id'])
    if current!=moved:
        require(current==data['cycle'],'보관 중 검사 회차가 변경됐습니다.')
        # Assets must not be shared with a different cycle before any deletion.
        with journal.connect() as db:
            for other,encoded in db.execute('SELECT id,data FROM cycles WHERE id!=?',(data['cycle_id'],)):
                require(not set(assets(json.loads(encoded))) & {s for s,_ in mappings},'다른 회차가 같은 원본을 사용합니다.')
        journal.save(moved,data['cycle_revision'],'ARCHIVED')
    with service.store.connect() as db:
        db.execute('INSERT INTO capture_storage(run_id,archived) VALUES(?,1) ON CONFLICT(run_id) DO UPDATE SET archived=1',(data['run_id'],))
    with service.queue.connect() as db:
        for source,target in mappings:
            db.execute('UPDATE jobs SET snapshot_path=? WHERE snapshot_path=?',(str(target),str(source)))
    service._intent(identity,'LINKED',data)
    for group,(source,_) in zip(data['assets'],mappings):
        if not source.exists(): continue
        remaining=inventory(source)
        require(all(k in group['files'] and v==group['files'][k] for k,v in remaining.items()),'정리 중 원본 파일이 변경됐습니다.')
        for name in remaining: within(source/name,source).unlink()
        for directory,folders,_ in os.walk(source,topdown=False,followlinks=False):
            for folder in folders: within(Path(directory)/folder,source).rmdir()
        managed(source,root).rmdir()
    service._intent(identity,'COMPLETED',data)
    service.store.event('STATION_ARCHIVED',{'cycle_id':data['cycle_id'],'path':str(destination),'operation_id':identity})
