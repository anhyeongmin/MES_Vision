"""Rebuildable search projection; reviews never modify inspection or motion data."""
from pathlib import Path
from copy import deepcopy
import json,time,sqlite3
from mes_vision.training.data import require
from mes_vision.vlm.snapshots import load_snapshot
from .sequence import ScanJournal,TERMINAL
from .worker import read_capture


def schema(db):
    db.executescript('''
      CREATE TABLE IF NOT EXISTS station_records(object_id TEXT PRIMARY KEY,run_id TEXT,track_id TEXT,revision INTEGER,
        decision TEXT,prior_object_id TEXT,created REAL,review TEXT,path TEXT,digest TEXT,session TEXT,
        product_id TEXT,product_name TEXT,product_version INTEGER,equipment_version INTEGER,codes TEXT,
        robot_state TEXT,source_object_id TEXT,data TEXT);
      CREATE TABLE IF NOT EXISTS station_index_cycles(id TEXT PRIMARY KEY,revision INTEGER,error TEXT);
      CREATE INDEX IF NOT EXISTS station_record_time ON station_records(created,object_id);
      CREATE INDEX IF NOT EXISTS station_record_cycle ON station_records(session);
    ''')


def sync(store,*,verify=False):
    if not (store.root/'station.sqlite3').exists(): return
    journal=ScanJournal(store.root)
    with journal.connect() as source:
        cycles=source.execute('SELECT id,revision FROM cycles').fetchall()
        with store.connect() as db:
            seen={r[0]:r[1] for r in db.execute('SELECT id,revision FROM station_index_cycles WHERE error IS NULL')}
            for identity,revision in cycles:
                if not verify and seen.get(identity)==revision: continue
                cycle=json.loads(source.execute('SELECT data FROM cycles WHERE id=?',(identity,)).fetchone()[0])
                created=source.execute('SELECT MIN(created) FROM cycle_events WHERE cycle_id=?',(identity,)).fetchone()[0]
                p=cycle['profile']; product=db.execute('SELECT data FROM products WHERE id=? AND version=?',(p['product_id'],p['product_version'])).fetchone()
                name=json.loads(product[0])['name'] if product else p['product_id']; errors=[]
                for target in cycle['targets']:
                    if target.get('decision') not in {'OK','NG','REVIEW'}: continue
                    receipt=target.get('result') or {}; codes=[]; error=None
                    prior=db.execute('SELECT digest,codes,data FROM station_records WHERE object_id=?',(target['id'],)).fetchone()
                    try:
                        if receipt.get('snapshot_path'):
                            if not verify and prior and prior['digest']==receipt['snapshot_digest'] and not json.loads(prior['data']).get('index_error'): codes=json.loads(prior['codes'])
                            else:
                                _,saved,_=load_snapshot(receipt['snapshot_path'],expected_digest=receipt['snapshot_digest'])
                                obj=next(o for o in saved['objects'] if o['object_id']==receipt['object_id'])
                                require(obj['final_decision']==target['decision'],'Saved station decision differs')
                                codes=obj.get('decision_details',{}).get('defect_codes',[])
                    except Exception as exc: error=str(exc); errors.append(error)
                    status=target['status']; robot={'SORTED':'RECAPTURE','SORTING':'REQUESTED','SORT_UNCONFIRMED':'RECOVERY'}.get(status,'NONE')
                    payload={'cycle_id':identity,'profile':p,'overview':cycle['overview'],'target':target,'cycle_state':cycle['state'],'index_error':error}
                    values=(target['id'],'station:'+identity,target['id'],1,target['decision'],None,created,None,
                        receipt.get('snapshot_path'),receipt.get('snapshot_digest'),identity,p['product_id'],name,p['product_version'],p['equipment_version'],
                        json.dumps(codes),robot,receipt.get('object_id'),json.dumps(payload,ensure_ascii=False))
                    db.execute('''INSERT INTO station_records VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(object_id) DO UPDATE SET
                        decision=excluded.decision,path=excluded.path,digest=excluded.digest,codes=excluded.codes,robot_state=excluded.robot_state,
                        source_object_id=excluded.source_object_id,data=excluded.data,product_name=excluded.product_name,created=excluded.created''',values)
                db.execute('INSERT OR REPLACE INTO station_index_cycles VALUES(?,?,?)',(identity,cycle['revision'],'; '.join(errors) or None))


def get_row(store,identity):
    sync(store)
    with store.connect() as db: row=db.execute('SELECT * FROM station_records WHERE object_id=?',(identity,)).fetchone()
    return dict(row) if row else None


def open_record(store,identity,*,row=None):
    row=get_row(store,identity) if row is None else row; require(row is not None,'검사 이력을 찾을 수 없습니다.')
    data=json.loads(row['data']); target=data['target']
    read_capture(data['overview'])
    if target.get('detail'): read_capture(target['detail'])
    if row['path']:
        manifest,saved,_=load_snapshot(row['path'],expected_digest=row['digest'])
        expected={'cycle_id':data['cycle_id'],'target_id':identity,'overview_frame_id':data['overview']['frame_id'],'detail_frame_id':target['detail']['frame_id']}
        require(saved['config'].get('station_link')==expected,'Station evidence link differs')
        from PIL import Image
        import hashlib
        with Image.open(Path(row['path'])/'frame.png') as image:
            require(image.mode=='RGB' and hashlib.sha256(image.tobytes()).hexdigest()==target['detail']['rgb_sha256'],'Detailed evidence pixels differ')
        obj=next(o for o in saved['objects'] if o['object_id']==row['source_object_id'])
        require(obj['final_decision']==row['decision'],'Station verdict differs from evidence')
        return row,manifest,saved,obj
    require(target['decision']=='REVIEW','A definite decision requires detailed evidence')
    read_capture(target.get('detail') or data['overview'])
    return row,None,None,None


def review(store,identity,operator,note):
    from mes_vision.operation.storage import StorageService
    require(isinstance(operator,str) and operator.strip() and isinstance(note,str) and note.strip(),'확인자와 검토 내용을 입력하세요.')
    with StorageService(store).exclusive():
        row,*_=open_record(store,identity)
        value={'operator':operator.strip(),'note':note.strip(),'at':time.time(),'snapshot_digest':row['digest']}
        with store.connect() as db:
            db.execute('UPDATE station_records SET review=? WHERE object_id=?',(json.dumps(value,ensure_ascii=False),identity))
            db.execute('INSERT INTO events(created,session,track_id,type,data) VALUES(?,?,?,?,?)',
                (value['at'],row['session'],identity,'OPERATOR_REVIEW',json.dumps({'object_id':identity,**value},ensure_ascii=False)))


def assets(cycle):
    """Cycle asset folders, deduplicated; full overview shared by all objects."""
    result=[]
    if cycle.get('overview'): result.append(Path(cycle['overview']['path']).parent)
    for target in cycle['targets']:
        if target.get('detail'): result.append(Path(target['detail']['path']).parent)
        if (target.get('result') or {}).get('snapshot_path'): result.append(Path(target['result']['snapshot_path']))
    return list(dict.fromkeys(result))


def relocate(value,mappings):
    if isinstance(value,str):
        candidate=Path(value)
        for source,destination in mappings:
            if candidate.is_absolute() and (candidate==source or candidate.is_relative_to(source)):
                target=destination/candidate.relative_to(source)
                return str(target) if target.is_absolute() else target.as_posix()
        return value
    if isinstance(value,list): return [relocate(v,mappings) for v in value]
    if isinstance(value,dict): return {k:relocate(v,mappings) for k,v in value.items()}
    return value


def export_record(service,row,staging,*,images,cache,db):
    from mes_vision.operation.storage import copy_tree_verified
    current,manifest,saved,obj=open_record(service.store,row['object_id'],row=dict(row,data=row['station_data'])); data=json.loads(current['data']); cycle=data['cycle_id']
    if cycle not in cache: cache[cycle]={'files':[],'mappings':[]}
    item=cache[cycle]
    if images:
        for source in assets({'overview':data['overview'],'targets':[data['target']]}):
            if any(source==existing for existing,_ in item['mappings']): continue
            relative=Path('station-evidence')/cycle/str(len(item['mappings']))
            copy_tree_verified(source,staging/relative); item['mappings'].append((source,relative)); item['files'].append(relative.as_posix())
    # Export report paths are package-relative; immutable original snapshot contents stay unchanged.
    report_data=relocate(data,item['mappings']) if images else data
    jobs=[dict(r) for r in db.execute('SELECT id,state,payload,result,error FROM analysis.jobs WHERE object_id=? ORDER BY created',(row['source_object_id'],))] if row['source_object_id'] else []
    events=[dict(r) for r in db.execute("SELECT created,type,data FROM events WHERE json_extract(data,'$.object_id')=? ORDER BY id",(row['object_id'],))]
    return {'record':row,'station':report_data,'evidence':item['files'],'inspection':obj,
        'configuration':saved['config'] if saved else data['profile'],'decision_policy':saved['decision_details'] if saved else None,
        'vlm':jobs,'events':events,'has_detail_inspection':saved is not None}
