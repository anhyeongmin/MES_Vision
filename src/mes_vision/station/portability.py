"""Include external station assets; restore historical paths without rearming motion."""
from pathlib import Path
import json,sqlite3,time
from uuid import uuid4
from mes_vision.training.data import read_json,write_json
from .settings import defaults
from .sequence import mark_unconfirmed_sort


def external_assets(runtime):
    root=Path(runtime); paths=[]
    if (root/'station.sqlite3').exists():
        from .sequence import ScanJournal
        from .retention import validate_cycle
        with ScanJournal(root).connect() as db:
            for row in db.execute('SELECT data FROM cycles'):
                cycle=json.loads(row[0])
                if cycle.get('archive_path'):
                    validate_cycle(cycle); paths.append(cycle['archive_path'])
    for path in (root/'station-recipes').glob('*.json'):
        recipe=read_json(path); asset=recipe.get('overview_asset')
        if asset: paths.append(asset['weights'])
    path=root/'station-settings.json'
    if path.exists():
        value=read_json(path); bundle=value.get('calibration_bundle')
        if bundle:
            paths.append(bundle); data=read_json(Path(bundle))['data']
            paths.extend(data[k]['path'] for k in ('capture_map','pick_map'))
    return paths


def restore_station(runtime,relocate):
    root=Path(runtime); path=root/'station.sqlite3'
    if path.exists():
        db=sqlite3.connect(path)
        try:
            with db:
                for identity,encoded in db.execute('SELECT id,data FROM cycles').fetchall():
                    data=relocate(json.loads(encoded)); data.update(coordinate_valid=False,pending=None,generation=data['generation']+1,revision=data['revision']+1)
                    mark_unconfirmed_sort(data)
                    if data['state'] not in {'COMPLETED','CANCELLED','FAILED','INTERRUPTED'}: data['state']='INTERRUPTED'
                    data['restored_history']=True; result=json.dumps(data,ensure_ascii=False,allow_nan=False)
                    db.execute('UPDATE cycles SET revision=?,state=?,data=? WHERE id=?',(data['revision'],data['state'],result,identity))
                    db.execute('INSERT INTO cycle_events(cycle_id,revision,event,data,created) VALUES(?,?,?,?,?)',
                        (identity,data['revision'],'RESTORED_HISTORY',result,time.time()))
        finally: db.close()
    # Search data is a rebuildable projection; keep operator reviews, invalidate
    # projected revisions so every restored path is read from the relocated cycle.
    operation=root/'operation.sqlite3'
    if operation.exists():
        db=sqlite3.connect(operation)
        try:
            with db:
                if db.execute("SELECT 1 FROM sqlite_master WHERE name='station_index_cycles'").fetchone():
                    db.execute('DELETE FROM station_index_cycles')
        finally: db.close()
    path=root/'station-settings.json'
    if path.exists():
        # Retain original installation evidence while requiring explicit registration.
        path.rename(root/('station-settings-restored-'+uuid4().hex+'.json')); write_json(path,defaults())
    for path in (root/'station-recipes').glob('*.json'):
        data=relocate(read_json(path)); data['validation_reference']=''; write_json(path,data)
