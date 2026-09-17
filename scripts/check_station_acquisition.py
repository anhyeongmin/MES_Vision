"""Report the new acquisition context without connecting to camera or robot."""
from pathlib import Path
import argparse,json,sqlite3,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    from mes_vision.operation.catalog import default_equipment
    from mes_vision.operation.acquisition import configure_equipment,inspection_camera,Acquisition,identity
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,default=ROOT/'artifacts/operation')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/ui-backend-handoff-v1/acquisition-context.json')
    args=parser.parse_args(); database=args.runtime/'operation.sqlite3'
    e=default_equipment()
    if database.exists():
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
            row=db.execute('SELECT data FROM equipment ORDER BY version DESC LIMIT 1').fetchone()
        if row: e=json.loads(row[0])
    configured=configure_equipment(ROOT,e); camera=configured['camera']; effective=inspection_camera(camera)
    acquisition=Acquisition(camera)  # Checks the lens asset; never opens a device.
    report={'status':'SOFTWARE_CONTEXT_ONLY','robot_commands_enabled':False,
        'sensor_mode':{k:camera.get(k) for k in ('driver','serial','width','height','fps','pixel_format')},
        'inspection_size':[effective['width'],effective['height']],
        'acquisition_identity':identity(camera),
        'expected_measured_map_context':{'camera_id':camera['serial'],'mount_revision':camera['mount_revision'],
            'acquisition_revision':effective['acquisition_revision'],'robot_base_id':camera['robot_base_id'],
            'tool_frame_id':camera['tool_frame_id'],'image_size':[effective['width'],effective['height']],'transformations':[]},
        'required_map_lens_mode':'none_verified' if acquisition.rectifier else 'measured_lens_model',
        'workspace':configured['workspace'],
        'workspace_confirmation_required':not bool(configured['workspace']['validation_reference']),
        'physical_calibration_approved':False,
        'note':'Expected context is not a measured or accepted robot calibration. No equipment records were modified.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(args.output); print('Sensor:',report['sensor_mode']); print('Inspection:',report['inspection_size'])
    print('No hardware opened; no calibration accepted.')

if __name__=='__main__': main()
