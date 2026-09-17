"""Audit saved inference output and render the actual photo-inspection Qt window."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from mes_vision.training.data import read_json,write_json,require,sha256
from mes_vision.station.photo_inspection import open_results
from mes_vision.vlm.snapshots import load_snapshot
from mes_vision.synthetic.detection_review import summarize


def main():
    directory=ROOT/'artifacts/model-ui-integration'; root,report=open_results(directory/'inference/results.json')
    previous={r['file_name']:r for r in read_json(ROOT/'artifacts/training/ash-reinforced-v3/review/reinforced/predicted_crop_chain-test-predictions.json')}
    rows=[]; mapped=0; objects=0; kinds={}; errors=[]; dispositions={}
    for row in report['records']:
        _,result,_=load_snapshot(root/row['snapshot'],expected_digest=row['snapshot_digest'])
        require(not result['robot_commands_enabled'] and not result['frame']['is_live'],'Unexpected live authority')
        kinds[row['role']]=kinds.get(row['role'],0)+1
        for obj in result['objects']:
            objects+=1; dispositions[obj['final_decision']]=dispositions.get(obj['final_decision'],0)+1
            for check in obj['checks']:
                if check['status']=='ERROR': errors.append({'photo':row['name'],'check':check['check_id']})
                for finding in check['findings']:
                    crop=finding['crop_box']; original=finding['original_box']; bounds=obj['crop']['bounds_original_xyxy']
                    require(original=={'x1':crop['x1']+bounds['x1'],'y1':crop['y1']+bounds['y1'],
                        'x2':crop['x2']+bounds['x1'],'y2':crop['y2']+bounds['y1']},'Defect coordinate shift changed')
                    mapped+=1
        if row['role']=='detail':
            reference=previous[row['name']]; require(row['source_sha256']==reference['image_sha256'],'Different comparison photograph')
            predictions=[{'label':f['defect_code'],'score':f['score'],
                'box':[f['original_box'][k] for k in ('x1','y1','x2','y2')]}
                for o in result['objects'] for c in o['checks'] for f in c['findings']]
            rows.append({'file_name':row['name'],'targets':reference['targets'],'predictions':predictions})
    counts=summarize(rows,.1,[f'NG{i:02d}' for i in range(1,7)])
    write_json(directory/'verification.json',{'status':'SOFTWARE_AND_SAVED_MODEL_VERIFIED','photos':kinds,'objects':objects,
        'mapped_defect_boxes':mapped,'inspection_errors':errors,'dispositions':dispositions,'candidate_metrics':counts,
        'image_origin':'same held-out synthetic CAD photographs as prior development evaluation',
        'physical_camera_tested':False,'robot_connected':False,'production_models_registered':False,
        'operational_policy_validated':False,'vlm_ran':False,'model_bundle_sha256':sha256(ROOT/'configs/inspection/ash-trained-v3.json')})
    require(not errors,'Inspection errors occurred')
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QPoint,Qt
    from PySide6.QtTest import QTest
    from mes_vision.station.window import StationWindow
    from mes_vision.station.photo_dialog import PhotoInspectionDialog
    from mes_vision.qt_i18n import apply_language
    app=QApplication.instance() or QApplication([])
    parent=StationWindow(ROOT,directory/'ui-runtime',start_worker=False); parent.timer.stop()
    window=PhotoInspectionDialog(ROOT,directory/'ui-runtime',parent)
    window.load_results(root/'results.json'); window.show(); app.processEvents()
    cases={'ng01':'reinforce-test-d0000-NG01.png','normal':'reinforce-test-d0000-OK.png',
           'remaining-error':'reinforce-test-d0004-OK.png','overview':'test-o0001.png'}
    for locale in ('ko','en','zh-CN','th'):
        apply_language(app,locale)
        for size in ((1320,820),(1100,700)):
            window.resize(*size); app.processEvents()
            for label,name in cases.items():
                index=next(i for i,r in enumerate(report['records']) if r['name']==name)
                window.files.selectRow(index); app.processEvents()
                require(window.inspection is not None,'No inspection visible')
                if window.inspection['objects']:
                    obj=window.inspection['objects'][-1]; identity=obj['object_id']; box=obj['effective_box']
                    canvas=window.original.canvas; scale,x,y=canvas.transform()
                    QTest.mouseClick(canvas,Qt.LeftButton,pos=QPoint(round(x+(box['x1']+box['x2'])*scale/2),
                                                                   round(y+(box['y1']+box['y2'])*scale/2)))
                    require(window.selected_id==identity,'Image selection did not bind to its result')
                window.grab().save(str(directory/f'photo-{locale}-{size[0]}-{label}.png'))
    window.close(); window.deleteLater(); parent.close(); app.processEvents()
    print('Verified',kinds,'mapped findings',mapped,'candidate totals',counts['total'],flush=True)


if __name__=='__main__': main()
