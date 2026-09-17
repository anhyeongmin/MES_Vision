"""Opt-in connected U20CAM worker/recording check; restores original controls."""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

def main():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    from mes_vision.operation.camera import CameraProcess
    from mes_vision.operation.uvc_camera import UVCCamera
    from mes_vision.operation.capture_controls import directshow_ranges
    from mes_vision.operation.catalog import default_equipment
    if '--hardware' not in sys.argv: raise SystemExit('Use --hardware with the camera available.')
    app=QApplication([]); device=next(d for d in UVCCamera.devices() if 'U20CAM-720P' in d['name'])
    before=directshow_ranges(device['path'])
    restore={'auto_exposure':before['exposure']['automatic'],'exposure':before['exposure']['current'],'gain':before['gain']['current']}
    settings=default_equipment()['camera']; settings.update(serial=device['serial'])
    camera=CameraProcess(settings); result={}; commands={}; finished=False
    def send(kind,payload,action): commands[camera.command(kind,payload)]=action
    def fail(message):
        result['error']=message; camera.stopping.set()
    def reply(event):
        action=commands.pop(event['id'],None)
        if 'error' in event: fail(event['error']); return
        if action=='apply':
            send('record_start',{'root':str(ROOT/'artifacts/capture-studio-check/recordings'),
                'metadata':{'label':'UNKNOWN','product_id':'hardware-check','purpose':'diagnostic'}},'start')
        elif action=='start':
            result['folder']=event['result']['folder']; QTimer.singleShot(2500,lambda:send('record_stop',{},'stop'))
        elif action=='stop':
            result['recording']=json.loads((Path(result['folder'])/'recording.json').read_text(encoding='utf-8'))
            send('controls',dict(restore,auto_exposure=False),'restore_manual')
        elif action=='restore_manual': send('controls',restore,'restore')
        elif action=='restore': camera.stopping.set()
    camera.connected.connect(lambda _:send('controls',dict(auto_exposure=False,exposure=-7,gain=0),'apply'))
    camera.command_finished.connect(reply); camera.failed.connect(fail); camera.finished.connect(app.quit)
    QTimer.singleShot(20000,lambda:fail('Worker check timed out'))
    camera.start(); app.exec()
    if 'error' in result:
        # Fresh process ended, so restoration can safely open a new capture.
        recovery=UVCCamera(settings)
        try:
            recovery.open(); recovery.apply_controls(dict(restore,auto_exposure=False)); recovery.apply_controls(restore)
        finally: recovery.close()
        raise RuntimeError(result['error'])
    import av
    path=Path(result['folder'])/result['recording']['video']
    with av.open(str(path)) as video:
        frames=list(video.decode(video=0))
        duration=float((frames[-1].pts-frames[0].pts)*frames[0].time_base)
    result['decoded_frames']=len(frames); result['measured_video_fps']=(len(frames)-1)/duration
    assert result['recording']['status']=='COMPLETED' and len(frames)==result['recording']['frames']
    assert result['measured_video_fps']>27 and len(frames)>50
    result['restored_controls']=directshow_ranges(device['path'])
    output=ROOT/'artifacts/capture-studio-check/recording-result.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'decoded_frames':len(frames),'measured_video_fps':result['measured_video_fps'],'status':'PASS'},ensure_ascii=False))

if __name__=='__main__': main()
