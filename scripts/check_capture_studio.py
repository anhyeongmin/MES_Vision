"""UI render check; --hardware also checks exposure/gain and restores controls."""
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

def main():
    import tempfile
    from unittest.mock import Mock,patch
    from PySide6.QtWidgets import QApplication,QWidget
    from mes_vision.operation.capture_dialog import CaptureDialog
    from mes_vision.operation.catalog import default_equipment,OperationStore,new_product
    from mes_vision.theme import apply_theme
    out=ROOT/'artifacts'/'capture-studio-check'; out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([])
    from mes_vision.i18n import install_display_font
    install_display_font(app); apply_theme('light',1)
    ranges={'exposure':dict(min=-13,max=-1,step=1,default=-6,manual=True,auto=True),
            'gain':dict(min=0,max=100,step=1,default=0,manual=True,auto=False)}
    result={'hardware_checked':False}
    if '--hardware' in sys.argv:
        from mes_vision.operation.uvc_camera import UVCCamera
        from mes_vision.operation.capture_controls import directshow_ranges
        devices=UVCCamera.devices(); device=next(d for d in devices if 'U20CAM-720P' in d['name'])
        ranges=directshow_ranges(device['path'])
        settings=default_equipment()['camera']; settings.update(serial=device['serial'],pixel_format='MJPG',fps=30,auto_exposure=ranges['exposure']['automatic'],exposure=ranges['exposure']['current'])
        camera=UVCCamera(settings)
        restore=dict(auto_exposure=ranges['exposure']['automatic'],exposure=ranges['exposure']['current'],gain=ranges['gain']['current'])
        try:
            camera.open()
            camera.apply_controls(dict(auto_exposure=False,exposure=-7,gain=1))
            frame=camera.read(); assert frame.width==1280 and frame.height==720
            after=directshow_ranges(device['path'])
            assert after['exposure']['current']==-7 and not after['exposure']['automatic'] and after['gain']['current']==1
            import time
            stamps=[]
            for _ in range(75): camera.read(); stamps.append(time.monotonic())
            measured=59/(stamps[-1]-stamps[-60]); assert measured>27
            result.update(hardware_checked=True,device=device['name'],ranges=ranges,frame_size=[frame.width,frame.height],
                requested_format='MJPG',verified_format='MJPG',requested_fps=30,measured_receive_fps=measured)
        finally:
            try:
                if camera.capture:
                    camera.apply_controls(dict(restore,auto_exposure=False))
                    camera.apply_controls(restore)
            finally: camera.close()
        result['restored']=directshow_ranges(device['path'])
    with tempfile.TemporaryDirectory() as root,patch.object(CaptureDialog,'connect_when_ready'):
        owner=QWidget(); owner.root=ROOT; owner.store=OperationStore(root); owner.equipment=default_equipment()
        owner.product=owner.store.save_product(new_product('ASH')); owner.camera=None; owner.refresh_equipment=Mock()
        dialog=CaptureDialog(owner); dialog.ranges=ranges; dialog.load_profile(); dialog.show(); app.processEvents()
        dialog.grab().save(str(out/'camera-settings.png'))
        dialog.tabs.setCurrentIndex(1); app.processEvents(); dialog.grab().save(str(out/'collection.png'))
        dialog.close(); app.processEvents(); owner.close()
    (out/('hardware-result.json' if '--hardware' in sys.argv else 'ui-result.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
