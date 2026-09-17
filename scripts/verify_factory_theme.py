"""Render real station screens without camera, robot or inference workers."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
import sys,time,tempfile,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from PySide6.QtWidgets import QApplication
from mes_vision.i18n import install_display_font,install_qt_translator
from mes_vision.operation.preferences import Preferences,DEFAULTS,validate
from mes_vision.operation.preferences_dialog import PreferencesDialog
from mes_vision.station.window import StationWindow
from mes_vision.theme import logo_path
from mes_vision.station.settings_dialog import StationSettingsDialog
from mes_vision.station.photo_dialog import PhotoInspectionDialog
from mes_vision.operation.product_dialog import ProductDialog
from mes_vision.operation.catalog import new_product

out=ROOT/'artifacts/ui-theme-alignment/screens';out.mkdir(parents=True,exist_ok=True)
app=QApplication([]);install_display_font(app);install_qt_translator(app)
checks=[]
def check(value,name):
    assert value,name
    checks.append(name)
def pump():
    for _ in range(5):app.processEvents();time.sleep(.01)
with tempfile.TemporaryDirectory() as temp:
    runtime=Path(temp)
    old=dict(DEFAULTS,language='ko',text_scale=100);old.pop('theme')
    (runtime/'preferences.json').write_text(json.dumps(old),encoding='utf-8')
    check(Preferences(runtime).value['theme']=='light','existing preferences migrate')
    window=StationWindow(ROOT,runtime,start_worker=False);window.timer.stop();window.show();pump()
    for mode in ['light','dark']:
        if window.preferences.value['theme']!=mode:window.theme_button.click()
        check(Preferences(runtime).value['theme']==mode,'theme saved '+mode)
        check(window.camera is None and window.robot is None and window.scan is None,'no equipment change '+mode)
        check(not window.brand_logo.pixmap().isNull(),'brand image '+mode)
        for size in [(1440,900),(1100,760)]:
            window.resize(*size);pump()
            for page in range(5):
                window.nav.setCurrentRow(page);pump()
                check(window.pages.currentIndex()==page,'navigation '+mode+str(size)+str(page))
                window.grab().save(str(out/f'{mode}-{size[0]}-page{page}.png'))
            check(window.global_stop.isVisible(),'robot stop visible '+mode+str(size))
            check(window.brand_logo.isVisible() or window.compact_logo.isVisible(),'logo visible '+mode+str(size))
        dialog=PreferencesDialog(Preferences(runtime),runtime,window);dialog.show();pump()
        check(dialog.theme.currentData()==mode,'settings selection '+mode)
        dialog.grab().save(str(out/f'{mode}-settings.png'))
        before=(runtime/'preferences.json').read_bytes();dialog.theme.setCurrentIndex(1-dialog.theme.currentIndex());dialog.reject()
        check(before==(runtime/'preferences.json').read_bytes(),'cancel unchanged '+mode)
        for name, extra in [('station-settings',StationSettingsDialog(runtime,window)),
                            ('photos',PhotoInspectionDialog(ROOT,runtime,window)),
                            ('product',ProductDialog(new_product('Theme check'),window.equipment,ROOT,runtime,window))]:
            extra.show();pump();extra.grab().save(str(out/f'{mode}-{name}.png'));extra.reject()
    # Save through the settings dialog, then use the same apply path as the modal caller.
    prefs=Preferences(runtime);dialog=PreferencesDialog(prefs,runtime,window)
    dialog.theme.setCurrentIndex(0);dialog.save();window.preferences=prefs;window.apply_preferences()
    check(Preferences(runtime).value['theme']=='light','dialog saves theme')
    window.close();pump()
    check(Preferences(runtime).value['theme']=='light','close preserves theme')
(out.parent/'verification.json').write_text(json.dumps({'status':'PASSED','checks':checks,'hardware_used':False},indent=2),encoding='utf-8')
print(f'{len(checks)} checks passed; 20 main screenshots and two settings screenshots.')
