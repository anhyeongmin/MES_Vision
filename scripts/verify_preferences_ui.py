"""Isolated Qt preference/language smoke test; no device or model connection."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))


def main():
    from mes_vision.i18n import LANGUAGES
    parser=argparse.ArgumentParser(); parser.add_argument("--language",choices=LANGUAGES,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--next-language",choices=LANGUAGES)
    parser.add_argument("--startup-runtime",type=Path)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    from mes_vision.operation.preferences import Preferences,DEFAULTS
    from mes_vision.i18n import set_language,tr,install_qt_translator,install_display_font,validate_catalog,qt_button_text,catalog
    next_language=args.next_language or ("en" if args.language=="ko" else "ko")
    runtime=args.startup_runtime or args.output/"runtime"; p=Preferences(runtime)
    if not args.startup_runtime: p.save(dict(DEFAULTS,language=args.language))
    if p.value["language"]!=args.language: raise AssertionError("persisted startup language mismatch")
    set_language(Preferences(runtime).value["language"]); validate_catalog()
    from PySide6.QtWidgets import QApplication,QDialog,QPushButton,QLabel,QDialogButtonBox
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QRawFont
    from mes_vision.operation.window import DesktopWindow
    from mes_vision.operation.preferences_dialog import PreferencesDialog
    from mes_vision.operation.product_dialog import ProductDialog
    from mes_vision.operation.catalog import new_product
    app=QApplication([]); install_qt_translator(app)
    install_display_font(app)
    window=DesktopWindow(ROOT,runtime,start_worker=False); window.show()
    def pump():
        for _ in range(8): app.processEvents(); time.sleep(.01)
    pump(); checks=[]
    def check(condition,name):
        if not condition: raise AssertionError(name)
        checks.append(name)
    check(window.settings_button.text()==tr("환경설정"),"settings button translated")
    check(window.start.text()==tr("검사 시작"),"inspection button translated")
    check(window.nav.item(0).text()==tr("1  실시간 검사"),"navigation translated")
    check(all(window.nav.fontMetrics().horizontalAdvance(window.nav.item(i).text())+30<=window.nav.width() for i in range(window.nav.count())),"navigation text fits")
    if args.language in ("zh-CN","th"):
        start,end=(0x4e00,0x9fff) if args.language=="zh-CN" else (0x0e00,0x0e7f)
        characters={c for text in catalog(args.language).values() for c in text if start<=ord(c)<=end}
        raw=QRawFont.fromFont(app.font())
        missing=[c for c in characters if not raw.supportsCharacter(ord(c))]
        check(bool(characters) and not missing,"primary font covers all translated script characters: "+str(len(characters)))
    if args.startup_runtime:
        check(window.pages.currentIndex()==p.value["start_page"],"saved start page applied on next launch")
        check("font-size:17px" in window.styleSheet(),"saved text size applied on next launch")
        check(not window.running and window.camera is None and window.robot is None and window.engine is None,"startup does not connect equipment")
        window.grab().save(str(args.output/"startup.png")); window.close()
        deadline=time.monotonic()+5
        while window.isVisible() and time.monotonic()<deadline: pump()
        check(not window.isVisible(),"clean shutdown")
        (args.output/"report.json").write_text(json.dumps({"passed":True,"checks":checks,"language":args.language},indent=2),encoding="utf-8")
        return
    window.grab().save(str(args.output/"inspection.png"))
    # Exercise the actual header button/modal entry point, including Cancel.
    def cancel_popup():
        dialog=app.activeModalWidget(); check(isinstance(dialog,PreferencesDialog),"header opens modal preferences")
        check(all(dialog.buttons.button(role).text()==qt_button_text(label) for role,label in ((QDialogButtonBox.Save,"Save"),(QDialogButtonBox.Cancel,"Cancel"),(QDialogButtonBox.RestoreDefaults,"Restore Defaults"))),"standard dialog buttons translated")
        check({dialog.language.itemData(i):dialog.language.itemText(i) for i in range(dialog.language.count())}==LANGUAGES,"all four language options available")
        check(dialog.language.currentData()==args.language,"current language selected")
        dialog.grab().save(str(args.output/"settings.png"))
        dialog.language.setCurrentIndex(dialog.language.findData(next_language)); dialog.reject()
    original=p.path.read_bytes(); QTimer.singleShot(150,cancel_popup); window.settings_button.click(); pump()
    check(p.path.read_bytes()==original,"cancel leaves disk untouched")
    def save_popup():
        dialog=app.activeModalWidget(); dialog.language.setCurrentIndex(dialog.language.findData(next_language))
        dialog.scale.setCurrentIndex(2); dialog.start_page.setCurrentIndex(3); dialog.remember.setChecked(False); dialog.save()
    QTimer.singleShot(100,save_popup); window.settings_button.click(); pump()
    stored=Preferences(runtime).value
    check(stored["text_scale"]==130 and stored["start_page"]==3 and not stored["remember_window"] and stored["language"]==next_language,"save persists preferences")
    check(window.pages.currentIndex()==0 and window.start.text()==tr("검사 시작"),"language/start page deferred until restart")
    check("font-size:17px" in window.styleSheet(),"text size applied immediately")
    check(not window.running and window.engine is None and window.camera is None and window.robot is None and not window.queue.enabled(),"no operating authority changed")
    for index in range(window.pages.count()):
        window.nav.setCurrentRow(index); pump()
        window.grab().save(str(args.output/f"page-{index+1}-large.png"))
    window.nav.setCurrentRow(0)
    product=new_product(); product["name"]="정상"; original_product=deepcopy(product)
    dialog=ProductDialog(product,window.equipment,ROOT,runtime,window); dialog.show(); pump()
    check(dialog.name.text()=="정상" and product==original_product,"user-entered product name unchanged")
    dialog.grab().save(str(args.output/"product.png")); dialog.reject()
    # Reopen Settings at the larger text size for overflow inspection.
    settings=PreferencesDialog(Preferences(runtime),runtime,window); settings.show(); pump()
    settings.grab().save(str(args.output/"settings-large.png")); settings.reject()
    window.close()
    deadline=time.monotonic()+5
    while window.isVisible() and time.monotonic()<deadline: pump()
    check(not window.isVisible(),"clean shutdown")
    check(Preferences(runtime).value["language"]==next_language,"shutdown preserves next-launch language")
    report={"passed":True,"language":args.language,"checks":checks,"hardware_connected":False}
    (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(report))


if __name__=="__main__": main()
