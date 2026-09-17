"""Actual local SAM/QProcess test using a declared synthetic photograph."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
import sys,time,json
from unittest.mock import patch,Mock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from PySide6.QtWidgets import QApplication,QDialog
from PySide6.QtCore import Qt,QPointF
from PySide6.QtTest import QTest
from mes_vision.i18n import install_display_font,install_qt_translator
from mes_vision.theme import apply_theme
from mes_vision.data_management.sam_dialog import SamDialog
from mes_vision.data_management.labeler import Labeler
from mes_vision.data_management.collection import Collection
from mes_vision.training.data import sha256

out=ROOT/'artifacts/sam-labeler/ui-final';out.mkdir(parents=True,exist_ok=False)
app=QApplication([]);install_display_font(app);install_qt_translator(app);apply_theme('dark')
source=ROOT/'artifacts/preprint-olive-white-v1/raw/olive-medium-d1-OK.png'
collection=Collection.create(out/'collection','SYNTHETIC-SAM-CHECK',kind='synthetic')
identity=collection.import_image(source,'sam-test-session')
record=collection.record(identity)
dialog=SamDialog(collection.image_path(record),record['image_sha256'],runtime=ROOT/'artifacts/operation')
dialog.show()
def pump():
    for _ in range(5):app.processEvents();time.sleep(.005)
pump();dialog.canvas.fitInView(dialog.canvas.sceneRect(),Qt.KeepAspectRatio)
dialog.mode.setCurrentIndex(2)
a=dialog.canvas.mapFromScene(QPointF(380,100));b=dialog.canvas.mapFromScene(QPointF(900,630))
QTest.mousePress(dialog.canvas.viewport(),Qt.LeftButton,pos=a)
QTest.mouseMove(dialog.canvas.viewport(),b)
QTest.mouseRelease(dialog.canvas.viewport(),Qt.LeftButton,pos=b)
assert dialog.box and not dialog.use_button.isEnabled()
dialog.run_button.click();deadline=time.monotonic()+180;events=0
while dialog.process is not None and time.monotonic()<deadline:
    app.processEvents();events+=1;time.sleep(.01)
assert dialog.process is None,'SAM child timeout'
assert dialog.result,dialog.status.text()
assert dialog.use_button.isEnabled()
# Refine the first mask with real positive and negative clicks, preserving the box.
for mode,point in [(0,(600,350)),(1,(845,470))]:
    dialog.mode.setCurrentIndex(mode)
    position=dialog.canvas.mapFromScene(QPointF(*point))
    QTest.mouseClick(dialog.canvas.viewport(),Qt.LeftButton,pos=position)
assert dialog.labels==[1,0] and not dialog.use_button.isEnabled()
dialog.run_button.click();deadline=time.monotonic()+180
while dialog.process is not None and time.monotonic()<deadline:
    app.processEvents();events+=1;time.sleep(.01)
assert dialog.process is None and dialog.result,dialog.status.text()
dialog.grab().save(str(out/'sam-dark.png'))
apply_theme('light');pump();dialog.grab().save(str(out/'sam-light.png'))
dialog.use_button.click();assert dialog.accepted_proposal
# Exercise the real Labeler acceptance/mutation/save path with the just-created mask.
labeler=Labeler(collection.root);labeler.show();pump()
fake=Mock();fake.exec.return_value=QDialog.Accepted;fake.accepted_proposal=dialog.accepted_proposal
with patch('mes_vision.data_management.sam_dialog.SamDialog',return_value=fake),patch('mes_vision.data_management.labeler.QInputDialog.getText',return_value=('SYNTHETIC-001',True)):
    labeler.sam_object.click()
assert len(labeler.draft['objects'])==1
obj=labeler.draft['objects'][0]
assert obj['condition']=='UNREVIEWED' and not labeler.draft['reviewed']
assert obj['bbox']==dialog.accepted_proposal['candidate']['box']
assert labeler.save()
saved=Collection(collection.root).record(identity)['objects'][0]
assert sha256(collection.root/saved['annotation_assist']['mask'])==saved['annotation_assist']['mask_sha256']
labeler.grab().save(str(out/'labeler-accepted.png'))
labeler.close();dialog.cleanup();dialog.close();pump()
# Cancellation ends the actual process and does not create an accepted proposal.
cancel=SamDialog(source,sha256(source));cancel.set_box([380,100,900,630]);cancel.start();cancel.cancel()
deadline=time.monotonic()+10
while cancel.process is not None and time.monotonic()<deadline:pump()
assert cancel.process is None and cancel.result is None and cancel.accepted_proposal is None
cancel.cleanup();cancel.close()
(out/'verification.json').write_text(json.dumps({'status':'PASSED','actual_sam_inference':True,
    'synthetic_photo':str(source),'qt_event_iterations':events,'box':saved['bbox'],
    'mask_retained':True,'automatic_review':False,'actual_child_cancellation':True,
    'production_models_changed':False},indent=2),encoding='utf-8')
print('Actual SAM dialog, original-coordinate box, label acceptance/save and cancellation passed.')
