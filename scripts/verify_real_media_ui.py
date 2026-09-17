"""Exercise the actual video dialog/worker using a generated, declared fixture."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QPushButton
from PySide6.QtGui import QFont, QFontDatabase
from mes_vision.data_management.collection import Collection
from mes_vision.data_management.labeler import Labeler
from mes_vision.i18n import set_language, tr
from mes_vision.training.data import write_json

app = QApplication.instance() or QApplication([])
font_path = Path('C:/Windows/Fonts/malgun.ttf')
if font_path.exists():
    QFontDatabase.addApplicationFont(str(font_path))
    app.setFont(QFont('Malgun Gothic', 9))
output = ROOT / 'artifacts/real-media-preparation'
output.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    video = root / 'fixture.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (320, 240))
    assert writer.isOpened()
    for index in range(25):
        frame = np.full((240, 320, 3), (45, 75, 65), dtype=np.uint8)
        cv2.rectangle(frame, (40 + index, 50), (170 + index, 190), (230, 230, 230), -1)
        writer.write(frame)
    writer.release()
    collection = Collection.create(root / 'collection', 'SYNTHETIC-UI-FIXTURE', kind='synthetic')
    window = Labeler(collection.root)
    window.show()
    with patch('mes_vision.data_management.labeler.QFileDialog.getOpenFileName', return_value=(str(video), '')), \
         patch('mes_vision.data_management.labeler.QInputDialog.getText', return_value=('fixture-session', True)), \
         patch('mes_vision.data_management.labeler.QInputDialog.getDouble', return_value=(1.0, True)), \
         patch('mes_vision.data_management.labeler.QInputDialog.getInt', return_value=(3, True)):
        next(b for b in window.findChildren(QPushButton) if b.text() == '영상 추가').click()
    deadline = time.monotonic() + 20
    while window.worker.isRunning() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert not window.worker.isRunning(), 'Video import did not finish'
    app.processEvents()
    assert window.central.isEnabled()
    assert len(window.collection.data['records']) == 3
    assert window.captures.count() == 3
    assert all(not r['reviewed'] for r in window.collection.data['records'])
    window.grab().save(str(output / 'video-import-ko.png'))
    window.close()
    for locale in ['en', 'zh-CN', 'th']:
        set_language(locale)
        translated = Labeler(collection.root)
        translated.show()
        app.processEvents()
        assert any(b.text() == str(tr('영상 추가')) for b in translated.findChildren(QPushButton))
        translated.grab().save(str(output / f'video-import-{locale}.png'))
        translated.close()
write_json(output / 'ui-verification.json', {'status': 'PASSED', 'generated_fixture_frames_imported': 3,
           'worker_finished': True, 'reviewed_automatically': False, 'locales': ['ko', 'en', 'zh-CN', 'th'],
           'physical_camera_used': False})
print('Video button, background worker, collection reload and four locale windows passed.')
