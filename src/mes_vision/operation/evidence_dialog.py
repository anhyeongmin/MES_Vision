from mes_vision.qt_i18n import ui_text
"""Read-only full-resolution evidence and the normal reference saved at inspection time."""
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog,QVBoxLayout,QLabel,QTabWidget,QWidget,QSplitter,QDialogButtonBox
from mes_vision.i18n import tr,trf
from mes_vision.vlm.snapshots import load_snapshot,object_record
from mes_vision.training.data import require
from .image_view import ImagePanel


def bounds(box): return [box[k] for k in ('x1','y1','x2','y2')] if isinstance(box,dict) else list(box)


def close_evidence(owner):
    dialog=getattr(owner,'evidence_window',None)
    if dialog is not None: dialog.close()


def show_evidence(owner,directory,object_id,expected_digest):
    dialog=EvidenceDialog(directory,object_id,expected_digest=expected_digest,parent=owner)
    close_evidence(owner); owner.evidence_window=dialog
    dialog.setAttribute(Qt.WA_DeleteOnClose)
    def finished(_):
        if getattr(owner,'evidence_window',None) is dialog: owner.evidence_window=None
    dialog.finished.connect(finished)
    # Keep the main inspection and stop controls available while reviewing evidence.
    dialog.setModal(False); dialog.show()
    return dialog


class EvidenceDialog(QDialog):
    def __init__(self,directory,object_id,*,expected_digest,parent=None):
        super().__init__(parent); ui_text(self.setWindowTitle, tr('검사 이미지 자세히 보기')); self.resize(1250,820)
        root=Path(directory)
        # Reload the saved evidence with its registered digest; never substitute current product settings.
        manifest,inspection,_=load_snapshot(root,expected_digest=expected_digest)
        record=object_record(manifest,inspection,object_id)
        source=next(item for item in manifest['objects'] if item['object_id']==object_id)
        layout=QVBoxLayout(self)
        self.identity=ui_text(QLabel, trf('검사 물체: {identity}',identity=object_id)); self.identity.setWordWrap(True); layout.addWidget(self.identity)
        self.tabs=QTabWidget(); layout.addWidget(self.tabs,1)
        self.full=ImagePanel(); self.full.canvas.set_file(root/'frame.png')
        self.full.canvas.tracks=[{'track_id':object_id,'box':bounds(record['effective_box']),
            'status':record['final_decision'] or tr('판정 연결 전')}]
        self.full.canvas.selected_id=object_id
        ui_text(self.tabs.addTab, self.full,tr('원본 전체'))
        comparison=QWidget(); compare_layout=QVBoxLayout(comparison); split=QSplitter(); compare_layout.addWidget(split,1)
        def pane(title):
            widget=QWidget(); inner=QVBoxLayout(widget); inner.setContentsMargins(0,0,0,0)
            label=ui_text(QLabel, title); label.setWordWrap(True); inner.addWidget(label)
            image=ImagePanel(); inner.addWidget(image,1); split.addWidget(widget); return image
        self.inspected=pane(tr('검사한 물체')); self.normal=pane(tr('검사 당시 정상 기준'))
        self.inspected.canvas.set_file(root/source['file'])
        findings=[]
        for check in record['checks']:
            if check['check_id']=='vlm': continue
            for finding in check['findings']:
                if finding.get('crop_box'):
                    findings.append({'track_id':finding.get('defect_code') or finding['label'],
                        'box':bounds(finding['crop_box']),'status':'NG' if check['status']=='FAIL' else 'REVIEW'})
        self.inspected.canvas.tracks=findings
        policy=self.normal.overlays.sizePolicy(); policy.setRetainSizeWhenHidden(True); self.normal.overlays.setSizePolicy(policy)
        self.normal.overlays.hide(); reference=manifest['reference']
        self.reference_status=ui_text(QLabel); self.reference_status.setWordWrap(True); compare_layout.addWidget(self.reference_status)
        if reference:
            self.normal.canvas.set_file(root/reference['file'])
            ui_text(self.reference_status.setText, trf('저장된 정상 기준: {collection} · 버전 {revision}',collection=reference['collection_id'],revision=reference['collection_revision']))
        else:
            self.normal.canvas.placeholder=tr('이 검사에는 정상 기준 이미지가 저장되어 있지 않습니다.')
            ui_text(self.reference_status.setText, tr('이 검사에는 정상 기준 이미지가 저장되어 있지 않습니다.'))
        require(not self.full.canvas.pixmap.isNull() and not self.inspected.canvas.pixmap.isNull()
                and (not reference or not self.normal.canvas.pixmap.isNull()),tr('저장 이미지를 읽을 수 없습니다.'))
        split.setSizes([600,600]); ui_text(self.tabs.addTab, comparison,tr('정상 기준 비교')); self.tabs.setCurrentIndex(1)
        hint=ui_text(QLabel, tr('각 이미지를 따로 확대·이동할 수 있습니다. 위치나 크기를 자동 정렬한 비교가 아닙니다.'))
        hint.setWordWrap(True); layout.addWidget(hint)
        buttons=QDialogButtonBox(QDialogButtonBox.Close); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
