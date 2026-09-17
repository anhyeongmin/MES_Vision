from copy import deepcopy
from PySide6.QtWidgets import QDialog,QVBoxLayout,QFormLayout,QLabel,QLineEdit,QCheckBox,QDialogButtonBox
from mes_vision.qt_i18n import ui_text
from mes_vision.i18n import tr
from mes_vision.training.data import require,read_json
from mes_vision.operation.widgets import PathField,Task
from mes_vision.operation.catalog import model_asset
from .recipe import recipe_path,save_recipe


class RecipeDialog(QDialog):
    def __init__(self,runtime,product,equipment,parent=None):
        super().__init__(parent); self.runtime=runtime; self.product=deepcopy(product); self.equipment=deepcopy(equipment); self.job=None
        path=recipe_path(runtime,product); self.previous=read_json(path) if path.exists() else None
        ui_text(self.setWindowTitle,tr('전체·상세 검사 설정')); self.resize(710,420); layout=QVBoxLayout(self); form=QFormLayout(); layout.addLayout(form)
        old=self.previous or {}
        self.overview=PathField((old.get('overview_asset') or {}).get('weights'))
        ui_text(form.addRow,tr('전체사진 물체 검출 모델'),self.overview)
        note=ui_text(QLabel,tr('품목 설정에 등록한 모델과 판정 기준은 상세사진에 사용합니다. 전체사진은 위치 검출만 수행합니다.'))
        note.setWordWrap(True); layout.addWidget(note)
        self.domains=ui_text(QCheckBox,tr('등록된 품목 모델·판정 기준을 근접 상세사진으로 검증했습니다.')); layout.addWidget(self.domains)
        self.full=ui_text(QCheckBox,tr('상세사진 전체 영역을 검사 영역으로 사용합니다.')); layout.addWidget(self.full)
        self.reference=QLineEdit(old.get('validation_reference','')); ui_text(form.addRow,tr('상세 촬영 검증 기록'),self.reference)
        # A previous version is visible but never silently accepted for edited product/equipment.
        same=old.get('product_version')==product['version'] and old.get('equipment_version')==equipment['version']
        from mes_vision.operation.acquisition import identity
        same= same and old.get('acquisition_identity')==identity(equipment['camera'])
        self.domains.setChecked(same); self.full.setChecked(same)
        self.message=QLabel(); self.message.setWordWrap(True); layout.addWidget(self.message); layout.addStretch()
        self.buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        ui_text(self.buttons.button(QDialogButtonBox.Save).setText,tr('저장')); ui_text(self.buttons.button(QDialogButtonBox.Cancel).setText,tr('취소'))
        self.buttons.accepted.connect(self.save); self.buttons.rejected.connect(self.reject); layout.addWidget(self.buttons)
    def save(self):
        try:
            require(self.job is None and self.domains.isChecked() and self.full.isChecked()
                and self.overview.value() and self.reference.text().strip(),tr('전체 모델과 상세 검사 검증 항목을 등록하세요.'))
            path=self.overview.value(); reference=self.reference.text().strip(); p=self.product; e=self.equipment
            def write():
                from mes_vision.operation.acquisition import inspection_camera, identity
                asset=model_asset(path,role='objects'); c=inspection_camera(e['camera']); w,h=c['width'],c['height']
                value={'schema_version':1,'product_id':p['id'],'product_version':p['version'],'equipment_version':e['version'],
                    'acquisition_identity':identity(e['camera']),
                    'overview_asset':asset,'image_size':[w,h],'validation_reference':reference,
                    'capture_domains':{'overview_objects':'overview',**{role:'detail' for role in ('objects','defects','anomaly','geometry') if p.get(role)}},
                    'detail_workspace':{'roi':[[0,0],[w,0],[w,h],[0,h]],'excluded':[],'validation_reference':reference}}
                return save_recipe(self.runtime,value,p,e,previous=self.previous)
            self.job=Task(write); self.buttons.setEnabled(False); ui_text(self.message.setText,tr('모델 파일과 촬영 기준을 확인하고 있습니다.'))
            self.saved=False; self.job.succeeded.connect(lambda _:setattr(self,'saved',True)); self.job.failed.connect(self.failed)
            self.job.finished.connect(self.finished_job); self.job.start()
        except Exception as exc: self.failed(str(exc))
    def failed(self,message): self.message.setText(str(message)); self.buttons.setEnabled(True)
    def finished_job(self):
        self.job.deleteLater(); self.job=None
        if self.saved: self.accept()
    def reject(self):
        if self.job is None: super().reject()
    def closeEvent(self,event):
        if self.job is not None: event.ignore()
        else: super().closeEvent(event)
