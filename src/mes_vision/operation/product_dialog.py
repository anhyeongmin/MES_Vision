from mes_vision.qt_i18n import ui_text
from mes_vision.i18n import text_join
from mes_vision.i18n import tr, trf
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QLineEdit,QPlainTextEdit,QLabel,
    QComboBox,QTabWidget,QWidget,QMessageBox,QFileDialog,QInputDialog,QDialogButtonBox)
from mes_vision.training.data import require,read_json,write_json
from mes_vision.anomaly.features import fingerprint
from .catalog import valid_product,model_asset
from .widgets import button,number,integer,PathField,Task
from .settings_ui import SettingsPages,add_review_controls,optional_number,optional_value


class ProductDialog(QDialog):
    def __init__(self,product,equipment,root,runtime,parent=None):
        super().__init__(parent); self.original=deepcopy(product); self.value=deepcopy(product); self.equipment=equipment; self.root=Path(root); self.runtime=Path(runtime); self.task=None
        ui_text(self.setWindowTitle, tr('품목 설정')); self.resize(900,780); layout=QVBoxLayout(self)
        self.pages=SettingsPages(layout); form=self.pages.form
        general=form(tr('품목 · 수량'))
        self.identity=QLineEdit(product["id"]); self.identity.setReadOnly(product["version"]>0); ui_text(general.addRow, tr('품목 ID'),self.identity)
        self.name=QLineEdit(product["name"]); ui_text(general.addRow, tr('품목 이름'),self.name)
        self.active=QComboBox(); ui_text(self.active.addItems, [tr('사용'),tr('비활성')]); self.active.setCurrentIndex(0 if product["active"] else 1); ui_text(general.addRow, tr('사용 상태'),self.active)
        self.count_mode=QComboBox(); ui_text(self.count_mode.addItems, [tr('자유 수량'),tr('고정 수량')]); self.count_mode.setCurrentIndex(product["count_mode"]=="fixed"); ui_text(general.addRow, tr('수량 조건'),self.count_mode)
        self.expected=integer(product["expected_count"] or 1,1,100); ui_text(general.addRow, tr('고정 수량'),self.expected)
        ui_text(general.addRow, ui_text(QLabel, tr('설정은 저장 후 품목을 적용할 때 활성화됩니다. 검사 중에는 변경할 수 없습니다.')))
        ui_text(general.addRow, ui_text(QLabel, tr('작업 환경 연결: ')+equipment["id"]+tr(' · 저장 시 이 환경에 연결합니다.')))
        models=form(tr('모델 · 정상 기준'),advanced=True); self.model_labels={}
        for role,label in (("objects",tr('물체 검출 모델')),("defects",tr('불량 검출 모델'))):
            row=QWidget(); h=QHBoxLayout(row); h.setContentsMargins(0,0,0,0)
            text=ui_text(QLabel); text.setWordWrap(True); self.model_labels[role]=text; h.addWidget(text,1)
            h.addWidget(button(tr('등록'),lambda _,r=role:self.import_model(r))); h.addWidget(button(tr('해제'),lambda _,r=role:self.clear_model(r))); ui_text(models.addRow, label,row)
        self.bank=PathField((product["anomaly"] or {}).get("bank"),folder=True); ui_text(models.addRow, tr('정상 특징 폴더'),self.bank)
        self.anomaly_criteria=PathField((product["anomaly"] or {}).get("criteria"),filter=tr('검사 기준 (*.json)')); ui_text(models.addRow, tr('이상 점수 기준'),self.anomaly_criteria)
        ui_text(models.addRow, button(tr('정상·이상 점수 기준 작성'),self.build_anomaly_criteria))
        self.geometry=PathField(product["geometry"],filter=tr('형상 기준 (*.json)')); ui_text(models.addRow, tr('형상·누락 검사 기준'),self.geometry)
        ui_text(models.addRow, button(tr('형상 검사 기준 작성'),self.build_geometry))
        self.reference=PathField(product["normal_reference"],folder=True); ui_text(models.addRow, tr('검토된 정상 참조 폴더'),self.reference)
        quality=form(tr('촬영 · 판정 기준'),advanced=True)
        q=product["quality"]; self.quality_version=QLineEdit(q["version"]); self.validation=QLineEdit(q["validation_reference"])
        ui_text(quality.addRow, tr('촬영 기준 버전'),self.quality_version); ui_text(quality.addRow, tr('현장 검증 기록'),self.validation)
        self.blur=optional_number(q["blur_min"],0,100000); self.brightness_min=optional_number(q["brightness_min"],0,255); self.brightness_max=optional_number(q["brightness_max"],0,255)
        ui_text(quality.addRow, tr('선명도 하한'),self.blur); ui_text(quality.addRow, tr('평균 밝기 하한'),self.brightness_min); ui_text(quality.addRow, tr('평균 밝기 상한'),self.brightness_max)
        self.policy=PathField(product["policy"],filter=tr('판정 기준 (*.json)')); ui_text(quality.addRow, tr('활성 판정 기준 파일'),self.policy)
        ui_text(quality.addRow, button(tr('검증 기록으로 판정 기준 작성'),self.build_policy))
        ui_text(quality.addRow, ui_text(QLabel, tr('파일 등록은 검증 완료가 아닙니다. 실물로 확인한 조건과 검증 기록을 입력하세요.')))
        grasp=form(tr('집기 기준'),advanced=True)
        g=product["grasp"]; self.grasp_version=QLineEdit(g["version"]); self.grasp_reference=QLineEdit(g["validation_reference"])
        self.grasp_u=number(g["u"],.01,.99,3); self.grasp_v=number(g["v"],.01,.99,3)
        self.grasp_z=optional_number(g["plane_z_mm"],-1000,1000,3); self.grasp_r=number(g["rotation_deg"],-180,180)
        for label,field in ((tr('집기 기준 버전'),self.grasp_version),(tr('검증 기록'),self.grasp_reference),(tr('물체 내 가로 비율'),self.grasp_u),
                            (tr('물체 내 세로 비율'),self.grasp_v),(tr('검사면 높이 Z(mm)'),self.grasp_z),(tr('집기 방향(도)'),self.grasp_r)): ui_text(grasp.addRow, label,field)
        ui_text(grasp.addRow, ui_text(QLabel, tr('현재 집기 기준은 검증된 고정 방향·물체 내부 비율입니다. 임의 방향 제품은 그대로 운전하지 마세요.')))
        vlm=form(tr('추가 분석'))
        self.vlm_mode=QComboBox(); ui_text(self.vlm_mode.addItems, [tr('외관 판단 보류만 자동 분석'),tr('NG와 외관 판단 보류 분석'),tr('수동 요청만')])
        self.vlm_mode.setCurrentIndex(["review_visual","ng_and_review","manual"].index(product["vlm_policy"])); ui_text(vlm.addRow, tr('자동 전달 조건'),self.vlm_mode)
        self.vlm_text=QPlainTextEdit(product["vlm_criteria"]); ui_text(vlm.addRow, tr('품목 검사 기준 설명'),self.vlm_text)
        self.count_mode.currentIndexChanged.connect(lambda index:self.expected.setEnabled(index==1)); self.expected.setEnabled(self.count_mode.currentIndex()==1)
        add_review_controls(self,layout,self.original,self.collect,'product',product['id'])
        self.message=ui_text(QLabel); self.message.setWordWrap(True); layout.addWidget(self.message)
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel); buttons.accepted.connect(self.save); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
        self.render_models()
    def render_models(self):
        if hasattr(self,'change_preview'): self.change_preview.hide()
        for role,label in self.model_labels.items():
            s=self.value[role]; ui_text(label.setText, tr('미등록') if s is None else Path(s["weights"]).name+"\n"+text_join(', ', s["class_names"]))
    def clear_model(self,role): self.value[role]=None; self.render_models()
    def import_model(self,role):
        if self.task and self.task.isRunning(): return
        path=QFileDialog.getOpenFileName(self,tr('제품 학습 모델 선택'),"",tr('학습 모델 (*.pth *.pt)'))[0]
        if not path: return
        ui_text(self.message.setText, tr('학습 모델의 클래스와 무결성을 확인하고 있습니다…'))
        self.task=Task(lambda:model_asset(path,role="objects"))
        def loaded(value):
            if role=="defects":
                codes={}
                for i,name in enumerate(value["class_names"]):
                    code,ok=QInputDialog.getItem(self,tr('불량 유형 연결'),name+tr('에 해당하는 불량 코드'),[tr('NG01 누락'),tr('NG02 돌출·버'),tr('NG03 균열'),tr('NG04 변형'),tr('NG05 구멍'),tr('NG06 표면')],editable=False)
                    if not ok: ui_text(self.message.setText, tr('불량 모델 등록을 취소했습니다.')); return
                    codes[str(i)]=code.split()[0]
                value["class_codes"]=codes
            self.value[role]=value; self.render_models(); ui_text(self.message.setText, tr('모델을 등록했습니다. 판정 기준과 검증 기록은 별도로 연결하세요.'))
        self.task.succeeded.connect(loaded); self.task.failed.connect(self.message.setText); self.task.start()
    def collect(self):
        p=deepcopy(self.value); p.update(id=self.identity.text().strip(),name=self.name.text().strip(),active=self.active.currentIndex()==0,
            count_mode="free" if self.count_mode.currentIndex()==0 else "fixed",expected_count=None if self.count_mode.currentIndex()==0 else self.expected.value(),
            workspace_id=self.equipment["id"],anomaly={"bank":self.bank.value(),"criteria":self.anomaly_criteria.value()} if self.bank.value() else None,
            geometry=self.geometry.value(),normal_reference=self.reference.value(),policy=self.policy.value(),
            quality={"version":self.quality_version.text().strip(),"validation_reference":self.validation.text().strip(),"blur_min":optional_value(self.blur),
                     "brightness_min":optional_value(self.brightness_min),"brightness_max":optional_value(self.brightness_max)},
            grasp={"version":self.grasp_version.text().strip(),"validation_reference":self.grasp_reference.text().strip(),"u":self.grasp_u.value(),
                   "v":self.grasp_v.value(),"rotation_deg":self.grasp_r.value(),"plane_z_mm":optional_value(self.grasp_z)},
            vlm_policy=["review_visual","ng_and_review","manual"][self.vlm_mode.currentIndex()],vlm_criteria=self.vlm_text.toPlainText().strip())
        for key,fields in (('quality',('blur_min','brightness_min','brightness_max')),('grasp',('u','v','plane_z_mm','rotation_deg'))):
            old=self.original[key]
            if old['version'] and any(old[field]!=p[key][field] for field in fields):
                require(p[key]['version'] and p[key]['version']!=old['version'],
                    tr('촬영 · 판정 기준' if key=='quality' else '집기 기준')+': '+tr('검증된 기준값을 변경했습니다. 재검증 후 해당 기준 버전을 새로 입력하세요.'))
        valid_product(p); return p
    def save(self):
        if self.task and self.task.isRunning(): ui_text(self.message.setText, tr('모델 확인이 끝난 뒤 저장하세요.')); return
        try: self.value=self.collect(); self.accept()
        except Exception as exc: ui_text(self.message.setText, str(exc))
    def generated(self,name,data):
        folder=self.runtime/"registered"/fingerprint(self.identity.text().strip()); folder.mkdir(parents=True,exist_ok=True)
        path=folder/(name+"-"+uuid4().hex+".json"); write_json(path,data); return str(path)
    def build_anomaly_criteria(self):
        try:
            require(self.bank.value(),tr('정상 특징 폴더를 먼저 선택하세요.'))
            meta=read_json(Path(self.bank.value())/"bank.json"); require(meta["kind"]=="real" and meta["product_id"]==self.identity.text().strip(),tr('이 품목의 실제 정상 특징이 필요합니다.'))
            dialog=QDialog(self); ui_text(dialog.setWindowTitle, tr('정상·이상 점수 기준')); f=QFormLayout(dialog)
            version=QLineEdit(); reference=QLineEdit(); lower=number(0,0,2,4); upper=number(0,0,2,4); pixel=number(0,0,2,4)
            for label,w in ((tr('기준 버전'),version),(tr('실물 검증 기록'),reference),(tr('정상 상한'),lower),(tr('불량 하한'),upper),(tr('의심 영역 하한'),pixel)): ui_text(f.addRow, label,w)
            ui_text(f.addRow, button(tr('검증 기준 등록'),dialog.accept))
            if dialog.exec()!=QDialog.Accepted: return
            from mes_vision.anomaly.scoring import Criteria
            data=Criteria(version.text(),fingerprint(meta),meta["product_id"],lower.value(),upper.value(),pixel.value(),True,reference.text(),"real")
            ui_text(self.anomaly_criteria.edit.setText, self.generated("anomaly",asdict(data)))
        except Exception as exc: ui_text(self.message.setText, str(exc))
    def build_geometry(self):
        from .setup_dialogs import GeometryDialog
        dialog=GeometryDialog(self.identity.text().strip(),self)
        if dialog.exec()==QDialog.Accepted: ui_text(self.geometry.edit.setText, self.generated("geometry",dialog.value))
    def build_policy(self):
        try:
            p=self.collect(); require(p["objects"] and p["defects"] and p["anomaly"] and p["anomaly"]["criteria"],tr('물체·불량 모델과 정상·이상 기준을 먼저 등록하세요.'))
            dialog=QDialog(self); ui_text(dialog.setWindowTitle, tr('판정 기준 등록')); f=QFormLayout(dialog)
            version=QLineEdit(); reference=QLineEdit(); reason=QLineEdit(); ui_text(f.addRow, tr('기준 버전'),version); ui_text(f.addRow, tr('실물 검증 기록'),reference)
            ui_text(f.addRow, tr('형상 검사 제외 사유(사용하지 않을 때)'),reason)
            thresholds={}
            for code in sorted(set(p["defects"]["class_codes"].values())):
                thresholds[code]=number(p["defects"]["threshold"],p["defects"]["threshold"],1,3); ui_text(f.addRow, code+tr(' 불량 확정 하한'),thresholds[code])
            ui_text(f.addRow, ui_text(QLabel, tr('실물 검증에서 확인한 값을 입력하세요. 모델 등록만으로 통과 기준이 생기지는 않습니다.')))
            ui_text(f.addRow, button(tr('검증 기록으로 등록'),dialog.accept))
            if dialog.exec()!=QDialog.Accepted: return
            from mes_vision.inspection import ModelRef
            from mes_vision.decision import DecisionPolicy,CheckRule
            from mes_vision.anomaly.scoring import Criteria
            from .quality import GeometryInspector
            criteria=Criteria(**read_json(Path(p["anomaly"]["criteria"])))
            require(criteria.validated and criteria.kind=="real" and criteria.product_id==p["id"],tr('이 품목의 검증된 이상 기준이 필요합니다.'))
            require(criteria.bank_digest==fingerprint(read_json(Path(p["anomaly"]["bank"])/"bank.json")),tr('정상 특징과 이상 기준이 다릅니다.'))
            require(p["quality"]["version"] and p["quality"]["validation_reference"],tr('촬영 품질 기준과 검증 기록을 먼저 입력하세요.'))
            ref=reference.text().strip(); v=version.text().strip()
            from mes_vision.inspection.rfdetr_adapter import model_reference
            def model(role): return model_reference(p[role]["sha256"],"product_"+role)
            rules=[CheckRule("known_defects",True,"defect_candidates",model=model("defects"),criteria_version=v,validated=True,validation_reference=ref,
                candidate_threshold=p["defects"]["threshold"],fail_thresholds={k:w.value() for k,w in thresholds.items()},class_codes=p["defects"]["class_codes"]),
                CheckRule("anomaly",True,"anomaly_distance",model=ModelRef("DINOv2 normal-reference anomaly","1","model",criteria.bank_digest,"product_normal_reference"),
                criteria_version=criteria.version,validated=True,validation_reference=criteria.validation_reference,criteria_digest=fingerprint(asdict(criteria)))]
            if p["geometry"]:
                geometry=GeometryInspector(p["geometry"]); rules.append(CheckRule("geometry",True,model=geometry.model,criteria_version=geometry.config["version"],validated=True,validation_reference=geometry.config["validation_reference"]))
            else: rules.append(CheckRule("geometry",False,optional_reason=reason.text().strip() or None))
            policy=DecisionPolicy(v,p["id"],tuple(rules),"real",True,ref,model("objects"),p["quality"]["version"],p["count_mode"]=="fixed")
            ui_text(self.policy.edit.setText, self.generated("policy",asdict(policy))); ui_text(self.message.setText, tr('판정 기준을 등록했습니다. 품목 설정을 저장하세요.'))
        except Exception as exc: ui_text(self.message.setText, str(exc))
    def reject(self):
        if self.task and self.task.isRunning(): ui_text(self.message.setText, tr('모델 확인 작업이 끝난 뒤 닫으세요.')); return
        super().reject()
