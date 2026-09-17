from mes_vision.theme import color as theme_color
from mes_vision.qt_i18n import ui_text
from mes_vision.i18n import text_join
from mes_vision.i18n import tr, trf
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import time
from uuid import uuid4
from PySide6.QtWidgets import QWidget,QHBoxLayout,QLabel,QInputDialog,QFileDialog
from mes_vision.training.data import require,write_json
from .widgets import button
from .shell import styled
from .faults import fault_code
from .storage import check_free_space


class RecoveryControls:
    def make_recovery_controls(self,outer):
        """Fault strip shown only while an operating fault is unresolved.

        Diagnostics export stays reachable at all times, so it moves to the page
        header instead of living inside the hidden strip.
        """
        self.fault_banner=styled(QWidget()); self.fault_banner.setObjectName('faultBanner')
        row=QHBoxLayout(self.fault_banner); row.setContentsMargins(12,7,12,7); row.setSpacing(10)
        self.fault_label=ui_text(QLabel); self.fault_label.setWordWrap(True); self.fault_label.setMaximumHeight(80)
        row.addWidget(self.fault_label,1)
        self.ack_fault_button=button(tr('오류 확인 · 재개 준비'),lambda:self.guarded(self.acknowledge_faults)); row.addWidget(self.ack_fault_button)
        self.diagnostics_button=button(tr('진단 내보내기'),lambda:self.guarded(self.export_diagnostics))
        header=getattr(self,'header',None)
        if header is not None:
            self.diagnostics_button.setObjectName('statusPill')
            header.actions.insertWidget(max(0,header.actions.count()-1),self.diagnostics_button)
        else:
            row.addWidget(self.diagnostics_button)
        outer.addWidget(self.fault_banner); self.refresh_faults()

    def refresh_faults(self):
        active=self.faults.active
        ui_text(self.fault_label.setText, tr('운전 차단 · ')+text_join(' / ', (e["detail"][:240] for e in active[:2]))+(trf(' · 외 {v0}건 (진단 내보내기에서 확인)', v0=len(active) - 2) if len(active)>2 else "") if active else tr('미해결 운전 오류 없음'))
        if self.faults.persistence_error: ui_text(self.fault_label.setText, self.fault_label.text()+tr(' · 오류 기록 저장/읽기 실패: ')+self.faults.persistence_error)
        self.fault_label.setStyleSheet(f"color:{theme_color('danger' if active else 'muted')};"); self.ack_fault_button.setEnabled(bool(active))
        if hasattr(self,'fault_banner'):
            self.fault_banner.setVisible(bool(active) or bool(self.faults.persistence_error))

    def trip_fault(self,code,message,*,stop_engine=True):
        # First remove authority to publish/dispatch, even if all persistence is unavailable.
        self.running=False; self.auto.setChecked(False); self.generation+=1; self.robot_pending=False; self.pick_block=None
        for track in self.tracks: track.update(result=None,status="LOST")
        self.canvas.tracks=[]; self.canvas.update(); self.record_key=None; self.select_track(self.selected_id)
        self.start.setEnabled(False); self.pick_button.setEnabled(False); self.recheck_button.setEnabled(False)
        if self.robot:
            try: self.robot.stop_motion()
            except Exception: pass
        if self.engine and stop_engine:
            self.engine_ready=False
            try: self.engine.stop()
            except Exception: pass
            if self.stop_at is None: self.stop_at=time.monotonic()
        entry,new=self.faults.raise_fault(code,message)
        if new:
            try: self.store.event("OPERATOR_FAULT",entry,session=self.session)
            except Exception: pass
        ui_text(self.status.setText, str(message)); self.refresh_faults()

    def acknowledge_faults(self):
        self.require_idle(camera=True,robot=True)
        require(not self.queue.enabled(),tr('오류 확인 전에 VLM을 OFF로 설정하세요.'))
        require(self.faults.active,tr('확인할 오류가 없습니다.'))
        check_free_space(self.store)
        with self.store.connect() as db: require(db.execute("PRAGMA quick_check").fetchone()[0]=="ok",tr('운영 기록 손상을 먼저 복원하세요.'))
        # A real write checks recovery of permission / read-only / full-disk faults.
        self.store.event("FAULT_RECOVERY_STORAGE_PROBE",{"active_faults":[e["id"] for e in self.faults.active]})
        operator,ok=QInputDialog.getText(self,tr('오류 확인'),tr('확인자'))
        if not ok: return
        note,ok=QInputDialog.getMultiLineText(self,tr('오류 확인'),tr('원인과 조치 내용 (장치 해제·설치 상태 확인 후 재연결, 검사 시작은 별도)'))
        if not ok: return
        identities=[e['id'] for e in self.faults.active]
        self.faults.acknowledge(operator,note)
        try: self.store.event("OPERATOR_FAULT_ACK",{"ids":identities,"operator":operator,"note":note})
        except Exception as exc:
            self.trip_fault("STORAGE_FAILURE",tr('오류 조치 기록을 완료하지 못했습니다: ')+str(exc)); raise
        self.refresh_faults()
        self.notice(tr('오류 조치를 기록했습니다. 모델·장치를 다시 준비하고 검사 시작을 누르세요.'))

    def export_diagnostics(self):
        destination=QFileDialog.getSaveFileName(self,tr('진단 파일 저장'),str(self.runtime.parent/("MES-diagnostics-"+datetime.now().strftime("%Y%m%d-%H%M%S")+".json")),"JSON (*.json)")[0]
        if not destination: return
        path=Path(destination); require(not path.exists(),tr('진단은 새로운 파일 이름으로 저장하세요.'))
        data={"schema_version":1,"created_utc":datetime.now(timezone.utc).isoformat(),"platform":platform.platform(),"runtime":str(self.runtime),
            "faults":self.faults.entries,"fault_persistence_error":self.faults.persistence_error,"generation":self.generation,"running":self.running,
            "camera_connected":bool(self.camera_info),"camera_frame_age_seconds":time.monotonic()-self.frame_received if self.frame_received else None,
            "engine_ready":self.engine_ready,"robot_state":self.robot_state,"session":self.session}
        try:
            with self.store.connect() as db: data["recent_events"]=[dict(r) for r in db.execute("SELECT * FROM events ORDER BY id DESC LIMIT 200")]
        except Exception as exc: data["database_error"]=str(exc)
        try: data["vlm_enabled"]=self.queue.enabled()
        except Exception as exc: data["vlm_error"]=str(exc)
        with path.open("x",encoding="utf-8") as stream: json.dump(data,stream,ensure_ascii=False,indent=2,default=str)
        ui_text(self.status.setText, tr('진단 기록 저장됨: ')+str(path))
