"""Reuse the existing pose form, binding its saved profile to both measured maps."""
from dataclasses import asdict
from uuid import uuid4
from pathlib import Path
from mes_vision.operation.setup_dialogs import RobotProfileDialog
from mes_vision.operation.settings_ui import optional_value
from mes_vision.robot import RobotProfile,Workspace,Pose
from mes_vision.training.data import require,write_json
from mes_vision.qt_i18n import ui_text
from mes_vision.i18n import tr
from .settings import StationSettings


class StationRobotProfileDialog(RobotProfileDialog):
    def __init__(self,product,equipment,runtime,parent=None):
        super().__init__(product,equipment,runtime,parent)
        self.calibration=StationSettings(runtime).calibrated()
        require(self.calibration.product_id==product['id'],tr('이 품목의 검증된 실측 보정이 필요합니다.'))
        for w in [*self.fields.values(),*(w for row in self.destinations.values() for w in row),self.age,self.timeout]:
            minimum=w.minimum()-10**(-w.decimals()); w.setMinimum(minimum); w.setValue(minimum); ui_text(w.setSpecialValueText,tr('미등록'))
        self.fields['travel'].setValue(self.calibration.value['travel_z']); self.fields['travel'].setEnabled(False)
        self.age.setMaximum(300)
        ui_text(self.message.setText,tr('촬영·집기 보정에 등록된 집기 높이와 이동 높이를 사용합니다. 모든 경로와 좌표 유효 시간을 실물로 확인하세요.'))
    def save(self):
        try:
            cal=StationSettings(self.runtime).calibrated(); require(cal.digest==self.calibration.digest,'Calibration changed')
            fields={k:optional_value(w) for k,w in self.fields.items()}; destinations={k:[optional_value(w) for w in row] for k,row in self.destinations.items()}
            require(all(v is not None for v in fields.values()) and all(v is not None for row in destinations.values() for v in row)
                and optional_value(self.age) is not None and optional_value(self.timeout) is not None,'Measured robot limits, destinations and deadlines required')
            require(self.equipment['robot']['input_address'] is not None,'Independent tool verification input required')
            profile=RobotProfile(self.version.text().strip(),self.product['id'],'real',True,self.reference.text().strip(),cal.identity,
                cal.value['grasp_version'],Workspace(**{k:v for k,v in fields.items() if k!='travel'}),cal.value['pick_z'],cal.value['travel_z'],
                {k:Pose(*v) for k,v in destinations.items()},'suction' if self.tool.currentIndex()==0 else 'gripper','digital_input',self.age.value(),self.timeout.value(),False)
            require(all(profile.workspace.contains(p) for p in profile.destinations.values()),'Destination outside registered workspace')
            folder=self.runtime/'robot-profiles'; folder.mkdir(exist_ok=True,parents=True); self.path=str(folder/(uuid4().hex+'.json'))
            write_json(Path(self.path),asdict(profile)); self.accept()
        except Exception as exc: ui_text(self.message.setText,str(exc))
