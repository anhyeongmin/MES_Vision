"""Human-confirmed SAM mask editor, isolated from live inspection."""
from pathlib import Path
import json,shutil,sys,tempfile
from uuid import uuid4
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt,Signal,QProcess,QTimer
from PySide6.QtGui import QImage,QPixmap,QPen,QColor
from PySide6.QtWidgets import QDialog,QVBoxLayout,QLabel,QComboBox,QPushButton
from mes_vision.qt_i18n import ui_text
from mes_vision.i18n import tr,trf
from mes_vision.operation.responsive import FlowLayout
from mes_vision.training.data import sha256,require
from .labeler import Canvas
from .sam_assist import mask_box,validate_prompt

ROOT=Path(__file__).resolve().parents[3]


class PromptCanvas(Canvas):
    prompt_clicked=Signal(object,int)
    def __init__(self):
        super().__init__();self.prompt_mode=0
    def mousePressEvent(self,event):
        if self.record and self.prompt_mode in (0,1) and event.button()==Qt.LeftButton:
            p=self.mapToScene(event.position().toPoint())
            if 0<=p.x()<self.record['width'] and 0<=p.y()<self.record['height']:
                self.prompt_clicked.emit([p.x(),p.y()],1 if self.prompt_mode==0 else 0)
            event.accept();return
        super().mousePressEvent(event)


class SamDialog(QDialog):
    def __init__(self,image_path,image_sha256,parent=None,runtime=None):
        super().__init__(parent)
        self.image_path=Path(image_path);self.image_sha256=image_sha256
        require(sha256(self.image_path)==image_sha256,'Source image changed')
        with Image.open(image_path) as image:self.width_px,self.height_px=image.size
        self.runtime=Path(runtime) if runtime else ROOT/'artifacts/operation'
        self.temporary=tempfile.TemporaryDirectory(prefix='mes-sam-')
        self.workdir=Path(self.temporary.name);self.points=[];self.labels=[];self.box=None
        self.process=None;self.result=None;self.result_dir=None;self.accepted_proposal=None
        self.cancelled=False;self.pending_close=False;self.output_log=''
        ui_text(self.setWindowTitle,tr('SAM 라벨링 보조'));self.resize(1150,800)
        layout=QVBoxLayout(self);row=FlowLayout();layout.addLayout(row)
        self.mode=QComboBox();ui_text(self.mode.addItems,[tr('포함 클릭'),tr('제외 클릭'),tr('상자 지정'),tr('이동')]);row.addWidget(self.mode)
        self.clear_button=ui_text(QPushButton,tr('입력 초기화'));row.addWidget(self.clear_button)
        self.run_button=ui_text(QPushButton,tr('마스크 생성'));row.addWidget(self.run_button)
        self.stop_button=ui_text(QPushButton,tr('취소'));self.stop_button.setEnabled(False);row.addWidget(self.stop_button)
        self.candidates=QComboBox();self.candidates.setMinimumWidth(210);row.addWidget(self.candidates)
        self.use_button=ui_text(QPushButton,tr('선택 마스크 적용'));self.use_button.setEnabled(False);row.addWidget(self.use_button)
        note=ui_text(QLabel,tr('물체 전체를 포함하는지 확인하세요. SAM 점수는 정답 확률이나 OK·NG 판정이 아닙니다.'));note.setWordWrap(True);layout.addWidget(note)
        self.canvas=PromptCanvas();layout.addWidget(self.canvas,1)
        self.canvas.show_record({'width':self.width_px,'height':self.height_px,'objects':[]},image_path)
        self.status=ui_text(QLabel,tr('포함할 부분을 클릭하거나 상자를 지정하세요.'));self.status.setWordWrap(True);layout.addWidget(self.status)
        self.mode.currentIndexChanged.connect(self.change_mode);self.clear_button.clicked.connect(self.clear_prompts)
        self.run_button.clicked.connect(self.start);self.stop_button.clicked.connect(self.cancel)
        self.use_button.clicked.connect(self.use);self.candidates.currentIndexChanged.connect(self.redraw)
        self.canvas.prompt_clicked.connect(self.point);self.canvas.box_drawn.connect(self.set_box)
        self.timeout=QTimer(self);self.timeout.setSingleShot(True);self.timeout.timeout.connect(self.cancel)

    def change_mode(self,index):
        self.canvas.prompt_mode=index;self.canvas.set_drawing(index==2)

    def invalidate(self):
        self.result=None;self.use_button.setEnabled(False);self.candidates.clear()

    def point(self,point,label):
        if len(self.points)>=64:return
        self.points.append(point);self.labels.append(label);self.invalidate();self.redraw()

    def set_box(self,box):
        self.box=box;self.invalidate();self.redraw()

    def clear_prompts(self):
        self.points=[];self.labels=[];self.box=None;self.invalidate();self.redraw()

    def redraw(self,*_):
        # Keep pan/zoom while replacing proposal overlays, never source pixels.
        self.canvas.show_record(self.canvas.record)
        if self.result and self.candidates.currentIndex()>=0:
            candidate=self.result['candidates'][self.candidates.currentIndex()]
            mask=np.asarray(Image.open(self.result_dir/candidate['mask']))>0
            rgba=np.zeros((*mask.shape,4),dtype=np.uint8);rgba[mask]=[40,190,245,100]
            image=QImage(rgba.data,self.width_px,self.height_px,4*self.width_px,QImage.Format_RGBA8888).copy()
            item=self.canvas.scene().addPixmap(QPixmap.fromImage(image));self.canvas.layers.append(item)
        for point,label in zip(self.points,self.labels):
            color=QColor('#00bd82' if label else '#ed5353');pen=QPen(color,2);pen.setCosmetic(True)
            item=self.canvas.scene().addEllipse(point[0]-4,point[1]-4,8,8,pen,color);self.canvas.layers.append(item)
        if self.box:
            x1,y1,x2,y2=self.box;pen=QPen(QColor('#56a9f9'),2);pen.setCosmetic(True)
            self.canvas.layers.append(self.canvas.scene().addRect(x1,y1,x2-x1,y2-y1,pen))

    def busy(self,enabled):
        for widget in (self.mode,self.clear_button,self.run_button,self.canvas,self.candidates):widget.setEnabled(not enabled)
        self.stop_button.setEnabled(enabled);self.use_button.setEnabled(not enabled and bool(self.result))

    def start(self):
        if self.process is not None:return
        request={'image':str(self.image_path),'image_sha256':self.image_sha256,'points':self.points,'labels':self.labels,
                 'box':self.box,'runtime':str(self.runtime),'output':str(self.workdir/uuid4().hex)}
        try:validate_prompt(request,self.width_px,self.height_px)
        except Exception as exc:self.status.setText(str(exc));return
        self.invalidate();self.cancelled=False;self.output_log='';self.result_dir=Path(request['output'])
        path=self.workdir/'request.json';path.write_text(json.dumps(request),encoding='utf-8')
        process=QProcess(self);self.process=process;process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self.read_output);process.finished.connect(self.process_finished)
        process.errorOccurred.connect(self.process_error)
        self.busy(True);ui_text(self.status.setText,tr('SAM 처리 중 · 처음에는 모델을 불러옵니다.'))
        self.timeout.start(180000);process.start(sys.executable,[str(ROOT/'scripts/sam_proposal.py'),str(path)])

    def read_output(self):
        if self.process:self.output_log=(self.output_log+bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace'))[-6000:]

    def process_error(self,error):
        if error==QProcess.FailedToStart:self.process_finished(-1,QProcess.CrashExit)

    def process_finished(self,code,status):
        if self.process is None:return
        self.timeout.stop();self.read_output();process=self.process;self.process=None;process.deleteLater()
        if not self.cancelled and code==0:
            try:
                result=json.loads((self.result_dir/'result.json').read_text(encoding='utf-8'))
                require(result['image_sha256']==self.image_sha256 and sha256(self.image_path)==self.image_sha256,'Source changed')
                for candidate in result['candidates']:
                    path=(self.result_dir/candidate['mask']).resolve()
                    require(path.is_relative_to(self.result_dir) and sha256(path)==candidate['sha256'],'Invalid mask artifact')
                    with Image.open(path) as image:mask=np.asarray(image)>0
                    require(mask.shape==(self.height_px,self.width_px) and mask_box(mask)==candidate['box'],'Invalid mask coordinates')
                self.result=result
                for index,candidate in enumerate(result['candidates']):self.candidates.addItem(f"{index+1} · SAM {candidate['model_score']:.3f}")
                ui_text(self.status.setText,tr('후보를 확인하고 선택 마스크 적용을 누르세요.'));self.redraw()
            except Exception as exc:self.invalidate();self.status.setText(str(exc))
        else:
            self.invalidate();self.status.setText(str(tr('SAM 작업이 취소되었거나 실패했습니다.'))+'\n'+self.output_log[-1200:])
        self.busy(False)
        if self.pending_close:self.reject()

    def cancel(self):
        if self.process is not None:
            self.cancelled=True;self.process.kill()

    def use(self):
        if self.process is not None or not self.result:return
        candidate=self.result['candidates'][self.candidates.currentIndex()]
        self.accepted_proposal={'candidate':candidate,'result':self.result,'directory':str(self.result_dir)}
        self.accept()

    def reject(self):
        if self.process is not None:
            self.pending_close=True;self.cancel();return
        super().reject()

    def closeEvent(self,event):
        if self.process is not None:self.pending_close=True;self.cancel();event.ignore()
        else:super().closeEvent(event)

    def cleanup(self):
        require(self.process is None,'SAM process still running');self.temporary.cleanup()


def retain_proposal(collection,proposal):
    """Preserve the accepted mask and model provenance; annotation remains unreviewed."""
    identity=uuid4().hex;folder=collection.root/'assists'/identity;folder.mkdir(parents=True)
    candidate=proposal['candidate'];source=Path(proposal['directory'])/candidate['mask']
    require(sha256(source)==candidate['sha256'],'Mask changed before acceptance')
    shutil.copyfile(source,folder/'mask.png')
    (folder/'proposal.json').write_text(json.dumps(proposal['result'],ensure_ascii=False,indent=2),encoding='utf-8')
    return {'kind':'sam_mask_to_box','mask':(folder/'mask.png').relative_to(collection.root).as_posix(),
            'mask_sha256':candidate['sha256'],'proposal':(folder/'proposal.json').relative_to(collection.root).as_posix(),
            'original_box':candidate['box'],'reviewed_automatically':False}
