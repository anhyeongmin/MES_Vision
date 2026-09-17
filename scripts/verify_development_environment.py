"""Local runtime checks. Synthetic fixtures are NOT product accuracy evidence.

Run without arguments for isolated component checks, or --component NAME to rerun
one failed component. Model assets are checked immediately before loading.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import gc
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts' / 'development-check'
ART.mkdir(parents=True, exist_ok=True)
for key, folder in {'HF_HOME':'huggingface','TORCH_HOME':'torch','MPLCONFIGDIR':'matplotlib','RF_HOME':'rfdetr'}.items():
    os.environ[key] = str(ROOT/'.cache'/folder)
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1', TOKENIZERS_PARALLELISM='false')
sys.stdout.reconfigure(encoding='utf-8')

def check_assets(prefix):
    from prepare_selected_assets import validate, target_path
    lock = json.loads((ROOT/'models/selection-lock.json').read_bytes())
    selected = [a for a in lock['assets'] if a['id'].startswith(prefix)]
    if not selected:
        raise RuntimeError('No pinned assets for ' + prefix)
    for asset in selected:
        validate(target_path(asset['path']), asset)

def torch_ready():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.manual_seed(23)
    torch.set_num_threads(4)
    return torch

def fixture_images():
    from PIL import Image, ImageDraw
    paths=[]
    for name, mark in [('reference',False),('comparison',True)]:
        image=Image.new('RGB',(448,448),'#e5e7eb')
        draw=ImageDraw.Draw(image)
        draw.rectangle((100,100,348,348),fill='#3874b8')
        draw.ellipse((185,185,263,263),fill='#e5e7eb')
        if mark:
            draw.line([(100,220),(148,238),(182,226)],fill='#111827',width=7)
        path=ART/(name+'.png')
        image.save(path)
        paths.append(path)
    return paths

def gpu_metrics(torch):
    torch.cuda.synchronize()
    return {'allocated_mib':round(torch.cuda.memory_allocated()/2**20,1),'reserved_mib':round(torch.cuda.memory_reserved()/2**20,1),'peak_allocated_mib':round(torch.cuda.max_memory_allocated()/2**20,1),'note':'PyTorch process memory, not total driver/desktop VRAM.'}

def check_dinov2():
    check_assets('dinov2/')
    torch=torch_ready()
    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image
    directory=ROOT/'models/dinov2-small'
    processor=AutoImageProcessor.from_pretrained(directory,local_files_only=True)
    model=AutoModel.from_pretrained(directory,local_files_only=True,trust_remote_code=False).to('cuda').eval()
    inputs=processor(images=Image.open(fixture_images()[0]),return_tensors='pt',do_resize=False,do_center_crop=False).to('cuda')
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        model(**inputs)
        torch.cuda.synchronize()
        started=time.perf_counter()
        output=model(**inputs).last_hidden_state
        torch.cuda.synchronize()
    if not torch.isfinite(output).all() or output.shape[1]<=1:
        raise RuntimeError('Invalid spatial features')
    return {'device':str(output.device),'input_shape':list(inputs['pixel_values'].shape),'feature_shape':list(output.shape),'finite':True,'forward_ms':round((time.perf_counter()-started)*1000,2),'memory':gpu_metrics(torch),'scope':'Feature extraction only; anomaly adapter, normal memory and thresholds not implemented.'}

def check_qwen():
    check_assets('qwen/')
    torch=torch_ready()
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    directory=ROOT/'models/qwen3.5-4b'
    paths=fixture_images()
    processor=AutoProcessor.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
    started=time.perf_counter()
    model=Qwen3_5ForConditionalGeneration.from_pretrained(directory,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
    torch.cuda.synchronize()
    load_seconds=time.perf_counter()-started
    messages=[{'role':'user','content':[{'type':'image','image':str(paths[0])},{'type':'image','image':str(paths[1])},{'type':'text','text':'Compare these two synthetic images. Briefly describe the visible difference in one sentence. Do not infer manufacturing quality.'}]}]
    inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False).to('cuda')
    if 'pixel_values' not in inputs:
        raise RuntimeError('Images were not encoded')
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    with torch.inference_mode():
        tokens=model.generate(**inputs,max_new_tokens=64,do_sample=False)
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    generated=tokens[:,inputs['input_ids'].shape[1]:]
    text=processor.batch_decode(generated,skip_special_tokens=True)[0].strip()
    if generated.numel()==0 or not text:
        raise RuntimeError('No generated response')
    result={'device':str(next(model.parameters()).device),'dtype':str(next(model.parameters()).dtype),'input_images':2,'input_tokens':inputs['input_ids'].shape[1],'generated_tokens':generated.shape[1],'response':text,'load_seconds':round(load_seconds,2),'generation_seconds':round(elapsed,2),'memory':gpu_metrics(torch),'scope':'Two-image local generation only; not an industrial defect evaluation.'}
    # Check the intended mixed workload without concurrent GPU training.
    check_assets('dinov2/')
    check_assets('rfdetr/')
    from transformers import AutoModel
    from rfdetr import RFDETRSmall
    from PIL import Image
    dino=AutoModel.from_pretrained(ROOT/'models/dinov2-small',local_files_only=True).to('cuda').eval()
    detector=RFDETRSmall(pretrain_weights=str(ROOT/'models/rf-detr-small.pth'),device='cuda')
    with torch.inference_mode():
        dino_output=dino(pixel_values=torch.zeros(1,3,224,224,device='cuda')).last_hidden_state
        detections=detector.predict(Image.open(paths[0]).convert('RGB'),threshold=0.0)
    if not torch.isfinite(dino_output).all() or len(detections)==0:
        raise RuntimeError('Co-resident model forward failed')
    # Generate once more while both other models remain on the GPU.
    with torch.inference_mode():
        combined_tokens=model.generate(**inputs,max_new_tokens=16,do_sample=False)
    if combined_tokens.shape[1]<=inputs['input_ids'].shape[1]:
        raise RuntimeError('Co-resident VLM generation failed')
    result['co_resident']={'models':['one RF-DETR Small','DINOv2 Small','Qwen3.5-4B BF16'],'forward_checks_passed':True,'memory':gpu_metrics(torch),'scope':'Three resident models, sequential inference; no product-specific second detector checkpoint or production concurrency test.'}
    return result

def check_training():
    check_assets('rfdetr/')
    torch=torch_ready()
    from PIL import Image,ImageDraw
    from rfdetr.config import RFDETRSmallConfig,TrainConfig
    from rfdetr.training import RFDETRModelModule,RFDETRDataModule,build_trainer
    from pytorch_lightning import Callback
    observed={'training_devices':[],'backward_checks':[]}
    class VerifyTrainingDevice(Callback):
        def on_train_batch_start(self,trainer,pl_module,batch,batch_idx):
            device=str(next(pl_module.parameters()).device)
            if not device.startswith('cuda'):
                raise RuntimeError('Training is not on CUDA')
            observed['training_devices'].append(device)
        def on_after_backward(self,trainer,pl_module):
            gradients=[p.grad for p in pl_module.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
                raise RuntimeError('Missing or non-finite gradients')
            nonzero=any(torch.count_nonzero(g).item()>0 for g in gradients)
            if not nonzero:
                raise RuntimeError('All gradients are zero')
            observed['backward_checks'].append({'finite':True,'nonzero':True})
    fixture=ART/'training-fixture'
    for split in ('train','valid','test'):
        directory=fixture/split
        directory.mkdir(parents=True,exist_ok=True)
        images=[]; annotations=[]
        for i in range(2):
            name=f'synthetic-{i}.png'
            image=Image.new('RGB',(512,512),'#dedede')
            ImageDraw.Draw(image).rectangle((100+i*30,130,300+i*30,340),fill='#3266aa')
            image.save(directory/name)
            images.append({'id':i+1,'file_name':name,'width':512,'height':512})
            annotations.append({'id':i+1,'image_id':i+1,'category_id':1,'bbox':[100+i*30,130,200,210],'area':42000,'iscrowd':0})
        (directory/'_annotations.coco.json').write_text(json.dumps({'info':{'description':'Runtime fixture only; not independent evaluation data'},'licenses':[],'images':images,'annotations':annotations,'categories':[{'id':1,'name':'synthetic_object','supercategory':'fixture'}]}),encoding='utf-8')
    model_config=RFDETRSmallConfig(pretrain_weights=str(ROOT/'models/rf-detr-small.pth'),num_classes=1,device='cuda')
    train_config=TrainConfig(dataset_dir=str(fixture),output_dir=str(ART/'training-output'),epochs=1,batch_size=1,grad_accum_steps=1,num_workers=0,use_ema=False,tensorboard=False,wandb=False,mlflow=False,multi_scale=False,expanded_scales=False,progress_bar=None,save_dataset_grids=False,augmentation_backend='torchvision')
    module=RFDETRModelModule(model_config,train_config)
    data=RFDETRDataModule(model_config,train_config)
    trainer=build_trainer(train_config,model_config,accelerator='gpu',devices=1,fast_dev_run=2,enable_model_summary=False)
    trainer.callbacks.append(VerifyTrainingDevice())
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    trainer.fit(module,datamodule=data)
    torch.cuda.synchronize()
    metrics={key:float(value.detach().cpu()) for key,value in trainer.callback_metrics.items() if torch.is_tensor(value) and value.numel()==1}
    if trainer.global_step!=2 or len(observed['backward_checks'])!=2 or any(not __import__('math').isfinite(v) for v in metrics.values()):
        raise RuntimeError('Training or metric computation failed')
    return {'optimizer_steps':trainer.global_step,**observed,'device_after_teardown':str(next(module.parameters()).device),'seconds':round(time.perf_counter()-started,2),'metrics':metrics,'memory':gpu_metrics(torch),'scope':'Two synthetic training and validation batches; fixtures intentionally reused. No accuracy/generalization evidence and no production weights.'}

def check_ui():
    import cv2
    import numpy as np
    from PIL import Image
    from PySide6.QtCore import QTimer,Qt,qVersion
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QLabel,QPushButton
    app=QApplication([])
    window=QWidget()
    window.setWindowTitle('MES Vision — 개발 환경 확인')
    window.resize(640,440)
    layout=QVBoxLayout(window)
    title=QLabel('MES Vision · 설치형 화면 실행 확인')
    title.setStyleSheet('font-size: 20px; font-weight: bold; padding: 12px;')
    layout.addWidget(title)
    picture=QLabel()
    image_path=fixture_images()[0]
    decoded=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
    if decoded is None or not np.array_equal(cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB),np.asarray(Image.open(image_path).convert('RGB'))):
        raise RuntimeError('OpenCV decoding or color conversion failed')
    picture.setPixmap(QPixmap(str(image_path)).scaled(250,250,Qt.KeepAspectRatio,Qt.SmoothTransformation))
    picture.setAlignment(Qt.AlignCenter)
    layout.addWidget(picture)
    status=QLabel('시험 화면입니다. 카메라·로봇은 연결하지 않았습니다.')
    layout.addWidget(status)
    button=QPushButton('이벤트 처리 확인')
    layout.addWidget(button)
    events=[]
    button.clicked.connect(lambda: (events.append('clicked'),status.setText('버튼 이벤트 처리 완료 · 제품 검사 화면은 이후 단계에서 구현')))
    result={}
    def finish():
        try:
            button.click()
            app.processEvents()
            visible=window.isVisible()
            saved=window.grab().save(str(ART/'desktop-smoke.png'))
            result.update(qt_version=qVersion(),opencv_version=cv2.__version__,image_decode_and_color_conversion=True,platform=app.platformName(),visible=visible,button_event_received=events==['clicked'],screenshot_saved=saved,scope='Temporary desktop runtime check, not the product UI.')
        except Exception as exc:
            result['error']=str(exc)
        finally:
            window.close(); app.quit()
    window.show()
    QTimer.singleShot(1200,finish)
    QTimer.singleShot(10000,app.quit)
    app.exec()
    if not result.get('visible') or not result.get('button_event_received') or not result.get('screenshot_saved') or result.get('platform')!='windows':
        raise RuntimeError('Native Windows UI check failed: '+str(result))
    return result

CHECKS={'dinov2':check_dinov2,'qwen':check_qwen,'training':check_training,'ui':check_ui}

def write_summary():
    summary={'time_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'os':platform.platform(),'scope':'Development runtime verification only. Actual D405, Dobot, laptop and product accuracy not tested.','components':{}}
    for component in CHECKS:
        path=ART/(component+'.json')
        report=json.loads(path.read_bytes()) if path.exists() else {'status':'not_run'}
        summary['components'][component]={'status':report['status'],'report':path.relative_to(ROOT).as_posix(),'time_utc':report.get('time_utc')}
    summary['status']='passed' if all(c['status']=='passed' for c in summary['components'].values()) else 'incomplete'
    summary['packages']={name:metadata.version(name) for name in ['torch','torchvision','rfdetr','transformers','accelerate','pytorch-lightning','peft','torchmetrics','opencv-python-headless','PySide6-Essentials','shiboken6']}
    (ART/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component',choices=list(CHECKS))
    args=parser.parse_args()
    if args.component:
        report={'component':args.component,'time_utc':datetime.now(timezone.utc).isoformat(),'status':'running'}
        (ART/(args.component+'.json')).write_text(json.dumps(report)+'\n',encoding='utf-8')
        write_summary()
        code=0
        try:
            report.update(result=CHECKS[args.component](),status='passed')
        except Exception as exc:
            report.update(status='failed',error=str(exc),traceback=traceback.format_exc())
            code=1
        (ART/(args.component+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        write_summary()
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        return code
    exit_codes=[]
    for component in CHECKS:
        with (ART/(component+'.log')).open('w',encoding='utf-8') as log:
            started=time.perf_counter()
            try:
                run=subprocess.run([sys.executable,str(Path(__file__)), '--component',component],stdout=log,stderr=subprocess.STDOUT,timeout=600)
                code=run.returncode
            except subprocess.TimeoutExpired:
                code=-1
                (ART/(component+'.json')).write_text(json.dumps({'component':component,'status':'failed','error':'Runtime check exceeded 600 seconds','time_utc':datetime.now(timezone.utc).isoformat()})+'\n',encoding='utf-8')
            if code!=0:
                report_path=ART/(component+'.json')
                current=json.loads(report_path.read_bytes()) if report_path.exists() else {}
                if current.get('status')!='failed':
                    report_path.write_text(json.dumps({'component':component,'status':'failed','error':f'Worker exited with code {code}; inspect {component}.log','time_utc':datetime.now(timezone.utc).isoformat()})+'\n',encoding='utf-8')
            exit_codes.append(code)
            print(component, 'passed' if code==0 else 'failed; inspect component log',flush=True)
    summary=write_summary()
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return 0 if summary['status']=='passed' and all(code==0 for code in exit_codes) else 1

if __name__=='__main__':
    raise SystemExit(main())
