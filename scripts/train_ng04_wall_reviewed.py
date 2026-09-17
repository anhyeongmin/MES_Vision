"""Explicit train-only real fine-tuning; no fabricated validation or release registration."""
from pathlib import Path
import sys,os,json,shutil,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
for key,folder in {'HF_HOME':'huggingface','TORCH_HOME':'torch','MPLCONFIGDIR':'matplotlib'}.items():os.environ[key]=str(ROOT/'.cache'/folder)
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['DO_NOT_TRACK']='1'
from mes_vision.data_management.collection import Collection,validate_record
from mes_vision.training.data import sha256

def main():
 import argparse
 from datetime import datetime
 parser=argparse.ArgumentParser(description='Local reviewed-data training. No independent validation; existing models are preserved.')
 parser.add_argument('--epochs',type=int,default=100)
 parser.add_argument('--check-only',action='store_true',help='Load every training image with augmentation; no training')
 args=parser.parse_args()
 if not 1 <= args.epochs <= 1000: parser.error('--epochs must be 1..1000')
 import torch
 from pytorch_lightning import Trainer,Callback,seed_everything
 from pytorch_lightning.callbacks import ModelCheckpoint
 from rfdetr.config import RFDETRSmallConfig,TrainConfig
 from rfdetr.training import RFDETRDataModule,RFDETRModelModule
 from filelock import FileLock
 from contextlib import ExitStack
 from mes_vision.vlm.gpu import GpuCoordinator
 from prepare_ng04_wall_review import prepare
 dataset=prepare()
 meta=json.loads((dataset/'provenance.json').read_text(encoding='utf-8'))
 assert meta['scope']=='train_only_unvalidated' and meta['images']==141
 coco=json.loads((dataset/'train/_annotations.coco.json').read_text(encoding='utf-8'))
 ims=coco['images'];anns=coco['annotations'];codes=[f'NG{i:02}' for i in range(1,7)]
 assert len(ims)==141 and len(anns)==118
 from PIL import Image
 for item,im in zip(meta['items'],ims):
  assert sha256(dataset/'train'/im['file_name'])==item['image_sha256']
  with Image.open(dataset/'train'/im['file_name']) as picture: assert picture.size==(im['width'],im['height'])
 for ann in anns:
  im=next(v for v in ims if v['id']==ann['image_id']);x,y,w,h=ann['bbox'];assert 0<=x and 0<=y and w>0 and h>0 and x+w<=im['width'] and y+h<=im['height']
 assert json.loads((dataset/'valid/_annotations.coco.json').read_text())['images']==[]
 out=ROOT/'artifacts/training'/('ash-wall-reviewed-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))/'detail-defects';out.mkdir(parents=True,exist_ok=False)
 shutil.copyfile(dataset/'provenance.json',out/'label-snapshot.json')
 weights=ROOT/'artifacts/training/ash-recorded-20260914-212558-590266/detail-defects/inference-unvalidated.pth'
 assert sha256(weights)=='4e58419200d37109981f697db4502c0db4a1ac4c494bcda08735cee748a84f55'
 manifest={'status':'PREPARED','scope':'Fine-tune recorded model on the same 141 real crops with 16 revised NG04 wall boxes; assistant-reviewed labels; independent evaluation pending','production_ready':False,'evaluation_performed':False,'validation_images':0,'train_images':len(ims),'annotations':len(anns),'epochs_requested':args.epochs,'initial_sha256':sha256(weights),'selection':'last epoch; no best-model metric','dataset':str(dataset),'preprocessing':'undistorted_full_frame_then_roi','calibration_sha256':meta['calibration_sha256'],'augmentation':__import__('rotation_training_support').SETTINGS,'negative_images':23,'sampling':meta['sampling'],'annotation_status':meta['annotation_status'],'prior_eval_normal_reused':True}
 def save(): (out/'run.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 save();print('OUTPUT:',out,flush=True);print('Epochs:',args.epochs,'No wall-clock cutoff.',flush=True);started=time.time()
 class Audit(Callback):
  def on_train_epoch_end(self,trainer,module):
   manifest.update(status='RUNNING',epoch=trainer.current_epoch+1,global_step=trainer.global_step,elapsed_seconds=time.time()-started);save();print('EPOCH',trainer.current_epoch+1,flush=True)
 try:
  assert torch.cuda.is_available();seed_everything(42,workers=True)
  mc=RFDETRSmallConfig(pretrain_weights=str(weights),num_classes=6,device='cuda',amp=True)
  tc=TrainConfig(dataset_dir=str(dataset),output_dir=str(out/'engine'),dataset_file='roboflow',epochs=args.epochs,batch_size=2,grad_accum_steps=4,lr=2e-5,lr_encoder=1e-5,num_workers=0,seed=42,use_ema=False,tensorboard=False,wandb=False,mlflow=False,run_test=False,multi_scale=False,expanded_scales=False,scale_jitter=False,progress_bar=None,save_dataset_grids=False,augmentation_backend='torchvision',aug_config={},class_names=codes)
  coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination')
  with ExitStack() as stack:
   stack.enter_context(FileLock(str(coord.root/'resident.lock'),timeout=0));stack.enter_context(coord.foreground(timeout=10))
   dm=RFDETRDataModule(mc,tc)
   from rotation_training_support import install,self_test
   self_test();install(dm)
   if args.check_only:
    dm.setup('fit')
    for i in range(len(dm._dataset_train)):
     image,target=dm._dataset_train[i]
     assert torch.isfinite(image).all() and torch.isfinite(target['boxes']).all()
     assert (target['boxes']>=0).all() and (target['boxes']<=1.001).all()
    for batch in dm.train_dataloader():break
    manifest.update(status='CHECK_ONLY_PASSED',images_checked=len(dm._dataset_train));save()
    print('CHECK PASSED: all 141 augmented samples loaded; no training performed',flush=True);return
   module=RFDETRModelModule(mc,tc)
   trainer=Trainer(accelerator='gpu',devices=1,max_epochs=args.epochs,precision='bf16-mixed',accumulate_grad_batches=4,gradient_clip_val=0.1,limit_val_batches=0,num_sanity_val_steps=0,log_every_n_steps=1,logger=__import__('pytorch_lightning.loggers',fromlist=['CSVLogger']).CSVLogger(str(out/'metrics')),enable_progress_bar=False,enable_model_summary=False,callbacks=[Audit(),ModelCheckpoint(dirpath=out/'engine',save_last=True,save_top_k=-1,every_n_epochs=25,filename='epoch-{epoch:03d}')],default_root_dir=out/'engine')
   manifest.update(status='RUNNING',gpu=torch.cuda.get_device_name(0));save()
   trainer.fit(module,datamodule=dm)
   assert trainer.global_step>0 and not trainer.interrupted
   dest=out/'inference-unvalidated.pth'
   torch.save({'model':{k:v.detach().cpu() for k,v in module.model.state_dict().items()},'model_name':'RFDETRSmall','model_config':mc.model_dump(mode='json'),'args':{'num_classes':6,'class_names':codes},'mes_vision':{'role':'known_defect_detector','production_ready':False,'evaluation_performed':False,'training_mode':'ng04_wall_relabel_train_only','preprocessing':'undistorted_full_frame_then_roi','calibration_sha256':meta['calibration_sha256'],'synthetic':False}},dest)
   manifest.update(status='COMPLETED_UNVALIDATED',weights=str(dest),weights_sha256=sha256(dest),global_step=trainer.global_step,elapsed_seconds=time.time()-started);save()
 except BaseException as exc:
  manifest.update(status='FAILED',error=repr(exc));save();raise
 print(json.dumps(manifest),flush=True);print('TRAINING FINISHED. Saved:',dest,flush=True)

if __name__=='__main__':main()
