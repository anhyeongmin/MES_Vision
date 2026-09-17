"""Explicit train-only real fine-tuning; no fabricated validation or release registration."""
from pathlib import Path
import sys,os,json,shutil,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
for key,folder in {'HF_HOME':'huggingface','TORCH_HOME':'torch','MPLCONFIGDIR':'matplotlib'}.items():os.environ[key]=str(ROOT/'.cache'/folder)
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['DO_NOT_TRACK']='1'
from mes_vision.data_management.collection import Collection,validate_record
from mes_vision.training.data import sha256

def main():
 import torch
 from pytorch_lightning import Trainer,Callback,seed_everything
 from pytorch_lightning.callbacks import ModelCheckpoint
 from rfdetr.config import RFDETRSmallConfig,TrainConfig
 from rfdetr.training import RFDETRDataModule,RFDETRModelModule
 from filelock import FileLock
 from contextlib import ExitStack
 from mes_vision.vlm.gpu import GpuCoordinator
 collection=Collection(ROOT/'collections/ash-real-detail-videos-20260914')
 selected=[r for r in collection.data['records'] if r['objects'] and not r['excluded']]
 assert len(selected)==32
 for r in selected:
  validate_record(r,complete=True);assert r['split']=='train';assert sha256(collection.image_path(r))==r['image_sha256']
 out=ROOT/'artifacts/training/ash-real-bootstrap-v1/detail-defects';out.mkdir(parents=True,exist_ok=False)
 dataset=out/'train-only-input';dataset.mkdir()
 codes=[f'NG{i:02}' for i in range(1,7)];cats=[{'id':i+1,'name':code,'supercategory':'defect'} for i,code in enumerate(codes)]
 ims=[];anns=[];provenance=[]
 (dataset/'train').mkdir();(dataset/'valid').mkdir()
 for i,r in enumerate(selected,1):
  o=r['objects'][0];assert o['condition'] in ['NORMAL','KNOWN_NG']
  b=o['bbox'];im=__import__('PIL.Image',fromlist=['Image']).open(collection.image_path(r)).convert('RGB').crop(b);name=r['id']+'.png';im.save(dataset/'train'/name)
  ims.append({'id':i,'file_name':name,'width':im.width,'height':im.height})
  for de in o['defects']:
   x,y,x2,y2=de['bbox'];box=[x-b[0],y-b[1],x2-x,y2-y];assert box[0]>=0 and box[1]>=0 and box[0]+box[2]<=im.width and box[1]+box[3]<=im.height
   anns.append({'id':len(anns)+1,'image_id':i,'category_id':codes.index(de['code'])+1,'bbox':box,'area':box[2]*box[3],'iscrowd':0})
  provenance.append({'record':r,'crop':b,'training_image_sha256':sha256(dataset/'train'/name)})
 for split,images,annotations in [('train',ims,anns),('valid',[],[])]:
  (dataset/split/'_annotations.coco.json').write_text(json.dumps({'images':images,'annotations':annotations,'categories':cats}),encoding='utf-8')
 # Empty valid is backend setup compatibility only. Trainer executes zero validation batches.
 (out/'label-snapshot.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
 weights=ROOT/'artifacts/training/ash-reinforced-v3/detail-defects/inference.pth'
 assert sha256(weights)=='d748f4a1736568553a17c8196f7d0f5a8b92f36ceb389fa7b779f3326c8f6b37'
 manifest={'status':'PREPARED','scope':'Initial fine-tune on visually inspected annotation drafts; independent evaluation pending','production_ready':False,'evaluation_performed':False,'validation_images':0,'train_images':len(ims),'annotations':len(anns),'epochs_requested':10,'initial_sha256':sha256(weights),'selection':'last epoch; no best-model metric','source_collection':str(collection.root)}
 def save(): (out/'run.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 save();started=time.time()
 class Audit(Callback):
  def on_train_epoch_end(self,trainer,module):
   manifest.update(status='RUNNING',epoch=trainer.current_epoch+1,global_step=trainer.global_step,elapsed_seconds=time.time()-started);save();print('EPOCH',trainer.current_epoch+1,flush=True)
 try:
  assert torch.cuda.is_available();seed_everything(42,workers=True)
  mc=RFDETRSmallConfig(pretrain_weights=str(weights),num_classes=6,device='cuda',amp=True)
  tc=TrainConfig(dataset_dir=str(dataset),output_dir=str(out/'engine'),dataset_file='roboflow',epochs=10,batch_size=2,grad_accum_steps=4,lr=2e-5,lr_encoder=1e-5,num_workers=0,seed=42,use_ema=False,tensorboard=False,wandb=False,mlflow=False,run_test=False,multi_scale=False,expanded_scales=False,scale_jitter=False,progress_bar=None,save_dataset_grids=False,augmentation_backend='torchvision',class_names=codes)
  coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination')
  with ExitStack() as stack:
   stack.enter_context(FileLock(str(coord.root/'resident.lock'),timeout=0));stack.enter_context(coord.foreground(timeout=10))
   module=RFDETRModelModule(mc,tc);dm=RFDETRDataModule(mc,tc)
   trainer=Trainer(accelerator='gpu',devices=1,max_epochs=10,precision='bf16-mixed',accumulate_grad_batches=4,gradient_clip_val=0.1,limit_val_batches=0,num_sanity_val_steps=0,logger=False,enable_progress_bar=False,enable_model_summary=False,callbacks=[Audit(),ModelCheckpoint(dirpath=out/'engine',save_last=True,save_top_k=0)],default_root_dir=out/'engine')
   manifest.update(status='RUNNING',gpu=torch.cuda.get_device_name(0));save()
   trainer.fit(module,datamodule=dm)
   assert trainer.global_step>0 and not trainer.interrupted
   dest=out/'inference-unvalidated.pth'
   torch.save({'model':{k:v.detach().cpu() for k,v in module.model.state_dict().items()},'model_name':'RFDETRSmall','model_config':mc.model_dump(mode='json'),'args':{'num_classes':6,'class_names':codes},'mes_vision':{'role':'known_defect_detector','production_ready':False,'evaluation_performed':False,'training_mode':'bootstrap_train_only','synthetic':False}},dest)
   manifest.update(status='COMPLETED_UNVALIDATED',weights=str(dest),weights_sha256=sha256(dest),global_step=trainer.global_step,elapsed_seconds=time.time()-started);save()
 except BaseException as exc:
  manifest.update(status='FAILED',error=repr(exc));save();raise
 print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
