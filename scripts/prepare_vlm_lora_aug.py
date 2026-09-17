from pathlib import Path
import json
from PIL import Image
from prepare_vlm_lora import ROOT,sha
from vlm_augmentation_support import caption,SETTINGS

def prepare():
 src=ROOT/'datasets/ash-vlm-lora-v1/dataset.json';data=json.loads(src.read_text(encoding='utf-8'))
 for r in data['items']:
  b=r['source_region'];box=[b[0],b[1],b[0]+b[2],b[1]+b[3]] if b else None
  with Image.open(r['image']) as im:r['target']=caption(r['target']['code'],box,im.size)
 data.update(augmentation=SETTINGS,parent_dataset_sha256=sha(src),scope='v2 development experiment, same86 train/55 eval sessions as v1; eval reused for development so NOT independent; short targets and augmentation both changed; no old eval images added to training')
 out=ROOT/'datasets/ash-vlm-lora-v2';out.mkdir(exist_ok=True);dest=out/'dataset.json'
 if dest.exists():assert json.loads(dest.read_text(encoding='utf-8'))==data
 else:dest.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
 return dest
if __name__=='__main__':print(prepare())
