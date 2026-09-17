"""Stage the candidate configuration; leaves the live selection unchanged."""
import argparse,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mes_vision.training.data import read_json,write_json,sha256,require
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
 run=args.run.resolve();meta=read_json(run/'run.json')
 require(meta['status']=='COMPLETED_UNVALIDATED','Training has not completed')
 selected=read_json(ROOT/'artifacts/vlm-all-classes-v1/before/backend.json')
 source=run/'adapter';destination=ROOT/'models'/('vlm-all-classes-'+run.name)
 require(not destination.exists(),'Candidate already exists; do not overwrite')
 shutil.copytree(source,destination)
 selected.update(adapter=str(destination.relative_to(ROOT)).replace('\\','/'),
   retained_adapter=selected['adapter'],retained_codes=['NG04','NG05'],
   supported_codes=['OK','NG01','NG02','NG03','NG04','NG05','NG06'])
 selected['files']['retained_adapter']=selected['files']['adapter']
 selected['files']['adapter']={p.name:sha256(p) for p in destination.iterdir() if p.is_file() and not p.name.startswith('.')}
 selected['training_run']=str(run.relative_to(ROOT));selected['production_validated']=False
 selected['description']='Single 2B base, all-class LoRA; original regional LoRA retained for NG04/NG05.'
 write_json(run/'candidate-backend.json',selected)
 print(run/'candidate-backend.json')
if __name__=='__main__':main()
