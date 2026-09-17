from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parents[1]
logdir=root/'artifacts/vlm-caption-review-v3'
status=logdir/'pipeline.json'
def save(**kw):status.write_text(json.dumps(kw,ensure_ascii=False,indent=2),encoding='utf-8')
try:
 save(status='TRAINING',production_ready=False)
 before=set((root/'artifacts/training').glob('vlm-2b-caption-v3-*'))
 subprocess.run([sys.executable,'-X','utf8',str(root/'scripts/train_vlm_caption_v3.py'),'--epochs','5'],check=True,cwd=root)
 runs=set((root/'artifacts/training').glob('vlm-2b-caption-v3-*'))-before
 assert len(runs)==1
 run=runs.pop();save(status='COMPARING',run=str(run),production_ready=False)
 subprocess.run([sys.executable,'-X','utf8',str(root/'scripts/compare_vlm_caption_v3.py'),'--run',str(run)],check=True,cwd=root)
 save(status='COMPLETED',run=str(run),report=str(run/'comparison/report.html'),production_ready=False)
except BaseException as e:
 save(status='FAILED',error=repr(e));raise
