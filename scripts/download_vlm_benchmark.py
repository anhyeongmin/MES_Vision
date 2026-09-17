import os,json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
root=Path.cwd();api=HfApi();records=[]
for name in ['Qwen3.5-0.8B','Qwen3.5-2B','Qwen3-VL-2B-Instruct']:
 repo='Qwen/'+name;info=api.model_info(repo);dest=root/'models'/(name.lower()+'-benchmark');print('DOWNLOAD',repo,info.sha,flush=True)
 snapshot_download(repo,revision=info.sha,local_dir=dest,allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','LICENSE','README.md','*.model'],max_workers=4)
 records.append(dict(repo=repo,revision=info.sha,path=str(dest)));(root/'models/vlm-benchmark-downloads.json').write_text(json.dumps(records,indent=2),encoding='utf-8');print('READY',repo,flush=True)
