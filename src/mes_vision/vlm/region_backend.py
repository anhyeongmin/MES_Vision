"""Advisory region observer. Snapshot coordinates are validated, never remapped."""
from pathlib import Path
import json
import math
import time

from PIL import Image
from mes_vision.inspection import Box
from mes_vision.training.data import require, read_json, sha256
from .backend import GenerationConfig, parse_response
from .snapshots import load_snapshot, object_record
from mes_vision.inspection.finding_scores import score_band, RULE

SYSTEM = '관심 영역 사진을 보고 질문에 해당하는 특징만 설명하세요. 불명확하면 확인하기 어렵다고 답하세요. 전체 물체의 OK/NG를 판정하지 마세요. observation 문자열 하나를 가진 JSON만 출력하세요.'
QUESTIONS = {'NG05': '확대 영역의 원통 구멍이 열려 있는지 설명하세요.',
             'NG04': '확대 영역의 원통과 벽 접합부에 뚜렷한 틈이 보이는지 설명하세요.',
             'OK': '물체 내부 원통의 개수와 보이는 배치를 설명하세요.',
             'NG01': '물체 내부 원통의 개수와 보이는 배치를 설명하세요.',
             'NG02': '확대 영역의 벽 안쪽에 사각 돌출물이 보이는지 설명하세요.',
             'NG03': '확대 영역의 바닥에 선 모양 균열이 보이는지 설명하세요.',
             'NG06': '확대 영역의 바닥에 원형 표면 흔적이 보이는지 설명하세요.'}

def supported_codes(selected):
    codes=selected.get('supported_codes',['NG04','NG05'])
    require(isinstance(codes,list) and bool(codes) and len(set(codes))==len(codes)
            and all(c in QUESTIONS for c in codes),'Unsupported VLM capability list')
    return set(codes)

def region_plan(job, codes=None):
    codes={'NG04','NG05'} if codes is None else set(codes)
    require(codes <= set(QUESTIONS),'Unsupported region question')
    manifest, inspection, _ = load_snapshot(job['snapshot_path'], expected_digest=job['snapshot_digest'])
    record = object_record(manifest, inspection, job['object_id'])
    crop = record['crop']; bounds = Box(**crop['bounds_original_xyxy'])
    require(crop['coordinate_space'] == 'crop_rgb_pixels' and crop['resized'] is False
            and crop['alignment'] == 'NOT_APPLIED', 'Unsupported region coordinate transform')
    require(crop['crop_to_original'] == [[1,0,bounds.x1],[0,1,bounds.y1],[0,0,1]], 'Region crop transform mismatch')
    source = next(x for x in manifest['objects'] if x['object_id'] == job['object_id'])
    require(source['image_size'] == [crop['width'],crop['height']], 'Region crop dimensions mismatch')
    regions, unsupported = [], []
    hidden = 0
    for check in record['checks']:
        if check['check_id'] != 'known_defects': continue
        for finding in check['findings']:
            code = finding['defect_code']
            if code not in codes or code == 'OK':
                unsupported.append(code); continue
            require(finding['original_box'] is not None and finding['crop_box'] is not None, 'Region finding has no coordinate pair')
            local, original = Box(**finding['crop_box']), Box(**finding['original_box'])
            require(local.clip(crop['width'],crop['height']) == local, 'Region finding outside object')
            require(local.translated(bounds.x1,bounds.y1) == original, 'Region finding coordinate mismatch')
            dx, dy = (local.x2-local.x1)*.25, (local.y2-local.y1)*.25
            box = [max(0,math.floor(local.x1-dx)),max(0,math.floor(local.y1-dy)),
                   min(crop['width'],math.ceil(local.x2+dx)),min(crop['height'],math.ceil(local.y2+dy))]
            # Missing cylinders need whole-object context, not an isolated empty patch.
            if code=='NG01': box=[0,0,crop['width'],crop['height']]
            score = finding.get('score')
            require(score is None or (type(score) in (int,float) and math.isfinite(score) and 0 <= score <= 1), 'Invalid region score')
            band=score_band(finding)
            if band=='HIDDEN':
                hidden+=1
                continue
            regions.append({'code':code,'box':box,'score':score or 0,'score_band':band})
    regions.sort(key=lambda r:-r['score'])
    # Multiple missing-feature boxes still ask the same whole-object question once.
    whole_seen=False; unique=[]
    for region in regions:
        if region['code']=='NG01':
            if whole_seen: continue
            whole_seen=True
        unique.append(region)
    regions=unique
    if not regions and not unsupported and not hidden and 'OK' in codes:
        regions=[{'code':'OK','box':[0,0,crop['width'],crop['height']],'score':0}]
    return {'image':str(Path(job['snapshot_path'])/source['file']), 'regions':regions[:3],
            'omitted':max(0,len(regions)-3), 'unsupported':unsupported,
            'hidden_below_display_min':hidden, 'display_rule':dict(RULE),
            'object_id':job['object_id'],'snapshot_digest':job['snapshot_digest']}

def parse_region(raw):
    # Reuse the strict duplicate-key, text-length and control-character validation.
    def unique(pairs):
        d={}
        for key,value in pairs:
            require(key not in d,'duplicate response field'); d[key]=value
        return d
    require(isinstance(raw,str) and 0 < len(raw) <= 6000,'invalid region response')
    d=json.loads(raw,object_pairs_hook=unique)
    require(isinstance(d,dict) and set(d)=={'observation'},'Region model must return only observation')
    return parse_response(json.dumps(dict(d,needs_review=True),ensure_ascii=False))['observation']

def build_backend(root):
    path=Path(root)/'configs/vlm/backend.json'
    if not path.exists():
        from .backend import QwenBackend
        return QwenBackend(root)
    selected=read_json(path)
    require(selected.get('backend')=='region_2b','Unsupported configured VLM backend')
    return RegionBackend(root,selected)

class RegionBackend:
    def __init__(self,root,selected):
        import os
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        root=Path(root).resolve()
        self.supported_codes=supported_codes(selected)
        def directory(key):
            p=(root/selected[key]).resolve()
            require(p.is_relative_to(root/'models'),'Region model path outside model directory')
            return p
        base,adapter=directory('base'),directory('adapter')
        retained=directory('retained_adapter') if selected.get('retained_adapter') else None
        require(selected['revision']=='15852e8c16360a2fea060d615a32b45270f8a8fc','Region base revision changed')
        for kind,d in [('base',base),('adapter',adapter)]+([('retained_adapter',retained)] if retained else []):
            assets=selected['files'][kind]
            actual={p.name for p in d.iterdir() if p.is_file() and not p.name.startswith('.')}
            require(actual==set(assets),'Region model file list changed')
            for name,digest in assets.items():
                require(Path(name).name==name and sha256(d/name)==digest,'Region model integrity mismatch')
        import torch
        from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
        from peft import PeftModel
        from .optimization import configure_patch_projection
        require(torch.cuda.is_available(),'CUDA unavailable; no automatic CPU fallback')
        torch.set_num_threads(4)
        self.processor=AutoProcessor.from_pretrained(base,local_files_only=True,trust_remote_code=False)
        model=Qwen3_5ForConditionalGeneration.from_pretrained(base,local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
        execution=configure_patch_projection(model,'linear_patch_v1')
        self.model=PeftModel.from_pretrained(model,adapter,local_files_only=True)
        self.retained_codes=set(selected.get('retained_codes',[]))
        require(self.retained_codes <= self.supported_codes and (not self.retained_codes or retained is not None),'Invalid retained adapter routing')
        if retained:
            self.model.load_adapter(retained,adapter_name='retained',local_files_only=True)
        else: self.model=self.model.merge_and_unload(safe_merge=True)
        self.model.eval()
        self.protocol='region-observation-v2-all-classes' if self.supported_codes-{'NG04','NG05'} else 'region-observation-v1'
        self.provenance={'id':'Qwen/Qwen3.5-2B-'+('all-classes' if self.supported_codes-{'NG04','NG05'} else 'region-abstain'),'revision':selected['revision'],
                         'files':selected['files'],'execution':execution,'advisory_only':True,'production_validated':False,
                         'supported_codes':sorted(self.supported_codes),'retained_codes':sorted(self.retained_codes)}

    def close(self):
        self.model=None; self.processor=None
        import torch
        torch.cuda.empty_cache()

    def generate(self,job):
        import torch
        from transformers import StoppingCriteria,StoppingCriteriaList
        config=GenerationConfig(**job['payload']['generation'])
        plan=region_plan(job,self.supported_codes); started=time.perf_counter(); details=[]; expired=False; truncated=False
        if config.language!='ko':
            observation='This region model supports Korean only; no additional analysis was performed.'
        elif not plan['regions']:
            observation='지원하는 검출 영역이 없어 추가 설명을 수행하지 않았습니다.'
        else:
            with Image.open(plan['image']) as im: source=im.convert('RGB')
            for region in plan['regions']:
                if self.retained_codes:
                    self.model.set_adapter('retained' if region['code'] in self.retained_codes else 'default')
                image=source.crop(region['box']);image.thumbnail((448,448))
                messages=[{'role':'system','content':SYSTEM},{'role':'user','content':[
                    {'type':'image','image':image},{'type':'text','text':QUESTIONS[region['code']]}]}]
                inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                    return_dict=True,return_tensors='pt',enable_thinking=False).to('cuda')
                require('pixel_values' in inputs and inputs['input_ids'].shape[1]<=4096,'Invalid region model input')
                class Deadline(StoppingCriteria):
                    def __call__(self,input_ids,scores,**kwargs):return time.perf_counter()-started>=config.timeout_seconds
                with torch.inference_mode():tokens=self.model.generate(**inputs,max_new_tokens=min(64,config.max_new_tokens),
                    do_sample=False,use_cache=True,stopping_criteria=StoppingCriteriaList([Deadline()]))
                torch.cuda.synchronize();gen=tokens[:,inputs['input_ids'].shape[1]:]
                raw=self.processor.batch_decode(gen,skip_special_tokens=True)[0].strip()
                eos=self.model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                truncated=gen.shape[1]>=min(64,config.max_new_tokens) and int(gen[0,-1]) not in eos
                expired=time.perf_counter()-started>=config.timeout_seconds
                if expired or truncated:break
                text=parse_region(raw);details.append(dict(region,raw_text=raw,observation=text,
                    adapter='retained' if region['code'] in self.retained_codes else 'default'))
            observation='\n'.join(f"{'전체 관찰' if r['code']=='OK' else r['code']+' ['+('의심' if r.get('score_band')=='SUSPECT' else '검출')+']'}: {r['observation']}" for r in details) or '추가 설명을 완료하지 못했습니다.'
            if plan['unsupported'] or plan['omitted']:observation+='\n지원 범위 밖이거나 처리 한도를 넘은 후보는 설명하지 않았습니다.'
        # Deliberately conservative: regional descriptions never approve a whole object.
        return {'raw_text':json.dumps({'observation':observation,'needs_review':True},ensure_ascii=False),
                'truncated':truncated,'deadline_expired':expired,'model':self.provenance,'prompt_version':self.protocol,
                'prompt':dict(plan,protocol=self.protocol,details=details),
                'metrics':{'region_count':len(details),'generation_seconds':time.perf_counter()-started}}
