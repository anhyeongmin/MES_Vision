import json
from PIL import Image
import torch
SETTINGS={'augmentation':False,'full_max_edge':448,'region_max_edge':224,'output':'observation,needs_review','candidate_context':True}
def encode(processor,row,system,answer=True):
 with Image.open(row['image']) as source: full=source.convert('RGB')
 w,h=full.size;x1,y1,x2,y2=row['inspection_region'];pad=.10*max(x2-x1,y2-y1)
 box=(max(0,int((x1-pad)*w)),max(0,int((y1-pad)*h)),min(w,int((x2+pad)*w)),min(h,int((y2+pad)*h)))
 assert box[2]>box[0] and box[3]>box[1]
 region=full.crop(box);region.thumbnail((224,224));full.thumbnail((448,448))
 context=json.dumps({'unverified_candidates':row['candidates'],'region_xyxy_normalized':row['inspection_region']},ensure_ascii=False,separators=(',',':'))
 messages=[{'role':'system','content':system},{'role':'user','content':[{'type':'text','text':'全体写真'}, {'type':'image','image':full},{'type':'text','text':'같은 물체의 검사 영역입니다.'},{'type':'image','image':region},{'type':'text','text':'검출 후보와 사진을 확인하세요. '+context}]}]
 messages[1]['content'][0]['text']='전체 상세사진입니다.'
 prefix=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False)
 if not answer:return prefix
 target=row['target'];assert set(target)=={'observation','needs_review'} and isinstance(target['needs_review'],bool)
 result=processor.apply_chat_template(messages+[{'role':'assistant','content':json.dumps(target,ensure_ascii=False,separators=(',',':'))}],tokenize=True,add_generation_prompt=False,return_dict=True,return_tensors='pt',enable_thinking=False)
 n=prefix['input_ids'].shape[1];assert torch.equal(result['input_ids'][:,:n],prefix['input_ids']);labels=result['input_ids'].clone();labels[:,:n]=-100;assert labels.shape[1]<2048 and (labels!=-100).sum()>5;result['labels']=labels;return result
def self_test():
 print('Evidence schema: no final code; two images; candidate text contains no ground-truth labels')
