import json
from PIL import Image
import torch
SETTINGS={'augmentation':False,'full_max_edge':448,'region_max_edge':224,'output':'observation,needs_review','candidate_context':True,'numeric_metadata':False,'fixed_crop':[.05,.05,.95,.95]}
def encode(processor,row,system,answer=True):
 with Image.open(row['image']) as source: full=source.convert('RGB')
 w,h=full.size;box=(int(.05*w),int(.05*h),max(1,int(.95*w)),max(1,int(.95*h)))
 region=full.crop(box);region.thumbnail((224,224));full.thumbnail((448,448))
 code=row['candidate_code'];assert code in ['NONE','NG04','NG05']
 messages=[{'role':'system','content':system},{'role':'user','content':[{'type':'text','text':'전체 상세사진입니다.'},{'type':'image','image':full},{'type':'text','text':'같은 사진의 중앙 영역입니다.'},{'type':'image','image':region},{'type':'text','text':'확인할 후보: '+code}]}]
 prefix=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False)
 if not answer:return prefix
 target=row['target'];assert set(target)=={'observation','needs_review'} and isinstance(target['needs_review'],bool)
 result=processor.apply_chat_template(messages+[{'role':'assistant','content':json.dumps(target,ensure_ascii=False,separators=(',',':'))}],tokenize=True,add_generation_prompt=False,return_dict=True,return_tensors='pt',enable_thinking=False)
 n=prefix['input_ids'].shape[1];assert torch.equal(result['input_ids'][:,:n],prefix['input_ids']);labels=result['input_ids'].clone();labels[:,:n]=-100;assert labels.shape[1]<2048 and (labels!=-100).sum()>5;result['labels']=labels;return result
def self_test():
 print('Evidence schema: no final code; two images; candidate text contains no ground-truth labels')
