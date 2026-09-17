"""Conservative image augmentation with synchronized spatial captions."""
import json,random
from pathlib import Path
import torch
from PIL import Image,ImageEnhance,ImageFilter
from rotation_training_support import geometry,self_test as geometry_test
SETTINGS={'quarter_turns':[0,90,180,270],'jitter_probability':.3,'jitter_degrees':10,'brightness_probability':.4,'brightness_range':[.9,1.1],'contrast_probability':.3,'contrast_range':[.9,1.1],'blur_probability':.15,'blur_radius':[.2,.5],'new_undistortion':False,'ground_truth_position_transformed':True}
FEATURES={'NG01':'원통 자리 비어 있음','NG02':'벽 안쪽 사각 돌출물','NG03':'바닥의 선 모양 균열','NG04':'벽 형상 변형','NG05':'원통 윗면 구멍 막힘','NG06':'바닥의 원형 패임'}
def caption(code,box,size):
 if code=='OK':return {'code':'OK','observation':'두 원통과 열린 구멍 확인, 뚜렷한 이상 없음'}
 x1,y1,x2,y2=box;w,h=size;cx=(x1+x2)/2/w;cy=(y1+y2)/2/h
 vertical='위쪽' if cy<.4 else '아래쪽' if cy>.6 else '중앙';horizontal='왼쪽' if cx<.4 else '오른쪽' if cx>.6 else ''
 return {'code':code,'observation':(' '.join([horizontal,vertical]).strip())+': '+FEATURES[code]}
def augment(image,box,turns=None,angle=None):
 turns=random.randrange(4) if turns is None else turns;angle=(random.uniform(-10,10) if random.random()<.3 else 0) if angle is None else angle
 boxes=torch.tensor([box],dtype=torch.float32) if box is not None else torch.empty((0,4))
 image,boxes=geometry(image,boxes,turns,angle)
 if random.random()<.4:image=ImageEnhance.Brightness(image).enhance(random.uniform(.9,1.1))
 if random.random()<.3:image=ImageEnhance.Contrast(image).enhance(random.uniform(.9,1.1))
 if random.random()<.15:image=image.filter(ImageFilter.GaussianBlur(random.uniform(.2,.5)))
 return image,boxes[0].tolist() if len(boxes) else None

def encode(processor,row,system,answer=True):
 with Image.open(row['image']) as im:im=im.convert('RGB')
 b=row['source_region'];box=[b[0],b[1],b[0]+b[2],b[1]+b[3]] if b else None
 if answer and row['split']=='train':im,box=augment(im,box)
 target=caption(row['target']['code'],box,im.size);im.thumbnail((448,448))
 messages=[{'role':'system','content':system},{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':'사진의 상태를 관찰하세요.'}]}]

 prefix=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False)
 if not answer:return prefix
 full=processor.apply_chat_template(messages+[{'role':'assistant','content':json.dumps(target,ensure_ascii=False,separators=(',',':'))}],tokenize=True,add_generation_prompt=False,return_dict=True,return_tensors='pt',enable_thinking=False)
 n=prefix['input_ids'].shape[1];assert torch.equal(full['input_ids'][:,:n],prefix['input_ids']);labels=full['input_ids'].clone();labels[:,:n]=-100;assert (labels!=-100).sum()>5 and labels.shape[1]<2048;full['labels']=labels;return full

def self_test():
 geometry_test();im=Image.new('RGB',(100,100));b=[5,65,25,90]
 for turns,expected in [(0,'왼쪽 아래쪽'),(1,'오른쪽 아래쪽'),(2,'오른쪽 위쪽'),(3,'왼쪽 위쪽')]:
  pic,box=augment(im,b,turns,0);assert caption('NG01',box,pic.size)['observation'].startswith(expected)
  _,empty=augment(im,None,turns,0);assert empty is None
 assert caption('OK',None,(100,100))['code']=='OK'
 print('PASS: rotated regions and Korean spatial captions agree; empty OK preserved')
if __name__=='__main__':self_test()
