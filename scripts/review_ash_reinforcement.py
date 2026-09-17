"""Visual QA of paired training-only reinforcement crops; never model predictions."""
from pathlib import Path
import argparse
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from mes_vision.synthetic.dataset_export import read,write,sha
from mes_vision.synthetic.reinforcement_export import group_tasks,crop_labels
from mes_vision.synthetic.reinforcement import crop_variant,xyxy


def main(root):
    root=root.resolve()
    selected=[]; strata=set()
    for group,tasks in group_tasks(read(root/"plan.json")).items():
        stratum=tasks[0]["condition_stratum"]
        if tasks[0]["split"]!='train' or stratum in strata:
            continue
        if not all((root/"quality"/(t['id']+'.json')).exists() for t in tasks):
            continue
        values=[read(root/"quality"/(t['id']+'.json')) for t in tasks]
        if not all(v['accepted'] for v in values):
            continue
        selected.append((group,tasks,values)); strata.add(stratum)
    if len(selected)!=3:
        raise ValueError('Three completed training strata required')
    folder=root/"review"; folder.mkdir(exist_ok=True)
    sheet=Image.new('RGB',(1280,1040),(25,28,34))
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    draw=ImageDraw.Draw(sheet)
    headers=['OK / base','NG01 / base','NG01 / crop variation','NG04 / same crop variation']
    for i,title in enumerate(headers): draw.text((i*320+10,15),title,fill='white',font=font)
    records=[]
    for row,(group,tasks,values) in enumerate(selected):
        variant=crop_variant('v3:'+group,2,[v['objects'][0]['bbox'] for v in values],[d['bbox'] for v in values for d in v['defects']])
        by_code={t['objects'][0]['code']:(t,v) for t,v in zip(tasks,values)}
        for col,code in enumerate(('OK','NG01','NG01','NG04')):
            task,value=by_code[code]
            with Image.open(root/'raw'/(task['id']+'.png')) as im: rgb=np.array(im)
            with Image.open(root/'masks'/(task['id']+'.png')) as im: mask=np.array(im)
            bounds=[int(v) for v in xyxy(value['objects'][0]['bbox'])] if col<2 else variant['bounds']
            crop,_,labels=crop_labels(rgb,mask,bounds,value['defects'])
            image=Image.fromarray(crop); d=ImageDraw.Draw(image)
            for label in labels:
                d.rectangle(xyxy(label['bbox']),outline=(255,75,65),width=2)
            image.thumbnail((305,280))
            x,y=col*320+8,45+row*330
            sheet.paste(image,(x+(305-image.width)//2,y+(280-image.height)//2))
            draw.text((x,y+282),task['condition_stratum'],fill='white',font=font)
            records.append(dict(scene=task['id'],bounds=bounds,source_sha256=sha(root/'raw'/(task['id']+'.png'))))
    draw.text((10,1020),'Synthetic training preview | Red: ground truth | Same CAD; not physical performance',fill='white',font=font)
    sheet.save(folder/'training-preview.png')
    write(folder/'preview-sources.json',records)
    print(folder/'training-preview.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    main(parser.parse_args().root)
