"""Verify the generated, unsplit review sample package without training or device access."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mes_vision.synthetic.projection import bbox


def load(path): return json.loads(path.read_text(encoding='utf-8'))


def verify(root):
    root=root.resolve(); render=load(root/'render-manifest.json'); summary=load(root/'sample-report.json')
    hashes=load(root/'sample-hashes.json')
    for name,expected in hashes.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('File changed or escaped: '+name)
    if len(render['frames'])!=24 or render['status']!='rendered': raise ValueError('Incomplete render')
    frames={f['id']:f for f in render['frames']}
    if len(frames)!=24: raise ValueError('Duplicate scenes')
    counts={}
    for role,expected_images,expected_annotations in (('detail-objects',21,21),('detail-defects',21,18),('overview-objects',3,21)):
        data=load(root/'annotations'/(role+'.coco.json'))
        if len(data['images'])!=expected_images or len(data['annotations'])!=expected_annotations:
            raise ValueError('Unexpected sample coverage: '+role)
        images={v['id']:v for v in data['images']}; categories={c['id']:c['name'] for c in data['categories']}
        if len(images)!=expected_images: raise ValueError('Duplicate images')
        for info in images.values():
            with Image.open(root/info['file_name']) as im:
                if im.size!=(1280,720) or im.mode!='RGB': raise ValueError('Invalid RGB image')
            frame=frames[info['scene_id']]
            if hashlib.sha256((root/info['file_name']).read_bytes()).hexdigest()!=frame['image_sha256']:
                raise ValueError('Raw RGB no longer matches renderer output')
        distribution=Counter(); seen=set()
        for a in data['annotations']:
            if a['id'] in seen: raise ValueError('Duplicate annotation')
            seen.add(a['id']); info=images[a['image_id']]; code=categories[a['category_id']]
            x,y,w,h=a['bbox']
            if not (0<=x<x+w<=1280 and 0<=y<y+h<=720 and a['area']==w*h): raise ValueError('Invalid box')
            distribution[code]+=1
            if role=='detail-defects':
                if info['cad_variants']!=[code]: raise ValueError('Wrong defect class')
                with Image.open(root/a['mask_file']) as im: mask=np.asarray(im)>0
                if mask.shape!=(720,1280) or bbox(mask,2)!=a['bbox'] or int(mask.sum())!=a['geometric_pixels']:
                    raise ValueError('Mask / box mismatch')
        if role=='detail-defects' and distribution!={f'NG{i:02d}':3 for i in range(1,7)}:
            raise ValueError('Missing defect class')
        counts[role]=dict(images=expected_images,annotations=expected_annotations,class_counts=dict(distribution))
    for detail in summary['details']:
        frame=frames[detail['id']]; normal=frames[frame['condition']['id']+'_OK']
        if frame['camera']!=normal['camera'] or frame['condition']!=normal['condition']:
            raise ValueError('Counterfactual camera / lighting mismatch')
        if frame['objects'][0]['matrix_world']!=normal['objects'][0]['matrix_world']:
            raise ValueError('Counterfactual pose mismatch')
        with Image.open(root/detail['mask']) as im: mask=np.asarray(im)>0
        if detail['code']=='OK' and (mask.any() or detail['defect_bbox'] is not None):
            raise ValueError('Normal image labelled as defective')
    result=dict(status='passed',files_verified=len(hashes),groups=counts,
        source_rgb_preserved=True,normal_negative_images=3,counterfactual_alignment_checked=True,
        real_camera_tested=False,model_trained=False,
        scope='Package integrity, class coverage, boxes, masks and paired scene alignment. Not physical accuracy.')
    (root/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--input',type=Path,required=True)
    print(json.dumps(verify(parser.parse_args().input),ensure_ascii=False,indent=2))
