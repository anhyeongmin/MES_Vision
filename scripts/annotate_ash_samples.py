"""Generate CAD geometric masks, COCO boxes, and separate review overlays for sample RGB."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mes_vision.synthetic.projection import world_triangles,raster_depth,difference_mask,bbox


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def coco(role):
    names=[f'NG{i:02d}' for i in range(1,7)] if role=='defects' else ['ASH']
    return dict(info=dict(description='CAD synthetic review samples; unsplit; not physical validation',version='1'),
        images=[],annotations=[],categories=[dict(id=i+1,name=n) for i,n in enumerate(names)])


def annotation(data,image_id,category,box,**extra):
    data['annotations'].append(dict(id=len(data['annotations'])+1,image_id=image_id,category_id=category,
        bbox=box,area=box[2]*box[3],iscrowd=0,**extra))


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args(); root=args.input.resolve()
    report=json.loads((root/'render-manifest.json').read_text(encoding='utf-8'))
    if report['status']!='rendered' or len(report['frames'])!=24: raise ValueError('Expected complete 24-image render')
    geometry_path=root/'geometry.json'
    if hashlib.sha256(geometry_path.read_bytes()).hexdigest()!=report['geometry_sha256']: raise ValueError('Geometry changed')
    geometry={k:np.asarray(v) for k,v in json.loads(geometry_path.read_text()).items()}
    for folder in ('annotations','masks','review'): (root/folder).mkdir(exist_ok=False)
    sets={role:coco('defects' if role=='detail-defects' else 'objects') for role in ('overview-objects','detail-objects','detail-defects')}
    details=[]; objects_count=0; defect_pixels={}; sheet=Image.new('RGB',(7*280,3*308+55),'#e5e7eb')
    rawsheet=sheet.copy()
    ImageDraw.Draw(sheet).text((12,12),'SYNTHETIC ONLY | defect boxes + visible geometry masks | 3 lighting / rotation conditions',font=font(20),fill='black')
    ImageDraw.Draw(rawsheet).text((12,12),'SYNTHETIC ONLY | cropped previews of unmarked 1280 x 720 RGB | not captured with a real camera',font=font(20),fill='black')
    for index,frame in enumerate(report['frames'],1):
        path=root/frame['file_name']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=frame['image_sha256']: raise ValueError('Rendered image changed')
        with Image.open(path) as img: image=img.convert('RGB')
        cam=frame['camera']
        if image.size!=(cam['width'],cam['height']) or not cam['projection_verified_against_blender']: raise ValueError('Camera mismatch')
        depths=[raster_depth(world_triangles(geometry[obj['code']],obj['matrix_world']),cam) for obj in frame['objects']]
        combined=np.stack(depths); nearest=np.argmin(combined,axis=0); seen=np.isfinite(np.min(combined,axis=0))
        image_info=dict(id=index,file_name=frame['file_name'],width=image.width,height=image.height,
            scene_id=frame['id'],synthetic=True,camera_model='U20CAM-720P nominal pinhole',
            cad_variants=[o['code'] for o in frame['objects']],condition=frame['condition']['id'])
        role=frame['domain']+'-objects'; sets[role]['images'].append(image_info)
        overlay=image.copy(); draw=ImageDraw.Draw(overlay)
        for i,obj in enumerate(frame['objects']):
            mask=seen&(nearest==i); box=bbox(mask)
            if box is None or int(mask.sum())<30: raise ValueError('Object missing from frame')
            if box[0]<=0 or box[1]<=0 or box[0]+box[2]>=image.width or box[1]+box[3]>=image.height: raise ValueError('Object cut off')
            annotation(sets[role],index,1,box,cad_variant=obj['code']); objects_count+=1
            if frame['domain']=='overview':
                x,y,w,h=box; draw.rectangle((x,y,x+w-1,y+h-1),outline='#16a34a',width=3)
                draw.text((x,y-22),'ASH / '+obj['code'],font=font(18),fill='#064e3b')
        if frame['domain']=='detail':
            sets['detail-defects']['images'].append(image_info)
            obj=frame['objects'][0]; code=obj['code']
            normal=raster_depth(world_triangles(geometry['OK'],obj['matrix_world']),cam)
            changed=difference_mask(depths[0],normal)
            defect_pixels[frame['id']]=int(changed.sum())
            if code=='OK' and changed.any(): raise ValueError('Normal image acquired a defect annotation')
            if code!='OK' and int(changed.sum())<8: raise ValueError('Defect has no usable visible geometric footprint')
            mask_name='masks/'+frame['id']+'.png'
            Image.fromarray(changed.astype(np.uint8)*255).save(root/mask_name)
            defect_box=bbox(changed,padding=2)
            if defect_box:
                annotation(sets['detail-defects'],index,int(code[2:]),defect_box,mask_file=mask_name,
                    geometric_pixels=int(changed.sum()),label_method='visible_counterfactual_depth_difference',
                    label_scope='synthetic_geometry_only')
                pixels=np.asarray(overlay).copy()
                pixels[changed]=(pixels[changed]*.72+np.array([255,55,35])*.28).astype(np.uint8)
                overlay=Image.fromarray(pixels); draw=ImageDraw.Draw(overlay)
                x,y,w,h=defect_box; draw.rectangle((x,y,x+w-1,y+h-1),outline='#ef2929',width=3)
                draw.text((x,max(0,y-28)),code,font=font(24),fill='#9b1111')
            # Crop review thumbnails only; original and COCO coordinates remain 1280x720.
            x,y,w,h=bbox(seen,padding=35)
            cell='ABC'.index(frame['condition']['id']); col=0 if code=='OK' else int(code[2:])
            for board,picture in ((sheet,overlay),(rawsheet,image)):
                crop=picture.crop((x,y,x+w,y+h)); crop.thumbnail((264,264))
                bx=col*280+(280-crop.width)//2; by=55+cell*308+32+(264-crop.height)//2
                board.paste(crop,(bx,by)); ImageDraw.Draw(board).text((col*280+12,55+cell*308+4),frame['id'],font=font(21),fill='black')
            details.append(dict(id=frame['id'],code=code,object_bbox=bbox(seen),defect_bbox=defect_box,mask=mask_name,
                geometric_pixels=int(changed.sum()),rgb=frame['file_name'],normal_pair=frame['normal_pair']))
        overlay.save(root/'review'/(frame['id']+'.png'))
        print('ANNOTATED '+frame['id'],flush=True)
    for role,data in sets.items(): write(root/'annotations'/(role+'.coco.json'),data)
    sheet.save(root/'review'/'detail-contact-sheet.png'); rawsheet.save(root/'review'/'detail-rgb-sheet.png')
    summary=dict(schema_version=1,status='generated_pending_visual_review',kind='synthetic',images=24,
        detail_images=21,overview_images=3,object_annotations=objects_count,defect_annotations=len(sets['detail-defects']['annotations']),
        normal_images_without_defect_annotations=3,split='unsplit_review_samples',
        depth_tolerance_metres=1e-6,bbox_padding_pixels=2,details=details,
        camera_target=report['camera_target'],camera_match=report['camera_match'],
        limitations=['Uncalibrated ideal pinhole; no measured lens distortion, focus, camera noise or MJPEG effects.',
            'Same seven CAD shapes reused across conditions. Not an independent physical or unseen-defect evaluation.',
            'Depth change marks visible geometry, not pixel-level radiometric changes or all appearance effects.',
            'RGB appearance is rendered; procedural fine roughness is not a measured print-layer reproduction.'],
        model_training_performed=False,production_camera_driver_changed=False)
    write(root/'sample-report.json',summary)
    files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for folder in ('rgb','annotations','masks','review') for p in (root/folder).rglob('*') if p.is_file()}
    write(root/'sample-hashes.json',files)


if __name__=='__main__': main()
