from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw
def review(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    checks=[];frames=m['frames'];selected=frames if len(frames)<=28 else frames[:28]
    sheet=Image.new('RGB',(1400,300*((len(selected)+6)//7)),'#eeeeee');draw=ImageDraw.Draw(sheet)
    wall=Image.new('RGB',(1200,360),'white');wi=0
    for i,r in enumerate(frames):
        im=Image.open(root/'rgb'/(r['id']+'.png')).convert('RGB');a=np.array(im)
        obj=np.array(Image.open(root/'masks'/(r['id']+'-object.png')).convert('L'))>127
        diff=np.array(Image.open(root/'masks'/(r['id']+'-difference.png')).convert('L'))>127
        assert im.size==(1280,720) and int(diff.sum())==r['difference_pixels']
        assert (not diff.any()) if r['code']=='OK' else diff.any()
        x,y,w,h=r['object_box_xywh'];box=(max(0,x-15),max(0,y-15),min(im.width,x+w+15),min(im.height,y+h+15))
        checks.append(dict(id=r['id'],object_pixels=int(obj.sum()),difference_pixels=int(diff.sum()),white_pixel_fraction_in_object=float(np.all(a[obj]>=250,axis=1).mean()),annotation_status='PROPOSAL_REQUIRES_REVIEW'))
        if i<len(selected):
            crop=im.crop(box);crop.thumbnail((195,260));sheet.paste(crop,(i%7*200,i//7*300));draw.text((i%7*200,i//7*300+265),r['id'],fill='black')
        if r['code']=='NG04' and wi<4:
            overlay=a.copy();overlay[diff]=(a[diff]*.5+np.array([255,40,40])*.5).astype(np.uint8)
            crop=Image.fromarray(overlay).crop(box);crop.thumbnail((290,320));wall.paste(crop,(wi*300,0));ImageDraw.Draw(wall).text((wi*300,330),r['id'],fill='black');wi+=1
    sheet.save(root/'samples.jpg');wall.save(root/'NG04-difference.jpg')
    (root/'quality-review.json').write_text(json.dumps(dict(images=len(frames),planned=m['planned_images'],complete=len(frames)==m['planned_images'],training_performed=False,labels_approved=False,checks=checks),indent=2))
    print('IMAGE/MASK REVIEW SAVED:',root/'samples.jpg')
if __name__=='__main__':
    import sys
    review(sys.argv[1])
