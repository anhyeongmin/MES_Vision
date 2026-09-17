"""Create stratified visual review sheets after export; never annotate model inputs."""
from pathlib import Path
import argparse,json
from PIL import Image,ImageDraw,ImageFont


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def font(size): return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)


def tile(root,task,size=(260,260),overlay=True):
    image=Image.open(root/'raw'/(task['id']+'.png')).convert('RGB')
    q=read(root/'quality'/(task['id']+'.json')); draw=ImageDraw.Draw(image)
    if overlay:
        for v in q['defects'] if task['domain']=='detail' else q['objects']:
            x,y,w,h=v['bbox']; draw.rectangle((x,y,x+w-1,y+h-1),outline='#e63228' if task['domain']=='detail' else '#16a34a',width=3)
    if task['domain']=='detail':
        x,y,w,h=q['objects'][0]['bbox']; image=image.crop((max(0,x-20),max(0,y-20),min(1280,x+w+20),min(720,y+h+20)))
    image.thumbnail(size)
    return image


def sheet(root,tasks,columns,path,title,size=(280,300),overlay=True):
    width,height=size; rows=(len(tasks)+columns-1)//columns
    board=Image.new('RGB',(columns*width,rows*height+45),'#e6e9ed'); draw=ImageDraw.Draw(board)
    draw.text((12,10),title,font=font(18),fill='black')
    for i,t in enumerate(tasks):
        img=tile(root,t,(width-14,height-42),overlay)
        x=(i%columns)*width; y=45+(i//columns)*height
        draw.text((x+8,y+3),t['id'],font=font(16),fill='black')
        board.paste(img,(x+(width-img.width)//2,y+32+(height-42-img.height)//2))
    board.save(path)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True); args=parser.parse_args()
    root=args.root.resolve(); plan=read(root/'plan.json'); exported=read(root/'training/export-report.json')
    accepted=[t for t in plan['tasks'] if t['group'] not in exported['rejected_groups']]
    out=root/'review'; out.mkdir(exist_ok=False)
    detailed=[]; overviews=[]; worst=[]
    for split in ('train','valid','test'):
        candidates=[t for t in accepted if t['split']==split and t['domain']=='detail']
        group=sorted({t['group'] for t in candidates})[0]
        detailed.extend(t for t in candidates if t['group']==group)
        candidates=[t for t in accepted if t['split']==split and t['domain']=='overview' and t['objects']]
        overviews.extend(candidates[::max(1,len(candidates)//3)][:3])
    for code in [f'NG{i:02d}' for i in range(1,7)]:
        candidates=[t for t in accepted if t['domain']=='detail' and t['objects'][0]['code']==code]
        worst.append(min(candidates,key=lambda t:read(root/'quality'/(t['id']+'.json'))['appearance']['mean_rgb_difference']))
    sheet(root,detailed,7,out/'detail-labels.png','SYNTHETIC | train / valid / test groups | defect boxes')
    sheet(root,detailed,7,out/'detail-rgb.png','SYNTHETIC | unmarked RGB crops | shared CAD across splits',overlay=False)
    sheet(root,overviews,3,out/'overview-labels.png','SYNTHETIC | varied object counts and positions',size=(480,300))
    sheet(root,worst,3,out/'lowest-contrast.png','SYNTHETIC | lowest accepted paired RGB difference per NG class',size=(400,390))
    (out/'selected-scenes.json').write_text(json.dumps(dict(detail=[t['id'] for t in detailed],overview=[t['id'] for t in overviews],
        lowest_contrast=[t['id'] for t in worst]),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__': main()
