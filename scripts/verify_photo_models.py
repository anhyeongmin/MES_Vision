"""Real saved-photo adapter execution on the established held-out synthetic set."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mes_vision.training.data import read_json,write_json,sha256
from mes_vision.station.photo_inspection import run_photos


def main():
    output=ROOT/'artifacts/model-ui-integration'
    detail=ROOT/'datasets/ash-synthetic-reinforced-v3/package/defect-crops/test/_annotations.coco.json'
    overview=ROOT/'datasets/ash-synthetic-v2/training/overview-objects/test'
    files=[('overview',overview/i['file_name']) for i in read_json(overview/'_annotations.coco.json')['images']]
    files += [('detail',ROOT/'datasets/ash-synthetic-reinforced-v3/raw'/(i['source_render_id']+'.png')) for i in read_json(detail)['images']]
    request={'output':str(output/'inference'),'runtime':str(ROOT/'artifacts/operation'),
        'bundle':str(ROOT/'configs/inspection/ash-trained-v3.json'),'image_kind':'synthetic',
        'images':[{'role':role,'path':str(path),'sha256':sha256(path)} for role,path in files]}
    write_json(output/'verification-request.json',request)
    report=run_photos(request,ROOT)
    print('COMPLETE',len(report['records']),'actual-model saved photographs',flush=True)


if __name__=='__main__': main()
