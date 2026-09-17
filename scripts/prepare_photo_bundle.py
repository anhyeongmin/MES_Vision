"""Bind evaluated ASH checkpoints to saved-photo inspection without live registration."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mes_vision.training.data import read_json,write_json,sha256,require


def main():
    old=ROOT/'artifacts/training/ash-synthetic-v2-first'; new=ROOT/'artifacts/training/ash-reinforced-v3'
    old_review=read_json(old/'review/summary.json'); thresholds=read_json(new/'review/thresholds.json')
    models={}
    for role,run,threshold in [('overview',old/'overview-objects',old_review['models']['overview-objects']['threshold']),
        ('objects',old/'detail-objects',old_review['models']['detail-objects']['threshold']),
        ('defects',new/'detail-defects',thresholds['models']['reinforced']['predicted_crop_chain'])]:
        metadata=read_json(run/'model.json'); weights=run/metadata['weights']['path']
        require(sha256(weights)==metadata['weights']['sha256'],'Trained artifact changed')
        spec={'backend':'rfdetr-small','weights':weights.relative_to(ROOT).as_posix(),
            'sha256':metadata['weights']['sha256'],'class_names':[c['name'] for c in metadata['categories']],
            'threshold':threshold,'inference_profile':'standard','max_batch_size':1,'device':'cuda'}
        if role=='defects': spec['class_codes']={str(i):n for i,n in enumerate(spec['class_names'])}
        models[role]=spec
    bundle={'schema_version':1,'purpose':'saved_photo_inspection','product_id':'ASH','production_ready':False,
        'training_origin':'synthetic CAD photographs','models':models,
        'threshold_scope':'Candidate display only; validation-selected thresholds, not approved OK/NG criteria',
        'threshold_sources':{str(p.relative_to(ROOT)):sha256(p) for p in (old/'review/summary.json',new/'review/thresholds.json')}}
    directory=ROOT/'configs/inspection'; directory.mkdir(exist_ok=True)
    write_json(directory/'ash-trained-v3.json',bundle)
    paths=[('overview',ROOT/'datasets/ash-synthetic-v2/training/overview-objects/test'/f'test-o{i:04d}.png') for i in (0,1,2)]
    paths += [('detail',ROOT/'datasets/ash-synthetic-reinforced-v3/raw'/f'reinforce-test-d0000-{code}.png')
              for code in ['OK',*[f'NG{i:02d}' for i in range(1,7)]]]
    # Preserve known residual errors for honest visual review, not just successful predictions.
    paths += [('detail',ROOT/'datasets/ash-synthetic-reinforced-v3/raw'/f'reinforce-test-d0004-{code}.png') for code in ('OK','NG03')]
    write_json(directory/'ash-photo-samples.json',{'image_kind':'synthetic','images':[
        {'role':role,'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path)} for role,path in paths]})
    print('Three checkpoint bindings and 12 review photographs prepared; no operational registration changed.')


if __name__=='__main__': main()
