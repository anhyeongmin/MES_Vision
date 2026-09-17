"""Product-version-specific capture domains; old overhead recipes are never inferred."""
from pathlib import Path
from copy import deepcopy
import json,os
from uuid import uuid4
from filelock import FileLock
from mes_vision.training.data import require,read_json,sha256


def recipe_path(runtime,product):
    # Product ids are data; use a digest to prevent path traversal.
    from mes_vision.anomaly.features import fingerprint
    return Path(runtime)/'station-recipes'/(fingerprint(product['id'])+'.json')


def validate_recipe(value,product,equipment,*,verify=True):
    require(value['schema_version']==1 and value['product_id']==product['id']
            and value['product_version']==product['version'] and value['equipment_version']==equipment['version'],
            'Capture recipe must match the current product and equipment versions')
    require(value['validation_reference'].strip(),'Capture-domain validation record required')
    from mes_vision.operation.acquisition import inspection_camera, identity
    c=inspection_camera(equipment['camera'])
    require(value.get('acquisition_identity')==identity(equipment['camera']),'Capture acquisition changed')
    require(value['image_size']==[c['width'],c['height']],'Capture resolution changed')
    domains=value['capture_domains']
    require(domains.get('overview_objects')=='overview','Overview model domain required')
    for role in ('objects','defects','anomaly','geometry'):
        if product.get(role): require(domains.get(role)=='detail','Detail model/criteria domain required: '+role)
    w=value['detail_workspace']; width,height=value['image_size']
    require(w=={'roi':[[0,0],[width,0],[width,height],[0,height]],'excluded':[],
        'validation_reference':value['validation_reference']},'Explicit full-detail-image workspace required')
    asset=value['overview_asset']
    require(asset and asset.get('class_names') and asset.get('sha256'),'Registered overview object model required')
    if verify: require(sha256(Path(asset['weights']))==asset['sha256'],'Overview model changed')
    return deepcopy(value)


def load_recipe(runtime,product,equipment,*,verify=True):
    return validate_recipe(read_json(recipe_path(runtime,product)),product,equipment,verify=verify)


def save_recipe(runtime,value,product,equipment,*,previous=None):
    value=validate_recipe(value,product,equipment); path=recipe_path(runtime,product); path.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(path)+'.lock',timeout=0):
        current=read_json(path) if path.exists() else None
        require(current==previous,'Capture recipe changed; reopen settings')
        tmp=path.with_name('.'+uuid4().hex+'.tmp')
        try:
            with tmp.open('x',encoding='utf-8') as f:
                json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
            tmp.replace(path)
        finally:
            if tmp.exists(): tmp.unlink()
    return value
