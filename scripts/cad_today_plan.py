"""Frozen 144-condition plan, full marginal grid coverage without Cartesian explosion."""
import json,random
from pathlib import Path
def prepare(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rng=random.Random(20260914);conditions=[]
    for i,yaw in enumerate(range(0,360,5)):
        conditions.append(dict(roll=-45+5*(i%19),pitch=-45+5*((i*7)%19),yaw=yaw,distance_mm=[110,130,150,170][i%4],light=(i//4)%4,group='wide_coverage'))
    for i,yaw in enumerate(range(0,360,5)):
        conditions.append(dict(roll=rng.choice([-10,-5,0,5,10]),pitch=rng.choice([-10,-5,0,5,10]),yaw=yaw,distance_mm=[110,130,150,170][(i+1)%4],light=(i//4+1)%4,group='near_topdown'))
    keys=[tuple(c[k] for k in ['roll','pitch','yaw','distance_mm','light']) for c in conditions]
    assert len(keys)==len(set(keys))==144
    for k,values in [('roll',range(-45,46,5)),('pitch',range(-45,46,5)),('yaw',range(0,360,5)),('distance_mm',[110,130,150,170]),('light',range(4))]:
        assert {c[k] for c in conditions}==set(values)
    plan=dict(seed=20260914,images=1008,conditions=conditions,scope='Every marginal grid value included for each class, not every RPY/light/distance combination. 72 wide conditions plus 72 near-top-down. No automatic label approval or training.')
    p=out/'plan.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==plan,'Plan changed; use new output'
    else:p.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    return plan
if __name__=='__main__':
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        a=prepare(d);assert prepare(d)==a
    print('Plan coverage, uniqueness, repeatability passed: 1008 images')
