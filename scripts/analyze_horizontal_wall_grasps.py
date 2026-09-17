"""Screen both X-closing wall grasps; no robot commands or physical approval."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from inspect_ash_cad import mesh, render
from analyze_wall_grasps import clear

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/wall-grasp-2x/horizontal'

def sweep(sign,z,margin=0):
    # Right wall center X=14; inner/outer contact planes X=12/16.
    # Open inner finger X=-11..-1; closed inner finger X=2..12.
    return [(min(sign*a,sign*b),max(sign*a,sign*b),z-4-margin,z+4+margin)
            for a,b in [(-11-margin,12),(16,39+margin)]]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    reg=json.loads((ROOT/'datasets/cad-sources/ash-v1/cad-sources.json').read_text(encoding='utf-8'))
    n=1000; span=100
    axis=(np.arange(n)+.5)*span/n-span/2
    xx,zz=np.meshgrid(axis,-axis)
    records=[]; depths=[]
    for v in reg['variants']:
        tri,source=mesh(Path(v['mesh_check']['path']))
        assert source['sha256']==v['mesh_check']['sha256']
        _,depth=render(tri*2,[[1,0,0],[0,0,1],[0,1,0]],np.zeros(3),span,n)
        depths.append(depth)
        rows=[]
        for sign in [1,-1]:
            for margin in [0,1,1.5]:
                for insertion in [3,6,8,10,12]:
                    zs=[float(z) for z in np.arange(-11,11.01,.25)
                        if clear(depth,xx,zz,sweep(sign,z,margin),28-insertion-margin)]
                    rows.append(dict(side='right' if sign==1 else 'left',margin_mm=margin,
                                     insertion_mm=insertion,accepted_z_samples_mm=zs))
        records.append(dict(code=v['code'],source=source,results=rows))
    common=[]
    for side in ['right','left']:
        for margin in [0,1,1.5]:
            sets=[set(next(a['accepted_z_samples_mm'] for a in r['results'] if a['side']==side and a['margin_mm']==margin and a['insertion_mm']==8)) for r in records]
            common.append(dict(side=side,margin_mm=margin,insertion_mm=8,
                               common_z_samples_mm=sorted(set.intersection(*sets))))
    # Geometry regression: open and closed envelopes and obstacle detection.
    assert sweep(1,0)==[(-11,12,-4,4),(16,39,-4,4)]
    assert sweep(-1,0)==[(-12,11,-4,4),(-39,-16,-4,4)]
    empty=np.full((n,n),-np.inf)
    assert clear(empty,xx,zz,sweep(1,0),20)
    empty[(abs(xx+10)<.2)&(abs(zz)<.2)]=28
    assert not clear(empty,xx,zz,sweep(1,0),20)
    report=dict(status='CAD_PATH_SCREEN_ONLY_NOT_ROBOT_COMMAND',scale=2,
        assumptions=['Original STL unit assumed mm, uniform 2x','Symmetric parallel 10x8x25mm fingers; open gap 30mm',
            'Nominal side wall contact planes X=+/-12 and +/-16; no adaptive contact fit to deformed wall'],
        method='0.1mm top height raster; complete closing sweep union; Z centers sampled 0.25mm. Margin expands tangential width, far sweep ends and downward depth; excludes intended contact faces.',
        limitations=['No grasp force/contact area validation; a clear path alone does not prove contact',
            'Exactly 1mm nominal opposite-wall clearance: 1mm expansion may touch without penetration, so it is not positive remaining clearance',
            'Body, mounting, other objects, motion errors and real finger kinematics excluded'],
        common_at_8mm=common,variants=records)
    (OUT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,axs=plt.subplots(2,4,figsize=(16,9),layout='constrained')
    for ax,r,d in zip(axs.flat,records,depths):
        ax.imshow(np.ma.masked_invalid(d),extent=(-50,50,-50,50),origin='upper',vmin=0,vmax=28,cmap='Blues')
        labels=[]
        for sign,color,label in [(1,'#cf7900','R'),(-1,'#008e89','L')]:
            ok=clear(d,xx,zz,sweep(sign,0,1),19)
            labels.append(f'{label}: '+('clear path' if ok else 'collision'))
            for a,b,c,e in sweep(sign,0):
                ax.add_patch(Rectangle((a,c),b-a,e-c,fill=False,edgecolor=color,linestyle='--'))
            ax.plot(sign*14,0,'o',color=color)
            ax.text(sign*14,6,label,color=color,ha='center',weight='bold')
        ax.set(title=r['code']+' | '+', '.join(labels),xlabel='X (mm)',ylabel='Z (mm)',xlim=(-43,43),ylim=(-24,24),aspect='equal')
        ax.grid(alpha=.15)
    axs.flat[-1].axis('off')
    axs.flat[-1].text(0,1,'Horizontal / X closing\n\nR: X=+14, Z=0 mm\nL: X=-14, Z=0 mm\nInsertion: 8 mm\n\nDashed = full closing sweep\nStatus uses 1 mm expansion\n1 mm opposite-wall gap nominal\n\nClear path is not proven grip.\nParallel-jaw assumption only.',va='top',fontsize=12)
    fig.suptitle('Horizontal wall grasp screening / 2x STL / seven variants',fontsize=16)
    fig.savefig(OUT/'horizontal.png',dpi=150)
    print(json.dumps(common))
    print(json.dumps({r['code']:{side:next(a['accepted_z_samples_mm'] for a in r['results'] if a['side']==side and a['margin_mm']==1 and a['insertion_mm']==8) for side in ['right','left']} for r in records}))

if __name__=='__main__': main()
