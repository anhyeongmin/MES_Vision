"""Conservative sampled top-down CAD screening; never generates robot commands."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from inspect_ash_cad import mesh, render

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/wall-grasp-2x'

def rectangles(sign, x, margin=0):
    # Union of finger footprints through the entire symmetric parallel close.
    # Target contact planes z=16,20 are not expanded into the intended wall.
    bounds = [(x-4-margin, x+4+margin, -7-margin, 16),
              (x-4-margin, x+4+margin, 20, 43+margin)]
    return [(a,b,min(sign*c,sign*d),max(sign*c,sign*d)) for a,b,c,d in bounds]

def clear(depth, xx, zz, rects, tip):
    for a,b,c,d in rects:
        occupied = (xx>a) & (xx<b) & (zz>c) & (zz<d)
        if np.any(depth[occupied] > tip + 1e-6):
            return False
    return True

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registry = json.loads((ROOT/'datasets/cad-sources/ash-v1/cad-sources.json').read_text(encoding='utf-8'))
    size, span = 800, 80
    axis = (np.arange(size)+.5)*span/size-span/2
    xx, zz = np.meshgrid(axis, -axis)
    records, depths = [], []
    for v in registry['variants']:
        tri, info = mesh(Path(v['mesh_check']['path']))
        assert info['sha256'] == v['mesh_check']['sha256'], 'Source changed'
        _, depth = render(tri*2, [[1,0,0],[0,0,1],[0,1,0]], np.zeros(3), span, size)
        depths.append(depth)
        results = []
        for sign in (1,-1):
            for insertion in (3,6,8,10,12):
                xs = [float(x) for x in np.arange(-7,7.01,.25)
                      if clear(depth,xx,zz,rectangles(sign,x,1),28-insertion-1)]
                results.append(dict(wall_sign=sign,insertion_mm=insertion,
                                    accepted_sampled_x_mm=xs))
        records.append(dict(code=v['code'],source=info,screening=results))
    candidates=[]
    for sign,x in ((1,2),(-1,-2)):
        checks=[clear(d,xx,zz,rectangles(sign,x,1),19) for d in depths]
        candidates.append(dict(id='A' if sign==1 else 'B',
            wall_center_stl_axes_mm=[x,24,sign*18],
            finger_tip_height_mm=20,insertion_mm=8,contact_height_range_mm=[20,28],
            closing_axis='STL Z',open_gap_mm=30,nominal_contact_gap_mm=4,
            passes_all_variants=bool(all(checks)),per_variant=dict(zip([r['code'] for r in records],checks))))
    # Sanity checks: empty field passes, a tall obstacle in swept region fails.
    test=np.full_like(depths[0],-np.inf)
    assert clear(test,xx,zz,rectangles(1,2,1),19)
    test[(abs(xx-2)<1)&(abs(zz)<1)]=24
    assert not clear(test,xx,zz,rectangles(1,2,1),19)
    report=dict(status='CAD_CANDIDATE_NOT_ROBOT_COMMAND',scale=2,
        units='mm assuming original STL coordinate unit is mm',
        coordinate_system='Original STL axes scaled 2x: X horizontal, Z horizontal, Y vertical; floor Y=0, rim Y=28. Plot has +Z upward.',
        assumptions=['Symmetric parallel translating fingers, not pivoting',
            'Each finger is a rectangular 10 x 8 mm cross section and 25 mm long',
            'Fixed gripper center during closing; nominal straight wall thickness 4 mm',
            'One isolated stationary part, open face +Y; gripper body and mount not modeled'],
        method='0.1 mm top-height raster; union of complete parallel finger closing sweep including open descent. Sweep inflated 1 mm tangentially and at far ends, tip lowered 1 mm. Intended wall contact faces not inflated. Candidate centers sampled every 0.25 mm.',
        limitations=['Raster screening is not exact collision proof or physical gripping validation',
            'No friction, force, deformation, tolerances, robot calibration or neighboring objects validated',
            'CAD-relative contact center is not robot TCP; physical finger kinematics must be checked'],
        candidates=candidates,variants=records)
    (OUT/'candidates.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,axs=plt.subplots(2,4,figsize=(16,10),layout='constrained')
    for ax,r,d in zip(axs.flat,records,depths):
        ax.imshow(np.ma.masked_invalid(d),extent=(-40,40,-40,40),origin='upper',vmin=0,vmax=28,cmap='Blues')
        for sign,x,color,label in ((1,2,'#df7a00','A'),(-1,-2,'#009777','B')):
            for a,b,c,e in rectangles(sign,x):
                ax.add_patch(Rectangle((a,c),b-a,e-c,fill=False,edgecolor=color,linewidth=1.4,linestyle='--'))
            ax.plot(x,sign*18,'o',color=color)
            ax.text(x+1,sign*18,label,color=color,weight='bold')
        ax.set(title=r['code'],xlabel='X (mm)',ylabel='Z (mm)',xlim=(-22,22),ylim=(-46,46),aspect='equal')
        ax.grid(alpha=.15)
    axs.flat[-1].axis('off')
    axs.flat[-1].text(0,1,'2x STL / 7 variants\n\nA: X=+2, Z=+18 mm\nB: X=-2, Z=-18 mm\n\nDashed: finger closing sweep\nFinger section: 10 x 8 mm\nOpen gap: 30 mm\nInsertion candidate: 8 mm\n\n+Y is vertical, rim Y=28\nCAD candidates only\nNot robot coordinates',va='top',fontsize=12)
    fig.suptitle('Wall grasp candidates — parallel-jaw assumption / top view',fontsize=17)
    fig.savefig(OUT/'candidates.png',dpi=150)
    print(json.dumps(candidates,indent=2))

if __name__=='__main__':
    main()
