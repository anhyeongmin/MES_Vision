"""Blender material-roughness sensitivity only; not a measured PETG optical model."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import render_ash_dataset as worker
original=worker.scene_setup


def setup(condition,overview,samples):
    scene,material,devices=original(condition,overview,samples)
    import bpy
    table=bpy.data.materials['TableMat'].node_tree.nodes.get('Principled BSDF')
    table.inputs['Roughness'].default_value=condition['table_roughness']
    return scene,material,devices


worker.scene_setup=setup
if __name__=='__main__': worker.main()
