"""Snapshot user CAD sources and class mapping; does not create images or train weights."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
from inspect_ash_cad import mesh

CLASSES = [
    ('OK', 'ASH_OK', '정상'),
    ('NG01', 'ASH_NG01_MissingBoss', '보스 누락'),
    ('NG02', 'ASH_NG02_ExtraMaterial', '추가 돌출 재료'),
    ('NG03', 'ASH_NG03_Crack', '균열'),
    ('NG04', 'ASH_NG04_WallDeformation', '벽 변형'),
    ('NG05', 'ASH_NG05_HoleDefect', '구멍 불량'),
    ('NG06', 'ASH_NG06_SurfaceDefect', '표면 결함'),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.input.resolve(), args.output.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Source and destination must be separate')
    records = []
    for code, name, label in CLASSES:
        paths = []
        for suffix in ('.stl', '.ipt'):
            matches = list(root.rglob(name + suffix))
            if len(matches) != 1:
                raise ValueError('Expected one file: ' + name + suffix)
            paths.append(matches[0])
        _, info = mesh(paths[0])
        if info['non_two_face_edges'] or info['degenerate_triangles']:
            raise ValueError('Mesh needs inspection: ' + name)
        records.append(dict(code=code, label_ko=label, product_class='ASH',
            paths=paths, hashes=[digest(p) for p in paths], mesh_check=info))
    output.mkdir(parents=True, exist_ok=False)
    for item in records:
        folder = output / item['code']; folder.mkdir()
        item['files'] = []
        for path, expected in zip(item.pop('paths'), item.pop('hashes')):
            dest = folder / path.name
            shutil.copyfile(path, dest)
            if digest(dest) != expected or digest(path) != expected:
                raise ValueError('Source changed while copying: ' + str(path))
            item['files'].append(dict(original=str(path), path=dest.relative_to(output).as_posix(), sha256=expected))
    manifest = dict(schema_version=1, kind='cad_source_registry', product_id='ASH',
        created_utc=datetime.now(timezone.utc).isoformat(), source=str(root),
        mesh_geometry_modified=False, user_scale_instruction='STL 그대로 출력',
        scale=dict(coordinate_multiplier=1.0, physical_unit_confirmed=False,
                   mm_per_coordinate_assumption=1.0,
                   note='Coordinates preserved. 16x14x20 mm only if the slicer imports coordinate units as mm at 100%.'),
        inspection_face=dict(camera_from_axis='+Y', image_right_axis='+X', image_up_axis='-Z',
            status='inferred_from_visible_geometry', note='All six NG visible-depth changes occur on this face in six-axis diagnostics.'),
        training=dict(status='source_prepared_only', images_generated=0, weights_trained=False,
            overview_categories=['ASH'], detail_object_categories=['ASH'],
            known_defect_categories=[v[0] for v in CLASSES if v[0] != 'OK'],
            normal_defect_annotations=[], defect_localization_approved=False,
            note='Depth-difference diagnostics are not RGB training inputs or approved annotations. '
                 'Synthetic evaluation with reused CAD identities does not establish new-shape or physical generalization.'),
        variants=records)
    (output / 'cad-sources.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(str(output / 'cad-sources.json'))


if __name__ == '__main__':
    main()
