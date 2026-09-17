"""Read binary STL assets and render diagnostic orthographic views, never training labels."""
from pathlib import Path
import argparse
import hashlib
import json
import struct
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageDraw


def mesh(path):
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError('Truncated binary STL: ' + str(path))
    count = struct.unpack_from('<I', data, 80)[0]
    if not count or len(data) != 84 + 50 * count:
        raise ValueError('Expected complete binary STL: ' + str(path))
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    triangles = np.frombuffer(data, dtype=dtype, offset=84)['vertices'].astype(float)
    if not np.isfinite(triangles).all():
        raise ValueError('Nonfinite vertices')
    vertices, inverse = np.unique(np.round(triangles.reshape(-1, 3), 5), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    signed_volume = np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
    return triangles, dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), triangles=count,
        bounds=[triangles.min((0, 1)).tolist(), triangles.max((0, 1)).tolist()],
        extents=np.ptp(triangles.reshape(-1, 3), axis=0).tolist(),
        welded_vertices=len(vertices), welding_decimal_places=5, non_two_face_edges=int(np.count_nonzero(counts != 2)),
        degenerate_triangles=int(np.count_nonzero(area2 < 1e-9)), signed_volume_coordinate_units=float(signed_volume))


VIEWS = [('from +X', [0, 0, -1], [0, 1, 0], [1, 0, 0]),
         ('from -X', [0, 0, 1], [0, 1, 0], [-1, 0, 0]),
         ('from +Y', [1, 0, 0], [0, 0, -1], [0, 1, 0]),
         ('from -Y', [1, 0, 0], [0, 0, 1], [0, -1, 0]),
         ('from +Z', [1, 0, 0], [0, 1, 0], [0, 0, 1]),
         ('from -Z', [-1, 0, 0], [0, 1, 0], [0, 0, -1])]


def render(triangles, basis, center, span, size=256):
    xyz = (triangles - center) @ np.array(basis).T
    points = xyz.copy()
    points[:, :, 0] = (xyz[:, :, 0] / span + .5) * size
    points[:, :, 1] = (.5 - xyz[:, :, 1] / span) * size
    depth = np.full((size, size), -np.inf)
    rgb = np.full((size, size, 3), 242, dtype=np.uint8)
    normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    light = np.array([-.4, .6, 1.]); light /= np.linalg.norm(light)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    brightness = .35 + .65 * np.maximum(normals @ light, 0)
    for p, b in zip(points, brightness):
        x0, y0 = np.maximum(np.floor(p[:, :2].min(0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(p[:, :2].max(0)).astype(int), size - 1)
        if x1 < x0 or y1 < y0:
            continue
        den = (p[1,1]-p[2,1])*(p[0,0]-p[2,0])+(p[2,0]-p[1,0])*(p[0,1]-p[2,1])
        if abs(den) < 1e-10:
            continue
        yy, xx = np.mgrid[y0:y1+1, x0:x1+1]; xx = xx + .5; yy = yy + .5
        a = ((p[1,1]-p[2,1])*(xx-p[2,0])+(p[2,0]-p[1,0])*(yy-p[2,1]))/den
        c = ((p[2,1]-p[0,1])*(xx-p[2,0])+(p[0,0]-p[2,0])*(yy-p[2,1]))/den
        d = 1-a-c
        z = a*p[0,2]+c*p[1,2]+d*p[2,2]
        patch = depth[y0:y1+1, x0:x1+1]
        select = (a >= -1e-8) & (c >= -1e-8) & (d >= -1e-8) & (z > patch)
        patch[select] = z[select]
        rgb[y0:y1+1, x0:x1+1][select] = np.array([112, 169, 205]) * b
    return rgb, depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.input.rglob('*.stl'), key=lambda p: (p.stem != 'ASH_OK', p.name))
    if len(files) != 7 or files[0].stem != 'ASH_OK':
        raise ValueError('Expected ASH_OK and six NG binary STL assets')
    assets = [mesh(p) for p in files]
    all_points = np.concatenate([tri.reshape(-1, 3) for tri, _ in assets])
    center = (all_points.min(0) + all_points.max(0)) / 2
    span = np.ptp(all_points, axis=0).max() * 1.18
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    sheet = Image.new('RGB', (6*256, 62+7*288), 'white')
    difference = Image.new('RGB', sheet.size, 'white')
    for canvas in (sheet, difference):
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), 'CAD DIAGNOSTIC - coordinate units unknown - not camera images or approved training labels', fill='black')
        for col, view in enumerate(VIEWS):
            draw.text((col*256+12, 36), view[0], fill='black')
    baseline = []
    detail_sheet = Image.new('RGB', (7*256, 610), 'white')
    detail_draw = ImageDraw.Draw(detail_sheet)
    detail_draw.text((12, 8), 'CAD ONLY - looking from +Y (top: depth shading / bottom: geometric differences in red)', fill='black')
    for row, (triangles, record) in enumerate(assets):
        record['visible_depth_difference_pixels'] = {}
        for col, (name, *basis) in enumerate(VIEWS):
            rgb, depth = render(triangles, basis, center, span)
            if row == 0:
                baseline.append(depth)
            ref = baseline[col]
            visible, ref_visible = np.isfinite(depth), np.isfinite(ref)
            delta = visible ^ ref_visible
            both = visible & ref_visible
            delta[both] |= np.abs(depth[both] - ref[both]) > 1e-4
            overlay = rgb.copy(); overlay[delta] = [224, 56, 55]
            record['visible_depth_difference_pixels'][name] = int(delta.sum())
            y = 62+row*288
            for canvas, pixels in ((sheet, rgb), (difference, overlay)):
                ImageDraw.Draw(canvas).text((col*256+8, y+4), files[row].stem, fill='black')
                canvas.paste(Image.fromarray(pixels), (col*256, y+28))
            Image.fromarray(rgb).save(output / f'{files[row].stem}_{col}.png')
            if col == 2:
                # Deliberate depth coloring makes level steps readable without
                # pretending the simple rasterizer simulates camera illumination.
                lo, hi = all_points[:, 1].min() - center[1], all_points[:, 1].max() - center[1]
                diagnostic = rgb.copy()
                scale = .45 + .55 * (depth[visible] - lo) / max(hi-lo, 1e-9)
                diagnostic[visible] = (diagnostic[visible] * scale[:, None]).astype(np.uint8)
                changed = diagnostic.copy(); changed[delta] = [224, 56, 55]
                detail_draw.text((row*256+8, 35), files[row].stem, fill='black')
                detail_sheet.paste(Image.fromarray(diagnostic), (row*256, 64))
                detail_sheet.paste(Image.fromarray(changed), (row*256, 340))
    sheet.save(output / 'six-views.png'); difference.save(output / 'differences.png')
    detail_sheet.save(output / 'inspection-face.png')
    inventory = [{'path':str(p.resolve()),'bytes':p.stat().st_size,
                  'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(args.input.rglob('*')) if p.is_file()]
    report = dict(schema_version=1, time_utc=datetime.now(timezone.utc).isoformat(),
        input=str(args.input.resolve()), physical_units=None, inspection_view=None, training_labels_approved=False,
        scope='Binary STL integrity and orthographic visible-depth diagnostics only. IPT files inventoried, not decoded. '
              'No mesh repair, self-intersection or printability validation. Red difference regions are not approved defect labels.',
        assets=[record for _, record in assets], inventory=inventory)
    (output / 'manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(str(output / 'six-views.png'))


if __name__ == '__main__':
    main()
