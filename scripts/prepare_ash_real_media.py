"""Prepare empty real collections and fine-tune configs without starting training."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mes_vision.data_management.collection import Collection
from mes_vision.training.config import JobConfig
from mes_vision.training.data import sha256, require, write_json


def main():
    bundle = json.loads((ROOT / 'configs/inspection/ash-trained-v3.json').read_text(encoding='utf-8'))
    for name in ('overview', 'detail'):
        path = ROOT / 'collections' / f'ash-real-{name}-v1'
        if not path.exists():
            Collection.create(path, 'ASH', kind='real')
        collection = Collection(path)
        require(collection.data['kind'] == 'real' and collection.data['product_id'] == 'ASH', 'Wrong collection')
    folder = ROOT / 'configs/training/ash-real-v1'
    folder.mkdir(parents=True, exist_ok=True)
    report = []
    for name, key, role in [('overview-objects', 'overview', 'object_detector'),
                            ('detail-objects', 'objects', 'object_detector'),
                            ('detail-defects', 'defects', 'known_defect_detector')]:
        model = bundle['models'][key]
        weights = ROOT / model['weights']
        require(sha256(weights) == model['sha256'], 'Initial checkpoint changed')
        config = dict(role=role, dataset_dir=f'../../../datasets/ash-real-v1/{name}',
                      initial_weights=Path(os.path.relpath(weights, folder)).as_posix(),
                      initial_sha256=model['sha256'], epochs=30, batch_size=2, grad_accum_steps=4,
                      lr=0.00005, lr_encoder=0.00005, num_workers=0, seed=42,
                      precision='bf16-mixed', synthetic=False)
        target = folder / f'{name}.json'
        if not target.exists():
            write_json(target, config)
        loaded = JobConfig.load(target)
        require(not loaded.synthetic and loaded.role == role, 'Wrong fine-tune role')
        require(sha256(Path(loaded.initial_weights)) == loaded.initial_sha256, 'Checkpoint mismatch')
        report.append({'config': str(target.relative_to(ROOT)), 'sha256': sha256(target),
                       'initial_weights_verified': True,
                       'dataset_present': Path(loaded.dataset_dir).exists()})
    write_json(ROOT / 'artifacts/real-media-preparation/config-verification.json',
               {'status': 'PREPARED_NOT_TRAINED', 'jobs': report,
                'note': 'Hyperparameters are starting values, not real-data validated.'})
    print('Two real collections and three verified fine-tune configurations ready. No training started.')


if __name__ == '__main__':
    main()
