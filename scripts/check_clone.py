"""Check relocated assets and start/close both UIs without camera or robot access."""
import argparse
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    from restore_runtime import restore
    restore(verify_only=True)
    import torch
    from PySide6.QtWidgets import QApplication
    from mes_vision.operation.catalog import default_equipment
    from mes_vision.station.window import StationWindow
    from mes_vision.station.manual_dialog import ManualInspectionDialog
    from mes_vision.training.data import read_json, sha256
    # Verify the currently selected inspection bundle independently of the download list.
    bundle=read_json(ROOT/'configs/inspection/ash-wall-reviewed-v2.json')
    for item in bundle['models'].values():
        assert sha256(ROOT/item['weights']) == item['sha256']
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix='mes-clone-check-', ignore_cleanup_errors=True) as temp:
        runtime=Path(temp)
        window=StationWindow(ROOT,runtime);window.show();app.processEvents();window.close();app.processEvents()
        manual=ManualInspectionDialog(ROOT,runtime,default_equipment()['camera'])
        manual.show();app.processEvents();manual.close();app.processEvents()
        window.deleteLater();manual.deleteLater();app.processEvents()
    if args.gpu:
        if not torch.cuda.is_available():raise RuntimeError('NVIDIA CUDA GPU unavailable. Check the driver.')
        value=torch.ones((16,16),device='cuda');assert (value@value).sum().item()==4096
        print('CUDA OK:',torch.cuda.get_device_name(0))
    print('CLONE CHECK PASSED: assets, bundle hashes and both UI constructors. No camera or robot connected.')


if __name__=='__main__':
    main()
