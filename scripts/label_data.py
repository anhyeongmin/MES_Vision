"""Open the local desktop annotation helper (no web server)."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PySide6.QtWidgets import QApplication, QMessageBox
from mes_vision.data_management.labeler import Labeler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--runtime",type=Path,default=Path(__file__).resolve().parents[1]/'artifacts/operation')
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("MES Vision 데이터 라벨링")
    from mes_vision.operation.preferences import Preferences
    from mes_vision.i18n import set_language,install_display_font,install_qt_translator
    from mes_vision.theme import apply_theme
    preferences=Preferences(args.runtime);set_language(preferences.value['language'])
    install_display_font(app);install_qt_translator(app)
    apply_theme(preferences.value['theme'],preferences.value['text_scale']/100)
    try:
        window = Labeler(args.workspace,runtime=args.runtime)
        window.show()
    except Exception as exc:
        QMessageBox.critical(None, "프로젝트 열기 실패", str(exc))
        return 1
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
