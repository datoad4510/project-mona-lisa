"""BCI Paint — entry point.

Usage
-----
Run with the real Neurosity Crown:
    python main.py

Run in mock / demo mode (no headset required):
    python main.py --mock

Add --debug-click to enable mouse-click region selection while testing:
    python main.py --mock --debug-click
"""

import sys
import argparse

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui     import QIcon, QPalette, QColor
from PyQt5.QtCore    import Qt

from ui.app import BCIPaintApp


def _apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    dark    = QColor(26, 26, 46)
    mid     = QColor(22, 33, 62)
    light   = QColor(233, 69, 96)
    text    = QColor(224, 224, 224)

    palette.setColor(QPalette.Window,          dark)
    palette.setColor(QPalette.WindowText,      text)
    palette.setColor(QPalette.Base,            mid)
    palette.setColor(QPalette.AlternateBase,   dark)
    palette.setColor(QPalette.ToolTipBase,     dark)
    palette.setColor(QPalette.ToolTipText,     text)
    palette.setColor(QPalette.Text,            text)
    palette.setColor(QPalette.Button,          mid)
    palette.setColor(QPalette.ButtonText,      text)
    palette.setColor(QPalette.BrightText,      light)
    palette.setColor(QPalette.Highlight,       light)
    palette.setColor(QPalette.HighlightedText, dark)
    app.setPalette(palette)


def main() -> None:
    parser = argparse.ArgumentParser(description="BCI Paint — Neurosity Crown")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run without a physical headset (synthetic EEG generation)",
    )
    parser.add_argument(
        "--debug-click",
        action="store_true",
        help="Allow mouse clicks to select/fill regions (useful for UI testing)",
    )
    args = parser.parse_args()

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("BCI Paint")
    _apply_dark_theme(qt_app)

    window = BCIPaintApp(mock=args.mock, debug_click=args.debug_click)
    window.show()

    sys.exit(qt_app.exec_())


if __name__ == "__main__":
    main()
