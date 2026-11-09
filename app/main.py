import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main():

    app = QApplication(sys.argv)

    # Load stylesheet
    project_root = Path(__file__).resolve().parent.parent

    style_path = (
        project_root
        /"app"
        / "ui"
        / "styles"
        / "dark_theme.qss"
    )

    with open(style_path, "r", encoding="utf-8") as file:
        app.setStyleSheet(file.read())

    # Main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()