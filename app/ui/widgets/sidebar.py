
from PySide6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSizePolicy,
)
from PySide6.QtCore import Signal, Qt


class Sidebar(QFrame):

    page_changed = Signal(str)

    def __init__(self):
        super().__init__()

        self.setObjectName("Sidebar")
        self.setFixedWidth(238)

        self.buttons = {}

        self.init_ui()

    def init_ui(self):

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 22, 16, 18)
        layout.setSpacing(8)

        # =====================================
        # BRAND
        # =====================================

        brand_layout = QHBoxLayout()
        brand_layout.setSpacing(11)

        logo = QLabel("TV")
        logo.setObjectName("BrandLogo")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setFixedSize(42, 42)

        brand_text = QVBoxLayout()
        brand_text.setSpacing(2)

        title = QLabel("TOMATO VISION")
        title.setObjectName("BrandTitle")

        subtitle = QLabel("QUALITY INSPECTION")
        subtitle.setObjectName("BrandSubtitle")

        brand_text.addWidget(title)
        brand_text.addWidget(subtitle)

        brand_layout.addWidget(logo)
        brand_layout.addLayout(brand_text)
        brand_layout.addStretch()

        layout.addLayout(brand_layout)

        layout.addSpacing(32)

        # =====================================
        # NAVIGATION SECTION
        # =====================================

        section_label = QLabel("WORKSPACE")
        section_label.setObjectName("SidebarSection")

        layout.addWidget(section_label)
        layout.addSpacing(7)

        pages = [
            ("▦", "Dashboard", "dashboard"),
            ("▣", "Camera Setup", "camera_setup"),
            ("◎", "Live Inspection", "live_inspection"),
            ("◷", "History", "history"),
            ("⚙", "Settings", "settings"),
        ]

        for icon, label, page_id in pages:

            button = QPushButton(f"  {icon}     {label}")

            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(44)
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )

            button.clicked.connect(
                lambda checked=False, p=page_id:
                self.change_page(p)
            )

            self.buttons[page_id] = button
            layout.addWidget(button)

        layout.addStretch()

        # =====================================
        # SYSTEM INFORMATION
        # =====================================

        separator = QFrame()
        separator.setObjectName("SidebarSeparator")
        separator.setFixedHeight(1)

        layout.addWidget(separator)
        layout.addSpacing(15)

        system_label = QLabel("SYSTEM STATUS")
        system_label.setObjectName("SidebarSection")

        layout.addWidget(system_label)

        status_layout = QHBoxLayout()
        status_layout.setSpacing(8)

        status_dot = QLabel("●")
        status_dot.setObjectName("SystemStatusDot")

        status_text = QLabel("Application Ready")
        status_text.setObjectName("SystemStatusText")

        status_layout.addWidget(status_dot)
        status_layout.addWidget(status_text)
        status_layout.addStretch()

        layout.addLayout(status_layout)

        version = QLabel("Tomato Vision  |  v1.0")
        version.setObjectName("SidebarVersion")

        layout.addSpacing(8)
        layout.addWidget(version)

        self.buttons["dashboard"].setChecked(True)

    def change_page(self, page_id):

        for key, button in self.buttons.items():
            button.setChecked(key == page_id)

        self.page_changed.emit(page_id)
