
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QMessageBox,
    QFrame,
)

from config.settings_manager import SettingsManager


class SettingsPage(QWidget):

    settings_saved = Signal(dict)

    def __init__(self, settings_manager, parent=None):
        super().__init__(parent)

        self.settings_manager = settings_manager

        self.setObjectName("SettingsPage")

        self.init_ui()
        self.load_settings()

    # ==========================================
    # INITIALIZE UI
    # ==========================================

    def init_ui(self):

        main_layout = QVBoxLayout(self)

        main_layout.setContentsMargins(35, 30, 35, 30)
        main_layout.setSpacing(20)

        # Page title
        title = QLabel("Settings")
        title.setObjectName("PageTitle")

        subtitle = QLabel(
            "Configure inspection and object detection parameters."
        )
        subtitle.setObjectName("PageSubtitle")

        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        # ======================================
        # INSPECTION SETTINGS
        # ======================================

        inspection_group = QGroupBox("Inspection Settings")
        inspection_layout = QFormLayout(inspection_group)

        inspection_layout.setContentsMargins(20, 25, 20, 20)
        inspection_layout.setSpacing(18)

        # Required frames
        self.required_frames = QSpinBox()
        self.required_frames.setRange(1, 300)
        self.required_frames.setSuffix(" frames")

        inspection_layout.addRow(
            "Inspection Frames:",
            self.required_frames
        )

        # Empty frames to release
        self.empty_frames = QSpinBox()
        self.empty_frames.setRange(1, 100)
        self.empty_frames.setSuffix(" frames")

        inspection_layout.addRow(
            "Empty Frames to Release:",
            self.empty_frames
        )

        # Required detections
        self.required_detections = QSpinBox()
        self.required_detections.setRange(1, 30)
        self.required_detections.setSuffix(" detections")

        inspection_layout.addRow(
            "Required Detections:",
            self.required_detections
        )

        main_layout.addWidget(inspection_group)

        # ======================================
        # DETECTION SETTINGS
        # ======================================

        detection_group = QGroupBox("Detection Settings")
        detection_layout = QFormLayout(detection_group)

        detection_layout.setContentsMargins(20, 25, 20, 20)
        detection_layout.setSpacing(18)

        # Confidence threshold
        self.confidence_threshold = QDoubleSpinBox()
        self.confidence_threshold.setRange(0.0, 1.0)
        self.confidence_threshold.setSingleStep(0.05)
        self.confidence_threshold.setDecimals(2)

        detection_layout.addRow(
            "Confidence Threshold:",
            self.confidence_threshold
        )

        # IoU threshold
        self.iou_threshold = QDoubleSpinBox()
        self.iou_threshold.setRange(0.0, 1.0)
        self.iou_threshold.setSingleStep(0.05)
        self.iou_threshold.setDecimals(2)

        detection_layout.addRow(
            "IoU Threshold:",
            self.iou_threshold
        )

        self.input_size = QSpinBox()
        self.input_size.setRange(640, 640)
        self.input_size.setValue(640)
        self.input_size.setSuffix(" px")
        self.input_size.setEnabled(False)

        detection_layout.addRow(
            "Input Image Size:",
            self.input_size
        )

        # Maximum detections
        self.max_detections = QSpinBox()
        self.max_detections.setRange(1, 1000)

        detection_layout.addRow(
            "Maximum Detections:",
            self.max_detections
        )

        main_layout.addWidget(detection_group)

        # ======================================
        # DESCRIPTION
        # ======================================

        note = QLabel(
            "Note: Changes are saved to settings.json. "
            "Some detection parameters may require restarting "
            "the inspection worker to take effect."
        )

        note.setWordWrap(True)
        note.setObjectName("SettingsNote")

        main_layout.addWidget(note)

        # ======================================
        # ACTION BUTTONS
        # ======================================

        main_layout.addStretch()

        button_layout = QHBoxLayout()

        button_layout.addStretch()

        self.reset_button = QPushButton("Restore Defaults")
        self.reset_button.setObjectName("SecondaryButton")

        self.save_button = QPushButton("Save Settings")
        self.save_button.setObjectName("PrimaryButton")

        self.reset_button.clicked.connect(
            self.restore_defaults
        )

        self.save_button.clicked.connect(
            self.save_settings
        )

        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.save_button)

        main_layout.addLayout(button_layout)

    # ==========================================
    # LOAD SETTINGS
    # ==========================================

    def load_settings(self):

        settings = self.settings_manager.get_settings()

        inspection = settings["inspection"]
        detection = settings["detection"]

        self.required_frames.setValue(
            inspection["required_frames"]
        )

        self.empty_frames.setValue(
            inspection["empty_frames_to_release"]
        )

        self.required_detections.setValue(
            inspection["required_detections"]
        )

        self.confidence_threshold.setValue(
            detection["confidence_threshold"]
        )

        self.iou_threshold.setValue(
            detection["iou_threshold"]
        )

        self.input_size.setValue(
            detection["input_size"]
        )

        self.max_detections.setValue(
            detection["max_detections"]
        )

    # ==========================================
    # COLLECT UI SETTINGS
    # ==========================================

    def collect_settings(self):

        return {
            "inspection": {
                "required_frames": self.required_frames.value(),
                "empty_frames_to_release": self.empty_frames.value(),
                "required_detections": self.required_detections.value()
            },

            "detection": {
                "confidence_threshold": self.confidence_threshold.value(),
                "iou_threshold": self.iou_threshold.value(),
                "input_size": self.input_size.value(),
                "max_detections": self.max_detections.value()
            }
        }

    # ==========================================
    # SAVE SETTINGS
    # ==========================================

    def save_settings(self):

        settings = self.collect_settings()

        try:

            self.settings_manager.save_settings(settings)

            self.settings_saved.emit(settings)

            QMessageBox.information(
                self,
                "Settings Saved",
                "Settings have been saved successfully."
            )

        except (ValueError, OSError) as exc:

            QMessageBox.critical(
                self,
                "Save Error",
                f"Unable to save settings:\n{exc}"
            )

    # ==========================================
    # RESTORE DEFAULT SETTINGS
    # ==========================================

    def restore_defaults(self):

        answer = QMessageBox.question(
            self,
            "Restore Defaults",
            "Are you sure you want to restore all default settings?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if answer != QMessageBox.StandardButton.Yes:
            return

        try:

            settings = self.settings_manager.reset_settings()

            self.load_settings()

            self.settings_saved.emit(settings)

            QMessageBox.information(
                self,
                "Settings Restored",
                "Default settings have been restored successfully."
            )

        except (ValueError, OSError) as exc:

            QMessageBox.critical(
                self,
                "Restore Error",
                f"Unable to restore settings:\n{exc}"
            )
