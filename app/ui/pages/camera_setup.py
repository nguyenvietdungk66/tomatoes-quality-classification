
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QGroupBox,
    QFormLayout,
    QFrame,
    QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage


class CameraSetupPage(QWidget):

    # Signals for CameraService integration
    scan_requested = Signal(str)
    connect_requested = Signal(str, str)
    disconnect_requested = Signal()
    camera_type_changed = Signal(str)

    def __init__(self):
        super().__init__()

        self._current_pixmap = None
        self._connected = False

        self.init_ui()

    def init_ui(self):

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 25, 30, 25)
        main_layout.setSpacing(20)

        # =================================================
        # PAGE HEADER
        # =================================================

        header_layout = QVBoxLayout()

        title = QLabel("Camera Setup")
        title.setObjectName("PageTitle")

        subtitle = QLabel(
            "Configure industrial cameras and acquisition settings"
        )
        subtitle.setObjectName("PageSubtitle")

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)

        main_layout.addLayout(header_layout)

        # =================================================
        # CONTENT
        # =================================================

        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # LEFT PANEL
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)

        # Camera configuration
        camera_group = QGroupBox("Camera Configuration")
        camera_form = QFormLayout(camera_group)
        camera_form.setSpacing(15)

        self.camera_type = QComboBox()
        self.camera_type.addItems([
            "Hikrobot",
            "Webcam / USB Camera"
        ])

        self.device_combo = QComboBox()
        self.device_combo.setPlaceholderText(
            "Scan for available cameras"
        )

        self.scan_button = QPushButton("Scan Devices")
        self.scan_button.setObjectName("SecondaryButton")

        camera_form.addRow("Camera Type:", self.camera_type)
        camera_form.addRow("Device:", self.device_combo)
        camera_form.addRow("", self.scan_button)

        left_panel.addWidget(camera_group)

        # Connection controls
        connection_group = QGroupBox("Connection")
        connection_layout = QVBoxLayout(connection_group)
        connection_layout.setSpacing(12)

        self.connect_button = QPushButton("Connect Camera")
        self.connect_button.setObjectName("PrimaryButton")

        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.setObjectName("DangerButton")
        self.disconnect_button.setEnabled(False)

        self.status_label = QLabel("● Disconnected")
        self.status_label.setObjectName("StatusDisconnected")

        connection_layout.addWidget(self.status_label)
        connection_layout.addWidget(self.connect_button)
        connection_layout.addWidget(self.disconnect_button)

        left_panel.addWidget(connection_group)

        # Device information
        info_group = QGroupBox("Device Information")
        info_form = QFormLayout(info_group)

        self.model_value = QLabel("--")
        self.serial_value = QLabel("--")
        self.resolution_value = QLabel("--")

        info_form.addRow("Model:", self.model_value)
        info_form.addRow("Serial Number:", self.serial_value)
        info_form.addRow("Resolution:", self.resolution_value)

        left_panel.addWidget(info_group)
        left_panel.addStretch()

        content_layout.addLayout(left_panel, 1)

        # =================================================
        # RIGHT PANEL - CAMERA PREVIEW
        # =================================================

        preview_group = QGroupBox("Live Camera Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(12, 20, 12, 12)

        self.preview_label = QLabel(
            "Camera preview will appear here"
        )

        self.preview_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.preview_label.setObjectName("CameraPreview")
        self.preview_label.setMinimumSize(500, 350)
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        preview_layout.addWidget(self.preview_label)

        content_layout.addWidget(preview_group, 2)

        main_layout.addLayout(content_layout)

        # =================================================
        # SIGNAL CONNECTIONS
        # =================================================

        self.camera_type.currentTextChanged.connect(
            self.on_camera_type_changed
        )

        self.scan_button.clicked.connect(
            self.on_scan_clicked
        )

        self.connect_button.clicked.connect(
            self.on_connect_clicked
        )

        self.disconnect_button.clicked.connect(
            self.disconnect_requested.emit
        )

    # =====================================================
    # UI EVENTS
    # =====================================================

    def on_camera_type_changed(self, camera_type):

        self.device_combo.clear()
        self.camera_type_changed.emit(camera_type)

    def on_scan_clicked(self):

        self.scan_requested.emit(
            self.camera_type.currentText()
        )

    def on_connect_clicked(self):

        device = self.device_combo.currentText()

        if not device:
            self.set_status(
                "No camera selected",
                connected=False
            )
            return

        self.connect_requested.emit(
            self.camera_type.currentText(),
            device
        )

    # =====================================================
    # METHODS FOR CAMERASERVICE
    # =====================================================

    def set_devices(self, devices):

        self.device_combo.clear()
        self.device_combo.addItems(devices)

    def set_status(self, message, connected=False):

        self._connected = connected
        self.status_label.setText(f"● {message}")

        if connected:
            self.status_label.setObjectName(
                "StatusConnected"
            )
        else:
            self.status_label.setObjectName(
                "StatusDisconnected"
            )

        # Refresh stylesheet after changing object name
        self.status_label.style().unpolish(
            self.status_label
        )
        self.status_label.style().polish(
            self.status_label
        )

        self.connect_button.setEnabled(not connected)
        self.disconnect_button.setEnabled(connected)

    def set_device_info(
        self,
        model="--",
        serial="--",
        resolution="--"
    ):

        self.model_value.setText(str(model))
        self.serial_value.setText(str(serial))
        self.resolution_value.setText(str(resolution))

    def set_frame(self, frame):

        if frame is None:
            return

        height, width, channels = frame.shape

        if channels != 3:
            return

        image = QImage(
            frame.data,
            width,
            height,
            frame.strides[0],
            QImage.Format.Format_BGR888
        ).copy()

        self._current_pixmap = QPixmap.fromImage(image)
        self.update_preview()

    def update_preview(self):

        if self._current_pixmap is None:
            return

        scaled_pixmap = self._current_pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.preview_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):

        super().resizeEvent(event)
        self.update_preview()

    def reset_connection_state(self):

        # Reset connection state
        self._connected = False

        # Clear device selection
        self.device_combo.clear()

        # Clear device information
        self.set_device_info(
            model="--",
            serial="--",
            resolution="--"
        )

        # Clear preview
        self._current_pixmap = None

        self.preview_label.clear()

        self.preview_label.setText(
            "Camera preview will appear here"
        )

        # Reset status and buttons
        self.set_status(
            "Disconnected",
            connected=False
        )