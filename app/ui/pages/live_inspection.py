
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QSizePolicy
)


class LiveInspectionPage(QWidget):

    start_requested = Signal()
    stop_requested = Signal()
    reset_requested = Signal()

    pause_requested = Signal()
    resume_requested = Signal()

    def __init__(self):
        super().__init__()

        self.is_paused = False

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # -------------------------------------------------
        # Page title
        # -------------------------------------------------
        title = QLabel("Live Inspection")
        title.setObjectName("PageTitle")

        description = QLabel(
            "Real-time tomato detection and quality classification"
        )
        description.setObjectName("PageDescription")

        layout.addWidget(title)
        layout.addWidget(description)

        # -------------------------------------------------
        # Cumulative statistics
        # -------------------------------------------------
        totals_layout = QHBoxLayout()
        totals_layout.setSpacing(12)

        self.total_inspected_label = QLabel("Total Inspected: 0")
        self.total_fresh_label = QLabel("Total Fresh: 0")
        self.total_rotten_label = QLabel("Total Rotten: 0")

        for label in (
            self.total_inspected_label,
            self.total_fresh_label,
            self.total_rotten_label,
        ):
            label.setObjectName("StatLabel")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumHeight(55)
            totals_layout.addWidget(label)

        layout.addLayout(totals_layout)

        # -------------------------------------------------
        # Camera preview
        # -------------------------------------------------
        preview_container = QFrame()
        preview_container.setObjectName("PreviewContainer")

        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(10, 10, 10, 10)

        self.preview_label = QLabel("Waiting for camera...")
        self.preview_label.setObjectName("CameraPreview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 400)
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_container, stretch=1)

        # -------------------------------------------------
        # Current frame statistics
        # -------------------------------------------------
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(20)

        self.fps_label = QLabel("FPS: 0")
        self.fresh_label = QLabel("Fresh: 0")
        self.rotten_label = QLabel("Rotten: 0")

        for label in (
            self.fps_label,
            self.fresh_label,
            self.rotten_label,
        ):
            label.setObjectName("StatLabel")
            stats_layout.addWidget(label)

        layout.addLayout(stats_layout)

        # -------------------------------------------------
        # Current inspection result
        # -------------------------------------------------
        self.result_label = QLabel("Inspection Result: Waiting")
        self.result_label.setObjectName("ResultLabel")
        layout.addWidget(self.result_label)

        # -------------------------------------------------
        # Status
        # -------------------------------------------------
        self.status_label = QLabel("Status: Idle")
        self.status_label.setObjectName("StatusLabel")
        layout.addWidget(self.status_label)

        # -------------------------------------------------
        # Control buttons
        # -------------------------------------------------
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        self.start_button = QPushButton("Start Inspection")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop Inspection")
        self.reset_button = QPushButton("Reset")

        self.start_button.clicked.connect(self.on_start_clicked)
        self.pause_button.clicked.connect(self.on_pause_clicked)
        self.stop_button.clicked.connect(self.on_stop_clicked)
        self.reset_button.clicked.connect(self.on_reset_clicked)

        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.reset_button)

        layout.addLayout(button_layout)

    # =====================================================
    # Camera frame
    # =====================================================
    def set_frame(self, frame):
        if frame is None:
            return

        if isinstance(frame, QImage):
            image = frame

        elif isinstance(frame, QPixmap):
            self.preview_label.setPixmap(
                frame.scaled(
                    self.preview_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
            return

        else:
            # Support OpenCV BGR numpy frames
            import cv2

            height, width, channels = frame.shape
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            image = QImage(
                rgb_frame.data,
                width,
                height,
                channels * width,
                QImage.Format_RGB888,
            ).copy()

        pixmap = QPixmap.fromImage(image)

        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    # =====================================================
    # Per-frame statistics
    # =====================================================
    def update_stats(self, fps, fresh, rotten):
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.fresh_label.setText(f"Fresh: {fresh}")
        self.rotten_label.setText(f"Rotten: {rotten}")

    # =====================================================
    # Cumulative statistics
    # =====================================================
    def update_totals(self, total, fresh, rotten):
        self.total_inspected_label.setText(
            f"Total Inspected: {total}"
        )
        self.total_fresh_label.setText(
            f"Total Fresh: {fresh}"
        )
        self.total_rotten_label.setText(
            f"Total Rotten: {rotten}"
        )

    # =====================================================
    # Final classification result
    # =====================================================
    def set_result(
        self,
        inspection_id,
        class_name,
        confidence,
        fresh_votes,
        rotten_votes,
    ):
        self.result_label.setText(
            f"Product #{inspection_id}: "
            f"{class_name} ({confidence:.1%})"
        )

    # =====================================================
    # Status
    # =====================================================
    def set_status(self, message):
        self.status_label.setText(f"Status: {message}")

    # ==========================================
    # BUTTON HANDLERS
    # ==========================================

    def on_start_clicked(self):

        self.start_requested.emit()


    def on_stop_clicked(self):

        self.stop_requested.emit()


    def on_reset_clicked(self):

        self.reset_requested.emit()

        self.update_totals(0, 0, 0)
        self.update_stats(0, 0, 0)

        self.result_label.setText("Inspection Result: Waiting")
        self.set_status("Reset")


    def on_pause_clicked(self):

        if not self.is_paused:

            # Request pause
            self.pause_requested.emit()

            self.set_paused(True)

        else:

            # Request resume
            self.resume_requested.emit()

            self.set_paused(False)


    def set_paused(self, paused):

        self.is_paused = paused

        self.pause_button.setText(
            "Resume" if paused else "Pause"
        )

        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)


    def set_running(self, running):

        self.start_button.setEnabled(not running)
        self.pause_button.setEnabled(running)
        self.stop_button.setEnabled(running)

        if not running:

            self.is_paused = False
            self.pause_button.setText("Pause")