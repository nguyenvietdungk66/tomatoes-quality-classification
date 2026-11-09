from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QStackedWidget,
    QMessageBox,
    QInputDialog,
    QVBoxLayout,
    QLabel,
    QFrame,
    QSizePolicy,
)

from ui.widgets.sidebar import Sidebar
from ui.pages.dashboard import DashboardPage
from ui.pages.camera_setup import CameraSetupPage
from ui.pages.live_inspection import LiveInspectionPage
from ui.pages.history import HistoryPage
from ui.pages.settings import SettingsPage

from services.camera_services import CameraService
from services.inspection_service import InspectionService

from config.settings_manager import SettingsManager

from database.database_manager import DatabaseManager
from database.inspection_repository import InspectionRepository

from PySide6.QtCore import Qt

class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Tomato Quality Inspection"
        )

        self.resize(1400, 850)

        self.init_ui()
        self.init_services()
        self.connect_signals()

    # ==========================================
    # INITIALIZE UI
    # ==========================================

    def init_ui(self):

        self.settings_manager = SettingsManager()

        # =====================================
        # LOAD APPLICATION STYLE
        # =====================================

        style_path = (
            Path(__file__).resolve().parent
            / "styles"
            / "dark_theme.qss"
        )

        if style_path.exists():

            with open(style_path, "r", encoding="utf-8") as file:
                self.setStyleSheet(file.read())

        # =====================================
        # CENTRAL WIDGET
        # =====================================

        central_widget = QWidget()
        central_widget.setObjectName("CentralWidget")

        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # =====================================
        # SIDEBAR
        # =====================================

        self.sidebar = Sidebar()

        main_layout.addWidget(self.sidebar)

        # =====================================
        # MAIN CONTENT
        # =====================================

        content_widget = QWidget()
        content_widget.setObjectName("MainContent")

        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        main_layout.addWidget(content_widget, stretch=1)

        # =====================================
        # TOP BAR
        # =====================================

        self.header = QFrame()
        self.header.setObjectName("TopBar")
        self.header.setFixedHeight(76)

        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(28, 10, 28, 10)
        header_layout.setSpacing(15)

        # Page title

        title_layout = QVBoxLayout()
        title_layout.setSpacing(3)

        self.page_title = QLabel("Dashboard")
        self.page_title.setObjectName("PageTitle")

        self.page_subtitle = QLabel(
            "Overview of inspection performance and system status"
        )
        self.page_subtitle.setObjectName("PageSubtitle")

        title_layout.addWidget(self.page_title)
        title_layout.addWidget(self.page_subtitle)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        # Camera indicator

        self.camera_indicator = QLabel("● CAMERA: UNKNOWN")
        self.camera_indicator.setObjectName("CameraIndicator")
        self.camera_indicator.setProperty("status", "unknown")

        # Model indicator

        self.model_indicator = QLabel("● YOLO: UNKNOWN")
        self.model_indicator.setObjectName("ModelIndicator")
        self.model_indicator.setProperty("status", "unknown")

        # Inspection indicator

        self.inspection_indicator = QLabel("● INSPECTION: STOPPED")
        self.inspection_indicator.setObjectName("InspectionIndicator")
        self.inspection_indicator.setProperty("status", "stopped")

        for indicator in (
            self.camera_indicator,
            self.model_indicator,
            self.inspection_indicator,
        ):

            indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
            indicator.setMinimumHeight(30)
            indicator.setSizePolicy(
                QSizePolicy.Policy.Minimum,
                QSizePolicy.Policy.Fixed,
            )

            header_layout.addWidget(indicator)

        content_layout.addWidget(self.header)

        # =====================================
        # PAGE CONTAINER
        # =====================================

        self.pages = QStackedWidget()
        self.pages.setObjectName("PageContainer")

        self.dashboard_page = DashboardPage()
        self.camera_page = CameraSetupPage()
        self.live_page = LiveInspectionPage()
        self.history_page = HistoryPage()

        self.settings_page = SettingsPage(
            self.settings_manager
        )

        self.page_map = {
            "dashboard": self.dashboard_page,
            "camera_setup": self.camera_page,
            "live_inspection": self.live_page,
            "history": self.history_page,
            "settings": self.settings_page,
        }

        for page in self.page_map.values():
            self.pages.addWidget(page)

        content_layout.addWidget(
            self.pages,
            stretch=1,
        )

        # =====================================
        # BOTTOM STATUS BAR
        # =====================================

        self.status_bar = QFrame()
        self.status_bar.setObjectName("BottomStatusBar")
        self.status_bar.setFixedHeight(34)

        status_layout = QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(20, 0, 20, 0)
        status_layout.setSpacing(18)

        self.system_status_label = QLabel("SYSTEM READY")
        self.system_status_label.setObjectName("BottomStatusText")

        self.bottom_camera_label = QLabel("Camera: Unknown")
        self.bottom_camera_label.setObjectName("BottomStatusText")

        self.bottom_model_label = QLabel("Model: Unknown")
        self.bottom_model_label.setObjectName("BottomStatusText")

        version_label = QLabel("Tomato Vision v1.0")
        version_label.setObjectName("BottomVersion")

        status_layout.addWidget(self.system_status_label)
        status_layout.addWidget(self.bottom_camera_label)
        status_layout.addWidget(self.bottom_model_label)

        status_layout.addStretch()

        status_layout.addWidget(version_label)

        content_layout.addWidget(self.status_bar)

        # =====================================
        # NAVIGATION
        # =====================================

        self.sidebar.page_changed.connect(
            self.switch_page
        )

        # Initial page
        self.update_page_header("dashboard")


    # ==========================================
    # INITIALIZE SERVICES
    # ==========================================

    def init_services(self):

        self.camera_service = CameraService()

        # Project root
        project_root = Path(
            __file__
        ).resolve().parents[2]

        # ONNX model path
        model_path = (
            project_root
            / "model"
            / "best.onnx"
        )

        print("Project root:", project_root)
        print("Model path:", model_path)
        print("Model exists:", model_path.exists())

        # Database
        self.database_manager = DatabaseManager()

        self.inspection_repository = InspectionRepository(
            self.database_manager
        )

        self.history_page.set_repository(
            self.inspection_repository
        )

        self.dashboard_page.set_repository(
            self.inspection_repository
        )

        # Inspection service
        self.inspection_service = InspectionService(
            model_path=str(model_path),
            settings_manager=self.settings_manager,
            repository=self.inspection_repository,
        )

    # ==========================================
    # CONNECT SIGNALS
    # ==========================================

    def connect_signals(self):

        # ======================================
        # CAMERA SETUP PAGE
        # ======================================

        self.camera_page.scan_requested.connect(
            self.camera_service.scan_devices
        )

        self.camera_page.connect_requested.connect(
            self.camera_service.connect_camera
        )

        self.camera_page.disconnect_requested.connect(
            self.disconnect_camera_with_confirmation
        )

        self.camera_page.camera_type_changed.connect(
            self.camera_service.prepare_camera
        )

        self.camera_service.devices_found.connect(
            self.camera_page.set_devices
        )

        self.camera_service.frame_ready.connect(
            self.camera_page.set_frame
        )

        self.camera_service.status_changed.connect(
            self.camera_page.set_status
        )

        self.camera_service.device_info_changed.connect(
            self.camera_page.set_device_info
        )

        self.camera_service.error.connect(
            lambda message: self.camera_page.set_status(
                message,
                connected=False
            )
        )

        # ======================================
        # CAMERA -> INSPECTION SERVICE
        # ======================================

        self.camera_service.frame_ready.connect(
            self.inspection_service.submit_frame
        )

        # ======================================
        # CAMERA -> DASHBOARD
        # ======================================

        self.camera_service.status_changed.connect(
            self.dashboard_page.set_camera_status
        )

        self.camera_service.camera_fps_changed.connect(
            self.dashboard_page.set_camera_fps
        )

        # ======================================
        # CAMERA CONNECTION LOST
        # ======================================

        self.camera_service.connection_lost.connect(
            self.handle_camera_connection_lost
        )

        # ======================================
        # INSPECTION SERVICE -> LIVE PAGE
        # ======================================

        self.inspection_service.frame_ready.connect(
            self.live_page.set_frame
        )

        self.inspection_service.stats_ready.connect(
            self.live_page.update_stats
        )

        self.inspection_service.status_changed.connect(
            self.live_page.set_status
        )

        self.inspection_service.result_ready.connect(
            self.live_page.set_result
        )

        self.inspection_service.totals_ready.connect(
            self.live_page.update_totals
        )

        self.inspection_service.error.connect(
            self.live_page.set_status
        )

        # ======================================
        # INSPECTION SERVICE -> DASHBOARD
        # ======================================

        self.inspection_service.performance_ready.connect(
            self.dashboard_page.set_processing_performance
        )

        self.inspection_service.model_status_changed.connect(
            self.dashboard_page.set_model_status
        )

        self.inspection_service.status_changed.connect(
            self.dashboard_page.set_inspection_status
        )

        # ======================================
        # LIVE INSPECTION PAGE
        # ======================================

        self.live_page.start_requested.connect(
            self.start_inspection
        )

        self.live_page.stop_requested.connect(
            self.stop_inspection
        )

        self.live_page.reset_requested.connect(
            self.reset_inspection
        )

        self.live_page.pause_requested.connect(
            self.inspection_service.pause_inspection
        )

        self.live_page.resume_requested.connect(
            self.inspection_service.resume_inspection
        )

        # ======================================
        # SETTINGS PAGE
        # ======================================

        self.settings_page.settings_saved.connect(
            self.inspection_service.apply_settings
        )

        # ======================================
        # HISTORY PAGE
        # ======================================

        self.history_page.resume_requested.connect(
            self.resume_saved_session
        )

        self.history_page.delete_requested.connect(
            self.delete_saved_session
        )

        # ======================================
        # MAIN WINDOW STATUS INDICATORS
        # ======================================

        self.camera_service.status_changed.connect(
            self.update_camera_header
        )

        self.inspection_service.model_status_changed.connect(
            self.update_model_header
        )

        self.inspection_service.status_changed.connect(
            self.update_inspection_header
        )

    def update_page_header(self, page_id):

        page_info = {
            "dashboard": (
                "Dashboard",
                "Overview of inspection performance and system status",
            ),
            "camera_setup": (
                "Camera Setup",
                "Camera connection and device configuration",
            ),
            "live_inspection": (
                "Live Inspection",
                "Real-time tomato quality monitoring",
            ),
            "history": (
                "Inspection History",
                "Review and manage previous inspection sessions",
            ),
            "settings": (
                "Settings",
                "Application and inspection configuration",
            ),
        }

        title, subtitle = page_info.get(
            page_id,
            ("Tomato Vision", ""),
        )

        self.page_title.setText(title)
        self.page_subtitle.setText(subtitle)


    def set_indicator_status(self, widget, status, text):

        widget.setText(text)
        widget.setProperty("status", status)

        # Refresh QSS property selector
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()


    def update_camera_header(self, *args):

        message = str(args[0]) if args else "Unknown"

        connected = False

        if len(args) > 1 and isinstance(args[1], bool):
            connected = args[1]
        else:
            normalized = message.lower()

            connected = (
                "connected" in normalized
                and "disconnected" not in normalized
                and "not connected" not in normalized
            )

        if connected:

            self.set_indicator_status(
                self.camera_indicator,
                "connected",
                "● CAMERA: CONNECTED",
            )

            self.bottom_camera_label.setText(
                "Camera: Connected"
            )

        else:

            self.set_indicator_status(
                self.camera_indicator,
                "disconnected",
                "● CAMERA: DISCONNECTED",
            )

            self.bottom_camera_label.setText(
                "Camera: Disconnected"
            )


    def update_model_header(self, message):

        message = str(message)
        normalized = message.lower()

        if any(word in normalized for word in (
            "loaded",
            "ready",
            "available",
        )):

            status = "ready"

        elif any(word in normalized for word in (
            "error",
            "failed",
            "not found",
        )):

            status = "error"

        else:

            status = "unknown"

        self.set_indicator_status(
            self.model_indicator,
            status,
            f"● YOLO: {message.upper()}",
        )

        self.bottom_model_label.setText(
            f"Model: {message}"
        )


    def update_inspection_header(self, message):

        message = str(message)
        normalized = message.lower()

        if "pause" in normalized:

            status = "paused"

        elif any(word in normalized for word in (
            "running",
            "inspecting",
            "started",
        )):

            status = "running"

        elif "error" in normalized:

            status = "error"

        else:

            status = "stopped"

        self.set_indicator_status(
            self.inspection_indicator,
            status,
            f"● INSPECTION: {message.upper()}",
        )

        self.system_status_label.setText(
            f"SYSTEM {message.upper()}"
        )

    # ==========================================
    # PAGE NAVIGATION
    # ==========================================

    def switch_page(self, page_id):

        page = self.page_map.get(page_id)

        if page is not None:

            self.pages.setCurrentWidget(page)

            self.update_page_header(page_id)

            # Automatically prepare camera when entering Camera Setup.
            if page is self.camera_page:

                camera_type = (
                    self.camera_page.camera_type.currentText()
                )

                self.camera_service.prepare_camera(
                    camera_type
                )

            if page is self.history_page:

                self.history_page.refresh_sessions()

            elif page is self.dashboard_page:

                self.dashboard_page.refresh_dashboard()

    # ==========================================
    # START INSPECTION
    # ==========================================

    def start_inspection(self):

        # --------------------------------------
        # CHECK CAMERA CONNECTION
        # --------------------------------------

        if not self.camera_service.is_connected():

            dialog = QMessageBox(self)

            dialog.setWindowTitle(
                "Camera Not Connected"
            )

            dialog.setText(
                "Camera has not been connected."
            )

            dialog.setInformativeText(
                "Please connect a camera before starting "
                "the inspection process."
            )

            connect_button = dialog.addButton(
                "Connect",
                QMessageBox.ButtonRole.AcceptRole
            )

            cancel_button = dialog.addButton(
                "Cancel",
                QMessageBox.ButtonRole.RejectRole
            )

            dialog.setDefaultButton(
                connect_button
            )

            dialog.exec()

            clicked = dialog.clickedButton()

            if clicked is connect_button:

                self.switch_page("camera_setup")

            elif clicked is cancel_button:

                return

            return

        # --------------------------------------
        # INPUT SESSION NAME
        # --------------------------------------

        session_name, accepted = QInputDialog.getText(
            self,
            "New Inspection Session",
            "Enter a name for this session:"
        )

        if not accepted:

            self.live_page.set_running(False)

            self.dashboard_page.set_processing_status(
                "Stopped"
            )

            return

        session_name = session_name.strip()

        if not session_name:

            QMessageBox.warning(
                self,
                "Invalid session name",
                "Please enter a non-empty session name."
            )

            self.live_page.set_running(False)

            self.dashboard_page.set_processing_status(
                "Stopped"
            )

            return

        # --------------------------------------
        # START YOLO INSPECTION
        # --------------------------------------

        self.dashboard_page.set_processing_status(
            "Starting"
        )

        if self.inspection_service.start_inspection(
            session_name=session_name
        ):

            self.live_page.set_running(True)

        else:

            self.live_page.set_running(False)

            self.dashboard_page.set_processing_status(
                "Error"
            )
    # ==========================================
    # STOP INSPECTION
    # ==========================================

    def stop_inspection(self):

        self.confirm_and_stop_inspection()


    # ==========================================
    # CONFIRM AND STOP INSPECTION
    # ==========================================

    def confirm_and_stop_inspection(self):

        if self.inspection_service.worker is None:

            self.live_page.set_running(False)

            self.dashboard_page.set_processing_status(
                "Stopped"
            )

            return True

        dialog = QMessageBox(self)

        dialog.setWindowTitle(
            "Stop Inspection"
        )

        dialog.setText(
            "Do you want to save this inspection session?"
        )

        dialog.setInformativeText(
            "Save stores the results and resume checkpoint. "
            "Don't Save discards the current unsaved changes. "
            "Cancel keeps inspection running."
        )

        save_button = dialog.addButton(
            "Save",
            QMessageBox.ButtonRole.AcceptRole
        )

        discard_button = dialog.addButton(
            "Don't Save",
            QMessageBox.ButtonRole.DestructiveRole
        )

        cancel_button = dialog.addButton(
            "Cancel",
            QMessageBox.ButtonRole.RejectRole
        )

        dialog.setDefaultButton(
            save_button
        )

        dialog.exec()

        clicked = dialog.clickedButton()

        # --------------------------------------
        # CANCEL
        # --------------------------------------

        if clicked is cancel_button or clicked is None:

            return False

        # --------------------------------------
        # SAVE
        # --------------------------------------

        if clicked is save_button:

            session_id = self.inspection_service.stop_inspection(
                save_results=True
            )

            if session_id is None:

                QMessageBox.warning(
                    self,
                    "Save failed",
                    "The session could not be saved. "
                    "The camera will remain connected."
                )

                return False

            self.live_page.set_running(False)

            self.dashboard_page.set_processing_status(
                "Stopped"
            )

            self.history_page.refresh_sessions()

            self.dashboard_page.refresh_dashboard()

            return True

        # --------------------------------------
        # DON'T SAVE
        # --------------------------------------

        elif clicked is discard_button:

            self.inspection_service.stop_inspection(
                save_results=False
            )

            self.live_page.set_running(False)

            self.dashboard_page.set_processing_status(
                "Stopped"
            )

            return True

        return False


    # ==========================================
    # DISCONNECT CAMERA WITH CONFIRMATION
    # ==========================================

    def disconnect_camera_with_confirmation(self):

        worker = self.inspection_service.worker

        # --------------------------------------
        # INSPECTION IS RUNNING
        # --------------------------------------

        if worker is not None and worker.isRunning():

            stopped = self.confirm_and_stop_inspection()

            # Cancel or save failure:
            # Keep camera connected.
            if not stopped:

                return

        # --------------------------------------
        # DISCONNECT CAMERA
        # --------------------------------------

        self.camera_service.disconnect_camera()

        self.camera_page.reset_connection_state()

        self.dashboard_page.set_processing_status(
            "Stopped"
        )

    # ==========================================
    # HANDLE UNEXPECTED CAMERA DISCONNECTION
    # ==========================================


    def handle_camera_connection_lost(self, reason):

        print("\n[MAIN WINDOW] handle_camera_connection_lost ENTERED")
        print("[MAIN WINDOW] Camera connection lost:", reason)

        # ==========================================
        # 1. RESET CAMERA SETUP PAGE
        # ==========================================

        print("[MAIN WINDOW] Resetting camera page...")

        self.camera_page.reset_connection_state()

        # ==========================================
        # 2. CHECK INSPECTION STATUS
        # ==========================================

        worker = self.inspection_service.worker

        inspection_running = (
            worker is not None and worker.isRunning()
        )

        print(
            "[MAIN WINDOW] Inspection running:",
            inspection_running
        )

        if inspection_running:

            # Pause inspection
            print("[MAIN WINDOW] Pausing inspection...")

            self.inspection_service.pause_inspection()

            self.live_page.set_paused(True)

            self.live_page.set_status(
                "Paused - Camera disconnected"
            )

            self.dashboard_page.set_processing_status("Paused")

        else:

            self.live_page.set_running(False)

            self.live_page.set_status(
                "Camera disconnected"
            )

            self.dashboard_page.set_processing_status("Stopped")

        # ==========================================
        # 3. SHOW CAMERA CONNECTION LOST WARNING
        # ==========================================

        print("[MAIN WINDOW] Preparing warning dialog...")

        warning_dialog = QMessageBox(self)

        warning_dialog.setWindowTitle(
            "Camera Connection Lost"
        )

        warning_dialog.setIcon(
            QMessageBox.Icon.Warning
        )

        warning_dialog.setText(
            "The camera has been disconnected unexpectedly."
        )

        warning_dialog.setInformativeText(
            f"Reason: {reason}\n\n"
            "The inspection process has been paused. "
            "Please reconnect the camera or stop inspection."
        )

        camera_button = warning_dialog.addButton(
            "Go to Camera Connection",
            QMessageBox.ButtonRole.AcceptRole
        )

        stop_button = warning_dialog.addButton(
            "Stop",
            QMessageBox.ButtonRole.DestructiveRole
        )

        warning_dialog.setDefaultButton(camera_button)

        print("[MAIN WINDOW] Showing warning dialog...")

        warning_dialog.exec()

        print("[MAIN WINDOW] Warning dialog closed.")

        clicked = warning_dialog.clickedButton()

        if clicked is camera_button:

            print("[MAIN WINDOW] User selected Camera Connection.")

            self.switch_page("camera_setup")

        elif clicked is stop_button:

            print("[MAIN WINDOW] User selected Stop.")

            self.switch_page("live_inspection")

        # ==========================================
        # 4. NO ACTIVE INSPECTION
        # ==========================================

        if not inspection_running:

            self.dashboard_page.refresh_dashboard()

            print("[MAIN WINDOW] No active inspection. Handler finished.")

            return

        # ==========================================
        # 5. ASK WHETHER TO SAVE INSPECTION SESSION
        # ==========================================

        print("[MAIN WINDOW] Preparing session confirmation dialog...")

        dialog = QMessageBox(self)

        dialog.setWindowTitle(
            "Inspection Session"
        )

        dialog.setIcon(
            QMessageBox.Icon.Warning
        )

        dialog.setText(
            "The inspection session has been paused."
        )

        dialog.setInformativeText(
            "Do you want to save the inspection results "
            "collected before the camera disconnected?"
        )

        save_button = dialog.addButton(
            "Save",
            QMessageBox.ButtonRole.AcceptRole
        )

        discard_button = dialog.addButton(
            "Don't Save",
            QMessageBox.ButtonRole.DestructiveRole
        )

        cancel_button = dialog.addButton(
            "Cancel",
            QMessageBox.ButtonRole.RejectRole
        )

        dialog.setDefaultButton(save_button)

        print("[MAIN WINDOW] Showing session confirmation dialog...")

        dialog.exec()

        clicked = dialog.clickedButton()

        # ==========================================
        # 6. SAVE SESSION
        # ==========================================

        if clicked is save_button:

            print("[MAIN WINDOW] User selected Save.")

            session_id = self.inspection_service.stop_inspection(
                save_results=True
            )

            if session_id is None:

                QMessageBox.warning(
                    self,
                    "Save Failed",
                    "The inspection session could not be saved. "
                    "Please check the database and try again."
                )

                print("[MAIN WINDOW] Session save failed.")

                return

            self.live_page.set_running(False)

            self.live_page.set_status(
                "Stopped - Session saved"
            )

            self.dashboard_page.set_processing_status(
                "Stopped"
            )

            self.history_page.refresh_sessions()

            self.dashboard_page.refresh_dashboard()

            print("[MAIN WINDOW] Session saved successfully.")

        # ==========================================
        # 7. DISCARD SESSION
        # ==========================================

        elif clicked is discard_button:

            print("[MAIN WINDOW] User selected Don't Save.")

            self.inspection_service.stop_inspection(
                save_results=False
            )

            self.live_page.set_running(False)

            self.live_page.set_status(
                "Stopped - Session discarded"
            )

            self.dashboard_page.set_processing_status(
                "Stopped"
            )

            self.dashboard_page.refresh_dashboard()

            print("[MAIN WINDOW] Session discarded.")

        # ==========================================
        # 8. CANCEL - KEEP SESSION PAUSED
        # ==========================================

        else:

            print("[MAIN WINDOW] User selected Cancel.")

            self.live_page.set_paused(True)

            self.live_page.set_status(
                "Paused - Waiting for camera reconnection"
            )

            self.dashboard_page.set_processing_status(
                "Paused"
            )

            print("[MAIN WINDOW] Inspection remains paused.")


    # ==========================================
    # RESET INSPECTION
    # ==========================================

    def reset_inspection(self):

        answer = QMessageBox.question(
            self,
            "Reset Inspection",
            "Reset the current session statistics and product ID sequence?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if answer == QMessageBox.StandardButton.Yes:

            self.inspection_service.reset_inspection()

    # ==========================================
    # RESUME SAVED SESSION
    # ==========================================

    def resume_saved_session(self, session_id):

        if (
            self.inspection_service.worker is not None
            and self.inspection_service.worker.isRunning()
        ):

            QMessageBox.warning(
                self,
                "Inspection is running",
                "Stop the current inspection before resuming another session."
            )

            return

        answer = QMessageBox.question(
            self,
            "Resume Session",
            "Resume this saved session? The camera must be connected. "
            "If a tomato was being evaluated when the session was saved, "
            "that unfinished evaluation will restart with a fresh product.",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if answer != QMessageBox.StandardButton.Yes:
            return

        self.dashboard_page.set_processing_status(
            "Starting"
        )

        if self.inspection_service.resume_session(session_id):

            self.live_page.set_running(True)

            self.pages.setCurrentWidget(
                self.live_page
            )

        else:

            self.live_page.set_running(False)

            self.dashboard_page.set_processing_status(
                "Error"
            )

    # ==========================================
    # DELETE SAVED SESSION
    # ==========================================

    def delete_saved_session(self, session_id):

        answer = QMessageBox.question(
            self,
            "Delete Session",
            "Permanently delete this session and all its product results?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if answer != QMessageBox.StandardButton.Yes:
            return

        try:

            self.inspection_repository.delete_session(
                session_id
            )

            self.history_page.refresh_sessions()

            self.dashboard_page.refresh_dashboard()

        except Exception as exc:

            QMessageBox.critical(
                self,
                "Delete failed",
                str(exc)
            )

    # ==========================================
    # CLOSE APPLICATION
    # ==========================================

    def closeEvent(self, event):

        worker = self.inspection_service.worker

        if worker is not None and worker.isRunning():

            dialog = QMessageBox(self)

            dialog.setWindowTitle(
                "Exit Application"
            )

            dialog.setText(
                "An inspection is running. "
                "What would you like to do before exiting?"
            )

            save_button = dialog.addButton(
                "Save and Exit",
                QMessageBox.ButtonRole.AcceptRole
            )

            discard_button = dialog.addButton(
                "Don't Save and Exit",
                QMessageBox.ButtonRole.DestructiveRole
            )

            cancel_button = dialog.addButton(
                "Cancel",
                QMessageBox.ButtonRole.RejectRole
            )

            dialog.setDefaultButton(
                save_button
            )

            dialog.exec()

            clicked = dialog.clickedButton()

            if clicked is cancel_button or clicked is None:

                event.ignore()
                return

            if clicked is save_button:

                session_id = self.inspection_service.stop_inspection(
                    save_results=True
                )

                if session_id is None:

                    QMessageBox.warning(
                        self,
                        "Save failed",
                        "The session could not be saved. "
                        "The application will remain open."
                    )

                    event.ignore()
                    return

            elif clicked is discard_button:

                self.inspection_service.stop_inspection(
                    save_results=False
                )

        try:

            self.camera_service.disconnect_camera()

        except Exception as exc:

            print(
                f"Camera shutdown error: {exc}"
            )

        event.accept()