from datetime import date, timedelta

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QColor, QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QFrame,
    QSizePolicy,
)


class StatCard(QFrame):
    """Card displaying one KPI."""

    def __init__(self, title, value="0", color="#E5E7EB"):
        super().__init__()

        self.setObjectName("DashboardCard")
        self.setMinimumHeight(100)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(
            "font-size: 13px; color: #9CA3AF;"
        )

        self.value_label = QLabel(str(value))
        self.value_label.setStyleSheet(
            f"font-size: 25px; font-weight: bold; color: {color};"
        )

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addStretch()

    def set_value(self, value):
        self.value_label.setText(str(value))


class QualityPieChart(QWidget):
    """Simple Fresh / Rotten pie chart using QPainter."""

    def __init__(self):
        super().__init__()

        self.fresh = 0
        self.rotten = 0

        self.setMinimumSize(220, 200)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    def set_data(self, fresh, rotten):

        self.fresh = max(0, int(fresh))
        self.rotten = max(0, int(rotten))

        self.update()

    def paintEvent(self, event):

        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )

        total = self.fresh + self.rotten

        center_x = self.width() / 2
        center_y = self.height() / 2

        size = min(
            self.width() - 35,
            self.height() - 35,
            170
        )

        rect = QRectF(
            center_x - size / 2,
            center_y - size / 2,
            size,
            size,
        )

        if total == 0:

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#4B5563"))
            painter.drawEllipse(rect)

            painter.setPen(QColor("#D1D5DB"))

            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No data",
            )

            return

        fresh_angle = round(
            self.fresh / total * 360 * 16
        )

        rotten_angle = 360 * 16 - fresh_angle

        painter.setPen(Qt.PenStyle.NoPen)

        if self.fresh > 0:

            painter.setBrush(QColor("#22C55E"))

            painter.drawPie(
                rect,
                90 * 16,
                fresh_angle,
            )

        if self.rotten > 0:

            painter.setBrush(QColor("#EF4444"))

            painter.drawPie(
                rect,
                90 * 16 + fresh_angle,
                rotten_angle,
            )

        # Inner circle creates a donut chart.
        inner_size = size * 0.55

        inner_rect = QRectF(
            center_x - inner_size / 2,
            center_y - inner_size / 2,
            inner_size,
            inner_size,
        )

        painter.setBrush(QColor("#1F2937"))
        painter.drawEllipse(inner_rect)

        painter.setPen(QColor("#FFFFFF"))

        font = QFont()
        font.setBold(True)
        font.setPointSize(15)

        painter.setFont(font)

        painter.drawText(
            inner_rect,
            Qt.AlignmentFlag.AlignCenter,
            str(total),
        )


class DailyBarChart(QWidget):
    """Bar chart showing inspected tomatoes over the last seven days."""

    def __init__(self):
        super().__init__()

        self.data = []

        self.setMinimumHeight(230)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    def set_data(self, data):

        self.data = data
        self.update()

    def paintEvent(self, event):

        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )

        width = self.width()
        height = self.height()

        left = 42
        right = 15
        top = 25
        bottom = 42

        chart_width = width - left - right
        chart_height = height - top - bottom

        painter.setPen(QColor("#9CA3AF"))

        painter.drawLine(
            left,
            top,
            left,
            top + chart_height,
        )

        painter.drawLine(
            left,
            top + chart_height,
            left + chart_width,
            top + chart_height,
        )

        if not self.data:

            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No inspection data",
            )

            return

        max_value = max(
            (item["total"] for item in self.data),
            default=0,
        )

        max_value = max(max_value, 1)

        count = len(self.data)

        slot_width = chart_width / count

        bar_width = min(
            32,
            slot_width * 0.55
        )

        for index, item in enumerate(self.data):

            value = item["total"]

            bar_height = (
                value / max_value * (chart_height - 20)
            )

            x = (
                left
                + index * slot_width
                + (slot_width - bar_width) / 2
            )

            y = top + chart_height - bar_height

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#3B82F6"))

            painter.drawRoundedRect(
                QRectF(
                    x,
                    y,
                    bar_width,
                    bar_height
                ),
                4,
                4,
            )

            painter.setPen(QColor("#D1D5DB"))

            painter.drawText(
                QRectF(
                    x - 10,
                    y - 22,
                    bar_width + 20,
                    18,
                ),
                Qt.AlignmentFlag.AlignCenter,
                str(value),
            )

            date_text = item["date"][5:]

            painter.drawText(
                QRectF(
                    left + index * slot_width,
                    top + chart_height + 8,
                    slot_width,
                    25,
                ),
                Qt.AlignmentFlag.AlignCenter,
                date_text,
            )


class DashboardPage(QWidget):

    def __init__(self):
        super().__init__()

        self.repository = None

        self.init_ui()

    def init_ui(self):

        main_layout = QVBoxLayout(self)

        main_layout.setContentsMargins(
            24,
            20,
            24,
            20
        )

        main_layout.setSpacing(16)

        # ==========================================
        # PAGE TITLE
        # ==========================================

        title = QLabel("Dashboard")
        title.setObjectName("PageTitle")

        description = QLabel(
            "Quality statistics and inspection overview"
        )

        description.setObjectName("PageDescription")

        main_layout.addWidget(title)
        main_layout.addWidget(description)

        # ==========================================
        # KPI CARDS
        # ==========================================

        cards_layout = QGridLayout()
        cards_layout.setSpacing(12)

        self.total_card = StatCard(
            "TOTAL INSPECTED",
            "0",
            "#60A5FA",
        )

        self.fresh_card = StatCard(
            "FRESH TOMATOES",
            "0",
            "#22C55E",
        )

        self.rotten_card = StatCard(
            "ROTTEN TOMATOES",
            "0",
            "#EF4444",
        )

        self.fresh_rate_card = StatCard(
            "FRESH RATE",
            "0%",
            "#22C55E",
        )

        cards_layout.addWidget(self.total_card, 0, 0)
        cards_layout.addWidget(self.fresh_card, 0, 1)
        cards_layout.addWidget(self.rotten_card, 0, 2)
        cards_layout.addWidget(self.fresh_rate_card, 0, 3)

        main_layout.addLayout(cards_layout)

        # ==========================================
        # CHARTS SECTION
        # ==========================================

        charts_layout = QHBoxLayout()
        charts_layout.setSpacing(12)

        # Quality distribution

        quality_frame = QFrame()
        quality_frame.setObjectName("DashboardCard")

        quality_layout = QVBoxLayout(quality_frame)

        quality_layout.setContentsMargins(
            16,
            14,
            16,
            14
        )

        quality_title = QLabel("Quality Distribution")

        quality_title.setStyleSheet(
            "font-size: 15px; font-weight: bold;"
        )

        self.pie_chart = QualityPieChart()

        self.fresh_legend = QLabel(
            "Fresh: 0 (0%)"
        )

        self.rotten_legend = QLabel(
            "Rotten: 0 (0%)"
        )

        self.fresh_legend.setStyleSheet(
            "color: #22C55E; font-weight: bold;"
        )

        self.rotten_legend.setStyleSheet(
            "color: #EF4444; font-weight: bold;"
        )

        quality_layout.addWidget(quality_title)

        quality_layout.addWidget(
            self.pie_chart,
            stretch=1,
        )

        quality_layout.addWidget(self.fresh_legend)
        quality_layout.addWidget(self.rotten_legend)

        # Daily production

        daily_frame = QFrame()
        daily_frame.setObjectName("DashboardCard")

        daily_layout = QVBoxLayout(daily_frame)

        daily_layout.setContentsMargins(
            16,
            14,
            16,
            14
        )

        daily_title = QLabel(
            "Inspection Volume - Last 7 Days"
        )

        daily_title.setStyleSheet(
            "font-size: 15px; font-weight: bold;"
        )

        self.daily_chart = DailyBarChart()

        daily_layout.addWidget(daily_title)

        daily_layout.addWidget(
            self.daily_chart,
            stretch=1,
        )

        charts_layout.addWidget(
            quality_frame,
            stretch=1
        )

        charts_layout.addWidget(
            daily_frame,
            stretch=2
        )

        main_layout.addLayout(charts_layout)

        # ==========================================
        # SYSTEM PERFORMANCE
        # ==========================================

        performance_frame = QFrame()
        performance_frame.setObjectName("DashboardCard")

        performance_layout = QVBoxLayout(performance_frame)

        performance_layout.setContentsMargins(
            16,
            14,
            16,
            14
        )

        performance_layout.setSpacing(12)

        performance_title = QLabel(
            "System Performance"
        )

        performance_title.setStyleSheet(
            "font-size: 15px; font-weight: bold;"
        )

        performance_layout.addWidget(
            performance_title
        )

        performance_grid = QGridLayout()
        performance_grid.setSpacing(10)

        self.camera_status_card = StatCard(
            "CAMERA STATUS",
            "Disconnected",
            "#EF4444",
        )

        self.camera_fps_card = StatCard(
            "CAMERA FPS",
            "--",
            "#60A5FA",
        )

        self.model_status_card = StatCard(
            "YOLO MODEL",
            "Not loaded",
            "#9CA3AF",
        )

        self.processing_status_card = StatCard(
            "INSPECTION STATUS",
            "Idle",
            "#9CA3AF",
        )

        self.processing_fps_card = StatCard(
            "PROCESSING FPS",
            "--",
            "#60A5FA",
        )

        self.inference_time_card = StatCard(
            "INFERENCE TIME",
            "--",
            "#FBBF24",
        )

        self.database_status_card = StatCard(
            "DATABASE STATUS",
            "Unknown",
            "#9CA3AF",
        )

        performance_grid.addWidget(
            self.camera_status_card,
            0,
            0
        )

        performance_grid.addWidget(
            self.camera_fps_card,
            0,
            1
        )

        performance_grid.addWidget(
            self.model_status_card,
            0,
            2
        )

        performance_grid.addWidget(
            self.processing_status_card,
            0,
            3
        )

        performance_grid.addWidget(
            self.processing_fps_card,
            1,
            0
        )

        performance_grid.addWidget(
            self.inference_time_card,
            1,
            1
        )

        performance_grid.addWidget(
            self.database_status_card,
            1,
            2
        )

        performance_layout.addLayout(
            performance_grid
        )

        main_layout.addWidget(
            performance_frame
        )

        # ==========================================
        # SESSION SUMMARY
        # ==========================================

        self.session_count_label = QLabel(
            "Saved Inspection Sessions: 0"
        )

        self.session_count_label.setObjectName(
            "StatLabel"
        )

        main_layout.addWidget(
            self.session_count_label
        )

        main_layout.addStretch()

    # ==========================================
    # REPOSITORY
    # ==========================================

    def set_repository(self, repository):

        self.repository = repository

        self.refresh_dashboard()

    def refresh_dashboard(self):

        if self.repository is None:
            return

        try:

            statistics = (
                self.repository.get_dashboard_statistics()
            )

            daily_statistics = (
                self.repository.get_daily_statistics(
                    days=7
                )
            )

            total = statistics["total_inspected"]
            fresh = statistics["total_fresh"]
            rotten = statistics["total_rotten"]

            session_count = statistics["total_sessions"]

            fresh_rate = (
                fresh / total * 100
                if total > 0
                else 0
            )

            rotten_rate = (
                rotten / total * 100
                if total > 0
                else 0
            )

            self.total_card.set_value(
                f"{total:,}"
            )

            self.fresh_card.set_value(
                f"{fresh:,}"
            )

            self.rotten_card.set_value(
                f"{rotten:,}"
            )

            self.fresh_rate_card.set_value(
                f"{fresh_rate:.1f}%"
            )

            self.pie_chart.set_data(
                fresh,
                rotten
            )

            self.fresh_legend.setText(
                f"Fresh: {fresh:,} ({fresh_rate:.1f}%)"
            )

            self.rotten_legend.setText(
                f"Rotten: {rotten:,} ({rotten_rate:.1f}%)"
            )

            self.daily_chart.set_data(
                daily_statistics
            )

            self.session_count_label.setText(
                f"Saved Inspection Sessions: {session_count:,}"
            )

            # Successful repository queries mean database is accessible.
            self.set_database_status("Connected")

        except Exception as exc:

            self.set_database_status("Error")

            self.session_count_label.setText(
                f"Dashboard data error: {exc}"
            )

            print(
                f"Dashboard refresh error: {exc}"
            )

    # ==========================================
    # SYSTEM PERFORMANCE UPDATE METHODS
    # ==========================================

    def set_camera_status(self, status, connected):

        if connected:

            if status == "Paused":
                display_status = "Paused"
            else:
                display_status = "Connected"

            self.camera_status_card.set_value(
                display_status
            )

            self.camera_status_card.value_label.setStyleSheet(
                "font-size: 25px; font-weight: bold; color: #22C55E;"
            )

        else:

            if status == "Connecting...":
                display_status = "Connecting"
                color = "#FBBF24"

            elif status == "Connection error":
                display_status = "Error"
                color = "#EF4444"

            else:
                display_status = "Disconnected"
                color = "#EF4444"

            self.camera_status_card.set_value(
                display_status
            )

            self.camera_status_card.value_label.setStyleSheet(
                f"font-size: 25px; font-weight: bold; color: {color};"
            )

            self.camera_fps_card.set_value("--")

    def set_camera_fps(self, fps):

        if fps <= 0:

            self.camera_fps_card.set_value("--")

        else:

            self.camera_fps_card.set_value(
                f"{fps:.1f}"
            )

    def set_model_status(self, status):

        self.model_status_card.set_value(status)

        if status == "Loaded":

            color = "#22C55E"

        elif status == "Loading":

            color = "#FBBF24"

        elif status == "Error":

            color = "#EF4444"

        else:

            color = "#9CA3AF"

        self.model_status_card.value_label.setStyleSheet(
            f"font-size: 25px; font-weight: bold; color: {color};"
        )

    def set_processing_performance(
        self,
        processing_fps,
        inference_ms
    ):

        if processing_fps <= 0:

            self.processing_fps_card.set_value("--")

        else:

            self.processing_fps_card.set_value(
                f"{processing_fps:.1f}"
            )

        if inference_ms <= 0:

            self.inference_time_card.set_value("--")

        else:

            self.inference_time_card.set_value(
                f"{inference_ms:.2f} ms"
            )

        if processing_fps > 0:

            self.set_processing_status("Running")

    def set_processing_status(self, status):

        self.processing_status_card.set_value(status)

        if status == "Running":

            color = "#22C55E"

        elif status == "Starting":

            color = "#FBBF24"

        elif status == "Paused":

            color = "#FBBF24"

        elif status == "Error":

            color = "#EF4444"

        else:

            color = "#9CA3AF"

        self.processing_status_card.value_label.setStyleSheet(
            f"font-size: 25px; font-weight: bold; color: {color};"
        )

        if status != "Running":

            self.processing_fps_card.set_value("--")
            self.inference_time_card.set_value("--")

    def set_inspection_status(self, message):

        if message == "Inspection paused":

            self.set_processing_status("Paused")

        elif message == "Inspection resumed":

            self.set_processing_status("Running")

        elif message == "Ready":

            self.set_processing_status("Running")

        elif message == "Stopped":

            self.set_processing_status("Stopped")

        elif message.startswith("Inspection error:"):

            self.set_processing_status("Error")

        elif message.startswith("Inspection stopped"):

            self.set_processing_status("Stopped")

    def set_database_status(self, status):

        self.database_status_card.set_value(status)

        if status == "Connected":

            color = "#22C55E"

        elif status == "Error":

            color = "#EF4444"

        else:

            color = "#9CA3AF"

        self.database_status_card.value_label.setStyleSheet(
            f"font-size: 25px; font-weight: bold; color: {color};"
        )