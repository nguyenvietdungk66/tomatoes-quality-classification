from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QTableWidget, QTableWidgetItem,
    QGroupBox, QHeaderView,
)


class HistoryPage(QWidget):
    resume_requested = Signal(int)
    delete_requested = Signal(int)

    def __init__(self, repository=None):
        super().__init__()
        self.repository = repository
        self.selected_session_id = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(14)

        title = QLabel("Inspection History")
        title.setObjectName("PageTitle")
        description = QLabel("Select a saved session to view its inspection results.")
        root.addWidget(title)
        root.addWidget(description)

        self.session_list = QListWidget()
        self.session_list.setMinimumHeight(180)
        self.session_list.currentItemChanged.connect(self._on_session_selected)
        root.addWidget(self.session_list, stretch=2)

        details_group = QGroupBox("Session Details")
        details_layout = QVBoxLayout(details_group)
        self.session_summary = QLabel("No session selected")
        self.session_summary.setWordWrap(True)
        details_layout.addWidget(self.session_summary)

        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(
            ["Product ID", "Evaluation", "Confidence", "Time"]
        )
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        details_layout.addWidget(self.results_table)
        root.addWidget(details_group, stretch=3)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.resume_button = QPushButton("Resume Session")
        self.delete_button = QPushButton("Delete Session")
        self.refresh_button = QPushButton("Refresh")
        self.resume_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        self.resume_button.clicked.connect(self._emit_resume)
        self.delete_button.clicked.connect(self._emit_delete)
        self.refresh_button.clicked.connect(self.refresh_sessions)
        buttons.addWidget(self.refresh_button)
        buttons.addWidget(self.resume_button)
        buttons.addWidget(self.delete_button)
        root.addLayout(buttons)

    def set_repository(self, repository):
        self.repository = repository
        self.refresh_sessions()

    @staticmethod
    def _format_time(value):
        if not value:
            return "—"
        try:
            return datetime.fromisoformat(value).astimezone().strftime("%d/%m/%Y %H:%M:%S")
        except (TypeError, ValueError):
            return str(value)

    def refresh_sessions(self):
        self.session_list.clear()
        self._clear_details()
        if self.repository is None:
            return
        try:
            sessions = self.repository.get_sessions()
            for session in sessions:
                name = session.get("session_name") or "Unnamed session"
                start = self._format_time(session.get("start_time"))
                end = self._format_time(session.get("end_time"))
                item = QListWidgetItem(f"{name}\n{start}  —  {end}")
                item.setData(Qt.ItemDataRole.UserRole, int(session["session_id"]))
                item.setToolTip(f"{name}\n{start} — {end}")
                self.session_list.addItem(item)
        except Exception as exc:
            self.session_summary.setText(f"Unable to load history: {exc}")

    def _clear_details(self):
        self.selected_session_id = None
        self.session_summary.setText("No session selected")
        self.results_table.setRowCount(0)
        self.resume_button.setEnabled(False)
        self.delete_button.setEnabled(False)

    def _on_session_selected(self, current, previous):
        if current is None:
            self._clear_details()
            return
        session_id = int(current.data(Qt.ItemDataRole.UserRole))
        self.selected_session_id = session_id
        self.resume_button.setEnabled(True)
        self.delete_button.setEnabled(True)
        session = self.repository.get_session(session_id)
        results = self.repository.get_session_results(session_id)
        if session is None:
            self._clear_details()
            return

        name = session.get("session_name") or "Unnamed session"
        start = self._format_time(session.get("start_time"))
        end = self._format_time(session.get("end_time"))
        status = session.get("status", "PAUSED")
        self.session_summary.setText(
            f"<b>{name}</b><br>Start: {start}<br>Last saved: {end}<br>"
            f"Status: {status}<br><br>"
            f"Total inspected: {session.get('total_inspected', 0)}   |   "
            f"Fresh: {session.get('total_fresh', 0)}   |   "
            f"Rotten: {session.get('total_rotten', 0)}"
        )

        self.results_table.setRowCount(len(results))
        for row, result in enumerate(results):
            values = [
                str(result["inspection_id"]),
                result["class_name"],
                f"{float(result['confidence']):.3f}",
                self._format_time(result["timestamp"]),
            ]
            for column, value in enumerate(values):
                self.results_table.setItem(row, column, QTableWidgetItem(value))

    def _emit_resume(self):
        if self.selected_session_id is not None:
            self.resume_requested.emit(self.selected_session_id)

    def _emit_delete(self):
        if self.selected_session_id is not None:
            self.delete_requested.emit(self.selected_session_id)
