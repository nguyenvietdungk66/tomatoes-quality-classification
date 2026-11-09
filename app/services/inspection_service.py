import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt

from core.onnx_engine import ONNXRuntimeEngine
from core.postprocessing import YOLOPostProcessor
from inspection.inspection_manager import InspectionManager
from inspection.temporal_classifier import TemporalClassifier


class InspectionWorker(QThread):

    frame_ready = Signal(object)
    stats_ready = Signal(float, int, int)
    status_changed = Signal(str)
    error = Signal(str)
    temporal_progress = Signal(int, int)
    result_ready = Signal(int, str, float, int, int)

    # System performance signals
    performance_ready = Signal(float, float)
    model_status_changed = Signal(str)

    def __init__(
        self,
        model_path,
        inspection_manager,
        settings,
        empty_frame_count=0,
        parent=None
    ):
        super().__init__(parent)

        self.model_path = Path(model_path)
        self.inspection_manager = inspection_manager

        self.running = False
        self.latest_frame = None

        self.frame_lock = threading.Lock()
        self.frame_event = threading.Event()
        self.reset_event = threading.Event()
        self.pause_event = threading.Event()

        self.settings_lock = threading.Lock()
        self.pending_settings = None

        self.active_settings = {
            "inspection": settings["inspection"].copy(),
            "detection": settings["detection"].copy(),
        }

        self.required_frames = settings["inspection"]["required_frames"]

        self.empty_frames_to_release = (
            settings["inspection"]["empty_frames_to_release"]
        )

        self.engine = None
        self.processor = None

        self.temporal_classifier = TemporalClassifier(
            required_frames=self.required_frames
        )

        self.empty_frame_count = max(
            0,
            int(empty_frame_count)
        )

    def update_settings(self, settings):

        with self.settings_lock:

            self.pending_settings = {
                "inspection": settings["inspection"].copy(),
                "detection": settings["detection"].copy(),
            }

    def apply_pending_settings(self):

        with self.settings_lock:

            if self.pending_settings is None:
                return

            new_settings = {
                "inspection": self.pending_settings["inspection"].copy(),
                "detection": self.pending_settings["detection"].copy(),
            }

            new_detection = new_settings["detection"]

            if new_detection != self.active_settings["detection"]:

                self.processor = YOLOPostProcessor(
                    conf_threshold=new_detection["confidence_threshold"],
                    iou_threshold=new_detection["iou_threshold"],
                    max_det=new_detection["max_detections"],
                )

                self.active_settings["detection"] = new_detection

            if (
                self.inspection_manager.state
                == self.inspection_manager.WAITING_PRODUCT
            ):

                new_inspection = new_settings["inspection"]

                self.required_frames = new_inspection["required_frames"]

                self.empty_frames_to_release = (
                    new_inspection["empty_frames_to_release"]
                )

                self.inspection_manager.required_detections = (
                    new_inspection["required_detections"]
                )

                self.temporal_classifier.required_frames = (
                    self.required_frames
                )

                self.active_settings["inspection"] = new_inspection
                self.pending_settings = None

                self.status_changed.emit(
                    "Settings applied successfully"
                )

    def submit_frame(self, frame):

        if (
            not self.running
            or self.pause_event.is_set()
            or frame is None
        ):
            return

        with self.frame_lock:

            self.latest_frame = frame.copy()

        self.frame_event.set()

    def pause(self):

        if not self.running:
            return False

        self.pause_event.set()

        with self.frame_lock:

            self.latest_frame = None
            self.frame_event.clear()

        self.performance_ready.emit(0.0, 0.0)

        self.status_changed.emit("Inspection paused")

        return True

    def resume(self):

        if not self.running:
            return False

        self.pause_event.clear()

        with self.frame_lock:

            self.latest_frame = None
            self.frame_event.clear()

        self.status_changed.emit("Inspection resumed")

        return True

    def request_reset(self):

        self.reset_event.set()
        self.frame_event.set()

    def process_control_events(self):

        if not self.reset_event.is_set():
            return

        self.reset_event.clear()

        self.inspection_manager.reset()
        self.temporal_classifier.reset()

        self.empty_frame_count = 0

        self.temporal_progress.emit(
            0,
            self.required_frames
        )

        self.status_changed.emit(
            "Inspection reset - Next ID: 1"
        )

    def process_temporal_classification(self, detections):

        current_id = self.inspection_manager.current_inspection_id

        if (
            current_id is None
            or self.inspection_manager.state
            != self.inspection_manager.INSPECTING
        ):
            return

        if self.temporal_classifier.inspection_id != current_id:

            self.temporal_classifier.start(current_id)

            self.temporal_progress.emit(
                0,
                self.required_frames
            )

        previous_count = self.temporal_classifier.frame_count

        temporal_result = self.temporal_classifier.update(
            detections
        )

        current_count = self.temporal_classifier.frame_count

        if current_count != previous_count:

            self.temporal_progress.emit(
                current_count,
                self.temporal_classifier.required_frames
            )

        if temporal_result is not None:

            inspection_id = temporal_result["inspection_id"]
            class_name = temporal_result["class_name"]

            self.result_ready.emit(
                inspection_id,
                class_name,
                temporal_result["confidence"],
                temporal_result["fresh_votes"],
                temporal_result["rotten_votes"],
            )

            if self.inspection_manager.complete_cycle():

                self.empty_frame_count = 0

                self.status_changed.emit(
                    f"Product #{inspection_id} classified: "
                    f"{class_name} - Waiting for product to leave"
                )

            else:

                self.status_changed.emit(
                    "Error: Unable to complete inspection cycle"
                )

    def update_product_departure(self, detections):

        if (
            self.inspection_manager.state
            != self.inspection_manager.WAITING_NEXT_PRODUCT
        ):
            return

        if len(detections) > 0:

            self.empty_frame_count = 0
            return

        self.empty_frame_count += 1

        if self.empty_frame_count < self.empty_frames_to_release:
            return

        if self.inspection_manager.allow_next_product():

            self.empty_frame_count = 0

            self.temporal_classifier.reset()

            self.temporal_progress.emit(
                0,
                self.required_frames
            )

            self.status_changed.emit(
                "Ready - Waiting for next product"
            )

    def run(self):

        self.running = True

        # Variables for performance measurement
        processing_window_start = time.perf_counter()
        processing_frame_count = 0

        inference_time_sum = 0.0
        inference_count = 0

        try:

            self.status_changed.emit(
                "Loading YOLO model..."
            )

            self.model_status_changed.emit("Loading")

            input_size = self.active_settings["detection"]["input_size"]

            if input_size != 640:

                raise ValueError(
                    "The current ONNX model only supports input size 640."
                )

            self.engine = ONNXRuntimeEngine(
                model_path=self.model_path,
                imgsz=input_size
            )

            detection_settings = self.active_settings["detection"]

            self.processor = YOLOPostProcessor(
                conf_threshold=detection_settings["confidence_threshold"],
                iou_threshold=detection_settings["iou_threshold"],
                max_det=detection_settings["max_detections"],
            )

            self.engine.warmup(iterations=2)

            self.model_status_changed.emit("Loaded")

            self.status_changed.emit("Ready")

            while self.running:

                self.process_control_events()

                if not self.running:
                    break

                if self.pause_event.is_set():

                    with self.frame_lock:

                        self.latest_frame = None
                        self.frame_event.clear()

                    self.frame_event.wait(timeout=0.1)
                    continue

                self.frame_event.wait(timeout=0.1)

                if not self.running:
                    break

                if self.pause_event.is_set():

                    with self.frame_lock:

                        self.latest_frame = None
                        self.frame_event.clear()

                    continue

                self.process_control_events()

                if not self.running:
                    break

                with self.frame_lock:

                    frame = self.latest_frame

                    self.latest_frame = None
                    self.frame_event.clear()

                if frame is None:
                    continue

                self.apply_pending_settings()

                # ======================================
                # ONNX INFERENCE TIME
                # ======================================

                start_time = time.perf_counter()

                inference_start = time.perf_counter()

                outputs, transform = self.engine.infer_raw(frame)

                inference_ms = (
                    time.perf_counter() - inference_start
                ) * 1000.0

                inference_time_sum += inference_ms
                inference_count += 1

                # ======================================
                # POSTPROCESSING AND INSPECTION
                # ======================================

                detections = self.processor.process(
                    outputs,
                    transform
                )

                new_inspection_id = (
                    self.inspection_manager.update_detection(
                        detections
                    )
                )

                if new_inspection_id is not None:

                    self.temporal_classifier.start(
                        new_inspection_id
                    )

                    self.empty_frame_count = 0

                    self.temporal_progress.emit(
                        0,
                        self.required_frames
                    )

                    self.status_changed.emit(
                        f"Inspecting Product #{new_inspection_id}"
                    )

                self.process_temporal_classification(
                    detections
                )

                self.update_product_departure(
                    detections
                )

                current_id = (
                    self.inspection_manager.current_inspection_id
                )

                fresh_count = sum(
                    d["class_id"] == 0
                    for d in detections
                )

                rotten_count = sum(
                    d["class_id"] == 1
                    for d in detections
                )

                annotated = self.draw_detections(
                    frame,
                    detections,
                    current_id
                )

                # Keep the original Live Inspection FPS behavior.
                elapsed = time.perf_counter() - start_time

                fps = 1.0 / max(elapsed, 1e-6)

                self.frame_ready.emit(annotated)

                self.stats_ready.emit(
                    fps,
                    fresh_count,
                    rotten_count
                )

                # ======================================
                # PROCESSING FPS
                # ======================================

                processing_frame_count += 1

                now = time.perf_counter()

                window_elapsed = (
                    now - processing_window_start
                )

                if window_elapsed >= 1.0:

                    processing_fps = (
                        processing_frame_count / window_elapsed
                    )

                    average_inference_ms = (
                        inference_time_sum / inference_count
                        if inference_count > 0
                        else 0.0
                    )

                    self.performance_ready.emit(
                        processing_fps,
                        average_inference_ms
                    )

                    processing_frame_count = 0
                    inference_time_sum = 0.0
                    inference_count = 0

                    processing_window_start = now

        except Exception as exc:

            self.model_status_changed.emit("Error")

            self.error.emit(str(exc))

            self.status_changed.emit(
                f"Inspection error: {exc}"
            )

        finally:

            self.running = False

            self.performance_ready.emit(0.0, 0.0)

            if self.isInterruptionRequested():
                self.status_changed.emit("Stopped")

            self.model_status_changed.emit("Not loaded")

    @staticmethod
    def draw_detections(frame, detections, inspection_id=None):

        image = frame.copy()

        show_id = (
            inspection_id is not None
            and len(detections) == 1
        )

        for detection in detections:

            class_id = detection["class_id"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]

            x1, y1, x2, y2 = detection["bbox"]

            color = (
                (0, 255, 0)
                if class_id == 0
                else (0, 0, 255)
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = (
                f"ID: {inspection_id} | {class_name}: {confidence:.2f}"
                if show_id
                else f"{class_name}: {confidence:.2f}"
            )

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return image

    def stop(self):

        self.running = False
        self.frame_event.set()


class InspectionService(QObject):

    frame_ready = Signal(object)
    stats_ready = Signal(float, int, int)
    status_changed = Signal(str)
    error = Signal(str)
    temporal_progress = Signal(int, int)
    result_ready = Signal(int, str, float, int, int)
    totals_ready = Signal(int, int, int)

    # System performance signals
    performance_ready = Signal(float, float)
    model_status_changed = Signal(str)

    def __init__(
        self,
        model_path,
        settings_manager,
        repository=None,
        parent=None
    ):

        super().__init__(parent)

        self.model_path = Path(model_path)
        self.settings_manager = settings_manager
        self.repository = repository

        self.worker = None

        settings = self.settings_manager.get_settings()

        self.inspection_manager = InspectionManager(
            required_detections=settings["inspection"]["required_detections"]
        )

        self.total_inspected = 0
        self.total_fresh = 0
        self.total_rotten = 0

        self.session_results = []
        self.session_start_time = None
        self.session_name = None
        self.active_session_id = None

        self._resume_pending = False
        self._restored_empty_frame_count = 0

        self.result_lock = threading.Lock()

    def start_inspection(self, session_name=None):

        if self.worker is not None:

            if self.worker.isRunning():
                return False

            self.worker = None

        if not self.model_path.exists():

            self.model_status_changed.emit("Error")

            self.error.emit(
                f"YOLO model not found: {self.model_path}"
            )

            return False

        settings = self.settings_manager.get_settings()

        if not self._resume_pending:

            self.inspection_manager.reset()

            self.total_inspected = 0
            self.total_fresh = 0
            self.total_rotten = 0

            self.session_results = []

            self.session_start_time = datetime.now().astimezone()

            self.session_name = (
                session_name or "Unnamed session"
            ).strip()

            self.active_session_id = None

            self._restored_empty_frame_count = 0

        elif session_name:

            self.session_name = session_name

        self.performance_ready.emit(0.0, 0.0)
        self.model_status_changed.emit("Loading")

        self.worker = InspectionWorker(
            model_path=self.model_path,
            inspection_manager=self.inspection_manager,
            settings=settings,
            empty_frame_count=self._restored_empty_frame_count,
        )

        self.worker.frame_ready.connect(
            self.frame_ready.emit
        )

        self.worker.stats_ready.connect(
            self.stats_ready.emit
        )

        self.worker.status_changed.connect(
            self.status_changed.emit
        )

        self.worker.error.connect(
            self.error.emit
        )

        self.worker.temporal_progress.connect(
            self.temporal_progress.emit
        )

        self.worker.performance_ready.connect(
            self.performance_ready.emit
        )

        self.worker.model_status_changed.connect(
            self.model_status_changed.emit
        )

        self.worker.result_ready.connect(
            self._handle_result,
            Qt.ConnectionType.DirectConnection
        )

        self._resume_pending = False

        self.worker.start()

        self.totals_ready.emit(
            self.total_inspected,
            self.total_fresh,
            self.total_rotten
        )

        return True

    @Slot(int, str, float, int, int)
    def _handle_result(
        self,
        inspection_id,
        class_name,
        confidence,
        fresh_votes,
        rotten_votes
    ):

        timestamp = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        with self.result_lock:

            self.total_inspected += 1

            if class_name == "Fresh Tomato":
                self.total_fresh += 1

            elif class_name == "Rotten Tomato":
                self.total_rotten += 1

            self.session_results.append({
                "inspection_id": inspection_id,
                "class_name": class_name,
                "confidence": confidence,
                "timestamp": timestamp,
            })

            totals = (
                self.total_inspected,
                self.total_fresh,
                self.total_rotten
            )

        self.result_ready.emit(
            inspection_id,
            class_name,
            confidence,
            fresh_votes,
            rotten_votes
        )

        self.totals_ready.emit(*totals)

    def submit_frame(self, frame):

        if self.worker is not None:
            self.worker.submit_frame(frame)

    def pause_inspection(self):

        if (
            self.worker is None
            or not self.worker.isRunning()
        ):

            self.status_changed.emit(
                "Cannot pause: inspection is not running"
            )

            return False

        return self.worker.pause()

    def resume_inspection(self):

        if (
            self.worker is None
            or not self.worker.isRunning()
        ):

            self.status_changed.emit(
                "Cannot resume: inspection is not running"
            )

            return False

        return self.worker.resume()

    def reset_inspection(self):

        with self.result_lock:

            self.total_inspected = 0
            self.total_fresh = 0
            self.total_rotten = 0

            self.session_results = []

            self.session_start_time = datetime.now().astimezone()

        self.totals_ready.emit(0, 0, 0)

        if (
            self.worker is not None
            and self.worker.isRunning()
        ):

            self.worker.request_reset()

        else:

            self.inspection_manager.reset()

            self.status_changed.emit(
                "Inspection reset - Next ID: 1"
            )

    def _build_checkpoint(self):

        worker = self.worker

        temporal_state = (
            worker.temporal_classifier.state_dict()
            if worker is not None
            else None
        )

        return {
            "version": 1,

            "saved_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),

            "inspection_manager": (
                self.inspection_manager.state_dict()
            ),

            "temporal_classifier": temporal_state,

            "empty_frame_count": (
                worker.empty_frame_count
                if worker is not None
                else self._restored_empty_frame_count
            ),

            "settings": self.settings_manager.get_settings(),
        }

    def stop_inspection(self, save_results=False):

        if (
            self.worker is not None
            and self.worker.isRunning()
        ):

            self.worker.stop()
            self.worker.wait(10000)

            if self.worker.isRunning():

                self.error.emit(
                    "Inspection thread is still stopping."
                )

                return None

        self.performance_ready.emit(0.0, 0.0)
        self.model_status_changed.emit("Not loaded")

        if not save_results:

            self.worker = None

            self._clear_active_session()

            self.status_changed.emit(
                "Inspection stopped. Unsaved session discarded."
            )

            return None

        if self.repository is None:

            self.error.emit(
                "Database repository is not configured."
            )

            return None

        if not self.session_name:

            self.error.emit(
                "Session name is missing."
            )

            return None

        with self.result_lock:

            start_time = (
                self.session_start_time
                or datetime.now().astimezone()
            )

            end_time = datetime.now().astimezone()

            results = list(self.session_results)

            totals = (
                self.total_inspected,
                self.total_fresh,
                self.total_rotten
            )

        checkpoint = self._build_checkpoint()

        try:

            session_id = self.repository.save_session(
                session_id=self.active_session_id,
                session_name=self.session_name,
                start_time=start_time.isoformat(
                    timespec="seconds"
                ),
                end_time=end_time.isoformat(
                    timespec="seconds"
                ),
                total_inspected=totals[0],
                total_fresh=totals[1],
                total_rotten=totals[2],
                results=results,
                checkpoint=checkpoint,
                status="PAUSED",
            )

            self.active_session_id = session_id
            self.worker = None

            self._restored_empty_frame_count = (
                checkpoint.get("empty_frame_count", 0)
            )

            self.status_changed.emit(
                f"Session '{self.session_name}' saved successfully."
            )

            return session_id

        except Exception as exc:

            self.worker = None

            self.error.emit(
                f"Failed to save inspection session: {exc}"
            )

            return None

    def resume_session(self, session_id):

        if (
            self.worker is not None
            and self.worker.isRunning()
        ):

            self.error.emit(
                "Stop the current inspection before resuming another session."
            )

            return False

        if self.repository is None:

            self.error.emit(
                "Database repository is not configured."
            )

            return False

        session = self.repository.get_session(session_id)

        if session is None:

            self.error.emit(
                "The selected session no longer exists."
            )

            return False

        checkpoint = session.get("checkpoint") or {}

        manager_state = checkpoint.get("inspection_manager")

        try:

            if manager_state:

                self.inspection_manager.load_state_dict(
                    manager_state
                )

            else:

                self.inspection_manager.reset()

                # Legacy sessions have no checkpoint.
                # Continue IDs after saved results.
                self.inspection_manager.next_inspection_id = max(
                    [
                        r["inspection_id"]
                        for r in self.repository.get_session_results(
                            session_id
                        )
                    ] + [0]
                ) + 1

            self.inspection_manager.prepare_for_resume()

            # Empty-frame streaks are tied to the previous camera run.
            self._restored_empty_frame_count = 0

            with self.result_lock:

                self.total_inspected = int(
                    session.get("total_inspected", 0)
                )

                self.total_fresh = int(
                    session.get("total_fresh", 0)
                )

                self.total_rotten = int(
                    session.get("total_rotten", 0)
                )

                self.session_results = (
                    self.repository.get_session_results(session_id)
                )

                self.session_start_time = datetime.fromisoformat(
                    session["start_time"]
                )

                self.session_name = (
                    session.get("session_name")
                    or "Unnamed session"
                )

                self.active_session_id = int(session_id)

            # Saved temporal votes are not replayed against a new tomato.
            self._resume_pending = True

            started = self.start_inspection()

            if started:

                self.status_changed.emit(
                    f"Resumed session: {self.session_name}"
                )

            return started

        except Exception as exc:

            self._resume_pending = False

            self.error.emit(
                f"Failed to resume session: {exc}"
            )

            return False

    def _clear_active_session(self):

        self.inspection_manager.reset()

        with self.result_lock:

            self.total_inspected = 0
            self.total_fresh = 0
            self.total_rotten = 0

            self.session_results = []
            self.session_start_time = None

        self.session_name = None
        self.active_session_id = None

        self._resume_pending = False
        self._restored_empty_frame_count = 0

        self.totals_ready.emit(0, 0, 0)

    def apply_settings(self, settings):

        if settings["detection"]["input_size"] != 640:

            self.error.emit(
                "The current ONNX model only supports input size 640. "
                "Please restore Input Image Size to 640."
            )

            return False

        if (
            self.worker is not None
            and self.worker.isRunning()
        ):

            self.worker.update_settings(settings)

        else:

            self.inspection_manager.required_detections = (
                settings["inspection"]["required_detections"]
            )

        return True