import re
import time
import cv2
import threading

from collections import deque

from PySide6.QtCore import QObject, QThread, Signal

from camera_setup.camera_source import create_camera


class CameraWorker(QThread):

    frame_ready = Signal(object)
    connected = Signal(str, str, str)
    prepared = Signal(str, str, str)
    error = Signal(str)
    disconnected = Signal()
    fps_ready = Signal(float)
    connection_lost = Signal(str)

    STARTUP_FRAME_TIMEOUT = 10.0
    READ_FAILURE_TIMEOUT = 1.5
    WATCHDOG_INTERVAL = 0.05

    READ_FAILURE_LIMIT = 10
    FAILURE_WINDOW_SECONDS = 2.0
    READ_FAILURE_INTERVAL = 0.05

    def __init__(self, source, camera_index=0, activated=False):
        super().__init__()

        self.source = source
        self.camera_index = camera_index

        self.camera = None
        self.running = True

        # False: camera is only being prewarmed.
        # True: user has pressed Connect.
        self.activated_event = threading.Event()

        if activated:
            self.activated_event.set()

        self.prepared_emitted = False
        self.connected_emitted = False

        self.model = source
        self.serial = "--"
        self.resolution = "--"

        self.pause_event = threading.Event()

        # Watchdog controls
        self.watchdog_stop = threading.Event()
        self.manual_stop = threading.Event()
        self.connection_lost_reported = threading.Event()

        self.last_frame_time = None
        self.watchdog_thread = None

        self.first_frame_received = threading.Event()
        self.camera_open_time = None

        self.loss_lock = threading.Lock()
        self.read_failure_times = deque()

    # ==========================================
    # ACTIVATE PREWARMED CAMERA
    # ==========================================

    def activate(self):
        self.activated_event.set()

    # ==========================================
    # CONNECTION WATCHDOG
    # ==========================================

    def start_watchdog(self):

        self.last_frame_time = None
        self.camera_open_time = time.monotonic()

        self.first_frame_received.clear()
        self.watchdog_stop.clear()

        self.watchdog_thread = threading.Thread(
            target=self.watch_connection,
            daemon=True
        )

        self.watchdog_thread.start()

    def watch_connection(self):

        print("[WATCHDOG] Started.")

        last_debug_time = time.monotonic()

        while not self.watchdog_stop.wait(self.WATCHDOG_INTERVAL):

            if self.manual_stop.is_set():
                print("[WATCHDOG] Manual stop detected.")
                return

            if not self.running:
                return

            if self.connection_lost_reported.is_set():
                return

            now = time.monotonic()

            # Wait for first valid frame.
            if not self.first_frame_received.is_set():

                elapsed = now - self.camera_open_time

                if now - last_debug_time >= 0.5:

                    print(
                        f"[WATCHDOG] Waiting for first frame: "
                        f"{elapsed:.2f}s | Running: {self.running}"
                    )

                    last_debug_time = now

                if elapsed >= self.STARTUP_FRAME_TIMEOUT:

                    reason = (
                        "Camera initialization failed: "
                        "no valid frame received within "
                        f"{self.STARTUP_FRAME_TIMEOUT:.1f} seconds."
                    )

                    print("[WATCHDOG] FIRST FRAME TIMEOUT!")

                    self.report_connection_lost(reason)
                    return

                continue

            # Normal connection monitoring.
            if self.last_frame_time is None:
                continue

            elapsed = now - self.last_frame_time

            if now - last_debug_time >= 0.5:

                print(
                    f"[WATCHDOG] No-frame time: {elapsed:.2f}s | "
                    f"Running: {self.running}"
                )

                last_debug_time = now

            if elapsed >= self.READ_FAILURE_TIMEOUT:

                reason = (
                    "Camera connection lost: "
                    f"no valid frame received for {elapsed:.1f} seconds."
                )

                print("[WATCHDOG] CONNECTION LOST DETECTED!")

                self.report_connection_lost(reason)
                return

    # ==========================================
    # REPORT CONNECTION LOST
    # ==========================================

    def report_connection_lost(self, reason):

        with self.loss_lock:

            if self.manual_stop.is_set():
                return

            if self.connection_lost_reported.is_set():
                return

            self.connection_lost_reported.set()
            self.running = False

        print(
            "[CAMERA WORKER] Emitting connection_lost:",
            reason
        )

        self.connection_lost.emit(reason)

    # ==========================================
    # READ FAILURE TRACKING
    # ==========================================

    def register_read_failure(self):

        now = time.monotonic()
        self.read_failure_times.append(now)

        while self.read_failure_times:

            oldest = self.read_failure_times[0]

            if now - oldest > self.FAILURE_WINDOW_SECONDS:
                self.read_failure_times.popleft()
            else:
                break

        failure_count = len(self.read_failure_times)

        print(
            f"[CAMERA READ] FAILED | "
            f"Recent failures: {failure_count}/"
            f"{self.READ_FAILURE_LIMIT} | "
            f"Window: {self.FAILURE_WINDOW_SECONDS:.1f}s"
        )

        if failure_count >= self.READ_FAILURE_LIMIT:

            reason = (
                "Camera connection lost: "
                f"{failure_count} frame read failures "
                f"within {self.FAILURE_WINDOW_SECONDS:.1f} seconds."
            )

            print("[CAMERA WORKER] Camera read failure limit reached.")

            self.report_connection_lost(reason)
            return True

        return False

    def clear_old_read_failures(self):

        now = time.monotonic()

        while self.read_failure_times:

            oldest = self.read_failure_times[0]

            if now - oldest > self.FAILURE_WINDOW_SECONDS:
                self.read_failure_times.popleft()
            else:
                break

    # ==========================================
    # EMIT CAMERA STATE
    # ==========================================

    def update_camera_state(self, frame):

        height, width = frame.shape[:2]

        self.resolution = f"{width} x {height}"

        # Camera is ready, but not yet connected in the UI.
        if not self.prepared_emitted:

            self.prepared_emitted = True

            print("[CAMERA WORKER] Camera is prepared and ready.")

            self.prepared.emit(
                self.model,
                self.serial,
                self.resolution
            )

        # Only emit Connected after the user presses Connect.
        if (
            self.activated_event.is_set()
            and not self.connected_emitted
        ):

            self.connected_emitted = True

            print("[CAMERA WORKER] Camera activated by user.")

            self.connected.emit(
                self.model,
                self.serial,
                self.resolution
            )

    # ==========================================
    # CAMERA WORKER
    # ==========================================

    def run(self):

        try:

            self.camera = create_camera(
                source=self.source,
                camera_index=self.camera_index,
                width=1280,
                height=720
            )

            open_start = time.perf_counter()

            self.camera.open()

            self.camera_open_time = time.monotonic()

            print(
                f"[CAMERA WORKER] Camera open completed in "
                f"{time.perf_counter() - open_start:.3f}s"
            )

            if self.source == "hikrobot":

                backend = self.camera.camera

                device_info = (
                    backend.device_list
                    .pDeviceInfo[self.camera_index]
                    .contents
                )

                try:
                    self.model = self.decode_string(
                        device_info.SpecialInfo.stGigEInfo.chModelName
                    )

                    self.serial = self.decode_string(
                        device_info.SpecialInfo.stGigEInfo.chSerialNumber
                    )

                except Exception:
                    self.model = "Hikrobot Camera"
                    self.serial = "--"

            else:
                self.model = "USB Camera"
                self.serial = "--"

            self.start_watchdog()

            fps_window_start = time.perf_counter()
            fps_frame_count = 0

            while self.running:

                try:
                    ret, frame = self.camera.read()

                except Exception as exc:

                    ret = False
                    frame = None

                    print(f"[CAMERA READ] Exception: {exc}")

                if self.connection_lost_reported.is_set():
                    break

                # Handle failed frame.
                if not ret or frame is None:

                    if not self.first_frame_received.is_set():
                        time.sleep(self.READ_FAILURE_INTERVAL)
                        continue

                    connection_lost = self.register_read_failure()

                    if connection_lost:
                        break

                    time.sleep(self.READ_FAILURE_INTERVAL)
                    continue

                # Successful frame.
                self.clear_old_read_failures()

                self.last_frame_time = time.monotonic()

                if not self.first_frame_received.is_set():

                    self.first_frame_received.set()

                    startup_elapsed = (
                        time.monotonic() - self.camera_open_time
                    )

                    print(
                        f"[CAMERA WORKER] First valid frame received "
                        f"after {startup_elapsed:.3f}s."
                    )

                # Update prepared / connected state.
                self.update_camera_state(frame)

                # FPS measurement.
                fps_frame_count += 1

                now = time.perf_counter()
                window_elapsed = now - fps_window_start

                if window_elapsed >= 1.0:

                    camera_fps = fps_frame_count / window_elapsed

                    self.fps_ready.emit(camera_fps)

                    fps_frame_count = 0
                    fps_window_start = now

                # During prewarming, keep reading frames but do not
                # forward them to the preview or inspection service.
                if not self.activated_event.is_set():
                    continue

                # Pause forwarding frames, but keep reading camera.
                if self.pause_event.is_set():
                    continue

                self.frame_ready.emit(frame)

        except Exception as exc:

            print("[CAMERA WORKER] ERROR:", exc)
            self.error.emit(str(exc))

        finally:

            print(
                f"[CAMERA WORKER] Entering finally | "
                f"running={self.running} | "
                f"manual_stop={self.manual_stop.is_set()} | "
                f"connection_lost="
                f"{self.connection_lost_reported.is_set()}"
            )

            self.watchdog_stop.set()

            if self.watchdog_thread is not None:

                if threading.current_thread() != self.watchdog_thread:

                    self.watchdog_thread.join(timeout=0.5)

            if self.camera is not None:

                try:
                    self.camera.release()

                except Exception as exc:
                    self.error.emit(f"Camera release error: {exc}")

                self.camera = None

            self.fps_ready.emit(0.0)
            self.disconnected.emit()

    # ==========================================
    # PAUSE / RESUME
    # ==========================================

    def pause(self):
        self.pause_event.set()

    def resume(self):
        self.pause_event.clear()

    # ==========================================
    # UTILITIES
    # ==========================================

    @staticmethod
    def decode_string(value):

        return bytes(value).split(
            b"\0"
        )[0].decode(errors="ignore")

    # ==========================================
    # STOP
    # ==========================================

    def stop(self):

        with self.loss_lock:

            self.manual_stop.set()
            self.watchdog_stop.set()
            self.running = False


class CameraService(QObject):

    frame_ready = Signal(object)
    status_changed = Signal(str, bool)
    device_info_changed = Signal(str, str, str)
    devices_found = Signal(list)
    error = Signal(str)

    camera_fps_changed = Signal(float)
    connection_lost = Signal(str)

    def __init__(self, parent=None):

        super().__init__(parent)

        self.worker = None
        self._connected = False

        self._prepared_source = None
        self._prepared_index = None

    # ==========================================
    # CREATE AND CONNECT WORKER SIGNALS
    # ==========================================

    def create_worker(self, source, camera_index, activated=False):

        worker = CameraWorker(
            source=source,
            camera_index=camera_index,
            activated=activated
        )

        worker.frame_ready.connect(self.frame_ready.emit)
        worker.connected.connect(self.on_connected)
        worker.prepared.connect(self.on_prepared)
        worker.error.connect(self.on_error)
        worker.disconnected.connect(self.on_disconnected)
        worker.fps_ready.connect(self.camera_fps_changed.emit)
        worker.connection_lost.connect(self.on_connection_lost)

        self.worker = worker

        self._prepared_source = source
        self._prepared_index = camera_index

        self._connected = False

        self.camera_fps_changed.emit(0.0)

        worker.start()

        return worker

    # ==========================================
    # STOP CURRENT WORKER
    # ==========================================

    def stop_current_worker(self):

        if self.worker is None:
            return True

        worker = self.worker

        if worker.isRunning():

            worker.stop()
            worker.wait(3000)

            if worker.isRunning():

                self.error.emit(
                    "Camera thread is still stopping. Please wait."
                )

                return False

        self.worker = None

        self._connected = False
        self._prepared_source = None
        self._prepared_index = None

        return True

    # ==========================================
    # PREPARE CAMERA IN BACKGROUND
    # ==========================================

    def prepare_camera(self, camera_type):

        source = (
            "hikrobot"
            if camera_type == "Hikrobot"
            else "webcam"
        )

        camera_index = 0

        # Reuse an existing worker if it is already preparing
        # or has prepared the same camera.
        if (
            self.worker is not None
            and self.worker.isRunning()
            and not self.worker.connection_lost_reported.is_set()
            and self._prepared_source == source
            and self._prepared_index == camera_index
        ):

            print("[CAMERA SERVICE] Camera is already being prepared.")

            return

        # Do not replace a camera currently connected by the user.
        if self.is_connected():
            return

        if not self.stop_current_worker():
            return

        device_label = (
            "0 - Hikrobot Camera"
            if source == "hikrobot"
            else "0 - USB Camera"
        )

        # Provide a default selection so the user can press Connect
        # without having to scan first.
        self.devices_found.emit([device_label])

        self.status_changed.emit(
            "Preparing camera...",
            False
        )

        print(
            f"[CAMERA SERVICE] Prewarming {source}, "
            f"camera index {camera_index}..."
        )

        self.create_worker(
            source=source,
            camera_index=camera_index,
            activated=False
        )

    # ==========================================
    # SCAN CAMERA DEVICES
    # ==========================================

    def scan_devices(self, camera_type):

        devices = []

        try:

            if camera_type == "Hikrobot":

                from camera_setup.hikrobot_camera import HikrobotCamera

                scanner = HikrobotCamera()
                count = scanner.enumerate_devices()

                for index in range(count):

                    info = (
                        scanner.device_list
                        .pDeviceInfo[index]
                        .contents
                    )

                    model = CameraWorker.decode_string(
                        info.SpecialInfo.stGigEInfo.chModelName
                    )

                    serial = CameraWorker.decode_string(
                        info.SpecialInfo.stGigEInfo.chSerialNumber
                    )

                    devices.append(
                        f"{index} - {model} (SN: {serial})"
                    )

            else:

                for index in range(5):

                    cap = cv2.VideoCapture(
                        index,
                        cv2.CAP_DSHOW
                    )

                    if cap.isOpened():
                        devices.append(f"{index} - USB Camera")

                    cap.release()

            self.devices_found.emit(devices)

            if not devices:
                self.error.emit("No camera devices found.")

        except Exception as exc:
            self.error.emit(str(exc))

    # ==========================================
    # CONNECT CAMERA
    # ==========================================

    def connect_camera(self, camera_type, device_name):

        try:

            match = re.match(r"^\s*(\d+)", device_name)

            if match is None:
                raise ValueError("Invalid camera device selection.")

            camera_index = int(match.group(1))

            source = (
                "hikrobot"
                if camera_type == "Hikrobot"
                else "webcam"
            )

            # If the selected camera is already being prewarmed,
            # activate the existing worker instead of opening again.
            if (
                self.worker is not None
                and self.worker.isRunning()
                and not self.worker.connection_lost_reported.is_set()
                and self._prepared_source == source
                and self._prepared_index == camera_index
            ):

                print(
                    "[CAMERA SERVICE] Reusing prewarmed camera."
                )

                self.status_changed.emit(
                    "Connecting...",
                    False
                )

                self.worker.activate()

                return

            # If a different camera is selected, replace the
            # prewarmed worker with the selected device.
            if not self.stop_current_worker():
                return

            self.status_changed.emit(
                "Connecting...",
                False
            )

            self.create_worker(
                source=source,
                camera_index=camera_index,
                activated=True
            )

        except Exception as exc:
            self.error.emit(str(exc))

    # ==========================================
    # WORKER EVENTS
    # ==========================================

    def on_prepared(self, model, serial, resolution):

        if (
            self.worker is None
            or not self.worker.isRunning()
            or self.worker.connection_lost_reported.is_set()
            or self.worker.manual_stop.is_set()
        ):
            return

        # Do not mark the camera as connected here.
        if not self._connected:

            print("[CAMERA SERVICE] Camera ready for Connect.")

            self.status_changed.emit(
                "Ready to connect",
                False
            )

    def on_connected(self, model, serial, resolution):

        if (
            self.worker is None
            or not self.worker.isRunning()
            or self.worker.connection_lost_reported.is_set()
            or self.worker.manual_stop.is_set()
        ):
            return

        self._connected = True

        self.device_info_changed.emit(
            model,
            serial,
            resolution
        )

        self.status_changed.emit(
            "Connected",
            True
        )

    def on_error(self, message):

        self.error.emit(message)

        if not self._connected:

            self.status_changed.emit(
                "Connection error",
                False
            )

    def on_connection_lost(self, reason):

        print(
            "[CAMERA SERVICE] Received connection_lost:",
            reason
        )

        self._connected = False
        self.camera_fps_changed.emit(0.0)

        # Camera lỗi trong lúc prewarming, chưa được người dùng Connect.
        if (
            self.worker is not None
            and not self.worker.activated_event.is_set()
        ):

            self.status_changed.emit(
                "Camera preparation failed",
                False
            )

            self.error.emit(reason)
            return

        # Camera đã được Connect: giữ nguyên luồng xử lý mất kết nối.
        self.status_changed.emit(
            "Connection lost",
            False
        )

        self.connection_lost.emit(reason)

    def on_disconnected(self):

        self._connected = False

        self.camera_fps_changed.emit(0.0)

        self.status_changed.emit(
            "Disconnected",
            False
        )

    # ==========================================
    # DISCONNECT CAMERA
    # ==========================================

    def disconnect_camera(self):

        if self.worker is None:

            self._connected = False
            return

        if not self.stop_current_worker():
            return

        self.camera_fps_changed.emit(0.0)

        self.status_changed.emit(
            "Disconnected",
            False
        )

    # ==========================================
    # PAUSE / RESUME CAMERA
    # ==========================================

    def pause_camera(self):

        if (
            self.worker is not None
            and self.worker.isRunning()
            and self._connected
        ):

            self.worker.pause()

            self.status_changed.emit(
                "Paused",
                True
            )

    def resume_camera(self):

        if (
            self.worker is not None
            and self.worker.isRunning()
            and self._connected
        ):

            self.worker.resume()

            self.status_changed.emit(
                "Connected",
                True
            )

    # ==========================================
    # CONNECTION STATUS
    # ==========================================

    def is_connected(self):

        return (
            self._connected
            and self.worker is not None
            and self.worker.isRunning()
            and not self.worker.connection_lost_reported.is_set()
        )