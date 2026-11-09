
"""
Camera backends for Tomato Quality Inspection.

Supported sources:
- "webcam": built-in webcam or ordinary USB UVC camera via OpenCV.
- "hikrobot": Hikrobot industrial camera via MVS SDK.
"""

import cv2


class OpenCVCamera:
    """OpenCV backend for laptop webcams and standard USB UVC cameras."""

    def __init__(
        self,
        camera_index=0,
        width=1280,
        height=720,
        backend=cv2.CAP_MSMF
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.backend = backend
        self.cap = None

    def open(self):
        import time

        total_start = time.perf_counter()

        print("[CAMERA OPEN] Starting camera initialization...")

        # Step 1: Initialize VideoCapture
        start = time.perf_counter()

        self.cap = cv2.VideoCapture(
            self.camera_index,
            self.backend
        )

        open_elapsed = time.perf_counter() - start

        print(
            f"[CAMERA OPEN] VideoCapture: "
            f"{open_elapsed:.3f}s"
        )

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None

            raise RuntimeError(
                f"Cannot open OpenCV camera index {self.camera_index}. "
                "Check the camera index and close other apps using the camera."
            )

        # Step 2: Read camera information
        start = time.perf_counter()

        actual_width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        actual_height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        info_elapsed = time.perf_counter() - start

        try:
            backend_name = self.cap.getBackendName()
        except Exception:
            backend_name = "Unknown"

        print(
            f"[CAMERA OPEN] Read camera information: "
            f"{info_elapsed:.3f}s"
        )

        print("\n========== OPENCV CAMERA ==========")
        print(f"Camera index: {self.camera_index}")
        print(f"Resolution: {actual_width}x{actual_height}")
        print(f"Backend: {backend_name}")
        print("Camera status: Connected")
        print(
            f"Total open time: "
            f"{time.perf_counter() - total_start:.3f}s"
        )
        print("===================================\n")

        return True

    def read(self):
        if self.cap is None:
            raise RuntimeError("OpenCV camera is not opened.")

        if not self.cap.isOpened():
            return False, None

        try:
            ret, frame = self.cap.read()

        except Exception as exc:
            print(f"[OPENCV READ ERROR] {exc}")
            return False, None

        if not ret or frame is None:
            return False, None

        return True, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        print("[INFO] OpenCV camera released.")


class HikrobotSource:
    """Adapter around the existing HikrobotCamera class."""

    def __init__(self, camera_index=0, **kwargs):
        self.camera_index = camera_index
        self.camera = None

    def open(self):
        try:
            from .hikrobot_camera import HikrobotCamera

        except ImportError as exc:
            raise ImportError(
                "Hikrobot mode requires "
                "camera_setup/hikrobot_camera.py "
                "containing the HikrobotCamera class."
            ) from exc

        self.camera = HikrobotCamera(
            camera_index=self.camera_index
        )

        self.camera.open()
        self.camera.start_grabbing()

        print("[INFO] Hikrobot camera connected and grabbing.")

        return True

    def read(self):
        if self.camera is None:
            raise RuntimeError("Hikrobot camera is not opened.")

        frame = self.camera.get_frame()

        if frame is None:
            return False, None

        return True, frame

    def release(self):
        if self.camera is not None:
            self.camera.close()
            self.camera = None

        print("[INFO] Hikrobot camera released.")


def create_camera(
    source="webcam",
    camera_index=0,
    width=1280,
    height=720
):
    """
    Factory returning a camera backend.
    """

    source = source.strip().lower()

    if source in ("webcam", "usb", "opencv"):
        return OpenCVCamera(
            camera_index=camera_index,
            width=width,
            height=height,
            backend=cv2.CAP_MSMF
        )

    if source in ("hikrobot", "mvs"):
        return HikrobotSource(
            camera_index=camera_index
        )

    raise ValueError(
        f"Unsupported camera source: {source!r}. "
        "Choose 'webcam' or 'hikrobot'."
    )
