import cv2
import time
import numpy as np

from pathlib import Path
from .camera_source import create_camera


class CameraManager:
    """
    Camera + ONNX inference manager.

    camera_type:
        "webcam"   -> built-in webcam or ordinary USB UVC camera
        "hikrobot" -> Hikrobot industrial camera through MVS SDK
    """

    def __init__(
        self,
        engine,
        processor,
        camera_type="webcam",
        camera_index=0,
        width=1280,
        height=720,
        output_dir=None
    ):
        self.engine = engine
        self.processor = processor

        self.camera_type = camera_type
        self.camera_index = camera_index
        self.width = width
        self.height = height

        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path("camera_results")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.camera = None
        self.is_opened = False

        self.frame_count = 0
        self.fps = 0.0
        self.inference_time = 0.0
        self.previous_frame_time = None

    # =====================================================
    # 1. OPEN CAMERA
    # =====================================================

    def open(self):
        self.camera = create_camera(
            source=self.camera_type,
            camera_index=self.camera_index,
            width=self.width,
            height=self.height
        )
        self.camera.open()
        self.is_opened = True
        return True

    # =====================================================
    # 2. READ FRAME
    # =====================================================

    def read(self):
        if not self.is_opened or self.camera is None:
            raise RuntimeError("Camera is not opened.")

        return self.camera.read()

    # =====================================================
    # 3. PROCESS FRAME
    # =====================================================

    def process_frame(self, frame):
        if frame is None:
            raise ValueError("Input frame is None.")

        start_time = time.perf_counter()

        outputs, transform = self.engine.infer_raw(frame)
        detections = self.processor.process(outputs, transform)

        end_time = time.perf_counter()
        self.inference_time = (end_time - start_time) * 1000

        annotated_frame = self.draw_detections(frame, detections)
        return annotated_frame, detections

    # =====================================================
    # 4. DRAW DETECTIONS
    # =====================================================

    def draw_detections(self, frame, detections):
        annotated = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = map(int, detection["bbox"])
            class_id = detection["class_id"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]

            # Fresh: green; Rotten: red.
            color = (0, 255, 0) if class_id == 0 else (0, 0, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            label_y = max(text_h + 8, y1)

            cv2.rectangle(
                annotated,
                (x1, label_y - text_h - 8),
                (x1 + text_w + 8, label_y + baseline - 4),
                color,
                -1
            )

            cv2.putText(
                annotated,
                label,
                (x1 + 4, label_y - 4),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

        return annotated

    # =====================================================
    # 5. UPDATE FPS
    # =====================================================

    def update_fps(self):
        current_time = time.perf_counter()

        if self.previous_frame_time is not None:
            delta = current_time - self.previous_frame_time

            if delta > 0:
                current_fps = 1.0 / delta
                alpha = 0.1

                if self.fps == 0:
                    self.fps = current_fps
                else:
                    self.fps = (
                        alpha * current_fps
                        + (1 - alpha) * self.fps
                    )

        self.previous_frame_time = current_time
        return self.fps

    # =====================================================
    # 6. DRAW SYSTEM INFORMATION
    # =====================================================

    def draw_system_info(self, frame, detections):
        total = len(detections)
        fresh_count = sum(d["class_id"] == 0 for d in detections)
        rotten_count = sum(d["class_id"] == 1 for d in detections)

        info = [
            f"Camera: {self.camera_type}",
            f"FPS: {self.fps:.1f}",
            f"Inference: {self.inference_time:.2f} ms",
            f"Total: {total}",
            f"Fresh: {fresh_count}",
            f"Rotten: {rotten_count}"
        ]

        x, y = 15, 30
        for line in info:
            cv2.putText(
                frame,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            y += 30

        return frame

    # =====================================================
    # 7. SAVE SNAPSHOT
    # =====================================================

    def save_snapshot(self, frame):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"tomato_inspection_{timestamp}.jpg"
        output_path = self.output_dir / filename

        success = cv2.imwrite(str(output_path), frame)
        if not success:
            raise RuntimeError(f"Failed to save snapshot: {output_path}")

        print(f"[INFO] Snapshot saved: {output_path.resolve()}")
        return output_path

    # =====================================================
    # 8. RUN LIVE INFERENCE
    # =====================================================

    def run(self):
        if not self.is_opened:
            self.open()

        print("[INFO] Starting live inference...")
        print("[INFO] Press Q or ESC to quit.")
        print("[INFO] Press S to save snapshot.")

        try:
            while self.is_opened:
                ret, frame = self.read()

                if not ret:
                    print("[WARNING] Cannot read camera frame.")
                    continue

                annotated_frame, detections = self.process_frame(frame)
                self.update_fps()

                display_frame = self.draw_system_info(
                    annotated_frame,
                    detections
                )

                self.frame_count += 1

                cv2.imshow(
                    "Tomato Quality Inspection - ONNX",
                    display_frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    break
                elif key == ord("s"):
                    self.save_snapshot(display_frame)

        finally:
            self.release()

    # =====================================================
    # 9. RELEASE CAMERA
    # =====================================================

    def release(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None

        self.is_opened = False
        cv2.destroyAllWindows()

        print("\n========== CAMERA STOPPED ==========")
        print(f"Processed frames: {self.frame_count}")
        print(f"Final FPS: {self.fps:.2f}")
        print("Camera released.")
        print("====================================\n")
