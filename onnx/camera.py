
import cv2
import time
import numpy as np

from pathlib import Path


class CameraManager:

    def __init__(
        self,
        engine,
        processor,
        camera_index=0,
        width=1280,
        height=720,
        output_dir=None
    ):

        self.engine = engine
        self.processor = processor

        self.camera_index = camera_index
        self.width = width
        self.height = height

        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path("camera_results")
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.cap = None

        self.is_opened = False

        self.frame_count = 0
        self.fps = 0.0
        self.inference_time = 0.0

        self.previous_frame_time = None

    # ==============================================
    # 1. OPEN CAMERA
    # ==============================================

    def open(self):

        # DirectShow is suitable for Windows webcams
        self.cap = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_DSHOW
        )

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None

            raise RuntimeError(
                f"Cannot open camera index {self.camera_index}"
            )

        # Request camera resolution
        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.width
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.height
        )

        # Read actual camera resolution
        actual_width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        actual_height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        self.is_opened = True

        print("\n========== CAMERA ==========")
        print(f"Camera index: {self.camera_index}")
        print(f"Resolution: {actual_width}x{actual_height}")
        print("Camera status: Connected")
        print("============================\n")

        return True

    # ==============================================
    # 2. READ FRAME
    # ==============================================

    def read(self):

        if not self.is_opened:
            raise RuntimeError("Camera is not opened.")

        ret, frame = self.cap.read()

        if not ret or frame is None:
            return False, None

        return True, frame

    # ==============================================
    # 3. PROCESS FRAME
    # ==============================================

    def process_frame(self, frame):

        if frame is None:
            raise ValueError("Input frame is None.")

        start_time = time.perf_counter()

        # ONNX Runtime inference
        outputs, transform = self.engine.infer_raw(frame)

        # YOLO postprocessing
        detections = self.processor.process(
            outputs,
            transform
        )

        end_time = time.perf_counter()

        self.inference_time = (
            end_time - start_time
        ) * 1000

        # Draw detection results
        annotated_frame = self.draw_detections(
            frame,
            detections
        )

        return annotated_frame, detections

    # ==============================================
    # 4. DRAW DETECTIONS
    # ==============================================

    def draw_detections(self, frame, detections):

        annotated = frame.copy()

        for detection in detections:

            x1, y1, x2, y2 = detection["bbox"]

            class_id = detection["class_id"]

            class_name = detection["class_name"]

            confidence = detection["confidence"]

            # Fresh: Green
            # Rotten: Red
            color = (
                (0, 255, 0)
                if class_id == 0
                else (0, 0, 255)
            )

            # Bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Label
            label = (
                f"{class_name}: {confidence:.2f}"
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )

            # Keep label inside image
            label_y = max(text_h + 8, y1)

            # Label background
            cv2.rectangle(
                annotated,
                (x1, label_y - text_h - 8),
                (x1 + text_w + 8, label_y + baseline - 4),
                color,
                -1
            )

            # Label text
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

    # ==============================================
    # 5. UPDATE FPS
    # ==============================================

    def update_fps(self):

        current_time = time.perf_counter()

        if self.previous_frame_time is not None:

            delta = (
                current_time - self.previous_frame_time
            )

            if delta > 0:

                current_fps = 1.0 / delta

                # Smooth FPS to reduce fluctuations
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

    # ==============================================
    # 6. DRAW SYSTEM INFORMATION
    # ==============================================

    def draw_system_info(
        self,
        frame,
        detections
    ):

        total = len(detections)

        fresh_count = sum(
            d["class_id"] == 0
            for d in detections
        )

        rotten_count = sum(
            d["class_id"] == 1
            for d in detections
        )

        info = [
            f"FPS: {self.fps:.1f}",
            f"Inference: {self.inference_time:.2f} ms",
            f"Total: {total}",
            f"Fresh: {fresh_count}",
            f"Rotten: {rotten_count}"
        ]

        x = 15
        y = 30

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

    # ==============================================
    # 7. SAVE SNAPSHOT
    # ==============================================

    def save_snapshot(self, frame):

        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = (
            f"tomato_inspection_{timestamp}.jpg"
        )

        output_path = self.output_dir / filename

        success = cv2.imwrite(
            str(output_path),
            frame
        )

        if not success:
            raise RuntimeError(
                f"Failed to save snapshot: {output_path}"
            )

        print(f"[INFO] Snapshot saved: {output_path.resolve()}")

        return output_path

    # ==============================================
    # 8. RUN CAMERA DEMO
    # ==============================================

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
                    break

                # Run inference
                annotated_frame, detections = (
                    self.process_frame(frame)
                )

                # Update FPS
                self.update_fps()

                # Draw system information
                display_frame = self.draw_system_info(
                    annotated_frame,
                    detections
                )

                self.frame_count += 1

                # Display
                cv2.imshow(
                    "Tomato Quality Inspection - ONNX",
                    display_frame
                )

                key = cv2.waitKey(1) & 0xFF

                # Quit
                if key == ord("q") or key == 27:
                    break

                # Save snapshot
                elif key == ord("s"):
                    self.save_snapshot(display_frame)

        finally:
            self.release()

    # ==============================================
    # 9. RELEASE CAMERA
    # ==============================================

    def release(self):

        if self.cap is not None:

            self.cap.release()
            self.cap = None

        self.is_opened = False

        cv2.destroyAllWindows()

        print("\n========== CAMERA STOPPED ==========")
        print(f"Processed frames: {self.frame_count}")
        print(f"Final FPS: {self.fps:.2f}")
        print("Camera released.")
        print("====================================\n")
