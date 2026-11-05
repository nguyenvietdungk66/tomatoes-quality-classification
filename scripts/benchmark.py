
from pathlib import Path
import time
import cv2
import numpy as np

from ultralytics import YOLO
import onnxruntime as ort


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

PT_PATH = BASE_DIR / "model" / "best.pt"
ONNX_PATH = BASE_DIR / "model" / "best.onnx"
IMAGE_DIR = BASE_DIR / "test_images"

IMG_SIZE = 640
CONF = 0.25
WARMUP = 10
ITERATIONS = 100


def get_images():

    images = []

    for path in IMAGE_DIR.iterdir():

        if path.suffix.lower() in [
            ".jpg", ".jpeg", ".png", ".bmp"
        ]:

            image = cv2.imread(str(path))

            if image is not None:
                images.append((path.name, image))

    if not images:
        raise RuntimeError("No test images found!")

    return images


def compare_detections(pt_result, onnx_result):

    pt_boxes = pt_result.boxes
    onnx_boxes = onnx_result.boxes

    if len(pt_boxes) != len(onnx_boxes):
        return False

    if len(pt_boxes) == 0:
        return True

    pt_cls = pt_boxes.cls.cpu().numpy()
    onnx_cls = onnx_boxes.cls.cpu().numpy()

    if not np.array_equal(pt_cls, onnx_cls):
        return False

    pt_conf = pt_boxes.conf.cpu().numpy()
    onnx_conf = onnx_boxes.conf.cpu().numpy()

    pt_xyxy = pt_boxes.xyxy.cpu().numpy()
    onnx_xyxy = onnx_boxes.xyxy.cpu().numpy()

    confidence_diff = np.max(
        np.abs(pt_conf - onnx_conf)
    )

    bbox_diff = np.max(
        np.abs(pt_xyxy - onnx_xyxy)
    )

    return (
        confidence_diff < 0.02
        and bbox_diff < 2.0
    )


def main():

    print("Loading models...")

    pt_model = YOLO(str(PT_PATH))
    onnx_model = YOLO(str(ONNX_PATH), task="detect")

    images = get_images()

    print("Test images:", len(images))

    pt_times = []
    onnx_times = []

    match_count = 0

    for name, image in images:

        # Warmup
        for _ in range(WARMUP):

            pt_model.predict(
                image,
                imgsz=IMG_SIZE,
                conf=CONF,
                device=0,
                verbose=False
            )

            onnx_model.predict(
                image,
                imgsz=IMG_SIZE,
                conf=CONF,
                device=0,
                verbose=False
            )

        # Benchmark PyTorch
        start = time.perf_counter()

        for _ in range(ITERATIONS):

            pt_result = pt_model.predict(
                image,
                imgsz=IMG_SIZE,
                conf=CONF,
                device=0,
                verbose=False
            )[0]

        pt_latency = (
            time.perf_counter() - start
        ) / ITERATIONS * 1000

        # Benchmark ONNX CPU
        start = time.perf_counter()

        for _ in range(ITERATIONS):

            onnx_result = onnx_model.predict(
                image,
                imgsz=IMG_SIZE,
                conf=CONF,
                device=0,
                verbose=False
            )[0]

        onnx_latency = (
            time.perf_counter() - start
        ) / ITERATIONS * 1000

        pt_times.append(pt_latency)
        onnx_times.append(onnx_latency)

        matched = compare_detections(
            pt_result,
            onnx_result
        )

        match_count += int(matched)

        print(
            f"{name}: "
            f"PT={pt_latency:.2f} ms | "
            f"ONNX={onnx_latency:.2f} ms | "
            f"Match={matched}"
        )

    print("\n========== SUMMARY ==========")

    print(
        f"PT average: {np.mean(pt_times):.2f} ms"
    )

    print(
        f"ONNX CPU average: {np.mean(onnx_times):.2f} ms"
    )

    print(
        f"PT FPS: {1000 / np.mean(pt_times):.2f}"
    )

    print(
        f"ONNX CPU FPS: {1000 / np.mean(onnx_times):.2f}"
    )

    print(
        f"Matching images: {match_count}/{len(images)}"
    )


if __name__ == "__main__":
    main()
