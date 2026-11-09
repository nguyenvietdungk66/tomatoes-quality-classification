import cv2
from pathlib import Path

from onnx_engine import ONNXRuntimeEngine
from postprocessing import YOLOPostProcessor


def main():

    BASE_DIR = Path(__file__).resolve().parent.parent

    MODEL_PATH = BASE_DIR / "model" / "best.onnx"

    IMAGE_PATH = (
        r"D:\Project YOLO detection\onnx results\resize16_9.jpg"
    )

    OUTPUT_PATH = BASE_DIR / "onnx results" / "test_detection.jpg"

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(IMAGE_PATH)

    # Initialize backend
    engine = ONNXRuntimeEngine(
        model_path=MODEL_PATH,
        imgsz=640
    )

    processor = YOLOPostProcessor(
        conf_threshold=0.25,
        iou_threshold=0.45,
        max_det=300
    )

    engine.warmup(iterations=10)

    # Inference
    outputs, transform = engine.infer_raw(image)

    # Postprocessing
    detections = processor.process(
        outputs,
        transform
    )

    # Draw detections
    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        class_name = detection["class_name"]
        confidence = detection["confidence"]

        color = (
            (0, 255, 0)
            if detection["class_id"] == 0
            else (0, 0, 255)
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = f"{class_name}: {confidence:.2f}"

        cv2.putText(
            image,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not cv2.imwrite(str(OUTPUT_PATH), image):
        raise RuntimeError(
            f"Failed to save image: {OUTPUT_PATH}"
        )

    print(f"Saved image: {OUTPUT_PATH.resolve()}")

    print("\n========== DETECTION RESULTS ==========")
    print(f"Total detections: {len(detections)}")

    for i, detection in enumerate(detections, 1):
        print(
            f"{i}. {detection['class_name']} | "
            f"Confidence: {detection['confidence']:.4f} | "
            f"BBox: {detection['bbox']}"
        )

    print(f"\nSaved image: {OUTPUT_PATH}")

    cv2.imshow("ONNX Tomato Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()