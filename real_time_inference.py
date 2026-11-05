
import torch
import cv2
import time

from ultralytics import YOLO
from pathlib import Path


#* ==========================================*
#* 1. CONFIGURATION
#* ==========================================*

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "best.pt"
)

IMAGE_DIR = BASE_DIR / "test_images"
OUTPUT_DIR = BASE_DIR / "inference_results"

IMAGE_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7

DEVICE = 0 if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {
    0: "Fresh Tomato",
    1: "Rotten Tomato"
}

# Select inference mode: "image" or "webcam"
MODE = "webcam"

# Webcam configuration
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


#* ==========================================*
#* 2. IMAGE INFERENCE
#* ==========================================*

def run_image_inference(model):

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Cannot find image directory: {IMAGE_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    image_paths = []

    for extension in image_extensions:
        image_paths.extend(IMAGE_DIR.glob(extension))

    image_paths = sorted(image_paths)

    if not image_paths:
        print("No images found!")
        return

    print(f"Found {len(image_paths)} images.")

    for image_path in image_paths:

        print(f"\nProcessing: {image_path.name}")

        results = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False
        )

        result = results[0]

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            print("No tomato detected.")

        else:
            for index, box in enumerate(boxes):

                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                x1, y1, x2, y2 = (
                    box.xyxy[0].cpu().numpy().astype(int)
                )

                class_name = CLASS_NAMES.get(
                    class_id,
                    f"Class {class_id}"
                )

                print(
                    f"Object {index + 1}: "
                    f"{class_name} | "
                    f"Confidence: {confidence:.4f} | "
                    f"BBox: ({x1}, {y1}, {x2}, {y2})"
                )

        annotated_image = result.plot()

        output_path = OUTPUT_DIR / image_path.name

        cv2.imwrite(str(output_path), annotated_image)

        cv2.imshow("Tomato Detection", annotated_image)

        print("Press any key for next image...")
        cv2.waitKey(0)

    cv2.destroyAllWindows()


#* ==========================================*
#* 3. WEBCAM INFERENCE
#* ==========================================*

def run_webcam_inference(model):

    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open webcam with index {CAMERA_INDEX}"
        )

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("\nWebcam started!")
    print("Press Q to quit.")
    print("Press S to save current frame.")

    frame_count = 0
    fps = 0.0

    try:

        while True:

            # 1. Capture frame
            ret, frame = cap.read()

            if not ret:
                print("Cannot receive frame from webcam.")
                break

            start_time = time.perf_counter()

            # 2. YOLO inference
            results = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=DEVICE,
                verbose=False
            )

            result = results[0]

            # 3. Draw bounding boxes
            annotated_frame = result.plot()

            # 4. Calculate FPS
            elapsed_time = time.perf_counter() - start_time

            if elapsed_time > 0:
                current_fps = 1.0 / elapsed_time

                # Smooth FPS to reduce fluctuations
                if frame_count == 0:
                    fps = current_fps
                else:
                    fps = 0.9 * fps + 0.1 * current_fps

            frame_count += 1

            # 5. Display FPS
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # 6. Display detection count
            detection_count = (
                len(result.boxes) if result.boxes is not None else 0
            )

            cv2.putText(
                annotated_frame,
                f"Objects: {detection_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "Real-time Tomato Detection",
                annotated_frame
            )

            # 7. Keyboard controls
            key = cv2.waitKey(1) & 0xFF

            # Quit
            if key == ord("q"):
                break

            # Save current frame
            elif key == ord("s"):

                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                timestamp = time.strftime("%Y%m%d_%H%M%S")

                output_path = (
                    OUTPUT_DIR / f"webcam_{timestamp}.jpg"
                )

                cv2.imwrite(
                    str(output_path),
                    annotated_frame
                )

                print(f"Saved: {output_path}")

    finally:

        # Release webcam
        cap.release()

        # Close OpenCV windows
        cv2.destroyAllWindows()

        print("\nWebcam stopped.")


#* ==========================================*
#* 4. MAIN
#* ==========================================*

def main():

    # Check model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find model: {MODEL_PATH}"
        )

    print("Loading trained YOLO model...")

    # Load best.pt
    model = YOLO(str(MODEL_PATH))

    print(f"Model loaded: {MODEL_PATH}")
    print(f"Device: {DEVICE}")

    # Select inference mode
    if MODE == "image":

        run_image_inference(model)

    elif MODE == "webcam":

        run_webcam_inference(model)

    else:

        raise ValueError(
            "MODE must be either 'image' or 'webcam'"
        )


if __name__ == "__main__":
    main()
