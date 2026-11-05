
import torch
import cv2
from ultralytics import YOLO
from pathlib import Path


def main():

    #* ==========================================*
    #* 1. CONFIGURATION
    #* ==========================================*

    BASE_DIR = Path(__file__).resolve().parent

    MODEL_PATH = (
        BASE_DIR
        / "runs"
        / "detect"
        / "runs"
        / "tomato_detection"
        / "yolo26n_baseline"
        / "weights"
        / "best.pt"
    )

    IMAGE_DIR = BASE_DIR / "rotten tomatoes"

    OUTPUT_DIR = BASE_DIR / "inference_results"

    IMAGE_SIZE = 640
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.7

    DEVICE = 0 if torch.cuda.is_available() else "cpu"

    CLASS_NAMES = {
        0: "Fresh Tomato",
        1: "Rotten Tomato"
    }


    #* ==========================================*
    #* 2. CHECK PATHS
    #* ==========================================*

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find model: {MODEL_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Cannot find image directory: {IMAGE_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    #* ==========================================*
    #* 3. LOAD TRAINED MODEL
    #* ==========================================*

    print("Loading trained model...")

    model = YOLO(str(MODEL_PATH))

    print(f"Model loaded: {MODEL_PATH}")
    print(f"Device: {DEVICE}")


    #* ==========================================*
    #* 4. GET INPUT IMAGES
    #* ==========================================*

    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp"
    ]

    image_paths = []

    for extension in image_extensions:
        image_paths.extend(IMAGE_DIR.glob(extension))

    image_paths = sorted(image_paths)

    if not image_paths:
        print("No images found!")
        return

    print(f"\nFound {len(image_paths)} images.")


    #* ==========================================*
    #* 5. INFERENCE
    #* ==========================================*

    for image_path in image_paths:

        print("\n" + "=" * 50)
        print(f"Processing: {image_path.name}")

        results = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False
        )

        result = results[0]

        # Get original image
        image = cv2.imread(str(image_path))

        if image is None:
            print("Cannot read image!")
            continue

        # Get detection information
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

        # Draw bounding boxes
        annotated_image = result.plot()

        # Save result
        output_path = OUTPUT_DIR / image_path.name

        cv2.imwrite(
            str(output_path),
            annotated_image
        )

        print(f"Saved result: {output_path}")

        # Display result
        cv2.imshow(
            "Tomato Detection",
            annotated_image
        )

        print("Press any key for next image...")
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    print("\nInference completed!")


if __name__ == "__main__":
    main()
