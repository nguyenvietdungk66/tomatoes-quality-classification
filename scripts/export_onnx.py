
from pathlib import Path
from ultralytics import YOLO
import onnx

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "model" / "best.pt"

IMG_SIZE = 640
BATCH_SIZE = 1


def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("=" * 50)
    print("YOLO ONNX EXPORT")
    print("=" * 50)

    # Load PyTorch model
    model = YOLO(str(MODEL_PATH))

    # Check class names
    print("\nModel classes:")
    print(model.names)

    # Export ONNX
    onnx_path = model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        half=False,
        dynamic=False,
        simplify=False,
        opset=17
    )

    onnx_path = Path(onnx_path)

    print("\nExport completed!")
    print("ONNX path:", onnx_path)

    # Validate ONNX structure
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    print("ONNX validation: PASSED")
    print("File size:",
          round(onnx_path.stat().st_size / 1024**2, 2),
          "MB")


if __name__ == "__main__":
    main()
