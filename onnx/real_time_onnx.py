
from pathlib import Path

from onnx_engine import ONNXRuntimeEngine
from postprocessing import YOLOPostProcessor
from camera import CameraManager


def main():

    BASE_DIR = Path(__file__).resolve().parent.parent

    MODEL_PATH = BASE_DIR / "model" / "best.onnx"

    OUTPUT_DIR = (
        BASE_DIR
        / "onnx results"
        / "camera snapshots"
    )

    # ==========================================
    # 1. INITIALIZE ONNX ENGINE
    # ==========================================

    engine = ONNXRuntimeEngine(
        model_path=MODEL_PATH,
        imgsz=640
    )

    # ==========================================
    # 2. INITIALIZE POSTPROCESSOR
    # ==========================================

    processor = YOLOPostProcessor(
        conf_threshold=0.25,
        iou_threshold=0.45,
        max_det=300
    )

    # ==========================================
    # 3. INITIALIZE CAMERA
    # ==========================================

    camera = CameraManager(
        engine=engine,
        processor=processor,
        camera_index=0,
        width=1280,
        height=720,
        output_dir=OUTPUT_DIR
    )

    # ==========================================
    # 4. WARMUP MODEL
    # ==========================================

    engine.warmup(iterations=10)

    # ==========================================
    # 5. START CAMERA
    # ==========================================

    camera.run()


if __name__ == "__main__":
    main()
