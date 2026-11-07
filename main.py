from pathlib import Path

from onnx_processing.onnx_engine import ONNXRuntimeEngine
from onnx_processing.postprocessing import YOLOPostProcessor
from camera_setup.camera import CameraManager


def main():

    BASE_DIR = Path(__file__).resolve().parent

    MODEL_PATH = BASE_DIR / "model" / "best.onnx"

    OUTPUT_DIR = (
        BASE_DIR
        / "onnx results"
        / "camera snapshots"
    )

    # ==========================================
    # 1. CAMERA CONFIGURATION
    # ==========================================

    CAMERA_TYPE = "webcam"  # "webcam" or "hikrobot"
    CAMERA_INDEX = 0

    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720

    # ==========================================
    # 2. INITIALIZE ONNX ENGINE
    # ==========================================

    engine = ONNXRuntimeEngine(
        model_path=MODEL_PATH,
        imgsz=640
    )

    # ==========================================
    # 3. INITIALIZE POSTPROCESSOR
    # ==========================================

    processor = YOLOPostProcessor(
        conf_threshold=0.25,
        iou_threshold=0.45,
        max_det=300
    )

    # ==========================================
    # 4. INITIALIZE CAMERA
    # ==========================================

    camera = CameraManager(
        engine=engine,
        processor=processor,
        camera_type=CAMERA_TYPE,
        camera_index=CAMERA_INDEX,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        output_dir=OUTPUT_DIR
    )

    # ==========================================
    # 5. WARMUP MODEL
    # ==========================================

    engine.warmup(iterations=10)

    # ==========================================
    # 6. START CAMERA
    # ==========================================

    camera.run()


if __name__ == "__main__":
    main()
