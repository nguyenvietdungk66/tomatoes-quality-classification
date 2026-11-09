
from pathlib import Path

import cv2
import numpy as np

from onnx_engine import ONNXRuntimeEngine


def main():

    BASE_DIR = Path(__file__).resolve().parent.parent

    MODEL_PATH = BASE_DIR / "model" / "best.onnx"

    IMAGE_PATH = (
        BASE_DIR
        / "dataset_module"
        / "data"
        / "test"
        / "images"
        / "22_jpg.rf.fe1ed5c26bd9f434d99d900c5e23a536.jpg"
    )

    # ==========================================
    # 1. LOAD IMAGE
    # ==========================================

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(IMAGE_PATH)

    # ==========================================
    # 2. INITIALIZE ENGINE
    # ==========================================

    engine = ONNXRuntimeEngine(
        model_path=MODEL_PATH,
        imgsz=640
    )

    # ==========================================
    # 3. WARMUP
    # ==========================================

    engine.warmup(iterations=10)

    # ==========================================
    # 4. INFERENCE
    # ==========================================

    outputs, transform = engine.infer_raw(image)

    print("\n========== OUTPUT INSPECTION ==========")

    print("Original image:", image.shape)
    print("Transform:", transform)

    for i, output in enumerate(outputs):

        print(f"\nOutput {i}")
        print("Shape:", output.shape)
        print("Dtype:", output.dtype)
        print("Min:", output.min())
        print("Max:", output.max())

        squeezed = np.squeeze(output)

        print("Squeezed shape:", squeezed.shape)

        if squeezed.ndim == 2:

            print("First 10 predictions:")
            print(squeezed[..., :10]
                  if squeezed.shape[0] < squeezed.shape[1]
                  else squeezed[:10, :])

        print("First 20 flattened values:")
        print(output.flatten()[:20])

    # ==========================================
    # 5. MODEL METADATA
    # ==========================================

    metadata = engine.session.get_modelmeta()

    print("\n========== MODEL METADATA ==========")
    print(metadata.custom_metadata_map)

    print("\n====================================")


if __name__ == "__main__":
    main()
