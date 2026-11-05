from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class ONNXRuntimeEngine:

    def __init__(
        self,
        model_path,
        imgsz=640,
        providers=None
    ):
        self.model_path = Path(model_path)
        self.imgsz = imgsz

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.model_path}"
            )

        # ==========================================
        # 1. LOAD CUDA DLLs
        # ==========================================

        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
                print("[INFO] CUDA DLL preload completed.")
            except Exception as e:
                print(f"[WARNING] DLL preload: {e}")

        # ==========================================
        # 2. CONFIGURE EXECUTION PROVIDERS
        # ==========================================

        if providers is None:
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ]

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers
        )

        # ==========================================
        # 3. GET MODEL INFORMATION
        # ==========================================

        self.input_info = self.session.get_inputs()[0]
        self.output_info = self.session.get_outputs()

        self.input_name = self.input_info.name
        self.output_names = [
            output.name for output in self.output_info
        ]

        self.input_shape = self.input_info.shape

        if self.input_shape != [1, 3, 640, 640]:
            raise ValueError(
                f"Unexpected model input shape: {self.input_shape}"
            )

        print("\n========== ONNX ENGINE ==========")
        print(f"Model: {self.model_path.name}")
        print(f"Input name: {self.input_name}")
        print(f"Input shape: {self.input_shape}")
        print(f"Output names: {self.output_names}")
        print(f"Available providers: {ort.get_available_providers()}")
        print(f"Session providers: {self.session.get_providers()}")
        print("=================================\n")

    # ==============================================
    # LETTERBOX PREPROCESSING
    # ==============================================

    def preprocess(self, image):

        if image is None:
            raise ValueError("Input image is None.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected BGR image with 3 channels, got {image.shape}"
            )

        original_h, original_w = image.shape[:2]

        target_h = self.imgsz
        target_w = self.imgsz

        # Calculate resize ratio
        ratio = min(
            target_w / original_w,
            target_h / original_h
        )

        new_w = round(original_w * ratio)
        new_h = round(original_h * ratio)

        # Resize image
        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )

        # Calculate padding
        dw = target_w - new_w
        dh = target_h - new_h

        left = round(dw / 2 - 0.1)
        right = round(dw / 2 + 0.1)

        top = round(dh / 2 - 0.1)
        bottom = round(dh / 2 + 0.1)

        # Add padding
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )

        # BGR -> RGB
        rgb = cv2.cvtColor(
            padded,
            cv2.COLOR_BGR2RGB
        )

        # HWC -> CHW
        tensor = rgb.transpose(2, 0, 1)

        # Normalize
        tensor = tensor.astype(np.float32) / 255.0

        # Add batch dimension
        tensor = np.expand_dims(tensor, axis=0)

        tensor = np.ascontiguousarray(tensor)

        transform = {
            "original_shape": (original_h, original_w),
            "ratio": ratio,
            "pad": (left, top)
        }

        return tensor, transform

    # ==============================================
    # RAW INFERENCE
    # ==============================================

    def infer_raw(self, image):

        input_tensor, transform = self.preprocess(image)

        outputs = self.session.run(
            self.output_names,
            {
                self.input_name: input_tensor
            }
        )

        return outputs, transform

    # ==============================================
    # WARMUP
    # ==============================================

    def warmup(self, iterations=10):

        dummy = np.zeros(
            (self.imgsz, self.imgsz, 3),
            dtype=np.uint8
        )

        for _ in range(iterations):
            self.infer_raw(dummy)

        print(
            f"[INFO] Warmup completed: {iterations} iterations."
        )
