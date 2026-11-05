
import torch
import onnxruntime as ort

MODEL_PATH = (
    r"D:\Project YOLO detection\model\best.onnx"
)

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("ONNX Runtime:", ort.__version__)

# Load CUDA/cuDNN DLLs from PyTorch
ort.preload_dlls()

# Initialize ONNX Runtime
session = ort.InferenceSession(
    MODEL_PATH,
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)

print("\nAvailable providers:")
print(ort.get_available_providers())

print("\nActive providers:")
print(session.get_providers())

assert "CUDAExecutionProvider" in session.get_providers(), (
    "CUDA initialization failed!"
)

print("\nSUCCESS: ONNX Runtime is using CUDA!")
