# Tomato Quality Inspection System

An AI-powered desktop application for automated tomato quality inspection using Deep Learning and industrial machine vision.

The system integrates a custom-trained YOLO26 object detection model, ONNX Runtime inference engine, and Hikrobot industrial camera to detect and classify fresh and rotten tomatoes in real time.



## 1. Project Overview

This project aims to develop an intelligent computer vision system for automated tomato quality inspection, demonstrating an end-to-end workflow from model development and optimization to real-time industrial camera integration and desktop application deployment.

The application detects tomatoes in camera frames and classifies them into two quality categories:

```Fresh Tomato```: Tomatoes identified as fresh and suitable for further processing.

```Rotten Tomato```: Tomatoes identified as rotten or unsuitable for further processing.

Unlike a standalone object detection model, this project integrates the trained model into a complete desktop application with live camera inspection, real-time visualization, inspection statistics, and historical data management.

## Project structure

The structure below is illustrative. Adjust it to match the actual folders in this repository.

## 2. System Architecture

The application follows a modular architecture that separates camera acquisition, AI inference, inspection processing, data management, and user interface components.

```text
 Industrial Camera
              (Hikrobot / USB)
                     |
                     v
              Camera Acquisition
                     |
                     v
               Frame Processing
                     |
                     v
              YOLO26n ONNX Model
                     |
                     v
              Detection Post-processing
                     |
                     v
              Inspection Management
                     |
              +------+------+
              |             |
              v             v
        Live Visualization  Inspection Results
              |             |
              v             v
           PySide6 UI     SQLite Database
              |
              v
         Dashboard / History
```

## Requirements

- Windows, Linux, or another OS supported by the installed Python packages
- Python 3.10+ (use the version tested by your project environment)
- A webcam or compatible camera for live inference
- A trained YOLO model exported to ONNX

Install the dependencies listed in `requirements.txt`:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

If `requirements.txt` has not yet been created, install the packages used by the project (for example, `onnxruntime` or `onnxruntime-gpu`, and `opencv-python`) and record the tested versions in that file.

## Model

Place the exported ONNX model in the location expected by the inference configuration. The example project structure uses:

```text
models/best.onnx
```

Update the model path in the relevant configuration or Python entry point if your model is stored elsewhere.

The ONNX model is a trained artifact. Its availability, size, training data, and usage rights should be documented separately where applicable.

## Run inference

Run the project's camera entry point from the repository root, for example:

```bash
python onnx/real_time_onnx.py
```

If your actual entry point or folder structure differs, use the corresponding script and update this command.

## Dataset and training

Large datasets are intentionally excluded from this repository by default. Download the dataset separately and place it in the directory expected by the training configuration.

Before training, check:

- The dataset directory paths in the YOLO data configuration.
- The class names and class IDs.
- The train/validation/test split.
- The model architecture and training parameters.

Do not commit datasets, private data, or large training outputs unless you have permission and a clear reason to distribute them.

## Git workflow

```bash
git add .
git commit -m "Describe your changes"
git push
```

## Notes

- Detection performance depends on the model, camera, lighting, image resolution, and inference provider.
- GPU acceleration requires a compatible runtime, drivers, and hardware. CPU inference may be used where GPU support is unavailable.
- This repository is a software project; validate the complete camera, actuator, and robot/PLC safety chain before any industrial deployment.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).

If you distribute a dataset, pretrained weights, or third-party code, check their individual licenses. The MIT license for this repository does not automatically grant rights to those materials.
