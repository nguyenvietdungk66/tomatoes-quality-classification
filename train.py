import torch
from ultralytics import YOLO
from pathlib import Path

def main():
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================

    DATA_YAML = Path("dataset_module/data/data.yaml")

    MODEL_NAME = "yolo26n.pt"

    EPOCHS = 100
    IMAGE_SIZE = 640
    BATCH_SIZE = 32

    DEVICE = 0       
    WORKERS = 4

    PROJECT = "runs/tomato_detection"
    EXPERIMENT_NAME = "yolo26n_baseline"


    # ==========================================
    # 2. CHECK DATASET
    # ==========================================

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Cannot find dataset config: {DATA_YAML.resolve()}"
        )


    # ==========================================
    # 3. LOAD MODEL
    # ==========================================

    model = YOLO(MODEL_NAME)


    # ==========================================
    # 4. TRAINING
    # ==========================================

    results = model.train(

        # Dataset
        data=str(DATA_YAML),

        # Training parameters
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        # Optimizer
        optimizer="auto",

        # Training control
        patience=20,
        save=True,
        save_period=10,

        # Hardware
        device=DEVICE,
        workers=WORKERS,

        # Reproducibility
        seed=42,
        deterministic=True,

        # Output
        project=PROJECT,
        name=EXPERIMENT_NAME,
        exist_ok=True,

        # Visualization
        plots=True,
        verbose=True
    )


    # ==========================================
    # 5. VALIDATION
    # ==========================================

    best_model_path = (
        Path(PROJECT)
        / EXPERIMENT_NAME
        / "weights"
        / "best.pt"
    )

    print("\nTraining completed!")
    print(f"Best model: {best_model_path}")

    best_model = YOLO(str(best_model_path))

    metrics = best_model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE
    )

    print("\nValidation results:")
    print(metrics.results_dict)


    # ==========================================
    # 6. TESTING
    # ==========================================

    test_metrics = best_model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE
    )

    print("\nTest results:")
    print(test_metrics.results_dict)

if __name__ == '__main__':
    main()
