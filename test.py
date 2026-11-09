
from pathlib import Path

import numpy as np
import pandas as pd

from ultralytics import YOLO

import plotly.graph_objects as go


# =====================================================
# 1. CONFIGURATION
# =====================================================

DATA_YAML = Path("dataset_module/data/data.yaml")
MODEL_PATH = Path("model/best.pt")

IMAGE_SIZE = 640
BATCH_SIZE = 32
DEVICE = 0

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

OUTPUT_DIR = Path("test_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD MODEL
# =====================================================

def load_model():

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_YAML.resolve()}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH.resolve()}"
        )

    model = YOLO(str(MODEL_PATH))

    class_names = model.names

    if isinstance(class_names, dict):
        class_names = [
            class_names[i] for i in range(len(class_names))
        ]
    else:
        class_names = list(class_names)

    if len(class_names) != 2:
        raise ValueError(
            f"Expected 2 classes, got {len(class_names)}"
        )

    print("\nClasses:", class_names)

    return model, class_names


# =====================================================
# 3. RUN TEST
# =====================================================

def run_test(model):

    print("\nRunning evaluation on TEST dataset...")

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        plots=True,
        project=str(OUTPUT_DIR),
        name="yolo_test",
        exist_ok=True,
        verbose=True
    )

    return metrics


# =====================================================
# 4. EXTRACT CONFUSION MATRIX
# =====================================================

def extract_confusion_matrix(metrics, num_classes, class_names):

    # Ultralytics confusion matrix:
    # Rows    = Ground Truth
    # Columns = Predicted
    # Last row/column = Background

    cm_full = metrics.confusion_matrix.matrix

    if cm_full.shape != (num_classes + 1, num_classes + 1):
        raise ValueError(
            f"Unexpected confusion matrix shape: {cm_full.shape}"
        )

    print("\nFull Confusion Matrix (including Background):")
    print(cm_full)

    # Only keep the two main classes
    cm = cm_full[:num_classes, :num_classes]

    print("\n2-Class Confusion Matrix:")
    print(cm)

    cm_df = pd.DataFrame(
        cm,
        index=[f"GT {name}" for name in class_names],
        columns=[f"Pred {name}" for name in class_names]
    )

    cm_df.to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    return cm_full, cm


# =====================================================
# 5. CALCULATE METRICS
# =====================================================

def calculate_metrics(cm_full, num_classes):

    cm = cm_full[:num_classes, :num_classes]

    # True Positive
    TP = np.diag(cm)

    # Background is still included in FP and FN calculations
    FP = cm_full[:, :num_classes].sum(axis=0) - TP
    FN = cm_full[:num_classes, :].sum(axis=1) - TP

    # Number of ground-truth objects
    support = TP + FN

    # Precision
    precision = np.divide(
        TP,
        TP + FP,
        out=np.zeros(num_classes, dtype=float),
        where=(TP + FP) != 0
    )

    # Recall
    recall = np.divide(
        TP,
        TP + FN,
        out=np.zeros(num_classes, dtype=float),
        where=(TP + FN) != 0
    )

    # F1-score
    f1_score = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=float),
        where=(precision + recall) != 0
    )

    # Macro Average
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1_score.mean()

    # Weighted Average
    total_support = support.sum()

    if total_support > 0:

        weighted_precision = np.average(
            precision,
            weights=support
        )

        weighted_recall = np.average(
            recall,
            weights=support
        )

        weighted_f1 = np.average(
            f1_score,
            weights=support
        )

    else:

        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0

    # Micro Average
    total_tp = TP.sum()
    total_fp = FP.sum()
    total_fn = FN.sum()

    micro_precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0 else 0.0
    )

    micro_recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0 else 0.0
    )

    micro_f1 = (
        2 * micro_precision * micro_recall /
        (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0 else 0.0
    )

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "support": support,

        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,

        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,

        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,

        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1
    }


# =====================================================
# 6. PRINT RESULTS
# =====================================================

def print_results(metrics, class_names, results):

    print("\n" + "=" * 70)
    print("              TOMATO OBJECT DETECTION RESULTS")
    print("=" * 70)

    TP = results["TP"]
    FP = results["FP"]
    FN = results["FN"]
    support = results["support"]

    precision = results["precision"]
    recall = results["recall"]
    f1_score = results["f1_score"]

    print("\nPER-CLASS RESULTS")
    print("-" * 70)

    for i, name in enumerate(class_names):

        print(f"\nClass: {name}")
        print(f"  TP        : {TP[i]}")
        print(f"  FP        : {FP[i]}")
        print(f"  FN        : {FN[i]}")
        print(f"  Support   : {support[i]}")
        print(f"  Precision : {precision[i]:.4f}")
        print(f"  Recall    : {recall[i]:.4f}")
        print(f"  F1-score  : {f1_score[i]:.4f}")

    print("\n" + "-" * 70)
    print("MACRO AVERAGE")

    print(f"  Precision : {results['macro_precision']:.4f}")
    print(f"  Recall    : {results['macro_recall']:.4f}")
    print(f"  F1-score  : {results['macro_f1']:.4f}")

    print("\nWEIGHTED AVERAGE")

    print(f"  Precision : {results['weighted_precision']:.4f}")
    print(f"  Recall    : {results['weighted_recall']:.4f}")
    print(f"  F1-score  : {results['weighted_f1']:.4f}")

    print("\nMICRO AVERAGE")

    print(f"  Precision : {results['micro_precision']:.4f}")
    print(f"  Recall    : {results['micro_recall']:.4f}")
    print(f"  F1-score  : {results['micro_f1']:.4f}")

    print("\nYOLO DETECTION METRICS")
    print("-" * 70)

    for key, value in metrics.results_dict.items():
        print(f"  {key}: {value}")


# =====================================================
# 7. SAVE CLASSIFICATION REPORT
# =====================================================

def save_classification_report(class_names, results):

    report_df = pd.DataFrame({

        "Class": class_names,

        "TP": results["TP"],
        "FP": results["FP"],
        "FN": results["FN"],
        "Support": results["support"],

        "Precision": results["precision"],
        "Recall": results["recall"],
        "F1-score": results["f1_score"]

    })

    report_df.to_csv(
        OUTPUT_DIR / "classification_report.csv",
        index=False,
        float_format="%.6f"
    )

    average_df = pd.DataFrame({

        "Average": ["Micro", "Macro", "Weighted"],

        "Precision": [
            results["micro_precision"],
            results["macro_precision"],
            results["weighted_precision"]
        ],

        "Recall": [
            results["micro_recall"],
            results["macro_recall"],
            results["weighted_recall"]
        ],

        "F1-score": [
            results["micro_f1"],
            results["macro_f1"],
            results["weighted_f1"]
        ]

    })

    average_df.to_csv(
        OUTPUT_DIR / "average_metrics.csv",
        index=False,
        float_format="%.6f"
    )


# =====================================================
# 8. CREATE INDIVIDUAL PLOTS WITH PLOTLY
# =====================================================

def create_individual_plots(metrics, cm_full, class_names, results):

    print("\nGenerating individual evaluation plots...")

    TP = results["TP"]
    FP = results["FP"]
    FN = results["FN"]

    precision = results["precision"]
    recall = results["recall"]
    f1_score = results["f1_score"]

    # YOLO detection metrics
    map50 = metrics.box.map50
    map5095 = metrics.box.map
    yolo_precision = metrics.box.mp
    yolo_recall = metrics.box.mr

    FONT = "Arial"
    TEMPLATE = "plotly_white"

    # -------------------------------------------------
    # COMMON FIGURE SETTINGS
    # -------------------------------------------------

    def save_figure(fig, filename, width=1500, height=950):

        fig.update_layout(
            template=TEMPLATE,

            font=dict(
                family=FONT,
                size=17
            ),

            width=width,
            height=height,

            # More space for title and bottom legend
            margin=dict(
                l=110,
                r=100,
                t=150,
                b=170
            ),

            title=dict(
                x=0.5,
                xanchor="center",
                y=0.98,
                yanchor="top",
                font=dict(size=24)
            ),

            hoverlabel=dict(
                font=dict(size=14)
            )
        )

        save_path = OUTPUT_DIR / filename

        try:

            fig.write_image(
                str(save_path),
                width=width,
                height=height,
                scale=3
            )

            print(f"Saved: {save_path.resolve()}")

        except Exception as error:

            print(f"Could not save {filename}: {error}")

            html_path = OUTPUT_DIR / filename.replace(".png", ".html")

            fig.write_html(str(html_path))

            print(f"Saved HTML instead: {html_path.resolve()}")

    # -------------------------------------------------
    # COMMON LEGEND SETTINGS
    # -------------------------------------------------

    def set_bottom_legend(fig):

        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
                font=dict(size=15),
                tracegroupgap=10
            )
        )

    # =================================================
    # FIGURE 1: CLASS-WISE PERFORMANCE
    # =================================================

    fig1 = go.Figure()

    fig1.add_trace(
        go.Bar(
            name="Precision",
            x=class_names,
            y=precision,
            text=[f"{v:.4f}" for v in precision],
            textposition="outside"
        )
    )

    fig1.add_trace(
        go.Bar(
            name="Recall",
            x=class_names,
            y=recall,
            text=[f"{v:.4f}" for v in recall],
            textposition="outside"
        )
    )

    fig1.add_trace(
        go.Bar(
            name="F1-score",
            x=class_names,
            y=f1_score,
            text=[f"{v:.4f}" for v in f1_score],
            textposition="outside"
        )
    )

    fig1.update_layout(
        title="Class-wise Performance",
        xaxis_title="Tomato Class",
        yaxis_title="Score",
        barmode="group",

        yaxis=dict(
            range=[0, 1.18],
            tickformat=".0%",
            automargin=True
        )
    )

    set_bottom_legend(fig1)

    save_figure(
        fig1,
        "class_performance.png"
    )

    # =================================================
    # FIGURE 2: 2-CLASS CONFUSION MATRIX
    # =================================================

    cm = cm_full[:len(class_names), :len(class_names)]

    fig2 = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,

            colorscale="Blues",

            text=cm.astype(int),
            texttemplate="%{text}",
            textfont=dict(size=25),

            colorbar=dict(
                title=dict(
                    text="Number of Objects",
                    font=dict(size=15)
                ),
                tickfont=dict(size=14)
            ),

            hovertemplate=(
                "Ground Truth: %{y}<br>"
                "Predicted: %{x}<br>"
                "Count: %{z}"
                "<extra></extra>"
            )
        )
    )

    fig2.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="Ground Truth Class",

        xaxis=dict(
            tickfont=dict(size=15)
        ),

        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=15)
        )
    )

    save_figure(
        fig2,
        "confusion_matrix.png",
        width=1200,
        height=1000
    )

    # =================================================
    # FIGURE 3: DETECTION STATISTICS
    # =================================================

    fig3 = go.Figure()

    fig3.add_trace(
        go.Bar(
            name="True Positive",
            x=class_names,
            y=TP,
            text=TP,
            textposition="outside"
        )
    )

    fig3.add_trace(
        go.Bar(
            name="False Positive",
            x=class_names,
            y=FP,
            text=FP,
            textposition="outside"
        )
    )

    fig3.add_trace(
        go.Bar(
            name="False Negative",
            x=class_names,
            y=FN,
            text=FN,
            textposition="outside"
        )
    )

    fig3.update_layout(
        title="Detection Statistics",
        xaxis_title="Tomato Class",
        yaxis_title="Number of Objects",
        barmode="group",

        yaxis=dict(
            rangemode="tozero",
            automargin=True
        )
    )

    set_bottom_legend(fig3)

    save_figure(
        fig3,
        "detection_statistics.png"
    )

    # =================================================
    # FIGURE 4: AVERAGE PERFORMANCE
    # =================================================

    average_names = [
        "Precision",
        "Recall",
        "F1-score"
    ]

    average_values = {

        "Micro": [
            results["micro_precision"],
            results["micro_recall"],
            results["micro_f1"]
        ],

        "Macro": [
            results["macro_precision"],
            results["macro_recall"],
            results["macro_f1"]
        ],

        "Weighted": [
            results["weighted_precision"],
            results["weighted_recall"],
            results["weighted_f1"]
        ]

    }

    fig4 = go.Figure()

    for name, values in average_values.items():

        fig4.add_trace(
            go.Bar(
                name=name,
                x=average_names,
                y=values,
                text=[f"{v:.4f}" for v in values],
                textposition="outside"
            )
        )

    fig4.update_layout(
        title="Average Performance Comparison",
        xaxis_title="Evaluation Metric",
        yaxis_title="Score",
        barmode="group",

        yaxis=dict(
            range=[0, 1.18],
            tickformat=".0%"
        )
    )

    set_bottom_legend(fig4)

    save_figure(
        fig4,
        "average_performance.png"
    )

    # =================================================
    # FIGURE 5: YOLO DETECTION METRICS
    # =================================================

    yolo_metric_names = [
        "Precision",
        "Recall",
        "mAP50",
        "mAP50-95"
    ]

    yolo_metric_values = [
        yolo_precision,
        yolo_recall,
        map50,
        map5095
    ]

    fig5 = go.Figure()

    fig5.add_trace(
        go.Bar(
            name="Score",
            x=yolo_metric_names,
            y=yolo_metric_values,

            text=[
                f"{v:.4f}"
                for v in yolo_metric_values
            ],

            textposition="outside"
        )
    )

    fig5.update_layout(
        title="YOLO Detection Evaluation Metrics",
        xaxis_title="Evaluation Metric",
        yaxis_title="Score",

        yaxis=dict(
            range=[0, 1.18],
            tickformat=".0%"
        ),

        showlegend=False
    )

    save_figure(
        fig5,
        "yolo_detection_metrics.png"
    )

    # =================================================
    # FINISH
    # =================================================

    print("\nAll individual plots generated!")

    print(
        "Output directory:",
        OUTPUT_DIR.resolve()
    )


# =====================================================
# 9. MAIN
# =====================================================

def main():

    # Load model
    model, class_names = load_model()

    num_classes = len(class_names)

    # Run testing
    metrics = run_test(model)

    # Extract confusion matrix
    cm_full, cm = extract_confusion_matrix(
        metrics,
        num_classes,
        class_names
    )

    # Calculate metrics
    results = calculate_metrics(
        cm_full,
        num_classes
    )

    # Print results
    print_results(
        metrics,
        class_names,
        results
    )

    # Save CSV reports
    save_classification_report(
        class_names,
        results
    )

    # Create individual plots
    create_individual_plots(
        metrics,
        cm_full,
        class_names,
        results
    )

    print("\n" + "=" * 70)
    print("TESTING COMPLETED!")
    print("Output directory:", OUTPUT_DIR.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()
