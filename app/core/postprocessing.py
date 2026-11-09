
import numpy as np


class YOLOPostProcessor:

    CLASS_NAMES = {
        0: "Fresh Tomato",
        1: "Rotten Tomato"
    }

    def __init__(
        self,
        conf_threshold=0.25,
        iou_threshold=0.45,
        max_det=300
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det

    # ==============================================
    # IOU CALCULATION
    # ==============================================

    @staticmethod
    def calculate_iou(box, boxes):

        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection_w = np.maximum(0, x2 - x1)
        intersection_h = np.maximum(0, y2 - y1)

        intersection = intersection_w * intersection_h

        area1 = (
            max(0, box[2] - box[0])
            * max(0, box[3] - box[1])
        )

        area2 = (
            np.maximum(0, boxes[:, 2] - boxes[:, 0])
            * np.maximum(0, boxes[:, 3] - boxes[:, 1])
        )

        union = area1 + area2 - intersection

        return intersection / np.maximum(union, 1e-7)

    # ==============================================
    # CLASS-AWARE NMS
    # ==============================================

    def nms(self, boxes, scores, class_ids):

        selected_indices = []

        for class_id in np.unique(class_ids):

            class_indices = np.where(
                class_ids == class_id
            )[0]

            class_scores = scores[class_indices]
            class_boxes = boxes[class_indices]

            # Highest confidence first
            order = np.argsort(class_scores)[::-1]

            while len(order) > 0:

                current = order[0]

                selected_indices.append(
                    class_indices[current]
                )

                if len(order) == 1:
                    break

                remaining = order[1:]

                ious = self.calculate_iou(
                    class_boxes[current],
                    class_boxes[remaining]
                )

                # Keep boxes with IoU <= threshold
                order = remaining[
                    ious <= self.iou_threshold
                ]

        # Global confidence sorting
        selected_indices.sort(
            key=lambda idx: scores[idx],
            reverse=True
        )

        return selected_indices[:self.max_det]

    # ==============================================
    # MAIN POSTPROCESSING
    # ==============================================

    def process(self, outputs, transform):

        if not outputs:
            raise ValueError("Model outputs are empty.")

        # ==========================================
        # 1. EXTRACT OUTPUT
        # ==========================================

        predictions = np.asarray(
            outputs[0],
            dtype=np.float32
        )

        # Expected: (1, 6, 8400)
        if predictions.ndim == 3:

            if predictions.shape[0] != 1:
                raise ValueError(
                    f"Unsupported batch size: {predictions.shape[0]}"
                )

            predictions = predictions[0]

        if (
            predictions.ndim != 2
            or predictions.shape[0] != 6
        ):
            raise ValueError(
                f"Unexpected output shape: {predictions.shape}"
            )

        # (6, 8400) -> (8400, 6)
        predictions = predictions.T

        if not np.all(np.isfinite(predictions)):
            raise ValueError(
                "Model output contains NaN or Inf values."
            )

        # ==========================================
        # 2. EXTRACT BOXES AND CLASS SCORES
        # ==========================================

        boxes_xywh = predictions[:, :4]

        class_scores = predictions[:, 4:6]

        class_ids = np.argmax(
            class_scores,
            axis=1
        )

        confidences = np.max(
            class_scores,
            axis=1
        )

        # ==========================================
        # 3. CONFIDENCE FILTERING
        # ==========================================

        valid_mask = (
            (confidences >= self.conf_threshold)
            & (confidences <= 1.0)
            & (boxes_xywh[:, 2] > 0)
            & (boxes_xywh[:, 3] > 0)
        )

        boxes_xywh = boxes_xywh[valid_mask]
        class_ids = class_ids[valid_mask]
        confidences = confidences[valid_mask]

        if len(boxes_xywh) == 0:
            return []

        # ==========================================
        # 4. XYWH -> XYXY
        # ==========================================

        cx = boxes_xywh[:, 0]
        cy = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]

        boxes_xyxy = np.stack(
            [
                cx - w / 2,
                cy - h / 2,
                cx + w / 2,
                cy + h / 2
            ],
            axis=1
        )

        # ==========================================
        # 5. CLASS-AWARE NMS
        # ==========================================

        selected_indices = self.nms(
            boxes=boxes_xyxy,
            scores=confidences,
            class_ids=class_ids
        )

        # ==========================================
        # 6. MAP TO ORIGINAL IMAGE
        # ==========================================

        original_h, original_w = (
            transform["original_shape"]
        )

        ratio = transform["ratio"]

        pad_x, pad_y = transform["pad"]

        results = []

        for idx in selected_indices:

            bx1, by1, bx2, by2 = boxes_xyxy[idx]

            # Remove letterbox padding
            bx1 = (bx1 - pad_x) / ratio
            by1 = (by1 - pad_y) / ratio

            bx2 = (bx2 - pad_x) / ratio
            by2 = (by2 - pad_y) / ratio

            # Clip to original image boundaries
            bx1 = np.clip(bx1, 0, original_w)
            by1 = np.clip(by1, 0, original_h)

            bx2 = np.clip(bx2, 0, original_w)
            by2 = np.clip(by2, 0, original_h)

            # Remove invalid boxes
            if bx2 <= bx1 or by2 <= by1:
                continue

            class_id = int(class_ids[idx])

            results.append({
                "class_id": class_id,
                "class_name": self.CLASS_NAMES[class_id],
                "confidence": float(confidences[idx]),
                "bbox": [
                    int(round(bx1)),
                    int(round(by1)),
                    int(round(bx2)),
                    int(round(by2))
                ]
            })

        return results
