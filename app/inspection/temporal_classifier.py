class TemporalClassifier:
    """Aggregate YOLO predictions across frames for one stable result."""

    CLASS_NAMES = {0: "Fresh Tomato", 1: "Rotten Tomato"}

    def __init__(self, required_frames=30):
        self.required_frames = int(required_frames)
        self.reset()

    def start(self, inspection_id):
        self.inspection_id = int(inspection_id)
        self.frame_count = 0
        self.class_counts = {0: 0, 1: 0}
        self.confidence_sums = {0: 0.0, 1: 0.0}
        self.result = None
        self.completed = False

    def update(self, detections):
        if self.inspection_id is None or self.completed or len(detections) != 1:
            return None

        detection = detections[0]
        class_id = int(detection["class_id"])
        confidence = float(detection["confidence"])
        if class_id not in self.class_counts:
            return None

        self.class_counts[class_id] += 1
        self.confidence_sums[class_id] += confidence
        self.frame_count += 1

        if self.frame_count >= self.required_frames:
            self._finalize()
            return self.result
        return None

    def _finalize(self):
        max_votes = max(self.class_counts.values())
        candidates = [
            class_id for class_id, count in self.class_counts.items()
            if count == max_votes
        ]
        winner = max(
            candidates,
            key=lambda class_id: (
                self.confidence_sums[class_id] / max(self.class_counts[class_id], 1)
            ),
        )
        average_confidence = (
            self.confidence_sums[winner] / max(self.class_counts[winner], 1)
        )
        self.result = {
            "inspection_id": self.inspection_id,
            "class_id": winner,
            "class_name": self.CLASS_NAMES[winner],
            "confidence": average_confidence,
            "fresh_votes": self.class_counts[0],
            "rotten_votes": self.class_counts[1],
            "total_frames": self.frame_count,
        }
        self.completed = True

    def state_dict(self):
        return {
            "required_frames": self.required_frames,
            "inspection_id": self.inspection_id,
            "frame_count": self.frame_count,
            "class_counts": {str(k): v for k, v in self.class_counts.items()},
            "confidence_sums": {str(k): v for k, v in self.confidence_sums.items()},
            "result": self.result,
            "completed": self.completed,
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise ValueError("Invalid temporal classifier checkpoint")
        self.required_frames = max(1, int(state.get("required_frames", self.required_frames)))
        inspection_id = state.get("inspection_id")
        self.inspection_id = None if inspection_id is None else int(inspection_id)
        self.frame_count = max(0, int(state.get("frame_count", 0)))
        counts = state.get("class_counts", {})
        sums = state.get("confidence_sums", {})
        self.class_counts = {0: int(counts.get("0", counts.get(0, 0))),
                             1: int(counts.get("1", counts.get(1, 0)))}
        self.confidence_sums = {0: float(sums.get("0", sums.get(0, 0.0))),
                                1: float(sums.get("1", sums.get(1, 0.0)))}
        self.result = state.get("result")
        self.completed = bool(state.get("completed", False))

    def reset(self):
        self.inspection_id = None
        self.frame_count = 0
        self.class_counts = {0: 0, 1: 0}
        self.confidence_sums = {0: 0.0, 1: 0.0}
        self.result = None
        self.completed = False
