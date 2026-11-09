class InspectionManager:
    WAITING_PRODUCT = "WAITING_PRODUCT"
    VERIFYING_PRODUCT = "VERIFYING_PRODUCT"
    INSPECTING = "INSPECTING"
    WAITING_NEXT_PRODUCT = "WAITING_NEXT_PRODUCT"

    VALID_STATES = {
        WAITING_PRODUCT,
        VERIFYING_PRODUCT,
        INSPECTING,
        WAITING_NEXT_PRODUCT,
    }

    def __init__(self, required_detections=5):
        self.required_detections = int(required_detections)
        self.next_inspection_id = 1
        self.current_inspection_id = None
        self.state = self.WAITING_PRODUCT
        self.detection_count = 0

    def update_detection(self, detections):
        if self.state in (self.INSPECTING, self.WAITING_NEXT_PRODUCT):
            return None

        if len(detections) != 1:
            self.detection_count = 0
            self.state = self.WAITING_PRODUCT
            return None

        self.state = self.VERIFYING_PRODUCT
        self.detection_count += 1

        if self.detection_count >= self.required_detections:
            self.current_inspection_id = self.next_inspection_id
            self.next_inspection_id += 1
            self.state = self.INSPECTING
            self.detection_count = 0
            return self.current_inspection_id
        return None

    def complete_cycle(self):
        if self.state != self.INSPECTING:
            return False
        self.current_inspection_id = None
        self.detection_count = 0
        self.state = self.WAITING_NEXT_PRODUCT
        return True

    def allow_next_product(self):
        if self.state != self.WAITING_NEXT_PRODUCT:
            return False
        self.state = self.WAITING_PRODUCT
        self.detection_count = 0
        return True

    def reset(self):
        self.next_inspection_id = 1
        self.current_inspection_id = None
        self.detection_count = 0
        self.state = self.WAITING_PRODUCT

    def state_dict(self):
        return {
            "required_detections": self.required_detections,
            "next_inspection_id": self.next_inspection_id,
            "current_inspection_id": self.current_inspection_id,
            "state": self.state,
            "detection_count": self.detection_count,
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise ValueError("Invalid inspection manager checkpoint")
        restored_state = state.get("state", self.WAITING_PRODUCT)
        if restored_state not in self.VALID_STATES:
            raise ValueError(f"Invalid inspection state: {restored_state}")

        self.required_detections = max(
            1, int(state.get("required_detections", self.required_detections))
        )
        self.next_inspection_id = max(1, int(state.get("next_inspection_id", 1)))
        current_id = state.get("current_inspection_id")
        self.current_inspection_id = None if current_id is None else int(current_id)
        self.state = restored_state
        self.detection_count = max(0, int(state.get("detection_count", 0)))

    def prepare_for_resume(self):
        """Require a fresh product after restart if a product was mid-inspection."""
        if self.state in (self.VERIFYING_PRODUCT, self.INSPECTING):
            # Reuse the unfinished product ID; it was never added to results.
            if self.current_inspection_id is not None:
                self.next_inspection_id = min(
                    self.next_inspection_id, self.current_inspection_id
                )
            self.current_inspection_id = None
            self.detection_count = 0
            self.state = self.WAITING_PRODUCT
        elif self.state == self.WAITING_PRODUCT:
            # Consecutive detection frames cannot span an application restart.
            self.detection_count = 0
        return self.state
