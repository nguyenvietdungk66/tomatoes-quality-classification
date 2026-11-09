
import json
from pathlib import Path


class SettingsManager:

    DEFAULT_SETTINGS = {
        "inspection": {
            "required_frames": 1,
            "empty_frames_to_release": 10,
            "required_detections": 5
        },
        "detection": {
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "input_size": 640,
            "max_detections": 300
        }
    }

    def __init__(self, settings_path=None):

        if settings_path is None:
            settings_path = (
                Path(__file__).resolve().parent
                / "settings.json"
            )

        self.settings_path = Path(settings_path)

        self.settings = self.load_settings()

    # ==========================================
    # LOAD SETTINGS
    # ==========================================

    def load_settings(self):

        if not self.settings_path.exists():

            self.settings_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            self.save_settings(
                self.DEFAULT_SETTINGS
            )

            return self.copy_defaults()

        try:

            with open(
                self.settings_path,
                "r",
                encoding="utf-8"
            ) as file:

                loaded_settings = json.load(file)

            settings = self.copy_defaults()

            # Merge inspection settings
            settings["inspection"].update(
                loaded_settings.get("inspection", {})
            )

            # Merge detection settings
            settings["detection"].update(
                loaded_settings.get("detection", {})
            )

            self.settings = settings

            if not self.validate_settings(settings):
                raise ValueError("Invalid settings")

            # Save merged settings to include missing keys
            self.save_settings(settings)

            return settings

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
            OSError
        ):

            settings = self.copy_defaults()

            self.save_settings(settings)

            return settings

    # ==========================================
    # SAVE SETTINGS
    # ==========================================

    def save_settings(self, settings=None):

        if settings is None:
            settings = self.settings

        if not self.validate_settings(settings):
            raise ValueError("Invalid settings configuration")

        self.settings_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        temporary_path = self.settings_path.with_suffix(".tmp")

        with open(
            temporary_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                settings,
                file,
                indent=4,
                ensure_ascii=False
            )

        temporary_path.replace(self.settings_path)

        self.settings = settings

    # ==========================================
    # GET SETTINGS
    # ==========================================

    def get_settings(self):

        return self.settings

    def get(self, section, key):

        return self.settings[section][key]

    # ==========================================
    # UPDATE SETTINGS
    # ==========================================

    def update_settings(self, section, key, value):

        if section not in self.settings:
            raise KeyError(f"Unknown settings section: {section}")

        if key not in self.settings[section]:
            raise KeyError(f"Unknown setting: {key}")

        updated_settings = {
            "inspection": self.settings["inspection"].copy(),
            "detection": self.settings["detection"].copy()
        }

        updated_settings[section][key] = value

        if not self.validate_settings(updated_settings):
            raise ValueError(
                f"Invalid value for {section}.{key}: {value}"
            )

        self.save_settings(updated_settings)

    # ==========================================
    # RESET SETTINGS
    # ==========================================

    def reset_settings(self):

        settings = self.copy_defaults()

        self.save_settings(settings)

        return settings

    # ==========================================
    # VALIDATION
    # ==========================================

    @staticmethod
    def validate_settings(settings):

        try:

            inspection = settings["inspection"]
            detection = settings["detection"]

            # Inspection settings
            required_frames = inspection["required_frames"]
            empty_frames = inspection["empty_frames_to_release"]
            required_detections = inspection["required_detections"]

            if not isinstance(required_frames, int) or not 1 <= required_frames <= 300:
                return False

            if not isinstance(empty_frames, int) or not 1 <= empty_frames <= 100:
                return False

            if not isinstance(required_detections, int) or not 1 <= required_detections <= 30:
                return False

            # Detection settings
            confidence = detection["confidence_threshold"]
            iou = detection["iou_threshold"]
            input_size = detection["input_size"]
            max_detections = detection["max_detections"]

            if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
                return False

            if not isinstance(iou, (int, float)) or not 0.0 <= iou <= 1.0:
                return False

            if not isinstance(input_size, int) or input_size != 640:
                return False

            if not isinstance(max_detections, int) or max_detections <= 0:
                return False

            return True

        except (KeyError, TypeError):
            return False

    # ==========================================
    # DEFAULT SETTINGS
    # ==========================================

    @classmethod
    def copy_defaults(cls):

        return {
            "inspection": cls.DEFAULT_SETTINGS["inspection"].copy(),
            "detection": cls.DEFAULT_SETTINGS["detection"].copy()
        }
