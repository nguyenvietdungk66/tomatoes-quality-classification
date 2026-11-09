from settings_manager import SettingsManager


settings_manager = SettingsManager()

print("Current settings:")
print(settings_manager.get_settings())

print("\nInspection frames:")
print(settings_manager.get("inspection", "required_frames"))

# Test updating a setting
settings_manager.update_settings(
    "inspection",
    "required_frames",
    5
)

print("\nUpdated inspection frames:")
print(settings_manager.get("inspection", "required_frames"))