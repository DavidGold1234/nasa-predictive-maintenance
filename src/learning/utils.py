import json

def load_features(path="src/learning/features.json"):
    with open(path, "r") as f:
        config = json.load(f)
    return config["selected_sensors"]
