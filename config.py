"""
Centralized Configuration for Fire & Smoke Detection Engine (v2)
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fire_smoke_model_v2.h5")
LEGACY_MODEL_PATH = os.path.join(MODEL_DIR, "fire_smoke_model.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
LOG_DIR = os.path.join(BASE_DIR, "logs")
HISTORY_FILE = os.path.join(LOG_DIR, "detection_history.json")

# Model Settings
IMG_SIZE = (224, 224)  # Standard input for MobileNetV2 / Transfer learning models
FIRE_THRESHOLD = 0.40  # Confidence threshold for Fire classification
SMOKE_THRESHOLD = 0.45 # Confidence threshold for Smoke classification
SUPPRESS_FIRE_ON_FACE = True

# Video Processing Settings
TEMPORAL_WINDOW_SIZE = 7  # Number of frames for rolling window average prediction
SMOOTHING_ALPHA = 0.3     # Exponential moving average factor

# Alert Settings
ALERT_COOLDOWN_SECONDS = 5.0

# Flask Settings
SECRET_KEY = os.environ.get("SECRET_KEY", "fire-smoke-v2-dev-key-change-in-prod")
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB max upload limit

# Create required directories automatically
for d in [MODEL_DIR, UPLOAD_FOLDER, LOG_DIR]:
    os.makedirs(d, exist_ok=True)
