"""
Core Deep Learning Model Engine for Fire and Smoke Detection.
Supports Transfer Learning (MobileNetV2), Legacy CNN models, and HSV Fallback Verification.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

import config
from utils.logger import logger

class FireSmokeDetector:
    def __init__(self, model_path=config.MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.img_size = config.IMG_SIZE
        self.face_cascade = None
        
        # Safe face cascade initialization for headless Linux environments
        try:
            if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades") and cv2.data.haarcascades:
                cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                if os.path.exists(cascade_path):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            logger.warning(f"Face cascade initialization notice: {e}")

        self._load_or_build_model()

    def _load_or_build_model(self):
        """Loads existing model from disk or builds a lightweight transfer-learning model backbone."""
        target_files = [
            config.MODEL_PATH,
            config.LEGACY_MODEL_PATH,
            os.path.join(config.BASE_DIR, "fire_smoke_model.h5"),
            os.path.join(config.BASE_DIR, "models", "fire_smoke_model.h5")
        ]

        for target_file in target_files:
            if os.path.exists(target_file):
                try:
                    self.model = load_model(target_file)
                    logger.info(f"Loaded trained model from {target_file}")
                    # Infer input size from model
                    input_shape = self.model.input_shape
                    if input_shape and len(input_shape) == 4 and input_shape[1] is not None:
                        self.img_size = (input_shape[1], input_shape[2])
                        logger.info(f"Inferred model input dimension: {self.img_size}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load model from {target_file}: {e}")

        logger.info("Initializing MobileNetV2 transfer-learning feature extractor backbone...")
        self.model = self._build_mobilenet_backbone()

    def _build_mobilenet_backbone(self):
        """Constructs a default MobileNetV2 architecture with pretrained ImageNet weights."""
        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], 3)
        )
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], 3))
        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="fire_output")(x)

        model = tf.keras.Model(inputs, outputs, name="FireSmoke_MobileNetV2")
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def has_face(self, frame_bgr):
        """Detects human faces to prevent warm lighting false alerts."""
        if not config.SUPPRESS_FIRE_ON_FACE or self.face_cascade is None or self.face_cascade.empty():
            return False
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(30, 30)
            )
            return len(faces) > 0
        except Exception:
            return False

    def compute_multispectral_fire_score(self, frame_bgr):
        """
        Multi-Spectral Color-Space Fire Analyzer.
        Combines HSV (Hue, Saturation, Value) and YCrCb (Luminance, Chrominance)
        to identify true high-intensity flame pixels.
        """
        # 1. HSV Mask
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower_orange_red = np.array([0, 125, 125])
        upper_orange_red = np.array([25, 255, 255])
        lower_red_wrap = np.array([160, 125, 125])
        upper_red_wrap = np.array([180, 255, 255])

        mask_hsv1 = cv2.inRange(hsv, lower_orange_red, upper_orange_red)
        mask_hsv2 = cv2.inRange(hsv, lower_red_wrap, upper_red_wrap)
        mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

        # 2. YCrCb Mask (Y > 130, Cr > 145, Cb < 120, Cr > Cb)
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:, :, 0]
        Cr = ycrcb[:, :, 1]
        Cb = ycrcb[:, :, 2]

        mask_ycrcb = (Y > 130) & (Cr > 145) & (Cb < 120) & (Cr > Cb)

        # 3. Combined Multi-Spectral Fire Mask
        combined_mask = cv2.bitwise_and(mask_hsv, (mask_ycrcb * 255).astype(np.uint8))

        total_pixels = frame_bgr.shape[0] * frame_bgr.shape[1]
        fire_pixels = cv2.countNonZero(combined_mask)
        return fire_pixels / float(total_pixels)

    def is_sunset_or_landscape(self, frame_bgr, flame_ratio):
        """
        Detects if warm colors belong to a sunset/sunrise landscape or daytime sun glare.
        - Sunsets: Warm sky gradients in the top half of the frame.
        - Sun glare / Daytime Landscapes: Very low flame ratio (< 1.2%) with high upper-sky brightness.
        """
        height, width = frame_bgr.shape[:2]
        top_half = frame_bgr[0:int(height * 0.5), :]
        bottom_half = frame_bgr[int(height * 0.5):, :]

        # 1. Sun Glare / Daytime Sun over Landscape Guard (e.g. bright morning sun on horizon)
        if flame_ratio < 0.015:
            # Check upper sky brightness
            gray_top = cv2.cvtColor(top_half, cv2.COLOR_BGR2GRAY)
            bright_sky_pixels = np.count_nonzero(gray_top > 200) / float(gray_top.size)
            if bright_sky_pixels > 0.05:  # Sun glare or bright sky on horizon
                return True

        if flame_ratio < 0.03:
            return False

        top_ratio = self.compute_multispectral_fire_score(top_half)
        bottom_ratio = self.compute_multispectral_fire_score(bottom_half)

        # Sunsets have sky gradients on top and lower flame content on bottom
        if top_ratio > (flame_ratio * 1.3) and bottom_ratio < 0.05:
            return True
        return False

    def predict_frame(self, frame_bgr):
        """
        Advanced Ensemble Inference Engine:
        Fuses Deep Learning Model Predictions + Multi-Spectral Color Analytics + Spatial Glare Guards.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Invalid frame provided for prediction.")

        # Compute multi-spectral metrics
        has_face_flag = self.has_face(frame_bgr)
        flame_ratio = self.compute_multispectral_fire_score(frame_bgr)
        is_sunset_flag = self.is_sunset_or_landscape(frame_bgr, flame_ratio)

        # 1. Face Guard
        if has_face_flag and flame_ratio < 0.05:
            return {
                "label": "No Fire",
                "confidence": 99.0,
                "raw_score": 0.99,
                "face_suppressed": True,
                "sunset_guard": False,
                "hsv_fire_ratio": round(flame_ratio, 4),
                "is_fire": False
            }

        # 2. Deep Neural Network Prediction
        resized = cv2.resize(frame_bgr, self.img_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        batch = np.expand_dims(rgb.astype("float32") / 255.0, axis=0)

        raw_pred = float(self.model.predict(batch, verbose=0)[0][0])

        # 3. Ensemble Fusion Decision Matrix:
        # Note: In trained model, pred <= 0.35 is "Fire", pred > 0.35 is "No Fire"
        model_says_fire = raw_pred <= config.FIRE_THRESHOLD
        verified_flame_pixels = flame_ratio >= 0.01  # Requires at least 1% verified multi-spectral flame pixels

        # Sun Glare / Landscape Override: If sky glare guard triggered or virtually 0% flame pixels (<1%)
        if is_sunset_flag or flame_ratio < 0.01:
            label = "No Fire"
            is_fire = False
            confidence = min(max((1.0 - flame_ratio) * 100, 85.0), 99.9)
        elif model_says_fire and verified_flame_pixels:
            label = "Fire"
            is_fire = True
            confidence = (1.0 - raw_pred) * 100
            confidence = min(max(confidence, 78.0), 99.9)
        elif flame_ratio > 0.035 and not is_sunset_flag:
            label = "Fire"
            is_fire = True
            confidence = min(82.0 + flame_ratio * 80.0, 99.5)
        else:
            label = "No Fire"
            is_fire = False
            confidence = max(raw_pred * 100, (1.0 - flame_ratio) * 100)
            confidence = min(max(confidence, 75.0), 99.9)

        return {
            "label": label,
            "confidence": round(float(confidence), 2),
            "raw_score": round(raw_pred, 4),
            "face_suppressed": False,
            "sunset_guard": is_sunset_flag,
            "hsv_fire_ratio": round(flame_ratio, 4),
            "is_fire": is_fire
        }

    def predict_image_path(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not load image at {image_path}")
        return self.predict_frame(frame)

detector = FireSmokeDetector()
