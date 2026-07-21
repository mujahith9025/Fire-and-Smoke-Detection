"""
Temporal Video Frame Processor with Rolling Window Smoothing & Real-time Overlay.
"""

import cv2
import time
from collections import deque
import config
from detector.model_engine import detector
from detector.gradcam import make_gradcam_heatmap, overlay_gradcam
from utils.alert_manager import alert_manager

class VideoProcessor:
    def __init__(self, window_size=config.TEMPORAL_WINDOW_SIZE):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.fps_tracker = deque(maxlen=30)
        self.last_frame_time = time.time()

    def process_frame(self, frame_bgr, generate_heatmap=False):
        start_time = time.time()

        if frame_bgr is None or frame_bgr.size == 0:
            return None, {}

        # 1. Single frame prediction
        pred_res = detector.predict_frame(frame_bgr)
        score = pred_res["raw_score"] if pred_res["is_fire"] else (1.0 - pred_res["raw_score"])

        # 2. Temporal rolling average
        self.history.append(score)
        avg_score = sum(self.history) / float(len(self.history))
        smoothed_fire = avg_score > 0.45 or pred_res["is_fire"]

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time + 1e-6)
        self.last_frame_time = current_time
        self.fps_tracker.append(fps)
        avg_fps = sum(self.fps_tracker) / len(self.fps_tracker)

        final_label = "Fire" if smoothed_fire else "No Fire"
        final_conf = min(max(pred_res["confidence"], 60.0), 99.9)

        # Log alert if fire is detected
        if final_label == "Fire":
            alert_manager.log_detection(
                label=final_label,
                confidence=final_conf,
                source="Live Camera Stream",
                metadata={"fps": round(avg_fps, 1)}
            )

        # 3. Draw HUD overlay on output frame
        annotated_frame = frame_bgr.copy()

        if generate_heatmap:
            try:
                resized = cv2.resize(frame_bgr, detector.img_size)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                batch = (rgb.astype("float32") / 255.0)[None, ...]
                heatmap = make_gradcam_heatmap(batch, detector.model)
                annotated_frame = overlay_gradcam(annotated_frame, heatmap, alpha=0.35)
            except Exception:
                pass

        # Banner Header
        color = (0, 0, 255) if final_label == "Fire" else (0, 200, 0)
        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 45), color, -1)

        hud_text = f"STATUS: {final_label.upper()} ({final_conf:.1f}%) | FPS: {avg_fps:.1f}"
        if pred_res.get("face_suppressed"):
            hud_text += " [FACE GUARD ACTIVE]"

        cv2.putText(
            annotated_frame,
            hud_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return annotated_frame, {
            "label": final_label,
            "confidence": round(final_conf, 1),
            "fps": round(avg_fps, 1),
            "is_fire": final_label == "Fire",
            "face_suppressed": pred_res.get("face_suppressed", False)
        }

video_processor = VideoProcessor()
