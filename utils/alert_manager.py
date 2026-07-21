"""
Alert Event Manager and Detection History Logger
"""

import json
import os
import csv
import time
from datetime import datetime
import config
from utils.logger import logger

class AlertManager:
    def __init__(self, history_file=config.HISTORY_FILE):
        self.history_file = history_file
        self.events = self._load_history()
        self.last_alert_time = 0

    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading detection history: {e}")
                return []
        return []

    def _save_history(self):
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.events, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving detection history: {e}")

    def log_detection(self, label, confidence, source="Image Upload", image_url=None, metadata=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        is_alert = label in ["Fire", "Smoke"]

        event = {
            "id": len(self.events) + 1,
            "timestamp": timestamp,
            "label": label,
            "confidence": round(confidence, 2),
            "source": source,
            "image_url": image_url or "",
            "is_alert": is_alert,
            "metadata": metadata or {}
        }

        self.events.insert(0, event)  # newest first
        if len(self.events) > 500:     # keep last 500 records
            self.events = self.events[:500]

        self._save_history()

        if is_alert:
            current_time = time.time()
            if current_time - self.last_alert_time >= config.ALERT_COOLDOWN_SECONDS:
                self.last_alert_time = current_time
                logger.warning(f"ALERT TRIGGERED: {label} detected ({confidence:.1f}%) via {source}")
                self._trigger_alert_notification(event)

        return event

    def _trigger_alert_notification(self, event):
        """Hook for external notifications (SMS, Email, Webhook)."""
        logger.info(f"Notification dispatched for event #{event['id']} [{event['label']}]")

    def get_history(self, limit=50, label_filter=None):
        if not label_filter or label_filter.lower() == "all":
            return self.events[:limit]
        return [e for e in self.events if e["label"].lower() == label_filter.lower()][:limit]

    def export_csv(self, output_path):
        fieldnames = ["id", "timestamp", "label", "confidence", "source", "is_alert", "image_url"]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for event in self.events:
                row = {k: event.get(k, "") for k in fieldnames}
                writer.writerow(row)
        return output_path

alert_manager = AlertManager()
