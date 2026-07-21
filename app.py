"""
Fire & Smoke Detection System (v2) - Flask Application & RESTful API Engine
"""

import os
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response, send_file
from werkzeug.utils import secure_filename

import config
from detector.model_engine import detector
from detector.gradcam import make_gradcam_heatmap, overlay_gradcam
from detector.video_processor import video_processor
from utils.alert_manager import alert_manager
from utils.logger import logger

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.secret_key = config.SECRET_KEY

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------------------------------------------------------------------
# Web Views
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    """MJPEG Live Webcam Video Stream with temporal HUD overlay."""
    def generate():
        cap = cv2.VideoCapture(0)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame, stats = video_processor.process_frame(frame, generate_heatmap=False)
                if annotated_frame is None:
                    continue

                ok, buffer = cv2.imencode(".jpg", annotated_frame)
                if not ok:
                    continue
                
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        finally:
            cap.release()

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ------------------------------------------------------------------------------
# RESTful API Endpoints (v1)
# ------------------------------------------------------------------------------
@app.route("/api/v1/health", methods=["GET"])
def api_health():
    return jsonify({
        "status": "healthy",
        "version": "2.0.0",
        "model_loaded": detector.model is not None,
        "input_size": list(detector.img_size)
    })

@app.route("/api/v1/predict/image", methods=["POST"])
def api_predict_image():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file parameter in request"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid or missing file format"}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    try:
        # 1. Standard prediction
        res = detector.predict_image_path(save_path)

        # 2. Grad-CAM visual heatmap generation
        frame_bgr = cv2.imread(save_path)
        resized = cv2.resize(frame_bgr, detector.img_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        batch = (rgb.astype("float32") / 255.0)[None, ...]

        heatmap = make_gradcam_heatmap(batch, detector.model)
        superimposed = overlay_gradcam(frame_bgr, heatmap, alpha=0.4)

        gradcam_filename = f"gradcam_{unique_name}"
        gradcam_save_path = os.path.join(app.config["UPLOAD_FOLDER"], gradcam_filename)
        cv2.imwrite(gradcam_save_path, superimposed)

        image_url = url_for("static", filename=f"uploads/{unique_name}")
        gradcam_url = url_for("static", filename=f"uploads/{gradcam_filename}")

        res["image_url"] = image_url
        res["gradcam_url"] = gradcam_url

        # Log event to history
        alert_manager.log_detection(
            label=res["label"],
            confidence=res["confidence"],
            source="API Image Upload",
            image_url=image_url,
            metadata={"hsv_score": res["hsv_fire_ratio"]}
        )

        return jsonify({
            "status": "success",
            "result": res
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/v1/history", methods=["GET"])
def api_history():
    label_filter = request.args.get("label", "all")
    events = alert_manager.get_history(limit=100, label_filter=label_filter)
    return jsonify({"status": "success", "count": len(events), "history": events})


@app.route("/api/v1/export/csv", methods=["GET"])
def api_export_csv():
    csv_path = os.path.join(config.LOG_DIR, "detection_report.csv")
    alert_manager.export_csv(csv_path)
    return send_file(csv_path, as_attachment=True, download_name="detection_report.csv")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Fire & Smoke Detection System v2 on port {port}")
    app.run(debug=True, host="0.0.0.0", port=port)
