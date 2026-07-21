# 🔥 Fire & Smoke Detection System (v2 Intermediate Level)

A modular, production-ready computer vision application featuring **MobileNetV2 Transfer Learning**, **Explainable AI (Grad-CAM Visual Heatmaps)**, **Temporal Video Frame-Smoothing**, **RESTful API**, and a modern **Dark Glassmorphism Web Dashboard**.

---

## 🌟 What's New in Version 2 (Intermediate Upgrade)

| Feature | Version 1 (Beginner) | Version 2 (Intermediate Upgraded) |
|---|---|---|
| **Architecture** | Monolithic script (`app.py`) | Modular package design (`detector/`, `api/`, `utils/`, `tests/`) |
| **Model** | Custom 3-layer CNN | **MobileNetV2 Transfer Learning Backbone** + HSV Spectrum Verification |
| **Explainable AI** | Text prediction label only | **Grad-CAM Visual Heatmaps** highlighting flame & smoke attention regions |
| **Video Engine** | Instant frame prediction | **Temporal Window Smoothing** (rolling average buffer) & FPS counter |
| **Alert System** | Basic terminal print | Event logger, Web Audio alarm synthesizer, and CSV report exporter |
| **API** | HTML Form post | **RESTful JSON API** (`/api/v1/predict/image`, `/api/v1/history`, `/api/v1/health`) |
| **Web Dashboard** | Basic page | Glassmorphism Dark Mode UI with interactive Inspector, Live HUD, Event Logs & Model Analytics |
| **Testing & DevOps** | None | **Pytest unit test suite**, `Dockerfile`, and `docker-compose.yml` |

---

## 📁 Upgraded Directory Structure

```
Fire_And_Smoke_Detection_v2/
├── app.py                          # Entrypoint: Flask Web App & RESTful API routes
├── config.py                       # Centralized configuration (thresholds, model paths, logging)
├── requirements.txt                # Dependency specifications
├── Dockerfile                      # Docker container definition
├── docker-compose.yml              # Docker Compose orchestration
├── README.md                       # Documentation & API Reference
├── Fire_and_Smoke_Detection_v2.ipynb # Notebook: Transfer Learning, Grad-CAM, & evaluation curves
│
├── detector/                       # Core Computer Vision & Deep Learning Engine
│   ├── __init__.py
│   ├── model_engine.py             # MobileNetV2 backbone, inference, face suppression
│   ├── gradcam.py                  # Grad-CAM Explainable AI heatmap overlay generator
│   └── video_processor.py          # Temporal rolling window, HUD renderer, alert triggers
│
├── utils/                          # Utilities & Event Logging
│   ├── __init__.py
│   ├── logger.py                   # Structured logging setup
│   └── alert_manager.py            # Event history manager, alert notification hooks, CSV export
│
├── templates/                      # UI Views
│   └── index.html                  # Responsive Dashboard layout with tabbed navigation
│
├── static/                         # Assets & Media
│   ├── css/
│   │   └── style.css               # Glassmorphism dark CSS design system
│   ├── js/
│   │   └── dashboard.js            # Interactive client JS, canvas feed, audio alarm synth
│   └── uploads/                    # Uploaded & generated Grad-CAM images
│
└── tests/                          # Automated Testing Suite
    ├── __init__.py
    ├── test_detector.py            # Unit tests for model engine & Grad-CAM
    └── test_api.py                 # Integration tests for REST API endpoints
```

---

## 🚀 Quickstart Guide

### 1. Local Setup
```bash
# Navigate to the new upgraded project folder
cd c:\Users\Mujahith\Downloads\Fire_And_Smoke_Detection_v2

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```
Open **http://127.0.0.1:5000** in your web browser.

---

### 2. Run Automated Unit Tests
```bash
pytest
```

---

### 3. Docker Deployment
```bash
docker-compose up --build
```

---

## 📡 RESTful API Documentation (v1)

### `GET /api/v1/health`
Checks backend engine status and loaded model parameters.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "model_loaded": true,
  "input_size": [224, 224]
}
```

### `POST /api/v1/predict/image`
Uploads an image for fire classification and generates a Grad-CAM heatmap overlay.

**Form Data:** `file` (Image file)

**Response:**
```json
{
  "status": "success",
  "result": {
    "label": "Fire",
    "confidence": 98.4,
    "is_fire": true,
    "image_url": "/static/uploads/uuid_image.jpg",
    "gradcam_url": "/static/uploads/gradcam_uuid_image.jpg",
    "hsv_fire_ratio": 0.185,
    "face_suppressed": false
  }
}
```

### `GET /api/v1/history?label=all`
Retrieves logged detection event history.

### `GET /api/v1/export/csv`
Downloads a CSV report of all logged alerts.

---

## 🔬 Explainable AI (Grad-CAM)

The `detector/gradcam.py` module uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to calculate visual heatmaps over input images. Heatmaps highlight specific spatial regions (bright red/orange spots) that triggered the neural network's fire or smoke prediction, enabling transparent model auditing.
