# YOLOv11 Training Pipeline on Kaggle 🚀

This repository provides a complete training pipeline for a **custom YOLOv11 object detection model**, fully integrated with Roboflow and running on **Kaggle Notebooks** using the **Ultralytics backend** (YOLOv8 framework).

---

## 📍 What is YOLOv11?

YOLOv11 is a **custom extension** or experimental variant of the YOLOv8 architecture. It uses the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) backend but supports custom model variants such as `yolo11n.pt`.

> **Note:** If you're using the official YOLOv8 models (e.g., `yolov8n.pt`), this pipeline will work just as well — just rename accordingly.

---

## ✅ Features

- 📥 Roboflow dataset integration via API
- 🧠 Custom model loading (`yolo11n.pt`, etc.)
- ⚙️ Full control over training hyperparameters and augmentation
- 🧪 Auto evaluation (mAP50, mAP75, etc.)
- 🖼️ Training results visualization (plots, metrics)
- 🔄 ONNX export support
- 🧠 Runs on Kaggle with GPU acceleration

---

## 📦 Requirements

No installation is needed locally.

All dependencies will be installed **inside the Kaggle Notebook**:

```python
!pip install -q roboflow ultralytics opencv-python-headless pillow pyyaml
