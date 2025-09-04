# ------------------------------------------------------------
# YOLOv11 Training Pipeline for Kaggle (Ultralytics + Roboflow)
# ------------------------------------------------------------
# This script trains a custom YOLOv11 model using:
# - Roboflow dataset (via API)
# - Ultralytics YOLO backend
# - Advanced training configurations
# - ONNX export and result visualization
# ------------------------------------------------------------

import os
import yaml
import torch
from roboflow import Roboflow
from ultralytics import YOLO
from PIL import Image
from IPython.display import display

# ------------------------------------------------------------
# Configuration Section
# ------------------------------------------------------------

# Roboflow Dataset Info
ROBOFLOW_API_KEY = ""
ROBOFLOW_WORKSPACE = ""
ROBOFLOW_PROJECT = ""
ROBOFLOW_VERSION = 1
YOLO_FORMAT = "yolov11"  # Custom format name (for clarity)

# Custom YOLOv11 Model
MODEL_VARIANT = "yolo11n.pt"  # Replace with your custom model path

# Training Configuration
EPOCHS = 70
BATCH_SIZE = 32
IMG_SIZE = 640
OPTIMIZER = "AdamW"
INITIAL_LEARNING_RATE = 0.001
FINAL_LEARNING_RATE_FACTOR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 5.0
SAVE_PERIOD = 10
PROJECT_DIR = "runs/train"
EXPERIMENT_NAME = "yolo11_experiment"
SEED = 42

# Augmentation Settings
AUGMENTATION = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.000,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}

# Loss Weights
LOSS_WEIGHTS = {
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
}

# ------------------------------------------------------------
# Device Setup
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cpu":
    print(" WARNING: CUDA not detected. Training will be slower on CPU.")

# ------------------------------------------------------------
# Dataset Download via Roboflow
# ------------------------------------------------------------
print(f"\n Downloading dataset from Roboflow: {ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT} (v{ROBOFLOW_VERSION})")
try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)
    dataset = version.download(YOLO_FORMAT)
    DATA_YAML_PATH = os.path.join(dataset.location, "data.yaml")
    print(f" Dataset downloaded to: {dataset.location}")
except Exception as e:
    print(f" Error downloading dataset: {e}")
    exit()

# ------------------------------------------------------------
# Validate data.yaml
# ------------------------------------------------------------
try:
    with open(DATA_YAML_PATH, 'r') as file:
        data_yaml = yaml.safe_load(file)

    num_classes = data_yaml.get('nc', 0)
    class_names = data_yaml.get('names', [])
    print(f" Number of classes: {num_classes}")
    print(f" Class names: {class_names}")

    if num_classes == 0 or not class_names:
        print(" No classes found in data.yaml. Please check your Roboflow dataset.")
        exit()
except Exception as e:
    print(f" Error reading data.yaml: {e}")
    exit()

# ------------------------------------------------------------
# Load Custom YOLOv11 Model
# ------------------------------------------------------------
print(f"\n Loading YOLOv11 model: {MODEL_VARIANT}")
try:
    model = YOLO(MODEL_VARIANT)
    print(" Model loaded successfully.")
except Exception as e:
    print(f" Error loading model: {e}")
    exit()

# ------------------------------------------------------------
# Start Model Training
# ------------------------------------------------------------
print("\n Starting model training...")
try:
    model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        optimizer=OPTIMIZER,
        lr0=INITIAL_LEARNING_RATE,
        lrf=FINAL_LEARNING_RATE_FACTOR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        save=True,
        save_period=SAVE_PERIOD,
        cache="ram",
        device=DEVICE,
        workers=os.cpu_count(),
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        exist_ok=False,
        pretrained=True,
        verbose=True,
        seed=SEED,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=True,
        close_mosaic=int(EPOCHS * 0.1),
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        freeze=None,
        overlap_mask=False,
        mask_ratio=4,
        dropout=0.0,
        box=LOSS_WEIGHTS["box"],
        cls=LOSS_WEIGHTS["cls"],
        dfl=LOSS_WEIGHTS["dfl"],
        pose=0.0,
        kobj=0.0,
        label_smoothing=0.05,
        **AUGMENTATION,
        patience=10,
    )
    print(" Model training completed.")
except Exception as e:
    print(f" Error during training: {e}")

# ------------------------------------------------------------
# Validation Metrics
# ------------------------------------------------------------
print("\n Starting model validation...")
try:
    metrics = model.val()
    print("\n Validation Metrics:")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP75: {metrics.box.map75:.4f}")
    print(f"   Per-class mAP: {metrics.box.maps}")
except Exception as e:
    print(f" Error during validation: {e}")

# ------------------------------------------------------------
# Visualize Training Results
# ------------------------------------------------------------
print("\n Visualizing training results...")
try:
    result_img_path = os.path.join(PROJECT_DIR, EXPERIMENT_NAME, "results.png")
    if os.path.exists(result_img_path):
        img = Image.open(result_img_path)
        display(img)
    else:
        print(" results.png not found.")
except Exception as e:
    print(f" Error displaying results: {e}")

# ------------------------------------------------------------
# Export Model to ONNX
# ------------------------------------------------------------
print("\n Exporting model to ONNX format...")
try:
    exported_model_path = model.export(format="onnx")
    print(f" Model exported to: {exported_model_path}")
except Exception as e:
    print(f" Error exporting model: {e}")

# ------------------------------------------------------------
# TensorBoard Launch Instructions
# ------------------------------------------------------------
print("\n To view training progress in TensorBoard:")
print("Run in terminal:\n    tensorboard --logdir runs/train")
print("Then open: http://localhost:6006")

print("\n Script completed successfully.")
