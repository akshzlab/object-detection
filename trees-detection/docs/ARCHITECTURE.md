# System Architecture

## Overview

The Tree Detection System is a complete pipeline for training and deploying YOLOv8 models for detecting trees in satellite/drone imagery. The system is organized into logical layers with clear separation of concerns.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Interface / CLI                          │
│  (train_model.py | predict_model.py | prepare_data.py | eval)   │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼──────────────────────────────────────────────────┐
│                    Configuration Layer                            │
│  (tree_data.yaml | Config Loading & Validation)                  │
└─────────────────┬──────────────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┬──────────────┐
    │             │             │              │
┌───▼──┐    ┌─────▼──┐   ┌──────▼─┐   ┌───────▼──┐
│ Data │    │Training│   │Inference│  │Evaluation│
│Prep  │    │Module  │   │ Module  │  │ Module   │
└────┬─┘    └───┬────┘   └───┬─────┘  └────┬─────┘
     │          │            │             │
┌────▼──────────▼────────────▼─────────────▼────┐
│           YOLOv8 Model (Ultralytics)          │
│                                                │
│  ┌──────────────┐         ┌──────────────┐   │
│  │   Backbone   │         │ Detection    │   │
│  │  (CSPDarknet)│────────▶│ Head         │   │
│  │   Feature    │         │ (Predictions)│   │
│  │ Extraction   │         └──────────────┘   │
│  └──────────────┘                            │
└────────────────────────────────────────────────┘
        │              │              │
        │              │              │
   ┌────▼─┐      ┌─────▼──┐     ┌────▼────┐
   │Model │      │Training│     │Inference│
   │Weights  │      │Results │     │Results │
   └────────┘      └─────────┘     └────────┘
```

## Core Modules

### 1. **Data Preparation Layer** (`src/prepare_data.py`)

**Purpose:** Handle dataset organization and validation

**Key Functions:**
- `organize_yolo_format_data()`: Split data into train/val sets
- `validate_yolo_format()`: Verify YOLO format compliance

**Input:** Raw images and annotations
```
data/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── labels/
    ├── img1.txt
    ├── img2.txt
    └── ...
```

**Output:** Organized training data
```
data/processed/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

**YOLO Format Specification:**
```
<class_id> <x_center> <y_center> <width> <height>
# Example:
0 0.5 0.5 0.3 0.4
```

---

### 2. **Training Layer** (`src/train_model.py`)

**Purpose:** Train YOLOv8 model with transfer learning

**Process:**
1. Load pre-trained YOLOv8 weights (n/s/m/l/x sizes available)
2. Load configuration from `config/tree_data.yaml`
3. Validate data structure
4. Initialize training with specified parameters
5. Monitor training progress
6. Save best weights and checkpoints

**Key Parameters:**
- `model_size`: nano(n) → xlarge(x)
- `epochs`: Number of passes through dataset
- `batch_size`: Samples per iteration
- `patience`: Early stopping threshold
- `device`: GPU (0) or CPU

**Output:**
```
runs/detect/tree_model_v1/
├── weights/
│   ├── best.pt      # Best model
│   └── last.pt      # Final checkpoint
├── results.csv      # Metrics per epoch
├── results.png      # Training curves
└── confusion_matrix.png
```

---

### 3. **Inference Layer** (`src/predict_model.py`)

**Purpose:** Run detection on images/videos

**TreeDetector Class:**
- Loads trained model weights
- Handles inference on various sources
- Manages confidence thresholds
- Saves detection results

**Key Methods:**
- `detect(source, conf)`: Run detection
- `detect_and_save(source, output_dir, save_crops)`: Detection with output
- `count_trees(source, conf)`: Count detections

**Input:** Image or video file

**Output:**
```
output/predictions/
├── images/          # Images with bounding boxes
├── crops/          # Cropped tree images (optional)
└── labels/         # Detection coordinates
```

---

### 4. **Evaluation Layer** (`src/evaluate.py`)

**Purpose:** Assess model performance

**ModelEvaluator Class:**
- Loads trained model
- Runs validation on test set
- Computes performance metrics
- Exports results to JSON

**Metrics Generated:**
- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: Average precision across IoU thresholds
- **Precision**: Ratio of correct detections
- **Recall**: Percentage of detected trees
- **Fitness**: Overall performance score

---

### 5. **Utility Layer** (`src/utils.py`)

**Purpose:** Shared utilities across all modules

**Key Functions:**
- `setup_logging()`: Configure logging with file/console handlers
- `load_yaml_config()`: Parse YAML configuration files
- `validate_data_structure()`: Verify YOLO format
- `create_directory_structure()`: Initialize project directories
- `get_absolute_path()`: Path resolution utilities

---

## Data Flow

### Training Pipeline

```
Input Data
    │
    ├─► Preparation (prepare_data.py)
    │   └─► Organize into train/val splits
    │       └─► Validate format
    │
    ├─► Configuration (tree_data.yaml)
    │   └─► Define paths, classes, parameters
    │
    └─► Training (train_model.py)
        ├─► Load pre-trained model
        ├─► Initialize training
        ├─► Train for N epochs
        └─► Save best weights
            └─► Output: best.pt, training curves
```
