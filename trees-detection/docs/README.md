# Tree Detection System (YOLOv8)

## 1. Project Overview
This project implements a Convolutional Neural Network (CNN) based object detection system using the YOLOv8 (You Only Look Once) architecture to identify and localize trees in satellite/drone imagery.

## 2. Theoretical Background

### CNN (Feature Extraction)
The backbone of this model is a Deep CNN (CSPDarknet). It processes the image through successive layers of convolution to extract:
- **Low-level features:** Edges, colors, gradients.
- **High-level features:** Foliage patterns, canopy shapes.

### YOLO Architecture (Detection Head)
Unlike two-stage detectors (like Faster R-CNN) which are slow, YOLO divides the input image into a grid. For every grid cell, it simultaneously predicts:
1. **Bounding Box Coordinates:** (x, y, width, height)
2. **Objectness Score:** Probability that an object exists.
3. **Class Probability:** Probability the object is a 'tree'.

## 3. Project Structure

```
trees-detection/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── config/
│   └── tree_data.yaml            # YOLO dataset configuration
├── data/
│   ├── raw/                      # Original raw images
│   ├── processed/                # Processed YOLO-format data
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── val/
│   │       ├── images/
│   │       └── labels/
│   ├── images/                   # Working directory for images
│   └── labels/                   # Working directory for labels
├── src/
│   ├── train_model.py            # Training script
│   ├── predict_model.py          # Inference script
│   ├── prepare_data.py           # Data preparation utilities
│   ├── evaluate.py               # Model evaluation script
│   └── utils.py                  # Utility functions
├── models/                       # Saved model weights
└── runs/                         # Training outputs and results
    └── detect/
        └── tree_model_v1/        # Named training run
            ├── weights/
            │   ├── best.pt       # Best model weights
            │   └── last.pt       # Last checkpoint
            └── results.png       # Training curves
```

## 4. Setup

### 4.1 Installation

1. **Clone the repository:**
   ```bash
   cd trees-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "from ultralytics import YOLO; print('YOLO imported successfully')"
   ```

### 4.2 Data Preparation

The dataset must follow YOLO format:
- Each image has a corresponding `.txt` annotation file
- Annotation file contains one line per object: `<class_id> <x_center> <y_center> <width> <height>`
- All coordinates are normalized (0-1)

Example annotation file (`.txt`):
```
0 0.5 0.5 0.3 0.4    # class=0(tree), center at (0.5,0.5), width=0.3, height=0.4
```

### 4.3 Organize Your Data

Place your images and labels in the following structure:
```
data/
├── images/          # All images (*.jpg, *.png)
└── labels/          # All annotations (*.txt)
```

Then organize into train/val splits:
```bash
python src/prepare_data.py organize data/images data/labels --output data/processed --train-ratio 0.8
```

### 4.4 Verify Data Structure

```bash
python src/prepare_data.py validate data/processed
```

## 5. Usage

### 5.1 Training

**Basic training (nano model, 100 epochs):**
```bash
python src/train_model.py
```

**Advanced training with custom parameters:**
```bash
python src/train_model.py \
    --config config/tree_data.yaml \
    --model m \
    --epochs 200 \
    --batch 32 \
    --patience 30 \
    --device 0
```

**Training parameters:**
- `--config`: Path to YOLO configuration file (default: config/tree_data.yaml)
- `--model`: Model size: n(nano), s(small), m(medium), l(large), x(xlarge) (default: n)
- `--epochs`: Number of training epochs (default: 100)
- `--batch`: Batch size (default: 16)
- `--patience`: Early stopping patience in epochs (default: 20)
- `--device`: GPU device ID or 'cpu' (default: 0)

**Model sizes comparison:**
| Size | Parameters | Speed | mAP50 |
|------|-----------|-------|-------|
| Nano | 2.6M | Fast | Lower |
| Small | 8.2M | Medium | Medium |
| Medium | 25.9M | Medium | Better |
| Large | 62.6M | Slow | Good |
| XLarge | 106.3M | Very Slow | Best |

### 5.2 Inference (Detection)

**Detect trees in a single image:**
```bash
python src/predict_model.py path/to/image.jpg \
    --model runs/detect/tree_model_v1/weights/best.pt
```

**Detect and save results:**
```bash
python src/predict_model.py path/to/image.jpg \
    --model runs/detect/tree_model_v1/weights/best.pt \
    --output results \
    --save-crops
```

**Count trees (text output only):**
```bash
python src/predict_model.py path/to/image.jpg \
    --model runs/detect/tree_model_v1/weights/best.pt \
    --count-only
```

**Inference parameters:**
- `source`: Path to image or video file
- `--model`: Path to trained model weights (required)
- `--conf`: Confidence threshold 0-1 (default: 0.25)
- `--output`: Output directory for results (default: output)
- `--save-crops`: Save cropped tree images
- `--count-only`: Only count trees, don't save visualizations

### 5.3 Model Evaluation

**Evaluate on validation set:**
```bash
python src/evaluate.py \
    --model runs/detect/tree_model_v1/weights/best.pt \
    --config config/tree_data.yaml
```

**Output metrics:**
- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: Mean Average Precision at IoU=0.5:0.95
- **Precision**: How many detections were correct
- **Recall**: What percentage of actual trees were found
- **Fitness**: Overall model performance score

## 6. Training Pipeline

### Step 1: Prepare Data
```bash
python src/prepare_data.py organize \
    data/images \
    data/labels \
    --output data/processed \
    --train-ratio 0.8
```

### Step 2: Validate Data

```bash
python src/prepare_data.py validate data/processed
```

### Step 3: Train Model

```bash
python src/train_model.py --model m --epochs 100
```

### Step 4: Evaluate Model

```bash
python src/evaluate.py \
    --model runs/detect/tree_model_v1/weights/best.pt \
    --config config/tree_data.yaml
```

### Step 5: Run Inference

```bash
python src/predict_model.py path/to/test_image.jpg \
    --model runs/detect/tree_model_v1/weights/best.pt
```

## 7. Output and Results

### Training Outputs
After training, results are saved in `runs/detect/tree_model_v1/`:
- `weights/best.pt` - Best model weights
- `weights/last.pt` - Last epoch checkpoint
- `results.png` - Training curves
- `confusion_matrix.png` - Confusion matrix visualization
- `results.csv` - Detailed metrics per epoch

### Inference Outputs
After inference, results are saved in the output directory:
- `predictions/` - Main output folder
  - `images/` - Images with bounding boxes drawn
  - `crops/` - Individual cropped tree images (if --save-crops)

## 8. Important Notes

### Dataset Requirements
- **Minimum:** 100-200 annotated images (but 500+ recommended)
- **Format:** YOLO format (.txt annotations)
- **Resolution:** 640x640 or higher recommended
- **Diversity:** Images from various lighting, seasons, and conditions

### Hardware Requirements
- **CPU only:** Very slow (days for training)
- **GPU (NVIDIA):** 6GB VRAM minimum, 12GB+ recommended
- **GPU (other):** May require CUDA compatibility

### Model Selection
- **Nano/Small:** Quick training, lower accuracy, deployment on edge devices
- **Medium/Large:** Balanced accuracy and speed
- **XLarge:** Best accuracy, requires more GPU memory and training time

### Preventing Overfitting
- Use data augmentation (handled by YOLOv8)
- Use appropriate patience (early stopping)
- Ensure diverse training data
- Use appropriate batch size
- Monitor validation metrics

## 9. Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
python src/train_model.py --batch 8
```

### No GPU Detected
Use CPU (slower):
```bash
python src/train_model.py --device cpu
```

### Model Not Found
Ensure the model path is correct and file exists:
```bash
ls -la runs/detect/tree_model_v1/weights/
```

### Data Format Issues
Validate data structure:
```bash
python src/prepare_data.py validate data/processed
```

## 10. Next Steps

1. **Find annotated dataset:**
   - NEON Tree Evaluation Benchmark
   - Kaggle forestry datasets
   - CVAT or Roboflow for manual annotation

2. **Prepare data in YOLO format**

3. **Train model with appropriate hardware**

4. **Evaluate and iterate on model**

5. **Deploy for inference

## 11. References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLOv8 Training Guide](https://docs.ultralytics.com/modes/train/)
- [YOLO Format Specification](https://docs.ultralytics.com/datasets/detect/)
- [Tree Detection Research](https://github.com/weecology/NeonTreeEvaluation)

http://googleusercontent.com/youtube_content/0 *YouTube video views will be stored in your YouTube History, and your data will be stored and used by YouTube according to its [Terms of Service](https://www.youtube.com/static?template=terms)*
