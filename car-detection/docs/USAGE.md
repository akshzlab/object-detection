# Car Detection - Usage Guide

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda
- GPU with CUDA support (recommended, but CPU works too)

### Quick Setup

```bash
cd car-detection
bash setup.sh
```

This will:
1. Install all dependencies
2. Create necessary directories
3. Download YOLOv8 models

### Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/{train,val,test}/{images,labels}
mkdir -p models runs/detect

# Download models
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Step 1: Prepare Your Data

### Using a Single Directory with Images and Labels

```bash
python src/prepare_data.py \
  --source /path/to/images \
  --output data \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --validate
```

### Using Separate Image and Label Directories

```bash
python src/prepare_data.py \
  --images /path/to/images \
  --labels /path/to/labels \
  --output data
```

### Expected Directory Structure After Preparation

```
data/
├── train/
│   ├── images/
│   │   ├── car_001.jpg
│   │   ├── car_002.jpg
│   │   └── ...
│   └── labels/
│       ├── car_001.txt
│       ├── car_002.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/
    └── images/
```

### Label Format

Each image must have a corresponding `.txt` file with the same name:

```
0 0.450 0.550 0.300 0.400
0 0.650 0.350 0.250 0.350
```

Format: `<class_id> <x_center> <y_center> <width> <height>`
- class_id: 0 (car)
- Coordinates: normalized to 0-1

## Step 2: Train the Model

### Basic Training

```bash
python src/train_model.py \
  --config config/car_data.yaml \
  --epochs 100
```

### Advanced Training with Custom Parameters

```bash
python src/train_model.py \
  --config config/car_data.yaml \
  --model yolov8m.pt \
  --epochs 150 \
  --batch-size 32 \
  --imgsz 800 \
  --device 0 \
  --patience 30
```

### Training Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| --config | config/car_data.yaml | Path to data config |
| --model | yolov8n.pt | Model to use (n/s/m/l/x) |
| --epochs | 100 | Number of epochs to train |
| --batch-size | 16 | Batch size |
| --imgsz | 640 | Training image size |
| --device | 0 | GPU device (0 for GPU, -1 for CPU) |
| --patience | 20 | Early stopping patience |
| --name | car_detection | Experiment name |
| --resume | - | Resume from checkpoint |

### Monitoring Training

Training results are saved in `runs/detect/car_detection/`:
- View metrics in the Results tab of VS Code
- Check plots in `runs/detect/car_detection/` directory
- Monitor console output for real-time metrics

### Expected Output

```
Epoch 1/100
train/loss: 2.456, val/loss: 2.123, mAP50: 0.234
Epoch 2/100
train/loss: 2.123, val/loss: 1.987, mAP50: 0.456
...
Best model saved: runs/detect/car_detection/weights/best.pt
```

## Step 3: Evaluate the Model

### Evaluate on Validation Set

```bash
python src/evaluate.py \
  --model runs/detect/car_detection/weights/best.pt \
  --data config/car_data.yaml
```

### Generate Training Curves

```bash
python src/evaluate.py \
  --model runs/detect/car_detection/weights/best.pt \
  --data config/car_data.yaml \
  --plot-training runs/detect/car_detection
```

### Interpret Metrics

- **mAP50**: Mean Average Precision at IoU=0.5 (box must have >50% overlap)
  - 0.9+: Excellent
  - 0.8-0.9: Very Good
  - 0.7-0.8: Good
  - <0.7: Needs improvement

- **mAP50-95**: Mean Average Precision across all IoU thresholds
  - More strict metric
  - Better indicator of overall model quality

## Step 4: Make Predictions

### Predict on Single Image

```bash
python src/predict_model.py \
  --model runs/detect/car_detection/weights/best.pt \
  --source /path/to/car.jpg \
  --output predictions
```

### Predict on Directory of Images

```bash
python src/predict_model.py \
  --model runs/detect/car_detection/weights/best.pt \
  --source data/test/images \
  --output predictions
```

### Adjust Confidence Threshold

```bash
python src/predict_model.py \
  --model runs/detect/car_detection/weights/best.pt \
  --source /path/to/images \
  --conf 0.6
```

### Output

- Annotated images saved to `predictions/`
- Bounding boxes with confidence scores
- Console output with detection counts

## Using in Your Code

### Python API

```python
from src.predict_model import CarDetector
import cv2

# Initialize detector
detector = CarDetector('runs/detect/car_detection/weights/best.pt', conf_threshold=0.5)

# Load image
image = cv2.imread('car.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect
annotated, detections = detector.detect(image)

# Process detections
for det in detections:
    x1, y1, x2, y2 = det['box']
    conf = det['conf']
    print(f"Car detected: confidence={conf:.2f}, box=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
```

### Batch Processing

```python
from src.predict_model import CarDetector
from src.utils import get_image_files, load_image

detector = CarDetector('best.pt')
image_files = get_image_files('data/test/images')

for img_path in image_files:
    image = load_image(img_path)
    annotated, detections = detector.detect(image)
    print(f"{img_path}: {len(detections)} cars")
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 8`
- Reduce image size: `--imgsz 512`
- Use smaller model: `--model yolov8n.pt`
- Use CPU: `--device -1`

### Low Accuracy
- Check data quality (clear images, proper labels)
- Increase training epochs: `--epochs 200`
- Use larger model: `--model yolov8l.pt`
- Increase image size: `--imgsz 1024`
- More training data needed

### Training Too Slow
- Use smaller model: `--model yolov8n.pt`
- Reduce image size: `--imgsz 512`
- Increase batch size: `--batch-size 32`
- Use GPU: `--device 0`

### Model Not Detecting Anything
- Lower confidence threshold: `--conf 0.3`
- Check if model is trained (mAP > 0)
- Verify input image format
- Check label format in training data

## Performance Tips

### For Production
1. Use YOLOv8m or YOLOv8l for better accuracy
2. Train for 150+ epochs
3. Use 800x800+ image size
4. Collect at least 100-200 diverse images

### For Real-time Applications
1. Use YOLOv8n or YOLOv8s for speed
2. Reduce image size to 416-640
3. Use smaller batch sizes during inference
4. Implement frame skipping

### For Edge Deployment
1. Export to ONNX: `model.export(format='onnx')`
2. Use TensorRT for NVIDIA devices
3. Quantize model for smaller size
4. Consider model distillation

## Next Steps

1. Prepare your car detection dataset
2. Run setup script
3. Train with default parameters
4. Evaluate results
5. Fine-tune hyperparameters
6. Deploy for production use

For more information, see:
- `docs/ARCHITECTURE.md` - System design
- `docs/CONFIG_REFERENCE.md` - Configuration options
- `docs/README.md` - Documentation index
