# Car Detection - Configuration Reference

## Data Configuration (car_data.yaml)

### Required Fields

```yaml
path: ../data                    # Root data directory (relative to config file)
train: train/images              # Training images directory
val: val/images                  # Validation images directory
test: test/images                # Test images directory (optional)

nc: 1                            # Number of classes
names: ['car']                   # Class names list
```

### Example Full Configuration

```yaml
# Dataset paths
path: ../data
train: train/images
val: val/images
test: test/images

# Classes
nc: 1
names: ['car']

# Optional: Class colors for visualization
colors:
  - [255, 0, 0]  # Blue for car (BGR format)
```

### Path Explanation

- **Relative Paths**: Relative to the YAML file location
  - `config/car_data.yaml` with `path: ../data` points to `data/`
  
- **Absolute Paths**: Use full path (e.g., `/home/user/data`)

## Training Configuration

### Command Line Arguments

```bash
python src/train_model.py [options]
```

#### Model Selection
```
--model {yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt}
```

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolov8n.pt | 3.2MB | Very Fast | Fair |
| yolov8s.pt | 11.2MB | Fast | Good |
| yolov8m.pt | 26.2MB | Medium | Better |
| yolov8l.pt | 52.3MB | Slow | Very Good |
| yolov8x.pt | 107MB | Very Slow | Excellent |

#### Training Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| epochs | 100 | 50-300 | Training epochs |
| batch-size | 16 | 1-128 | Batch size (depends on GPU) |
| imgsz | 640 | 320-1280 | Training image size |
| device | 0 | 0-N / -1 | GPU device ID or -1 for CPU |
| project | runs/detect | - | Output directory |
| patience | 20 | 5-50 | Early stopping patience |
| lr0 | 0.01 | 0.0001-0.1 | Initial learning rate |
| momentum | 0.937 | 0.0-1.0 | SGD momentum |

#### Data Augmentation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| hsv_h | 0.015 | HSV hue augmentation |
| hsv_s | 0.7 | HSV saturation augmentation |
| hsv_v | 0.4 | HSV value augmentation |
| degrees | 10 | Rotation augmentation (±degrees) |
| translate | 0.1 | Translation augmentation (fraction) |
| scale | 0.5 | Scale augmentation |
| flipud | 0.5 | Vertical flip probability |
| fliplr | 0.5 | Horizontal flip probability |

### Recommended Configurations

#### Quick Testing
```bash
python src/train_model.py \
  --config config/car_data.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --batch-size 16 \
  --imgsz 640
```

#### Standard Training
```bash
python src/train_model.py \
  --config config/car_data.yaml \
  --model yolov8m.pt \
  --epochs 100 \
  --batch-size 32 \
  --imgsz 800
```

#### High Accuracy
```bash
python src/train_model.py \
  --config config/car_data.yaml \
  --model yolov8l.pt \
  --epochs 200 \
  --batch-size 32 \
  --imgsz 1024 \
  --patience 30
```

#### High Speed
```bash
python src/train_model.py \
  --config config/car_data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --batch-size 64 \
  --imgsz 416
```

## Prediction Configuration

### Command Line Arguments

```bash
python src/predict_model.py [options]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| --model | Yes | - | Path to trained model |
| --source | Yes | - | Image or directory path |
| --output | No | predictions | Output directory |
| --conf | No | 0.5 | Confidence threshold (0-1) |

### Confidence Threshold Guide

- **0.3**: Very permissive (more false positives)
- **0.5**: Default (balanced)
- **0.7**: More strict (fewer false positives)
- **0.9**: Very strict (only high confidence detections)

## Data Preparation Configuration

### Command Line Arguments

```bash
python src/prepare_data.py [options]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --source | path | - | Source directory |
| --images | path | - | Images directory |
| --labels | path | - | Labels directory |
| --output | path | data | Output directory |
| --train-ratio | float | 0.8 | Training split ratio |
| --val-ratio | float | 0.1 | Validation split ratio |
| --validate | flag | False | Validate after prep |

### Data Split Ratios

| Train | Val | Test | Use Case |
|-------|-----|------|----------|
| 0.8 | 0.1 | 0.1 | Standard |
| 0.7 | 0.2 | 0.1 | Small dataset |
| 0.9 | 0.1 | 0.0 | Large dataset |
| 0.6 | 0.2 | 0.2 | Testing emphasis |

## Environment Variables

### GPU Configuration
```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Use multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1,2

# Disable GPU
export CUDA_VISIBLE_DEVICES=""
```

### YOLO Configuration
```bash
# Disable analytics
export YOLO_ANALYTICS=False

# Set log level
export YOLO_VERBOSE=False
```

## Advanced Configurations

### Custom Training Loop

```python
from ultralytics import YOLO

model = YOLO('yolov8m.pt')

# Custom training
results = model.train(
    data='config/car_data.yaml',
    epochs=100,
    imgsz=800,
    batch=32,
    # Custom parameters
    patience=30,
    optimizer='SGD',
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
)
```

### Model Export

```python
from ultralytics import YOLO

model = YOLO('runs/detect/car_detection/weights/best.pt')

# Export to different formats
model.export(format='onnx')      # ONNX
model.export(format='torchscript')  # TorchScript
model.export(format='tflite')    # TensorFlow Lite
model.export(format='pb')        # TensorFlow SavedModel
```

## Monitoring and Debugging

### TensorBoard Monitoring
```bash
tensorboard --logdir runs/detect/car_detection
```

### Validation Metrics to Watch

1. **train/loss**: Should decrease steadily
2. **val/loss**: Should follow train/loss with small gap
3. **metrics/mAP50**: Should increase toward 1.0
4. **fitness**: Overall metric for model selection

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Loss not decreasing | Lower learning rate, check data quality |
| Overfitting | Increase augmentation, reduce model size |
| Memory error | Reduce batch size or image size |
| Slow training | Use smaller model or reduce image size |
| Low accuracy | Increase training data or epochs |

## Performance Optimization

### For Faster Training
- Reduce image size: `--imgsz 416`
- Smaller model: `--model yolov8n.pt`
- Mixed precision: Enabled by default
- Multi-GPU training: `--device 0,1`

### For Better Accuracy
- Larger model: `--model yolov8x.pt`
- Larger image size: `--imgsz 1280`
- More epochs: `--epochs 200+`
- More diverse data: 500+ images recommended

### GPU Memory Optimization

| Model | imgsz=640 | imgsz=800 | imgsz=1024 |
|-------|-----------|-----------|------------|
| nano | 2GB | 2GB | 3GB |
| small | 2GB | 3GB | 4GB |
| medium | 3GB | 4GB | 6GB |
| large | 4GB | 6GB | 8GB |
| xlarge | 8GB | 12GB | 16GB |

## Checkpoints and Resume

### Resume Training
```bash
python src/train_model.py \
  --config config/car_data.yaml \
  --resume
```

Looks for checkpoint in: `runs/detect/car_detection/weights/last.pt`

## Documentation References

- YOLO Official: https://docs.ultralytics.com
- PyTorch: https://pytorch.org/docs
- OpenCV: https://docs.opencv.org
