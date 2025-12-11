# Car Detection - Architecture Documentation

## System Architecture

### Overview
This car detection system is built using YOLOv8 (You Only Look Once version 8), a state-of-the-art real-time object detection framework. The system supports three main operations:
1. Data Preparation
2. Model Training
3. Inference & Evaluation

## Component Breakdown

### 1. Data Preparation (`src/prepare_data.py`)

**Purpose**: Organize and split raw car detection data into train/validation/test sets.

**Key Functions**:
- `organize_images_and_labels()` - Organize data from a single directory
- `split_existing_data()` - Split separate image and label directories
- `validate_dataset()` - Validate dataset structure

**Features**:
- Random shuffling for unbiased splits
- Customizable train/val/test ratios
- Support for multiple image formats
- YOLO format label handling

### 2. Training Module (`src/train_model.py`)

**Purpose**: Train YOLO model on car detection dataset.

**Key Components**:
- `setup_training()` - Initialize training configuration
- `train_yolo_model()` - Main training function with hyperparameter tuning
- `validate_model()` - Validate trained model

**Training Parameters**:
- Model variants: YOLOv8n/s/m/l/x
- Epochs, batch size, image size customization
- Data augmentation (HSV, rotation, translation, flip)
- Early stopping with patience
- SGD optimizer with learning rate scheduling

**Output**:
- Best model weights (`best.pt`)
- Last model weights (`last.pt`)
- Training curves and metrics
- Validation results

### 3. Inference Module (`src/predict_model.py`)

**Purpose**: Perform car detection on new images.

**Key Classes**:
- `CarDetector` - Main detection class
  - `detect()` - Single image detection
  - `detect_batch()` - Batch processing

**Key Functions**:
- `predict_image()` - Predict single image with visualization
- `predict_directory()` - Batch predict all images in directory

**Features**:
- Confidence threshold filtering
- Bounding box visualization
- Batch processing support
- JSON output format

### 4. Evaluation Module (`src/evaluate.py`)

**Purpose**: Evaluate model performance on validation/test sets.

**Key Functions**:
- `evaluate_model()` - Run validation and compute metrics
- `plot_training_curves()` - Visualize training progress

**Metrics Computed**:
- mAP50: Mean Average Precision at IoU=0.5
- mAP50-95: Mean Average Precision at IoU=0.5:0.95
- Per-class metrics
- Training/validation losses

### 5. Utilities (`src/utils.py`)

**Helper Functions**:
- `load_yaml_config()` - Load YAML configurations
- `save_yaml_config()` - Save YAML configurations
- `load_image()` - Load image from file
- `save_image()` - Save image to file
- `draw_boxes()` - Draw bounding boxes on image
- `get_image_files()` - List image files in directory
- `validate_dataset_structure()` - Check data directory structure

## Data Flow

```
Raw Images/Labels
       ↓
prepare_data.py (Organize & Split)
       ↓
train/val/test directories
       ↓
train_model.py (Train YOLO)
       ↓
Trained Weights (best.pt)
       ↓
predict_model.py (Inference)
       ↓
Detected Cars (Bounding Boxes)
       ↓
evaluate.py (Metrics)
```

## YOLO Format Details

### Input Format
- Images: PNG, JPEG, BMP, TIFF
- Labels: TXT files with one box per line

### Label Format
```
<class_id> <x_center> <y_center> <width> <height>
```
Where:
- `class_id`: 0 (only one class: car)
- Coordinates: Normalized to [0, 1] range

### Example
```
0 0.5 0.5 0.3 0.4
```
This represents a car with center at (0.5, 0.5) and normalized dimensions 0.3×0.4

## Model Architecture

YOLOv8 uses a modified CSPDarknet backbone with:
- Feature extraction
- Neck (FPN - Feature Pyramid Network)
- Head (Detection head with 3 scales)

## Configuration

`config/car_data.yaml`:
```yaml
path: ../data
train: train/images
val: val/images
test: test/images
nc: 1
names: ['car']
```

## Performance Considerations

### Speed vs Accuracy Trade-off
- **YOLOv8n**: ~45 FPS, lower accuracy
- **YOLOv8s**: ~60 FPS, medium accuracy
- **YOLOv8m**: ~38 FPS, good accuracy
- **YOLOv8l**: ~25 FPS, high accuracy
- **YOLOv8x**: ~15 FPS, highest accuracy

### Recommended Settings
- **Quick Testing**: YOLOv8n, 640px images
- **Production**: YOLOv8m/l, 800-1024px images
- **High Accuracy**: YOLOv8x, 1280px images

## GPU Memory Requirements
- YOLOv8n: ~2GB
- YOLOv8s: ~4GB
- YOLOv8m: ~6GB
- YOLOv8l: ~8GB
- YOLOv8x: ~16GB

## Future Enhancements

1. Multi-class detection (car, truck, bus, etc.)
2. Vehicle tracking across frames
3. Speed/direction estimation
4. License plate detection
5. Model quantization for edge deployment
6. ONNX export for cross-platform inference
