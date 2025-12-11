# Car Detection Project

Complete YOLO-based car detection system with training, inference, and evaluation capabilities.

## Quick Start

### 1. Setup
```bash
cd car-detection
bash setup.sh
```

### 2. Prepare Data
```bash
python src/prepare_data.py --source /path/to/images --output data
```

### 3. Train Model
```bash
python src/train_model.py --config config/car_data.yaml --epochs 100 --batch-size 16
```

### 4. Evaluate Model
```bash
python src/evaluate.py --model runs/detect/car_detection/weights/best.pt --data config/car_data.yaml
```

### 5. Predict
```bash
python src/predict_model.py --model runs/detect/car_detection/weights/best.pt --source /path/to/image
```

## Project Structure

- `config/` - Configuration files
  - `car_data.yaml` - Dataset configuration for YOLO
- `data/` - Dataset directory
  - `train/` - Training data
  - `val/` - Validation data
  - `test/` - Test data
- `docs/` - Documentation
- `jupyter/` - Jupyter notebooks for interactive exploration
- `models/` - Pre-trained and custom models
- `src/` - Source code
  - `train_model.py` - Training script
  - `predict_model.py` - Prediction script
  - `evaluate.py` - Evaluation script
  - `prepare_data.py` - Data preparation
  - `utils.py` - Utility functions

## Usage Examples

### Training with Custom Parameters
```bash
python src/train_model.py \
  --config config/car_data.yaml \
  --model yolov8m.pt \
  --epochs 150 \
  --batch-size 32 \
  --imgsz 800
```

### Batch Prediction
```bash
python src/predict_model.py \
  --model runs/detect/car_detection/weights/best.pt \
  --source data/test/images \
  --output predictions
```

### Data Preparation with Split
```bash
python src/prepare_data.py \
  --images /data/images \
  --labels /data/labels \
  --output data \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --validate
```

## Dataset Format

The project uses YOLO format:
- Images: JPEG, PNG, BMP, etc.
- Labels: TXT files with format: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0-1)

For car detection, class_id is always 0 (one class).

## Model Selection

Available YOLO models:
- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (slowest, most accurate)

## Configuration

Edit `config/car_data.yaml` to point to your dataset:
```yaml
path: ../data
train: train/images
val: val/images
test: test/images
nc: 1
names: ['car']
```

## Requirements

- Python 3.8+
- PyTorch with CUDA support (recommended)
- See `requirements.txt` for full dependencies

## Development

Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

Run tests, linting, and formatting:
```bash
pytest
black src/
flake8 src/
isort src/
```

## Results

Training results are saved in `runs/detect/car_detection/`:
- `weights/best.pt` - Best model checkpoint
- `weights/last.pt` - Last model checkpoint
- Training curves and metrics plots
- Validation results

## Performance Metrics

- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: Mean Average Precision at IoU=0.5:0.95
- **Fitness**: Combined metric for model selection

## License

MIT License
