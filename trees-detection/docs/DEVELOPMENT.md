# Development Guide

This guide is for developers working on the tree detection system.

## Project Architecture

```
Tree Detection System
├── Data Layer
│   └── Data Preparation & Validation (prepare_data.py)
├── Model Layer
│   ├── YOLOv8 Backbone (CNN feature extraction)
│   ├── Detection Head (bounding box prediction)
│   └── Training (train_model.py)
├── Inference Layer
│   └── Prediction & Detection (predict_model.py)
└── Utilities
    ├── Configuration (utils.py, tree_data.yaml)
    ├── Logging & Monitoring (utils.py)
    └── Evaluation (evaluate.py)
```

## File Organization

### `src/utils.py`
Core utility functions:
- `setup_logging()`: Configure logging
- `load_yaml_config()`: Load YAML configs
- `validate_data_structure()`: Verify YOLO format
- `create_directory_structure()`: Initialize project dirs

### `src/train_model.py`
Training pipeline:
- `train_tree_detector()`: Main training function
- Argument parsing for CLI
- Config validation
- Error handling

### `src/predict_model.py`
Inference engine:
- `TreeDetector` class: Encapsulates model loading and inference
- `detect()`: Run detection
- `detect_and_save()`: Detection with output
- `count_trees()`: Tree counting utility

### `src/prepare_data.py`
Data handling:
- `organize_yolo_format_data()`: Split train/val
- `validate_yolo_format()`: Format validation
- Command-line interface for data processing

### `src/evaluate.py`
Model evaluation:
- `ModelEvaluator` class: Evaluation interface
- `evaluate()`: Run validation metrics
- JSON results export

## Adding New Features

### 1. Adding a New Data Augmentation

Edit `src/train_model.py`:

```python
# In train_tree_detector() function
results = model.train(
    data=config_path,
    epochs=epochs,
    # Add augmentation parameters
    hsv_h=0.015,  # HSV-Hue augmentation
    hsv_s=0.7,    # HSV-Saturation augmentation
    hsv_v=0.4,    # HSV-Value augmentation
    ...
)
```

### 2. Adding Custom Metrics

Edit `src/evaluate.py`:

```python
def evaluate(self, data_config: str, imgsz: int = 640) -> dict:
    results = self.model.val(...)
    
    # Add custom metrics
    return {
        'mAP50': float(results.box.map50),
        'custom_metric': calculate_custom_metric(results),
    }
```

### 3. Adding Batch Processing

Create `src/batch_process.py`:

```python
def process_directory(input_dir, model_path, output_dir):
    detector = TreeDetector(model_path)
    
    for image_path in Path(input_dir).glob('*.jpg'):
        results = detector.detect_and_save(image_path, output_dir)
```

## Testing

### Manual Testing

1. **Test data preparation:**
```bash
mkdir -p test_data/images test_data/labels
# Add test images and labels
python src/prepare_data.py organize test_data/images test_data/labels
python src/prepare_data.py validate data/processed
```

2. **Test training (nano, few epochs):**
```bash
python src/train_model.py --model n --epochs 2 --batch 4
```

3. **Test inference:**
```bash
python src/predict_model.py test_image.jpg --model runs/detect/tree_model_v1/weights/best.pt
```

### Unit Testing

Create `tests/test_utils.py`:

```python
import unittest
from src.utils import load_yaml_config, validate_data_structure

class TestUtils(unittest.TestCase):
    def test_load_yaml_config(self):
        config = load_yaml_config('config/tree_data.yaml')
        self.assertIn('nc', config)
        self.assertEqual(config['nc'], 1)
    
    def test_validate_data_structure(self):
        result = validate_data_structure('data/processed')
        self.assertIsNotNone(result)
```

Run tests:
```bash
python -m pytest tests/
```

## Debugging

### Enable Verbose Logging

In any script:
```python
logger = setup_logging('debug.log')
```

### Debug Training

Add to `train_model.py`:
```python
results = model.train(
    ...,
    verbose=True,  # Enable verbose output
    device=0,      # Use GPU for debugging
)
```

### Profile Performance

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
detector = TreeDetector(model_path)
results = detector.detect(image_path)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```

## Code Standards

### Style Guide
- Use PEP 8 formatting
- Type hints for all functions
- Docstrings for all modules/functions/classes
- Comments for complex logic

### Example Function:
```python
def detect_trees(image_path: str, conf_threshold: float = 0.25) -> list:
    """
    Detect trees in an image.
    
    Args:
        image_path: Path to input image
        conf_threshold: Confidence threshold (0-1)
    
    Returns:
        List of detections
        
    Raises:
        FileNotFoundError: If image doesn't exist
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Implementation
    return results
```

## Contributing

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes
3. Test thoroughly: `python src/train_model.py --model n --epochs 2`
4. Commit with clear message: `git commit -m "Add feature X"`
5. Push and create PR

## Performance Optimization

### For Training
- Use larger batch size (if GPU memory allows)
- Use medium or large model for better accuracy
- Increase epochs and patience

### For Inference
- Use smaller model (nano/small) for speed
- Batch process images
- Reduce image size if possible

### For Memory
- Reduce batch size
- Use smaller model
- Use mixed precision: Add `amp=True` to train()

## Deployment

### Export Model

```python
from ultralytics import YOLO

model = YOLO('runs/detect/tree_model_v1/weights/best.pt')
model.export(format='onnx')  # For deployment
```

### Integration

```python
from src.predict_model import TreeDetector

detector = TreeDetector('best.pt')
results = detector.detect_and_save(image_path, output_dir)
```

## Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLOv8 GitHub Repository](https://github.com/ultralytics/ultralytics)
- [Object Detection Fundamentals](https://docs.ultralytics.com/yolov8/)
- [Custom Dataset Training](https://docs.ultralytics.com/modes/train/)
