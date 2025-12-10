# Tree Detection System - Usage Guide

This guide provides detailed instructions for using the tree detection system.

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from ultralytics import YOLO; print('✓ YOLO installed')"
```

### 2. Prepare Your Data

Before training, you need annotated data in YOLO format.

**Data Format:**
- Images: `.jpg`, `.png`, etc.
- Annotations: `.txt` files (one per image) with format:
  ```
  <class_id> <x_center> <y_center> <width> <height>
  ```
  where coordinates are normalized (0-1).

**Example:**
```
0 0.5 0.5 0.3 0.4
0 0.2 0.8 0.15 0.2
```

### 3. Organize Data

Place images and labels in separate directories:

```
data/
├── images/
│   ├── forest_1.jpg
│   ├── forest_2.jpg
│   └── ...
└── labels/
    ├── forest_1.txt
    ├── forest_2.txt
    └── ...
```

Then organize into train/val splits:

```bash
python src/prepare_data.py organize data/images data/labels \
    --output data/processed \
    --train-ratio 0.8
```

### 4. Validate Data

```bash
python src/prepare_data.py validate data/processed
```

Output should show:
- Number of images in train/val
- Number of corresponding labels
- Any validation issues

### 5. Train Model

**Basic training (fastest):**
```bash
python src/train_model.py
```

**Custom training:**
```bash
python src/train_model.py \
    --model m \
    --epochs 200 \
    --batch 32 \
    --patience 30
```

**Options:**
- `--model {n,s,m,l,x}`: Model size (default: n)
- `--epochs INT`: Training epochs (default: 100)
- `--batch INT`: Batch size (default: 16)
- `--patience INT`: Early stopping patience (default: 20)
- `--device {0,cpu}`: GPU device or CPU (default: 0)

### 6. Run Inference

**Detect in single image:**
```bash
python src/predict_model.py path/to/image.jpg \
    --model runs/detect/tree_model_v1/weights/best.pt
```

**Save results and crops:**
```bash
python src/predict_model.py path/to/image.jpg \
    --model runs/detect/tree_model_v1/weights/best.pt \
    --output results \
    --save-crops
```

**Count only:**
```bash
python src/predict_model.py path/to/image.jpg \
    --model runs/detect/tree_model_v1/weights/best.pt \
    --count-only
```

### 7. Evaluate Model

```bash
python src/evaluate.py \
    --model runs/detect/tree_model_v1/weights/best.pt \
    --config config/tree_data.yaml
```

Outputs metrics:
- **mAP50**: Precision at IoU=0.5
- **mAP50-95**: Precision at IoU=0.5-0.95
- **Precision**: Detection accuracy
- **Recall**: Coverage of all trees
- **Fitness**: Overall score

## Advanced Usage

### Model Selection

Choose based on your requirements:

| Model | Speed | Accuracy | Memory |
|-------|-------|----------|--------|
| nano (n) | ⚡⚡⚡ | ⭐⭐ | Low |
| small (s) | ⚡⚡ | ⭐⭐⭐ | Low |
| medium (m) | ⚡ | ⭐⭐⭐⭐ | Medium |
| large (l) | 🐢 | ⭐⭐⭐⭐⭐ | High |
| xlarge (x) | 🐢🐢 | ⭐⭐⭐⭐⭐ | Very High |

### Training with Custom Parameters

```bash
# High accuracy training (slow)
python src/train_model.py \
    --model l \
    --epochs 300 \
    --batch 64 \
    --patience 50

# Fast training (low accuracy)
python src/train_model.py \
    --model n \
    --epochs 50 \
    --batch 8 \
    --patience 10
```

### Inference with Different Thresholds

```bash
# Strict (fewer false positives)
python src/predict_model.py image.jpg \
    --model best.pt \
    --conf 0.5

# Lenient (catch more trees)
python src/predict_model.py image.jpg \
    --model best.pt \
    --conf 0.1
```

### Batch Processing

```bash
# Create a script to process multiple images
for image in data/test_images/*.jpg; do
    python src/predict_model.py "$image" \
        --model runs/detect/tree_model_v1/weights/best.pt \
        --output results
done
```

## File Locations

### Inputs
- **Images:** `data/images/` or `data/processed/train/images/`
- **Annotations:** `data/labels/` or `data/processed/train/labels/`
- **Config:** `config/tree_data.yaml`
- **Model weights:** `runs/detect/tree_model_v1/weights/best.pt`

### Outputs
- **Training results:** `runs/detect/tree_model_v1/`
- **Model weights:** `runs/detect/tree_model_v1/weights/`
- **Training logs:** `runs/detect/tree_model_v1/results.csv`
- **Inference results:** `output/predictions/`
- **Eval results:** `runs/evaluate/eval_results.json`

## Common Issues and Solutions

### Out of Memory

**Error:** `CUDA out of memory`

**Solution:** Reduce batch size
```bash
python src/train_model.py --batch 8
```

### Slow Training on CPU

**Error:** Training taking hours per epoch

**Solution:** Use GPU or reduce model size
```bash
# Use smaller model
python src/train_model.py --model n --batch 8

# Or switch to GPU (if available)
python src/train_model.py --device 0
```

### Model Not Found

**Error:** `Model not found: best.pt`

**Solution:** Check path
```bash
ls -la runs/detect/tree_model_v1/weights/
```

### Data Format Errors

**Error:** Config file issues

**Solution:** Validate data structure
```bash
python src/prepare_data.py validate data/processed
```

## Tips for Best Results

### Data Quality
1. **Annotation Accuracy:** Labels must be precise
2. **Data Diversity:** Include various:
   - Seasons (spring, summer, fall, winter)
   - Weather (sunny, cloudy, rainy)
   - Lighting conditions
   - Tree types and sizes
3. **Dataset Size:** Minimum 100-200, ideal 500+

### Training
1. **Start with small model:** nano or small
2. **Monitor training curves:** Check `runs/detect/tree_model_v1/results.png`
3. **Early stopping:** Use patience parameter
4. **Data augmentation:** YOLOv8 handles this automatically

### Inference
1. **Confidence tuning:** Adjust threshold based on false positives/negatives
2. **Batch inference:** Process multiple images for efficiency
3. **Save crops:** Useful for manual review

## Next Steps

1. **Collect Data:**
   - Public datasets: NEON, Kaggle, OpenEarthMap
   - Manual annotation: CVAT, Roboflow, Labelimg

2. **Experiment:**
   - Try different model sizes
   - Tune hyperparameters
   - Compare results

3. **Deploy:**
   - Export to ONNX format
   - Integrate with application
   - Set up monitoring

## Support

For issues or questions:
- Check [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- Review [YOLO Format Spec](https://docs.ultralytics.com/datasets/detect/)
- Search [YOLOv8 Issues](https://github.com/ultralytics/ultralytics/issues)
