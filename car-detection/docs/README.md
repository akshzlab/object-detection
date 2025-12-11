# Car Detection Documentation Index

## Quick Links

- **[README.md](../README.md)** - Project overview and quick start
- **[USAGE.md](./USAGE.md)** - Detailed usage guide with examples
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - System design and components
- **[CONFIG_REFERENCE.md](./CONFIG_REFERENCE.md)** - Configuration options
- **[DEVELOPMENT.md](./DEVELOPMENT.md)** - Development guide

## Getting Started

### First Time Users
1. Start with [README.md](../README.md) for overview
2. Follow [USAGE.md](./USAGE.md) installation steps
3. Run quick training example
4. Make predictions on sample images

### Experienced Users
1. Review [ARCHITECTURE.md](./ARCHITECTURE.md) for system design
2. Check [CONFIG_REFERENCE.md](./CONFIG_REFERENCE.md) for advanced options
3. Customize configurations for your dataset
4. Deploy model to production

## By Task

### Setting Up the Project
- [README.md - Quick Start](../README.md#quick-start)
- [USAGE.md - Installation](./USAGE.md#installation)

### Preparing Data
- [USAGE.md - Step 1: Prepare Data](./USAGE.md#step-1-prepare-your-data)
- [CONFIG_REFERENCE.md - Data Configuration](./CONFIG_REFERENCE.md#data-configuration-car_datayaml)

### Training Models
- [USAGE.md - Step 2: Train](./USAGE.md#step-2-train-the-model)
- [CONFIG_REFERENCE.md - Training Configuration](./CONFIG_REFERENCE.md#training-configuration)
- [ARCHITECTURE.md - Training Module](./ARCHITECTURE.md#2-training-module-src-train_modelpy)

### Making Predictions
- [USAGE.md - Step 4: Predict](./USAGE.md#step-4-make-predictions)
- [ARCHITECTURE.md - Inference Module](./ARCHITECTURE.md#3-inference-module-src-predict_modelpy)

### Evaluating Performance
- [USAGE.md - Step 3: Evaluate](./USAGE.md#step-3-evaluate-the-model)
- [CONFIG_REFERENCE.md - Validation Metrics](./CONFIG_REFERENCE.md#monitoring-and-debugging)

### Development and Extension
- [DEVELOPMENT.md - Development Guide](./DEVELOPMENT.md)
- [ARCHITECTURE.md - System Overview](./ARCHITECTURE.md#system-architecture)

## File Structure

```
car-detection/
├── README.md                    # Quick start guide
├── setup.sh                     # Automated setup script
├── requirements.txt             # Python dependencies
├── requirements-dev.txt         # Development dependencies
├── config/
│   └── car_data.yaml           # Dataset configuration
├── data/                        # Dataset directory
│   ├── train/
│   ├── val/
│   └── test/
├── docs/
│   ├── README.md               # This file
│   ├── USAGE.md                # Usage guide
│   ├── ARCHITECTURE.md         # System design
│   ├── CONFIG_REFERENCE.md     # Configuration reference
│   └── DEVELOPMENT.md          # Development guide
├── jupyter/
│   └── car_detection_YOLOv8n.ipynb
├── models/                      # Pre-trained models
└── src/
    ├── __init__.py
    ├── train_model.py          # Training script
    ├── predict_model.py        # Inference script
    ├── evaluate.py             # Evaluation script
    ├── prepare_data.py         # Data preparation
    └── utils.py                # Utility functions
```

## Common Commands

### Setup
```bash
bash setup.sh
```

### Data Preparation
```bash
python src/prepare_data.py --source /path/to/images --output data
```

### Training
```bash
python src/train_model.py --config config/car_data.yaml --epochs 100
```

### Evaluation
```bash
python src/evaluate.py --model runs/detect/car_detection/weights/best.pt
```

### Prediction
```bash
python src/predict_model.py --model runs/detect/car_detection/weights/best.pt --source /path/to/image
```

## Key Concepts

### YOLO Format
- **Images**: JPEG, PNG, BMP, etc.
- **Labels**: TXT files with normalized coordinates
- **Format**: `<class_id> <x_center> <y_center> <width> <height>`

### Model Selection
- **Small**: yolov8n (fast, less accurate)
- **Medium**: yolov8m (balanced)
- **Large**: yolov8l (slow, more accurate)

### Key Metrics
- **mAP50**: Precision-Recall average at 50% IoU
- **mAP50-95**: Average across all IoU thresholds
- **Fitness**: Combined metric for model selection

## Troubleshooting

### Common Issues

**Installation Problems**
- See [USAGE.md - Installation](./USAGE.md#installation)
- Check Python version (3.8+)
- Verify CUDA installation

**Training Issues**
- See [USAGE.md - Troubleshooting](./USAGE.md#troubleshooting)
- Check data format and quality
- Monitor memory usage
- Review loss curves

**Prediction Issues**
- See [ARCHITECTURE.md - Model Architecture](./ARCHITECTURE.md#model-architecture)
- Verify model file exists
- Check input image format
- Adjust confidence threshold

## Performance Tuning

### Speed Optimization
- Use smaller model (yolov8n)
- Reduce image size (512-640)
- Increase batch size during training
- Use GPU if available

### Accuracy Improvement
- Use larger model (yolov8l/x)
- Increase image size (800-1280)
- Collect more training data
- Train for longer epochs (150+)

See [CONFIG_REFERENCE.md - Recommended Configurations](./CONFIG_REFERENCE.md#recommended-configurations)

## Advanced Topics

### Custom Modifications
- Modify model architecture: See DEVELOPMENT.md
- Add custom losses: See ARCHITECTURE.md
- Implement post-processing: See USAGE.md - Using in Your Code

### Deployment
- Export to ONNX: See CONFIG_REFERENCE.md
- Docker containerization: See DEVELOPMENT.md
- Edge deployment: See USAGE.md - Performance Tips

### Integration
- Python API: See USAGE.md - Using in Your Code
- Batch processing: See ARCHITECTURE.md - Inference Module
- Real-time applications: See CONFIG_REFERENCE.md

## External Resources

- [YOLOv8 Official Documentation](https://docs.ultralytics.com)
- [PyTorch Documentation](https://pytorch.org/docs)
- [OpenCV Documentation](https://docs.opencv.org)
- [Python Documentation](https://docs.python.org)

## Contact and Support

For issues, questions, or suggestions:
1. Check this documentation
2. Search existing issues
3. Review similar projects
4. File a detailed issue with:
   - Error message
   - Steps to reproduce
   - System information
   - Expected vs actual behavior

## Version History

- **v1.0.0**: Initial release with YOLOv8n support
  - Training module
  - Inference module
  - Evaluation module
  - Data preparation tools

## License

MIT License - See LICENSE file for details

---

Last Updated: 2025-12-11
