# Tree Detection System (YOLOv8)

A complete, production-ready YOLOv8-based object detection pipeline for identifying and localizing trees in satellite/drone imagery.

## 📚 Documentation

All comprehensive documentation is in the `docs/` directory:

| Document | Purpose |
|----------|---------|
| [docs/INDEX.md](docs/INDEX.md) | **Start here** - Master index and navigation |
| [docs/README.md](docs/README.md) | Detailed project overview |
| [docs/USAGE.md](docs/USAGE.md) | Complete usage instructions with examples |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and data flow |
| [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md) | Configuration file reference |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Developer guidelines and testing |
| [docs/COMPLETION_SUMMARY.md](docs/COMPLETION_SUMMARY.md) | Project status and what was built |

## 🗂️ Project Structure

```
trees-detection/
├── README.md                    # This file
├── USAGE.md                     # Quick usage pointer (see docs/USAGE.md)
├── requirements.txt             # Python dependencies
├── requirements-dev.txt         # Development dependencies
├── setup.sh                     # Quick start setup script
├── .gitignore                   # Git configuration
│
├── config/
│   └── tree_data.yaml           # YOLO dataset configuration
│
├── src/                         # Python source modules
│   ├── train_model.py           # Training pipeline
│   ├── predict_model.py         # Inference engine
│   ├── prepare_data.py          # Data preparation and validation
│   ├── evaluate.py              # Model evaluation framework
│   └── utils.py                 # Shared utilities (logging, config, etc)
│
├── data/                        # Dataset directory
│   ├── train/                   # Training images/labels (organize here)
│   └── val/                     # Validation images/labels (organize here)
│
├── docs/                        # Complete documentation (7 guides)
│   ├── INDEX.md                 # Master index
│   ├── README.md                # Detailed project overview
│   ├── USAGE.md                 # Usage guide
│   ├── ARCHITECTURE.md          # System architecture
│   ├── CONFIG_REFERENCE.md      # Configuration reference
│   ├── DEVELOPMENT.md           # Developer guide
│   └── COMPLETION_SUMMARY.md    # Completion summary
│
├── models/                      # Model weights storage
│   └── (saved model files go here)
│
└── runs/                        # Training outputs
    └── (training logs, results, weights go here)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
```bash
# Organize images and labels into train/val splits
python src/prepare_data.py organize data/images data/labels \
    --output data/processed --train-ratio 0.8

# Validate the structure
python src/prepare_data.py validate data/processed
```

### 3. Train Model
```bash
# Quick training (nano model)
python src/train_model.py --model n --epochs 100

# Production training (medium model)
python src/train_model.py --model m --epochs 200 --batch 32
```

### 4. Evaluate Performance
```bash
python src/evaluate.py \
    --model runs/detect/tree_model_v1/weights/best.pt \
    --config config/tree_data.yaml
```

### 5. Run Inference
```bash
# Detect and save results
python src/predict_model.py image.jpg \
    --model runs/detect/tree_model_v1/weights/best.pt \
    --output results

# Count trees only
python src/predict_model.py image.jpg \
    --model runs/detect/tree_model_v1/weights/best.pt \
    --count-only
```

## 📦 Core Components

| Module | Purpose |
|--------|---------|
| `src/train_model.py` | Complete training pipeline with transfer learning, early stopping, and checkpointing |
| `src/predict_model.py` | TreeDetector class for inference, single image/batch processing, crop saving |
| `src/prepare_data.py` | YOLO format data organization, train/val splitting, format validation |
| `src/evaluate.py` | ModelEvaluator class for validation, mAP/precision/recall metrics, JSON export |
| `src/utils.py` | Logging, YAML config loading, data validation, directory creation |

## ⚙️ Configuration

Dataset configuration in `config/tree_data.yaml`:

```yaml
path: ../data/processed
train: train/images
val: val/images
nc: 1              # Number of classes (1 = trees only)
names:
  0: tree          # Class name mapping
```

## 📋 Requirements

**Core dependencies:**
- ultralytics >= 8.3.0 (YOLOv8)
- torch >= 2.0.0
- torchvision >= 0.15.0
- opencv-python-headless
- numpy, scipy, scikit-learn
- PyYAML, matplotlib, Pillow

See `requirements.txt` for full list and versions.

## 🎯 Features

✅ **Complete pipeline** - Data prep → Training → Evaluation → Inference  
✅ **Multiple model sizes** - nano, small, medium, large, xlarge  
✅ **Transfer learning** - Pre-trained YOLOv8 weights  
✅ **Flexible inference** - Single images, batches, video  
✅ **Comprehensive metrics** - mAP50, mAP50-95, precision, recall  
✅ **Production-ready** - Error handling, logging, validation  
✅ **Well documented** - 7 comprehensive guides  

## 📖 For More Information

- **First time?** → Read [docs/INDEX.md](docs/INDEX.md)
- **How to use?** → See [docs/USAGE.md](docs/USAGE.md)
- **How it works?** → Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Configuration?** → Check [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md)
- **Development?** → Review [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)

---

**Status:** ✅ Production-Ready | **Updated:** December 10, 2025
