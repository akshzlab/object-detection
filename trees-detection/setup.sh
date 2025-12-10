#!/bin/bash
# Tree Detection System - Quick Start Script
# This script sets up the project and provides common commands

set -e

echo "=========================================="
echo "Tree Detection System - YOLOv8"
echo "=========================================="
echo ""

# Check if requirements are installed
echo "Checking dependencies..."
python -c "from ultralytics import YOLO" 2>/dev/null && echo "✓ YOLOv8 is installed" || echo "✗ Installing dependencies..."

# Install if needed
if ! python -c "from ultralytics import YOLO" 2>/dev/null; then
    echo "Installing required packages..."
    pip install -r requirements.txt
fi

echo ""
echo "Setup complete! Here are some useful commands:"
echo ""
echo "1. Prepare Data:"
echo "   python src/prepare_data.py organize data/images data/labels --output data/processed"
echo ""
echo "2. Validate Data:"
echo "   python src/prepare_data.py validate data/processed"
echo ""
echo "3. Train Model (nano, 100 epochs):"
echo "   python src/train_model.py"
echo ""
echo "4. Train Model (medium, 200 epochs):"
echo "   python src/train_model.py --model m --epochs 200 --batch 32"
echo ""
echo "5. Run Inference:"
echo "   python src/predict_model.py path/to/image.jpg --model runs/detect/tree_model_v1/weights/best.pt"
echo ""
echo "6. Count Trees:"
echo "   python src/predict_model.py path/to/image.jpg --model runs/detect/tree_model_v1/weights/best.pt --count-only"
echo ""
echo "7. Evaluate Model:"
echo "   python src/evaluate.py --model runs/detect/tree_model_v1/weights/best.pt --config config/tree_data.yaml"
echo ""
echo "=========================================="
