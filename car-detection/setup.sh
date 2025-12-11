#!/bin/bash

# Setup script for car detection project

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Creating necessary directories..."
mkdir -p data/train/images data/train/labels
mkdir -p data/val/images data/val/labels
mkdir -p data/test/images
mkdir -p models
mkdir -p runs/detect

echo "Downloading YOLOv8 models..."
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo "Setup complete!"
echo "Next steps:"
echo "1. Prepare your dataset: python src/prepare_data.py --source /path/to/images"
echo "2. Train model: python src/train_model.py --config config/car_data.yaml"
echo "3. Predict: python src/predict_model.py --model runs/detect/car_detection/weights/best.pt --source /path/to/image"
