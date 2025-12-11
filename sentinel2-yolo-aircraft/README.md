# Sentinel-2 Aircraft Detection with YOLO

This repository contains a complete workflow for training and running a YOLO-based aircraft detector on Sentinel-2 satellite imagery. It includes:

* A synthetic sample dataset
* YOLO training scripts
* Sliding-window geospatial inference
* ONNX export support
* Jupyter notebooks for training and inference

## Features

* End-to-end training pipeline (Ultralytics YOLO)
* Large-scale GeoTIFF inference with overlap-aware sliding windows
* GeoJSON export of detections
* Notebook-based visualization and monitoring

## Repository Structure

```
sentinel2-yolo-aircraft/
├── README.md
├── LICENSE
├── pyproject.toml
├── setup.py
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── synthetic_sample/
│   │   ├── images/{train,val,test}/
│   │   └── labels/{train,val,test}/
│   ├── data.yaml
│   └── detections/
│
├── sentinel2_yolo/
│   ├── __init__.py
│   ├── dataset.py
│   ├── inference.py
│   ├── tiling.py
│   └── utils.py
│
├── scripts/
│   ├── train.py
│   ├── infer_scene.py
│   └── export_onnx.py
│
├── notebooks/
│   ├── training_notebook_enhanced.ipynb
│   └── inference_notebook.ipynb
│
└── models/
    └── yolov8s-aircraft.pt
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Training

```bash
python scripts/train.py --data data/data.yaml --epochs 50 --img 640
```

## Inference

```bash
python scripts/infer_scene.py --geotiff path/to/scene.tif --weights models/yolov8s-aircraft.pt
```

## License

MIT License
