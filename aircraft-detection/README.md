# Aircraft Detection using YOLOv8

A deep learning project for detecting and classifying military aircraft using YOLOv8. This project includes both model training and inference capabilities with a web-based interface for real-time detection.

## Features

- **YOLOv8 Object Detection**: Trained model for detecting various aircraft types
- **Image & Video Support**: Process both images and video files
- **Web Interface**: User-friendly Flask web application for easy detection
- **Real-time Inference**: Fast detection with confidence scores
- **Cross-platform**: Works on Windows, macOS, and Linux

## Project Structure

```
aircraft-detection/
├── app.py                          # Flask backend server
├── index.html                      # Web interface frontend
├── .env                           # Environment variables (API keys)
├── requirements.txt               # Python dependencies
├── notebooks/
│   ├── train_aircraft_yolov8.ipynb    # Model training notebook
│   ├── train_aircraft_yolo11.ipynb    # Alternative YOLOv11 training
│   └── runs/
│       └── detect/train/
│           └── weights/
│               └── best.pt        # Trained model weights
├── output/                        # Directory for processed results
└── README.md                      # This file
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8+
- pip or conda package manager
- Git

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/object-detection.git
cd object-detection/aircraft-detection
```

### 3. Create Environment Variables

Create a `.env` file in the project directory:

```env
ROBOFLOW_API_KEY=your_roboflow_api_key_here
```

Get your Roboflow API key from [Roboflow](https://roboflow.com/)

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download the Model

The trained model weights (`best.pt`) should be located at:
```
./notebooks/runs/detect/train/weights/best.pt
```

If you need to train a new model, see [Training](#training) section.

## Usage

### Web Interface (Recommended)

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Upload an image or video file
4. Click "Submit" to run detection
5. View results with bounding boxes and confidence scores

### Python API

```python
from ultralytics import YOLO
from PIL import Image

# Load model
model = YOLO("./notebooks/runs/detect/train/weights/best.pt")

# Detect on image
results = model("image.jpg")

# Get detections
for result in results:
    boxes = result.boxes
    for box in boxes:
        class_name = result.names[int(box.cls)]
        confidence = box.conf.item()
        print(f"Detected: {class_name} ({confidence:.2%})")
    
    # Display result
    result.show()
```

## Dataset

- **Source**: [Roboflow Aircraft Detection Dataset](https://universe.roboflow.com/)
- **Size**: 14,200+ annotated images
- **Classes**: 77 different aircraft types (fighters, helicopters, transports, etc.)
- **Format**: YOLO format with .txt label files

### Dataset Preparation (Optional)

If you want to download and prepare the dataset:

1. Get your Roboflow API key
2. Run the notebook cell:
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("personal-workspace-gf3wi").project("aircraft-zccgz-cdl21")
version = project.version(1)
dataset = version.download("yolov8")
```

## Training

To train a new model from scratch:

1. Open `notebooks/train_aircraft_yolov8.ipynb` in Jupyter or Google Colab
2. Set your Roboflow API key in the `.env` file
3. Run the notebook cells in order
4. Training will be performed on 100 epochs with:
   - Base model: YOLOv8n (nano - fastest)
   - Image size: 640x640
   - Dataset: Roboflow aircraft detection dataset

### Google Colab Training

The notebook includes Google Colab support:
1. Click the "Open in Colab" badge at the top of the notebook
2. Set up your Roboflow API key
3. Run all cells to train the model

Training takes approximately 2-4 hours on a GPU.

## Model Performance

- **YOLOv8n (nano)**: Fast inference, suitable for real-time detection
- **Inference Speed**: ~50-100ms per image (depending on hardware)
- **Accuracy**: Trained on 14,200+ annotated aircraft images

## API Endpoints

### POST /detect

Upload an image or video for detection.

**Request:**
```
multipart/form-data:
  file: <image or video file>
```

**Response (Image):**
```json
{
  "detections": [
    {
      "class": "F-16",
      "confidence": 0.95
    },
    {
      "class": "Helicopter",
      "confidence": 0.87
    }
  ],
  "image_url": "/output/processed_image.jpg"
}
```

**Response (Video):**
```json
{
  "detections": "Processed video successfully",
  "video_url": "/output/predict/video.mp4"
}
```

### GET /output/<filename>

Retrieve processed images or videos.

## File Formats Supported

**Images:**
- JPG/JPEG
- PNG
- BMP
- GIF

**Videos:**
- MP4
- AVI
- MOV
- MKV

## Requirements

See [requirements.txt](requirements.txt):

```
Flask
Pillow
ultralytics
roboflow
python-dotenv
opencv-python
torch
torchvision
numpy
```

## Troubleshooting

### Model not found error
- Ensure the trained model is located at: `./notebooks/runs/detect/train/weights/best.pt`
- Check that the path in `app.py` matches your actual model location

### API key not loaded
- Create a `.env` file with `ROBOFLOW_API_KEY=your_key`
- Ensure `python-dotenv` is installed

### CUDA/GPU errors
- Install CUDA-compatible PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- Or use CPU-only mode (slower)

## Performance Tips

1. **Use smaller images for faster inference**: Adjust `imgsz` parameter
2. **Batch processing**: Process multiple images together
3. **GPU Acceleration**: Install CUDA for ~10x speedup
4. **Model Size**: Use YOLOv8n for speed, YOLOv8m/l for accuracy

## Future Improvements

- [ ] Model compression (quantization, pruning)
- [ ] Real-time webcam detection
- [ ] Batch processing API
- [ ] Model serving with TensorRT
- [ ] Docker containerization
- [ ] Deployment to cloud platforms (AWS, GCP, Azure)

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Object Detection Guide](https://roboflow.com/tutorials)
- [YOLO Paper](https://arxiv.org/abs/2201.12741)

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.

---

**Last Updated**: December 2025
