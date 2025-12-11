# Car Detection - Development Guide

## Setting Up Development Environment

### Install Development Dependencies
```bash
pip install -r requirements-dev.txt
```

### IDE Configuration

#### VS Code Extensions (Recommended)
- Python
- Pylance
- Jupyter
- Black Formatter
- Even Better TOML

#### PyCharm
- Built-in Python support
- Enable Jupyter support
- Configure Black formatter

## Code Style and Quality

### Code Formatting
```bash
# Format all Python files
black src/

# Check formatting
black --check src/
```

### Linting
```bash
# Run flake8 checks
flake8 src/

# Common issues to check
flake8 src/ --select E,W
```

### Import Sorting
```bash
# Sort imports in all files
isort src/

# Check import order
isort --check-only src/
```

### Pre-commit Hook (Optional)
```bash
# Create .git/hooks/pre-commit with:
#!/bin/bash
black src/
isort src/
flake8 src/
```

## Testing

### Run Tests
```bash
pytest tests/

# With coverage
pytest --cov=src tests/

# Verbose output
pytest -v tests/
```

### Test Structure
```
tests/
├── test_utils.py
├── test_train.py
├── test_predict.py
└── test_data_prep.py
```

## Adding New Features

### 1. Create Feature Branch
```bash
git checkout -b feature/car-tracking
```

### 2. Implement Feature
- Follow existing code style
- Add docstrings to all functions
- Add type hints

### 3. Test Feature
```bash
pytest -v
black --check src/
flake8 src/
```

### 4. Create Pull Request
- Clear description of changes
- Link related issues
- Request review

## Common Development Tasks

### Adding New Training Hyperparameter

1. Add to `src/train_model.py`:
```python
parser.add_argument('--new-param', type=float, default=0.5)
```

2. Pass to training:
```python
model.train(..., new_param=args.new_param)
```

3. Document in `docs/CONFIG_REFERENCE.md`

### Adding New Prediction Feature

1. Add method to `CarDetector` class in `src/predict_model.py`
2. Add helper function to `src/utils.py` if needed
3. Update docstrings
4. Test with sample images

### Adding New Evaluation Metric

1. Add to `src/evaluate.py`:
```python
custom_metric = compute_custom_metric(results)
```

2. Add to output dictionary
3. Add to documentation
4. Create plot if applicable

## Debugging

### Enable Verbose Logging
```python
# In your script
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Debug YOLO Training
```python
from ultralytics import YOLO

model = YOLO('yolov8m.pt')
results = model.train(
    data='config/car_data.yaml',
    verbose=True,  # Enable verbose output
    # ... other params
)
```

### Profile Code Performance
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
detector = CarDetector('model.pt')
image = load_image('car.jpg')
detector.detect(image)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

## Documentation

### Update Documentation
1. Edit relevant `.md` file in `docs/`
2. Follow Markdown style guide
3. Include code examples where applicable
4. Keep README.md in sync

### Documentation Files
- `README.md` - Quick start and overview
- `USAGE.md` - Detailed usage guide
- `ARCHITECTURE.md` - System design
- `CONFIG_REFERENCE.md` - Configuration options
- `DEVELOPMENT.md` - This file

## Deployment

### Prepare for Production

1. **Export Model**:
```python
from ultralytics import YOLO

model = YOLO('best.pt')
model.export(format='onnx')  # Or other format
```

2. **Performance Testing**:
```python
import time
detector = CarDetector('best.pt')
times = []

for i in range(100):
    start = time.time()
    detector.detect(image)
    times.append(time.time() - start)

avg_time = sum(times) / len(times)
fps = 1 / avg_time
print(f"Average time: {avg_time:.3f}s, FPS: {fps:.1f}")
```

3. **Create Docker Image**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY models/ models/
COPY config/ config/

CMD ["python", "src/predict_model.py"]
```

## Performance Benchmarking

### Compare Models
```bash
# Compare training speed
time python src/train_model.py --model yolov8n.pt --epochs 10
time python src/train_model.py --model yolov8m.pt --epochs 10

# Compare inference speed
python src/predict_model.py --model yolov8n.pt --source image.jpg
python src/predict_model.py --model yolov8m.pt --source image.jpg
```

### Memory Profiling
```python
from memory_profiler import profile

@profile
def detect_cars(image_path):
    detector = CarDetector('best.pt')
    image = load_image(image_path)
    return detector.detect(image)

detect_cars('car.jpg')
```

Run with:
```bash
python -m memory_profiler script.py
```

## CI/CD Pipeline (GitHub Actions Example)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      - name: Format check
        run: black --check src/
      - name: Lint
        run: flake8 src/
      - name: Tests
        run: pytest tests/
```

## Troubleshooting Development

### Import Errors
```python
# Add src to path
import sys
sys.path.insert(0, '/path/to/car-detection/src')
```

### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU for development
export CUDA_VISIBLE_DEVICES=""
```

### Dependency Conflicts
```bash
# Update pip
pip install --upgrade pip

# Create fresh environment
python -m venv venv_new
source venv_new/bin/activate
pip install -r requirements.txt
```

## Contributing Guidelines

1. **Code Style**: Follow PEP 8, use Black for formatting
2. **Documentation**: Add docstrings to all functions/classes
3. **Testing**: Write tests for new features
4. **Commits**: Clear, descriptive commit messages
5. **PR Description**: Explain what, why, and how

## Resources

- [YOLO Documentation](https://docs.ultralytics.com)
- [PyTorch Tutorial](https://pytorch.org/tutorials/)
- [Python Best Practices](https://pep8.org/)
- [Git Workflow](https://git-scm.com/book/en/v2)

## Getting Help

1. Check documentation files
2. Search existing issues
3. Review similar code in repository
4. Ask in project discussions
5. File a detailed issue
