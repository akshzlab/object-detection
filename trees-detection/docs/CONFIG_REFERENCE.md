# Configuration Reference

## YAML Configuration File (`config/tree_data.yaml`)

The dataset configuration file tells YOLOv8 where to find your training data.

### Basic Structure

```yaml
# Root path to the dataset
path: ../data/processed

# Relative paths from root to train/val images
train: train/images
val: val/images

# Number of classes
nc: 1

# Class names (map class IDs to names)
names:
  0: tree
```

### Detailed Explanation

#### `path`
- **Type:** string
- **Required:** Yes
- **Description:** Root directory containing train/val data
- **Can be:** Absolute or relative path (relative to config file)
- **Example:** `../data/processed` or `/absolute/path/to/data`

#### `train`
- **Type:** string
- **Required:** Yes
- **Description:** Relative path to training images (from `path`)
- **Example:** `train/images`
- **Structure:** Directory containing `.jpg`, `.png`, etc.

#### `val`
- **Type:** string
- **Required:** Yes
- **Description:** Relative path to validation images (from `path`)
- **Example:** `val/images`
- **Structure:** Directory containing `.jpg`, `.png`, etc.

#### `nc`
- **Type:** integer
- **Required:** Yes
- **Description:** Number of object classes
- **For trees:** Always `1`
- **Range:** 1-999

#### `names`
- **Type:** dictionary
- **Required:** Yes
- **Description:** Maps class IDs to human-readable names
- **For trees:** `0: tree`
- **Format:** 
  ```yaml
  names:
    0: tree
  ```

### Optional Parameters

```yaml
# For test set (if you have one)
test: test/images
```

## Expected Directory Structure

Your configuration should match this structure:

```
/absolute/or/relative/path/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── labels/
│       ├── img_001.txt
│       ├── img_002.txt
│       └── ...
├── val/
│   ├── images/
│   │   ├── img_101.jpg
│   │   ├── img_102.jpg
│   │   └── ...
│   └── labels/
│       ├── img_101.txt
│       ├── img_102.txt
│       └── ...
└── test/ (optional)
    ├── images/
    └── labels/
```

## YOLO Format Annotation Files

Each image must have a corresponding `.txt` file with the same name.

### Format: One line per object
```
<class_id> <x_center> <y_center> <width> <height>
```

### Parameters
- **class_id:** 0 (for trees)
- **x_center:** Horizontal center (0.0 to 1.0, normalized)
- **y_center:** Vertical center (0.0 to 1.0, normalized)
- **width:** Bounding box width (0.0 to 1.0, normalized)
- **height:** Bounding box height (0.0 to 1.0, normalized)

### Example (file: forest_001.txt)
```
0 0.5 0.5 0.3 0.4
0 0.2 0.8 0.15 0.2
0 0.8 0.3 0.2 0.25
```

This annotates 3 trees in forest_001.jpg:
- Tree 1: center at (50%, 50%), width 30%, height 40%
- Tree 2: center at (20%, 80%), width 15%, height 20%
- Tree 3: center at (80%, 30%), width 20%, height 25%

## Coordinate System

```
        0                    1.0
      0 +--------------------+
        |                    |
        |  (x_center,        |
        |   y_center)        |
      | |        ↓            |
        |       +--+          |
        |       |  |← width   |
        |       +--+          |
        |        ↑            |
        |      height         |
        |                    |
    1.0 +--------------------+
```

## Creating Configuration Files

### Step 1: Create config directory
```bash
mkdir -p config
```

### Step 2: Create YAML file
```bash
cat > config/tree_data.yaml << 'EOF'
path: ../data/processed
train: train/images
val: val/images
nc: 1
names:
  0: tree
EOF
```

### Step 3: Verify paths
```bash
# Check that paths are correct
ls -la ../data/processed/train/images
ls -la ../data/processed/val/images
```

## Configuration Validation

The system automatically validates:
- ✓ File exists and is valid YAML
- ✓ `path` directory exists
- ✓ `train` images directory exists
- ✓ `val` images directory exists
- ✓ Images and labels match (in pairs)
- ✓ `nc` is a positive integer
- ✓ `names` has entries for all class IDs

### Manual Validation
```bash
# Validate your data structure
python src/prepare_data.py validate data/processed
```

## Common Issues

### Issue: "Config file not found"
```
ERROR: Config file not found: config/tree_data.yaml
```
**Solution:**
```bash
# Create the config directory and file
mkdir -p config
```