# Sentinel-2 Dataset Migration Guide

## Overview

Both notebooks have been updated to work with **real Sentinel-2 satellite imagery** instead of synthetic data. This guide explains the changes and how to use them.

---

## Key Changes

### Training Notebook (`notebooks/training_notebook.ipynb`)

#### Data Configuration
- **Before**: Used synthetic images in `data/synthetic_sample/` with standard RGB format
- **After**: Uses Sentinel-2 GeoTIFF files in `data/train/`, `data/val/`, `data/test/`

#### New Features
1. **Sentinel-2 Band Management**
   - Supports all 11 Sentinel-2 bands at different resolutions
   - Configurable band selection for training
   - RGB visualization bands separate from training bands

2. **GeoTIFF Processing**
   - Reads multi-band GeoTIFF files with automatic band validation
   - Handles 12-bit and 16-bit Sentinel-2 data
   - Normalization for display

3. **Enhanced Validation**
   - Checks for GeoTIFF structure and band count
   - Validates data.yaml for Sentinel-2 format
   - Displays band information and spatial resolution

#### Configuration Parameters
```python
# Band selection for training
TRAINING_BANDS = [4, 3, 2]          # RGB bands (can expand to include NIR)

# Band selection for visualization
RGB_BANDS = [4, 3, 2]               # Natural color display

# Tile configuration
TILE_SIZE = 256                     # Sentinel-2 10m tiles (256px = 2560m)
TILE_OVERLAP = 0.1                  # 10% overlap

# Data structure
data/
  train/
    scene_01.tif, scene_02.tif, ... # Sentinel-2 GeoTIFFs
  val/
    scene_xx.tif, ...
  test/
    scene_yy.tif, ...
```

---

### Inference Notebook (`notebooks/inference_notebook.ipynb`)

#### Data Configuration
- **Before**: Generic GeoTIFF files in `data/` directory
- **After**: Optimized for Sentinel-2 GeoTIFFs in `data/test/`

#### New Features
1. **Band Combination Support**
   - Natural Color: Bands 4, 3, 2 (Red, Green, Blue)
   - False Color: Bands 8, 4, 3 (NIR, Red, Green) for vegetation
   - SWIR: Bands 12, 11, 4 (for cloud detection)
   - Agriculture: Bands 11, 8, 2

2. **Sentinel-2 Specific Improvements**
   - Auto-detection of test scenes
   - Smart band normalization (2% to 98% percentile stretch)
   - Resolution metadata display
   - Band availability checking

3. **Enhanced Visualization**
   - Proper 16-bit to 8-bit conversion
   - Percentile-based stretching for better contrast
   - Geographic bounds and CRS display

#### Configuration Parameters
```python
# Band selection for visualization (1-indexed)
BAND_INDICES = [4, 3, 2]            # Natural Color (RGB)
BAND_COMBO_NAME = "Natural Color (RGB)"

# Inference settings
TILE_SIZE = 1024                    # Tile size in pixels (~10.24km at 10m)
OVERLAP = 200                       # Pixel overlap
CONF_THRESHOLD = 0.5                # Detection confidence
NMS_IOU = 0.3                       # NMS deduplication threshold

# Data structure
data/
  test/
    scene.tif, scene2.tif, ...      # Sentinel-2 GeoTIFFs to process
```

---

## Data Preparation

### Expected Directory Structure
```
sentinel2-yolo-aircraft/
├── data/
│   ├── train/
│   │   ├── S2A_MSIL2A_scene1.tif
│   │   ├── S2A_MSIL2A_scene2.tif
│   │   └── ... (training scenes)
│   ├── val/
│   │   ├── S2B_MSIL2A_scene3.tif
│   │   └── ... (validation scenes)
│   ├── test/
│   │   ├── S2A_MSIL2A_scene4.tif
│   │   └── ... (test scenes)
│   ├── data.yaml
│   └── detections/  (outputs)
├── notebooks/
│   ├── training_notebook.ipynb
│   └── inference_notebook.ipynb
└── src/
```

### Downloading Sentinel-2 Data

1. **Copernicus Data Space Ecosystem**
   - https://dataspace.copernicus.eu/
   - Free registration required
   - Downloads L2A (Bottom-of-Atmosphere) processed data

2. **USGS EarthExplorer**
   - https://earthexplorer.usgs.gov/
   - Alternative source for Sentinel-2 data

3. **Cloud Platforms**
   - **AWS**: `sentinel-cogs` bucket (free within AWS)
   - **Google Cloud**: `sentinel-2-l1c` dataset
   - **Microsoft Azure**: `sentinel-2-l2a` dataset

### Data Format
- **Format**: GeoTIFF (Cloud-Optimized GeoTIFF preferred)
- **Bands**: 11 bands from Sentinel-2 L2A or L1C
- **Resolution**: 10m (RGB), 20m (Red Edge, SWIR), 60m (Coastal Aerosol)
- **Encoding**: 12-bit or 16-bit unsigned integer
- **CRS**: EPSG:32631 (UTM Zone 31N) - or any valid UTM zone

### Creating Labels for Training

Sentinel-2 GeoTIFFs need YOLO format labels (class, x_center, y_center, width, height normalized):

```
# Label file structure for data/train/scene1.txt:
0 0.45 0.52 0.08 0.10
0 0.67 0.34 0.06 0.07
1 0.23 0.78 0.05 0.06
```

Tools for annotation:
- **Roboflow**: Cloud-based annotation
- **CVAT**: Open-source annotation platform
- **LabelImg**: GUI tool for bounding boxes

---

## Sentinel-2 Band Reference

| Band | Name | Resolution | Wavelength | Use |
|------|------|-----------|-----------|-----|
| 1 | Coastal Aerosol | 60m | 0.433-0.453 μm | Atmospheric correction |
| 2 | Blue | 10m | 0.457-0.522 μm | Water, atmospheric scattering |
| 3 | Green | 10m | 0.542-0.578 μm | Vegetation |
| 4 | Red | 10m | 0.650-0.680 μm | Vegetation boundaries |
| 5 | Vegetation Red Edge | 20m | 0.698-0.713 μm | Fine vegetation details |
| 6 | Vegetation Red Edge | 20m | 0.733-0.748 μm | Vegetation stress |
| 7 | Vegetation Red Edge | 20m | 0.773-0.793 μm | LAI, biomass |
| 8 | NIR (Near-Infrared) | 10m | 0.785-0.900 μm | Vegetation vigor |
| 8A | Vegetation Red Edge | 20m | 0.855-0.875 μm | Vegetation classification |
| 11 | SWIR (Short-Wave IR) | 20m | 1.565-1.657 μm | Cloud/snow detection |
| 12 | SWIR | 20m | 2.100-2.280 μm | Cloud/moisture detection |

### Common Band Combinations

| Combination | Bands | Use Case |
|------------|-------|----------|
| Natural Color | 4, 3, 2 | Visual inspection |
| False Color (Vegetation) | 8, 4, 3 | Vegetation analysis |
| False Color (Urban) | 12, 11, 4 | Urban development |
| Agriculture | 11, 8, 2 | Crop monitoring |
| NDVI (Normalized Difference Vegetation Index) | (8-4)/(8+4) | Vegetation health |

---

## Usage Instructions

### Training a Model

1. **Prepare Data**
   ```bash
   # Organize Sentinel-2 GeoTIFFs
   mkdir -p data/{train,val,test}
   # Copy your scene GeoTIFFs to respective folders
   ```

2. **Update Labels**
   - Create corresponding `.txt` label files in same structure
   - Or use `prepare_data.py` to auto-generate labels

3. **Configure Notebook**
   - Open `notebooks/training_notebook.ipynb`
   - Adjust `TRAINING_BANDS` if needed
   - Modify `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`

4. **Run Training**
   - Execute all cells in order
   - Monitor metrics in real-time
   - Best weights saved to `runs/train/notebook_enhanced/weights/best.pt`

### Running Inference

1. **Prepare Sentinel-2 Scene**
   ```bash
   # Place test GeoTIFF in data/test/
   cp your_sentinel2_scene.tif data/test/
   ```

2. **Configure Notebook**
   - Open `notebooks/inference_notebook.ipynb`
   - Auto-detection finds latest trained model
   - Adjust `BAND_INDICES` for visualization (optional)
   - Set `CONF_THRESHOLD` (0.3-0.7 recommended)

3. **Run Inference**
   - Execute all cells
   - View detections overlaid on satellite imagery
   - Results exported to `data/detections/sentinel2_inference_results.geojson`

### Exporting Results

Results are automatically saved in multiple formats:
- **GeoJSON** - For web mapping (Leaflet, Mapbox)
- **GeoPackage** - For desktop GIS (QGIS, ArcGIS)
- **Shapefile** - Legacy GIS format
- **CSV** - For data analysis

---

## Performance Tips for Sentinel-2

1. **Memory Usage**
   - Sentinel-2 scenes are large (10,000-12,000 px)
   - Use `TILE_SIZE = 512` if GPU memory is limited
   - Increase `OVERLAP` for better edge detection

2. **Processing Speed**
   - GPU inference: ~5-10x faster than CPU
   - Batch processing multiple scenes
   - Use `DEVICE = '0'` for GPU acceleration

3. **Detection Quality**
   - Aircraft detection at 10m resolution: ~15-20m minimum object size
   - Lower `CONF_THRESHOLD` for small objects (0.3-0.4)
   - Increase `NMS_IOU` to remove more duplicates (0.4-0.5)

4. **Cloud Masking**
   - Consider filtering cloudy pixels using SCL band
   - Or train model to be robust to clouds

---

## Troubleshooting

### Common Issues

**Error: "GeoTIFF not found"**
- Ensure files are in correct directory structure
- Check file extensions (`.tif` or `.tiff`)
- Verify file paths in configuration cells

**Error: "Band indices out of range"**
- Check actual band count in your GeoTIFFs
- Sentinel-2 L2A has 11 bands (1-indexed)
- Adjust `BAND_INDICES` or `TRAINING_BANDS`

**Poor Detection Results**
- Verify training data quality
- Check if model was trained on similar imagery
- Try different band combinations
- Lower confidence threshold
- Increase training epochs

**Out of Memory (OOM)**
- Reduce `BATCH_SIZE` in training
- Reduce `TILE_SIZE` in inference
- Use `DEVICE = 'cpu'` instead of GPU
- Process smaller Sentinel-2 scenes

---

## See Also

- [Sentinel-2 User Guide](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi)
- [Copernicus Dataspace](https://dataspace.copernicus.eu/)
- [YOLO Documentation](https://docs.ultralytics.com/)
- [Rasterio Documentation](https://rasterio.readthedocs.io/)

---

**Last Updated**: December 2024
**Author**: Sentinel2-YOLO Team
