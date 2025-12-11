# Sentinel-2 Dataset Sources for Aircraft Detection

This guide covers where to download Sentinel-2 satellite imagery for training and validating aircraft detection models.

---

## Official Sources

### 1. **Copernicus Data Space Ecosystem** (Recommended)
**🌐 Website**: https://dataspace.copernicus.eu/

**Best For**: Free, official Sentinel-2 data with web browser download

**Features**:
- Free access to Sentinel-2 L2A (Bottom-of-Atmosphere corrected)
- Graphical interface for easy browsing
- Covers global coverage from 2015-present
- Data available within 3-5 days of acquisition

**How to Use**:
```
1. Visit https://dataspace.copernicus.eu/
2. Create free account (requires email verification)
3. Click "Browse Data"
4. Search by:
   - Geographic area (draw bounding box on map)
   - Date range
   - Cloud coverage threshold
5. Select Sentinel-2 L2A product
6. Download GeoTIFF files
```

**Data Format**:
- Product: `S2A_MSIL2A_*.zip` or `S2B_MSIL2A_*.zip`
- Contains 11 bands in JPEG2000 format
- Cloud mask (SCL band) included
- Size: ~500-800 MB per scene

**Pros**:
✓ Official European Space Agency source
✓ Free tier (10 GB/month)
✓ No authentication delays
✓ Includes cloud mask

**Cons**:
✗ Limited monthly quota (free tier)
✗ Web interface slower for bulk downloads
✗ Requires manual search for each scene

---

### 2. **USGS EarthExplorer**
**🌐 Website**: https://earthexplorer.usgs.gov/

**Best For**: Official US source, Sentinel-2 L1C data

**Features**:
- Maintained by US Geological Survey
- Global Sentinel-2 coverage
- L1C (Top-of-Atmosphere) data available
- Can export coordinate lists for batch processing

**How to Use**:
```
1. Visit https://earthexplorer.usgs.gov/
2. Create free account
3. Use search criteria:
   - Coordinates or bounding box
   - Date range
   - Cloud cover (slider 0-100%)
4. Select Sentinel-2 L1C dataset
5. Order or download directly
```

**Pros**:
✓ Official US government source
✓ Reliable, stable infrastructure
✓ Good for bulk ordering
✓ Metadata well-documented

**Cons**:
✗ L1C data (requires atmospheric correction)
✗ Can be slow for large areas
✗ Registration and login required

---

## Cloud Platforms (Free or Low-Cost)

### 3. **AWS S3 (Amazon Web Services)**
**🌐 Website**: https://registry.opendata.aws/sentinel-2/

**Best For**: Fast bulk downloads, programmatic access, processing on AWS

**Products Available**:
- **Sentinel-2 L1C**: `sentinel-2-l1c` bucket
- **Sentinel-2 L2A**: `sentinel-cogs` (Cloud-Optimized GeoTIFFs)

**Key Features**:
- Free egress within AWS region (us-west-2)
- COG format optimized for cloud access
- Minimal latency, very fast downloads
- Integration with AWS Lambda, EC2, SageMaker

**How to Access**:
```python
# Using boto3 (AWS SDK for Python)
import boto3
from botocore import UNSIGNED
from botocore.config import Config

# No credentials needed for public data
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# List Sentinel-2 L2A scenes
response = s3.list_objects_v2(
    Bucket='sentinel-cogs',
    Prefix='sentinel-2-l2a/2024/01/',  # Year/Month
    Delimiter='/'
)

# Download a specific scene
s3.download_file(
    'sentinel-cogs',
    's3://sentinel-cogs/sentinel-2-l2a/2024/01/S2A_MSIL2A_xxx.tif',
    'local_scene.tif'
)
```

**Tools for Discovery**:
- **Sentinel Hub**: https://www.sentinel-hub.com/
- **STAC API**: https://stacspec.org/ (Spatio-temporal Asset Catalog)

**Pros**:
✓ Fastest downloads (100+ Mbps)
✓ COG format optimized for tiling
✓ No download quota
✓ Easy programmatic access

**Cons**:
✗ Requires AWS account (free tier available)
✗ Charges apply outside us-west-2 region
✗ Need Python/CLI knowledge

---

### 4. **Google Cloud Storage**
**🌐 Website**: https://cloud.google.com/storage/docs/public-datasets/sentinel-2

**Best For**: Integration with Google Earth Engine, BigQuery analysis

**Products Available**:
- **Sentinel-2 L1C**: `gcp-public-data-sentinel-2` bucket
- **Sentinel-2 L2A**: Available via Google Earth Engine

**Key Features**:
- Free storage access for public datasets
- Integration with Google Earth Engine (excellent visualization)
- Easy filtering by cloud coverage, date
- BigQuery for large-scale analysis

**How to Use with Google Earth Engine**:
```javascript
// JavaScript in Earth Engine Code Editor (https://code.earthengine.google.com/)

// Filter Sentinel-2 L2A by location and date
var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(ee.Geometry.Point([longitude, latitude]))
  .filterDate('2024-01-01', '2024-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));

// Select RGB bands for visualization
var visualization = {
  bands: ['B4', 'B3', 'B2'],
  min: 0,
  max: 3000
};

Map.centerObject(collection.first(), 10);
Map.addLayer(collection.first(), visualization, 'Sentinel-2 RGB');

// Export to Google Drive or Cloud Storage
Export.image.toDrive({
  image: collection.first(),
  description: 'sentinel2_scene',
  scale: 10,  // 10m resolution
  region: ee.Geometry.Point([longitude, latitude]).buffer(50000)
});
```

**Pros**:
✓ Excellent web interface (Earth Engine)
✓ Easy cloud filtering
✓ No download limits
✓ Free BigQuery queries (1TB/month)

**Cons**:
✗ Requires Google Cloud account
✗ Earth Engine has learning curve
✗ Data export can be slow

---

### 5. **Microsoft Azure**
**🌐 Website**: https://azure.microsoft.com/en-us/services/open-datasets/

**Best For**: Azure infrastructure integration, large-scale processing

**Products Available**:
- **Sentinel-2 L2A**: `sentinel-2-l2a` dataset

**How to Access**:
```python
# Using Azure SDK
from azure.storage.blob import BlobServiceClient

account_name = "azureopendatastorage"
container_name = "sentinel2"
connection_string = f"BlobEndpoint=https://{account_name}.blob.core.windows.net/;SharedAccessSignature=sv=..."

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

# List blobs
blobs = container_client.list_blobs()
for blob in blobs:
    print(blob.name)
```

**Pros**:
✓ Integrated with Azure ML
✓ Good for large-scale training
✓ Synapse Analytics integration

**Cons**:
✗ Requires Azure account
✗ Less intuitive than GCP/AWS
✗ Limited free tier

---

## Specialized Datasets for Aircraft Detection

### 6. **FAIR1M Dataset** (Aircraft Focus)
**🌐 Website**: https://www.gaofeiwujing.net/fairm

**Description**: 
- Focused on Fine-grained Aerial Image Recognition
- Contains aircraft and other aerial objects
- ~1 million patches extracted from Google Maps
- Already annotated with bounding boxes

**Features**:
- Aircraft, ship, vehicle annotations
- Multi-resolution images
- High-quality labels
- Research-focused (good for transfer learning)

**How to Use**:
```bash
# Download from official source
wget https://www.gaofeiwujing.net/fairm/FAIR1M.zip

# Extract and organize
unzip FAIR1M.zip
# Contains: images/, annotations/, split info
```

**Pros**:
✓ Already annotated with aircraft labels
✓ High-quality bounding boxes
✓ Good for transfer learning
✓ Multi-object detection

**Cons**:
✗ Not raw Sentinel-2 (Google Maps resolution)
✗ Limited free download bandwidth
✗ Not real satellite data (higher resolution)

---

### 7. **HRSC2016 Dataset** (Aerial/Satellite)
**🌐 Website**: https://sites.google.com/site/hrsc2016/home

**Description**:
- High Resolution Ship Collections dataset
- Contains aerial and satellite imagery
- 1000+ annotated ship images (but method works for aircraft)
- Bounding box annotations

**Features**:
- 1:1000 to 1:6000 resolution range
- Grayscale and color images
- Well-established benchmark

**Pros**:
✓ Good for aerial object detection
✓ Established benchmark
✓ Free access

**Cons**:
✗ Ship-focused (not aircraft)
✗ Not Sentinel-2 format
✗ Limited dataset size

---

## Tools for Batch Download

### 8. **Sentinelsat** (Python Package)
**Installation**:
```bash
pip install sentinelsat
```

**Quick Start**:
```python
from sentinelsat import SentinelAPI, geojson_to_wkt
from datetime import date

# Initialize API connection
api = SentinelAPI('your_username', 'your_password', 'https://apihub.copernicus.eu/apihub')

# Define search area (bounding box)
# Format: (minlon, minlat, maxlon, maxlat)
footprint = geojson_to_wkt({
    "type": "Polygon",
    "coordinates": [[[
        [-73.97, 40.76],  # New York example
        [-73.90, 40.76],
        [-73.90, 40.82],
        [-73.97, 40.82],
        [-73.97, 40.76]
    ]]]
})

# Search for Sentinel-2 L2A products
products = api.query(
    footprint,
    date=('20240101', '20240131'),
    platformname='Sentinel-2',
    producttype='S2MSI2A',  # L2A = atmospherically corrected
    cloudcoverpercentage=(0, 20)  # Max 20% cloud cover
)

# Download products
api.download_all(products, directory_path='./sentinel2_data')

# Or download specific product
product_id = list(products.keys())[0]
api.download(product_id, directory_path='./sentinel2_data')
```

**Pros**:
✓ Programmatic batch download
✓ Easy filtering (clouds, date, area)
✓ Works with Copernicus Hub
✓ Well-documented

**Cons**:
✗ Slower than AWS/GCP
✗ Subject to hub quotas
✗ Requires registration with ESA

---

### 9. **rasterio + rio-cogeo** (Local Processing)
**Installation**:
```bash
pip install rasterio rio-cogeo
```

**Usage**:
```python
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

# Open Sentinel-2 GeoTIFF
with rasterio.open('S2A_MSIL2A_scene.tif') as src:
    # Read RGB bands (4=Red, 3=Green, 2=Blue)
    rgb = src.read([4, 3, 2])
    
    # Normalize for display
    rgb_norm = rgb.astype(float) / 3000  # Sentinel-2 is ~12-bit
    
    # Display
    plt.imshow(np.transpose(rgb_norm, (1, 2, 0)))
    plt.show()
    
    # Get metadata
    print(f"Size: {src.width} x {src.height}")
    print(f"Bands: {src.count}")
    print(f"CRS: {src.crs}")
```

---

## Recommended Workflow for Aircraft Detection

### **Option A: Quick Start (Free, Small-Scale)**
```
1. Use Google Earth Engine
   ↓
2. Download 10-20 scenes with low cloud cover
   ↓
3. Crop to areas with visible aircraft (ports, airports)
   ↓
4. Use CVAT or Roboflow for manual annotation
   ↓
5. Train model on collected data
```

### **Option B: Production Scale (Cost ~$50-200)**
```
1. Use Copernicus Dataspace (free tier: 10GB/month)
   ↓
2. Download 100+ scenes across multiple regions
   ↓
3. Use semi-automated annotation:
   - Existing model (FAIR1M transfer learning)
   - Manual review and correction
   ↓
4. Organize in train/val/test splits
   ↓
5. Train fine-tuned model
```

### **Option C: Enterprise Scale (Cost $500-5000)**
```
1. Set up AWS/GCP account
   ↓
2. Automated batch download (10,000+ scenes)
   ↓
3. Cloud-based processing pipeline
   ↓
4. Commercial annotation service (Scale.com, Labelbox)
   ↓
5. Distributed training on cloud infrastructure
```

---

## Specific Regions Recommended for Aircraft Detection

### **High Aircraft Activity Regions**
1. **Middle East** (Dubai, Qatar)
   - Coordinates: `[50.0, 24.0, 56.0, 27.0]`
   - Dubai International Airport, numerous ports
   
2. **Southeast Asia** (Singapore)
   - Coordinates: `[103.5, 1.0, 104.5, 1.7]`
   - Changi Airport, major shipping hub
   
3. **Mediterranean** (Port of Rotterdam)
   - Coordinates: `[3.8, 51.9, 4.5, 52.0]`
   - Busiest European port, frequent aircraft
   
4. **US East Coast** (Port of Long Beach)
   - Coordinates: `[-118.3, 33.7, -118.0, 33.9]`
   - Major port, aircraft routes overhead
   
5. **Asia-Pacific** (Hong Kong)
   - Coordinates: `[113.8, 22.2, 114.3, 22.4]`
   - International airport, harbor

---

## Data Organization for Training

### **Recommended Directory Structure**
```
sentinel2-yolo-aircraft/
├── data/
│   ├── raw/                    # Original downloaded data
│   │   ├── S2A_MSIL2A_xxx.zip
│   │   ├── S2A_MSIL2A_yyy.zip
│   │   └── ...
│   │
│   ├── processed/              # Extracted and processed
│   │   ├── scene_001.tif
│   │   ├── scene_002.tif
│   │   └── ...
│   │
│   ├── train/                  # Training set
│   │   ├── scene_001.tif
│   │   ├── scene_001.txt       # YOLO labels
│   │   ├── scene_002.tif
│   │   ├── scene_002.txt
│   │   └── ...
│   │
│   ├── val/                    # Validation set
│   │   ├── scene_101.tif
│   │   ├── scene_101.txt
│   │   └── ...
│   │
│   ├── test/                   # Test set
│   │   ├── scene_201.tif
│   │   ├── scene_201.txt
│   │   └── ...
│   │
│   └── data.yaml               # YOLO dataset config
│
└── scripts/
    └── download_sentinel2.py    # Custom download script
```

---

## Cost Comparison

| Source | Cost (100 Scenes) | Speed | Quality | Ease |
|--------|-----------------|-------|---------|------|
| Copernicus (Free Tier) | Free (10GB/mo limit) | Slow | L2A ✓ | Medium |
| USGS EarthExplorer | Free | Medium | L1C | Medium |
| AWS S3 | Free* ($0-50) | Fast | L2A ✓ | Medium |
| Google Earth Engine | Free ($0-50) | Fast | L2A ✓ | Medium |
| Azure | Free* ($0-50) | Fast | L2A ✓ | Medium |
| FAIR1M (Pre-annotated) | Free | N/A | Aerial | Easy |

*Free tier applies; charges for egress/processing in some cases

---

## Quick Reference: Command Examples

### **Download with Sentinelsat**
```bash
# Install
pip install sentinelsat

# Download script
python << 'EOF'
from sentinelsat import SentinelAPI, geojson_to_wkt

api = SentinelAPI('user', 'password')
footprint = geojson_to_wkt({"type": "Polygon", "coordinates": [[[-73.97, 40.76], [-73.90, 40.76], [-73.90, 40.82], [-73.97, 40.82], [-73.97, 40.76]]]})
products = api.query(footprint, date=('20240101', '20240131'), platformname='Sentinel-2', producttype='S2MSI2A', cloudcoverpercentage=(0, 20))
api.download_all(products)
EOF
```

### **Download with AWS CLI**
```bash
# Install AWS CLI
pip install awscli

# List available scenes
aws s3 ls s3://sentinel-cogs/sentinel-2-l2a/2024/01/ --no-sign-request

# Download scene
aws s3 cp s3://sentinel-cogs/sentinel-2-l2a/2024/01/S2A_MSIL2A_xxx.tif . --no-sign-request
```

### **Download with gsutil (Google Cloud)**
```bash
# Install gsutil
curl https://sdk.cloud.google.com | bash

# List and download
gsutil ls gs://gcp-public-data-sentinel-2/tiles/32/U/UD/2024/
gsutil -m cp gs://gcp-public-data-sentinel-2/tiles/.../S2A*.tif .
```

---

## Next Steps

1. **Choose a data source** based on your needs (start with Copernicus or Google Earth Engine)
2. **Download 50-100 scenes** with low cloud cover over airports/ports
3. **Crop scenes** to relevant areas with visible aircraft
4. **Annotate** using CVAT, Roboflow, or manual annotation tools
5. **Organize** into train/val/test directories
6. **Use the training notebook** in this repository to train your model

**See Also**:
- [SENTINEL2_MIGRATION.md](./SENTINEL2_MIGRATION.md) - Using Sentinel-2 in notebooks
- [training_notebook.ipynb](./notebooks/training_notebook.ipynb) - Training workflow
- [inference_notebook.ipynb](./notebooks/inference_notebook.ipynb) - Inference workflow

---

**Last Updated**: December 2024
