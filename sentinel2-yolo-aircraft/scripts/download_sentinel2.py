#!/usr/bin/env python3
"""
Sentinel-2 Dataset Download Tool

Download Sentinel-2 imagery from various sources for aircraft detection training.

Usage:
    python download_sentinel2.py --method copernicus --bbox 3.8 51.9 4.5 52.0 --date 2024-01-01 2024-12-31 --output data/
    python download_sentinel2.py --method aws --bbox -118.3 33.7 -118.0 33.9 --output data/
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import subprocess
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Sentinel2Downloader:
    """Download Sentinel-2 imagery from various sources"""
    
    def __init__(self, output_dir: str = 'data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def check_dependencies(self, method: str):
        """Check if required packages are installed"""
        dependencies = {
            'copernicus': ['sentinelsat'],
            'aws': ['boto3'],
            'gcp': ['google-cloud-storage'],
            'azure': ['azure-storage-blob'],
        }
        
        if method not in dependencies:
            return True
        
        required = dependencies[method]
        missing = []
        
        for pkg in required:
            try:
                __import__(pkg.replace('-', '_'))
            except ImportError:
                missing.append(pkg)
        
        if missing:
            logger.warning(f"Missing packages for {method}: {', '.join(missing)}")
            logger.info(f"Install with: pip install {' '.join(missing)}")
            return False
        
        return True
    
    def download_copernicus(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: Tuple[str, str],
        cloud_cover: float = 20,
        username: Optional[str] = None,
        password: Optional[str] = None
    ) -> List[Path]:
        """
        Download from Copernicus Data Space Ecosystem
        
        Args:
            bbox: (minlon, minlat, maxlon, maxlat)
            date_range: (start_date, end_date) as 'YYYY-MM-DD'
            cloud_cover: Maximum cloud coverage percentage (0-100)
            username: Copernicus username (prompted if not provided)
            password: Copernicus password (prompted if not provided)
        
        Returns:
            List of downloaded file paths
        """
        try:
            from sentinelsat import SentinelAPI, geojson_to_wkt
        except ImportError:
            logger.error("sentinelsat not installed. Install with: pip install sentinelsat")
            return []
        
        if not username or not password:
            logger.warning("Username and password required for Copernicus")
            logger.info("Get free account at: https://dataspace.copernicus.eu/")
            username = input("Enter Copernicus username: ")
            password = input("Enter Copernicus password: ")
        
        logger.info(f"Connecting to Copernicus Hub as {username}...")
        api = SentinelAPI(username, password, 'https://dataspace.copernicus.eu/api/SentinelHub')
        
        # Create polygon from bbox
        minlon, minlat, maxlon, maxlat = bbox
        footprint = geojson_to_wkt({
            "type": "Polygon",
            "coordinates": [[
                [minlon, minlat],
                [maxlon, minlat],
                [maxlon, maxlat],
                [minlon, maxlat],
                [minlon, minlat]
            ]]
        })
        
        logger.info(f"Searching for Sentinel-2 L2A scenes...")
        logger.info(f"  Area: {bbox}")
        logger.info(f"  Date: {date_range[0]} to {date_range[1]}")
        logger.info(f"  Max cloud cover: {cloud_cover}%")
        
        products = api.query(
            footprint,
            date=(date_range[0].replace('-', ''), date_range[1].replace('-', '')),
            platformname='Sentinel-2',
            producttype='S2MSI2A',  # L2A = bottom-of-atmosphere corrected
            cloudcoverpercentage=(0, cloud_cover)
        )
        
        if not products:
            logger.warning("No products found matching criteria")
            return []
        
        logger.info(f"Found {len(products)} products")
        
        # Download products
        downloaded_files = []
        for i, (product_id, product_info) in enumerate(products.items(), 1):
            logger.info(f"[{i}/{len(products)}] Downloading {product_info['title']}...")
            
            try:
                api.download(product_id, directory_path=str(self.output_dir))
                downloaded_files.append(self.output_dir / f"{product_info['title']}.zip")
            except Exception as e:
                logger.error(f"Failed to download {product_info['title']}: {e}")
        
        logger.info(f"Downloaded {len(downloaded_files)} products to {self.output_dir}")
        return downloaded_files
    
    def download_aws(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: Optional[Tuple[str, str]] = None,
        use_cog: bool = True
    ) -> List[Path]:
        """
        Download from AWS S3 (sentinel-cogs bucket)
        
        Args:
            bbox: (minlon, minlat, maxlon, maxlat)
            date_range: Optional (start_date, end_date)
            use_cog: Use Cloud-Optimized GeoTIFF format (faster)
        
        Returns:
            List of downloaded file paths
        """
        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config
        except ImportError:
            logger.error("boto3 not installed. Install with: pip install boto3")
            return []
        
        logger.info("Connecting to AWS S3 (no authentication required)...")
        
        # Create S3 client without authentication
        s3 = boto3.client(
            's3',
            region_name='us-west-2',
            config=Config(signature_version=UNSIGNED)
        )
        
        bucket = 'sentinel-cogs'
        prefix = 'sentinel-2-l2a/'
        
        if date_range:
            year = date_range[0].split('-')[0]
            month = date_range[0].split('-')[1]
            prefix += f'{year}/{month}/'
        
        logger.info(f"Listing objects in s3://{bucket}/{prefix}...")
        
        try:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=50)
        except Exception as e:
            logger.error(f"Failed to list objects: {e}")
            return []
        
        if 'Contents' not in response:
            logger.warning("No objects found")
            return []
        
        # Download files
        downloaded_files = []
        files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.tif')]
        
        for i, key in enumerate(files[:10], 1):  # Limit to first 10 for demo
            filename = Path(key).name
            filepath = self.output_dir / filename
            
            logger.info(f"[{i}/{len(files[:10])}] Downloading {filename}...")
            
            try:
                s3.download_file(bucket, key, str(filepath))
                downloaded_files.append(filepath)
                logger.info(f"  Size: {filepath.stat().st_size / (1024**2):.1f} MB")
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")
        
        logger.info(f"Downloaded {len(downloaded_files)} files to {self.output_dir}")
        return downloaded_files
    
    def download_gcp(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: Optional[Tuple[str, str]] = None
    ) -> List[Path]:
        """
        Download from Google Cloud Storage
        
        Args:
            bbox: (minlon, minlat, maxlon, maxlat)
            date_range: Optional (start_date, end_date)
        
        Returns:
            List of downloaded file paths
        """
        logger.info("Google Cloud Storage download requires Earth Engine or custom setup")
        logger.info("Recommended: Use Google Earth Engine instead")
        logger.info("  https://code.earthengine.google.com/")
        logger.info("  See SENTINEL2_DATA_SOURCES.md for example code")
        return []
    
    def download_azure(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: Optional[Tuple[str, str]] = None
    ) -> List[Path]:
        """
        Download from Azure Storage
        
        Args:
            bbox: (minlon, minlat, maxlon, maxlat)
            date_range: Optional (start_date, end_date)
        
        Returns:
            List of downloaded file paths
        """
        logger.info("Azure download requires account setup")
        logger.info("See SENTINEL2_DATA_SOURCES.md for instructions")
        return []
    
    def extract_archives(self, archives: List[Path]) -> List[Path]:
        """Extract downloaded ZIP files and organize"""
        extracted_files = []
        
        for archive in archives:
            if not archive.exists():
                logger.warning(f"Archive not found: {archive}")
                continue
            
            if archive.suffix == '.zip':
                logger.info(f"Extracting {archive.name}...")
                
                # Create extraction directory
                extract_dir = archive.parent / archive.stem
                extract_dir.mkdir(exist_ok=True)
                
                try:
                    import zipfile
                    with zipfile.ZipFile(archive, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    
                    # Find GeoTIFF files
                    tiff_files = list(extract_dir.rglob('*.tif')) + list(extract_dir.rglob('*.tiff'))
                    extracted_files.extend(tiff_files)
                    
                    logger.info(f"  Extracted {len(tiff_files)} GeoTIFF files")
                    
                except Exception as e:
                    logger.error(f"Failed to extract {archive}: {e}")
        
        return extracted_files
    
    def organize_for_training(self, scenes: List[Path], split_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1)):
        """
        Organize scenes into train/val/test directories
        
        Args:
            scenes: List of scene file paths
            split_ratio: (train%, val%, test%)
        """
        if not scenes:
            logger.warning("No scenes to organize")
            return
        
        train_dir = self.output_dir / 'train'
        val_dir = self.output_dir / 'val'
        test_dir = self.output_dir / 'test'
        
        for dir_ in [train_dir, val_dir, test_dir]:
            dir_.mkdir(exist_ok=True)
        
        # Split scenes
        num_scenes = len(scenes)
        num_train = int(num_scenes * split_ratio[0])
        num_val = int(num_scenes * split_ratio[1])
        
        logger.info(f"Organizing {num_scenes} scenes into train/val/test...")
        logger.info(f"  Train: {num_train} ({split_ratio[0]*100:.0f}%)")
        logger.info(f"  Val: {num_val} ({split_ratio[1]*100:.0f}%)")
        logger.info(f"  Test: {num_scenes - num_train - num_val} ({split_ratio[2]*100:.0f}%)")
        
        for i, scene in enumerate(sorted(scenes)):
            if i < num_train:
                dest = train_dir / scene.name
            elif i < num_train + num_val:
                dest = val_dir / scene.name
            else:
                dest = test_dir / scene.name
            
            # Copy or symlink
            try:
                import shutil
                shutil.copy2(scene, dest)
            except Exception as e:
                logger.error(f"Failed to copy {scene}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Download Sentinel-2 imagery for aircraft detection training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Download from Copernicus (Rotterdam)
  python download_sentinel2.py --method copernicus \\
    --bbox 3.8 51.9 4.5 52.0 \\
    --date 2024-01-01 2024-12-31 \\
    --output data/

  # Download from AWS (Long Beach)
  python download_sentinel2.py --method aws \\
    --bbox -118.3 33.7 -118.0 33.9 \\
    --output data/

  # Show available regions
  python download_sentinel2.py --regions
        '''
    )
    
    parser.add_argument('--method', choices=['copernicus', 'aws', 'gcp', 'azure', 'all'],
                       default='aws', help='Download method (default: aws)')
    
    parser.add_argument('--bbox', type=float, nargs=4, metavar=('MIN_LON', 'MIN_LAT', 'MAX_LON', 'MAX_LAT'),
                       help='Bounding box (minlon minlat maxlon maxlat)')
    
    parser.add_argument('--date', type=str, nargs=2, metavar=('START', 'END'),
                       default=['2024-01-01', '2024-12-31'],
                       help='Date range YYYY-MM-DD (default: 2024-01-01 to 2024-12-31)')
    
    parser.add_argument('--cloud-cover', type=float, default=20,
                       help='Maximum cloud coverage %% (default: 20)')
    
    parser.add_argument('--output', type=str, default='data/',
                       help='Output directory (default: data/)')
    
    parser.add_argument('--username', type=str,
                       help='Copernicus username (prompted if not provided)')
    
    parser.add_argument('--password', type=str,
                       help='Copernicus password (prompted if not provided)')
    
    parser.add_argument('--organize', action='store_true',
                       help='Organize downloaded scenes into train/val/test')
    
    parser.add_argument('--regions', action='store_true',
                       help='Show recommended regions for aircraft detection')
    
    args = parser.parse_args()
    
    # Show regions
    if args.regions:
        regions = {
            'Rotterdam (Netherlands)': (3.8, 51.9, 4.5, 52.0),
            'Dubai (UAE)': (50.0, 24.0, 56.0, 27.0),
            'Singapore': (103.5, 1.0, 104.5, 1.7),
            'Long Beach (USA)': (-118.3, 33.7, -118.0, 33.9),
            'Hong Kong': (113.8, 22.2, 114.3, 22.4),
            'Shanghai (China)': (121.2, 30.7, 121.8, 31.0),
            'Hamburg (Germany)': (9.8, 53.5, 10.3, 53.6),
        }
        
        print("\n📍 Recommended Regions for Aircraft Detection:\n")
        for name, bbox in regions.items():
            print(f"{name}")
            print(f"  Bbox: {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
        
        return
    
    # Validate arguments
    if not args.bbox:
        logger.error("--bbox required. Use --regions to see options")
        sys.exit(1)
    
    # Download
    downloader = Sentinel2Downloader(args.output)
    
    downloaded = []
    
    if args.method in ['copernicus', 'all']:
        if downloader.check_dependencies('copernicus'):
            logger.info("\n=== Downloading from Copernicus ===")
            files = downloader.download_copernicus(
                bbox=tuple(args.bbox),
                date_range=tuple(args.date),
                cloud_cover=args.cloud_cover,
                username=args.username,
                password=args.password
            )
            extracted = downloader.extract_archives(files)
            downloaded.extend(extracted)
    
    if args.method in ['aws', 'all']:
        if downloader.check_dependencies('aws'):
            logger.info("\n=== Downloading from AWS ===")
            files = downloader.download_aws(
                bbox=tuple(args.bbox),
                date_range=tuple(args.date) if args.date else None
            )
            downloaded.extend(files)
    
    if args.method in ['gcp']:
        downloader.download_gcp(bbox=tuple(args.bbox), date_range=tuple(args.date))
    
    if args.method in ['azure']:
        downloader.download_azure(bbox=tuple(args.bbox), date_range=tuple(args.date))
    
    # Organize if requested
    if args.organize and downloaded:
        logger.info("\n=== Organizing for Training ===")
        downloader.organize_for_training(downloaded)
    
    logger.info("\n✓ Download complete!")
    logger.info(f"Data saved to: {downloader.output_dir}")


if __name__ == '__main__':
    main()
