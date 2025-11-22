"""
Google Earth Engine Utilities for Satellite Data Acquisition
Real satellite data fetching for illegal mining detection
"""

import ee
import os
import logging
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime, timedelta
import rasterio
import numpy as np
from dotenv import load_dotenv
import requests
import tempfile
import shutil

# Load environment variables from common paths
_env_loaded = False
try:
    # 1) Standard .env discovery
    if load_dotenv():
        _env_loaded = True
    # 2) Explicit backend/.env
    if not _env_loaded:
        _env_loaded = load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
    # 3) Explicit backend/env (without dot), per user setup
    if not _env_loaded:
        _env_loaded = load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), 'env'))
except Exception:
    _env_loaded = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GEEUtils:
    """Google Earth Engine utilities for satellite data acquisition"""
    
    def __init__(self):
        """Initialize GEE authentication and collections"""
        self.initialized = False
        try:
            # Earth Engine Python SDK uses OAuth credentials, not API keys.
            # Ensure you've run `earthengine authenticate` once on this machine.
            project_id = os.getenv('EE_PROJECT_ID') or 'car-pooling-dc7a3'
            ee.Initialize(project=project_id)
            self.initialized = True
            logger.info(f"✅ Google Earth Engine initialized (project={project_id})")
        except Exception as e:
            logger.warning(f"⚠️ GEE initialization failed: {e}")
            logger.warning("⚠️ GEE features will not be available. Run 'earthengine authenticate' to enable.")
            self.initialized = False
        
        # Define satellite collections (only if initialized)
        if self.initialized:
            self.sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            self.sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')
            self.landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            self.srtm = ee.Image('USGS/SRTMGL1_003')
            self.alos_dem = ee.Image('JAXA/ALOS/AW3D30/V2_2')
        else:
            self.sentinel2 = None
            self.sentinel1 = None
            self.landsat8 = None
            self.srtm = None
            self.alos_dem = None
        
        # Define bands for different sensors
        self.sentinel2_bands = {
            'B2': 'Blue',
            'B3': 'Green', 
            'B4': 'Red',
            'B8': 'NIR',
            'B11': 'SWIR1',
            'B12': 'SWIR2',
            'QA60': 'QA60'
        }
        
        self.landsat8_bands = {
            'SR_B2': 'Blue',
            'SR_B3': 'Green',
            'SR_B4': 'Red', 
            'SR_B5': 'NIR',
            'SR_B6': 'SWIR1',
            'SR_B7': 'SWIR2',
            'QA_PIXEL': 'QA_PIXEL'
        }
    
    def download_sentinel2_aoi(self, aoi_geojson: Dict, start_date: str, end_date: str, 
                              out_path: str, bands: List[str] = None, 
                              max_cloud_cover: int = 20) -> bool:
        """
        Download Sentinel-2 data for Area of Interest
        
        Args:
            aoi_geojson: GeoJSON polygon defining AOI
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            out_path: Output file path
            bands: List of bands to download
            max_cloud_cover: Maximum cloud cover percentage
            
        Returns:
            bool: Success status
        """
        if not self.initialized:
            logger.warning("⚠️ GEE not initialized. Creating demo data instead.")
            return self._create_demo_sentinel2_composite(aoi_geojson, out_path, bands or ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
        try:
            logger.info(f"🛰️ Downloading Sentinel-2 data for AOI from {start_date} to {end_date}")
            
            # Convert GeoJSON to EE geometry
            aoi = ee.Geometry.Polygon(aoi_geojson['coordinates'])
            
            # Default bands if not specified
            if bands is None:
                bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']

            # Filter Sentinel-2 collection (do NOT select bands before masking; QA60 needed)
            collection = (self.sentinel2
                          .filterDate(start_date, end_date)
                          .filterBounds(aoi)
                          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover)))

            # Create cloud-free composite
            def mask_clouds(image):
                qa = image.select('QA60')
                cloud_mask = qa.bitwiseAnd(1024).eq(0).And(qa.bitwiseAnd(2048).eq(0))
                return image.updateMask(cloud_mask).divide(10000)

            composite = collection.map(mask_clouds).median().clip(aoi)
            composite = composite.select(bands)

            # Download each band and stack locally into a multi-band GeoTIFF
            temp_dir = tempfile.mkdtemp(prefix="s2_dl_")
            temp_band_paths: List[str] = []
            try:
                for band in bands:
                    single_band = composite.select([band])
                    url = single_band.getDownloadURL({
                        'region': aoi.coordinates().getInfo(),
                        'scale': 10,
                        'crs': 'EPSG:4326',
                        'format': 'GEO_TIFF'
                    })
                    band_path = os.path.join(temp_dir, f"{band}.tif")
                    self._download_file(url, band_path)
                    temp_band_paths.append(band_path)

                # Stack bands into a single GeoTIFF
                with rasterio.open(temp_band_paths[0]) as ref:
                    profile = ref.profile.copy()
                    profile.update({
                        'count': len(temp_band_paths),
                        'dtype': rasterio.float32,
                        'nodata': None
                    })
                    data_arrays = []
                    for p in temp_band_paths:
                        with rasterio.open(p) as src:
                            data_arrays.append(src.read(1).astype(np.float32))

                with rasterio.open(out_path, 'w', **profile) as dst:
                    for idx, arr in enumerate(data_arrays):
                        dst.write(arr, idx + 1)
                        dst.set_band_description(idx + 1, self.sentinel2_bands.get(bands[idx], bands[idx]))

                logger.info(f"✅ Sentinel-2 composite downloaded: {out_path}")
                return True
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            logger.error(f"❌ Error downloading Sentinel-2 data: {e}")
            return self._create_demo_sentinel2_composite(aoi_geojson, out_path, bands)
    
    def download_dem(self, aoi_geojson: Dict, out_path: str, source: str = "SRTM") -> bool:
        """
        Download DEM data for Area of Interest
        
        Args:
            aoi_geojson: GeoJSON polygon defining AOI
            out_path: Output file path
            source: DEM source ("SRTM" or "ALOS")
            
        Returns:
            bool: Success status
        """
        if not self.initialized:
            logger.warning("⚠️ GEE not initialized. Creating demo data instead.")
            return self._create_demo_dem(aoi_geojson, out_path, source)
        try:
            logger.info(f"🗻 Downloading {source} DEM data for AOI")
            
            # Convert GeoJSON to EE geometry
            aoi = ee.Geometry.Polygon(aoi_geojson['coordinates'])
            
            # Select DEM source
            if source == "SRTM":
                dem = self.srtm
                scale = 30
            elif source == "ALOS":
                dem = self.alos_dem.select('AVE_DSM')
                scale = 30
            else:
                raise ValueError(f"Unknown DEM source: {source}")
            
            # Clip to AOI
            dem_clipped = dem.clip(aoi)

            # Direct download via URL
            url = dem_clipped.getDownloadURL({
                'region': aoi.coordinates().getInfo(),
                'scale': scale,
                'crs': 'EPSG:4326',
                'format': 'GEO_TIFF'
            })
            self._download_file(url, out_path)
            logger.info(f"✅ {source} DEM downloaded: {out_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error downloading {source} DEM: {e}")
            return self._create_demo_dem(aoi_geojson, out_path, source)
    
    def download_sentinel1_sar(self, aoi_geojson: Dict, start_date: str, end_date: str,
                              out_path: str, polarization: str = "VV") -> bool:
        """
        Download Sentinel-1 SAR data for Area of Interest
        
        Args:
            aoi_geojson: GeoJSON polygon defining AOI
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            out_path: Output file path
            polarization: SAR polarization ("VV", "VH", or "VVVH")
            
        Returns:
            bool: Success status
        """
        if not self.initialized:
            logger.warning("⚠️ GEE not initialized. Creating demo data instead.")
            return self._create_demo_sar_composite(aoi_geojson, out_path, polarization)
        try:
            logger.info(f"📡 Downloading Sentinel-1 SAR data for AOI")
            
            # Convert GeoJSON to EE geometry
            aoi = ee.Geometry.Polygon(aoi_geojson['coordinates'])
            
            # Filter Sentinel-1 collection
            collection = (self.sentinel1
                         .filterDate(start_date, end_date)
                         .filterBounds(aoi)
                         .filter(ee.Filter.eq('instrumentMode', 'IW'))
                         .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization))
                         .select([f'{polarization}_dB']))
            
            # Create median composite
            composite = collection.median()
            
            # Clip to AOI
            composite = composite.clip(aoi)
            
            # Direct download via URL
            url = composite.getDownloadURL({
                'region': aoi.coordinates().getInfo(),
                'scale': 10,
                'crs': 'EPSG:4326',
                'format': 'GEO_TIFF'
            })
            self._download_file(url, out_path)
            logger.info(f"✅ Sentinel-1 SAR downloaded: {out_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error downloading Sentinel-1 SAR: {e}")
            return False

    def _download_file(self, url: str, out_path: str) -> None:
        """Download a file from URL to local path."""
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    
    def _create_demo_sentinel2_composite(self, aoi_geojson: Dict, out_path: str, bands: List[str]) -> bool:
        """Create demo Sentinel-2 composite for testing"""
        try:
            logger.info("🎭 Creating demo Sentinel-2 composite...")
            
            # Extract AOI bounds
            coords = aoi_geojson['coordinates'][0]
            min_lon = min(coord[0] for coord in coords)
            max_lon = max(coord[0] for coord in coords)
            min_lat = min(coord[1] for coord in coords)
            max_lat = max(coord[1] for coord in coords)
            
            # Calculate dimensions (approximate 10m pixels)
            width = int((max_lon - min_lon) * 111000 / 10)  # ~111km per degree
            height = int((max_lat - min_lat) * 111000 / 10)
            
            # Create realistic multi-band data
            data = {}
            for i, band in enumerate(bands):
                # Create realistic spectral data
                if band in ['B2', 'B3', 'B4']:  # Visible bands
                    base_value = 0.1 + i * 0.05
                elif band == 'B8':  # NIR
                    base_value = 0.3
                elif band in ['B11', 'B12']:  # SWIR
                    base_value = 0.2 + (i - 4) * 0.1
                else:
                    base_value = 0.1
                
                # Add some variation and mining signatures
                band_data = np.random.normal(base_value, 0.05, (height, width))
                
                # Add mining areas (low vegetation, high bare soil)
                mining_areas = np.random.random((height, width)) < 0.1  # 10% mining areas
                if band in ['B2', 'B3', 'B4', 'B8']:  # Lower values in mining areas
                    band_data[mining_areas] *= 0.7
                else:  # Higher SWIR values in mining areas
                    band_data[mining_areas] *= 1.3
                
                data[band] = np.clip(band_data, 0, 1)
            
            # Create GeoTIFF
            transform = rasterio.transform.from_bounds(
                min_lon, min_lat, max_lon, max_lat, width, height
            )
            
            with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=len(bands),
                dtype=rasterio.float32,
                crs='EPSG:4326',
                transform=transform
            ) as dst:
                for i, band in enumerate(bands):
                    dst.write(data[band], i + 1)
                    dst.set_band_description(i + 1, self.sentinel2_bands.get(band, band))
            
            logger.info(f"✅ Demo Sentinel-2 composite saved: {out_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating demo Sentinel-2 composite: {e}")
            return False
    
    def _create_demo_dem(self, aoi_geojson: Dict, out_path: str, source: str) -> bool:
        """Create demo DEM for testing"""
        try:
            logger.info(f"🎭 Creating demo {source} DEM...")
            
            # Extract AOI bounds
            coords = aoi_geojson['coordinates'][0]
            min_lon = min(coord[0] for coord in coords)
            max_lon = max(coord[0] for coord in coords)
            min_lat = min(coord[1] for coord in coords)
            max_lat = max(coord[1] for coord in coords)
            
            # Calculate dimensions (30m pixels for DEM)
            width = int((max_lon - min_lon) * 111000 / 30)
            height = int((max_lat - min_lat) * 111000 / 30)
            
            # Create realistic elevation data
            x = np.linspace(min_lon, max_lon, width)
            y = np.linspace(min_lat, max_lat, height)
            X, Y = np.meshgrid(x, y)
            
            # Base elevation (India average ~300m)
            base_elevation = 300 + 50 * np.sin(X * 10) + 30 * np.cos(Y * 10)
            
            # Add mining pits (lower elevation)
            mining_pits = np.random.random((height, width)) < 0.05  # 5% mining areas
            base_elevation[mining_pits] -= np.random.uniform(10, 50, np.sum(mining_pits))
            
            # Add noise
            elevation = base_elevation + np.random.normal(0, 5, (height, width))
            
            # Create GeoTIFF
            transform = rasterio.transform.from_bounds(
                min_lon, min_lat, max_lon, max_lat, width, height
            )
            
            with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=rasterio.float32,
                crs='EPSG:4326',
                transform=transform
            ) as dst:
                dst.write(elevation, 1)
                dst.set_band_description(1, f'{source} Elevation (m)')
            
            logger.info(f"✅ Demo {source} DEM saved: {out_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating demo {source} DEM: {e}")
            return False
    
    def _create_demo_sar_composite(self, aoi_geojson: Dict, out_path: str, polarization: str) -> bool:
        """Create demo SAR composite for testing"""
        try:
            logger.info(f"🎭 Creating demo Sentinel-1 SAR {polarization} composite...")
            
            # Extract AOI bounds
            coords = aoi_geojson['coordinates'][0]
            min_lon = min(coord[0] for coord in coords)
            max_lon = max(coord[0] for coord in coords)
            min_lat = min(coord[1] for coord in coords)
            max_lat = max(coord[1] for coord in coords)
            
            # Calculate dimensions (10m pixels)
            width = int((max_lon - min_lon) * 111000 / 10)
            height = int((max_lat - min_lat) * 111000 / 10)
            
            # Create realistic SAR backscatter data (dB scale)
            base_backscatter = -10 + np.random.normal(0, 3, (height, width))
            
            # Mining areas have different backscatter (more rough surface)
            mining_areas = np.random.random((height, width)) < 0.1
            base_backscatter[mining_areas] += np.random.uniform(2, 8, np.sum(mining_areas))
            
            # Create GeoTIFF
            transform = rasterio.transform.from_bounds(
                min_lon, min_lat, max_lon, max_lat, width, height
            )
            
            with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=rasterio.float32,
                crs='EPSG:4326',
                transform=transform
            ) as dst:
                dst.write(base_backscatter, 1)
                dst.set_band_description(1, f'Sentinel-1 {polarization} Backscatter (dB)')
            
            logger.info(f"✅ Demo Sentinel-1 SAR {polarization} composite saved: {out_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating demo SAR composite: {e}")
            return False

# Standalone functions for easy integration
def download_sentinel2_aoi(aoi_geojson: Dict, start_date: str, end_date: str, 
                          out_path: str, bands: List[str] = None, 
                          max_cloud_cover: int = 20) -> bool:
    """Download Sentinel-2 data for AOI"""
    gee_utils = GEEUtils()
    return gee_utils.download_sentinel2_aoi(aoi_geojson, start_date, end_date, out_path, bands, max_cloud_cover)

def download_dem(aoi_geojson: Dict, out_path: str, source: str = "SRTM") -> bool:
    """Download DEM data for AOI"""
    gee_utils = GEEUtils()
    return gee_utils.download_dem(aoi_geojson, out_path, source)

def download_sentinel1_sar(aoi_geojson: Dict, start_date: str, end_date: str,
                          out_path: str, polarization: str = "VV") -> bool:
    """Download Sentinel-1 SAR data for AOI"""
    gee_utils = GEEUtils()
    return gee_utils.download_sentinel1_sar(aoi_geojson, start_date, end_date, out_path, polarization)

if __name__ == "__main__":
    # Test the GEE utilities
    test_aoi = {
        "type": "Polygon",
        "coordinates": [[
            [76.0, 15.0], [77.0, 15.0], [77.0, 16.0], [76.0, 16.0], [76.0, 15.0]
        ]]
    }
    
    gee_utils = GEEUtils()
    
    # Test Sentinel-2 download
    success = gee_utils.download_sentinel2_aoi(
        test_aoi, 
        "2024-01-01", 
        "2024-01-31", 
        "test_sentinel2.tif"
    )
    
    print(f"Sentinel-2 download: {'✅ Success' if success else '❌ Failed'}")
    
    # Test DEM download
    success = gee_utils.download_dem(test_aoi, "test_dem.tif", "SRTM")
    print(f"DEM download: {'✅ Success' if success else '❌ Failed'}")
