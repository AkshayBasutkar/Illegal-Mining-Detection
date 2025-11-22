# Illegal Mining Detection System - Complete Technical Documentation

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Workflow Diagram](#workflow-diagram)
3. [Module Deep Dive with Code](#module-deep-dive-with-code)
4. [Data Flow](#data-flow)
5. [API Endpoints](#api-endpoints)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ILLEGAL MINING DETECTION SYSTEM                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  React Application (port 5173)                                              │
│  ├── Leaflet Map Visualization                                              │
│  ├── Interactive 2D/3D Views                                                │
│  ├── Report Generation UI                                                   │
│  └── API Integration                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ HTTP/REST API
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BACKEND LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  FastAPI Application (app.py) - port 8000                                   │
│  ├── /api/mining-boundaries     → Get legal lease boundaries                │
│  ├── /api/satellite-data        → Get detected mining areas                 │
│  ├── /api/analyze/illegal-mining-detection → Run full analysis              │
│  └── /api/illegal-mining-results/{id} → Get analysis results                │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
┌──────────────────────┐  ┌──────────────────┐  ┌─────────────────────┐
│  DATA ACQUISITION    │  │  PREPROCESSING   │  │  DETECTION          │
│  (gee_utils.py)      │  │  (preprocess.py) │  │  (detect_indices.py)│
│                      │  │                  │  │                     │
│ • Sentinel-2 imagery │  │ • Reproject      │  │ • Spectral indices  │
│ • DEM data           │  │ • Clip           │  │ • Threshold masks   │
│ • SAR data           │  │ • Normalize      │  │ • Polygonization    │
│ • Cloud filtering    │  │ • Align rasters  │  │ • Area calculation  │
└──────────────────────┘  └──────────────────┘  └─────────────────────┘
                                     │
                                     ▼
                          ┌────────────────────────┐
                          │  ILLEGAL MINING        │
                          │  ANALYSIS              │
                          │  (compare_with_lease.py)│
                          │                        │
                          │ • Spatial overlay      │
                          │ • Intersection calc    │
                          │ • Legal/Illegal class  │
                          │ • Statistics           │
                          └────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────────────────┐   │
│  │ Google Earth    │  │ Sentinel-2   │  │ Legal Lease Boundaries      │   │
│  │ Engine API      │  │ Imagery      │  │ (Shapefile/KML/GeoJSON)     │   │
│  └─────────────────┘  └──────────────┘  └─────────────────────────────┘   │
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────────────────┐   │
│  │ SRTM/ALOS DEM   │  │ Sentinel-1   │  │ Government WFS Endpoints    │   │
│  │ Elevation Data  │  │ SAR Data     │  │ (Optional)                  │   │
│  └─────────────────┘  └──────────────┘  └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Workflow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    ILLEGAL MINING DETECTION WORKFLOW                      │
└──────────────────────────────────────────────────────────────────────────┘

START
  │
  ├─→ [1] USER INPUT
  │   ├── Area of Interest (GeoJSON polygon)
  │   ├── Date Range (start_date, end_date)
  │   ├── Legal Lease Boundaries (optional)
  │   └── Analysis Parameters
  │
  ├─→ [2] DATA ACQUISITION (gee_utils.py)
  │   │
  │   ├─→ Download Sentinel-2 Imagery
  │   │   ├── Filter by AOI and date range
  │   │   ├── Apply cloud filtering (QA60 mask)
  │   │   ├── Create median composite
  │   │   └── Export as multi-band GeoTIFF
  │   │       Bands: B2(Blue), B3(Green), B4(Red), B8(NIR), B11(SWIR1), B12(SWIR2)
  │   │
  │   ├─→ Download DEM Data
  │   │   ├── Select source (SRTM/ALOS)
  │   │   ├── Clip to AOI
  │   │   └── Export as GeoTIFF
  │   │
  │   └─→ Optional: Download Sentinel-1 SAR
  │
  ├─→ [3] PREPROCESSING (preprocess.py)
  │   │
  │   ├─→ Reproject to EPSG:4326
  │   ├─→ Clip to exact AOI bounds
  │   ├─→ Normalize bands (0-1 range)
  │   │   └── Use 2nd-98th percentile for normalization
  │   ├─→ Fill DEM voids
  │   │   └── Use scipy interpolation (linear → nearest neighbor)
  │   └─→ Align raster grids
  │
  ├─→ [4] MINING DETECTION (detect_indices.py)
  │   │
  │   ├─→ Calculate Spectral Indices
  │   │   ├── NDVI = (NIR - Red) / (NIR + Red)
  │   │   ├── BSI = ((SWIR1+Red)-(NIR+Blue)) / ((SWIR1+Red)+(NIR+Blue))
  │   │   ├── NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
  │   │   ├── NDWI = (Green - NIR) / (Green + NIR)
  │   │   ├── SAVI = ((NIR-Red)/(NIR+Red+L)) × (1+L)
  │   │   ├── EVI = 2.5 × ((NIR-Red)/(NIR+6×Red-7.5×Blue+1))
  │   │   └── NBR = (NIR - SWIR2) / (NIR + SWIR2)
  │   │
  │   ├─→ Create Mining Mask
  │   │   ├── Apply Thresholds:
  │   │   │   • NDVI < 0.2 (low vegetation)
  │   │   │   • BSI > 0.3 (high bare soil)
  │   │   │   • NDBI > 0.1 (built-up areas)
  │   │   │   • NDWI < 0.2 (low water)
  │   │   │   • SAVI < 0.1
  │   │   │   • EVI < 0.1
  │   │   │   • NBR < 0.1
  │   │   └── Flag pixel as mining if ≥4 conditions met
  │   │
  │   ├─→ Clean Mask (Morphological Operations)
  │   │   ├── Remove small objects (< 50 pixels)
  │   │   ├── Binary opening (3×3 kernel)
  │   │   ├── Binary closing (5×5 kernel)
  │   │   ├── Binary dilation (3×3 to connect nearby pixels)
  │   │   └── Binary erosion (2×2 to restore size)
  │   │
  │   └─→ Polygonize Mask
  │       ├── Convert raster mask to vector polygons
  │       ├── Calculate area (hectares)
  │       ├── Calculate perimeter and compactness
  │       └── Filter by minimum area (0.001 ha)
  │
  ├─→ [5] ILLEGAL MINING ANALYSIS (compare_with_lease.py)
  │   │
  │   ├─→ Load Legal Lease Boundaries
  │   │   ├── Read from shapefile/KML/GeoJSON
  │   │   ├── Standardize column names
  │   │   └── Create union of all lease polygons
  │   │
  │   ├─→ For Each Detected Mining Polygon:
  │   │   ├── Reproject to equal-area CRS (EPSG:3857)
  │   │   ├── Apply buffer tolerance (10m)
  │   │   │
  │   │   ├── Calculate Intersection
  │   │   │   inside_geom = detected_poly ∩ lease_union_buffered
  │   │   │   inside_area_ha = inside_geom.area / 10000
  │   │   │
  │   │   ├── Calculate Difference
  │   │   │   outside_geom = detected_poly - lease_union
  │   │   │   outside_area_ha = outside_geom.area / 10000
  │   │   │
  │   │   ├── Calculate Overlap Percentage
  │   │   │   overlap_% = (inside_area / total_area) × 100
  │   │   │
  │   │   ├── Classify Status
  │   │   │   IF outside_area ≤ 0.01 ha → LEGAL
  │   │   │   ELSE IF overlap_% ≥ 80% → MIXED
  │   │   │   ELSE → ILLEGAL
  │   │   │
  │   │   └── Calculate Confidence Score
  │   │       confidence = base_confidence × area_factor × lease_factor
  │   │       where:
  │   │         base_confidence = f(overlap_%)
  │   │         area_factor = f(total_area_ha)
  │   │         lease_factor = f(num_overlapping_leases)
  │   │
  │   └─→ Generate Summary Statistics
  │       ├── Total detected areas count
  │       ├── Legal/Illegal/Mixed counts
  │       ├── Total areas in hectares
  │       ├── Compliance rate (%)
  │       └── Violation rate (%)
  │
  ├─→ [6] RESULTS & VISUALIZATION (app.py + frontend)
  │   │
  │   ├─→ Backend Processing
  │   │   ├── Export results to GeoJSON
  │   │   ├── Export results to Shapefile
  │   │   ├── Export statistics to JSON
  │   │   └── Store in analysis_results dict
  │   │
  │   └─→ Frontend Visualization
  │       ├── Display on Leaflet map
  │       │   ├── Green polygons: Legal leases
  │       │   ├── Red zones: Critical violations
  │       │   ├── Orange zones: Warning areas
  │       │   └── Blue markers: Satellite detections
  │       │
  │       ├── Generate PDF Report
  │       │   ├── Summary statistics
  │       │   ├── Map snapshots
  │       │   └── Violation details
  │       │
  │       └── Export Data Files
  │           ├── GeoJSON for web mapping
  │           ├── Shapefile for GIS software
  │           └── CSV for tabular analysis
  │
  └─→ END
```

---

## Module Deep Dive with Code

### 1. Data Acquisition Module (`gee_utils.py`)

**Purpose**: Downloads satellite imagery and elevation data from Google Earth Engine.

**Key Code Sections**:

```python
class GEEUtils:
    def __init__(self):
        # Initialize Google Earth Engine with project ID
        project_id = os.getenv('EE_PROJECT_ID') or 'car-pooling-dc7a3'
        ee.Initialize(project=project_id)
        
        # Define satellite collections
        self.sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        self.sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')
        self.srtm = ee.Image('USGS/SRTMGL1_003')
```

**Cloud Masking Function**:
```python
def mask_clouds(image):
    # QA60 band contains cloud mask information
    # Bit 10: Opaque clouds
    # Bit 11: Cirrus clouds
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1024).eq(0).And(qa.bitwiseAnd(2048).eq(0))
    return image.updateMask(cloud_mask).divide(10000)
```

**Download Process**:
```python
def download_sentinel2_aoi(self, aoi_geojson, start_date, end_date, 
                          out_path, bands=None, max_cloud_cover=20):
    # 1. Convert GeoJSON to Earth Engine geometry
    aoi = ee.Geometry.Polygon(aoi_geojson['coordinates'])
    
    # 2. Filter collection
    collection = (self.sentinel2
                  .filterDate(start_date, end_date)
                  .filterBounds(aoi)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover)))
    
    # 3. Create cloud-free composite
    composite = collection.map(mask_clouds).median().clip(aoi)
    composite = composite.select(bands)
    
    # 4. Download and stack bands
    for band in bands:
        single_band = composite.select([band])
        url = single_band.getDownloadURL({
            'region': aoi.coordinates().getInfo(),
            'scale': 10,  # 10m resolution
            'crs': 'EPSG:4326',
            'format': 'GEO_TIFF'
        })
        self._download_file(url, band_path)
```

---

### 2. Preprocessing Module (`preprocess.py`)

**Purpose**: Prepares raw satellite data for analysis.

**Normalization Code**:
```python
def normalize_bands(self, raster_path, dst_path, scale_factor=10000):
    with rasterio.open(raster_path) as src:
        data = src.read().astype(np.float32)
        normalized_data = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            band = data[i]
            valid_mask = band != src.nodata if src.nodata else np.ones_like(band, dtype=bool)
            
            if np.any(valid_mask):
                # Use 2nd and 98th percentile for robust normalization
                band_min = np.percentile(band[valid_mask], 2)
                band_max = np.percentile(band[valid_mask], 98)
                
                if band_max > band_min:
                    normalized_band = (band - band_min) / (band_max - band_min)
                    normalized_band = np.clip(normalized_band, 0, 1)
                else:
                    normalized_band = band / scale_factor
                
                normalized_data[i] = normalized_band
```

**DEM Void Filling**:
```python
def fill_dem_voids(self, dem_path, dst_path, method="gdal"):
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1).astype(np.float32)
        
        # Create mask for valid data
        if nodata is not None:
            valid_mask = data != nodata
        
        # Get coordinates of valid and invalid points
        y, x = np.mgrid[0:data.shape[0], 0:data.shape[1]]
        valid_points = np.column_stack((y[valid_mask], x[valid_mask]))
        valid_values = data[valid_mask]
        invalid_points = np.column_stack((y[~valid_mask], x[~valid_mask]))
        
        # Interpolate using griddata (linear → nearest neighbor fallback)
        filled_values = griddata(valid_points, valid_values, invalid_points, 
                               method='linear', fill_value=np.nan)
```

---

### 3. Mining Detection Module (`detect_indices.py`)

**Purpose**: Identifies mining areas using spectral analysis.

**Spectral Indices Calculation**:
```python
def _calculate_spectral_indices(self, blue, green, red, nir, swir1, swir2):
    epsilon = 1e-8  # Avoid division by zero
    
    # Vegetation indices
    ndvi = (nir - red) / (nir + red + epsilon)
    savi = ((nir - red) / (nir + red + 0.5)) * (1 + 0.5)
    evi = 2.5 * ((nir - red) / (nir + 6*red - 7.5*blue + 1 + epsilon))
    
    # Soil/bare earth indices
    bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + epsilon)
    ndbi = (swir1 - nir) / (swir1 + nir + epsilon)
    
    # Water index
    ndwi = (green - nir) / (green + nir + epsilon)
    
    # Disturbance index
    nbr = (nir - swir2) / (nir + swir2 + epsilon)
    
    return {
        'ndvi': ndvi, 'bsi': bsi, 'ndbi': ndbi, 'ndwi': ndwi,
        'savi': savi, 'evi': evi, 'nbr': nbr
    }
```

**Mining Mask Creation**:
```python
def _create_mining_mask(self, indices):
    # Define conditions for mining detection
    low_vegetation = indices['ndvi'] < 0.2
    high_bare_soil = indices['bsi'] > 0.3
    low_water = indices['ndwi'] < 0.2
    high_builtup = indices['ndbi'] > 0.1
    low_savi = indices['savi'] < 0.1
    low_evi = indices['evi'] < 0.1
    disturbed_nbr = indices['nbr'] < 0.1
    
    # Stack all conditions
    conditions = np.stack([
        low_vegetation, high_bare_soil, low_water, high_builtup,
        low_savi, low_evi, disturbed_nbr
    ], axis=0)
    
    # Count how many conditions are met
    condition_count = np.sum(conditions, axis=0)
    
    # Mining mask: at least 4 out of 7 conditions met
    mining_mask = condition_count >= 4
    
    # Additional strong signature
    mining_signature = (
        (indices['ndvi'] < 0.15) &
        (indices['bsi'] > 0.4) &
        (indices['ndbi'] > 0.2)
    )
    
    # Combine masks
    final_mask = mining_mask | mining_signature
    return final_mask.astype(np.uint8)
```

**Morphological Cleaning**:
```python
def _clean_mask(self, mask):
    # Remove small noise objects
    min_size = 50  # pixels
    cleaned = remove_small_objects(mask.astype(bool), min_size=min_size)
    
    # Morphological operations
    cleaned = binary_opening(cleaned, footprint=np.ones((3, 3)))
    cleaned = binary_closing(cleaned, footprint=np.ones((5, 5)))
    
    # Connect nearby pixels
    cleaned = binary_dilation(cleaned, structure=np.ones((3, 3)))
    cleaned = binary_erosion(cleaned, structure=np.ones((2, 2)))
    
    # Final cleanup
    cleaned = remove_small_objects(cleaned, min_size=min_size)
    return cleaned.astype(np.uint8)
```

**Polygonization**:
```python
def polygonize_mask(self, mask, transform, crs, min_area_ha=None):
    mask_bool = mask.astype(bool)
    polygons = []
    properties = []
    
    # Generate GeoJSON-like shapes from raster
    for geom, value in rio_shapes(mask.astype(np.uint8), mask=mask_bool, transform=transform):
        if int(value) != 1:
            continue
        
        poly = shape(geom)
        if not poly.is_valid or poly.is_empty:
            continue
        
        # Calculate area in hectares (rough WGS84 conversion)
        area_ha = (poly.area * 111000.0 * 111000.0) / 10000.0
        
        if min_area_ha is not None and area_ha < min_area_ha:
            continue
        
        properties.append({
            'area_ha': round(area_ha, 2),
            'area_m2': round(area_ha * 10000.0, 0),
            'perimeter_m': round(poly.length * 111000.0, 0),
            'compactness': round(4 * np.pi * poly.area / (poly.length ** 2 + 1e-9), 3),
            'mining_id': f"mining_{len(polygons)+1}"
        })
        polygons.append(poly)
    
    return gpd.GeoDataFrame(properties, geometry=polygons, crs=crs)
```

---

### 4. Illegal Mining Analysis Module (`compare_with_lease.py`)

**Purpose**: Compares detected mining with legal boundaries.

**Spatial Overlay Analysis**:
```python
def compare_with_lease(self, detected_polygons, lease_polygons, equal_area_crs="EPSG:3857"):
    # Project to equal-area CRS for accurate area calculations
    detected_ea = detected_polygons.to_crs(equal_area_crs)
    lease_ea = lease_polygons.to_crs(equal_area_crs)
    
    # Create union of all lease boundaries
    lease_union = lease_ea.unary_union
    
    # Add buffer for tolerance
    if self.buffer_meters > 0:
        lease_union_buffered = lease_union.buffer(self.buffer_meters)
    else:
        lease_union_buffered = lease_union
    
    results = []
    for idx, detected_poly in detected_ea.iterrows():
        result = self._analyze_single_polygon(
            detected_poly, lease_union_buffered, lease_ea, equal_area_crs
        )
        results.append(result)
    
    return gpd.GeoDataFrame(results, crs=equal_area_crs)
```

**Single Polygon Analysis**:
```python
def _analyze_single_polygon(self, detected_poly, lease_union, lease_gdf, crs):
    # Calculate total area
    total_area_m2 = detected_poly.geometry.area
    total_area_ha = total_area_m2 / 10000
    
    # Find intersection with lease boundaries
    inside_geom = detected_poly.geometry.intersection(lease_union)
    inside_area_m2 = inside_geom.area if inside_geom.area > 0 else 0
    inside_area_ha = inside_area_m2 / 10000
    
    # Calculate area outside lease boundaries
    outside_geom = detected_poly.geometry.difference(lease_union)
    outside_area_m2 = outside_geom.area if outside_geom.area > 0 else 0
    outside_area_ha = outside_area_m2 / 10000
    
    # Calculate overlap percentage
    overlap_percentage = (inside_area_ha / total_area_ha * 100) if total_area_ha > 0 else 0
    
    # Find overlapping leases
    overlapping_leases = []
    for _, lease in lease_gdf.iterrows():
        if detected_poly.geometry.intersects(lease.geometry):
            overlap_area = detected_poly.geometry.intersection(lease.geometry).area / 10000
            overlapping_leases.append({
                'lease_id': lease.get('lease_id', 'unknown'),
                'lease_name': lease.get('lease_name', 'unknown'),
                'overlap_area_ha': round(overlap_area, 2)
            })
    
    # Classify status
    status = self._classify_mining_status(outside_area_ha, overlap_percentage)
    
    # Calculate confidence
    confidence = self._calculate_confidence_score(
        total_area_ha, overlap_percentage, len(overlapping_leases)
    )
    
    return {
        'geometry': detected_poly.geometry,
        'total_area_ha': round(total_area_ha, 2),
        'inside_area_ha': round(inside_area_ha, 2),
        'outside_area_ha': round(outside_area_ha, 2),
        'overlap_percentage': round(overlap_percentage, 1),
        'status': status,
        'confidence': round(confidence, 2),
        'overlapping_leases': overlapping_leases,
        'num_overlapping_leases': len(overlapping_leases),
        'illegal_area_ha': round(outside_area_ha, 2) if status in ['illegal', 'mixed'] else 0
    }
```

**Classification Logic**:
```python
def _classify_mining_status(self, outside_area_ha, overlap_percentage):
    # Legal: minimal area outside (< 0.01 ha tolerance)
    if outside_area_ha <= self.tolerance_ha:
        return 'legal'
    
    # Mixed: mostly within but some spillover (≥80% overlap)
    elif overlap_percentage >= 80:
        return 'mixed'
    
    # Illegal: significant area outside
    else:
        return 'illegal'
```

**Confidence Score Calculation**:
```python
def _calculate_confidence_score(self, total_area_ha, overlap_percentage, num_overlapping_leases):
    # Base confidence on overlap
    if overlap_percentage >= 95:
        base_confidence = 0.95
    elif overlap_percentage >= 80:
        base_confidence = 0.85
    elif overlap_percentage >= 50:
        base_confidence = 0.70
    else:
        base_confidence = 0.60
    
    # Adjust for area size (larger = more reliable)
    if total_area_ha >= 10:
        area_factor = 1.0
    elif total_area_ha >= 1:
        area_factor = 0.9
    else:
        area_factor = 0.8
    
    # Adjust for lease complexity
    if num_overlapping_leases == 1:
        lease_factor = 1.0
    elif num_overlapping_leases > 1:
        lease_factor = 0.9
    else:
        lease_factor = 0.8
    
    confidence = base_confidence * area_factor * lease_factor
    return min(1.0, max(0.0, confidence))
```

---

### 5. API Layer (`app.py`)

**Purpose**: Provides REST API endpoints for frontend integration.

**FastAPI Application Setup**:
```python
app = FastAPI(
    title="Illegal Mining Detection API",
    description="End-to-end system for detecting illegal mining activities",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processing modules
gee_utils = GEEUtils()
preprocessor = Preprocessor()
mining_detector = MiningDetector()
illegal_detector = IllegalMiningDetector()
```

**Main Detection Endpoint**:
```python
@app.post("/api/detect")
async def detect_illegal_mining(request: DetectionRequest, background_tasks: BackgroundTasks):
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Store initial status
    analysis_results[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "message": "Analysis initiated...",
        "progress": 0
    }
    
    # Run analysis in background
    background_tasks.add_task(_run_illegal_mining_analysis, job_id, request)
    
    return {
        "job_id": job_id,
        "status": "processing",
        "estimated_completion_time": "5-15 minutes"
    }
```

**Background Analysis Function**:
```python
async def _run_illegal_mining_analysis(job_id: str, request: DetectionRequest):
    try:
        # Step 1: Download satellite data (10% progress)
        analysis_results[job_id]["progress"] = 10
        sentinel2_path = os.path.join(temp_dir, "sentinel2.tif")
        dem_path = os.path.join(temp_dir, "dem.tif")
        
        gee_utils.download_sentinel2_aoi(
            request.aoi_geojson, request.start_date, request.end_date,
            sentinel2_path, max_cloud_cover=20
        )
        gee_utils.download_dem(request.aoi_geojson, dem_path, "SRTM")
        
        # Step 2: Preprocess (30% progress)
        analysis_results[job_id]["progress"] = 30
        normalized_sentinel2 = os.path.join(temp_dir, "sentinel2_normalized.tif")
        filled_dem = os.path.join(temp_dir, "dem_filled.tif")
        
        preprocessor.normalize_bands(sentinel2_path, normalized_sentinel2)
        preprocessor.fill_dem_voids(dem_path, filled_dem)
        
        # Step 3: Detect mining (50% progress)
        analysis_results[job_id]["progress"] = 50
        detection_results = mining_detector.detect_mining_areas(
            normalized_sentinel2, temp_dir
        )
        
        # Step 4: Compare with leases (70% progress)
        analysis_results[job_id]["progress"] = 70
        if request.lease_file_path:
            lease_gdf = illegal_detector.read_lease_shapefile(request.lease_file_path)
        else:
            lease_gdf = illegal_detector.fetch_government_leases(aoi_bbox)
        
        comparison_results = illegal_detector.compare_with_lease(
            detection_results['polygons'], lease_gdf, request.buffer_meters
        )
        
        # Step 5: Generate results (90% progress)
        analysis_results[job_id]["progress"] = 90
        summary_stats = illegal_detector.generate_summary_statistics(comparison_results)
        export_files = illegal_detector.export_results(comparison_results, temp_dir, 'all')
        
        # Final results
        analysis_results[job_id].update({
            "status": "completed",
            "progress": 100,
            "results": {
                "detection_results": detection_results,
                "comparison_results": comparison_results.to_dict('records'),
                "summary_statistics": summary_stats,
                "export_files": export_files
            }
        })
    except Exception as e:
        analysis_results[job_id].update({
            "status": "failed",
            "message": f"Analysis failed: {str(e)}"
        })
```

---

## Data Flow

```
INPUT DATA
│
├─→ Satellite Imagery (Sentinel-2)
│   • 6 spectral bands: Blue, Green, Red, NIR, SWIR1, SWIR2
│   • 10m spatial resolution
│   • Cloud-filtered composite
│   • Date range: user-specified
│
├─→ Elevation Data (DEM)
│   • SRTM or ALOS source
│   • 30m resolution
│   • Void-filled
│
└─→ Legal Boundaries
    • Shapefile/KML/GeoJSON
    • Lease polygons with attributes
    • CRS: various (standardized to EPSG:4326)
    
    ↓
    
PROCESSING PIPELINE
│
├─→ Stage 1: Spectral Index Calculation
│   Input: 6-band Sentinel-2 raster
│   Output: 7 index rasters (NDVI, BSI, NDBI, NDWI, SAVI, EVI, NBR)
│   Format: Float32 arrays, normalized -1 to 1
│
├─→ Stage 2: Threshold Classification
│   Input: 7 index rasters
│   Output: Binary mask (0=no mining, 1=mining)
│   Logic: Pixel flagged if ≥4 conditions met
│
├─→ Stage 3: Morphological Cleaning
│   Input: Binary mask
│   Output: Cleaned binary mask
│   Operations: Opening, Closing, Dilation, Erosion
│
├─→ Stage 4: Vectorization
│   Input: Cleaned mask + geotransform
│   Output: Vector polygons (GeoDataFrame)
│   Attributes: area_ha, perimeter_m, compactness, mining_id
│
├─→ Stage 5: Spatial Overlay
│   Input: Mining polygons + Lease polygons
│   Output: Classified polygons with intersection/difference
│   CRS: EPSG:3857 (equal-area projection)
│   Calculations:
│     • inside_area = detected ∩ leases
│     • outside_area = detected - leases
│     • overlap_% = (inside / total) × 100
│
└─→ Stage 6: Classification & Statistics
    Input: Spatial overlay results
    Output: Status (legal/mixed/illegal), confidence, summary stats
    
    ↓
    
OUTPUT DATA
│
├─→ GeoJSON Files
│   • Web-compatible format
│   • Includes all attributes
│   • CRS: EPSG:4326
│
├─→ Shapefiles
│   • GIS-compatible format
│   • Standard format for desktop GIS
│
├─→ Summary Statistics (JSON)
│   • Total areas detected
│   • Legal/Illegal counts
│   • Compliance rates
│   • Violation percentages
│
└─→ Visualizations
    • Interactive Leaflet map
    • Color-coded zones
    • PDF reports
```

---

## API Endpoints

### Demo Endpoints

| Endpoint | Method | Purpose | Response |
|----------|--------|---------|----------|
| `/api/mining-boundaries` | GET | Get demo legal leases | 12 sample leases across India with GeoJSON |
| `/api/satellite-data` | GET | Get detected mining areas | 6 demo detection polygons with properties |
| `/api/analyze/quick` | POST | Quick analysis | Immediate success response |
| `/api/analyze/illegal-mining-detection` | POST | Full detection | Analysis job ID with violation zones |
| `/api/illegal-mining-results/{id}` | GET | Get analysis results | Complete results with red/orange zones |

### Production Endpoints

| Endpoint | Method | Purpose | Input | Output |
|----------|--------|---------|-------|--------|
| `/api/upload-lease` | POST | Upload lease file | Shapefile/KML/GeoJSON | File info with bounds |
| `/api/detect` | POST | Run full analysis | AOI + dates + params | Job ID |
| `/api/results/{job_id}` | GET | Get job results | Job ID | Status + results |
| `/api/report/{job_id}` | GET | Download PDF | Job ID | PDF file |
| `/api/download/{job_id}/{type}` | GET | Download files | Job ID + file type | GeoJSON/SHP/CSV |
| `/api/health` | GET | Health check | None | Status of modules |

---

## Key Algorithms Summary

### Mining Detection Algorithm
```
FOR each pixel in satellite image:
    1. Calculate 7 spectral indices
    2. Apply 7 threshold conditions
    3. IF ≥4 conditions are TRUE:
         Mark pixel as mining
    4. ELSE:
         Mark pixel as non-mining
END FOR

Apply morphological cleaning:
    • Remove noise (< 50 pixels)
    • Smooth boundaries
    • Connect nearby pixels
    • Fill small holes

Convert to polygons:
    • Vectorize binary mask
    • Calculate areas
    • Filter by minimum size
```

### Illegal Mining Classification Algorithm
```
FOR each detected mining polygon:
    1. Reproject to equal-area CRS
    2. Create lease union with buffer
    3. Calculate intersection:
         inside_area = polygon ∩ lease_union
    4. Calculate difference:
         outside_area = polygon - lease_union
    5. Calculate overlap:
         overlap_% = (inside / total) × 100
    
    6. Classify:
         IF outside_area ≤ 0.01 ha:
             status = LEGAL
         ELSE IF overlap_% ≥ 80%:
             status = MIXED
         ELSE:
             status = ILLEGAL
    
    7. Calculate confidence score
    8. Find overlapping leases
END FOR
```

---

## Performance Characteristics

| Operation | Time Complexity | Space Complexity | Typical Time |
|-----------|----------------|------------------|--------------|
| Sentinel-2 Download | O(n × m) | O(n × m) | 30-60 seconds |
| Spectral Index Calc | O(n × m × k) | O(n × m × k) | 5-10 seconds |
| Morphological Ops | O(n × m × kernel²) | O(n × m) | 2-5 seconds |
| Polygonization | O(n × m) | O(p) | 10-20 seconds |
| Spatial Overlay | O(p × l) | O(p + l) | 5-15 seconds |

Where:
- n × m = raster dimensions (pixels)
- k = number of bands/indices
- p = number of detected polygons
- l = number of lease polygons

---

## Technology Stack

**Backend**:
- FastAPI: REST API framework
- Rasterio: Raster I/O and processing
- GeoPandas: Vector data processing
- Shapely: Geometric operations
- NumPy: Numerical computations
- SciPy: Scientific algorithms
- scikit-image: Morphological operations
- Google Earth Engine API: Satellite data

**Frontend**:
- React: UI framework
- Leaflet: Interactive mapping
- Plotly: 3D visualization
- Axios: HTTP client

**Data Formats**:
- Input: GeoTIFF, Shapefile, KML, GeoJSON
- Output: GeoJSON, Shapefile, CSV, PDF
- Intermediate: NumPy arrays, Pandas DataFrames

---

This documentation provides a complete technical overview of the Illegal Mining Detection System with detailed code examples, architecture diagrams, and workflow explanations.
