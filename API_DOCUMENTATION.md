# PyWake Service - API Documentation

## Overview
This API service provides wind farm simulation capabilities using PyWake library. It accepts GeoJSON polygon boundaries and returns wind turbine placement and energy production calculations.

---

## API Endpoints

### 1. Wind Farm Simulation
**Endpoint:** `POST /pywake/wind-farm/{geojson_name}`

Generates wind turbine layout and calculates power output for a single wind condition.

#### Input Structure

**Path Parameter:**
- `geojson_name` (string): Name identifier for the GeoJSON area

**Request Body:** `GeoJSONQuery`
```json
{
  "properties": {
    "dp": [number array],    // Directional probability distribution (12 sectors, values in permille)
    "c_s": [number array],   // Weibull scale parameter (c) for each wind direction sector
    "k_s": [number array],   // Weibull shape parameter (k) for each wind direction sector
    "mys": number           // Mean yearly wind speed (m/s)
  },
  "geometry": {
    "type": "Polygon",
    "coordinates": [
      [
        [longitude, latitude],  // First point (EPSG:4326)
        [longitude, latitude],  // Second point
        ...
        [longitude, latitude]   // Last point (must match first point)
      ]
    ]
  }
}
```

**Properties Details:**
- `dp`: Array of 12 numbers representing wind frequency for each 30° sector (values sum to 1000)
- `c_s`: Array of 12 numbers for Weibull scale parameters per sector
- `k_s`: Array of 12 numbers for Weibull shape parameters per sector
- `mys`: Single number for average wind speed

**Geometry Details:**
- Coordinates must be in **WGS84 (EPSG:4326)** - Longitude/Latitude format
- Polygon must form a closed loop (first point = last point)
- Area is projected to **UTM Zone 24S (EPSG:32724)** for calculations
- Minimum area required: **2,000,000 m²** (2 km²)

#### Output Structure

**Response:** `FeatureCollection`
```json
{
  "type": "FeatureCollection",
  "properties": {
    "Weibull_A": [number array],     // Weibull scale parameters for all sectors
    "Weibull_k": [number array],     // Weibull shape parameters for all sectors
    "WS": [number array],            // Wind speeds evaluated
    "WD": [number array],            // Wind directions evaluated
    "Farm_AEP_GWh": number           // Total Annual Energy Production in GWh
  },
  "features": [
    {
      "type": "Feature",
      "properties": {
        "wt": number,                    // Wind turbine number (sequential)
        "Aerogerador": string,           // Turbine model ("IEA_Reference_15MW_240")
        "Pot": number,                   // Rated power (MW) - 15.0
        "WS_eff": number,                // Effective wind speed at turbine (m/s)
        "TI_eff": number,                // Effective turbulence intensity
        "Power": number,                 // Power output (kW)
        "CT": number,                    // Thrust coefficient
        "h": number,                     // Hub height (m) - 150
        "type": "WTG"                    // Type identifier (Wind Turbine Generator)
      },
      "geometry": {
        "type": "Point",
        "coordinates": [longitude, latitude]  // EPSG:4674 (SIRGAS 2000)
      }
    }
    // ... more turbine features
  ]
}
```

**Output Details:**
- Each feature represents one wind turbine
- Coordinates returned in **SIRGAS 2000 (EPSG:4674)**
- `Farm_AEP_GWh`: Total farm Annual Energy Production in GigaWatt-hours
- Turbine positions optimized with:
  - Downwind spacing: 15D (15 × rotor diameter)
  - Crosswind spacing: 3D (3 × rotor diameter)
  - Stagger offset: 1.5D for improved wake management

---

### 2. Dashboard Data
**Endpoint:** `POST /pywake/wind-farm/dash-data/{geojson_name}`

Returns detailed AEP data broken down by wind speed and wind direction for dashboard visualization.

#### Input Structure

Same as Wind Farm Simulation endpoint (see above).

#### Output Structure

**Response:** Dashboard Data Object
```json
{
  "wind_speed_x": [number array],      // Wind speeds (m/s) - [3, 4, 5, ..., 25]
  "aep_vs_ws_y": [number array],       // AEP per wind speed bin (GWh)
  "wind_direction_x": [number array],  // Wind directions (degrees) - [0, 30, 60, ..., 330]
  "aep_vs_wd_y": [number array]        // AEP per wind direction sector (GWh)
}
```

**Output Details:**
- `wind_speed_x`: Array of wind speeds from 3 to 25 m/s (23 values)
- `aep_vs_ws_y`: Annual Energy Production for each wind speed bin summed across all turbines and directions
- `wind_direction_x`: Array of 12 wind direction sectors (0° to 330°, 30° intervals)
- `aep_vs_wd_y`: Annual Energy Production for each direction sector summed across all turbines and speeds
- All AEP values in GWh (GigaWatt-hours)

---

## Turbine Specifications

**Model:** IEA Reference 15MW 240m

| Parameter | Value |
|-----------|-------|
| Rated Power | 15 MW |
| Rotor Diameter | 240 m |
| Hub Height | 150 m |
| Cut-in Speed | ~3 m/s |
| Cut-out Speed | ~25 m/s |
| Data Source | `data/IEA_Reference_15MW_240.csv` |

**Power Curve:** The turbine uses a tabular power-CT curve loaded from CSV containing:
- Wind Speed [m/s]
- Power [kW]
- Thrust Coefficient (Ct) [-]

---

## Simulation Parameters

### Site Configuration
- **Turbulence Intensity (TI):** 10% (0.1) for all sectors
- **Wind Shear Model:** Power law with:
  - Reference height: 100 m
  - Alpha (shear exponent): 0.1

### Wind Farm Model
- **Model Type:** All2AllIterative
- **Wake Deficit Model:** FUGA (Flow and wakes in complex terrain)
- **Superposition Model:** Linear Sum
- **Blockage Model:** FUGA Deficit

### Layout Generation
- **Spacing Strategy:**
  - Downwind (parallel to wind): 15D
  - Crosswind (perpendicular to wind): 3D
  - Stagger offset: 1.5D
- **Alignment:** Grid rotated to align with predominant wind direction
- **Anchor Point:** Positioned at upwind boundary vertex
- **Filtering:** Only turbines inside polygon boundary are included

---

## Coordinate Reference Systems

| Purpose | CRS | EPSG Code |
|---------|-----|-----------|
| Input Geometry | WGS 84 (Lon/Lat) | EPSG:4326 |
| Internal Calculations | UTM Zone 24S | EPSG:32724 |
| Output Coordinates | SIRGAS 2000 | EPSG:4674 |

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Simulation area is too small. Please select a larger area."
}
```
**Cause:** Polygon area < 2,000,000 m²

```json
{
  "detail": "No turbines could be placed in the selected area. Please select a larger or different area."
}
```
**Cause:** Layout algorithm could not place any turbines within boundary

### 404 Not Found
```json
{
  "detail": "Could not retrieve point properties for {geojson_name}"
}
```
**Cause:** External service could not find wind resource data for the specified area

### 500 Internal Server Error
```json
{
  "detail": "Simulation failed. Check turbine data or geometry."
}
```
**Cause:** PyWake simulation encountered an error

### 503 Service Unavailable
```json
{
  "detail": "Service unavailable: {exception details}"
}
```
**Cause:** External wind resource service is not responding

---

## Example Request

```bash
curl -X POST "http://localhost:8000/pywake/wind-farm/my_wind_farm" \
  -H "Content-Type: application/json" \
  -d '{
    "properties": {
      "dp": [100, 120, 90, 85, 75, 70, 80, 95, 110, 105, 85, 85],
      "c_s": [8.5, 8.7, 8.3, 8.1, 7.9, 7.8, 8.0, 8.4, 8.8, 8.6, 8.2, 8.4],
      "k_s": [2.1, 2.2, 2.0, 2.1, 2.0, 1.9, 2.0, 2.1, 2.2, 2.1, 2.0, 2.1],
      "mys": 8.3
    },
    "geometry": {
      "type": "Polygon",
      "coordinates": [[
        [-48.5, -28.5],
        [-48.5, -28.4],
        [-48.4, -28.4],
        [-48.4, -28.5],
        [-48.5, -28.5]
      ]]
    }
  }'
```

---

## Notes

1. **Performance:** Simulation time increases with polygon size and number of turbines
2. **Validation:** Input geometry must be a valid closed polygon
3. **Units:** All power values in kW, AEP in GWh, distances in meters, speeds in m/s
4. **Wind Data:** Requires connection to external wind resource service at `{MAIN_SERVICE}/geojsons/query/centroid/`
5. **Turbine Data:** CSV file must exist at `./data/IEA_Reference_15MW_240.csv`

---

## Version Information

- **PyWake:** Wind farm simulation engine
- **FastAPI:** Web framework
- **Projection:** pyproj for coordinate transformations
- **GeoJSON:** RFC 7946 compliant

---

*Last Updated: January 20, 2026*
