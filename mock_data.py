
import numpy as np
import xarray as xr
import pandas as pd
from typing import List, Tuple

# ============================================================================
# MOCK DATA GENERATION
# ============================================================================
def get_reduced_gaussian_grid(n_lat: int = 64) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Generate a reduced Gaussian grid structure.
    
    Returns:
        latitudes: Array of latitude values
        points: 1D array of all grid points
        points_per_lat: Number of points at each latitude
    """
    # Simplified Gaussian latitudes
    x = np.linspace(-1, 1, n_lat)
    latitudes = np.arcsin(x) * 180 / np.pi
    latitudes = latitudes[::-1]  # Descending order
    
    # Reduced grid: fewer points near poles
    # N320 grid example pattern (simplified)
    points_per_lat = []
    for i, lat in enumerate(latitudes):
        # Fewer points near poles, more at equator
        lat_factor = np.cos(np.radians(lat))
        n_points = max(20, int(160 * lat_factor))  # Simplified formula
        points_per_lat.append(n_points)
    
    total_points = sum(points_per_lat)
    
    # Create 1D coordinate array with lat/lon encoded
    points = np.arange(total_points, dtype=np.float32)
    
    return latitudes, points, points_per_lat


def create_mock_weather_data(path: str = "weather_data.zarr") -> xr.Dataset:
    """Creates a mock ERA5-like dataset in Zarr format with reduced Gaussian grid."""
    
    # Create synthetic data with reduced Gaussian grid
    time_points = 500
    times = pd.date_range("2020-01-01", periods=time_points, freq="6h")
    latitudes, points, points_per_lat = get_reduced_gaussian_grid(64)
    total_points = len(points)
    
    print(f"Reduced Gaussian grid: {len(latitudes)} latitudes, {total_points} total points")
    print(f"Points per latitude - Min: {min(points_per_lat)}, Max: {max(points_per_lat)}")
    
    # Generate correlated weather fields
    np.random.seed(42)
    base_pattern = np.random.randn(total_points)
    
    # Create mapping from 1D points to lat/lon for data generation
    point_to_lat = []
    point_to_lon_idx = []
    idx = 0
    for lat_idx, n_points in enumerate(points_per_lat):
        for lon_idx in range(n_points):
            point_to_lat.append(latitudes[lat_idx])
            point_to_lon_idx.append(lon_idx / n_points * 360)  # Longitude in degrees
            idx += 1
    
    data_vars = {}
    for i, var in enumerate(["temperature_2m", "u_wind_10m", "v_wind_10m", "surface_pressure"]):
        print("Generating variable:", var)
        # Create time-varying data with spatial correlation
        data = np.zeros((time_points, total_points))
        for t in range(time_points):
            noise = np.random.randn(total_points) * 0.1
            seasonal = np.sin(2 * np.pi * t / 365) * (i + 1)
            # Add latitude-dependent variation
            lat_weights = np.array([np.cos(np.radians(lat)) for lat in point_to_lat])
            data[t] = base_pattern * (i + 1) + noise + seasonal * lat_weights
            
        data_vars[var] = (["time", "points"], data)
    
    # Add cell area as a coordinate (important for reduced Gaussian grids)
    cell_areas = np.zeros(total_points)
    idx = 0
    for lat_idx, (lat, n_points) in enumerate(zip(latitudes, points_per_lat)):
        # Approximate cell area (proportional to cos(lat) / n_points)
        area = np.cos(np.radians(lat)) / n_points
        for _ in range(n_points):
            cell_areas[idx] = area
            idx += 1
    
    # Create xarray dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": times,
            "points": points,
            "cell_area": ("points", cell_areas),
            # Store grid structure as attributes or coordinates
            "latitude": ("points", np.array(point_to_lat)),
            "longitude": ("points", np.array(point_to_lon_idx))
        },
        attrs={
            "description": "Mock ERA5 reanalysis data on reduced Gaussian grid",
            "grid_type": "reduced_gaussian",
            "n_latitudes": len(latitudes),
            "points_per_latitude": points_per_lat,  # Store as attribute
            "unique_latitudes": latitudes.tolist(),
            "total_points": total_points,
            "units": {
                "temperature_2m": "K",
                "u_wind_10m": "m/s",
                "v_wind_10m": "m/s",
                "surface_pressure": "Pa"
            }
        }
    )
    
    print("Chunking and saving dataset to Zarr format...")
    ds = ds.chunk({"time": 10, "points": 500})
    ds.to_zarr(path, mode="w", zarr_format=3, consolidated=False)
    print(f"Mock data created at {path}")
    print(f"Total grid points: {total_points}")
    print(f"Time steps: {len(times)}")
    return ds

if __name__ == "__main__":
    ds = create_mock_weather_data()