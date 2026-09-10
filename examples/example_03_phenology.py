import numpy as np
import cdts

def main():
    print("CDTS Phenofit: End-to-End Phenology Extraction Example")
    
    # 1. Simulate a multi-year vegetation time-series (e.g., NDVI for 3 years)
    # We will create a synthetic dataset for 10 pixels over 3 years (16-day composites)
    years = 3
    time_steps_per_year = 23
    total_time_steps = years * time_steps_per_year
    n_pixels = 10
    
    # Create dates array (Day of Year values spanning multiple years)
    # E.g., [1, 17, 33, ... 353, 366, 382, ...]
    dates = np.array([t * 16 + (t // 23) * 365 for t in range(total_time_steps)])
    
    # Create synthetic NDVI datacube (Time, Y, X) -> (69, 10, 1)
    # Using a sine wave to simulate seasonal growth, adding noise
    np.random.seed(42)
    cube_data = np.zeros((total_time_steps, n_pixels, 1))
    
    for t in range(total_time_steps):
        # A simple curve peaking around middle of the year
        doy = dates[t] % 365
        growth = np.sin((doy / 365.0) * np.pi - np.pi/2) # -1 to 1
        growth = (growth + 1) / 2.0 # 0 to 1
        
        # Add a baseline of 0.2 (soil) and peak of 0.8 (healthy veg)
        ndvi = 0.2 + growth * 0.6
        
        # Add some random noise
        cube_data[t, :, 0] = ndvi + np.random.normal(0, 0.05, n_pixels)
        
    print(f"Created synthetic datacube with shape: {cube_data.shape}")
    
    # 2. Convert to an xarray DataArray (Dask backed)
    # In a real scenario, this would be returned by cdts.build_time_series()
    import xarray as xr
    import dask.array as da
    
    da_cube = da.from_array(cube_data, chunks=(total_time_steps, 5, 1))
    xr_cube = xr.DataArray(da_cube, dims=["time", "y", "x"], coords={"time": dates})
    
    # 3. Run Phenology Extraction
    # We will use the HANTS smoother, double logistic (Beck) curve, and the DERIVATIVE extraction method.
    print("\nRunning Phenology extraction (this will process via C++ / OpenMP)...")
    pheno_results = xr_cube.cdts.run_phenology(
        dates=dates,
        curve_type=cdts.CurveType.BECK,
        extraction_method=cdts.ExtractionMethod.DERIVATIVE,
        max_seasons=years,               # Expecting 1 season per year
        
        # Smoothing Configuration
        apply_whittaker=False,           # Disable Whittaker
        apply_hants=True,                # Enable HANTS
        hants_frequencies=3,
        hants_threshold=0.1,
        
        # Fine-Grained Season Control
        min_season_length=90,            # A real season must last at least 90 days
        min_amplitude=0.2,               # A real season must have an NDVI jump of at least 0.2
        
        n_jobs=-1                        # Use all available CPU cores
    )
    
    # 4. Trigger computation
    # The output shape is (metric, season, y, x)
    # where metric: 0=SOS, 1=EOS, 2=LOS, 3=POP
    result_array = pheno_results.compute()
    
    print("\nExtraction Complete! Results shape:", result_array.shape)
    
    # 5. Inspect Results
    print("\nAnalyzing Pixel 0, Season 0:")
    sos = result_array[0, 0, 0, 0]
    eos = result_array[1, 0, 0, 0]
    los = result_array[2, 0, 0, 0]
    pop = result_array[3, 0, 0, 0]
    
    print(f"  Start of Season (SOS): DOY {sos:.1f}")
    print(f"  Peak of Season  (POP): DOY {pop:.1f}")
    print(f"  End of Season   (EOS): DOY {eos:.1f}")
    print(f"  Length of Season(LOS): {los:.1f} days")
    
if __name__ == "__main__":
    main()
