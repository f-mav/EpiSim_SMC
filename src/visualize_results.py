import xarray as xr
import matplotlib.pyplot as plt
import os

# POINT THIS TO YOUR FILE
# Based on your image, it is inside runs/output runs/output/observables.nc
file_path = os.path.join("runs", "output", "compartments_full.nc") 
# If that doesn't work, try full path or "compartments_full.nc"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    print("Please check the path.")
else:
    print(f"Opening {file_path}...")
    
    # Open the NetCDF file
    ds = xr.open_dataset(file_path)
    
    print("\n--- Available Variables ---")
    print(ds)

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # We loop through data variables to find what to plot
    # Common variables in EpiSim might be 'S', 'I', 'R', or 'susceptible', etc.
    # We skip 'time' or coordinate variables
    for var_name in ds.data_vars:
        data = ds[var_name]
        
        # Plot only 1D data (time series)
        if data.ndim == 1:
            data.plot.line(label=var_name)
            
    plt.title("Simulation Observables")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()