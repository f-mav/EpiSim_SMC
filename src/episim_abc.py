import numpy as np
import pandas as pd
import xarray as xr
import json
import os
import sys
import logging  # <--- Essential

# 1. Import your wrapper FIRST
try:
    from epi_sim import EpiSim, pardir
except ImportError:
    sys.exit("Error: Could not import 'epi_sim.py'.")

# 2. SILENCE THE LOGGER HERE (After Import)
# The wrapper sets itself to DEBUG when it loads. We must override it now.
epi_logger = logging.getLogger("epi_sim")
epi_logger.setLevel(logging.ERROR)

# Also silence the specific handler the wrapper attaches to stdout
for handler in epi_logger.handlers:
    handler.setLevel(logging.ERROR)

# --- 1. CONFIGURATION & CONSTANTS ---
REAL_DATA_CSV = "real_data_spain.csv"
DATA_FOLDER = os.path.join(pardir(), "models/mitma")
INSTANCE_FOLDER = os.path.join(pardir(), "runs")
INITIAL_CONDITIONS = os.path.join(DATA_FOLDER, "A0_initial_conditions_seeds.csv")
CONFIG_FILE = os.path.join(DATA_FOLDER, "config_MMCACovid19.json")

# ABC-SMC SETTINGS
N_PARTICLES = 10      # Number of successful particles to keep per generation
N_GENERATIONS = 4     # Number of times to "tighten" the threshold
QUANTILE = 0.5        # Keep the best 50% of particles for the next threshold

# --- 2. DATA HANDLING WITH NORMALIZATION ---

def normalize_curve(curve):
    """
    Scales a curve between 0 and 1 based on its maximum value.
    This makes the distance metric independent of population size.
    """
    peak = np.max(curve)
    if peak == 0: return curve # Avoid division by zero
    return curve / peak

def get_ground_truth():
    """Loads and normalizes the real-world data."""
    if not os.path.exists(REAL_DATA_CSV):
        sys.exit(f"Error: {REAL_DATA_CSV} not found. Run your generation script first.")
    
    df = pd.read_csv(REAL_DATA_CSV)
    raw_data = df['new_infected'].values
    return normalize_curve(raw_data)

# --- 3. SIMULATION RUNNER ---

def run_simulation(beta, base_config, norm_ground_truth):
    """
    Runs model, normalizes output, and calculates distance.
    Returns: (distance, raw_output_path)
    """
    model = EpiSim(base_config, DATA_FOLDER, INSTANCE_FOLDER, INITIAL_CONDITIONS)
    model.setup(executable_type='interpreter')

    # Update Config
    cfg = base_config.copy()
    # Adjust key based on your specific JSON structure
    if "params" in cfg: cfg["params"]["beta"] = beta
    elif "epidemic_params" in cfg: cfg["epidemic_params"]["beta"] = beta
    else: cfg["beta"] = beta
    
    model.update_config(cfg)

    # Run Settings
    override = {
        "start_date": "2020-02-09",
        "end_date": "2020-05-31",
        "save_time_step": -1
    }

    try:
        model.run_model(override_config=override, override_model_state=model.model_state)
        
        # Path logic (handling the .nc file)
        output_nc = os.path.join(model.model_state_folder, "output", "observables.nc")
        if not os.path.exists(output_nc):
            output_nc = os.path.join(model.model_state_folder, "output", f"compartments_t_{override['end_date']}.nc")
            if not os.path.exists(output_nc): return float('inf')

        # Read & Normalize
        ds = xr.open_dataset(output_nc)
        sim_raw = ds['new_infected'].sum(dim=['M', 'G']).values
        ds.close()

        # Normalize the simulation EXACTLY like the ground truth
        sim_norm = normalize_curve(sim_raw)

        # Calculate Distance (RMSE on Normalized Data)
        n = min(len(sim_norm), len(norm_ground_truth))
        diff = sim_norm[:n] - norm_ground_truth[:n]
        distance = np.sqrt(np.mean(diff**2))
        
        return distance

    except Exception as e:
        print(f" [Crash: {e}]", end="")
        return float('inf')

# --- 4. THE ADAPTIVE ABC-SMC ALGORITHM ---

if __name__ == "__main__":
    print(f"--- Starting Adaptive ABC-SMC ---")
    print(f"Particles per Gen: {N_PARTICLES} | Generations: {N_GENERATIONS}")
    
    # Load Data
    norm_truth = get_ground_truth()
    
    with open(CONFIG_FILE, 'r') as f:
        base_config = json.load(f)

    # Initial State
    # We start with a huge epsilon (infinite) to accept the first batch
    current_epsilon = float('inf')
    
    # Store the previous generation's results (weights could be added here later)
    # List of tuples: (beta_value, distance)
    population = []

    for gen in range(N_GENERATIONS):
        print(f"\n\n=== GENERATION {gen + 1} (Threshold ε = {current_epsilon:.4f}) ===")
        
        new_population = []
        attempts = 0
        
        while len(new_population) < N_PARTICLES:
            attempts += 1
            
            # --- A. PROPOSE PARAMETER ---
            # Gen 0: Random Uniform from Prior
            # Gen > 0: Perturb a particle from previous generation (Basic Kernel)
            if gen == 0:
                beta = np.random.uniform(0.01, 0.5)
            else:
                # Pick a survivor from previous gen and add small noise
                parent_beta = population[np.random.randint(len(population))][0]
                sigma = 0.02 # Perturbation width
                beta = parent_beta + np.random.normal(0, sigma)
                beta = np.clip(beta, 0.01, 0.5) # Keep within bounds

            # --- B. RUN SIMULATION ---
            # Print a dot for every attempt to show life
            print(".", end="", flush=True) 
            
            dist = run_simulation(beta, base_config, norm_truth)
            
            # --- C. ACCEPT / REJECT ---
            if dist < current_epsilon:
                new_population.append((beta, dist))
                print(f" [Acc: {dist:.4f}]", end="") 

        # --- D. ADAPTIVE THRESHOLDING ---
        population = new_population
        
        # Sort population by distance (best to worst)
        population.sort(key=lambda x: x[1])
        
        # Calculate new epsilon for next generation
        # We take the distance of the particle at the "Quantile" (e.g. 50th percentile)
        # This ensures the next generation MUST be better than the average of this one.
        cutoff_index = int(len(population) * QUANTILE)
        new_epsilon = population[cutoff_index][1]
        
        print(f"\n\n>>> Generation {gen+1} Complete.")
        print(f"    Best Distance: {population[0][1]:.4f}")
        print(f"    Best Beta: {population[0][0]:.4f}")
        print(f"    New Threshold for Gen {gen+2}: {new_epsilon:.4f}")
        current_epsilon = new_epsilon

    # --- FINAL RESULTS ---
    print("\n" + "="*30)
    print("FINAL ESTIMATION")
    print("="*30)
    betas = [p[0] for p in population]
    print(f"Estimated Beta: {np.mean(betas):.4f} +/- {np.std(betas):.4f}")
    print(f"Best Particle:  Beta={population[0][0]:.4f} (Error={population[0][1]:.4f})")