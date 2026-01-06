import pandas as pd
import time
import json
import sys
import os
import math

# --- CONFIGURATION ---
POSSIBLE_FILENAMES = ['mitbih_test.csv', '100.csv']
SAMPLE_RATE_HZ = 100 
SLEEP_TIME = 1.0 / SAMPLE_RATE_HZ

def generate_synthetic_data():
    """Generates a fake ECG signal if no CSV is found."""
    print("WARNING: CSV not found. Generating synthetic data...", file=sys.stderr)
    data = []
    for i in range(500):
        phase = i % 100
        val = 0.8 if (phase > 10 and phase < 20) else 0.1
        data.append(val)
    return data

def load_and_stitch_data():
    # 1. Get the folder where THIS script lives (hospital_sim)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Also check the current working directory just in case
    cwd = os.getcwd()
    
    found_path = None
    
    # Check both locations for the file
    search_dirs = [script_dir, cwd, os.path.join(cwd, 'hospital_sim')]
    
    for folder in search_dirs:
        for name in POSSIBLE_FILENAMES:
            path = os.path.join(folder, name)
            if os.path.exists(path):
                found_path = path
                break
        if found_path: break
            
    if not found_path:
        print(f"Error: Could not find CSV in: {search_dirs}", file=sys.stderr)
        return generate_synthetic_data()
        
    print(f"Loading {found_path}...", file=sys.stderr)
    try:
        df = pd.read_csv(found_path, header=None)
        subset = df.iloc[0:100, :-1] 
        continuous_signal = subset.values.flatten().tolist()
        print(f"Stitched {len(continuous_signal)} samples.")
        return continuous_signal
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        return generate_synthetic_data()

def stream_heartbeat(data):
    if not data: return
    index = 0
    total_samples = len(data)
    print("Starting Continuous Stream...", file=sys.stderr)
    while True:
        packet = {"val": data[index], "status": "ok"}
        print(json.dumps(packet))
        sys.stdout.flush() 
        index += 1
        if index >= total_samples: index = 0
        time.sleep(SLEEP_TIME)

if __name__ == "__main__":
    patient_data = load_and_stitch_data()
    try:
        stream_heartbeat(patient_data)
    except KeyboardInterrupt:
        pass