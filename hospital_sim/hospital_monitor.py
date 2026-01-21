import pandas as pd
import serial
import time
import json
import os
import requests
import sys
import urllib.parse

# config
SERIAL_PORT = '/dev/ttyACM0' 
BAUD_RATE = 115200
CSV_FILE = 'mitbih_test.csv'

# google cloud config
GCP_API_KEY = os.getenv("GCP_API_KEY") 
GCP_BUCKET_NAME = "heart-data-repo-1" 
GCP_OBJECT_NAME = "mitbih_test.csv"

def get_google_cloud_url():
    
    encoded_name = urllib.parse.quote(GCP_OBJECT_NAME, safe='')
    url = f"https://storage.googleapis.com/storage/v1/b/{GCP_BUCKET_NAME}/o/{encoded_name}?alt=media&key={GCP_API_KEY}"
    return url

# Set the source to the Function above
DATA_SOURCE_NAME = "Google Cloud Storage"

def fetch_data_from_cloud():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, CSV_FILE)

    # If file exists and isn't empty, we are good
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        print(f"Data found locally: {file_path}")
        return file_path

    target_url = get_google_cloud_url()

    print(f"Dataset not found locally.")
    print(f"Downloading from {DATA_SOURCE_NAME}...")
    # Don't print the full URL to keep the API Key secret in logs
    print(f"       Target Bucket: {GCP_BUCKET_NAME}")

    try:
        response = requests.get(target_url, stream=True)
        
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Download Complete: Saved to {file_path}")
            return file_path
        elif response.status_code == 403:
            print(f"Access Denied (403). Check your API Key permissions.")
            return None
        elif response.status_code == 404:
            print(f"File Not Found (404). Check Bucket/Object name.")
            return None
        else:
            print(f"Cloud Error: Status Code {response.status_code}")
            print(f"      Response: {response.text}")
            return None
    except Exception as e:
        print(f"Connection Error: {e}")
        return None

def load_data():
    # 1. Ensure we have the data (Download if missing)
    file_path = fetch_data_from_cloud()
    
    if not file_path:
        return None
    
    print(f"Loading heartbeats from {os.path.basename(file_path)}...")
    try:
        df = pd.read_csv(file_path, header=None)
        
        # Filter for Normal (0), S-Type (1), and V-Type (2)
        df = df[df[187].isin([0.0, 1.0, 2.0])]
        
        # Use data AFTER row 16000 to simulate "New Patients" (Validation Set)
        data = df.iloc[16000:, :187].values
        
        print(f"Loaded {len(data)} 'Unseen' Test Samples.")
        return data
    except Exception as e:
        print(f"CSV Read Error: {e}")
        return None

def main():
    # 1. Setup Serial Connection
    try:
        if not os.path.exists(SERIAL_PORT):
            print(f"Warning: Pico not found at {SERIAL_PORT}")
            print(f"       (Running in simulation mode - Screen only)")
            ser = None
        else:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            print(f"Connected to Pico at {SERIAL_PORT}")
    except Exception as e:
        print(f"Connection Failed: {e}")
        return

    # 2. Load Data (Auto-Download included)
    heartbeats = load_data()
    if heartbeats is None: return

    print("-" * 50)
    print("STARTING SIMULATION: Cloud Data -> Pi 5 -> Pico")
    print("-" * 50)

    try:
        for i, beat in enumerate(heartbeats):
            print(f"\n--- Sending Heartbeat #{i+1} ---")
            
            for val in beat:
                packet = {"val": float(val)}
                msg = json.dumps(packet) + "\n"
                
                # Send to Pico if connected
                if ser:
                    ser.write(msg.encode('utf-8'))

                    if ser.in_waiting > 0:
                        try:
                            raw_data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                            if raw_data.strip():
                                print(f"PICO DIAGNOSIS: {raw_data.strip()}") 
                        except:
                            pass
                
                time.sleep(0.01) 

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nStopping...")
        if ser: ser.close()

if __name__ == "__main__":
    main()