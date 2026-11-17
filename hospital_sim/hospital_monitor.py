import pandas as pd
import serial
import time
import json
import os
import requests
import sys
import urllib.parse
import argparse
from dotenv import load_dotenv

load_dotenv()


# config
SERIAL_PORT = '/dev/ttyACM0' 
BAUD_RATE = 115200
CSV_FILE = 'mitbih_test.csv'
# Flat baseline streamed after every beat, so the Pico's trigger always sees a
# settled baseline before the next R-peak. holdout.json overrides this value.
INTER_BEAT_GAP = 8
SAMPLE_RATE_HZ = 125   # the Kaggle MIT-BIH rows are resampled to 125 Hz

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        df = pd.read_csv(file_path, header=None)
        
        # Filter for Normal (0), S-Type (1), and V-Type (2)
        df = df[df[187].isin([0.0, 1.0, 2.0])]
        
        # Stream only beats the model never trained on. train_tinyml.py writes
        # the CSV row numbers of its holdout split to holdout.json next to this
        # script; fall back to the old "rows after 16000" rule if it is missing.
        # Dataset rows are 1.2 heart periods long, so each contains the onset of
        # the NEXT beat; holdout.json also says how many samples to stream so
        # that one row is exactly one beat on the wire.
        holdout_path = os.path.join(script_dir, "holdout.json")
        if os.path.exists(holdout_path):
            with open(holdout_path) as f:
                meta = json.load(f)
            cut = {int(r): int(n) for r, n in meta["rows"]}
            gap_len = int(meta.get("inter_beat_gap", INTER_BEAT_GAP))
            unseen = df[df.index.isin(cut.keys())]
            lengths = [cut[r] for r in unseen.index]
            print(f"Holdout split from holdout.json: {len(unseen)} beats never used in training.")
        else:
            unseen = df.iloc[16000:]
            lengths = [187] * len(unseen)
            gap_len = INTER_BEAT_GAP
            print("Warning: holdout.json not found; streaming full rows after 16000, which may overlap training.")
        gap = [0.0] * gap_len
        data = [list(row[:n]) + gap for row, n in zip(unseen.iloc[:, :187].values, lengths)]
        labels = unseen[187].values.astype(int)

        print(f"Loaded {len(data)} 'Unseen' Test Samples.")
        return data, labels
    except Exception as e:
        print(f"CSV Read Error: {e}")
        return None

CLASS_NAMES = {0: "NORMAL", 1: "S-TYPE", 2: "V-TYPE"}

def build_playlist(data, labels, per_class):
    """Demo order for the portfolio video: V-TYPE first, then S-TYPE, then NORMAL."""
    order = []
    for cls in (2, 1, 0):
        idx = [i for i, l in enumerate(labels) if l == cls][:per_class]
        if len(idx) < per_class:
            print(f"Warning: only {len(idx)} unseen beats of class {CLASS_NAMES[cls]}")
        order.extend(idx)
    return [data[i] for i in order], labels[order]

def parse_args():
    ap = argparse.ArgumentParser(description="Stream MIT-BIH beats to the Pico over USB serial.")
    ap.add_argument("--port", default=SERIAL_PORT,
                    help="serial device (Pi: /dev/ttyACM0, Mac: /dev/tty.usbmodem*)")
    ap.add_argument("--demo", action="store_true",
                    help="play a fixed playlist: N x V-TYPE, N x S-TYPE, N x NORMAL")
    ap.add_argument("--per-class", type=int, default=10, help="beats per class in --demo (default 10)")
    return ap.parse_args()

def main():
    args = parse_args()
    port = args.port
    # 1. Setup Serial Connection
    try:
        if not os.path.exists(port):
            print(f"Warning: Pico not found at {port}")
            print(f"       (Running in simulation mode - Screen only)")
            ser = None
        else:
            ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
            print(f"Connected to Pico at {port}")
    except Exception as e:
        print(f"Connection Failed: {e}")
        return

    # 2. Load Data (Auto-Download included)
    loaded = load_data()
    if loaded is None: return
    heartbeats, labels = loaded
    if args.demo:
        heartbeats, labels = build_playlist(heartbeats, labels, args.per_class)
        print(f"DEMO MODE: {len(heartbeats)} beats, order V-TYPE -> S-TYPE -> NORMAL")
    matched = 0

    print("-" * 50)
    print("STARTING SIMULATION: Cloud Data -> Pi 5 -> Pico")
    print("-" * 50)

    try:
        for i, beat in enumerate(heartbeats):
            expected = CLASS_NAMES[labels[i]]
            print(f"\n--- Sending Heartbeat #{i+1}  (expected: {expected}) ---")
            replies = ""
            
            for val in beat:
                packet = {"val": float(val)}
                msg = json.dumps(packet) + "\n"
                
                # Send to Pico if connected
                if ser:
                    ser.write(msg.encode('utf-8'))

                    if ser.in_waiting > 0:
                        raw_data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                        if raw_data.strip():
                            replies += raw_data
                            print(f"PICO DIAGNOSIS: {raw_data.strip()}")

                time.sleep(1 / SAMPLE_RATE_HZ)

            time.sleep(0.2)
            if ser and expected in replies:
                matched += 1

        if ser:
            print("-" * 50)
            print(f"Pico agreed with the dataset label on {matched}/{len(heartbeats)} beats.")

    except KeyboardInterrupt:
        print("\nStopping...")
        if ser: ser.close()

if __name__ == "__main__":
    main()