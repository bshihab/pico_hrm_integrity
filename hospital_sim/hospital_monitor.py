import pandas as pd
import serial
import time
import json
import os

# config
BAUD_RATE = 115200
CSV_FILE = 'mitbih_test.csv'

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, CSV_FILE)

    if not os.path.exists(file_path):
        if os.path.exists(CSV_FILE):
            file_path = CSV_FILE
        else:
            print(f"CSV file not found at {file_path}")
            return None
    
    print(f"Loading heartbeats from {file_path}...")
    try:
        df = pd.read_csv(file_path, header=None)
        # Filter for Normal (0) and PVC (2) only
        # Column 187 is the label
        df = df[df[187].isin([0.0, 2.0])]
        data = df.iloc[:, :187].values
        return data
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"Connected to Pico at {SERIAL_PORT}")
    except Exception as e:
        print(f"Connection Failed: {e}")
        return

    # 2. Load Data
    heartbeats = load_data()
    if heartbeats is None: return

    print("starting simulation: Sending Data + Listening for Diagnosis")
    print("-" * 50)

    try:
        for i, beat in enumerate(heartbeats):
            print(f"\n--- Sending Heartbeat #{i+1} ---")
            
            # Send the heartbeat point by point
            for val in beat:
                packet = {"val": float(val)}
                msg = json.dumps(packet) + "\n"
                ser.write(msg.encode('utf-8'))

                if ser.in_waiting > 0:
                    try:
                        raw_data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                        if raw_data.strip():
                            # Pico sends JSON like {"pico_status": "DANGER", "prob": 0.9}
                            print(f"PICO DIAGNOSIS: {raw_data.strip()}") 
                    except:
                        pass

                time.sleep(0.01) 

            # Small pause between heartbeats
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nStopping...")
        ser.close()

if __name__ == "__main__":
    main()