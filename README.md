# Cloud-to-Edge Heart Rate Monitor

A distributed embedded system that replays ECG data like a patient monitor and classifies each heartbeat in real time on a dual-core microcontroller, using a small int8-weight neural network with inference code written by hand in C.


---

## Overview

This project borrows a pattern from safety-critical medical devices, one core doing the work and the other watching it, on a Raspberry Pi Pico (RP2040). It dedicates **Core 0** to real-time AI inference (detecting heart arrhythmias) and **Core 1** to a software watchdog that raises an alarm when the data stream or Core 0 stops.

The system is validated using a **Hardware-in-the-Loop (HIL)** test bench where a Raspberry Pi 5 acts as a "Patient Simulator," streaming clinical data from Google Cloud Storage over USB Serial.

---

## System Architecture

The system is designed with a **"Doctor and Bodyguard"** dual-core architecture:

| Component | Role | Technology Stack |
|-----------|------|------------------|
| **Raspberry Pi 5** | The Patient (Simulator) | Python, Pandas, Google Cloud API |
| **Pico Core 0** | The Doctor (AI) | C on the Pico SDK, int8-weight MLP |
| **Pico Core 1** | The Bodyguard (Safety) | Polling watchdog (every 50 ms), no RTOS |

### Key Features

- **Hand-Written Inference, No TensorFlow Lite Micro**: A 32-32-3 MLP in plain C. Weights are stored as int8 with one scale per layer (1,120 bytes); biases and arithmetic stay in float. The quantization saves memory, not compute: each weight is converted back to float before the multiply, and the Cortex-M0+ has no floating-point unit, so the math runs in software float. The int8 model agrees with the float model on 99.90% of held-out beats (`training/RESULTS.md`).

- **One Window Definition, Both Sides**: The firmware finds the R-peak with a threshold-and-slope rule and classifies the 32 samples that follow it. The training script cuts its windows from the dataset with the same rule, so the model is trained on exactly what the device feeds it. A dry run in `train_tinyml.py` streams every held-out beat through a copy of the firmware's trigger state machine and checks the windows match.

- **Dual-Core Watchdog**: Core 0 stamps a shared timestamp every time it processes a packet. Core 1 checks that timestamp every 50 ms and, if no packet has been processed for 1 s, turns on the panic LED. One check catches both failures: a stopped data stream and a hung Core 0.

- **Cloud Data Pipeline**: The training script and the simulator download the MIT-BIH beats from a Google Cloud Storage bucket, with the API key read from a local `.env` file that is not in git. Without a key, both use a local copy of the CSV (see below).

- **Closed-Loop Verification**: The Pico reports diagnosis results (NORMAL, S-TYPE, V-TYPE) back to the host via USB Serial for automated validation.

---

## Visual Indicators

Four external LEDs on a breadboard plus the onboard LED provide real-time diagnostic feedback:

| LED | GPIO (pin) | Pattern | Meaning |
|-----|------------|---------|---------|
| Green | GP16 (21) | 1 blink | **NORMAL**: healthy heartbeat |
| Yellow | GP17 (22) | 2 blinks | **S-TYPE**: supraventricular ectopic beat |
| Red | GP18 (24) | 3 blinks | **V-TYPE**: ventricular ectopic beat (danger) |
| Blue | GP19 (25) | solid 2 s, then fast strobe | **PANIC**: data stream halted / watchdog triggered |
| Onboard | GP25 | on | link armed: packets are arriving |
| All, in turn | | lamp test, then 3 onboard blinks | **BOOT**: every indicator is proven working |

Wiring, per LED: GPIO to the LED's long leg, short leg through 330 Ω to a Pico GND pin (pin 23). The resistor can sit on either side of the LED.

---

## Getting Started

### Prerequisites

- **Hardware**: Raspberry Pi Pico (with headers), Raspberry Pi 5 (or Mac/PC), Micro-USB data cable, breadboard, 4 LEDs, 4 × 330 Ω resistors
- **Tools**: VS Code, CMake, ARM GCC Toolchain, Python 3.10+
- **Pico SDK**: cloned locally, with `PICO_SDK_PATH` pointing at it
- **Data**: `mitbih_test.csv` from the Kaggle [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) (MIT-BIH beats, resampled to 125 Hz). Put it in `training/` and `hospital_sim/`, or host it in your own Cloud Storage bucket and put `GCP_API_KEY=...` in a `.env` file. Only this file is used (re-split here 70/15/15 per class into train / validation / holdout); the larger Kaggle train file is not. Fusion (F) and unclassifiable (Q) beats are dropped, leaving three classes

### 1. Train the Model (Python)

The training script downloads data from Google Cloud (or uses a local copy), shuffles it and splits it per class into train / validation / holdout, trains a Dense Neural Network using Keras with class weights, quantizes it, and generates the C header (`model_weights.h`) together with the list of held-out beats the simulator is allowed to stream.

```bash
# Install requirements
pip install tensorflow pandas numpy requests python-dotenv

# Run training (add --csv /path/to/mitbih_test.csv to skip the download)
python3 training/train_tinyml.py
```

**Output:** `firmware/src/model_weights.h`, `hospital_sim/holdout.json`, `training/RESULTS.md` (confusion matrices, int8-vs-float agreement, stream dry run)

### 2. Build the Firmware (C)

Compile the firmware using the Pico SDK.

```bash
export PICO_SDK_PATH=/path/to/pico-sdk   # e.g. ~/pico-sdk
cd firmware
mkdir build && cd build
cmake ..
make
```

If CMake cannot find `arm-none-eabi-gcc`, also `export PICO_TOOLCHAIN_PATH=/path/to/toolchain/bin`.

**Output:** `hrm_firmware.uf2`

### 3. Flash the Device

Hold the **BOOTSEL** button on the Pico while plugging it in, then drag-and-drop the `.uf2` file onto the `RPI-RP2` drive.

### 4. Run the Patient Simulator

The simulator connects to the Pico via USB Serial (`/dev/ttyACM0` on Linux/Pi) and streams only beats from the holdout split, one beat per row, cut just before the next beat's onset.

```bash
# On Raspberry Pi 5
pip install pandas pyserial requests python-dotenv
python3 hospital_sim/hospital_monitor.py            # stream every held-out beat
python3 hospital_sim/hospital_monitor.py --demo     # 10 V-TYPE, 10 S-TYPE, 10 NORMAL, with a tally
python3 hospital_sim/hospital_monitor.py --port /dev/tty.usbmodem1101   # from a Mac
```

Each verdict the Pico sends back is printed next to the dataset label, and `--demo` ends with how many beats agreed.

---

## Safety Demonstration (Watchdog)

To see the watchdog fire:

1. Run the simulator (`hospital_monitor.py`)
2. Observe the LED blinking in sync with the data
3. Kill the simulator (Ctrl + C) to simulate a sensor failure or software hang

**Observation:** Within 1.1 seconds every indicator goes dark, the blue LED turns **solid** (flatline detected) and two seconds later switches to a **hyper-fast strobe** (system halted). Power-cycle the Pico to recover.

---

## Repository Structure

```
.
├── firmware/                   # C Source Code for RP2040
│   ├── src/main.c              # Dual-core logic & Inference engine
│   ├── src/model_weights.h     # Auto-generated Quantized Weights
│   └── CMakeLists.txt          # Build configuration
├── hospital_sim/               # Python Patient Simulator
│   ├── hospital_monitor.py     # Streamer & Cloud Fetcher
│   ├── holdout.json            # Beats never used in training (generated)
│   └── mitbih_test.csv         # Clinical Data (Auto-downloaded, not in git)
├── training/                   # AI Training Pipeline
│   ├── train_tinyml.py         # TensorFlow script + C Code Generator + stream dry run
│   └── RESULTS.md              # Latest training report (generated)
└── README.md
```

---

## Results

Holdout split (15% of beats, never seen in training), int8 model as the Pico computes it:

| | NORMAL | S-TYPE | V-TYPE |
|---|---|---|---|
| Recall | 96.9% | 41.7% | 85.8% |
| Precision | 97.2% | 83.3% | 69.8% |

Overall accuracy 94.6%. The int8 model gives the same answer as the float model on 99.90% of holdout beats. Full confusion matrices are in `training/RESULTS.md`.

`training/RESULTS.md` is regenerated on every training run. It reports per-class recall and precision on the validation and holdout splits for the float model and for the int8 model as the Pico computes it, the agreement between the two, and a dry run of the full stream through a copy of the firmware's trigger. V-type beats are the strong class; S-type beats differ from normal beats mainly in timing rather than shape, so a shape-only window finds fewer than half of them (41.7% on the holdout), and the report says so rather than hiding it behind an overall accuracy figure.

### Limitations

- **The split is by beat, not by patient.** The Kaggle CSV has no patient or record IDs, so beats are shuffled and split at random. Held-out beats can come from the same people as training beats, which makes the holdout numbers optimistic for a patient the model has never seen.
- **S-type beats are mostly missed.** The network sees one 256 ms window of beat shape and no RR interval, and supraventricular beats differ from normal beats mostly in timing.
- **LED signalling blocks Core 0.** A V-type verdict blinks for about 300 ms with `sleep_ms`, during which Core 0 reads no samples; USB flow control holds them back and Core 0 catches up afterwards, but classification lags the stream. Moving the blinks to Core 1 or a hardware timer would remove that.
- **The input is a replayed dataset, not an ECG front end.** There is no ADC, filtering or electrode noise in the loop.

