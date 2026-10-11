# Dual-Core Safety-Critical Arrhythmia Detector

A distributed embedded system that simulates a patient monitor and analyzes ECG data in real-time using a custom quantized neural network on a dual-core microcontroller.


---

## Overview

This project demonstrates a safety-critical medical device architecture using the Raspberry Pi Pico (RP2040). It dedicates **Core 0** to real-time AI inference (detecting heart arrhythmias) and **Core 1** to a hard-real-time safety watchdog.

The system is validated using a **Hardware-in-the-Loop (HIL)** test bench where a Raspberry Pi 5 acts as a "Patient Simulator," streaming clinical data from Google Cloud Storage over USB Serial.

---

## System Architecture

The system is designed with a **"Doctor and Bodyguard"** dual-core architecture:

| Component | Role | Technology Stack |
|-----------|------|------------------|
| **Raspberry Pi 5** | The Patient (Simulator) | Python, Pandas, Google Cloud API |
| **Pico Core 0** | The Doctor (AI) | Bare-Metal C, Quantized TinyML |
| **Pico Core 1** | The Bodyguard (Safety) | RTOS-style Watchdog, Interrupts |

### Key Features

- **Zero-Overhead Inference**: Implemented a custom C inference engine for a Hybrid-Quantized Neural Network (Int8 weights / Float math), eliminating the need for heavy libraries like TensorFlow Lite Micro. The whole model is 1120 int8 weights and 35 float biases.

- **One Window Definition, Both Sides**: The firmware finds the R-peak with a threshold-and-slope rule and classifies the 32 samples that follow it. The training script cuts its windows from the dataset with the same rule, so the model is trained on exactly what the device feeds it. A dry run in `train_tinyml.py` streams every held-out beat through a copy of the firmware's trigger state machine and checks the windows match.

- **Dual-Core Safety**: Core 1 monitors Core 0's execution time. If the data stream halts or the AI hangs for >1 second, Core 1 triggers a hardware panic strobe.

- **Cloud Data Pipeline**: The simulator securely fetches the MIT-BIH Arrhythmia Database from a Google Cloud Storage bucket using API Key authentication, ensuring deterministic testing data.

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
- **Cloud**: A Google Cloud Project with the Storage JSON API enabled

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
cd firmware
mkdir build && cd build
cmake ..
make
```

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

To verify the safety-critical watchdog:

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

---

## Results

`training/RESULTS.md` is regenerated on every training run. It reports per-class recall and precision on the validation and holdout splits for the float model and for the int8 model as the Pico computes it, the agreement between the two, and a dry run of the full stream through a copy of the firmware's trigger. V-type beats are the strong class; S-type beats differ from normal beats mainly in timing rather than shape, so a shape-only window finds about half of them, and the report says so rather than hiding it behind an overall accuracy figure.
