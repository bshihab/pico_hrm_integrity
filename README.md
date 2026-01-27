# Dual-Core Safety-Critical Arrhythmia Detector

A distributed embedded system that simulates a patient monitor and analyzes ECG data in real-time using a custom quantized neural network on a dual-core microcontroller.

**Status:** Complete & Verified (Hardware-in-the-Loop)

---

## ⚡ Overview

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

- **Zero-Overhead Inference**: Implemented a custom C inference engine for a Hybrid-Quantized Neural Network (Int8 weights / Float math), eliminating the need for heavy libraries like TensorFlow Lite Micro.

- **Dual-Core Safety**: Core 1 monitors Core 0's execution time. If the data stream halts or the AI hangs for >1 second, Core 1 triggers a hardware panic strobe.

- **Cloud Data Pipeline**: The simulator securely fetches the MIT-BIH Arrhythmia Database from a Google Cloud Storage bucket using API Key authentication, ensuring deterministic testing data.

- **Closed-Loop Verification**: The Pico reports diagnosis results (NORMAL, S-TYPE, V-TYPE) back to the host via USB Serial for automated validation.

---

## Visual Indicators

The Pico's onboard LED provides real-time diagnostic feedback:

| Pattern | Meaning | Condition |
|---------|---------|-----------|
| 1 Blink | **NORMAL** | Healthy Heartbeat Detected |
| 2 Blinks | **S-TYPE** | Supraventricular Ectopic Beat Detected |
| 3 Blinks | **V-TYPE** | Premature Ventricular Contraction (Danger) |
| Fast Strobe | **PANIC** | System Crash / Watchdog Triggered |
| 3 Slow Blinks | **BOOT** | System Startup |

---

## Getting Started

### Prerequisites

- **Hardware**: Raspberry Pi Pico, Raspberry Pi 5 (or Mac/PC), Micro-USB Data Cable
- **Tools**: VS Code, CMake, ARM GCC Toolchain, Python 3.10+
- **Cloud**: A Google Cloud Project with the Storage JSON API enabled

### 1. Train the Model (Python)

The training script downloads data from Google Cloud, trains a Dense Neural Network using Keras, and automatically generates a C header file (`model_weights.h`) with quantized weights.

```bash
# Install requirements
pip install tensorflow pandas numpy requests python-dotenv

# Run training
python3 training/train_tinyml.py
```

**Output:** Generates `firmware/src/model_weights.h`

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

The simulator connects to the Pico via USB Serial (`/dev/ttyACM0` on Linux/Pi) and streams ECG data.

```bash
# On Raspberry Pi 5
python3 hospital_sim/hospital_monitor.py
```

---

## Safety Demonstration (Watchdog)

To verify the safety-critical watchdog:

1. Run the simulator (`hospital_monitor.py`)
2. Observe the LED blinking in sync with the data
3. Kill the simulator (Ctrl + C) to simulate a sensor failure or software hang

**Observation:** Within 1.1 seconds, the Pico LED will turn **Solid Green** (Flatline detected) and then switch to a **Hyper-Fast Strobe** (System Halted).

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
│   └── mitbih_test.csv         # Clinical Data (Auto-downloaded)
├── training/                   # AI Training Pipeline
│   └── train_tinyml.py         # TensorFlow script + C Code Generator
└── README.md
```

---
