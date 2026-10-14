# AirTalk 4: Low-Latency Inertial Air-Writing & Gesture Pipeline

A deterministic, edge-based air-writing system utilizing an inertial measurement unit (IMU) and magnetometer to map free-space hand trajectories directly to text, desktop control actions, and air-mouse navigation without cloud inference or heavy neural networks.

## Core System Architecture

* **Embedded Firmware (`AirTalk4.ino`)**: Samples the MPU6050 and auto-detected magnetometers (QMC5883L/P or HMC5883L) at a strict **100 Hz** rate on an ESP32. Core 1 handles time-critical sensor sampling; Core 0 manages transport networking (Wi-Fi SoftAP UDP broadcasting or Bluetooth LE GATT notifications).

* **Dual Transport System**: Seamlessly switch between local Wi-Fi UDP streaming and **Bluetooth LE (BLE)** with automatic connection parameter tuning and an embedded BLE watchdog to eliminate orphaned link states.

* **Telemetry Pipeline**: Streams 41-byte packed binary packets directly to a host PC with minimal jitter and zero parsing overhead.

* **Calibration Engine**: Features an automated host-side calibration wizard for gyro bias re-zeroing, pen-axis mounting extraction, and automated magnetometer hard/soft-iron ellipse fitting with axis-alignment verification.

## Recognition Engine & Active Learning

* **Feature Extraction (`airpen_common.py`)**: Computes orientation via Madgwick IMU and MARG filters combining gyro, accelerometer, and calibrated magnetometer data. Evaluates **seven distinct feature variants (A–G)** covering raw body frame, gravity-removed world frames, and magnetic-north alignment.

* **Classification**: Employs **Dynamic Time Warping (1-NN DTW)** with Sakoe-Chiba banding for template matching against user-defined handwriting samples.

* **Active Training (`dtw_eval.py --active`)**: Interactive "test-then-train" dictation system that tests recognition before learning, dynamically saving misclassifications or low-margin captures directly into the active template pool.

* **Real-time Quality Gates (`collect.py`)**: Evaluates captured strokes live against existing templates to filter out redundant strokes (`--sim-threshold`) or cross-class overmatches before saving to disk.

* **Performance Metrics**: Achieves **86.0% leave-one-out (LOO) accuracy across 10 digits** and **85.6% across a 36-class digits+letters corpus** using minimal training exemplars.

## Air Mouse & Live Gestures

* **Air Mouse (`airmouse.py`)**: Ballistic non-linear cursor acceleration with gesture triggers (Single Tap = Left Click, Double Tap = Middle Click, Long Hold $\ge 0.5\text{s}$ = Right Click) for Wayland desktops via `ydotool`.

* **Gesture Mode (`airtalk_server.py`)**: Full dynamic 3D gesture support for text editing (Flick Up/Down/Left/Right, Twist Clockwise/Anticlockwise, Thrust Forward/Backward). Fully remappable live via the `airtalk_live.html` HUD.

## Repository Layout

| Component | Technical Scope | 
| ----- | ----- | 
| `AirTalk4.ino` | FreeRTOS multitasking, 100 Hz IMU/mag sampling, UDP & BLE transport switches. | 
| `airmouse.py` | Wayland air mouse controller with velocity ballistics and `ydotool` integration. | 
| `airpen_common.py` | Sensor fusion, Madgwick MARG filter, magnetometer calibration, variants A–G. | 
| `collect.py` | Interactive sample gathering with live DTW overmatch & redundancy gates. | 
| `dtw_eval.py` | Offline cross-validation, console live inference, and active training dictation loops. | 
| `airtalk_server.py` | WebSocket server, remappable gesture engine, and dynamic config state. | 
| `airtalk_live.html` | Client-side DOM rendering, real-time trajectory canvas, and settings HUD. | 
| `prune.py` | Multi-pass dataset pruning tool (duplicate removal, boundary polluter cleaning, class capacity caps). | 

## Quick Start & Common Workflow

1. **Flash Firmware**: Open `AirTalk4.ino` in Arduino IDE, set `#define USE_BLE 1` (or `0` for UDP), and flash to your ESP32.

2. **Calibrate Hardware**: Run the full wizard to extract pen axis mounting and magnetometer profile:

   ```
   python3 collect.py --calibrate
   
   ```

3. **Collect Baseline Dataset**:

   ```
   python3 collect.py --labels digits,upper --reps 10
   
   ```

4. **Refine via Active Training**:

   ```
   python3 dtw_eval.py --active --variant A --text "the quick brown fox jumps over the lazy dog"
   
   ```

5. **Prune Dataset & Run Web Application**:

   ```
   python3 prune.py --sim-threshold 4.5 --max-per-class 10 --apply
   python3 airtalk_server.py --live --variant A
   
   ```