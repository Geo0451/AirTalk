# AirTalk: Low-Latency Inertial Air-Writing & Gesture Pipeline

A deterministic, edge-based air-writing system utilizing an inertial measurement unit (IMU) and magnetometer to map free-space hand trajectories directly to text and desktop control actions without cloud inference or heavy neural networks.

---

## System Architecture

* **Embedded Firmware (`AirTalk3.ino`)**: Samples the MPU6050 and an auto-detected magnetometer (QMC5883L/P or HMC5883L) at a strict **100 Hz** rate on an ESP32. Core 1 handles sensor sampling; Core 0 manages Wi-Fi SoftAP and low-overhead binary UDP packet streaming.
* **Telemetry Pipeline**: Streams 41-byte packed binary packets directly to a host PC over an independent access point, removing router pairing friction.
* **Calibration Engine**: Features an automated host-side calibration wizard for gyro bias re-zeroing, pen-axis mounting extraction, and automated magnetometer hard/soft-iron ellipse fitting with axis-alignment verification.

---

## Recognition Engine (DTW 1-NN)

* **Feature Extraction (`airpen_common.py`)**: Computes orientation via a Madgwick MARG filter combining gyro, accelerometer, and calibrated magnetometer data. Evaluates **seven distinct feature variants (A–G)** covering raw body frame, gravity-removed world frames, and magnetic-north alignment.
* **Classification**: Employs **Dynamic Time Warping (1-NN DTW)** with Sakoe-Chiba banding for template matching against user-defined handwriting samples.
* **Performance Metrics**: Achieves **86.0% leave-one-out (LOO) accuracy across 10 digits** and **85.6% across a 36-class digits+letters corpus** using minimal training exemplars.

---

## Dataset Management & Quality Control

* **`collect.py`**: Guided data collection featuring real-time DTW quality gates to intercept misclassifications (inter-class overmatching) and intra-class redundancies at point-of-capture.
* **`prune.py`**: Automated dataset hygiene script that scans and removes intra-class duplicates, boundary-polluting templates, and enforces per-class capacity caps.
* **Active Training (`dtw_eval.py`)**: Sentence-driven training loop that tests recognition before learning, utilizing runtime misclassifications to incrementally expand the template library.

---

## Gesture HUD & WebSocket Control

* **Motion Mapping (`airtalk_server.py`, `airtalk_live.html`)**: Detects **8 native physical motions** (flick up/down/left/right, twist clockwise/anticlockwise, thrust forward/backward).
* **Live Configuration**: Fully remappable action layer (backspace, space, enter, case toggle, cursor navigation) synchronized instantly across connected clients via WebSockets with zero server restarts.

---

## Repository Layout

| Component | Technical Scope |
| :--- | :--- |
| `AirTalk3.ino` | FreeRTOS multitasking, 100 Hz IMU/mag sampling, UDP binary telemetry. |
| `airpen_common.py` | Sensor fusion, Madgwick MARG filter, magnetometer calibration, variants A–G. |
| `collect.py` | Interactive sample gathering with live DTW validation gates. |
| `dtw_eval.py` | Offline cross-validation, console inference, and active training loops. |
| `airtalk_server.py` | WebSocket server, Typist state machine, and dynamic config state. |
| `airtalk_live.html` | Client-side DOM rendering, real-time trajectory canvas, and settings HUD. |
| `prune.py` | Template set optimization and boundary cleaning utility. |