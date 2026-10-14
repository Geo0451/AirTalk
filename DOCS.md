# AirTalk 4 Technical Documentation

## 1. System Overview

AirTalk 4 is a deterministic, low-latency inertial air-writing and gesture recognition pipeline designed for embedded microcontrollers (ESP32) and host-side Python systems. By bypassing heavy cloud-based neural inference and utilizing edge-based sensor fusion alongside Dynamic Time Warping (DTW 1-NN), the system maps free-space hand trajectories directly into text and desktop control actions with minimal latency.

---

## 2. Firmware Architecture (`AirTalk4.ino`)

The firmware runs on an ESP32 microcontroller utilizing **FreeRTOS** to separate time-critical sensor acquisition from network transmission.

### 2.1 Multitasking & Core Allocation
* **Core 1 (`IMU_Task`)**: Dedicated to high-speed sensor polling at a strict $100\text{ Hz}$ rate ($10\text{ ms}$ intervals). It reads raw accelerometer, gyroscope, and magnetometer values, applies touch debouncing, computes timestamped telemetry packets, and pushes them to a FreeRTOS queue (`telemetryQueue`).
* **Core 0 (`NET_Task` / BLE tasks)**: Handles communication overhead, network buffering, and transmission without blocking the deterministic sensor loop on Core 1.

### 2.2 Dual Transport Modes (`USE_BLE` Switch)
The firmware supports two distinct transport protocols, toggled via `#define USE_BLE`:
1. **`USE_BLE 0` (Wi-Fi SoftAP + UDP)**:
   * Boots a Wi-Fi Access Point (`AirTalk_AP`, channel 6, max 1 client) with power-saving sleep disabled (`WiFi.setSleep(false)`).
   * Streams 41-byte binary packets via UDP broadcast to `192.168.4.255:5005`.
2. **`USE_BLE 1` (Bluetooth LE)**:
   * Bypasses Wi-Fi entirely to save power and eliminate router-pairing friction.
   * Advertises as **`AirTalk`** with a custom UUID service (`4f3c5a10-...`).
   * Sends each sample as a GATT notification (`PROPERTY_NOTIFY`) over characteristic `4f3c5a11-...`.
   * **Connection Parameters**: Requests fast connection intervals ($\text{min}=7.5\text{ ms}$, $\text{max}=15\text{ ms}$) to keep per-sample transmission latency low.
   * **BLE Watchdog**: Automatically detects orphaned links (when a host crashes or disconnects uncleanly without un-subscribing from notifications) and forces a server disconnect after a $10\text{ second}$ grace period to resume advertising automatically.

### 2.3 Sensor Integration & Auxiliary Bus Bypass
* **IMU**: MPU6050 configured over Fast I2C ($400\text{ kHz}$) with a hardware Digital Low Pass Filter (DLPF $\approx 42\text{ Hz}$) and programmable full-scale ranges ($\pm 8g$ accelerometer, $\pm 1000\text{ dps}$ gyroscope).
* **Magnetometer Auto-Detection**: Automatically probes the I2C bus at startup to initialize one of three magnetometers:
  * **QMC5883L** at `0x0D` ($\pm 8G$, $200\text{ Hz}$)
  * **HMC5883L** at `0x1E` ($\pm 1.3\text{Ga}$, $75\text{ Hz}$)
  * **QMC5883P** at `0x2C` ($\pm 8G$, $200\text{ Hz}$)
* **AUX Bus Bypass**: Configures the MPU6050 bypass register (`0x37 = 0x02`) to expose auxiliary-mounted magnetometers directly on the primary SDA/SCL pins.

### 2.4 Binary Packet Structure (`TelemetryPacket`)
Each sample is packed into a deterministic **41-byte binary structure** (`__attribute__((packed))`), ensuring zero parsing overhead:

```cpp
struct TelemetryPacket {
  uint32_t timestampUs; // Microsecond timestamp
  uint8_t  touched;     // Touch sensor state (0 or 1)
  float    ax, ay, az;  // Accelerometer raw (g)
  float    gx, gy, gz;  // Gyroscope raw minus bias (deg/s)
  float    mx, my, mz;  // Magnetometer raw (Gauss)
};
```

---

## 3. Python Processing Pipeline (`airpen_common.py`)

The host-side codebase handles multi-transport streaming, calibration wizards, sensor fusion, and feature extraction.

### 3.1 Unified `PenStream` Abstraction
* Supports **UDP**, **BLE** (via `bleak`), and **Auto** mode (which listens on UDP while actively monitoring Bluetooth availability).
* Includes robust background connection management, automatic stale-link cleanup via BlueZ D-Bus integration on Linux, and real-time stream statistics (packet loss, arrival jitter percentiles).

### 3.2 Magnetometer Calibration & Axis Alignment
Magnetometer readings suffer from hard-iron biases and soft-iron distortions. The calibration engine:
1. Fits an axis-aligned ellipsoid (with sphere fallback) across a tumble dataset to extract offset vectors and per-axis scaling factors:
   $$\mathbf{m}_{\text{calibrated}} = (\mathbf{m}_{\text{raw}} - \mathbf{offset}) \odot \mathbf{scale}$$
2. Verifies axis alignment against gravity using a dip-angle consistency check across 24 proper rotation matrices, ensuring the magnetic sensor frame perfectly matches the IMU body frame.

### 3.3 Sensor Fusion & Madgwick Filters
* **Tilt Quaternions**: Initialized from steady-state accelerometer readings.
* **Madgwick IMU / MARG Filter**: Dynamically tracks orientation using gyroscope integration corrected by accelerometer gravity vectors (and calibrated magnetometer magnetic north references).
* **Gated Corrections**: Automatically disables accelerometer correction during high-acceleration writing motions and disables magnetometer correction during magnetic interference anomalies.

### 3.4 Feature Extraction Variants (A–G)
Raw streams are translated into normalized feature matrices across 7 distinct variants:
* **Variant A**: Raw body frame ($6\text{-channel}$: $\text{acc}_b, \text{gyr}_b$).
* **Variant B**: Gravity-removed, world-frame + heading-aligned ($6\text{-channel}$: $\text{lin}_h, \text{gyr}_h$).
* **Variant C**: Variant B plus acceleration and gyroscope magnitudes ($8\text{-channel}$).
* **Variant D**: Yaw-invariant vertical and horizontal projections ($4\text{-channel}$).
* **Variant E**: Variant A plus calibrated body-frame magnetometer ($9\text{-channel}$).
* **Variant F**: Magnetic-north world frame without gravity ($6\text{-channel}$).
* **Variant G**: Variant B with automatic fallback to magnetic-north when the pen is held vertically.

---

## 4. Active Learning, Quality Gates & Dataset Hygiene

AirTalk 4 introduces real-time dataset sanitization and interactive training paradigms to maximize DTW 1-NN performance with minimal template redundancy.

### 4.1 Real-Time Quality Gates (`collect.py`)
`collect.py` evaluates every newly drawn stroke against existing disk templates using live DTW comparison before committing to disk:
* **Overmatch Protection**: Flags strokes closer to a different class template than the requested label (e.g., writing an `o` that matches `6`). Prevents mislabeled template pollution at capture time.
* **Redundancy Suppression**: Checks if DTW distance to same-class templates falls below `--sim-threshold` (default `4.5`). Warns and prompts user to avoid redundant template storage.

### 4.2 Active Training Mode (`dtw_eval.py --active`)
A "test-then-train" active learning pipeline:
1. Dictates target sentences or customized text.
2. Performs recognition using current baseline templates.
3. Automatically prompts correction and saves misclassifications (or low-margin matches below `--learn-margin`) into session files tagged with `_active`.
4. Immediately folds newly captured templates into the active comparison pool.

### 4.3 Automated Dataset Pruning (`prune.py`)
Multi-pass offline template optimization:
1. **Intra-Class Duplicates**: Drops redundant templates within same class below `--sim-threshold`.
2. **Inter-Class Boundary Polluters**: Eliminates templates whose nearest neighbor across the global dataset belongs to a different class.
3. **Capacity Trimming**: Trims class counts down to `--max-per-class` by removing outer outliers based on mean intra-class DTW distance.

---

## 5. Companion Tools & Utilities

* **`airmouse.py`**: Converts the pen into a Wayland Air Mouse using exponential smoothing, velocity-based acceleration ballistics, and `ydotool` bindings. Supports single-tap (Left Click), double-tap (Middle Click), and long-hold (Right Click).
* **`collect.py`**: Interactive sample collection utility featuring real-time DTW validation gates to prevent misclassifications and capture redundancy.
* **`dtw_eval.py`**: Cross-validation script providing leave-one-out (LOO) accuracy metrics, live recognition, and active training loops (`--active`).
* **`airtalk_server.py`**: WebSocket server and live typist engine supporting dynamic, remappable air gestures (flicks, twists, thrusts).
* **`prune.py`**: Template dataset hygiene tool that cleans boundary-polluting exemplars and enforces per-class capacity caps.