// ===================================================================================
// TRANSPORT SWITCH
//   USE_BLE 0 : the original behaviour, unchanged: WiFi SoftAP "AirStylus_AP" + UDP broadcast.
//   USE_BLE 1 : boots on Bluetooth LE instead (WiFi is never started). The same 41-byte
//               TelemetryPacket is sent as a GATT notification once per sample.
// Everything BLE-related lives inside `#if USE_BLE` blocks, so with 0 the compiler never even
// sees it. Flip it, re-upload, done.
// ===================================================================================
#define USE_BLE 1

#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>

#if USE_BLE
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#endif

#define MPU_ADDR         0x68
#define TOUCH_PIN        27
#define LED_PIN          2

// Sensor ranges. Keep each pair in sync: the register value written in setup() and the divisor.
//   Gyro : 0x00=±250  0x08=±500  0x10=±1000  0x18=±2000 dps  -> 131 / 65.5 / 32.8 / 16.4 LSB per dps
//   Accel: 0x00=±2g   0x08=±4g   0x10=±8g     0x18=±16g      -> 16384 / 8192 / 4096 / 2048 LSB per g
#define GYRO_FS_REG       0x10
#define GYRO_LSB_PER_DPS  32.8f
#define ACCEL_FS_REG      0x08
#define ACCEL_LSB_PER_G   8192.0f

// ---------------- Magnetometer (same I2C bus as the MPU6050) ----------------
// Auto-detected at boot:  QMC5883L @ 0x0D, HMC5883L @ 0x1E, or QMC5883P @ 0x2C.
// Output units: Gauss, RAW (no hard/soft-iron calibration here; do that on the host).
#define QMC_ADDR          0x0D
#define HMC_ADDR          0x1E
#define QMCP_ADDR         0x2C   // QMC5883P (newer GY-271-style boards; different registers from QMC5883L)
#define QMC_LSB_PER_GAUSS 3000.0f   // ±8G range
#define HMC_LSB_PER_GAUSS 1090.0f   // ±1.3Ga range
#define QMCP_LSB_PER_GAUSS 3750.0f  // ±8G range

// Axis remap: mag sensor axes -> pen body frame (same axes as the MPU6050 accel/gyro).
// body[i] = MAG_SIGN[i] * sensor[MAG_AXIS[i]]. Sensor order is {x, y, z}.
// Defaults assume the mag chip is mounted with axes parallel to the MPU6050. If not, fix here
// (see the check at the bottom of this file's notes: tilt test).
const uint8_t MAG_AXIS[3] = {0, 1, 2};
const float   MAG_SIGN[3] = {1.0f, 1.0f, 1.0f};

enum MagType { MAG_NONE, MAG_QMC5883L, MAG_HMC5883L, MAG_QMC5883P };
MagType magType = MAG_NONE;
float magLast[3] = {0, 0, 0};   // held if a read fails, so the stream never glitches to 0

// 41-byte packed binary packet: original 29 bytes + 3 floats (mx my mz) appended at the end.
struct __attribute__((packed)) TelemetryPacket {
  uint32_t timestampUs;
  uint8_t touched;
  float ax, ay, az;
  float gx, gy, gz;
  float mx, my, mz;
};

QueueHandle_t telemetryQueue;
WiFiUDP udp;
const IPAddress broadcastIP(192, 168, 4, 255); // UDP broadcast over SoftAP
const uint16_t udpPort = 5005;

float gBias[3] = {0};  // gyro bias only. Accel is sent raw (see imuTask)
bool touchState = false;

void writeRegAt(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

void writeReg(uint8_t reg, uint8_t val) { writeRegAt(MPU_ADDR, reg, val); }

uint8_t readReg(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0xFF;
}

// Two separate statements: C++ doesn't define the order of the two Wire.read() calls
// in `(Wire.read() << 8) | Wire.read()`, so a compiler could swap the bytes.
static inline int16_t read16() {
  int16_t hi = Wire.read();
  int16_t lo = Wire.read();
  return (int16_t)((hi << 8) | (lo & 0xFF));
}

static inline int16_t read16LE() {
  int16_t lo = Wire.read();
  int16_t hi = Wire.read();
  return (int16_t)((hi << 8) | (lo & 0xFF));
}

void readRawIMU(int16_t raw[6]) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14);
  
  for (int i = 0; i < 3; i++) raw[i] = read16();
  Wire.read(); Wire.read(); // Skip temp bytes
  for (int i = 3; i < 6; i++) raw[i] = read16();
}

// ---------------- Magnetometer ----------------
bool i2cPresent(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

uint8_t readRegAt(uint8_t addr, uint8_t reg) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0xFF;
  Wire.requestFrom(addr, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0xFF;
}

// Prints every address that ACKs on the bus. Use it to confirm the mag is physically visible.
void i2cScan() {
  Serial.print("[I2C] Scan:");
  int n = 0;
  for (uint8_t a = 1; a < 127; a++) {
    if (i2cPresent(a)) { Serial.printf(" 0x%02X", a); n++; }
  }
  Serial.println(n ? "" : " (nothing found)");
}

void initMag() {
  if (i2cPresent(QMC_ADDR)) {
    writeRegAt(QMC_ADDR, 0x0B, 0x01);  // set/reset period (recommended by datasheet)
    // 0x1D = OSR 512 | ±8G | 200 Hz | continuous
    writeRegAt(QMC_ADDR, 0x09, 0x1D);
    magType = MAG_QMC5883L;
  } else if (i2cPresent(HMC_ADDR)) {
    writeRegAt(HMC_ADDR, 0x00, 0x18);  // 1 sample avg, 75 Hz, normal measurement
    writeRegAt(HMC_ADDR, 0x01, 0x20);  // gain ±1.3Ga (1090 LSB/Ga)
    writeRegAt(HMC_ADDR, 0x02, 0x00);  // continuous-measurement mode
    magType = MAG_HMC5883L;
    // ID registers 0x0A..0x0C should read 'H','4','3' (0x48 0x34 0x33) on HMC5883L
    Serial.printf("[MAG] HMC ID regs: 0x%02X 0x%02X 0x%02X\n",
                  readRegAt(HMC_ADDR, 0x0A), readRegAt(HMC_ADDR, 0x0B), readRegAt(HMC_ADDR, 0x0C));
  } else if (i2cPresent(QMCP_ADDR)) {
    writeRegAt(QMCP_ADDR, 0x0B, 0x08);  // CTRL2: ±8G range, set/reset on
    writeRegAt(QMCP_ADDR, 0x0A, 0xCF);  // CTRL1: continuous mode, 200 Hz, OSR 8, downsample 8
    magType = MAG_QMC5883P;
    // Chip ID register 0x00 should read 0x80 on QMC5883P
    Serial.printf("[MAG] QMC5883P chip ID reg: 0x%02X (expect 0x80)\n", readRegAt(QMCP_ADDR, 0x00));
  } else {
    magType = MAG_NONE;
  }
}

// Reads the latest mag sample into out[3] (body frame, Gauss). Returns false on failure,
// in which case the caller should keep the previous value.
bool readMag(float out[3]) {
  float s[3];

  if (magType == MAG_QMC5883L) {
    Wire.beginTransmission(QMC_ADDR);
    Wire.write(0x00);
    if (Wire.endTransmission(false) != 0) return false;
    if (Wire.requestFrom((uint8_t)QMC_ADDR, (uint8_t)6) != 6) return false;
    // Order: X, Y, Z, little-endian
    s[0] = read16LE() / QMC_LSB_PER_GAUSS;
    s[1] = read16LE() / QMC_LSB_PER_GAUSS;
    s[2] = read16LE() / QMC_LSB_PER_GAUSS;
  } else if (magType == MAG_HMC5883L) {
    Wire.beginTransmission(HMC_ADDR);
    Wire.write(0x03);
    if (Wire.endTransmission(false) != 0) return false;
    if (Wire.requestFrom((uint8_t)HMC_ADDR, (uint8_t)6) != 6) return false;
    // Order: X, Z, Y (yes, really), big-endian
    float x = read16() / HMC_LSB_PER_GAUSS;
    float z = read16() / HMC_LSB_PER_GAUSS;
    float y = read16() / HMC_LSB_PER_GAUSS;
    s[0] = x; s[1] = y; s[2] = z;
  } else if (magType == MAG_QMC5883P) {
    Wire.beginTransmission(QMCP_ADDR);
    Wire.write(0x01);                    // data starts at 0x01 (0x00 is the chip ID)
    if (Wire.endTransmission(false) != 0) return false;
    if (Wire.requestFrom((uint8_t)QMCP_ADDR, (uint8_t)6) != 6) return false;
    // Order: X, Y, Z, little-endian
    s[0] = read16LE() / QMCP_LSB_PER_GAUSS;
    s[1] = read16LE() / QMCP_LSB_PER_GAUSS;
    s[2] = read16LE() / QMCP_LSB_PER_GAUSS;
  } else {
    return false;
  }

  for (int i = 0; i < 3; i++) out[i] = MAG_SIGN[i] * s[MAG_AXIS[i]];
  return true;
}

void calibrate() {
  digitalWrite(LED_PIN, HIGH);
  Serial.println("[IMU] Calibrating biases...");
  delay(1000);

  double sG[3] = {0};
  uint32_t count = 0;
  uint32_t start = millis();

  while (millis() - start < 2000 || count < 800) {
    int16_t r[6];
    readRawIMU(r);
    for (int i = 0; i < 3; i++) sG[i] += r[i + 3] / GYRO_LSB_PER_DPS;
    count++;
    delay(2);
  }

  for (int i = 0; i < 3; i++) gBias[i] = sG[i] / count;

  digitalWrite(LED_PIN, LOW);
  Serial.printf("[IMU] Calibrated. gBias:[%.3f,%.3f,%.3f] dps\n", gBias[0], gBias[1], gBias[2]);
}

bool readTouch() {
  static bool cand = false;
  static uint8_t cnt = 0;
  bool raw = (digitalRead(TOUCH_PIN) == HIGH);
  if (raw == cand) {
    if (cnt < 2) cnt++;
  } else {
    cand = raw;
    cnt = 1;
  }
  if (cnt >= 2) touchState = cand;
  return touchState;
}

// -------------------------------------------------------------------
// CORE 1 TASK: High-Speed IMU & Touch Sampling (100 Hz)
// -------------------------------------------------------------------
void imuTask(void *pvParameters) {
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xFrequency = pdMS_TO_TICKS(10); // 10ms = 100Hz

  for (;;) {
    vTaskDelayUntil(&xLastWakeTime, xFrequency);

    int16_t r[6];
    readRawIMU(r);

    static uint32_t magFail = 0;
    float m[3];
    if (readMag(m)) {
      magLast[0] = m[0]; magLast[1] = m[1]; magLast[2] = m[2];
    } else if (magType != MAG_NONE) {
      if (magFail++ % 500 == 0) Serial.printf("[MAG] read failed (%lu so far)\n", (unsigned long)magFail);
    }

    TelemetryPacket pkt;
    pkt.timestampUs = micros();
    pkt.touched = readTouch() ? 1 : 0;

    // Accel in g, RAW (no bias/gravity removal: the host needs the true gravity direction).
    // Gyro in deg/s minus the boot-time bias.
    pkt.ax = r[0] / ACCEL_LSB_PER_G;
    pkt.ay = r[1] / ACCEL_LSB_PER_G;
    pkt.az = r[2] / ACCEL_LSB_PER_G;
    
    pkt.gx = (r[3] / GYRO_LSB_PER_DPS) - gBias[0];
    pkt.gy = (r[4] / GYRO_LSB_PER_DPS) - gBias[1];
    pkt.gz = (r[5] / GYRO_LSB_PER_DPS) - gBias[2];

    // Magnetometer in Gauss, raw, appended at the end of the packet.
    pkt.mx = magLast[0];
    pkt.my = magLast[1];
    pkt.mz = magLast[2];

    xQueueSend(telemetryQueue, &pkt, 0);
  }
}

// -------------------------------------------------------------------
// CORE 0 TASK: SoftAP Network Operations & UDP Binary Streaming
// -------------------------------------------------------------------
void netTask(void *pvParameters) {
  TelemetryPacket pkt;

  for (;;) {
    if (xQueueReceive(telemetryQueue, &pkt, pdMS_TO_TICKS(10)) == pdTRUE) {
      udp.beginPacket(broadcastIP, udpPort);
      udp.write((const uint8_t*)&pkt, sizeof(TelemetryPacket));
      udp.endPacket();
    }
  }
}

#if USE_BLE
// -------------------------------------------------------------------
// BLE TRANSPORT (only compiled when USE_BLE is 1)
//   Device name : AirStylus
//   Service     : BLE_SERVICE_UUID
//   Telemetry   : BLE_TELEMETRY_UUID (notify). Every notification is exactly one 41-byte
//                 TelemetryPacket, so the host's decoder is identical to the UDP one. It needs an
//                 ATT MTU of at least 44 (41 + 3 header); the host should check what was negotiated.
// -------------------------------------------------------------------
#define BLE_DEVICE_NAME     "AirStylus"
#define BLE_SERVICE_UUID    "4f3c5a10-7b2e-4d6a-9a51-3c8e0b1a0001"
#define BLE_TELEMETRY_UUID  "4f3c5a11-7b2e-4d6a-9a51-3c8e0b1a0001"

// Connection interval the pen asks the host for, in 1.25 ms units (6 = 7.5 ms ... 12 = 15 ms).
// 100 Hz = one sample per 10 ms, so a short interval keeps per-sample latency low.
#define BLE_CONN_MIN_INTERVAL  6
#define BLE_CONN_MAX_INTERVAL  12
#define BLE_CONN_LATENCY       0     // never skip connection events
#define BLE_CONN_TIMEOUT       400   // supervision timeout, 10 ms units = 4 s

BLEServer*         bleServer    = nullptr;
BLECharacteristic* bleTelemetry = nullptr;
volatile bool      bleConnected = false;
volatile bool      bleInitDone  = false;

class AirServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* s, esp_ble_gatts_cb_param_t* param) override {
    bleConnected = true;
    // Ask for the fast connection parameters (the host may still choose its own).
    s->updateConnParams(param->connect.remote_bda, BLE_CONN_MIN_INTERVAL, BLE_CONN_MAX_INTERVAL,
                        BLE_CONN_LATENCY, BLE_CONN_TIMEOUT);
    Serial.println("[BLE] Host connected");
  }
  void onDisconnect(BLEServer* s) override {
    bleConnected = false;
    Serial.println("[BLE] Host disconnected, advertising again");
    BLEDevice::startAdvertising();
  }
};

void bleBegin() {
  BLEDevice::init(BLE_DEVICE_NAME);
  BLEDevice::setMTU(517);   // ask for the biggest MTU; the host decides what it accepts

  bleServer = BLEDevice::createServer();
  bleServer->setCallbacks(new AirServerCallbacks());

  BLEService* svc = bleServer->createService(BLE_SERVICE_UUID);
  bleTelemetry = svc->createCharacteristic(BLE_TELEMETRY_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  bleTelemetry->addDescriptor(new BLE2902());   // lets the host switch notifications on
  svc->start();

  BLEAdvertising* adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(BLE_SERVICE_UUID);
  adv->setScanResponse(true);                   // the device name goes in the scan response
  adv->setMinPreferred(BLE_CONN_MIN_INTERVAL);
  adv->setMaxPreferred(BLE_CONN_MAX_INTERVAL);
  BLEDevice::startAdvertising();

  Serial.println("[BLE] Advertising as \"" BLE_DEVICE_NAME "\"");
}

// BLE is started from a short-lived task pinned to Core 0 (not from setup(), which runs on Core 1):
// the Bluetooth controller's interrupts end up on the core that enabled it, and we want them on
// the radio core, away from the 100 Hz I2C sampling on Core 1.
void bleInitTask(void *pvParameters) {
  bleBegin();
  bleInitDone = true;
  vTaskDelete(NULL);
}

// CORE 0 TASK (BLE version of netTask): pull samples off the queue and notify them, one per sample.
// Not connected -> samples are simply discarded, so the queue never backs up.
void netTaskBLE(void *pvParameters) {
  TelemetryPacket pkt;

  for (;;) {
    if (xQueueReceive(telemetryQueue, &pkt, pdMS_TO_TICKS(10)) == pdTRUE) {
      if (bleConnected && bleTelemetry != nullptr) {
        bleTelemetry->setValue((uint8_t*)&pkt, sizeof(TelemetryPacket));
        bleTelemetry->notify();
      }
    }
  }
}
#endif  // USE_BLE

#if USE_BLE
// -------------------------------------------------------------------
// BLE WATCHDOG (BLE build only; set BLE_WATCHDOG to 0 to compile it out)
//   A host that dies hard (kill -9, crash, power loss) can leave its Bluetooth link open with nobody
//   attached. A connected pen stops advertising, so the next host run can't find it and only a reset
//   helped. When the host process goes away BlueZ switches the notification subscription off, so
//   "connected but not subscribed for a while" means the link is orphaned: drop it, and the existing
//   onDisconnect callback restarts advertising.
//   It never acts unless it can positively see "connected AND not subscribed" for the whole grace
//   period, and a host that is merely slow to subscribe has plenty of time (the normal connect ->
//   subscribe takes a second or two).
// -------------------------------------------------------------------
#define BLE_WATCHDOG            1
#define BLE_SUBSCRIBE_GRACE_MS  10000

#if BLE_WATCHDOG
void bleWatchdogTask(void *pvParameters) {
  BLE2902* cccd = nullptr;     // the descriptor the host writes to switch notifications on
  bool     counting = false;
  uint32_t since = 0;

  for (;;) {
    vTaskDelay(pdMS_TO_TICKS(500));

    if (!bleConnected || bleTelemetry == nullptr || bleServer == nullptr) {
      counting = false;
      continue;
    }

    if (cccd == nullptr) {
      cccd = (BLE2902*)bleTelemetry->getDescriptorByUUID(BLEUUID((uint16_t)0x2902));
      if (cccd == nullptr) continue;          // can't tell -> never kick anyone
    }

    if (cccd->getNotifications()) {           // someone is listening: all good
      counting = false;
      continue;
    }

    uint32_t now = millis();
    if (!counting) {
      counting = true;
      since = now;
      continue;
    }
    if (now - since >= BLE_SUBSCRIBE_GRACE_MS) {
      Serial.println("[BLE] Connected but not subscribed for too long (orphaned link?), disconnecting");
      bleServer->disconnect(bleServer->getConnId());
      counting = false;
    }
  }
}
#endif  // BLE_WATCHDOG
#endif  // USE_BLE

void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(TOUCH_PIN, INPUT_PULLDOWN);

  Serial.begin(115200);
  Wire.begin(21, 22);
  Wire.setClock(400000); // 400 kHz Fast I2C

  writeReg(0x6B, 0x00); delay(50); // Wake MPU6050
  writeReg(0x6B, 0x01);            // Gyro PLL Clock
  writeReg(0x1A, 0x03);            // DLPF ~42Hz Hardware LPF
  writeReg(0x1B, GYRO_FS_REG);     // Gyro range (see defines at top)
  writeReg(0x1C, ACCEL_FS_REG);    // Accel range (see defines at top)

  uint8_t who = readReg(0x75);
  if (who != 0x68 && who != 0x70) {
    while (1) { digitalWrite(LED_PIN, !digitalRead(LED_PIN)); delay(100); }
  }

  // If the mag sits behind the MPU6050's auxiliary I2C (e.g. GY-87 boards), it is invisible
  // on the main bus until bypass is enabled. Harmless if the mag is on the external bus.
  writeReg(0x6A, 0x00);   // I2C master off
  writeReg(0x37, 0x02);   // BYPASS_EN: expose the aux bus on SDA/SCL
  delay(20);              // let the mag finish power-up (needs >= 6 ms)
  i2cScan();
  initMag();
  switch (magType) {
    case MAG_QMC5883L: Serial.println("[MAG] QMC5883L found @0x0D"); break;
    case MAG_HMC5883L: Serial.println("[MAG] HMC5883L found @0x1E"); break;
    case MAG_QMC5883P: Serial.println("[MAG] QMC5883P found @0x2C"); break;
    default:           Serial.println("[MAG] NOT FOUND - streaming mx,my,mz = 0"); break;
  }

  calibrate();

  telemetryQueue = xQueueCreate(20, sizeof(TelemetryPacket));

#if !USE_BLE
  // Configure ESP32 SoftAP Mode
  WiFi.mode(WIFI_AP);
  // Fixed channel 6, SSID visible, max 1 client. Keep the radio awake for steady latency.
  WiFi.softAP("AirStylus_AP", "stylus1234", 6, 0, 1);
  WiFi.setSleep(false);
  Serial.print("[SoftAP] Started AP. Gateway IP: ");
  Serial.println(WiFi.softAPIP());

  udp.begin(udpPort);
#else
  Serial.println("[Transport] BLE");
  xTaskCreatePinnedToCore(bleInitTask, "BLE_Init", 8192, NULL, 1, NULL, 0);
  while (!bleInitDone) delay(10);
#if BLE_WATCHDOG
  xTaskCreatePinnedToCore(bleWatchdogTask, "BLE_WD", 4096, NULL, 1, NULL, 0);
#endif
#endif

  // Pin sampling to Core 1 and network streaming to Core 0
  xTaskCreatePinnedToCore(imuTask, "IMU_Task", 4096, NULL, 2, NULL, 1);
#if !USE_BLE
  xTaskCreatePinnedToCore(netTask, "NET_Task", 4096, NULL, 1, NULL, 0);
#else
  xTaskCreatePinnedToCore(netTaskBLE, "NET_Task", 4096, NULL, 1, NULL, 0);
#endif
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000));
}