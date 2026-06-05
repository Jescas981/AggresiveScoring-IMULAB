/**
 * Teensy 4.1 — MPU6050 IMU + NEO-6M GPS Session Logger (I2Cdevlib version)
 *
 * Packet layout (36 bytes, packed):
 *   [timestamp_ms(4) | session_id(4) | type(1) | _pad(3) | data[6](24)]
 *
 *   TYPE_IMU → data = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
 *   TYPE_GPS → data = [lat, lon, speed_mps, course_deg, altitude_m, utc_seconds]
 */

#define I2CDEV_IMPLEMENTATION I2CDEV_ARDUINO_WIRE

#include <Arduino.h>
#include <Wire.h>
#include <SD.h>
#include <QNEthernet.h>
#include <EEPROM.h>
#include <I2Cdev.h>
#include <MPU6050.h>
#include <PubSubClient.h>
#include <TinyGPSPlus.h>


using namespace qindesign::network;

// ── Config ────────────────────────────────────────────────────
static constexpr int      EEPROM_ADDR        = 0;
static constexpr int      EEPROM_MAGIC_ADDR  = 4;
static constexpr uint32_t SESSION_MAGIC      = 0xA5B6C7D8UL;
static constexpr uint32_t LOG_INTERVAL_MS    = 10;
static constexpr uint32_t SD_FLUSH_INTERVAL  = 1000;
static constexpr uint32_t SERIAL_PRINT_EVERY = 50;
static constexpr size_t   SD_BATCH_SIZE      = 10;

// GPS
static constexpr int GPS_BAUD = 9600;   // Serial2: RX pin 7, TX pin 8

// Network
static const IPAddress STATIC_IP(192, 168, 100, 30);
static const IPAddress SUBNET   (255, 255, 255,  0);
static const IPAddress GATEWAY  (192, 168, 100,  1);
static const IPAddress DNS      (  8,   8,   8,  8);

// MQTT
static const char* MQTT_BROKER = "192.168.100.6";
static const int   MQTT_PORT   = 1883;
static const char* MQTT_CLIENT = "teensy41";
static const char* MQTT_TOPIC  = "imu/raw";

// ── Packet types ──────────────────────────────────────────────
static constexpr uint8_t TYPE_IMU = 0x01;
static constexpr uint8_t TYPE_GPS = 0x02;

// ── Unified log packet (36 bytes) ─────────────────────────────
#pragma pack(push, 1)
struct LogPacket {
    uint32_t timestamp_ms;   //  4
    uint32_t session_id;     //  4
    uint8_t  type;           //  1
    uint8_t  _pad[3];        //  3
    float    data[6];        // 24
};                           // = 36 bytes
#pragma pack(pop)

static_assert(sizeof(LogPacket) == 36, "LogPacket must be 36 bytes");

// ── Globals ───────────────────────────────────────────────────
MPU6050        mpu;
TinyGPSPlus    gps;
File           sessionFile;
uint32_t       sessionID     = 0;
uint32_t       lastLogTime   = 0;
uint32_t       lastFlushTime = 0;
uint32_t       packetCount   = 0;
EthernetClient ethClient;
PubSubClient   mqtt(ethClient);

LogPacket      sdBatch[SD_BATCH_SIZE];
size_t         sdBatchCount = 0;

// ── EEPROM ────────────────────────────────────────────────────
uint32_t loadAndIncrementSession() {
    uint32_t magic, id;
    EEPROM.get(EEPROM_MAGIC_ADDR, magic);

    if (magic != SESSION_MAGIC) {
        id = 1;
        EEPROM.put(EEPROM_MAGIC_ADDR, SESSION_MAGIC);
        Serial.println("[EEPROM] First boot - session 1.");
    } else {
        EEPROM.get(EEPROM_ADDR, id);
        id += 1;
        Serial.printf("[EEPROM] Session %lu -> %lu\n", id - 1, id);
    }

    EEPROM.put(EEPROM_ADDR, id);
    return id;
}

// ── MPU6050 ───────────────────────────────────────────────────
bool initMPU() {
    Wire.begin();
    Wire.setClock(400000);

    Serial.println("[MPU6050] Initializing...");
    mpu.initialize();

    if (!mpu.testConnection()) {
        Serial.println("[MPU6050] ERROR: not found.");
        return false;
    }

    mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_4);
    mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_500);
    mpu.setDLPFMode(MPU6050_DLPF_BW_42);
    mpu.setRate(4);
    mpu.setSleepEnabled(false);

    Serial.println("[MPU6050] OK — I2C @ 400 kHz");
    return true;
}

LogPacket readIMU() {
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    constexpr float ACCEL_SCALE = 9.81f / 8192.0f;
    constexpr float GYRO_SCALE  = DEG_TO_RAD / 65.5f;

    LogPacket pkt;
    pkt.timestamp_ms = millis();
    pkt.session_id   = sessionID;
    pkt.type         = TYPE_IMU;
    pkt._pad[0] = pkt._pad[1] = pkt._pad[2] = 0;
    pkt.data[0] = ax * ACCEL_SCALE;   // accel_x  (m/s²)
    pkt.data[1] = ay * ACCEL_SCALE;   // accel_y
    pkt.data[2] = az * ACCEL_SCALE;   // accel_z
    pkt.data[3] = gx * GYRO_SCALE;    // gyro_x   (rad/s)
    pkt.data[4] = gy * GYRO_SCALE;    // gyro_y
    pkt.data[5] = gz * GYRO_SCALE;    // gyro_z
    return pkt;
}

// ── GPS ───────────────────────────────────────────────────────
bool pollGPS(LogPacket& out) {
    bool newFix = false;

    while (Serial2.available()) {
        if (gps.encode(Serial2.read()) &&
            gps.location.isUpdated()   &&
            gps.location.isValid()) {

            float utcSeconds = 0.0f;
            if (gps.time.isValid())
                utcSeconds = gps.time.hour()        * 3600.0f
                           + gps.time.minute()      *   60.0f
                           + gps.time.second()
                           + gps.time.centisecond() *    0.01f;

            out.timestamp_ms = millis();
            out.session_id   = sessionID;
            out.type         = TYPE_GPS;
            out._pad[0] = out._pad[1] = out._pad[2] = 0;
            out.data[0] = (float)gps.location.lat();
            out.data[1] = (float)gps.location.lng();
            out.data[2] = gps.speed.isValid()    ? (float)gps.speed.mps()       : 0.0f;
            out.data[3] = gps.course.isValid()   ? (float)gps.course.deg()      : 0.0f;
            out.data[4] = gps.altitude.isValid() ? (float)gps.altitude.meters() : 0.0f;
            out.data[5] = utcSeconds;
            newFix = true;
        }
    }

    return newFix;
}

// ── SD ────────────────────────────────────────────────────────
bool createSessionFile(uint32_t id) {
    char path[32];
    snprintf(path, sizeof(path), "/session_%05lu.bin", id);

    sessionFile = SD.open(path, FILE_WRITE);
    if (!sessionFile) {
        Serial.printf("[SD] ERROR: Could not create %s\n", path);
        return false;
    }

    Serial.printf("[SD] %s (%u bytes/packet, batch=%u)\n",
                  path, sizeof(LogPacket), SD_BATCH_SIZE);
    return true;
}

void flushSDBuffer(bool forceFsync = false) {
    if (sdBatchCount == 0) return;

    sessionFile.write(
        reinterpret_cast<const uint8_t*>(sdBatch),
        sdBatchCount * sizeof(LogPacket)
    );
    sdBatchCount = 0;

    if (forceFsync) sessionFile.flush();
}

// ── MQTT ──────────────────────────────────────────────────────
void maintainMQTT() {
    if (!mqtt.connected()) {
        Serial.print("[MQTT] Reconnecting...");
        if (mqtt.connect(MQTT_CLIENT)) {
            Serial.println(" OK.");
        } else {
            Serial.printf(" failed (rc=%d)\n", mqtt.state());
        }
    }
    mqtt.loop();
}

// ── Unified log ───────────────────────────────────────────────
void logPacket(const LogPacket& pkt) {
    sdBatch[sdBatchCount++] = pkt;
    if (sdBatchCount >= SD_BATCH_SIZE)
        flushSDBuffer(false);

    if (mqtt.connected())
        mqtt.publish(MQTT_TOPIC,
                     reinterpret_cast<const uint8_t*>(&pkt),
                     sizeof(pkt));

    if (++packetCount % SERIAL_PRINT_EVERY == 0) {
        if (pkt.type == TYPE_IMU) {
            Serial.printf("[IMU][%8lu] a=[%6.2f,%6.2f,%6.2f] g=[%6.3f,%6.3f,%6.3f]\n",
                          pkt.timestamp_ms,
                          pkt.data[0], pkt.data[1], pkt.data[2],
                          pkt.data[3], pkt.data[4], pkt.data[5]);
        } else {
            Serial.printf("[GPS][%8lu] lat=%9.5f lon=%10.5f spd=%5.2f crs=%6.1f alt=%6.1f utc=%9.2f\n",
                          pkt.timestamp_ms,
                          pkt.data[0], pkt.data[1], pkt.data[2],
                          pkt.data[3], pkt.data[4], pkt.data[5]);
        }
    }
}

// ── Setup ─────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(200);
    Serial.println("=== Teensy 4.1 IMU + GPS Logger ===");

    Serial2.begin(GPS_BAUD);
    Serial.println("[GPS] NEO-6M on Serial2 RX=7 TX=8 @ 9600 baud");

    if (!initMPU()) while (true) delay(500);

    bool sdOk = false;
    for (int i = 0; i < 5; i++) {
        delay(100);
        if (SD.begin(BUILTIN_SDCARD)) { sdOk = true; break; }
    }
    Serial.println(sdOk ? "[SD] OK" : "[SD] FAILED");
    if (!sdOk) while (true) delay(500);

    sessionID = loadAndIncrementSession();
    if (!createSessionFile(sessionID)) while (true) delay(500);

    Ethernet.begin();
    Ethernet.setDHCPEnabled(false);
    Ethernet.setLocalIP(STATIC_IP);
    Ethernet.setSubnetMask(SUBNET);
    Ethernet.setGatewayIP(GATEWAY);
    Ethernet.setDNSServerIP(DNS);

    uint32_t t0 = millis();
    while (Ethernet.linkStatus() != LinkON && millis() - t0 < 5000)
        delay(100);

    Serial.printf("[ETH] Link: %s\n",
                  Ethernet.linkStatus() == LinkON ? "ON" : "OFF");

    mqtt.setServer(MQTT_BROKER, MQTT_PORT);
    mqtt.setBufferSize(sizeof(LogPacket) + 64);
    maintainMQTT();

    lastLogTime   = millis();
    lastFlushTime = millis();
    Serial.println("[Setup] Done\n");
}

// ── Loop ──────────────────────────────────────────────────────
void loop() {
    uint32_t now = millis();

    // IMU — 100 Hz
    if (now - lastLogTime >= LOG_INTERVAL_MS) {
        lastLogTime += LOG_INTERVAL_MS;
        logPacket(readIMU());
    }

    // GPS — logs on each new valid fix (~1 Hz)
    LogPacket gpsPkt;
    if (pollGPS(gpsPkt))
        logPacket(gpsPkt);

    // SD flush
    if (now - lastFlushTime >= SD_FLUSH_INTERVAL) {
        lastFlushTime = now;
        flushSDBuffer(true);
    }

    maintainMQTT();
}