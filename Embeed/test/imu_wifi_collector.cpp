/**
 * ESP32 — MPU6050 (I2Cdevlib) + NEO-6M GPS Session Logger (Wi-Fi + MQTT)
 *
 * Packet layout (36 bytes, packed):
 *   [timestamp_ms(4) | session_id(4) | type(1) | _pad(3) | data[6](24)]
 *
 *   TYPE_IMU → data = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
 *   TYPE_GPS → data = [lat, lon, speed_mps, course_deg, altitude_m, utc_seconds]
 */

#include <Arduino.h>
#include <Wire.h>
#include <SD.h>
#include <EEPROM.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <TinyGPSPlus.h>

#include <I2Cdev.h>
#include <MPU6050.h>

// ── Config ────────────────────────────────────────────────
static constexpr int      EEPROM_ADDR        = 0;
static constexpr int      EEPROM_MAGIC_ADDR  = 4;
static constexpr uint32_t SESSION_MAGIC      = 0xA5B6C7D8UL;
static constexpr uint32_t LOG_INTERVAL_MS    = 10;
static constexpr uint32_t SD_FLUSH_INTERVAL  = 1000;
static constexpr uint32_t SERIAL_PRINT_EVERY = 50;
static constexpr size_t   SD_BATCH_SIZE      = 10;

// GPS
static constexpr int GPS_RX_PIN = 16;
static constexpr int GPS_TX_PIN = 17;
static constexpr int GPS_BAUD   = 9600;

// Wi-Fi
const char* WIFI_SSID = "A56 de Jesus";
const char* WIFI_PASS = "12345678";

// MQTT
const char* MQTT_BROKER = "10.212.9.167";
const int   MQTT_PORT   = 1883;
const char* MQTT_CLIENT = "esp32";
const char* MQTT_TOPIC  = "imu/raw";

// ── Packet types ──────────────────────────────────────────
static constexpr uint8_t TYPE_IMU = 0x01;
static constexpr uint8_t TYPE_GPS = 0x02;

// ── Log packet ─────────────────────────────────────────────
#pragma pack(push, 1)
struct LogPacket {
    uint32_t timestamp_ms;
    uint32_t session_id;
    uint8_t  type;
    uint8_t  _pad[3];
    float    data[6];
};
#pragma pack(pop)

static_assert(sizeof(LogPacket) == 36, "LogPacket must be 36 bytes");

// ── Globals ───────────────────────────────────────────────
MPU6050 mpu;

TinyGPSPlus gps;
File sessionFile;

uint32_t sessionID = 0;
uint32_t lastLogTime = 0;
uint32_t lastFlushTime = 0;
uint32_t packetCount = 0;

WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

LogPacket sdBatch[SD_BATCH_SIZE];
size_t sdBatchCount = 0;

// ── EEPROM ────────────────────────────────────────────────
uint32_t loadAndIncrementSession() {
    uint32_t magic, id;

    EEPROM.begin(16);
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
    EEPROM.commit();

    return id;
}

// ── MPU6050 (I2Cdevlib) ───────────────────────────────────
bool initMPU() {
    Wire.begin(21, 22);
    Wire.setClock(400000);

    Serial.println("[MPU6050] Initializing...");

    mpu.initialize();

    if (!mpu.testConnection()) {
        Serial.println("[MPU6050] ERROR: not found.");
        return false;
    }

    mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_8);
    mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_500);
    mpu.setDLPFMode(MPU6050_DLPF_BW_42);
    mpu.setSleepEnabled(false);

    Serial.println("[MPU6050] OK — I2Cdevlib @ 400 kHz");
    return true;
}

// ── IMU READ ──────────────────────────────────────────────
LogPacket readIMU() {
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    constexpr float ACCEL_SCALE = 9.81f / 4096.0f;
    constexpr float GYRO_SCALE  = (3.1415926535f / 180.0f) / 65.5f;

    LogPacket pkt;
    pkt.timestamp_ms = millis();
    pkt.session_id   = sessionID;
    pkt.type         = TYPE_IMU;

    pkt._pad[0] = pkt._pad[1] = pkt._pad[2] = 0;

    pkt.data[0] = ax * ACCEL_SCALE;
    pkt.data[1] = ay * ACCEL_SCALE;
    pkt.data[2] = az * ACCEL_SCALE;

    pkt.data[3] = gx * GYRO_SCALE;
    pkt.data[4] = gy * GYRO_SCALE;
    pkt.data[5] = gz * GYRO_SCALE;

    return pkt;
}

// ── GPS ───────────────────────────────────────────────────
bool pollGPS(LogPacket& out) {
    bool newFix = false;

    while (Serial2.available()) {
        if (gps.encode(Serial2.read()) &&
            gps.location.isUpdated() &&
            gps.location.isValid()) {

            float utcSeconds = 0.0f;
            if (gps.time.isValid()) {
                utcSeconds =
                    gps.time.hour() * 3600.0f +
                    gps.time.minute() * 60.0f +
                    gps.time.second() +
                    gps.time.centisecond() * 0.01f;
            }

            out.timestamp_ms = millis();
            out.session_id   = sessionID;
            out.type         = TYPE_GPS;

            out._pad[0] = out._pad[1] = out._pad[2] = 0;

            out.data[0] = gps.location.lat();
            out.data[1] = gps.location.lng();
            out.data[2] = gps.speed.isValid() ? gps.speed.mps() : 0.0f;
            out.data[3] = gps.course.isValid() ? gps.course.deg() : 0.0f;
            out.data[4] = gps.altitude.isValid() ? gps.altitude.meters() : 0.0f;
            out.data[5] = utcSeconds;

            newFix = true;
        }
    }

    return newFix;
}

// ── (resto igual que tu código original) ──────────────────
bool createSessionFile(uint32_t id) {
    char path[32];
    snprintf(path, sizeof(path), "/session_%05lu.bin", id);

    sessionFile = SD.open(path, FILE_WRITE);
    if (!sessionFile) return false;

    Serial.printf("[SD] %s\n", path);
    return true;
}

void flushSDBuffer(bool force = false) {
    if (!sdBatchCount) return;

    sessionFile.write((uint8_t*)sdBatch, sdBatchCount * sizeof(LogPacket));
    sdBatchCount = 0;

    if (force) sessionFile.flush();
}

void maintainMQTT() {
    if (!mqtt.connected()) {
        mqtt.connect(MQTT_CLIENT);
    }
    mqtt.loop();
}

void logPacket(const LogPacket& pkt) {
    sdBatch[sdBatchCount++] = pkt;

    if (sdBatchCount >= SD_BATCH_SIZE)
        flushSDBuffer(false);

    if (mqtt.connected())
        mqtt.publish(MQTT_TOPIC, (uint8_t*)&pkt, sizeof(pkt));

    if (++packetCount % SERIAL_PRINT_EVERY == 0) {
        Serial.printf("[%s] t=%lu\n",
            pkt.type == TYPE_IMU ? "IMU" : "GPS",
            pkt.timestamp_ms);
    }
}

// ── SETUP ────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(500);

    Serial2.begin(GPS_BAUD, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);

    if (!initMPU()) while (true) delay(500);

    if (!SD.begin()) while (true) delay(500);

    sessionID = loadAndIncrementSession();
    if (!createSessionFile(sessionID)) while (true) delay(500);

    WiFi.begin(WIFI_SSID, WIFI_PASS);
    while (WiFi.status() != WL_CONNECTED) delay(300);

    mqtt.setServer(MQTT_BROKER, MQTT_PORT);
    mqtt.setBufferSize(sizeof(LogPacket) + 64);

    lastLogTime = millis();
    lastFlushTime = millis();

    Serial.println("[Setup] Done");
}

// ── LOOP ────────────────────────────────────────────────
void loop() {
    uint32_t now = millis();

    if (now - lastLogTime >= LOG_INTERVAL_MS) {
        lastLogTime += LOG_INTERVAL_MS;
        logPacket(readIMU());
    }

    LogPacket gpsPkt;
    if (pollGPS(gpsPkt))
        logPacket(gpsPkt);

    if (now - lastFlushTime >= SD_FLUSH_INTERVAL) {
        lastFlushTime = now;
        flushSDBuffer(true);
    }

    maintainMQTT();
}