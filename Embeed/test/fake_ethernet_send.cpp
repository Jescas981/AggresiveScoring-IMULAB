/**
 * Teensy 4.1 — IMU Session Logger
 * ─────────────────────────────────
 * - Gyroscope + accelerometer struct stored as raw binary on SD
 * - Same struct published to MQTT broker as raw bytes
 * - Non-volatile session counter via EEPROM
 */

#include <Arduino.h>
#include <SD.h>
#include <QNEthernet.h>
#include <EEPROM.h>
#include <PubSubClient.h>

using namespace qindesign::network;

// ── Config ────────────────────────────────────────────────────
static constexpr int      EEPROM_ADDR       = 0;
static constexpr int      EEPROM_MAGIC_ADDR = 4;
static constexpr uint32_t SESSION_MAGIC     = 0xA5B6C7D8UL;
static constexpr uint32_t LOG_INTERVAL_MS   = 10;   // 100 Hz

// Network
static const IPAddress STATIC_IP  (192, 168, 100, 30);
static const IPAddress SUBNET     (255, 255, 255,  0);
static const IPAddress GATEWAY    (192, 168, 100,  1);
static const IPAddress DNS        (  8,   8,   8,  8);

// MQTT
static const char* MQTT_BROKER   = "192.168.100.6";  // <- your broker IP
static const int   MQTT_PORT     = 1883;
static const char* MQTT_CLIENT   = "teensy41";
static const char* MQTT_TOPIC    = "imu/raw";

// ── IMU struct ────────────────────────────────────────────────
// Packed: no padding bytes — exact same layout on wire and disk
#pragma pack(push, 1)
struct ImuPacket {
    uint32_t timestamp_ms;   // millis()
    uint32_t session_id;     // EEPROM session counter
    float    accel_x;        // m/s^2
    float    accel_y;
    float    accel_z;
    float    gyro_x;         // rad/s
    float    gyro_y;
    float    gyro_z;
};
#pragma pack(pop)

static_assert(sizeof(ImuPacket) == 32, "ImuPacket must be exactly 32 bytes");

// ── Globals ───────────────────────────────────────────────────
File           sessionFile;
uint32_t       sessionID   = 0;
uint32_t       lastLogTime = 0;
EthernetClient ethClient;
PubSubClient   mqtt(ethClient);

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

// ── SD ────────────────────────────────────────────────────────
bool createSessionFile(uint32_t id) {
    char path[32];
    snprintf(path, sizeof(path), "/session_%05lu.bin", id);
    sessionFile = SD.open(path, FILE_WRITE);
    if (!sessionFile) {
        Serial.printf("[SD] ERROR: Could not create %s\n", path);
        return false;
    }
    Serial.printf("[SD] Session file: %s  (packet=%u bytes)\n",
                  path, sizeof(ImuPacket));
    return true;
}

// ── MQTT ──────────────────────────────────────────────────────
void mqttConnect() {
    if (mqtt.connected()) return;
    Serial.print("[MQTT] Connecting...");
    int tries = 0;
    while (!mqtt.connected() && tries++ < 5) {
        if (mqtt.connect(MQTT_CLIENT)) {
            Serial.println(" OK.");
        } else {
            Serial.printf(" failed (rc=%d), retry...\n", mqtt.state());
            delay(500);
        }
    }
}

// ── Fake IMU — replace with real sensor driver ────────────────
ImuPacket readIMU() {
    float t = millis() / 1000.0f;
    ImuPacket pkt;
    pkt.timestamp_ms = millis();
    pkt.session_id   = sessionID;
    pkt.accel_x = sinf(t * 1.1f) * 9.81f;
    pkt.accel_y = cosf(t * 0.9f) * 9.81f;
    pkt.accel_z = 9.81f + sinf(t * 0.3f) * 0.5f;
    pkt.gyro_x  = sinf(t * 2.0f) * 0.5f;
    pkt.gyro_y  = cosf(t * 1.5f) * 0.5f;
    pkt.gyro_z  = sinf(t * 0.7f) * 0.3f;
    return pkt;
}

// ── Log: SD binary + MQTT raw bytes ──────────────────────────
void logPacket(const ImuPacket& pkt) {
    // 1. Raw binary to SD
    sessionFile.write(reinterpret_cast<const uint8_t*>(&pkt), sizeof(pkt));
    sessionFile.flush();

    // 2. Raw bytes to MQTT
    mqttConnect();
    if (mqtt.connected()) {
        mqtt.publish(MQTT_TOPIC,
                     reinterpret_cast<const uint8_t*>(&pkt),
                     sizeof(pkt));
    }

    // 3. Human-readable Serial debug
    Serial.printf("[%8lu] sess=%lu | a=[%6.2f,%6.2f,%6.2f] g=[%6.3f,%6.3f,%6.3f]\n",
                  pkt.timestamp_ms, pkt.session_id,
                  pkt.accel_x, pkt.accel_y, pkt.accel_z,
                  pkt.gyro_x,  pkt.gyro_y,  pkt.gyro_z);
}

// ── Setup ─────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(200);
    Serial.println("=== Teensy 4.1 IMU Logger ===");

    // SD
    bool sdOk = false;
    for (int i = 1; i <= 5; i++) {
        delay(100);
        if (SD.begin(BUILTIN_SDCARD)) { sdOk = true; break; }
    }
    Serial.println(sdOk ? "[SD] OK" : "[SD] FAILED");
    if (!sdOk) { while (true) delay(500); }

    // Session
    sessionID = loadAndIncrementSession();
    if (!createSessionFile(sessionID)) { while (true) delay(500); }

    // Ethernet static IP
    Serial.println("[ETH] Starting...");
    Ethernet.begin();
    Ethernet.setDHCPEnabled(false);
    Ethernet.setLocalIP(STATIC_IP);
    Ethernet.setSubnetMask(SUBNET);
    Ethernet.setGatewayIP(GATEWAY);
    Ethernet.setDNSServerIP(DNS);

    uint32_t t0 = millis();
    while (Ethernet.linkStatus() != LinkON && millis() - t0 < 5000) delay(100);
    Serial.printf("[ETH] Link: %s  IP: %s\n",
                  Ethernet.linkStatus() == LinkON ? "ON" : "OFF",
                  Ethernet.localIP());

    // MQTT
    mqtt.setServer(MQTT_BROKER, MQTT_PORT);
    mqtt.setBufferSize(sizeof(ImuPacket) + 64);
    mqttConnect();

    Serial.println("\n[Setup] Done - logging at 100 Hz\n");
}

// ── Loop ──────────────────────────────────────────────────────
void loop() {
    mqtt.loop();  // keep MQTT alive

    uint32_t now = millis();
    if (now - lastLogTime >= LOG_INTERVAL_MS) {
        lastLogTime = now;
        logPacket(readIMU());
    }
}