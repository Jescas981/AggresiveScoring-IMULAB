#pragma once

#include <IPAddress.h>

// ── EEPROM ────────────────────────────────────────────────────
static constexpr int      EEPROM_ADDR        = 0;
static constexpr int      EEPROM_MAGIC_ADDR  = 4;
static constexpr uint32_t SESSION_MAGIC      = 0xA5B6C7D8UL;

// ── Timing ────────────────────────────────────────────────────
// IMU logged at 100 Hz (raw read)
static constexpr uint32_t LOG_INTERVAL_MS    = 10;
// SD hard-flush to filesystem every second
static constexpr uint32_t SD_FLUSH_INTERVAL  = 1000;
// Print to Serial every N packets
static constexpr uint32_t SERIAL_PRINT_EVERY = 50;

// ── SD ────────────────────────────────────────────────────────
static constexpr size_t   SD_BATCH_SIZE      = 10;

// ── IMU windowed mean ─────────────────────────────────────────
// 3 raw samples × 10 ms = 30 ms per block → ~33 Hz decimated output
static constexpr size_t   IMU_WINDOW_SIZE    = 3;

// ── Event window ──────────────────────────────────────────────
// 100 mean samples × 30 ms = ~3 s event window at ~33 Hz
static constexpr size_t   EVENT_WINDOW_SIZE  = 100;
static constexpr size_t   EVENT_OVERLAP      = 0; 

// ── GPS ───────────────────────────────────────────────────────
static constexpr int      GPS_BAUD           = 9600;   // Serial2: RX=7, TX=8

// ── Network ───────────────────────────────────────────────────
static const IPAddress STATIC_IP(192, 168, 100,  30);
static const IPAddress SUBNET   (255, 255, 255,   0);
static const IPAddress GATEWAY  (192, 168, 100,   1);
static const IPAddress DNS      (  8,   8,   8,   8);

// ── MQTT ──────────────────────────────────────────────────────
static const char* MQTT_BROKER      = "192.168.100.6";
static const int   MQTT_PORT        = 1883;
static const char* MQTT_CLIENT      = "teensy41";
static const char* MQTT_TOPIC_RAW   = "imu/raw";   // 100 Hz raw samples
static const char* MQTT_TOPIC_MEAN  = "imu/mean";    // ~33 Hz decimated mean (3-sample blocks)
static const char* MQTT_TOPIC_EVENT_SCORE  = "score/event";
static const char* MQTT_TOPIC_SESSION_SCORE  = "score/session";

// static const char* MQTT_TOPIC_EVENT     = "imu/event";      // ~0.33 Hz feature vectors (100-sample event window)
// static const char* MQTT_TOPIC_INFERENCE = "imu/inference"; // ~0.33 Hz inference results (label + timing)