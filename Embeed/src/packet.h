#pragma once

#include <stdint.h>

/**
 * Unified log packet — 36 bytes, packed.
 *
 *  TYPE_IMU → data = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
 *  TYPE_GPS → data = [lat, lon, speed_mps, course_deg, altitude_m, utc_seconds]
 *
 * When type == TYPE_IMU and the windowed mean is active, each data[] value
 * is the mean of IMU_WINDOW_SIZE raw samples (≈33 samples at 100 Hz → 33 Hz output).
 */

static constexpr uint8_t TYPE_IMU = 0x01;
static constexpr uint8_t TYPE_GPS = 0x02;

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

#pragma once

#include <Arduino.h>
#include "config.h"
#include "score.h"

// ============================================================
// EVENT SCORE PACKET
// ============================================================

#pragma pack(push, 1)
struct EventScorePacket
{
    uint32_t timestamp_ms;
    uint32_t session_id;

    uint8_t  event;        // EventType
    float    peak_ax;
    float    jerk;
    float    rms;

    float    event_score;
};
#pragma pack(pop)

// ============================================================
// SESSION SCORE PACKET
// ============================================================

#pragma pack(push, 1)
struct SessionScorePacket
{
    uint32_t timestamp_ms;
    uint32_t session_id;

    float session_score;

    uint32_t n_frenado;
    uint32_t n_giro;
    uint32_t n_rompemuelle;
};
#pragma pack(pop)