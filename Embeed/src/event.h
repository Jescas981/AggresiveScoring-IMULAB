#pragma once

#include <Arduino.h>
#include <math.h>
#include <string.h>

#include "config.h"
#include "packet.h"

// ── Constants ─────────────────────────────────────────────────
static constexpr uint8_t IMU_CHANNELS   = 6;
static constexpr uint8_t FEATURES_PER_CH = 5;
static constexpr uint8_t FEATURE_COUNT  = IMU_CHANNELS * FEATURES_PER_CH; // 30

// ── Feature packet ────────────────────────────────────────────
#pragma pack(push, 1)
struct FeaturePacket
{
    uint32_t timestamp_ms;
    uint32_t session_id;
    uint8_t  window_size;
    uint8_t  _pad[3];

    float features[FEATURE_COUNT]; // ML features

    // ── extra physics features ──
    float peak_ax;
    float jerk;
    float rms;
};
#pragma pack(pop)

// ── Channel remapping ────────────────────────────────────────
// [ax, ay, az, gx, gy, gz] -> [gx, gy, gz, ax, ay, az]
static constexpr uint8_t CHANNEL_REMAP[IMU_CHANNELS] = {3, 4, 5, 0, 1, 2};

// ── Event window ──────────────────────────────────────────────
struct EventWindow
{
    float buf[EVENT_WINDOW_SIZE][IMU_CHANNELS];
    size_t count;
};

static inline void eventWindowToTensor(
    const EventWindow& ew,
    float* out,
    uint8_t C,
    uint8_t T
) {
    for (uint8_t c = 0; c < C; c++) {
        for (uint8_t t = 0; t < T; t++) {
            out[c * T + t] = ew.buf[t][c];
        }
    }
}

inline void eventWindowReset(EventWindow &ew)
{
    memset(ew.buf, 0, sizeof(ew.buf));
    ew.count = 0;
}

// ── Feature extraction (mean/std/min/max/energy) ──────────────
static void extractChannel(const float *samples, size_t n, float out[5])
{
    float sum = 0.0f;
    float sum_sq = 0.0f;
    float vmin =  1e38f;
    float vmax = -1e38f;

    for (size_t i = 0; i < n; i++)
    {
        float v = samples[i];
        sum += v;
        sum_sq += v * v;
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    float mean = sum / (float)n;
    float var  = (sum_sq / (float)n) - (mean * mean);

    out[0] = mean;
    out[1] = (var > 0.0f) ? sqrtf(var) : 0.0f;
    out[2] = vmin;
    out[3] = vmax;
    out[4] = sum_sq;
}

// ── Public API ────────────────────────────────────────────────
inline bool eventWindowPush(EventWindow &ew,
                            const LogPacket &pkt,
                            uint32_t sessionID,
                            FeaturePacket &out)
{
    // store in canonical order: gx gy gz ax ay az
    for (uint8_t c = 0; c < IMU_CHANNELS; c++)
        ew.buf[ew.count][c] = pkt.data[CHANNEL_REMAP[c]];

    ew.count++;

    if (ew.count < EVENT_WINDOW_SIZE)
        return false;

    // ─────────────────────────────────────────────
    // FEATURE EXTRACTION (ML FEATURES)
    // ─────────────────────────────────────────────
    float col[EVENT_WINDOW_SIZE];

    for (uint8_t c = 0; c < IMU_CHANNELS; c++)
    {
        for (size_t i = 0; i < EVENT_WINDOW_SIZE; i++)
            col[i] = ew.buf[i][c];

        extractChannel(col,
                       EVENT_WINDOW_SIZE,
                       &out.features[c * FEATURES_PER_CH]);
    }

    // ─────────────────────────────────────────────
    // PHYSICS FEATURES (IMPORTANT PART)
    // ─────────────────────────────────────────────

    float peak_ax = 0.0f;
    float rms_ax  = 0.0f;
    float jerk_ax = 0.0f;

    // ax is channel index 3 after remap
    for (size_t i = 0; i < EVENT_WINDOW_SIZE; i++)
    {
        float ax = ew.buf[i][3];

        float abs_ax = fabsf(ax);
        if (abs_ax > peak_ax)
            peak_ax = abs_ax;

        rms_ax += ax * ax;
    }

    rms_ax = sqrtf(rms_ax / EVENT_WINDOW_SIZE);

    for (size_t i = 1; i < EVENT_WINDOW_SIZE; i++)
    {
        float ax1 = ew.buf[i][3];
        float ax0 = ew.buf[i - 1][3];
        jerk_ax += fabsf(ax1 - ax0);
    }

    jerk_ax /= (EVENT_WINDOW_SIZE - 1);

    out.peak_ax = peak_ax;
    out.rms     = rms_ax;
    out.jerk    = jerk_ax;

    // ─────────────────────────────────────────────
    // PACKET METADATA
    // ─────────────────────────────────────────────
    out.timestamp_ms = millis();
    out.session_id    = sessionID;
    out.window_size   = (uint8_t)EVENT_WINDOW_SIZE;

    memset(out._pad, 0, sizeof(out._pad));

    // ─────────────────────────────────────────────
    // OVERLAP HANDLING
    // ─────────────────────────────────────────────
#if EVENT_OVERLAP == 0
    eventWindowReset(ew);
#else
    constexpr size_t STRIDE = EVENT_WINDOW_SIZE - EVENT_OVERLAP;

    memmove(ew.buf,
            ew.buf + STRIDE,
            EVENT_OVERLAP * IMU_CHANNELS * sizeof(float));

    ew.count = EVENT_OVERLAP;
#endif

    return true;
}