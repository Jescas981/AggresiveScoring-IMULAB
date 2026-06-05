#pragma once

/**
 * imu.h — MPU6050 driver wrapper with windowed mean decimation.
 *
 * Strategy
 * ─────────
 * Raw reads happen at 100 Hz (every LOG_INTERVAL_MS = 10 ms).
 * Samples are pushed into a circular ring buffer of IMU_WINDOW_SIZE (33) slots.
 * Once the buffer is full, a mean packet is emitted and the buffer resets,
 * giving an effective output rate of ~100/33 ≈ 3 Hz … wait, that's not what
 * you want.
 *
 * Correct interpretation of "window of 33 Hz":
 *   You want a *sliding* mean that is published at the same 100 Hz cadence
 *   but each published value is the running mean of the last 33 raw samples.
 *   This acts as a simple low-pass / anti-alias FIR that removes noise above
 *   ~1.5 Hz half-power (33-tap boxcar), while still streaming at 100 Hz.
 *
 * Two modes are provided — choose by setting IMU_DECIMATED in config or
 * calling the appropriate function:
 *
 *   imuReadRaw()        – returns one packet from the latest single read.
 *   imuReadMean()       – returns a packet only when a full window of
 *                         IMU_WINDOW_SIZE samples has been accumulated,
 *                         flushing the ring after each emission (~3 Hz net).
 *   imuReadSlidingMean()– updates the ring every call and always returns a
 *                         packet containing the mean of the last 33 samples
 *                         (100 Hz output, smoothed).
 *
 * Use imuWindowReady() to check if imuReadMean() has a new packet this cycle.
 */

#define I2CDEV_IMPLEMENTATION I2CDEV_ARDUINO_WIRE

#include <Wire.h>
#include <I2Cdev.h>
#include <MPU6050.h>
#include <Arduino.h>

#include "config.h"
#include "packet.h"

// ── Internal state ────────────────────────────────────────────
namespace imu_detail {

    MPU6050 _mpu;

    // Ring buffer for windowed mean — 6 axes × IMU_WINDOW_SIZE samples
    float   _ring[IMU_WINDOW_SIZE][6];
    size_t  _head     = 0;   // next write index
    size_t  _count    = 0;   // samples accumulated (capped at IMU_WINDOW_SIZE)
    float   _runSum[6] = {};  // running sum for O(1) sliding mean update

    // Scale factors (computed once at init)
    float   ACCEL_SCALE = 0.0f;
    float   GYRO_SCALE  = 0.0f;

    // Read raw int16 from sensor → fill float[6] (m/s², rad/s)
    inline void readRawFloats(float out[6]) {
        int16_t ax, ay, az, gx, gy, gz;
        _mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
        out[0] = ax * ACCEL_SCALE;
        out[1] = ay * ACCEL_SCALE;
        out[2] = az * ACCEL_SCALE;
        out[3] = gx * GYRO_SCALE;
        out[4] = gy * GYRO_SCALE;
        out[5] = gz * GYRO_SCALE;
    }

    // Push one sample into the ring buffer, update running sums.
    inline void pushSample(const float sample[6]) {
        if (_count == IMU_WINDOW_SIZE) {
            // Evict oldest slot from running sum
            for (int i = 0; i < 6; i++)
                _runSum[i] -= _ring[_head][i];
        }
        for (int i = 0; i < 6; i++) {
            _ring[_head][i] = sample[i];
            _runSum[i]     += sample[i];
        }
        _head = (_head + 1) % IMU_WINDOW_SIZE;
        if (_count < IMU_WINDOW_SIZE) _count++;
    }

    // Fill a LogPacket with the current running mean.
    inline LogPacket meanPacket(uint32_t sessionID) {
        LogPacket pkt;
        pkt.timestamp_ms        = millis();
        pkt.session_id          = sessionID;
        pkt.type                = TYPE_IMU;
        pkt._pad[0] = pkt._pad[1] = pkt._pad[2] = 0;
        float n = (float)_count;
        for (int i = 0; i < 6; i++)
            pkt.data[i] = _runSum[i] / n;
        return pkt;
    }

} // namespace imu_detail

// ── Public API ────────────────────────────────────────────────

/**
 * Initialize MPU6050.  Call once in setup().
 * Returns true on success.
 */
bool imuInit() {
    Wire.begin();
    Wire.setClock(400000);

    Serial.println("[MPU6050] Initializing...");
    imu_detail::_mpu.initialize();

    if (!imu_detail::_mpu.testConnection()) {
        Serial.println("[MPU6050] ERROR: not found.");
        return false;
    }

    imu_detail::_mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_4);
    imu_detail::_mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_500);
    imu_detail::_mpu.setDLPFMode(MPU6050_DLPF_BW_42);
    imu_detail::_mpu.setRate(4);          // internal ODR = 200 Hz (1 kHz / (1+4))
    imu_detail::_mpu.setSleepEnabled(false);

    // FS_4  → 8192 LSB/g  (±4 g range)
    // FS_500 → 65.5 LSB/(°/s)
    imu_detail::ACCEL_SCALE = 9.81f / 8192.0f;
    imu_detail::GYRO_SCALE  = DEG_TO_RAD / 65.5f;

    Serial.println("[MPU6050] OK — I2C @ 400 kHz, DLPF 42 Hz, ODR 200 Hz");
    return true;
}

/**
 * Read a single raw sample. No filtering.
 * Call at 100 Hz from loop().
 */
LogPacket imuReadRaw(uint32_t sessionID) {
    float s[6];
    imu_detail::readRawFloats(s);

    LogPacket pkt;
    pkt.timestamp_ms        = millis();
    pkt.session_id          = sessionID;
    pkt.type                = TYPE_IMU;
    pkt._pad[0] = pkt._pad[1] = pkt._pad[2] = 0;
    for (int i = 0; i < 6; i++) pkt.data[i] = s[i];
    return pkt;
}

/**
 * Sliding-window mean (100 Hz input → 100 Hz smoothed output).
 *
 * Every call:
 *   1. Reads one raw sample from the sensor.
 *   2. Pushes it into the IMU_WINDOW_SIZE ring buffer (oldest evicted).
 *   3. Returns a packet whose data[] is the mean of the last ≤33 samples.
 *
 * Output rate matches input rate; latency = IMU_WINDOW_SIZE/2 samples ≈ 165 ms.
 */
LogPacket imuReadSlidingMean(uint32_t sessionID) {
    float s[6];
    imu_detail::readRawFloats(s);
    imu_detail::pushSample(s);
    return imu_detail::meanPacket(sessionID);
}

enum class DecimationMethod {
    Mean,
    Median,
    Last
};

template<DecimationMethod Method>
bool imuReadDecimated(
    uint32_t sessionID,
    const LogPacket& in,
    LogPacket& out)
{
    static size_t fillCount = 0;
    static float acc[6] = {};
    static float buffer[IMU_WINDOW_SIZE][6];

    if constexpr (Method == DecimationMethod::Mean) {
        for (int i = 0; i < 6; i++)
            acc[i] += in.data[i];
    }

    if constexpr (Method == DecimationMethod::Median) {
        memcpy(buffer[fillCount], in.data, sizeof(in.data));
    }

    fillCount++;

    if (fillCount < IMU_WINDOW_SIZE)
        return false;

    out.timestamp_ms = millis();
    out.session_id = sessionID;
    out.type = TYPE_IMU;

    if constexpr (Method == DecimationMethod::Mean) {
        for (int i = 0; i < 6; i++) {
            out.data[i] = acc[i] / IMU_WINDOW_SIZE;
            acc[i] = 0;
        }
    }

    else if constexpr (Method == DecimationMethod::Last) {
        memcpy(out.data, in.data, sizeof(out.data));
    }

    else if constexpr (Method == DecimationMethod::Median) {
        for (int ch = 0; ch < 6; ch++) {
            float tmp[IMU_WINDOW_SIZE];

            for (size_t k = 0; k < IMU_WINDOW_SIZE; k++)
                tmp[k] = buffer[k][ch];

            std::sort(tmp, tmp + IMU_WINDOW_SIZE);

            out.data[ch] = tmp[IMU_WINDOW_SIZE / 2];
        }
    }

    fillCount = 0;
    return true;
}