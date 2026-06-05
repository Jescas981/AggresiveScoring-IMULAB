#pragma once

/**
 * lpf.h — IIR Low-Pass Filters for IMU axes.
 *
 * Two independent filters, both implemented in Direct Form II Transposed
 * (numerically stable, minimal state storage, suitable for fixed-point-like
 * float precision on Cortex-M7).
 *
 * ── Accelerometer — 4th-order Butterworth LPF ────────────────────────────
 *   B = [4.54666568e-06, 1.81866627e-05, 2.72799941e-05,
 *        1.81866627e-05, 4.54666568e-06]
 *   A = [1.0, -3.75127641, 5.28429506, -3.31262008, 0.77967418]
 *   Order  : 4  (5 coefficients)
 *   Gain   : very narrow passband (DC-pass, aggressive HF rejection)
 *
 * ── Gyroscope — 2nd-order Butterworth LPF ────────────────────────────────
 *   B = [0.0133592, 0.0267184, 0.0133592]
 *   A = [1.0, -1.56101808, 0.64135154]
 *   Order  : 2  (3 coefficients)
 *   Gain   : wider passband, suitable for angular-rate dynamics
 *
 * Usage
 * ─────
 *   // Declare one filter instance per axis (they hold independent state):
 *   AccelLPF ax_f, ay_f, az_f;
 *   GyroLPF  gx_f, gy_f, gz_f;
 *
 *   // Reset all state (call once, or after a gap in data):
 *   lpfResetAccel(ax_f);  lpfResetGyro(gx_f);
 *
 *   // Feed a sample, get filtered output:
 *   float ax_filtered = lpfUpdateAccel(ax_f, ax_raw);
 *   float gx_filtered = lpfUpdateGyro (gx_f, gx_raw);
 *
 * Convenience wrappers operate on a full 6-element float array
 * [ax, ay, az, gx, gy, gz] matching LogPacket::data layout:
 *
 *   ImuLPF imuFilter;
 *   lpfResetImu(imuFilter);
 *   lpfUpdateImu(imuFilter, pkt.data);   // filters pkt.data in-place
 */

#include <string.h>   // memset

// ── Accel filter — order 4 ────────────────────────────────────────────────

static constexpr int   ACCEL_LPF_ORDER = 4;

static constexpr float ACCEL_LPF_B[ACCEL_LPF_ORDER + 1] = {
     4.54666568e-06f,
     1.81866627e-05f,
     2.72799941e-05f,
     1.81866627e-05f,
     4.54666568e-06f
};

static constexpr float ACCEL_LPF_A[ACCEL_LPF_ORDER + 1] = {
     1.0f,
    -3.75127641f,
     5.28429506f,
    -3.31262008f,
     0.77967418f
};

struct AccelLPF {
    // Direct Form II Transposed delay line: w[0..order-1]
    float w[ACCEL_LPF_ORDER] = {};
};

inline void lpfResetAccel(AccelLPF& f) {
    memset(f.w, 0, sizeof(f.w));
}

/**
 * Push one sample through the 4th-order accel IIR.
 * Returns the filtered output.
 *
 * Direct Form II Transposed recurrence:
 *   y[n]   = b[0]*x[n] + w[0]
 *   w[k]   = b[k+1]*x[n] - a[k+1]*y[n] + w[k+1]   (k = 0..N-2)
 *   w[N-1] = b[N]*x[n]   - a[N]*y[n]
 */
inline float lpfUpdateAccel(AccelLPF& f, float x) {
    float y = ACCEL_LPF_B[0] * x + f.w[0];

    for (int k = 0; k < ACCEL_LPF_ORDER - 1; k++)
        f.w[k] = ACCEL_LPF_B[k + 1] * x - ACCEL_LPF_A[k + 1] * y + f.w[k + 1];

    f.w[ACCEL_LPF_ORDER - 1] = ACCEL_LPF_B[ACCEL_LPF_ORDER] * x
                              - ACCEL_LPF_A[ACCEL_LPF_ORDER] * y;
    return y;
}

// ── Gyro filter — order 2 ─────────────────────────────────────────────────

static constexpr int   GYRO_LPF_ORDER = 2;

static constexpr float GYRO_LPF_B[GYRO_LPF_ORDER + 1] = {
    0.0133592f,
    0.0267184f,
    0.0133592f
};

static constexpr float GYRO_LPF_A[GYRO_LPF_ORDER + 1] = {
    1.0f,
   -1.56101808f,
    0.64135154f
};

struct GyroLPF {
    float w[GYRO_LPF_ORDER] = {};
};

inline void lpfResetGyro(GyroLPF& f) {
    memset(f.w, 0, sizeof(f.w));
}

inline float lpfUpdateGyro(GyroLPF& f, float x) {
    float y = GYRO_LPF_B[0] * x + f.w[0];

    for (int k = 0; k < GYRO_LPF_ORDER - 1; k++)
        f.w[k] = GYRO_LPF_B[k + 1] * x - GYRO_LPF_A[k + 1] * y + f.w[k + 1];

    f.w[GYRO_LPF_ORDER - 1] = GYRO_LPF_B[GYRO_LPF_ORDER] * x
                             - GYRO_LPF_A[GYRO_LPF_ORDER] * y;
    return y;
}

// ── Convenience: full 6-axis IMU filter ───────────────────────────────────

/**
 * One ImuLPF holds independent filter state for all 6 IMU axes.
 * Axis layout matches LogPacket::data:
 *   [0]=ax  [1]=ay  [2]=az  [3]=gx  [4]=gy  [5]=gz
 */
struct ImuLPF {
    AccelLPF accel[3];   // ax, ay, az
    GyroLPF  gyro[3];    // gx, gy, gz
};

inline void lpfResetImu(ImuLPF& f) {
    for (int i = 0; i < 3; i++) {
        lpfResetAccel(f.accel[i]);
        lpfResetGyro (f.gyro[i]);
    }
}

/**
 * Filter a 6-element float array in-place.
 * data[0..2] → accel axes, data[3..5] → gyro axes.
 */
inline void lpfUpdateImu(ImuLPF& f, float data[6]) {
    for (int i = 0; i < 3; i++)
        data[i]     = lpfUpdateAccel(f.accel[i], data[i]);
    for (int i = 0; i < 3; i++)
        data[i + 3] = lpfUpdateGyro (f.gyro[i],  data[i + 3]);
}