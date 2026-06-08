#pragma once

#include <Arduino.h>
#include "model_interface.h"
#include "event.h"

struct InferenceResult {
    uint32_t timestamp_ms;
    uint32_t session_id;
    int32_t  label;
    uint32_t elapsed_us;
    const char* label_name;
};

// ======================================================
// RF / XGB
// ======================================================
#if defined(MODEL_TYPE_RF) || defined(MODEL_TYPE_XGB)

static inline InferenceResult infer(const FeaturePacket& fv)
{
    float features[N_FEATURES];

    for (int i = 0; i < N_FEATURES; i++) {
        features[i] =
            (fv.features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }

    uint32_t t0 = micros();
    int label = model_predict(features);
    uint32_t t1 = micros();

    return {
        millis(),
        fv.session_id,
        label,
        t1 - t0,
        model_get_class_name(label)
    };
}

#endif

// ======================================================
// CNN
// ======================================================
#if defined(MODEL_TYPE_CNN)

static inline InferenceResult infer(
    const float* window,
    uint32_t session_id
)
{
    uint32_t t0 = micros();
    int label = model_predict(window);
    uint32_t t1 = micros();

    return {
        millis(),
        session_id,
        label,
        t1 - t0,
        model_get_class_name(label)
    };
}

#endif