#pragma once

#include <Arduino.h>
#include <math.h>
#include <algorithm>
#include <vector>

// ============================================================
// EVENT TYPES
// ============================================================

enum class EventType : uint8_t
{
    FRENADO     = 0,
    GIRO        = 1,
    NORMAL      = 2,
    ROMPEMUELLE = 3
};

// ============================================================
// EVENT PARAMETERS
// ============================================================

struct EventScoreParams
{
    float peak_n10;
    float peak_b90;

    float jerk_n10;
    float jerk_b90;

    float rms_n10;
    float rms_b90;

    float w_peak;
    float w_jerk;
    float w_rms;
};

// ============================================================
// PARAMETERS
// ============================================================

constexpr EventScoreParams FRENADO_PARAMS =
{
    0.7534f, 5.3115f,
    0.0088f, 0.0973f,
    0.5642f, 2.9125f,
    0.357f, 0.372f, 0.271f
};

constexpr EventScoreParams GIRO_PARAMS =
{
    0.3065f, 2.6135f,
    0.0109f, 0.0593f,
    0.1807f, 1.4979f,
    0.401f, 0.402f, 0.197f
};

constexpr EventScoreParams ROMPEMUELLE_PARAMS =
{
    1.1830f, 4.3928f,
    0.0476f, 0.1193f,
    0.5850f, 1.4346f,
    0.341f, 0.347f, 0.312f
};

// ============================================================
// SESSION WEIGHTS
// ============================================================

constexpr float SESSION_W_FRENADO     = 0.31f;
constexpr float SESSION_W_GIRO        = 0.42f;
constexpr float SESSION_W_ROMPEMUELLE = 0.27f;

// ============================================================
// SAFE SCORE NORMALIZATION
// ============================================================

inline float scoreFeature(float value,
                          float p10,
                          float p90)
{
    float denom = (p90 - p10);

    if (fabsf(denom) < 1e-6f)
        return 0.0f;

    float score = (value - p10) / denom;
    score *= 100.0f;

    if (score < 0.0f) score = 0.0f;
    if (score > 100.0f) score = 100.0f;

    return score;
}

// ============================================================
// EVENT SCORE
// ============================================================

inline float computeEventScore(
    EventType event,
    float peak_ax,
    float jerk,
    float rms)
{
    const EventScoreParams* p = nullptr;

    switch (event)
    {
        case EventType::FRENADO:     p = &FRENADO_PARAMS; break;
        case EventType::GIRO:        p = &GIRO_PARAMS; break;
        case EventType::ROMPEMUELLE: p = &ROMPEMUELLE_PARAMS; break;
        case EventType::NORMAL:
        default:
            return 0.0f;
    }

    float s_peak = scoreFeature(peak_ax, p->peak_n10, p->peak_b90);
    float s_jerk = scoreFeature(jerk,    p->jerk_n10, p->jerk_b90);
    float s_rms  = scoreFeature(rms,     p->rms_n10,  p->rms_b90);

    return
        p->w_peak * s_peak +
        p->w_jerk * s_jerk +
        p->w_rms  * s_rms;
}

// ============================================================
// PERCENTIL 75
// ============================================================

inline float scorePercentile75(const std::vector<float>& values)
{
    if (values.empty()) return 0.0f;
    if (values.size() == 1) return values[0];

    std::vector<float> sorted = values;
    size_t idx = static_cast<size_t>(0.75f * (sorted.size() - 1));

    std::nth_element(sorted.begin(),
                     sorted.begin() + idx,
                     sorted.end());

    return sorted[idx];
}

// ============================================================
// SESSION STATE
// ============================================================

struct SessionScore
{
    uint32_t n_frenado     = 0;
    uint32_t n_giro        = 0;
    uint32_t n_rompemuelle = 0;

    std::vector<float> frenado_scores;
    std::vector<float> giro_scores;
    std::vector<float> rompemuelle_scores;
};

// ============================================================
// RESET
// ============================================================

inline void scoreReset(SessionScore& s)
{
    s.n_frenado = 0;
    s.n_giro = 0;
    s.n_rompemuelle = 0;
    s.frenado_scores.clear();
    s.giro_scores.clear();
    s.rompemuelle_scores.clear();
}

// ============================================================
// ADD EVENT
// ============================================================

inline void scoreAddEvent(SessionScore& s,
                          EventType event,
                          float eventScore)
{
    if (!isfinite(eventScore))
        return;

    switch (event)
    {
        case EventType::FRENADO:
            s.frenado_scores.push_back(eventScore);
            s.n_frenado++;
            break;

        case EventType::GIRO:
            s.giro_scores.push_back(eventScore);
            s.n_giro++;
            break;

        case EventType::ROMPEMUELLE:
            s.rompemuelle_scores.push_back(eventScore);
            s.n_rompemuelle++;
            break;

        default:
            break;
    }
}

// ============================================================
// SESSION SCORE (usando percentil 75)
// ============================================================

inline float computeSessionScore(const SessionScore& s)
{
    float weightedSum = 0.0f;
    float totalWeight = 0.0f;

    if (s.n_frenado > 0)
    {
        weightedSum += SESSION_W_FRENADO *
            scorePercentile75(s.frenado_scores);
        totalWeight += SESSION_W_FRENADO;
    }

    if (s.n_giro > 0)
    {
        weightedSum += SESSION_W_GIRO *
            scorePercentile75(s.giro_scores);
        totalWeight += SESSION_W_GIRO;
    }

    if (s.n_rompemuelle > 0)
    {
        weightedSum += SESSION_W_ROMPEMUELLE *
            scorePercentile75(s.rompemuelle_scores);
        totalWeight += SESSION_W_ROMPEMUELLE;
    }

    if (totalWeight <= 0.0f)
        return 0.0f;

    return weightedSum / totalWeight;
}