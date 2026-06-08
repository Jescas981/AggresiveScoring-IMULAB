/**
 * logger.ino — Teensy 4.1 IMU + GPS Session Logger
 *
 * Module responsibilities
 * ───────────────────────
 *   config.h   – compile-time constants (timing, window sizes, IP, MQTT topics)
 *   packet.h   – LogPacket struct + TYPE_IMU / TYPE_GPS constants
 *   imu.h      – MPU6050 init + imuReadRaw() / imuReadDecimated()
 *   lpf.h      – IIR LPF for accel (4th-order) and gyro (2nd-order), DF2T
 *   event.h    – Event window accumulator + feature extraction (30 floats)
 *   inference.h – RF model wrapper: normalize → predict → InferenceResult
 *   gps.h      – NEO-6M polling via TinyGPSPlus (~1 Hz fix)
 *   storage.h  – EEPROM session counter + SD batched binary writer
 *   network.h  – QNEthernet static-IP + PubSubClient MQTT
 *
 * Data flow (every 10 ms, 100 Hz)
 * ────────────────────────────────
 *   raw = imuReadRaw()
 *        │
 *        ├─► SD + MQTT "imu/raw"              (100 Hz, unfiltered)
 *        │
 *        └─► lpfUpdateImu()                   (IIR LPF in-place on copy)
 *                 │
 *                 └─► imuReadDecimated()       (mean of 3 LPF samples → ~33 Hz)
 *                          │
 *                          ├─► SD + MQTT "imu/mean"    (~33 Hz)
 *                          │
 *                          └─► eventWindowPush()       (accumulate 100 mean samples)
 *                                   └─► MQTT "imu/event"  (~0.33 Hz, 30 features)
 *
 *   GPS fix (~1 Hz) → SD + MQTT "imu/raw"
 */

#include "config.h"
#include "packet.h"
#include "imu.h"
#include "lpf.h"
#include "event.h"
#include "inference.h"
#include "gps.h"
#include "storage.h"
#include "network.h"
#include "score.h"

// ── State ─────────────────────────────────────────────────────
static uint32_t sessionID = 0;
static uint32_t lastLogTime = 0;
static uint32_t lastFlushTime = 0;
static uint32_t packetCount = 0;
static ImuLPF imuFilter;
static EventWindow eventWin;
static SessionScore sessionScore;
static uint32_t lastScorePrint = 0;
// ── Helpers ───────────────────────────────────────────────────

static void serialPrint(const LogPacket &pkt)
{
    if (++packetCount % SERIAL_PRINT_EVERY != 0)
        return;

    if (pkt.type == TYPE_IMU)
    {
        Serial.printf("[IMU][%8lu] a=[%6.2f,%6.2f,%6.2f] g=[%6.3f,%6.3f,%6.3f]\n",
                      pkt.timestamp_ms,
                      pkt.data[0], pkt.data[1], pkt.data[2],
                      pkt.data[3], pkt.data[4], pkt.data[5]);
    }
    else
    {
        Serial.printf("[GPS][%8lu] lat=%9.5f lon=%10.5f spd=%5.2f crs=%6.1f alt=%6.1f utc=%9.2f\n",
                      pkt.timestamp_ms,
                      pkt.data[0], pkt.data[1], pkt.data[2],
                      pkt.data[3], pkt.data[4], pkt.data[5]);
    }
}

// ── Arduino entrypoints ───────────────────────────────────────

void setup()
{
    Serial.begin(115200);
    delay(200);
    Serial.println("=== Teensy 4.1 IMU + GPS Logger ===");

    gpsInit();

    if (!imuInit())
    {
        while (true)
            delay(500);
    }

    sessionID = storageLoadSession();
    if (!storageInit(sessionID))
    {
        while (true)
            delay(500);
    }

    networkInit();
    lpfResetImu(imuFilter);
    eventWindowReset(eventWin);

    lastLogTime = millis();
    lastFlushTime = millis();
    Serial.println("[Setup] Done\n");
}

void loop()
{
    uint32_t now = millis();

    // ── IMU @ 100 Hz ──────────────────────────────────────────
    if (now - lastLogTime >= LOG_INTERVAL_MS)
    {
        lastLogTime += LOG_INTERVAL_MS;

        // 1. Raw → SD + imu/raw  (100 Hz, unfiltered)
        LogPacket rawPkt = imuReadRaw(sessionID);
        storageWrite(rawPkt);
        networkPublish(rawPkt, MQTT_TOPIC_RAW);
        // serialPrint(rawPkt);

        // Apply Rz(180°)
        rawPkt.data[0] = -rawPkt.data[0]; // ax
        rawPkt.data[1] = -rawPkt.data[1]; // ay
        rawPkt.data[3] = -rawPkt.data[3]; // gx
        rawPkt.data[4] = -rawPkt.data[4]; // gy

        // 2. LPF applied on a copy of the raw data
        LogPacket filtPkt = rawPkt;
        lpfUpdateImu(imuFilter, filtPkt.data);

        // 3. Accumulate 3 LPF samples → emit block mean at ~33 Hz
        LogPacket meanPkt;
        if (imuReadDecimated<DecimationMethod::Last>(
                sessionID, filtPkt, meanPkt))
        {
            storageWrite(meanPkt);
            networkPublish(meanPkt, MQTT_TOPIC_MEAN);

            // 4. Accumulate 100 mean samples (~3 s) → extract features
            FeaturePacket featPkt;
            if (eventWindowPush(eventWin, meanPkt, sessionID, featPkt))
            {
#if defined(MODEL_TYPE_CNN)

                float window[IMU_CHANNELS * EVENT_WINDOW_SIZE];

                eventWindowToTensor(
                    eventWin,
                    window,
                    IMU_CHANNELS,
                    EVENT_WINDOW_SIZE);
                InferenceResult result = infer(window, sessionID);
#else
                InferenceResult result = infer(featPkt);
#endif
                // networkPublish(featPkt, MQTT_TOPIC_EVENT);
                // Ignorar NORMAL
                if (result.label != static_cast<uint8_t>(EventType::NORMAL))
                {
                    EventType event =
                        static_cast<EventType>(result.label);

                    float eventScore =
                        computeEventScore(
                            event,
                            featPkt.peak_ax,
                            featPkt.jerk,
                            featPkt.rms);

                    scoreAddEvent(
                        sessionScore,
                        event,
                        eventScore);

                    EventScorePacket evtPkt;
                    evtPkt.timestamp_ms = result.timestamp_ms;
                    evtPkt.session_id = sessionID;
                    evtPkt.event = (uint8_t)event;
                    evtPkt.peak_ax = featPkt.peak_ax;
                    evtPkt.jerk = featPkt.jerk;
                    evtPkt.rms = featPkt.rms;
                    evtPkt.event_score = eventScore;

                    networkPublish(evtPkt, MQTT_TOPIC_EVENT_SCORE);

                    Serial.printf(
                        "[SCORE][%8lu] event=%d peak_ax=%.4f jerk=%.4f rms=%.4f score=%.2f\n",
                        result.timestamp_ms,
                        (int)event,
                        featPkt.peak_ax,
                        featPkt.jerk,
                        featPkt.rms,
                        eventScore);
                }

                // Serial.printf(
                //     "[INF][%8lu] session=%lu label=%ld (%s) time=%lu us\n",
                //     result.timestamp_ms,
                //     result.session_id,
                //     result.label,
                //     result.label_name,
                //     result.elapsed_us);
            }
        }
    }

    // ── GPS @ ~1 Hz (new valid fix) ───────────────────────────
    LogPacket gpsPkt;
    if (gpsPoll(sessionID, gpsPkt))
    {
        storageWrite(gpsPkt);
        networkPublish(gpsPkt, MQTT_TOPIC_MEAN);
        serialPrint(gpsPkt);
    }

    // ── SD periodic fsync ─────────────────────────────────────
    if (now - lastFlushTime >= SD_FLUSH_INTERVAL)
    {
        lastFlushTime = now;
        storageFlush(true);
    }

    // Last score print
    if (now - lastScorePrint >= 10000)
    {
        lastScorePrint = now;

        float sessionScoreValue = computeSessionScore(sessionScore);

        Serial.printf(
            "[SESSION SCORE][%8lu] score=%.2f | n_f=%lu n_g=%lu n_r=%lu\n",
            now,
            sessionScoreValue,
            (unsigned long)sessionScore.n_frenado,
            (unsigned long)sessionScore.n_giro,
            (unsigned long)sessionScore.n_rompemuelle);

        SessionScorePacket sessPkt;
        sessPkt.timestamp_ms = now;
        sessPkt.session_id = sessionID;
        sessPkt.session_score = sessionScoreValue;
        sessPkt.n_frenado = sessionScore.n_frenado;
        sessPkt.n_giro = sessionScore.n_giro;
        sessPkt.n_rompemuelle = sessionScore.n_rompemuelle;
        networkPublish(sessPkt, MQTT_TOPIC_SESSION_SCORE);
    }

    // ── MQTT keepalive ────────────────────────────────────────
    networkMaintain();
}