#pragma once

/**
 * gps.h — NEO-6M GPS polling via TinyGPSPlus.
 *
 * Connects on Serial2 (RX=7, TX=8).  Call gpsInit() once in setup(),
 * then gpsPoll() every loop iteration.  gpsPoll() returns true and fills
 * the output packet only when a new valid fix is parsed (~1 Hz).
 */

#include <Arduino.h>
#include <TinyGPSPlus.h>

#include "config.h"
#include "packet.h"

namespace gps_detail {
    TinyGPSPlus _gps;
} // namespace gps_detail

// ── Public API ────────────────────────────────────────────────

/** Start Serial2 at GPS_BAUD. Call once in setup(). */
void gpsInit() {
    Serial2.begin(GPS_BAUD);
    Serial.println("[GPS] NEO-6M on Serial2 RX=7 TX=8 @ 9600 baud");
}

/**
 * Drain Serial2 and parse NMEA sentences.
 *
 * Returns true and writes a TYPE_GPS packet into `out` only when a new
 * valid location fix is received.  Call every loop() iteration.
 */
bool gpsPoll(uint32_t sessionID, LogPacket& out) {
    bool newFix = false;

    while (Serial2.available()) {
        if (!gps_detail::_gps.encode(Serial2.read())) continue;

        if (gps_detail::_gps.location.isUpdated() &&
            gps_detail::_gps.location.isValid()) {

            float utcSeconds = 0.0f;
            if (gps_detail::_gps.time.isValid()) {
                utcSeconds = gps_detail::_gps.time.hour()        * 3600.0f
                           + gps_detail::_gps.time.minute()      *   60.0f
                           + gps_detail::_gps.time.second()
                           + gps_detail::_gps.time.centisecond() *    0.01f;
            }

            out.timestamp_ms        = millis();
            out.session_id          = sessionID;
            out.type                = TYPE_GPS;
            out._pad[0] = out._pad[1] = out._pad[2] = 0;
            out.data[0] = (float)gps_detail::_gps.location.lat();
            out.data[1] = (float)gps_detail::_gps.location.lng();
            out.data[2] = gps_detail::_gps.speed.isValid()
                              ? (float)gps_detail::_gps.speed.mps()       : 0.0f;
            out.data[3] = gps_detail::_gps.course.isValid()
                              ? (float)gps_detail::_gps.course.deg()      : 0.0f;
            out.data[4] = gps_detail::_gps.altitude.isValid()
                              ? (float)gps_detail::_gps.altitude.meters() : 0.0f;
            out.data[5] = utcSeconds;

            newFix = true;
        }
    }

    return newFix;
}