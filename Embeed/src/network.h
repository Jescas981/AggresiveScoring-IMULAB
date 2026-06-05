#pragma once

/**
 * network.h — Static-IP Ethernet + MQTT via QNEthernet / PubSubClient.
 *
 * Call networkInit() once in setup() after SD/IMU are ready.
 * Call networkMaintain() every loop() to keep the MQTT connection alive.
 * Call networkPublish() to push a packet to the broker.
 */

#include <Arduino.h>
#include <QNEthernet.h>
#include <PubSubClient.h>

#include "config.h"
#include "packet.h"

using namespace qindesign::network;

namespace network_detail {
    EthernetClient _ethClient;
    PubSubClient   _mqtt(_ethClient);
} // namespace network_detail

// ── Public API ────────────────────────────────────────────────

/** Reconnect to the MQTT broker if disconnected. */
static void _mqttReconnect() {
    if (network_detail::_mqtt.connected()) return;

    Serial.print("[MQTT] Reconnecting...");
    if (network_detail::_mqtt.connect(MQTT_CLIENT)) {
        Serial.println(" OK.");
    } else {
        Serial.printf(" failed (rc=%d)\n", network_detail::_mqtt.state());
    }
}

/**
 * Configure static IP, wait up to 5 s for link, then connect to MQTT.
 * Call once in setup().
 */
void networkInit() {
    Ethernet.begin();
    Ethernet.setDHCPEnabled(false);
    Ethernet.setLocalIP(STATIC_IP);
    Ethernet.setSubnetMask(SUBNET);
    Ethernet.setGatewayIP(GATEWAY);
    Ethernet.setDNSServerIP(DNS);

    uint32_t t0 = millis();
    while (Ethernet.linkStatus() != LinkON && millis() - t0 < 5000)
        delay(100);

    Serial.printf("[ETH] Link: %s  IP: %s\n",
                  Ethernet.linkStatus() == LinkON ? "ON" : "OFF",
                  Ethernet.localIP());

    network_detail::_mqtt.setServer(MQTT_BROKER, MQTT_PORT);
    network_detail::_mqtt.setBufferSize(sizeof(LogPacket) + 64);
    _mqttReconnect();
}

/**
 * Keep MQTT alive and process incoming messages.
 * Call every loop() iteration.
 */
void networkMaintain() {
    _mqttReconnect();
    network_detail::_mqtt.loop();
}

/**
 * Publish a packet to an explicit MQTT topic.
 * Silently skipped if not connected.
 *
 * Usage:
 *   networkPublish(pkt, MQTT_TOPIC_RAW);   // 100 Hz
 *   networkPublish(pkt, MQTT_TOPIC_MEAN);  // ~3 Hz decimated
 */
void networkPublish(const LogPacket& pkt, const char* topic) {
    if (!network_detail::_mqtt.connected()) return;
    network_detail::_mqtt.publish(
        topic,
        reinterpret_cast<const uint8_t*>(&pkt),
        sizeof(pkt)
    );
}

/** Publish any arbitrary payload by raw pointer + length. */
void networkPublish(const void* payload, const char* topic, size_t len) {
    if (!network_detail::_mqtt.connected()) return;
    network_detail::_mqtt.publish(
        topic,
        reinterpret_cast<const uint8_t*>(payload),
        len
    );
}

void networkPublish(const EventScorePacket& pkt, const char* topic)
{
    if (!network_detail::_mqtt.connected()) return;

    network_detail::_mqtt.publish(
        topic,
        reinterpret_cast<const uint8_t*>(&pkt),
        sizeof(pkt)
    );
}

void networkPublish(const SessionScorePacket& pkt, const char* topic)
{
    if (!network_detail::_mqtt.connected()) return;

    network_detail::_mqtt.publish(
        topic,
        reinterpret_cast<const uint8_t*>(&pkt),
        sizeof(pkt)
    );
}