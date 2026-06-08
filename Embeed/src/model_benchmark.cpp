#include <Arduino.h>
#include "model_interface.h"

// ─────────────────────────────
// CONFIG
// ─────────────────────────────
#if defined(MODEL_TYPE_CNN)
#define MAX_FEATURES (6 * 90)
#else
#define MAX_FEATURES 128
#endif

static float features[MAX_FEATURES];

// ─────────────────────────────
// PROTOCOL
// ─────────────────────────────
static const uint8_t MAGIC = 0xAA;

static const uint8_t CMD_INFO = 0x01;
static const uint8_t CMD_DATA = 0x02;

static const uint8_t RESP_INFO = 0x11;
static const uint8_t RESP_DATA = 0x12;
static const uint8_t RESP_LOG = 0x13;
// ─────────────────────────────
// READ EXACT
// ─────────────────────────────
static bool readExact(uint8_t *buf, int n)
{
    int got = 0;
    unsigned long t0 = millis();

    while (got < n)
    {
        if (Serial.available())
        {
            got += Serial.readBytes((char *)buf + got, n - got);
        }
        if (millis() - t0 > 200)
            return false;
    }
    return got == n;
}

void sendLogPacket(const char *msg)
{
    Serial.write(MAGIC);
    Serial.write(RESP_LOG);

    uint16_t len = strlen(msg);

    Serial.write((uint8_t *)&len, sizeof(len));
    Serial.write((uint8_t *)msg, len);
}

// ─────────────────────────────
// INFO RESPONSE (PACKET)
// ─────────────────────────────
void sendInfoPacket()
{
    const char *name = model_get_name();
    uint8_t n_features = (uint8_t)model_get_n_features();
    uint8_t n_classes = (uint8_t)model_get_n_classes();

    uint8_t name_len = strlen(name);

    Serial.write(MAGIC);
    Serial.write(RESP_INFO);

    Serial.write(&name_len, 1);
    Serial.write((uint8_t *)name, name_len);

    Serial.write(&n_features, 1);
    Serial.write(&n_classes, 1);
}

// ─────────────────────────────
// DATA RESPONSE (PACKET)
// ─────────────────────────────
void sendDataPacket(int32_t cls, uint32_t dt_us)
{
    Serial.write(MAGIC);
    Serial.write(RESP_DATA);

    Serial.write((uint8_t *)&cls, sizeof(cls));
    Serial.write((uint8_t *)&dt_us, sizeof(dt_us));
}

// ─────────────────────────────
// SETUP
// ─────────────────────────────
void setup()
{
    Serial.begin(115200);

#if defined(ARDUINO_ARCH_ESP32) || defined(__IMXRT1062__)
    while (!Serial)
    {
    }
#endif

    delay(500);

    Serial.println("BOOT");
    Serial.println("READY");
    Serial.println(model_get_name());
}

// ─────────────────────────────
// LOOP
// ─────────────────────────────
void loop()
{

    if (Serial.available() < 2)
        return;

    if (Serial.read() != MAGIC)
        return;

    uint8_t cmd = Serial.read();

    // ── INFO ─────────────────────────────
    if (cmd == CMD_INFO)
    {
        sendInfoPacket();
        return;
    }

    // ── DATA ─────────────────────────────
    if (cmd == CMD_DATA)
    {

        uint16_t n;
        if (!readExact((uint8_t *)&n, sizeof(n)))
            return;

        if (n > MAX_FEATURES)
        {
            sendLogPacket("ERROR: n > MAX_FEATURES");
            return;
        }

        if (!readExact((uint8_t *)features, n * sizeof(float)))
        {
            sendLogPacket("ERROR: readExact failed");
            return;
        }

        uint32_t t0 = micros();
        int32_t cls = model_predict_raw(features);
        uint32_t t1 = micros();

        sendDataPacket(cls, t1 - t0);
        return;
    }
}