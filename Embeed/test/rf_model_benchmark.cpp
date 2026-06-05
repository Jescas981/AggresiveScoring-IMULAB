#include <Arduino.h>
#include "rf_model.h"
#include "scaler.h"

Eloquent::ML::Port::RandomForest clf;

void setup() {
    Serial.begin(115200);
    while (!Serial);
    Serial.println("READY");
}

void loop() {
    if (!Serial.available()) return;

    // Leer línea CSV: "0.1,0.2,...,0.36\n"
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.length() == 0) return;

    // Parsear features
    float features[N_FEATURES];
    int idx = 0;
    char buf[512];
    line.toCharArray(buf, sizeof(buf));
    char* token = strtok(buf, ",");

    while (token != nullptr && idx < N_FEATURES) {
        features[idx++] = atof(token);
        token = strtok(nullptr, ",");
    }

    if (idx != N_FEATURES) {
        Serial.println("ERROR:features_incompletas");
        return;
    }

    // Normalizar
    for (int i = 0; i < N_FEATURES; i++)
        features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];

    // Inferencia
    uint32_t t0 = micros();
    int clase = clf.predict(features);
    uint32_t t1 = micros();

    // Respuesta: "clase,tiempo_us"
    Serial.print(clase);
    Serial.print(",");
    Serial.println(t1 - t0);
}