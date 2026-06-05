#include <Arduino.h>
#include "model_interface.h"

#if defined(ESP32)
  #include "esp_heap_caps.h"
#endif

// ─── Utilidad RAM ────────────────────────────────────────────────────────────
uint32_t getFreeRAM() {
#if defined(ESP32)
  return esp_get_free_heap_size();
// #elif defined(TEENSYDUINO)
//   char top;
//   return (uint32_t)(&top) - (uint32_t)(__malloc_heap_start);
return 0; // Teensy no tiene función nativa para RAM libre, pero se podría implementar con extern char __heap_start y __brkval
#else
  return 0;
#endif
}

uint32_t getMinFreeRAM() {
#if defined(ESP32)
  return esp_get_minimum_free_heap_size();
#else
  return 0; // Teensy no tiene historial nativo
#endif
}
// ─────────────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  while (!Serial);
  Serial.println("READY");
}

void handleSerialCommand() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();

  // ── Comando INFO ────────────────────────────────────────────────────────────
  if (line == "INFO" || line == "info") {
    Serial.print("Model: ");
    Serial.println(model_get_name());
    Serial.print("Features: ");
    Serial.println(model_get_n_features());
    Serial.print("Classes: ");
    Serial.println(model_get_n_classes());
    Serial.print("Free RAM: ");
    Serial.println(getFreeRAM());
#if defined(ESP32)
    Serial.print("Min Free RAM: ");
    Serial.println(getMinFreeRAM());
    Serial.print("Largest free block: ");
    Serial.println(heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
#endif
    Serial.println("READY");
    return;
  }

  // ── Ignorar líneas vacías ───────────────────────────────────────────────────
  if (line.length() == 0) return;

  // ── Parsear features ────────────────────────────────────────────────────────
  float features[model_get_n_features()];
  int idx = 0;
  char buf[512];
  line.toCharArray(buf, sizeof(buf));

  char *token = strtok(buf, ",");
  while (token != nullptr && idx < model_get_n_features()) {
    features[idx++] = atof(token);
    token = strtok(nullptr, ",");
  }

  if (idx != model_get_n_features()) {
    Serial.println("ERROR:features_incompletas");
    return;
  }

  // ── Inferencia con medición de RAM y tiempo ─────────────────────────────────
  uint32_t ram_antes  = getFreeRAM();
  uint32_t t0         = micros();

  int clase = model_predict_raw(features);

  uint32_t t1        = micros();
  uint32_t ram_despues = getFreeRAM();

  // ── Respuesta: clase,nombre,tiempo_us,ram_antes,ram_delta ──────────────────
  Serial.print(clase);
  Serial.print(",");
  Serial.print(model_get_class_name(clase));
  Serial.print(",");
  Serial.print(t1 - t0);                          // latencia en µs
  Serial.print(",");
  Serial.print(ram_antes);                         // RAM libre antes
  Serial.print(",");
  Serial.println((int32_t)ram_antes - (int32_t)ram_despues); // delta (>0 = consumió)
}

void loop() {
  handleSerialCommand();
}