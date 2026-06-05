#include <Arduino.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

extern const unsigned char g_model[];
extern const unsigned int g_model_len;

#define N_FEATURES  36
#define N_CLASSES    6

static tflite::MicroMutableOpResolver<4> resolver;

constexpr int kArenaSize = 16 * 1024;
static uint8_t tensor_arena[kArenaSize];

static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input  = nullptr;
static TfLiteTensor* output = nullptr;

void setup() {
    Serial.begin(115200);
    delay(1000);

    tflite::InitializeTarget();

    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddReshape();

    const tflite::Model* model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Schema mismatch!");
        while (1);
    }

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kArenaSize
    );
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors FAILED — aumenta kArenaSize");
        while (1);
    }

    input  = interpreter->input(0);
    output = interpreter->output(0);

    // Verificar que el modelo coincide con lo esperado
    if (input->dims->data[1] != N_FEATURES) {
        Serial.print("ERROR: modelo espera ");
        Serial.print(input->dims->data[1]);
        Serial.println(" features, no 36");
        while (1);
    }

    Serial.print("Input features: ");
    Serial.println(input->dims->data[1]);   // debe imprimir 36
    Serial.print("Output clases:  ");
    Serial.println(output->dims->data[1]);
    Serial.print("Arena usado:    ");
    Serial.print(interpreter->arena_used_bytes());
    Serial.println(" bytes");
    Serial.println("Listo!");
}

void loop() {
    float features[N_FEATURES] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };

    for (int i = 0; i < N_FEATURES; i++) {
        input->data.f[i] = features[i];
    }

    uint32_t t_start = micros();  // ← antes

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke FAILED");
        return;
    }

    uint32_t t_end = micros();    // ← después

    int best = 0;
    for (int i = 1; i < N_CLASSES; i++) {
        if (output->data.f[i] > output->data.f[best])
            best = i;
    }

    Serial.print("Clase: ");
    Serial.print(best);
    Serial.print(" | Prob: ");
    Serial.print(output->data.f[best]);
    Serial.print(" | Inference: ");
    Serial.print(t_end - t_start);  // ← imprime
    Serial.println(" us");

    delay(100);
}