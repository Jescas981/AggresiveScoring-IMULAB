# =========================================================
# EXPORT MODELS WITH COMPILE-TIME SELECTION ONLY
# =========================================================

import os
import joblib
import json
from datetime import datetime
from micromlgen import port

# =========================================================
# CONFIGURATION
# =========================================================

EXPORT_DIR = 'model_input' 
MODELS_BASE_DIR = os.path.join(EXPORT_DIR, 'models')
SCALERS_BASE_DIR = os.path.join(EXPORT_DIR, 'scalers')
OUTPUT_BASE_DIR = "model_cc"

# Configurations to export
CONFIGS = [
    'FULL_IMU_LPF',
    'SELECTED_IMU', 
    'SELECTED_GPS',
    'IMU_WITH_GPS',
    'FULL_IMU_GPS'
]

# SOLO RF
MODEL_TYPES = ['RF','XGB']

# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def export_scaler_h(scaler, path):
    """Export scaler as C header file"""
    mean = scaler.mean_
    scale = scaler.scale_
    
    def arr(name, v):
        return (
            f"static const float {name}[] = {{ "
            + ", ".join([f"{x:.6f}" for x in v])
            + " };\n"
        )
    
    txt = ""
    txt += "#pragma once\n\n"
    txt += f"#define N_FEATURES {len(mean)}\n\n"
    txt += arr("SCALER_MEAN", mean)
    txt += "\n"
    txt += arr("SCALER_SCALE", scale)
    txt += "\n\n"
    txt += "inline void scale_features(const float* features, float* scaled, int n) {\n"
    txt += "    for (int i = 0; i < n; i++) {\n"
    txt += "        scaled[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];\n"
    txt += "    }\n"
    txt += "}\n"
    
    with open(path, "w") as f:
        f.write(txt)


def export_label_encoder_h(le, path):
    """Export label encoder as C header"""
    classes = le.classes_
    
    txt = "#pragma once\n\n"
    txt += f"#define N_CLASSES {len(classes)}\n\n"
    txt += "static const char* CLASS_NAMES[] = {\n"
    for cls in classes:
        txt += f'    "{cls}",\n'
    txt += "};\n\n"
    
    txt += "inline const char* get_class_name(int idx) {\n"
    txt += "    if (idx >= 0 && idx < N_CLASSES) return CLASS_NAMES[idx];\n"
    txt += "    return \"UNKNOWN\";\n"
    txt += "}\n"
    
    with open(path, "w") as f:
        f.write(txt)


def export_rf_header(rf_model, output_path, class_names):
    """Export Random Forest as C++ header using micromlgen"""
    try:
        cpp_code = port(rf_model)
        
        # Añadir función get_class_name
        cpp_code += "\n\n// Get class name from prediction\n"
        cpp_code += "inline const char* get_class_name(int idx) {\n"
        cpp_code += "    static const char* class_names[] = {\n"
        for name in class_names:
            cpp_code += f'        "{name}",\n'
        cpp_code += "    };\n"
        cpp_code += "    if (idx >= 0 && idx < sizeof(class_names)/sizeof(class_names[0])) {\n"
        cpp_code += "        return class_names[idx];\n"
        cpp_code += "    }\n"
        cpp_code += "    return \"UNKNOWN\";\n"
        cpp_code += "}\n"
        
        with open(output_path, "w") as f:
            f.write(cpp_code)
        return True
    except Exception as e:
        print(f"    ⚠ micromlgen failed: {e}")
        return False


# =========================================================
# CREATE COMPILE-TIME SELECTOR HEADER (CORREGIDO)
# =========================================================

def create_compile_time_selector(output_dir, configs, model_types):
    """
    Create header for compile-time model selection.
    Uses the RandomForest class, not global functions.
    """
    
    # Collect existing models
    available_models = []
    for config in configs:
        for model_type in model_types:
            model_name = f"{config}_{model_type}"
            model_dir = os.path.join(output_dir, model_name)
            model_file = os.path.join(model_dir, f"{config}_RF_model.h")
            if os.path.exists(model_dir) and os.path.exists(model_file):
                available_models.append({
                    'name': model_name,
                    'config': config,
                    'type': model_type,
                })
    
    if not available_models:
        print("⚠ No models found!")
        return None, []
    
    # Generar selector header (usando la clase, no funciones globales)
    header = """// =========================================================
// COMPILE-TIME MODEL SELECTOR
// =========================================================
//
// USAGE:
//   #define MODEL_SELECTION IMU_WITH_GPS_RF
//   #include "model_selector.h"
//
//   // En tu código:
//   float scaled[N_FEATURES];
//   scale_features(raw, scaled, N_FEATURES);
//   int result = model_predict(scaled);
//   const char* label = model_get_class_name(result);
//
// =========================================================

#pragma once

// =========================================================
// DEFINE YOUR MODEL (cambia esta línea)
// =========================================================

#ifndef MODEL_SELECTION
    // Si no se define, usar el primer modelo disponible
    #define MODEL_SELECTION """ + available_models[0]['name'] + """
#endif

// =========================================================
// MODELOS DISPONIBLES
// =========================================================

"""
    
    for model in available_models:
        header += f"#define MODEL_{model['name']} {model['name']}\n"
    
    header += "\n// =========================================================\n"
    header += "// INCLUIR MODELO SELECCIONADO\n"
    header += "// =========================================================\n\n"
    
    # Incluir solo el modelo seleccionado
    for model in available_models:
        header += f"#ifdef MODEL_{model['name']}\n"
        header += f"#if MODEL_SELECTION == MODEL_{model['name']}\n"
        header += f'    #include "{model["name"]}/scaler.h"\n'
        header += f'    #include "{model["name"]}/label_encoder.h"\n'
        header += f'    #include "{model["name"]}/{model["config"]}_RF_model.h"\n'
        
        # Crear alias del clasificador
        header += f"\n    // Alias del clasificador para este modelo\n"
        header += f"    using ModelClassifier = Eloquent::ML::Port::RandomForest;\n"
        header += f"    static ModelClassifier clf;\n"
        header += f"#endif\n"
        header += f"#endif\n\n"
    
    header += """
// =========================================================
// FUNCIONES UNIFICADAS
// =========================================================

// Escalar features
static inline void model_scale(const float* input, float* output, int n) {
    scale_features(input, output, n);
}

// Predicción (usa el clasificador)
static inline int model_predict(const float* features) {
"""
    
    for model in available_models:
        header += f"#ifdef MODEL_{model['name']}\n"
        header += f"#if MODEL_SELECTION == MODEL_{model['name']}\n"
        header += f"    return clf.predict(features);\n"
        header += f"#endif\n"
        header += f"#endif\n"
    
    header += """    return -1;
}

// Predicción completa (escala + predice)
static inline int model_predict_raw(const float* raw_features) {
    float scaled[N_FEATURES];
    model_scale(raw_features, scaled, N_FEATURES);
    return model_predict(scaled);
}

// Obtener nombre de clase
static inline const char* model_get_class_name(int idx) {
    return get_class_name(idx);
}

// Obtener nombre del modelo
static inline const char* model_get_name(void) {
"""
    
    for model in available_models:
        header += f"#ifdef MODEL_{model['name']}\n"
        header += f"#if MODEL_SELECTION == MODEL_{model['name']}\n"
        header += f'    return "{model["name"]}";\n'
        header += f"#endif\n"
        header += f"#endif\n"
    
    header += """    return "UNKNOWN";
}

// Obtener número de features
static inline int model_get_n_features(void) {
    return N_FEATURES;
}

// Obtener número de clases
static inline int model_get_n_classes(void) {
    return N_CLASSES;
}

#endif // MODEL_SELECTOR_H
"""
    
    selector_path = os.path.join(output_dir, "model_selector.h")
    with open(selector_path, 'w') as f:
        f.write(header)
    
    return selector_path, available_models


# =========================================================
# MAIN EXPORT FUNCTION
# =========================================================

def export_all_models():
    """Main function to export all models"""
    
    print("="*70)
    print("EXPORTING RF MODELS TO model_cc/")
    print("="*70)
    
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    exported_count = 0
    exported_models = []
    
    for config in CONFIGS:
        for model_type in MODEL_TYPES:
            
            model_name = f"{config}_{model_type}"
            model_dir = os.path.join(OUTPUT_BASE_DIR, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            print(f"\n📁 Exporting: {model_name}")
            
            # Paths
            model_file = os.path.join(MODELS_BASE_DIR, f"{config}_{model_type}.pkl")
            scaler_file = os.path.join(SCALERS_BASE_DIR, f"{config}_{model_type}_scaler.pkl")
            label_encoder_file = os.path.join(MODELS_BASE_DIR, f"{config}_{model_type}_label_encoder.pkl")
            
            # Check files
            if not os.path.exists(model_file):
                print(f"    ⚠ Model not found: {model_file}")
                continue
            
            if not os.path.exists(scaler_file):
                print(f"    ⚠ Scaler not found: {scaler_file}")
                continue
            
            if not os.path.exists(label_encoder_file):
                print(f"    ⚠ Label encoder not found: {label_encoder_file}")
                continue
            
            # Load
            scaler = joblib.load(scaler_file)
            label_encoder = joblib.load(label_encoder_file)
            
            # Export scaler.h
            export_scaler_h(scaler, os.path.join(model_dir, "scaler.h"))
            print(f"    ✓ Exported scaler.h")
            
            # Export label_encoder.h
            export_label_encoder_h(label_encoder, os.path.join(model_dir, "label_encoder.h"))
            print(f"    ✓ Exported label_encoder.h")
            
            # Export RF model
            rf_data = joblib.load(model_file)
            rf_model = rf_data['model'] if isinstance(rf_data, dict) else rf_data
            
            cpp_file = os.path.join(model_dir, f"{config}_RF_model.h")
            if export_rf_header(rf_model, cpp_file, label_encoder.classes_):
                print(f"    ✓ Exported {config}_RF_model.h")
                exported_count += 1
                exported_models.append(model_name)
            
            # Save config
            config_info = {
                'config_name': config,
                'model_type': model_type,
                'input_dim': int(scaler.mean_.shape[0]),
                'n_classes': int(len(label_encoder.classes_)),
                'classes': [str(c) for c in label_encoder.classes_]
            }
            
            with open(os.path.join(model_dir, "model_info.json"), 'w') as f:
                json.dump(config_info, f, indent=2)
            print(f"    ✓ Exported model_info.json")
    
    # Create selector
    selector_path, available_models = create_compile_time_selector(OUTPUT_BASE_DIR, CONFIGS, MODEL_TYPES)
    
    if selector_path:
        print(f"\n✓ Created selector: {selector_path}")
        print(f"\n✅ Models exported: {len(available_models)}")
        for model in available_models:
            print(f"   - {model['name']}")
    
    print("\n" + "="*70)
    print("EXPORT COMPLETE!")
    print("="*70)
    print(f"\n📁 Output: {os.path.abspath(OUTPUT_BASE_DIR)}")
    print("\n📝 To use in Arduino/PlatformIO:")
    print(f"   1. Copy '{OUTPUT_BASE_DIR}' to your project's 'include/' folder")
    print(f"   2. In your code: #define MODEL_SELECTION IMU_WITH_GPS_RF")
    print(f"   3. #include \"model_selector.h\"")
    print(f"   4. Use model_predict(), model_get_class_name(), etc.")
    
    return True


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    export_all_models()