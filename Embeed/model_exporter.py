# =========================================================
# EXPORT MODELS WITH COMPILE-TIME SELECTION + CNN1D (TFLITE MICRO SAFE)
# =========================================================

import os
import joblib
import json
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from micromlgen import port

# =========================================================
# CONFIGURATION
# =========================================================

EXPORT_DIR = 'model_input'
MODELS_BASE_DIR = os.path.join(EXPORT_DIR, 'models')
SCALERS_BASE_DIR = os.path.join(EXPORT_DIR, 'scalers')
OUTPUT_BASE_DIR = "model_cc"

CONFIGS = ['IMU6']
MODEL_TYPES = ['RF', 'XGB', 'CNN1D']

# =========================================================
# LOAD CALIBRATION DATA (FOR INT8)
# =========================================================

CALIB_DATA_PATH = os.path.join(EXPORT_DIR, "datasets", "IMU6", "X_train.npy")

if os.path.exists(CALIB_DATA_PATH):
    X_calib = np.load(CALIB_DATA_PATH).astype(np.float32)
    X_calib = X_calib[:300]  # reduce for calibration
else:
    X_calib = None
    print("⚠️ No calibration dataset found - INT8 may degrade")

# =========================================================
# CNN1D MODEL
# =========================================================

class CNN1D(nn.Module):
    def __init__(self, C, n_classes, dropout=0.3):
        super().__init__()

        def block(in_ch, out_ch, k=3):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )

        self.features = nn.Sequential(
            nn.Conv1d(C, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            block(32, 64),
            block(64, 128),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# =========================================================
# UTILS
# =========================================================

def export_scaler_h(scaler, path):
    mean = scaler.mean_
    scale = scaler.scale_

    def arr(name, v):
        return (
            f"static const float {name}[] = {{ "
            + ", ".join([f"{x:.6f}" for x in v])
            + " };\n"
        )

    txt = "#pragma once\n\n"
    txt += f"#define N_FEATURES {len(mean)}\n\n"
    txt += arr("SCALER_MEAN", mean)
    txt += "\n"
    txt += arr("SCALER_SCALE", scale)
    txt += "\n"
    txt += """
inline void scale_features(const float* features, float* scaled, int n) {
    for (int i = 0; i < n; i++) {
        scaled[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }
}
"""
    with open(path, "w") as f:
        f.write(txt)


def export_label_encoder_h(le, path):
    classes = le.classes_

    txt = "#pragma once\n\n"
    txt += f"#define N_CLASSES {len(classes)}\n\n"
    txt += "static const char* CLASS_NAMES[] = {\n"

    for c in classes:
        txt += f'    "{c}",\n'

    txt += "};\n\n"

    txt += """
inline const char* get_class_name(int idx) {
    if (idx >= 0 and idx < N_CLASSES) return CLASS_NAMES[idx];
    return "UNKNOWN";
}
"""
    with open(path, "w") as f:
        f.write(txt)

# =========================================================
# RF / XGB EXPORT
# =========================================================

def export_rf_header(model, output_path, class_names):
    try:
        cpp = port(model)

        cpp += "\n\ninline const char* get_class_name(int idx) {\n"
        cpp += "static const char* names[] = {\n"

        for n in class_names:
            cpp += f'    "{n}",\n'

        cpp += """};
    if (idx >= 0 and idx < sizeof(names)/sizeof(names[0]))
        return names[idx];
    return "UNKNOWN";
}
"""

        with open(output_path, "w") as f:
            f.write(cpp)

        return True

    except Exception as e:
        print("RF export error:", e)
        return False

# =========================================================
# CNN LOAD
# =========================================================

def load_cnn1d(path, C, n_classes):
    model = CNN1D(C, n_classes)
    state = torch.load(path, map_location="cpu")

    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    return model

# =========================================================
# REPRESENTATIVE DATASET (FIX PRINCIPAL)
# =========================================================

def rep_data():
    if X_calib is None:
        return

    for i in range(len(X_calib)):
        x = X_calib[i].astype(np.float32)

        # expected CNN format: (B, C, T)
        yield [x[np.newaxis, :, :]]

# =========================================================
# CNN → TFLITE EXPORT
# =========================================================

def export_cnn1d_tflite(model, C, input_len, out_path):
    model.eval()
    dummy = torch.randn(1, C, input_len)

    onnx_path = out_path.replace(".tflite", ".onnx")

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=12,
        input_names=["input"],
        output_names=["output"]
    )

    import subprocess
    import tempfile

    tf_dir = tempfile.mkdtemp()

    subprocess.run([
        "onnx2tf",
        "-i", onnx_path,
        "-o", tf_dir
    ], check=True)

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)

    # 🔥 INT8 QUANTIZATION
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # converter.representative_dataset = rep_data

    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    # ]

    # converter.inference_input_type = tf.int8
    # converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(out_path, "wb") as f:
        f.write(tflite_model)

# =========================================================
# TFLITE HEADER
# =========================================================

def tflite_to_header(tflite_path, header_path):
    with open(tflite_path, "rb") as f:
        data = f.read()

    txt = "#pragma once\n\n"
    txt += "alignas(16) const unsigned char cnn1d_model[] = {\n"

    for i, b in enumerate(data):
        txt += f"0x{b:02x}, "
        if i % 16 == 0:
            txt += "\n"

    txt += "\n};\n\n"
    txt += f"const int cnn1d_model_len = {len(data)};\n"

    with open(header_path, "w") as f:
        f.write(txt)

# =========================================================
# MAIN EXPORT PIPELINE
# =========================================================

def export_all_models():

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    for config in CONFIGS:
        for model_type in MODEL_TYPES:

            name = f"{config}_{model_type}"
            model_dir = os.path.join(OUTPUT_BASE_DIR, name)
            os.makedirs(model_dir, exist_ok=True)

            print("\n📦", name)

            scaler_file = os.path.join(SCALERS_BASE_DIR, f"{name}_scaler.pkl")
            label_file = os.path.join(MODELS_BASE_DIR, f"{name}_label_encoder.pkl")

            if not os.path.exists(scaler_file):
                print("missing scaler")
                continue

            scaler = joblib.load(scaler_file)
            le = joblib.load(label_file)

            export_scaler_h(scaler, os.path.join(model_dir, "scaler.h"))
            export_label_encoder_h(le, os.path.join(model_dir, "label_encoder.h"))

            C = scaler.mean_.shape[0]
            n_classes = len(le.classes_)

            # -------------------------
            # RF / XGB
            # -------------------------
            if model_type in ["RF", "XGB"]:

                model_file = os.path.join(MODELS_BASE_DIR, f"{name}.pkl")

                if os.path.exists(model_file):
                    model = joblib.load(model_file)
                    model = model["model"] if isinstance(model, dict) else model

                    export_rf_header(
                        model,
                        os.path.join(model_dir, f"{name}_model.h"),
                        le.classes_
                    )

                    print("  ✓ RF/XGB")

            # -------------------------
            # CNN1D
            # -------------------------
            if model_type == "CNN1D":

                model_file = os.path.join(MODELS_BASE_DIR, f"{name}.pth")

                if not os.path.exists(model_file):
                    print("missing CNN")
                    continue

                cnn = load_cnn1d(model_file, C, n_classes)

                tflite_path = os.path.join(model_dir, "cnn1d.tflite")
                header_path = os.path.join(model_dir, "cnn1d_model.h")

                export_cnn1d_tflite(
                    cnn,
                    C=C,
                    input_len=90,
                    out_path=tflite_path
                )

                tflite_to_header(tflite_path, header_path)

                print("  ✓ CNN1D TFLite Micro")

            meta = {
                "config": config,
                "type": model_type,
                "C": int(C),
                "classes": list(map(str, le.classes_))
            }

            with open(os.path.join(model_dir, "model_info.json"), "w") as f:
                json.dump(meta, f, indent=2)

    print("\n====================")
    print("EXPORT DONE")
    print("====================")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    export_all_models()