# run_benchmark.py
import serial
import numpy as np
import pandas as pd
import os
import glob
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# CONFIGURACIÓN DE RUTAS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_BASE_DIR = os.path.join(BASE_DIR, "model_input", "datasets")
EXPORT_DIR = os.path.join(BASE_DIR, f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

PORT = "/dev/ttyACM0"
BAUDRATE = 115200

print(f"📁 Base directory: {BASE_DIR}")
print(f"📁 Datasets directory: {DATASET_BASE_DIR}")
print(f"📁 Existe: {os.path.exists(DATASET_BASE_DIR)}")

# =========================================================
# MAPEO DE MODELOS A SUS DATASETS DE PRUEBA
# =========================================================

MODEL_DATASETS = {
    "FULL_IMU_LPF_RF":      "FULL_IMU_LPF_RF/validation_dataset.csv",
    "SELECTED_IMU_RF":      "SELECTED_IMU_RF/validation_dataset.csv",
    "SELECTED_GPS_RF":      "SELECTED_GPS_RF/validation_dataset.csv",
    "IMU_WITH_GPS_RF":      "IMU_WITH_GPS_RF/validation_dataset.csv",
    "FULL_IMU_GPS_RF":      "FULL_IMU_GPS_RF/validation_dataset.csv",

    "FULL_IMU_LPF_XGB":     "FULL_IMU_LPF_XGB/validation_dataset.csv",
    "SELECTED_IMU_XGB":     "SELECTED_IMU_XGB/validation_dataset.csv",
    "SELECTED_GPS_XGB":     "SELECTED_GPS_XGB/validation_dataset.csv",
    "IMU_WITH_GPS_XGB":     "IMU_WITH_GPS_XGB/validation_dataset.csv",
    "FULL_IMU_GPS_XGB":     "FULL_IMU_GPS_XGB/validation_dataset.csv",

    "FULL_IMU_LPF_ANN_FEAT": "FULL_IMU_LPF_ANN_FEAT/validation_dataset.csv",
    "SELECTED_IMU_ANN_FEAT": "SELECTED_IMU_ANN_FEAT/validation_dataset.csv",
    "SELECTED_GPS_ANN_FEAT": "SELECTED_GPS_ANN_FEAT/validation_dataset.csv",
    "IMU_WITH_GPS_ANN_FEAT": "IMU_WITH_GPS_ANN_FEAT/validation_dataset.csv",
    "FULL_IMU_GPS_ANN_FEAT": "FULL_IMU_GPS_ANN_FEAT/validation_dataset.csv",
}


# =========================================================
# DETECCIÓN DE MODELO
# =========================================================

def get_model_from_arduino(port=PORT, baud=BAUDRATE):
    """Detecta el modelo actual desde Arduino enviando comando INFO"""
    try:
        ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        print("  Enviando INFO...")
        ser.write(b"INFO\n")
        ser.flush()
        time.sleep(0.5)

        model_name = None
        board_info = {}
        start = time.time()

        while (time.time() - start) < 3:
            if ser.in_waiting:
                line = ser.readline().decode().strip()
                if line:
                    print(f"    Recibido: {line}")
                if line.startswith("Model:"):
                    model_name = line.replace("Model:", "").strip()
                elif line.startswith("Features:"):
                    board_info["features"] = line.replace("Features:", "").strip()
                elif line.startswith("Classes:"):
                    board_info["classes"] = line.replace("Classes:", "").strip()
                elif line.startswith("Free RAM:"):
                    board_info["free_ram"] = int(line.replace("Free RAM:", "").strip())
                elif line.startswith("Min Free RAM:"):
                    board_info["min_free_ram"] = int(line.replace("Min Free RAM:", "").strip())
                elif line.startswith("Largest free block:"):
                    board_info["largest_block"] = int(line.replace("Largest free block:", "").strip())
                elif line == "READY":
                    break
            time.sleep(0.05)

        ser.close()

        if board_info:
            print("\n  📋 Info del board:")
            for k, v in board_info.items():
                label = {
                    "features":     "  Features",
                    "classes":      "  Classes",
                    "free_ram":     "  Free RAM",
                    "min_free_ram": "  Min Free RAM (histórico)",
                    "largest_block":"  Largest free block",
                }.get(k, f"  {k}")
                unit = " bytes" if "ram" in k or "block" in k else ""
                print(f"{label}: {v}{unit}")

        return model_name, board_info

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None, {}


def find_dataset_for_model(model_name):
    """Encuentra el dataset correspondiente al modelo"""
    if model_name in MODEL_DATASETS:
        dataset_path = os.path.join(DATASET_BASE_DIR, MODEL_DATASETS[model_name])
        if os.path.exists(dataset_path):
            print(f"  ✅ Encontrado: {dataset_path}")
            return dataset_path
        else:
            print(f"  ❌ No existe: {dataset_path}")

    pattern = f"*{model_name}*.csv"
    files = glob.glob(os.path.join(DATASET_BASE_DIR, pattern))
    if files:
        print(f"  ✅ Encontrado por patrón: {files[0]}")
        return files[0]

    print(f"  ❌ No se encontró dataset para {model_name}")
    return None


def list_available_datasets():
    """Lista todos los datasets disponibles"""
    csv_files = glob.glob(os.path.join(DATASET_BASE_DIR, "*dataset.csv"))
    print("\n📂 Datasets disponibles:")
    for f in csv_files:
        name = os.path.basename(f)
        for model, dataset in MODEL_DATASETS.items():
            if dataset == name:
                print(f"  {name} -> {model}")
                break
        else:
            print(f"  {name}")


# =========================================================
# PLOTS
# =========================================================

def plot_confusion_matrix(y_true, y_pred, class_names, model_name, output_dir):
    """Genera y guarda la matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"  📊 Confusion matrix saved: {output_path}")
    return cm


def plot_ram_usage(df_ok, model_name, output_dir):
    """
    Genera dos gráficas de RAM:
      1. RAM libre a lo largo de las inferencias
      2. Histograma del delta de RAM por inferencia
    """
    if 'ram_antes' not in df_ok.columns or df_ok['ram_antes'].eq(-1).all():
        print("  ⚠️  Sin datos de RAM (firmware antiguo o placa sin soporte)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'RAM Usage During Inference - {model_name}', fontsize=13)

    # ── Gráfica 1: RAM libre a lo largo del tiempo ──────────────────────────
    axes[0].plot(df_ok['sample'], df_ok['ram_antes'], color='steelblue', linewidth=0.8)
    axes[0].set_title('Free RAM per Inference')
    axes[0].set_xlabel('Sample index')
    axes[0].set_ylabel('Free RAM (bytes)')
    axes[0].grid(True, alpha=0.3)

    # ── Gráfica 2: Histograma de delta RAM ───────────────────────────────────
    deltas = df_ok['ram_delta'].values
    axes[1].hist(deltas, bins=30, color='salmon', edgecolor='white')
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_title('RAM Delta per Inference\n(>0 = consumed, 0 = no alloc)')
    axes[1].set_xlabel('RAM delta (bytes)')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'ram_usage_{model_name}.png')
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"  📊 RAM plot saved: {output_path}")


# =========================================================
# BENCHMARK PRINCIPAL
# =========================================================

def run_benchmark(input_csv=None, model_name=None, port=PORT, baud=BAUDRATE, output_csv=None):
    """Ejecuta benchmark del modelo en Arduino"""

    os.makedirs(EXPORT_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("BENCHMARK DE MODELO EN ARDUINO")
    print("="*60)

    # ── Determinar dataset ───────────────────────────────────────────────────
    board_info = {}

    if input_csv is None:
        if model_name is None:
            print("\n🔍 Detectando modelo en Arduino...")
            model_name, board_info = get_model_from_arduino(port, baud)
        
        if model_name:
            print(f"\n✅ Modelo detectado: {model_name}")
            input_csv = find_dataset_for_model(model_name)
            if input_csv:
                print(f"📁 Dataset: {os.path.basename(input_csv)}")

        if input_csv is None:
            print("\n📋 Selecciona el modelo que estás probando:")
            models = list(MODEL_DATASETS.keys())
            for i, m in enumerate(models):
                print(f"  {i+1}. {m}")
            try:
                choice = int(input("\nSelecciona (número): ")) - 1
                if 0 <= choice < len(models):
                    model_name = models[choice]
                    input_csv = os.path.join(DATASET_BASE_DIR, MODEL_DATASETS[model_name])
                    print(f"✅ Usando modelo: {model_name}")
                    print(f"📁 Dataset: {os.path.basename(input_csv)}")
                else:
                    print("❌ Selección inválida")
                    return None
            except ValueError:
                print("❌ Entrada inválida")
                return None

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Archivo no encontrado: {input_csv}")

    # ── Cargar CSV ───────────────────────────────────────────────────────────
    df_input = pd.read_csv(input_csv)
    exclude_cols = ['label', 'label_name', 'Unnamed: 0']
    feature_cols = [c for c in df_input.columns if c not in exclude_cols]
    X = df_input[feature_cols].values.astype(np.float32)

    if 'label' in df_input.columns:
        y_true = df_input['label'].values
        if 'label_name' in df_input.columns:
            class_names = sorted(df_input['label_name'].unique(),
                                 key=lambda x: df_input[df_input['label_name'] == x]['label'].iloc[0])
        else:
            class_names = [str(i) for i in np.unique(y_true)]
    else:
        y_true = None
        class_names = None

    print(f"\n📊 Dataset info:")
    print(f"   Archivo:  {os.path.basename(input_csv)}")
    print(f"   Muestras: {X.shape[0]}")
    print(f"   Features: {X.shape[1]}")
    if class_names:
        print(f"   Clases:   {class_names}")

    results = []

    # ── Serial ───────────────────────────────────────────────────────────────
    try:
        ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)
        ser.reset_input_buffer()

        print("\n🔌 Conectado")
        total_samples = len(X)
        print(f"\n🚀 Ejecutando benchmark sobre {total_samples} muestras...\n")

        for i, row in enumerate(X):
            csv_line = ",".join(f"{v:.6f}" for v in row) + "\n"
            ser.write(csv_line.encode())
            resp = ser.readline().decode().strip()

            base = {
                "sample":      i,
                "prediccion":  -1,
                "nombre_clase": "TIMEOUT",
                "tiempo_us":   -1,
                "ram_antes":   -1,
                "ram_delta":   -1,
                "esperado":    y_true[i] if y_true is not None else -1,
            }

            if not resp:
                results.append(base)
                continue

            if resp.startswith("ERROR"):
                base["nombre_clase"] = "ERROR"
                results.append(base)
                continue

            parts = resp.split(",")

            # Firmware nuevo: clase,nombre,tiempo_us,ram_antes,ram_delta
            if len(parts) == 5:
                base.update({
                    "prediccion":   int(parts[0]),
                    "nombre_clase": parts[1],
                    "tiempo_us":    int(parts[2]),
                    "ram_antes":    int(parts[3]),
                    "ram_delta":    int(parts[4]),
                })
            # Firmware anterior: clase,nombre,tiempo_us
            elif len(parts) == 3:
                base.update({
                    "prediccion":   int(parts[0]),
                    "nombre_clase": parts[1],
                    "tiempo_us":    int(parts[2]),
                })
            elif len(parts) == 2:
                base.update({
                    "prediccion":   int(parts[0]),
                    "tiempo_us":    int(parts[1]),
                    "nombre_clase": "UNKNOWN",
                })

            results.append(base)

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{total_samples}] procesadas...")

        ser.close()

    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return None

    # ── Guardar resultados ───────────────────────────────────────────────────
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_csv:
        output_file = os.path.join(EXPORT_DIR, output_csv)
    else:
        base_name = os.path.splitext(os.path.basename(input_csv))[0]
        output_file = os.path.join(EXPORT_DIR, f"results_{base_name}_{timestamp}.csv")

    df.to_csv(output_file, index=False)

    df_ok = df[df["prediccion"] != -1]

    # ── Stats de tiempos ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTADOS DEL BENCHMARK - TIEMPOS")
    print("="*60)
    print(f"Muestras válidas:   {len(df_ok)}/{len(df)}")

    if len(df_ok) > 0:
        tiempos = df_ok['tiempo_us'].values
        print(f"Tiempo promedio:    {np.mean(tiempos):.1f} us")
        print(f"Tiempo máximo:      {np.max(tiempos)} us")
        print(f"Tiempo mínimo:      {np.min(tiempos)} us")
        print(f"Frecuencia:         {1e6/np.mean(tiempos):.1f} Hz")

    # ── Stats de RAM ─────────────────────────────────────────────────────────
    has_ram = 'ram_antes' in df_ok.columns and not df_ok['ram_antes'].eq(-1).all()

    if has_ram:
        print("\n" + "="*60)
        print("RESULTADOS DEL BENCHMARK - RAM")
        print("="*60)

        if board_info.get("free_ram"):
            print(f"Free RAM (boot):     {board_info['free_ram']} bytes")
        if board_info.get("min_free_ram"):
            print(f"Min Free RAM (boot): {board_info['min_free_ram']} bytes")
        if board_info.get("largest_block"):
            print(f"Largest block (boot):{board_info['largest_block']} bytes")

        ram_vals  = df_ok['ram_antes'].values
        ram_deltas = df_ok['ram_delta'].values

        print(f"\nRAM libre (inferencias):")
        print(f"  Promedio:  {np.mean(ram_vals):.0f} bytes")
        print(f"  Mínimo:    {np.min(ram_vals)} bytes")
        print(f"  Máximo:    {np.max(ram_vals)} bytes")

        print(f"\nDelta RAM por inferencia (consumo dinámico):")
        print(f"  Promedio:  {np.mean(ram_deltas):.1f} bytes")
        print(f"  Máximo:    {np.max(ram_deltas)} bytes")
        print(f"  Mínimo:    {np.min(ram_deltas)} bytes")

        leaks = (ram_deltas > 0).sum()
        if leaks == 0:
            print(f"\n  ✅ Sin allocations dinámicas detectadas (delta siempre 0)")
        else:
            print(f"\n  ⚠️  {leaks}/{len(ram_deltas)} inferencias con delta > 0 (posible malloc interno)")

        plot_ram_usage(df_ok, model_name or "modelo", EXPORT_DIR)

    # ── Métricas de clasificación ────────────────────────────────────────────
    if y_true is not None and len(df_ok) > 0:
        y_true_valid = df_ok['esperado'].values
        y_pred_valid = df_ok['prediccion'].values

        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_true_valid, y_pred_valid, target_names=class_names))

        accuracy  = accuracy_score(y_true_valid, y_pred_valid)
        precision = precision_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)
        recall    = recall_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)
        f1        = f1_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)

        print("-"*60)
        print("MACRO AVERAGE:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print("="*60)

        plot_confusion_matrix(y_true_valid, y_pred_valid, class_names,
                              model_name or "modelo", EXPORT_DIR)

        metrics_row = {
            'modelo':           model_name,
            'accuracy':         accuracy,
            'precision_macro':  precision,
            'recall_macro':     recall,
            'f1_macro':         f1,
            'tiempo_us_mean':   np.mean(tiempos) if len(df_ok) > 0 else -1,
            'tiempo_us_max':    np.max(tiempos)  if len(df_ok) > 0 else -1,
            'timestamp':        timestamp,
        }

        # Agregar RAM a métricas si disponible
        if has_ram:
            metrics_row.update({
                'ram_libre_mean':  np.mean(ram_vals),
                'ram_libre_min':   np.min(ram_vals),
                'ram_delta_mean':  np.mean(ram_deltas),
                'ram_delta_max':   np.max(ram_deltas),
            })

        metrics_df = pd.DataFrame([metrics_row])
        metrics_file = os.path.join(EXPORT_DIR, f"metrics_{model_name}_{timestamp}.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"\n📊 Metrics saved: {metrics_file}")

    print(f"\n💾 Results saved in: {EXPORT_DIR}/")
    return df


# =========================================================
# MAIN
# =========================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark de modelo en Arduino')
    parser.add_argument('--input',  '-i', type=str, help='Archivo CSV de entrada')
    parser.add_argument('--model',  '-m', type=str, help='Nombre del modelo (ej: IMU_WITH_GPS_RF)')
    parser.add_argument('--port',   '-p', type=str, default=PORT, help='Puerto serial')
    parser.add_argument('--output', '-o', type=str, help='Archivo de salida')
    parser.add_argument('--list',   '-l', action='store_true', help='Listar datasets disponibles')
    parser.add_argument('--detect', '-d', action='store_true', help='Detectar modelo desde Arduino')

    args = parser.parse_args()

    if args.list:
        list_available_datasets()
        return

    if args.detect:
        print("\n🔍 Detectando modelo...")
        model, info = get_model_from_arduino(args.port, BAUDRATE)
        if model:
            print(f"\n✅ Modelo detectado: {model}")
        else:
            print("\n❌ No se pudo detectar el modelo")
        return

    run_benchmark(
        input_csv=args.input,
        model_name=args.model,
        port=args.port,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()