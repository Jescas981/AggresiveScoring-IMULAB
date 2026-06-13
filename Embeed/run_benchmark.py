import serial
import numpy as np
import os
import time
import struct
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# CONFIG
# =========================================================
PORT = "/dev/ttyACM0"
BAUDRATE = 115200

MAGIC = 0xAA

CMD_INFO = 0x01
CMD_DATA = 0x02

RESP_INFO = 0x11
RESP_DATA = 0x12
RESP_LOG  = 0x13


# =========================================================
# REQUEST INFO
# =========================================================
def request_info(ser):
    ser.reset_input_buffer()
    ser.write(struct.pack("<BB", MAGIC, CMD_INFO))
    ser.flush()


# =========================================================
# READ INFO PACKET
# =========================================================
def read_info(ser):

    while True:
        b = ser.read(1)

        if len(b) == 0:
            return None

        if b[0] == MAGIC:
            break

    cmd = ser.read(1)

    if len(cmd) == 0:
        return None

    if cmd[0] != RESP_INFO:
        return None

    ln = ser.read(1)

    if len(ln) == 0:
        return None

    name_len = ln[0]

    name = ser.read(name_len)

    if len(name) != name_len:
        return None

    meta = ser.read(2)

    if len(meta) != 2:
        return None

    n_features, n_classes = meta[0], meta[1]

    return {
        "name": name.decode(errors="ignore"),
        "n_features": n_features,
        "n_classes": n_classes
    }


# =========================================================
# READ RESPONSE (DATA + LOGS)
# =========================================================
def read_response(ser):

    try:

        while True:

            # ---------------------------------
            # WAIT MAGIC
            # ---------------------------------
            b = ser.read(1)

            if len(b) == 0:
                return None

            if b[0] != MAGIC:
                continue

            # ---------------------------------
            # READ COMMAND
            # ---------------------------------
            cmd = ser.read(1)

            if len(cmd) == 0:
                return None

            cmd = cmd[0]

            # =================================
            # LOG PACKET
            # =================================
            if cmd == RESP_LOG:

                hdr = ser.read(2)

                if len(hdr) != 2:
                    continue

                msg_len = struct.unpack("<H", hdr)[0]

                msg = ser.read(msg_len)

                if len(msg) != msg_len:
                    continue

                try:
                    txt = msg.decode("utf-8")
                except Exception:
                    txt = str(msg)

                print(f"[ESP] {txt}")

                # seguir esperando RESP_DATA
                continue

            # =================================
            # DATA PACKET
            # =================================
            if cmd == RESP_DATA:

                payload = ser.read(8)

                if len(payload) != 8:
                    return None

                cls, dt = struct.unpack("<ii", payload)

                return cls

            # =================================
            # UNKNOWN PACKET
            # =================================
            print(f"[WARN] Unknown packet: 0x{cmd:02X}")

    except Exception as e:
        print("read_response error:", e)
        return None


# =========================================================
# DATASET DISCOVERY
# =========================================================
def discover_datasets(base_dir):

    datasets = {}

    for name in os.listdir(base_dir):

        path = os.path.join(base_dir, name)

        if not os.path.isdir(path):
            continue

        files = os.listdir(path)

        if (
            any(f.startswith("X_") for f in files)
            and
            any(f.startswith("y_") for f in files)
        ):
            datasets[name] = path

    return datasets


# =========================================================
# LOAD DATASET
# =========================================================
def load_dataset(folder):

    x_path = os.path.join(folder, "X_test.npy")
    y_path = os.path.join(folder, "y_test.npy")

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path)

    return X, y


# =========================================================
# FLATTEN
# =========================================================
def flatten(X):
    return X.reshape(X.shape[0], -1).astype(np.float32)


# =========================================================
# SEND SAMPLE
# =========================================================
def send_row(ser, row):

    row = np.asarray(row, dtype=np.float32)

    n = len(row)

    packet = struct.pack(
        "<BBH",
        MAGIC,
        CMD_DATA,
        n
    )

    packet += row.tobytes()

    ser.write(packet)


# =========================================================
# MAIN
# =========================================================
def run():

    DATASET_BASE_DIR = "./model_input/datasets"

    datasets = discover_datasets(DATASET_BASE_DIR)

    ser = serial.Serial(
        PORT,
        BAUDRATE,
        timeout=1
    )

    time.sleep(2)

    # -----------------------------------------------------
    # MODEL INFO
    # -----------------------------------------------------
    request_info(ser)

    model_info = None

    start = time.time()

    while time.time() - start < 5:

        model_info = read_info(ser)

        if model_info is not None:
            break

    print("MODEL INFO:", model_info)

    if model_info is None:
        raise RuntimeError(
            "Failed to read model info"
        )

    # -----------------------------------------------------
    # DATASET
    # -----------------------------------------------------
    if model_info["name"] not in datasets:
        raise ValueError(
            f"No dataset for {model_info['name']}"
        )

    X, y = load_dataset(
        datasets[model_info["name"]]
    )

    # X = np.transpose(X, (0, 2, 1))

    X = flatten(X)

    print("Dataset shape:", X.shape)

    # -----------------------------------------------------
    # INFERENCE
    # -----------------------------------------------------
    results = []
    times = []

    print("\n🚀 RUNNING...\n")

    for i in range(len(X)):

        t0 = time.perf_counter()

        send_row(ser, X[i])
        pred = read_response(ser)

        t1 = time.perf_counter()

        dt_ms = (t1 - t0) * 1000.0
        times.append(dt_ms)

        print(f"{i:5d} | true={y[i]} pred={pred} | {dt_ms:.2f} ms")

        results.append({
            "true": int(y[i]),
            "pred": int(pred) if pred is not None else -1
        })

    ser.close()

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------
    df = pd.DataFrame(results)
    df_ok = df[df["pred"] != -1]

    print("\n================ METRICS ================")

    if len(df_ok) == 0:
        print("❌ No valid predictions")
        return

    print("Accuracy :", accuracy_score(df_ok["true"], df_ok["pred"]))

    print("Precision:", precision_score(df_ok["true"], df_ok["pred"],
                                        average="macro", zero_division=0))

    print("Recall   :", recall_score(df_ok["true"], df_ok["pred"],
                                    average="macro", zero_division=0))

    print("F1       :", f1_score(df_ok["true"], df_ok["pred"],
                                average="macro", zero_division=0))

    # =========================================================
    # CONFUSION MATRIX
    # =========================================================
    labels = ["Frenado", "Giro", "Normal", "Resalto"]

    cm_norm = confusion_matrix(df_ok["true"], df_ok["pred"], normalize='true')

    # ---- normalized
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title("Matriz de confusión")
    plt.xlabel("Predecido")
    plt.ylabel("Verdadero")

    plt.suptitle(model_info["name"])
    plt.tight_layout()
    plt.show()

    # =========================================================
    # TIMING METRICS
    # =========================================================
    times = np.array(times)

    print("\n================ TIMING ================")
    print(f"Avg latency : {times.mean():.2f} ms")
    print(f"Min latency : {times.min():.2f} ms")
    print(f"Max latency : {times.max():.2f} ms")
    print(f"Std dev     : {times.std():.2f} ms")

    from sklearn.metrics import classification_report

    print("\n================ CLASS REPORT ================\n")

    print(classification_report(
        df_ok["true"],
        df_ok["pred"],
        target_names=["Frenado", "Giro", "Normal", "Resalto"],
        zero_division=0
    ))

if __name__ == "__main__":
    run()