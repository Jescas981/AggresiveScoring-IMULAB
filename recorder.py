import struct
import csv
import os
import time
import threading
import collections
from datetime import datetime
from flask import Flask, jsonify, request, render_template, Response
from flask_cors import CORS
import paho.mqtt.client as mqtt
import subprocess
import cv2

# ── Config ──────────────────────────────────────────────────────────────────
BROKER_IP    = "192.168.100.6"
PORT         = 1883
TOPIC        = "imu/raw"
DATA_DIR     = "data"
CAMERA_INDEX = 1
CAMERA_FPS   = 25
CAMERA_W     = 640
CAMERA_H     = 480

PACKET_RING_SIZE = 6000

os.makedirs(DATA_DIR, exist_ok=True)

# ── Struct format (36 bytes, little-endian) ──────────────────────────────────
# uint32 timestamp_ms | uint32 session_id | uint8 type | uint8 pad[3] | float data[6]
FMT         = "<IIB3x6f"
PACKET_SIZE = struct.calcsize(FMT)   # 36

TYPE_IMU = 0x01
TYPE_GPS = 0x02

assert PACKET_SIZE == 36, f"Expected 36 bytes, got {PACKET_SIZE}"

# ── State ────────────────────────────────────────────────────────────────────
state = {
    "receiving":             False,
    "labeling":              False,
    "label_start":           None,
    "label_name":            None,
    "session_id":            None,
    "packet_count":          0,
    "last_packet":           None,
    "labels":                [],
    # IMU
    "imu_csv_path":          None,
    "imu_csv_file":          None,
    "imu_csv_writer":        None,
    # GPS
    "gps_csv_path":          None,
    "gps_csv_file":          None,
    "gps_csv_writer":        None,
    # Labels
    "labels_path":           None,
    "_labels_file":          None,
    "_labels_writer":        None,
    # Camera
    "cam_path":              None,
    "cam_timestamps_path":   None,
    "cam_frame_count":       0,
}
state_lock = threading.Lock()

# ── Packet ring buffer ───────────────────────────────────────────────────────
_packet_ring = collections.deque(maxlen=PACKET_RING_SIZE)
_packet_seq  = 0
_packet_lock = threading.Lock()

# ── MQTT event log ───────────────────────────────────────────────────────────
_mqtt_events = collections.deque(maxlen=200)
_events_lock = threading.Lock()

def _log_event(level, msg):
    with _events_lock:
        _mqtt_events.append({
            "ts":    int(time.time() * 1000),
            "level": level,
            "msg":   msg,
        })
    print(f"[MQTT/{level.upper()}] {msg}")

# ── Camera ───────────────────────────────────────────────────────────────────
_latest_frame_lock = threading.Lock()
_latest_frame_jpeg = None
_camera_running    = False
_camera_thread     = None

def _get_monotonic_ms():
    return int(time.time() * 1000)

def _camera_loop():
    global _latest_frame_jpeg, _camera_running
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

    if not cap.isOpened():
        print(f"[Camera] ERROR: Could not open camera index {CAMERA_INDEX}")
        _camera_running = False
        return

    print(f"[Camera] Opened {CAMERA_W}x{CAMERA_H} @ {CAMERA_FPS} fps")
    fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
    writer    = None
    ts_file   = None
    ts_csv    = None
    frame_idx = 0

    while _camera_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        now_ms = _get_monotonic_ms()

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with _latest_frame_lock:
                _latest_frame_jpeg = buf.tobytes()

        with state_lock:
            recording = state["receiving"]
            cam_path  = state["cam_path"]
            ts_path   = state["cam_timestamps_path"]

        if recording and cam_path:
            if writer is None:
                writer    = cv2.VideoWriter(cam_path, fourcc, CAMERA_FPS, (CAMERA_W, CAMERA_H))
                ts_file   = open(ts_path, "w", newline="")
                ts_csv    = csv.DictWriter(ts_file, fieldnames=["frame_idx", "timestamp_ms"])
                ts_csv.writeheader()
                frame_idx = 0
                print(f"[Camera] Recording → {cam_path}")
            writer.write(frame)
            ts_csv.writerow({"frame_idx": frame_idx, "timestamp_ms": now_ms})
            ts_file.flush()
            with state_lock:
                state["cam_frame_count"] = frame_idx
            frame_idx += 1

        elif not recording and writer is not None:
            writer.release(); writer = None
            if ts_file: ts_file.close(); ts_file = None
            ts_csv = None; frame_idx = 0
            print("[Camera] Writer closed.")

    if writer:  writer.release()
    if ts_file: ts_file.close()
    cap.release()
    print("[Camera] Thread exited.")

def start_camera():
    global _camera_running, _camera_thread
    if _camera_running:
        return
    _camera_running = True
    _camera_thread  = threading.Thread(target=_camera_loop, daemon=True)
    _camera_thread.start()

# ── CSV helpers ───────────────────────────────────────────────────────────────
IMU_HEADERS = ["timestamp_ms", "session_id",
               "accel_x", "accel_y", "accel_z",
               "gyro_x",  "gyro_y",  "gyro_z"]

GPS_HEADERS = ["timestamp_ms", "session_id",
               "latitude", "longitude",
               "speed_mps", "course_deg",
               "altitude_m", "utc_seconds"]

def open_csv(path, headers):
    f = open(path, "w", newline="")
    w = csv.DictWriter(f, fieldnames=headers)
    w.writeheader()
    return f, w

def open_labels_csv(base_path):
    path = base_path.replace(".csv", "_labels.csv")
    f    = open(path, "w", newline="")
    w    = csv.DictWriter(f, fieldnames=["label", "start_ms", "end_ms", "duration_ms"])
    w.writeheader()
    return path, f, w

def camera_paths(ts):
    return (os.path.join(DATA_DIR, f"video_{ts}.mp4"),
            os.path.join(DATA_DIR, f"video_{ts}_frame_timestamps.csv"))

# ── Packet decoder ────────────────────────────────────────────────────────────
def decode_packet(payload):
    """
    Returns a dict with all fields, or None if the payload is invalid.
    
    Struct layout (36 bytes):
      uint32  timestamp_ms
      uint32  session_id
      uint8   type          (0x01=IMU, 0x02=GPS)
      uint8   _pad[3]       (ignored, consumed by '3x')
      float   data[6]
    
    IMU  → data = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    GPS  → data = [lat, lon, speed_mps, course_deg, altitude_m, utc_seconds]
    """
    if len(payload) != PACKET_SIZE:
        return None

    ts, sid, ptype, d0, d1, d2, d3, d4, d5 = struct.unpack(FMT, payload)

    base = {"timestamp_ms": ts, "session_id": sid, "type": ptype}

    if ptype == TYPE_IMU:
        base.update({
            "accel_x": round(d0, 4), "accel_y": round(d1, 4), "accel_z": round(d2, 4),
            "gyro_x":  round(d3, 4), "gyro_y":  round(d4, 4), "gyro_z":  round(d5, 4),
        })
    elif ptype == TYPE_GPS:
        base.update({
            "latitude":    round(d0, 6),
            "longitude":   round(d1, 6),
            "speed_mps":   round(d2, 3),
            "course_deg":  round(d3, 2),
            "altitude_m":  round(d4, 2),
            "utc_seconds": round(d5, 2),
        })
    else:
        return None  # unknown type

    return base

# ── MQTT callbacks ────────────────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    codes = {0: "OK", 1: "Bad protocol", 2: "Client ID rejected",
             3: "Server unavailable", 4: "Bad credentials", 5: "Not authorized"}
    if rc == 0:
        _log_event("info", f"Connected to {BROKER_IP}:{PORT}")
        client.subscribe(TOPIC)
        _log_event("info", f"Subscribed to {TOPIC}")
    else:
        _log_event("error", f"Connect failed: {codes.get(rc, rc)}")

def on_disconnect(client, userdata, rc):
    if rc == 0:
        _log_event("info", "Disconnected cleanly")
    else:
        _log_event("warn", f"Unexpected disconnect (rc={rc}), reconnecting…")

def on_subscribe(client, userdata, mid, granted_qos):
    _log_event("info", f"Subscription confirmed (mid={mid}, qos={granted_qos})")

def on_message(client, userdata, msg):
    global _packet_seq

    row = decode_packet(msg.payload)
    if row is None:
        _log_event("warn", f"Bad packet: size={len(msg.payload)} (expected {PACKET_SIZE})")
        return

    ptype = row["type"]

    # ── Ring buffer ──
    with _packet_lock:
        row["_seq"] = _packet_seq
        _packet_seq += 1
        _packet_ring.append(row)

    with state_lock:
        state["last_packet"] = row
        state["session_id"]  = row["session_id"]

        if state["receiving"]:
            state["packet_count"] += 1
            clean = {k: v for k, v in row.items() if not k.startswith("_") and k != "type"}

            if ptype == TYPE_IMU and state["imu_csv_writer"]:
                state["imu_csv_writer"].writerow(clean)
                state["imu_csv_file"].flush()

            elif ptype == TYPE_GPS and state["gps_csv_writer"]:
                state["gps_csv_writer"].writerow(clean)
                state["gps_csv_file"].flush()

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route("/status")
def status():
    with state_lock:
        return jsonify({
            "receiving":       state["receiving"],
            "labeling":        state["labeling"],
            "label_name":      state["label_name"],
            "label_start":     state["label_start"],
            "packet_count":    state["packet_count"],
            "imu_csv_path":    state["imu_csv_path"],
            "gps_csv_path":    state["gps_csv_path"],
            "cam_path":        state["cam_path"],
            "cam_frame_count": state["cam_frame_count"],
            "last_packet":     state["last_packet"],
            "labels":          state["labels"],
        })

@app.route("/packets")
def packets():
    since = int(request.args.get("since", -1))
    ptype = request.args.get("type", None)   # optional: "imu" or "gps"
    with _packet_lock:
        result = [p for p in _packet_ring if p["_seq"] > since]
    if ptype == "imu":
        result = [p for p in result if p.get("type") == TYPE_IMU]
    elif ptype == "gps":
        result = [p for p in result if p.get("type") == TYPE_GPS]
    return jsonify(result[-500:])

@app.route("/events")
def events():
    with _events_lock:
        return jsonify(list(_mqtt_events))

@app.route("/start_recording", methods=["POST"])
def start_recording():
    with state_lock:
        if state["receiving"]:
            return jsonify({"error": "Already recording"}), 400

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        session_dir = os.path.join(DATA_DIR, f"session_{ts}")
        os.makedirs(session_dir, exist_ok=True)

        imu_path = os.path.join(session_dir, "imu.csv")
        gps_path = os.path.join(session_dir, "gps.csv")

        imu_f, imu_w = open_csv(imu_path, IMU_HEADERS)
        gps_f, gps_w = open_csv(gps_path, GPS_HEADERS)

        lpath = os.path.join(session_dir, "labels.csv")
        lf = open(lpath, "w", newline="")
        lw = csv.DictWriter(lf, fieldnames=["label", "start_ms", "end_ms", "duration_ms"])
        lw.writeheader()
        # vid_path, fts_path = camera_paths(ts)
        vid_path = os.path.join(session_dir, "video.mp4")
        fts_path = os.path.join(session_dir, "video_frame_timestamps.csv")

        state.update({
            "receiving":           True,
            "packet_count":        0,
            "labels":              [],
            "cam_frame_count":     0,
            # IMU
            "imu_csv_path":        imu_path,
            "imu_csv_file":        imu_f,
            "imu_csv_writer":      imu_w,
            # GPS
            "gps_csv_path":        gps_path,
            "gps_csv_file":        gps_f,
            "gps_csv_writer":      gps_w,
            # Labels
            "labels_path":         lpath,
            "_labels_file":        lf,
            "_labels_writer":      lw,
            # Camera
            "cam_path":            vid_path,
            "cam_timestamps_path": fts_path,
            "session_dir": session_dir,
        })

    _log_event("info", f"Recording started → IMU:{imu_path}  GPS:{gps_path}")
    return jsonify({"status": "recording", "imu_csv": imu_path,
                    "gps_csv": gps_path, "video": vid_path})

@app.route("/stop_recording", methods=["POST"])
def stop_recording():
    with state_lock:
        if not state["receiving"]:
            return jsonify({"error": "Not recording"}), 400

        if state["labeling"] and state["last_packet"]:
            _close_label(state["last_packet"]["timestamp_ms"])

        video_path = state["cam_path"]
        state["receiving"]           = False
        state["cam_path"]            = None
        state["cam_timestamps_path"] = None

        for key in ("imu_csv_file", "gps_csv_file", "_labels_file"):
            if state.get(key):
                state[key].close()
                state[key] = None

        state["imu_csv_writer"] = state["gps_csv_writer"] = state["_labels_writer"] = None

    _log_event("info", f"Recording stopped ({state['packet_count']} packets)")

    # ── Convert to H264 ──
    if video_path and os.path.exists(video_path):
        output_path = video_path.replace(".mp4", "_h264.mp4")
        cmd = ["ffmpeg", "-y", "-i", video_path,
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", output_path]
        try:
            subprocess.run(cmd, check=True)
            os.replace(output_path, video_path)
            print(f"[FFMPEG] Converted → {video_path}")
        except Exception as e:
            print(f"[FFMPEG] ERROR: {e}")

    return jsonify({"status": "stopped", "packets": state["packet_count"],
                    "video": video_path})

@app.route("/start_label", methods=["POST"])
def start_label():
    data  = request.json or {}
    label = data.get("label", "unknown").strip()
    with state_lock:
        if not state["receiving"]:
            return jsonify({"error": "Not recording"}), 400
        if state["labeling"]:
            return jsonify({"error": "Label window already open"}), 400
        ts = state["last_packet"]["timestamp_ms"] if state["last_packet"] else _get_monotonic_ms()
        state.update({"labeling": True, "label_start": ts, "label_name": label})
    return jsonify({"status": "labeling", "label": label, "start_ms": ts})

@app.route("/stop_label", methods=["POST"])
def stop_label():
    with state_lock:
        if not state["labeling"]:
            return jsonify({"error": "No label window open"}), 400
        ts    = state["last_packet"]["timestamp_ms"] if state["last_packet"] else _get_monotonic_ms()
        entry = _close_label(ts)
    return jsonify({"status": "label_saved", "entry": entry})

def _close_label(end_ms):
    entry = {
        "label":       state["label_name"],
        "start_ms":    state["label_start"],
        "end_ms":      end_ms,
        "duration_ms": end_ms - state["label_start"],
    }
    state["labels"].append(entry)
    if state.get("_labels_writer"):
        state["_labels_writer"].writerow(entry)
        state["_labels_file"].flush()
    state.update({"labeling": False, "label_start": None, "label_name": None})
    return entry

@app.route("/labels")
def get_labels():
    with state_lock:
        return jsonify(state["labels"])

@app.route("/sessions")
def sessions():
    sessions = []

    if not os.path.exists(DATA_DIR):
        return jsonify([])

    for folder in sorted(os.listdir(DATA_DIR), reverse=True):
        session_path = os.path.join(DATA_DIR, folder)

        if not os.path.isdir(session_path):
            continue

        files = {
            "session_id": folder,
            "imu": None,
            "gps": None,
            "video": None,
            "frame_ts": None,
            "labels": None,
        }

        for fname in os.listdir(session_path):
            fpath = os.path.join(session_path, fname)

            if fname == "imu.csv":
                files["imu"] = fpath
            elif fname == "gps.csv":
                files["gps"] = fpath
            elif fname == "video.mp4":
                files["video"] = fpath
            elif fname == "video_frame_timestamps.csv":
                files["frame_ts"] = fpath
            elif fname == "labels.csv":
                files["labels"] = fpath

        sessions.append(files)

    return jsonify(sessions)

@app.route("/data/<path:filename>")
def serve_data(filename):
    from flask import send_from_directory
    return send_from_directory(os.path.abspath(DATA_DIR), filename)

def _mjpeg_generator():
    while True:
        with _latest_frame_lock:
            frame = _latest_frame_jpeg
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(1 / CAMERA_FPS)

@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    start_camera()
    client = mqtt.Client()
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_subscribe  = on_subscribe
    client.on_message    = on_message
    client.connect(BROKER_IP, PORT, 60)
    threading.Thread(target=client.loop_forever, daemon=True).start()
    print(f"[Server] Packet size: {PACKET_SIZE} bytes (TYPE_IMU=0x{TYPE_IMU:02X}, TYPE_GPS=0x{TYPE_GPS:02X})")
    print("[Server] Running on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
