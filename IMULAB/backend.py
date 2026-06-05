import struct
import csv
import os
import time
import threading
import collections
from datetime import datetime
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import paho.mqtt.client as mqtt
import subprocess
import cv2

# ── Config ───────────────────────────────────────────────────────────────────
BROKER_IP    = "192.168.100.6"
PORT         = 1883
TOPIC        = "imu/raw"
DATA_DIR     = "data"
CAMERA_INDEX = 0
CAMERA_FPS   = 25
CAMERA_W     = 1920
CAMERA_H     = 1080

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
    "session_id":            None,
    "packet_count":          0,
    "last_packet":           None,
    # IMU
    "imu_csv_path":          None,
    "imu_csv_file":          None,
    "imu_csv_writer":        None,
    # GPS
    "gps_csv_path":          None,
    "gps_csv_file":          None,
    "gps_csv_writer":        None,
    # Camera
    "cam_path":              None,
    "cam_timestamps_path":   None,
    "cam_frame_count":       0,
    "session_dir":           None,
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
# Preview is throttled: every PREVIEW_EVERY frames we resize to PREVIEW_W
# and encode a JPEG — keeps CPU cost negligible at 1080p.
# The VideoWriter always gets the full-res frame.

PREVIEW_EVERY = 5          # encode preview every N capture frames (~5 fps preview at 25fps)
PREVIEW_W     = 480        # preview width; height computed from aspect ratio

_camera_running    = False
_camera_thread     = None
_latest_frame_jpeg = None
_latest_frame_lock = threading.Lock()

# Live camera stats (read by /status)
_cam_stats      = {"fps": 0.0, "recording": False}
_cam_stats_lock = threading.Lock()

def _get_monotonic_ms():
    return int(time.time() * 1000)

def _camera_loop():
    global _camera_running, _latest_frame_jpeg
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # drain stale frames — key fix for 1080p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

    if not cap.isOpened():
        print(f"[Camera] ERROR: Could not open camera index {CAMERA_INDEX}")
        _camera_running = False
        return
    
    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    prev_h     = int(PREVIEW_W * actual_h / actual_w)
    print(f"[Camera] Opened {actual_w}x{actual_h} @ {actual_fps:.1f} fps  "
          f"(preview {PREVIEW_W}x{prev_h} every {PREVIEW_EVERY} frames)")

    fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
    writer    = None
    ts_file   = None
    ts_csv    = None
    frame_idx = 0

    # FPS measurement
    fps_counter = 0
    fps_ts      = time.time()

    while _camera_running:
        ret, frame = cap.read()
        if not ret:
            continue   # keep draining without sleeping

        now_ms = _get_monotonic_ms()

        # ── FPS counter ──
        fps_counter += 1
        now_t = time.time()
        if now_t - fps_ts >= 1.0:
            measured = fps_counter / (now_t - fps_ts)
            fps_counter = 0
            fps_ts = now_t
            with _cam_stats_lock:
                _cam_stats["fps"] = round(measured, 1)

        # ── Throttled preview encode (cheap: small resize before JPEG) ──
        if frame_idx % PREVIEW_EVERY == 0:
            small = cv2.resize(frame, (PREVIEW_W, prev_h), interpolation=cv2.INTER_LINEAR)
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with _latest_frame_lock:
                    _latest_frame_jpeg = buf.tobytes()

        with state_lock:
            recording = state["receiving"]
            cam_path  = state["cam_path"]
            ts_path   = state["cam_timestamps_path"]

        if recording and cam_path:
            if writer is None:
                writer    = cv2.VideoWriter(cam_path, fourcc, CAMERA_FPS, (actual_w, actual_h))
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
            with _cam_stats_lock:
                _cam_stats["recording"] = True
            frame_idx += 1

        elif not recording and writer is not None:
            writer.release(); writer = None
            if ts_file: ts_file.close(); ts_file = None
            ts_csv = None; frame_idx = 0
            with _cam_stats_lock:
                _cam_stats["recording"] = False
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

# ── Packet decoder ────────────────────────────────────────────────────────────
def decode_packet(payload):
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
        return None

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

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route("/status")
def status():
    with state_lock:
        with _cam_stats_lock:
            cs = dict(_cam_stats)
        return jsonify({
            "receiving":       state["receiving"],
            "packet_count":    state["packet_count"],
            "imu_csv_path":    state["imu_csv_path"],
            "gps_csv_path":    state["gps_csv_path"],
            "cam_path":        state["cam_path"],
            "cam_frame_count": state["cam_frame_count"],
            "cam_fps":         cs["fps"],
            "cam_recording":   cs["recording"],
            "last_packet":     state["last_packet"],
            "session_dir":     state["session_dir"],
        })

@app.route("/packets")
def packets():
    since = int(request.args.get("since", -1))
    ptype = request.args.get("type", None)
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

        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(DATA_DIR, f"session_{ts}")
        os.makedirs(session_dir, exist_ok=True)

        imu_path = os.path.join(session_dir, "imu.csv")
        gps_path = os.path.join(session_dir, "gps.csv")
        vid_path = os.path.join(session_dir, "video.mp4")
        fts_path = os.path.join(session_dir, "video_frame_timestamps.csv")

        imu_f, imu_w = open_csv(imu_path, IMU_HEADERS)
        gps_f, gps_w = open_csv(gps_path, GPS_HEADERS)

        state.update({
            "receiving":           True,
            "packet_count":        0,
            "cam_frame_count":     0,
            "imu_csv_path":        imu_path,
            "imu_csv_file":        imu_f,
            "imu_csv_writer":      imu_w,
            "gps_csv_path":        gps_path,
            "gps_csv_file":        gps_f,
            "gps_csv_writer":      gps_w,
            "cam_path":            vid_path,
            "cam_timestamps_path": fts_path,
            "session_dir":         session_dir,
        })

    _log_event("info", f"Recording started → {session_dir}")
    return jsonify({"status": "recording", "session_dir": session_dir,
                    "imu_csv": imu_path, "gps_csv": gps_path, "video": vid_path})

@app.route("/stop_recording", methods=["POST"])
def stop_recording():
    with state_lock:
        if not state["receiving"]:
            return jsonify({"error": "Not recording"}), 400

        video_path             = state["cam_path"]
        state["receiving"]     = False
        state["cam_path"]      = None
        state["cam_timestamps_path"] = None

        for key in ("imu_csv_file", "gps_csv_file"):
            if state.get(key):
                state[key].close()
                state[key] = None
        state["imu_csv_writer"] = state["gps_csv_writer"] = None

    pkt_count = state["packet_count"]
    _log_event("info", f"Recording stopped ({pkt_count} packets)")

    # ── Convert to H.264 ──
    if video_path and os.path.exists(video_path):
        output_path = video_path.replace(".mp4", "_h264.mp4")
        cmd = ["ffmpeg", "-y", "-i", video_path,
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", output_path]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.replace(output_path, video_path)
            print(f"[FFMPEG] Converted → {video_path}")
        except Exception as e:
            print(f"[FFMPEG] ERROR: {e}")

    return jsonify({"status": "stopped", "packets": pkt_count,
                    "video": video_path})

def _mjpeg_generator():
    while True:
        with _latest_frame_lock:
            frame = _latest_frame_jpeg
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(1 / (CAMERA_FPS / PREVIEW_EVERY))   # pace to preview rate

@app.route("/video_feed")
def video_feed():
    from flask import Response
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
    print(f"[Server] Packet size: {PACKET_SIZE} bytes")
    print("[Server] Running on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)