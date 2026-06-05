import struct
import os
import time
import threading
import collections
from flask import Flask, jsonify, render_template_string, Response
from flask_cors import CORS
import paho.mqtt.client as mqtt
import cv2

BROKER_IP = "192.168.100.6"
PORT = 1883

TOPIC_MEAN_IMU = "imu/mean"
TOPIC_EVENT_SCORE = "score/event"
TOPIC_SESSION_SCORE = "score/session"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
os.makedirs(TEMPLATES_DIR, exist_ok=True)

FMT_IMU = "<IIB3x6f"
PACKET_IMU = struct.calcsize(FMT_IMU)
FMT_EVENT_SCORE = "<IIB4f"
PACKET_EVENT = struct.calcsize(FMT_EVENT_SCORE)
FMT_SESSION_SCORE = "<IIfIII"
PACKET_SESSION = struct.calcsize(FMT_SESSION_SCORE)

EVENT_TYPES = {
    0: "frenado", 1: "giro", 2: "normal",
    3: "resalto"
}

MAX_POINTS = 500
mean_buffer = collections.deque(maxlen=MAX_POINTS)
event_buffer = collections.deque(maxlen=MAX_POINTS)
last_session = None
last_event = None
packet_count = 0
event_count = 0
buffer_lock = threading.Lock()

# ── Camera ──────────────────────────────────────────────────────────────────
_latest_frame_lock = threading.Lock()
_latest_frame_jpeg = None
_camera_running = False
_camera_thread = None
CAMERA_INDEX = 0
CAMERA_FPS = 20
CAMERA_W = 640
CAMERA_H = 480


def _camera_loop():
    global _latest_frame_jpeg, _camera_running

    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"[Camera] Opened index {idx}")
            break
        cap.release()
    else:
        print("[Camera] No camera found")
        _camera_running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    while _camera_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # Resize for performance
        frame = cv2.resize(frame, (320, 240))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with _latest_frame_lock:
                _latest_frame_jpeg = buf.tobytes()
        time.sleep(1 / CAMERA_FPS)

    cap.release()


def start_camera():
    global _camera_running, _camera_thread
    if _camera_running:
        return
    _camera_running = True
    _camera_thread = threading.Thread(target=_camera_loop, daemon=True)
    _camera_thread.start()


def _log(msg): print(f"[Server] {msg}")

# ── MQTT ────────────────────────────────────────────────────────────────────


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        _log(f"Connected to {BROKER_IP}:{PORT}")
        client.subscribe(
            [(TOPIC_MEAN_IMU, 0), (TOPIC_EVENT_SCORE, 0), (TOPIC_SESSION_SCORE, 0)])
    else:
        _log(f"Connection failed: rc={rc}")


def on_message(client, userdata, msg):
    global last_session, last_event, packet_count, event_count
    p = msg.payload
    t = msg.topic
    try:
        if t == TOPIC_MEAN_IMU and len(p) == PACKET_IMU:
            ts, sid, _, ax, ay, az, gx, gy, gz = struct.unpack(FMT_IMU, p)
            with buffer_lock:
                mean_buffer.append({"ts": ts, "ax": round(ax, 3), "ay": round(ay, 3), "az": round(az, 3),
                                    "gx": round(gx, 3), "gy": round(gy, 3), "gz": round(gz, 3)})
                packet_count += 1
        elif t == TOPIC_EVENT_SCORE and len(p) == PACKET_EVENT:
            ts, sid, et, pax, j, r, es = struct.unpack(FMT_EVENT_SCORE, p)
            ev = {"ts": ts, "event_type": et, "event_name": EVENT_TYPES.get(et, f"type_{et}"),
                  "peak_ax": round(pax, 3), "jerk": round(j, 3), "rms": round(r, 3),
                  "event_score": round(es, 3), "score_pct": round(es, 3)}
            with buffer_lock:
                event_buffer.append(ev)
                last_event = ev
                event_count += 1
            _log(f"Event: {ev['event_name']} score={ev['score_pct']}%")
        elif t == TOPIC_SESSION_SCORE and len(p) == PACKET_SESSION:
            ts, sid, ss, nf, ng, nr = struct.unpack(FMT_SESSION_SCORE, p)
            last_session = {"ts": ts, "session_score": round(ss, 3), "score_pct": round(ss, 3),
                            "n_frenado": nf, "n_giro": ng, "n_rompemuelle": nr}
            _log(
                f"Session: {last_session['score_pct']}% | B:{nf} T:{ng} P:{nr}")
    except Exception as e:
        _log(f"Parse error: {e}")


# ── Flask ───────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=TEMPLATES_DIR)
CORS(app)


@app.route("/")
def index():
    with open(os.path.join(TEMPLATES_DIR, "player.html"), 'r', encoding='utf-8') as f:
        return render_template_string(f.read())


@app.route("/api/live")
def api_live():
    with buffer_lock:
        return jsonify({
            "mean": list(mean_buffer), "events": list(event_buffer),
            "last_event": last_event, "last_session": last_session,
            "packet_count": packet_count, "event_count": event_count,
        })


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with _latest_frame_lock:
                frame = _latest_frame_jpeg
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(1 / CAMERA_FPS)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/frame")
def api_frame():
    """Return latest frame as base64 for manual refresh"""
    import base64
    with _latest_frame_lock:
        frame = _latest_frame_jpeg
    if frame:
        return jsonify({"frame": "data:image/jpeg;base64," + base64.b64encode(frame).decode()})
    return jsonify({"frame": None})


if __name__ == "__main__":
    start_camera()
    try:
        client = mqtt.Client(client_id="py_dash", protocol=mqtt.MQTTv311,
                             callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(BROKER_IP, PORT, 60)
        threading.Thread(target=client.loop_forever, daemon=True).start()
        _log(f"MQTT connected")
    except Exception as e:
        _log(f"MQTT error: {e}")
    app.run(host="0.0.0.0", port=5000, debug=False)
