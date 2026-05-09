import os
import csv
import io
import cv2
from flask import Flask, jsonify, request, send_from_directory, render_template, send_file, abort
from flask_cors import CORS

DATA_DIR = "data"
app = Flask(__name__)
CORS(app)

# ── Per-session video capture cache ────────────────────────────────────────
# Keeps one open VideoCapture per session so we don't reopen on every request
_cap_cache = {}  # session_id -> cv2.VideoCapture

def get_cap(session_id):
    """Return a cached VideoCapture for the session, opening it if needed."""
    if session_id in _cap_cache:
        cap = _cap_cache[session_id]
        if cap.isOpened():
            return cap
        # Stale handle — reopen
        cap.release()

    video_path = os.path.join(DATA_DIR, session_id, "video.mp4")
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    _cap_cache[session_id] = cap
    return cap


@app.route("/")
def index():
    return render_template("playback.html")


@app.route("/session/<session_id>/frame/<int:frame_idx>")
def get_frame(session_id, frame_idx):
    """Extract a specific frame by index directly from video — no temp files."""
    print('okkk')

    cap = get_cap(session_id)

    if cap is None:
        abort(404, "Video not found")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx < 0 or frame_idx >= total:
        abort(400, f"Frame {frame_idx} out of range (0–{total-1})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        abort(500, "Could not decode frame")

    # Encode directly into memory — no disk touch at all
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        abort(500, "JPEG encode failed")


    return send_file(
        io.BytesIO(buf.tobytes()),
        mimetype="image/jpeg",
        max_age=0           # don't let the browser cache stale frames
    )


@app.route("/sessions")
def sessions():
    result = []
    if not os.path.exists(DATA_DIR):
        return jsonify([])

    for folder in sorted(os.listdir(DATA_DIR), reverse=True):
        path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(path):
            continue

        files = {
            "session_id":   folder,
            "imu":          None,
            "imu_filtered": None,
            "gps":          None,
            "frame_ts":     None,
            "labels":       None,
            "has_video":    False,
        }

        for fname in os.listdir(path):
            fp = os.path.join(path, fname)
            if   fname == "imu.csv":                        files["imu"]          = fp
            elif fname == "imu_filtered.csv":               files["imu_filtered"] = fp
            elif fname == "gps.csv":                        files["gps"]          = fp
            elif fname == "video_frame_timestamps.csv":     files["frame_ts"]     = fp
            elif fname == "labels.csv":                     files["labels"]       = fp
            elif fname == "video.mp4":                      files["has_video"]    = True

        result.append(files)

    return jsonify(result)


@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory(os.path.abspath(DATA_DIR), filename)


@app.route("/session/<session_id>/labels", methods=["POST"])
def save_labels(session_id):
    body = request.json or []
    session_path = os.path.join(DATA_DIR, session_id)
    os.makedirs(session_path, exist_ok=True)
    out_path = os.path.join(session_path, "labels.csv")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "start_ms", "end_ms", "duration_ms"])
        w.writeheader()
        for lbl in body:
            w.writerow({
                "label":       lbl["label"],
                "start_ms":    int(lbl["start_ms"]),
                "end_ms":      int(lbl["end_ms"]),
                "duration_ms": int(lbl["end_ms"] - lbl["start_ms"]),
            })

    return jsonify({"status": "saved", "count": len(body), "path": out_path})


if __name__ == "__main__":
    print("[server_playback] Running at http://localhost:5001")
    print(f"[server_playback] Serving sessions from ./{DATA_DIR}/")
    app.run(host="0.0.0.0", port=5001, debug=False)