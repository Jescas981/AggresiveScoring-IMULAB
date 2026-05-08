import os
import csv
from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS

DATA_DIR = "data"
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("playback.html")

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
            "video":        None,       # legacy mp4 (optional)
            "frames_dir":   None,       # NEW: folder of frame images
            "frame_ts":     None,
            "labels":       None,
        }
        for fname in os.listdir(path):
            fp = os.path.join(path, fname)
            if   fname == "imu.csv":                              files["imu"]          = fp
            elif fname == "imu_filtered.csv":                     files["imu_filtered"] = fp
            elif fname == "gps.csv":                              files["gps"]          = fp
            elif fname == "fixed_video.mp4":                      files["video"]        = fp
            elif fname == "video_frame_timestamps.csv":           files["frame_ts"]     = fp
            elif fname == "fixed_video_frame_timestamps.csv":     files["frame_ts"]     = fp
            elif fname == "labels.csv":                           files["labels"]       = fp
            elif fname == "frames" and os.path.isdir(fp):
                # frames/ subfolder — store relative path
                files["frames_dir"] = f"{folder}/frames"
        result.append(files)
    return jsonify(result)

@app.route("/data/<path:filename>")
def serve_data(filename):
    """Serve any file under the data/ directory (CSV, MP4, frames/…)."""
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