import os
import csv
import io
import cv2
import threading
from flask import Flask, jsonify, request, send_from_directory, render_template, send_file, abort
from flask_cors import CORS

DATA_DIR = "data"
app = Flask(__name__)
CORS(app)

# Global lock for video operations
video_lock = threading.Lock()

@app.route("/")
def index():
    return render_template("playback.html")


@app.route("/session/<session_id>/frame/<int:frame_idx>")
def get_frame(session_id, frame_idx):
    """Extract a specific frame directly from video (thread-safe)."""
    
    video_path = os.path.join(DATA_DIR, session_id, "video.mp4")
    
    if not os.path.exists(video_path):
        abort(404, "Video not found")
    
    # Use lock to prevent concurrent FFmpeg access
    with video_lock:
        cap = None
        try:
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
                # Try with default backend as fallback
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    abort(500, "Could not open video")
            
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_idx < 0 or frame_idx >= total:
                abort(400, f"Frame {frame_idx} out of range (0–{total-1})")
            
            # Seek frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                abort(500, "Could not decode frame")
            
            # Encode in memory (no disk)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if not ok:
                abort(500, "JPEG encode failed")
            
            return send_file(
                io.BytesIO(buf.tobytes()),
                mimetype="image/jpeg",
                max_age=0,
                conditional=False
            )
            
        except Exception as e:
            app.logger.error(f"Error processing frame {frame_idx}: {str(e)}")
            abort(500, f"Error processing frame: {str(e)}")
        finally:
            if cap is not None:
                cap.release()


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


@app.errorhandler(500)
def handle_500(error):
    app.logger.error(f"500 error: {error}")
    return jsonify({"error": str(error)}), 500


@app.errorhandler(404)
def handle_404(error):
    return jsonify({"error": str(error)}), 404


@app.errorhandler(400)
def handle_400(error):
    return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    # Set environment variables to help with FFmpeg stability
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
    
    print("[server_playback] Running at http://localhost:5001")
    print(f"[server_playback] Serving sessions from ./{DATA_DIR}/")
    
    # Run with threaded=False to avoid concurrency issues with FFmpeg
    # Or use a production WSGI server like gunicorn with a single worker
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)