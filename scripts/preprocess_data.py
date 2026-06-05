import os
import pandas as pd

from utils.helper import (
    load_sensor,
    dedup_gps_per_second,
    clip_to_overlap
)

from utils.view import ProgressBar
from utils.ekf_processor import EKFProcessor
from utils.video_processor import VideoProcessor


# ─────────────────────────────
# ROOT CONFIG
# ─────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, "data")


# ─────────────────────────────
# SESSIONS
# ─────────────────────────────

sessions = sorted([
    d for d in os.listdir(DATA_ROOT)
])

print(f"Found {len(sessions)} sessions:")

global_bar = ProgressBar(len(sessions), label="ALL SESSIONS")


# ─────────────────────────────
# PROCESS
# ─────────────────────────────

for session in sessions:

    print(f"\n🔍 Checking {session}...")

    DATA_DIR = os.path.join(DATA_ROOT, session)

    GPS_PATH = os.path.join(DATA_DIR, "gps.csv")
    IMU_PATH = os.path.join(DATA_DIR, "imu.csv")
    VIDEO_PATH = os.path.join(DATA_DIR, "video.mp4")

    OUT_IMU = os.path.join(DATA_DIR, "imu_filtered.csv")
    FRAMES_DIR = os.path.join(DATA_DIR, "frames")

    # skip invalid sessions
    if not all(map(os.path.exists, [GPS_PATH, IMU_PATH, VIDEO_PATH])):
        print(f"[SKIP] incomplete: {session}")
        global_bar.update(1)
        continue

    print(f"\n🚀 Processing {session}")

    # os.makedirs(FRAMES_DIR, exist_ok=True)


    # ─────────────────────────────
    # LOAD DATA
    # ─────────────────────────────

    GPS_COLS = [
        "timestamp_ms", "latitude", "longitude",
        "speed_mps", "course_deg", "altitude_m"
    ]

    IMU_COLS = [
        "timestamp_ms", "accel_x", "accel_y",
        "accel_z", "gyro_x", "gyro_y", "gyro_z"
    ]

    gps_raw = load_sensor(GPS_PATH, GPS_COLS)
    imu_raw = load_sensor(IMU_PATH, IMU_COLS)

    gps_dedup = dedup_gps_per_second(gps_raw)
    gps, imu = clip_to_overlap(gps_dedup, imu_raw)

    gps["timestamp_ms"] = gps["timestamp_ms"].astype("int64")
    imu["timestamp_ms"] = imu["timestamp_ms"].astype("int64")


    # ─────────────────────────────
    # MERGE IMU + GPS
    # ─────────────────────────────

    imu_with_gps = pd.merge_asof(
        imu.sort_values("timestamp_ms"),
        gps[["timestamp_ms", "speed_mps", "course_deg","latitude","longitude","altitude_m"]],
        on="timestamp_ms",
        direction="nearest",
        tolerance=30
    )

    imu_with_gps.to_csv(
        os.path.join(DATA_DIR, "imu_with_gps.csv"),
        index=False
    )


    # ─────────────────────────────
    # EKF (INDEPENDENT)
    # ─────────────────────────────

    if not os.path.exists(OUT_IMU):

        ekf_bar = ProgressBar(len(imu_with_gps), label=f"EKF {session}")

        ekf_proc = EKFProcessor(progress_bar=ekf_bar)
        df = ekf_proc.run(imu_with_gps)

        ekf_bar.close()

        df.to_csv(OUT_IMU, index=False)

    else:
        print(f"[SKIP] EKF already done: {session}")


    # ─────────────────────────────
    # VIDEO (INDEPENDENT)
    # ─────────────────────────────

    # if not (os.path.exists(FRAMES_DIR) and len(os.listdir(FRAMES_DIR)) > 0):

    #     video_proc = VideoProcessor(
    #         video_path=VIDEO_PATH,
    #         out_dir=FRAMES_DIR
    #     )

    #     frame_df = video_proc.extract()

    # else:
    #     print(f"[SKIP] frames already exist: {session}")


    # ─────────────────────────────
    # SAVE CLEAN DATA
    # ─────────────────────────────

    gps.to_csv(os.path.join(DATA_DIR, "gps_clean.csv"), index=False)
    imu.to_csv(os.path.join(DATA_DIR, "imu_clean.csv"), index=False)

    print(f"✅ Done: {session}")

    global_bar.update(1)


global_bar.close()

print("\n🎉 ALL SESSIONS PROCESSED")