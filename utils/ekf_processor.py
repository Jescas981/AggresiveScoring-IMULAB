import numpy as np
import pandas as pd

from utils.helper import lpf, gps_course_to_enu_deg
from utils.attitude_ekf import AttitudeEKF


class EKFProcessor:
    def __init__(
        self,
        alpha_gyro=0.95,
        alpha_acc=0.90,
        gps_heading_gain=0.5,
        dt_max=0.1,
        progress_bar=None
    ):
        self.alpha_gyro = alpha_gyro
        self.alpha_acc = alpha_acc
        self.gps_heading_gain = gps_heading_gain
        self.dt_max = dt_max

        self.progress_bar = progress_bar

        self.reset()

    # ─────────────────────────────────────────────
    def reset(self):
        self.ekf = AttitudeEKF()
        self.gyro_f = np.zeros(3)
        self.acc_f = np.zeros(3)

    # ─────────────────────────────────────────────
    def _step(self, row, dt, t_curr):

        gyro_raw = row[["gyro_x", "gyro_y", "gyro_z"]].values.astype(float)
        acc_raw = row[["accel_x", "accel_y", "accel_z"]].values.astype(float)

        # ── LPF ─────────────────────────────
        self.gyro_f = lpf(self.gyro_f, gyro_raw, self.alpha_gyro)
        self.acc_f = lpf(self.acc_f, acc_raw, self.alpha_acc)

        # ── EKF ─────────────────────────────
        self.ekf.predict(self.gyro_f, dt)
        self.ekf.update_accel(self.acc_f)

        # ── GPS heading ─────────────────────
        course = row.get("course_deg", np.nan)
        speed = row.get("speed_mps", np.nan)

        if (
            not np.isnan(course)
            and not np.isnan(speed)
            and speed > 1
        ):
            self.ekf.update_heading(
                np.radians(gps_course_to_enu_deg(course)),
                self.gps_heading_gain
            )

        lin_acc = self.ekf.linear_acceleration(self.acc_f)
        roll, pitch, yaw = self.ekf.orientation()

        return {
            "t": t_curr,

            "gx": self.gyro_f[0],
            "gy": self.gyro_f[1],
            "gz": self.gyro_f[2],

            "ax": self.acc_f[0],
            "ay": self.acc_f[1],
            "az": self.acc_f[2],

            "alin_x": lin_acc[0],
            "alin_y": lin_acc[1],
            "alin_z": lin_acc[2],

            "roll": np.degrees(roll),
            "pitch": np.degrees(pitch),
            "yaw": np.degrees(yaw),
        }

    # ─────────────────────────────────────────────
    def run(self, imu_with_gps: pd.DataFrame):

        self.reset()

        results = []
        total = len(imu_with_gps)

        for i in range(1, total):

            row = imu_with_gps.iloc[i]

            t_curr = row["timestamp_ms"]
            t_prev = imu_with_gps["timestamp_ms"].iloc[i - 1]

            dt = (t_curr - t_prev) / 1000.0

            if dt <= 0 or dt > self.dt_max:
                continue

            results.append(self._step(row, dt, t_curr))

            # ── progress bar (optional) ──
            if self.progress_bar is not None:
                self.progress_bar.update(i)

            elif i % 500 == 0 or i == total - 1:
                pct = 100 * i / total
                print(f"\rEKF: {pct:5.1f}% ({i}/{total})", end="")

        print("\nEKF done.")
        return pd.DataFrame(results)