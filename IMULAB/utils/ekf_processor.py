import numpy as np
import pandas as pd

from utils.helper import LPF2, LPF4, gps_course_to_enu_deg
from utils.attitude_ekf import AttitudeEKF


# GYRO
_LPF_B = [0.0133592, 0.0267184, 0.0133592]
_LPF_A = [1.0, -1.56101808, 0.64135154]

# ACC
_LPF_AX = [4.54666568e-06, 1.81866627e-05, 2.72799941e-05, 1.81866627e-05, 4.54666568e-06]
_LPF_BX = [1.         ,-3.75127641  ,5.28429506 ,-3.31262008 , 0.77967418]


class EKFProcessor:
    def __init__(
        self,
        gps_heading_gain=0.5,
        dt_max=0.1,
        progress_bar=None
    ):
        self.gps_heading_gain = gps_heading_gain
        self.dt_max = dt_max
        self.prev_lin_acc = np.zeros(3)
        self.jerk_initialized = False
        self.progress_bar = progress_bar
        self.lpf_acc_xy = LPF4(_LPF_AX, _LPF_BX)  # 4Hz:x,y
        self.lpf_acc_z = LPF2([0.06745527, 0.13491055, 0.06745527], [
                              1., -1.1429805, 0.4128016])  # 10Hz: z
        self.lpf_gyro = LPF2(_LPF_B, _LPF_A)  # 4Hz
        self.reset()

    # ─────────────────────────────────────────────
    def reset(self):
        self.ekf = AttitudeEKF()
        self.lpf_acc_xy = LPF4(_LPF_AX, _LPF_BX)
        self.lpf_acc_z = LPF2([0.06745527, 0.13491055, 0.06745527], [
                              1., -1.1429805, 0.4128016])  # 10Hz: z
        self.lpf_gyro = LPF2(_LPF_B, _LPF_A)
        self.prev_lin_acc = np.zeros(3)
        self.jerk_initialized = False

    def compute_jerk(self, lin_acc: np.ndarray, dt: float):
        if dt <= 0:
            return np.zeros(3)

        if not self.jerk_initialized:
            self.prev_lin_acc = lin_acc.copy()
            self.jerk_initialized = True
            return np.zeros(3)

        jerk = (lin_acc - self.prev_lin_acc) / dt

        self.prev_lin_acc = lin_acc.copy()

        return jerk

    # ─────────────────────────────────────────────
    def _step(self, row, dt, t_curr):

        gyro_raw = row[["gyro_x", "gyro_y", "gyro_z"]].values.astype(float)
        acc_raw = row[["accel_x", "accel_y", "accel_z"]].values.astype(float)

        Rz_180 = np.array([
            [-1,  0,  0],
            [0, -1,  0],
            [0,  0,  1]
        ])

        gyro_raw = Rz_180 @ gyro_raw
        acc_raw = Rz_180 @ acc_raw

        # ── EKF ─────────────────────────────
        self.ekf.predict(gyro_raw, dt)
        self.ekf.update_accel(acc_raw)

        # ── GPS heading ─────────────────────
        gps_course = row.get("course_deg", np.nan)
        gps_speed = row.get("speed_mps", np.nan)
        gps_lat = row.get("latitude", np.nan)
        gps_lon = row.get("longitude", np.nan)
        gps_alt = row.get("altitude_m", np.nan)

        if (
            not np.isnan(gps_course)
            and not np.isnan(gps_speed)
            and gps_speed > 1
        ):
            self.ekf.update_heading(
                np.radians(gps_course_to_enu_deg(gps_course)),
                self.gps_heading_gain
            )

        ekf_acc = self.ekf.linear_acceleration(acc_raw)
        ekf_gyro = self.ekf.angular_velocity(gyro_raw)
        roll, pitch, yaw = self.ekf.orientation()

        acc_lpf_xy = self.lpf_acc_xy.step(acc_raw)
        acc_lpf_z = self.lpf_acc_z.step(acc_raw)
        gyro_lpf = self.lpf_gyro.step(gyro_raw)

        return {
            "t": t_curr,

            "lpf_gx": gyro_lpf[0],
            "lpf_gy": gyro_lpf[1],
            "lpf_gz": gyro_lpf[2],

            "lpf_ax": acc_lpf_xy[0],
            "lpf_ay": acc_lpf_xy[1],
            "lpf_az": acc_lpf_z[2],

            "ax": acc_raw[0],
            "ay": acc_raw[1],
            "az": acc_raw[2],

            "gx": gyro_raw[0],
            "gy": gyro_raw[1],
            "gz": gyro_raw[2],

            "ekf_gx": ekf_gyro[0],
            "ekf_gy": ekf_gyro[1],
            "ekf_gz": ekf_gyro[2],

            "ekf_ax": ekf_acc[0],
            "ekf_ay": ekf_acc[1],
            "ekf_az": ekf_acc[2],

            "gps_speed": gps_speed,
            "gps_lat": gps_lat,
            "gps_lon": gps_lon,
            "gps_alt": gps_alt,
            "gps_course": gps_course,

            "roll": np.degrees(roll),
            "pitch": np.degrees(pitch),
            "yaw": np.degrees(yaw)
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
