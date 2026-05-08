import numpy as np
from utils.mymath import skew, rotvec_to_quat, quat_mult, quat_to_rot
GRAVITY = np.array([0, 0, 9.80665])

class AttitudeEKF:
    def __init__(self):
        # ── Nominal state ─────────────────────────────────────────────────
        self.q  = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion [w, x, y, z]
        self.bg = np.zeros(3)                        # gyro bias estimate (rad/s)
        self.ba = np.zeros(3)                        # acc bias estimate (rad/s)

        # ── Error state δx = [δθ(3), δbg(3), δba(3)] ─────────────────────────────
        # Reset to zero after every update step (error-state convention)
        self.dx = np.zeros(9)

        # ── Covariances ───────────────────────────────────────────────────
        self.P = np.diag([
            0.1,  0.1,  0.1,    # orientation uncertainty  (rad²)
            0.01, 0.01, 0.01,   # bias uncertainty         (rad²/s²)
            0.01, 0.01, 0.01,   # bias uncertainty         (m²/s²)
        ])

        self.Q = np.diag([
            5e-3, 5e-3, 5e-3,   # gyro white noise         (rad²/s)
            1e-6, 1e-6, 1e-6,   # bias random walk         (rad²/s³)
            1e-5, 1e-5, 1e-5,   # bias random walk         (m²/s³)
        ])

        # ── Noise ───────────────────────────────────────────────────
        self.R_acc = np.diag([5e-2, 5e-2, 5e-2])    # accel noise (tune per sensor)


    # ── Predict (gyroscope integration) ──────────────────────────────────────
    def predict(self, gyro: np.ndarray, dt: float):
        w = gyro - self.bg                      # bias-corrected angular velocity

        # 1. Propagate nominal quaternion
        dq = rotvec_to_quat(w * dt)
        self.q = quat_mult(self.q, dq)
        self.q /= np.linalg.norm(self.q)        # keep unit norm

        # 2. Error-state transition matrix
        F = np.eye(9)
        F[0:3, 0:3] = np.eye(3) - skew(w) * dt  # ∂δθ/∂δθ
        F[0:3, 3:6] = -np.eye(3) * dt            # ∂δθ/∂δbg

        # 3. Discrete noise input matrix
        G = np.zeros((9, 9))
        G[0:3, 0:3] = np.eye(3) * dt
        G[3:6, 3:6] = np.eye(3) * dt
        G[6:9, 6:9] = np.eye(3) * dt
        # 4. Propagate error-state covariance
        self.P = F @ self.P @ F.T + G @ self.Q @ G.T

    def _apply_correction(self, dx: np.ndarray):
        """Inject error-state correction into nominal state, then reset."""
        self.q  = quat_mult(self.q, rotvec_to_quat(dx[0:3]))
        self.q /= np.linalg.norm(self.q)
        self.bg += dx[3:6]
        self.ba += dx[6:9]
        self.dx  = np.zeros(9)

    def _kalman_update(self, residual: np.ndarray, H: np.ndarray, R: np.ndarray):
        """Generic KF measurement update. Returns corrected dx."""
        S  = H @ self.P @ H.T + R                        # innovation covariance
        K  = self.P @ H.T @ np.linalg.inv(S)             # Kalman gain
        dx = K @ residual                                 # error-state correction
        self.P = (np.eye(9) - K @ H) @ self.P            # covariance update
        return dx

    # ── Update 1: accelerometer (roll + pitch) ────────────────────────────────
    def update_accel(self, acc: np.ndarray, gate: float = 0.5):
        """
        Correct roll/pitch using accelerometer as gravity reference.
        Skipped when vehicle dynamics dominate (quasi-static gate).

        gate : max allowed deviation from 9.81 m/s² to accept update (m/s²)
        """
        acc_corr = acc - self.ba
        acc_mag = np.linalg.norm(acc_corr)
        if abs(acc_mag - 9.81) > gate:       # vehicle is accelerating → skip
            return

        a_norm = acc_corr / acc_mag               # unit gravity direction, body frame

        R      = quat_to_rot(self.q)
        g_pred = R.T @ (GRAVITY / 9.81)     # expected unit gravity in body frame

        residual = a_norm - g_pred           # (3,)

        # H: gravity measurement is sensitive to δθ, not δbg
        H = np.zeros((3, 9))
        H[0:3, 0:3] = skew(g_pred)          # ∂g_body/∂δθ
        H[0:3, 6:9] = -np.eye(3)

        dx = self._kalman_update(residual, H, self.R_acc)
        self._apply_correction(dx)

    def update_heading(self, heading_rad: float, heading_std_rad: float = 0.1):
        """
        Correct yaw using GPS-derived heading (course over ground).
        Only reliable above ~1 m/s — caller should gate on speed.

        heading_rad     : course-over-ground in ENU (rad, 0=East, CCW+)
        heading_std_rad : 1-sigma heading noise (rad)
        """
        R   = quat_to_rot(self.q)
        yaw = np.arctan2(R[1, 0], R[0, 0])  # yaw extracted from rotation matrix

        residual = np.array([heading_rad - yaw])

        # wrap to [-π, π]
        residual[0] = (residual[0] + np.pi) % (2 * np.pi) - np.pi

        # H: only yaw component of δθ affects heading measurement
        H       = np.zeros((1, 9))
        H[0, 2] = 1.0                        # ∂heading/∂δθ_z

        R_hdg = np.array([[heading_std_rad ** 2]])

        dx = self._kalman_update(residual, H, R_hdg)
        self._apply_correction(dx)

    def angular_velocity(self, gyro: np.ndarray):
        return gyro - self.bg

    def linear_acceleration(self, acc: np.ndarray) -> np.ndarray:
        """
        Compute linear acceleration in world frame (gravity removed).

        acc : raw accelerometer (body frame, m/s^2)

        returns: linear acceleration in world frame (m/s^2)
        """
        # 1. Rotación body → world
        acc_corr = acc - self.ba
        R = quat_to_rot(self.q)
        acc_world = R @ acc_corr
        # 2. Quitar gravedad
        lin_acc = acc_world - GRAVITY

        return lin_acc

    def acceleration(self, acc: np.ndarray):
        return acc - self.ba

    def orientation(self) -> dict:
        """
        Returns current attitude in three representations.

        euler_deg : (roll, pitch, yaw) in degrees  — intuitive for debugging
        euler_rad : (roll, pitch, yaw) in radians  — use for further math
        quaternion: [w, x, y, z]                   — canonical internal form
        R         : 3×3 rotation matrix body→world
        """
        R = quat_to_rot(self.q)

        # Aerospace ZYX convention: yaw → pitch → roll
        pitch = np.arcsin(-R[2, 0])
        roll  = np.arctan2(R[2, 1], R[2, 2])
        yaw   = np.arctan2(R[1, 0], R[0, 0])

        euler_rad = np.array([roll, pitch, yaw])

        return euler_rad