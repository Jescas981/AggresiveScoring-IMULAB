import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple


def load_sensor(path: str, required_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("timestamp_ms").reset_index(drop=True)
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df

def dedup_gps_per_second(gps: pd.DataFrame) -> pd.DataFrame:
    def pick_median(g):
        median_t = g["timestamp_ms"].median()
        return g.iloc[(g["timestamp_ms"] - median_t).abs().argmin()]
    return (
        gps.groupby("utc_seconds", group_keys=False)
           .apply(pick_median)
           .reset_index(drop=True)
    )

def clip_to_overlap(
    *dfs: pd.DataFrame, col: str = "timestamp_ms"
) -> Tuple[pd.DataFrame, ...]:
    t_start = max(df[col].min() for df in dfs)
    t_end = min(df[col].max() for df in dfs)
    overlap_s = (t_end - t_start) / 1_000
    if overlap_s <= 0:
        raise ValueError("No temporal overlap between sensors.")
    print(f"Overlap window : {t_start} → {t_end}  ({overlap_s:.1f} s)")
    return tuple(
        df[(df[col] >= t_start) & (df[col] <= t_end)].reset_index(drop=True)
        for df in dfs
    )

import numpy as np

class LPF4:

    def __init__(self, b, a):

        self.b = np.asarray(b, dtype=np.float64)
        self.a = np.asarray(a, dtype=np.float64)

        # x[n-1 ... n-4]
        self.x1 = None
        self.x2 = None
        self.x3 = None
        self.x4 = None

        # y[n-1 ... n-4]
        self.y1 = None
        self.y2 = None
        self.y3 = None
        self.y4 = None

    def step(self, x):

        x = np.asarray(x, dtype=np.float64)

        # =================================================
        # warm start
        # =================================================

        if self.x1 is None:

            self.x1 = x.copy()
            self.x2 = x.copy()
            self.x3 = x.copy()
            self.x4 = x.copy()

            self.y1 = x.copy()
            self.y2 = x.copy()
            self.y3 = x.copy()
            self.y4 = x.copy()

            return x

        # =================================================
        # 4th-order IIR
        # =================================================

        y = (
            self.b[0] * x +
            self.b[1] * self.x1 +
            self.b[2] * self.x2 +
            self.b[3] * self.x3 +
            self.b[4] * self.x4 -

            self.a[1] * self.y1 -
            self.a[2] * self.y2 -
            self.a[3] * self.y3 -
            self.a[4] * self.y4
        ) / self.a[0]

        # =================================================
        # shift states
        # =================================================

        self.x4 = self.x3.copy()
        self.x3 = self.x2.copy()
        self.x2 = self.x1.copy()
        self.x1 = x.copy()

        self.y4 = self.y3.copy()
        self.y3 = self.y2.copy()
        self.y2 = self.y1.copy()
        self.y1 = y.copy()

        return y

class LPF2:
    def __init__(self, b, a):
        self.b = np.asarray(b, dtype=np.float64)
        self.a = np.asarray(a, dtype=np.float64)

        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None

    def step(self, x):
        x = np.asarray(x, dtype=np.float64)

        # ── init (warm start) ─────────────────────────────
        if self.x1 is None:
            self.x1 = x.copy()
            self.x2 = x.copy()
            self.y1 = x.copy()
            self.y2 = x.copy()
            return x

        # ── IIR biquad ───────────────────────────────────
        y = (
            self.b[0] * x +
            self.b[1] * self.x1 +
            self.b[2] * self.x2 -
            self.a[1] * self.y1 -
            self.a[2] * self.y2
        )

        # ── shift states ────────────────────────────────
        self.x2 = self.x1.copy()
        self.x1 = x.copy()
        self.y2 = self.y1.copy()
        self.y1 = y.copy()

        return y
    
def gps_course_to_enu_deg(course_deg):
    # GPS: 0° = North, clockwise positive
    # ENU: 0° = East, CCW positive

    yaw_rad = np.deg2rad(90.0 - course_deg)

    # wrap to [-pi, pi]
    yaw_rad = np.arctan2(np.sin(yaw_rad), np.cos(yaw_rad))

    return np.rad2deg(yaw_rad)