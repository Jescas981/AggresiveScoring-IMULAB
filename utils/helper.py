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

def lpf(prev, new, alpha=0.9):
    return alpha * prev + (1 - alpha) * new

def gps_course_to_enu_deg(course_deg):
    # GPS: 0° = North, clockwise positive
    # ENU: 0° = East, CCW positive

    yaw_rad = np.deg2rad(90.0 - course_deg)

    # wrap to [-pi, pi]
    yaw_rad = np.arctan2(np.sin(yaw_rad), np.cos(yaw_rad))

    return np.rad2deg(yaw_rad)