import cv2
import os
import pandas as pd


class VideoProcessor:
    def __init__(self, video_path, out_dir="frames", progress_bar=None):
        self.video_path = video_path
        self.out_dir = out_dir
        self.progress_bar = progress_bar

    def extract(self):

        os.makedirs(self.out_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[VIDEO] fps={fps:.2f}, total_frames={total_frames}")

        # ─────────────────────────────
        # INIT PROGRESS BAR PROPERLY
        # ─────────────────────────────
        if self.progress_bar is not None:
            self.progress_bar.total = total_frames
            self.progress_bar.reset()

        timestamps = []
        saved_idx = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            filename = f"{saved_idx:06d}.png"
            path = os.path.join(self.out_dir, filename)

            cv2.imwrite(path, frame)

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            timestamps.append({
                "frame_idx": saved_idx,
                "timestamp_ms": timestamp_ms,
                "filename": filename
            })

            saved_idx += 1
            frame_idx += 1

            # ─────────────────────────────
            # PROGRESS FIX (IMPORTANT)
            # ─────────────────────────────
            if self.progress_bar is not None:
                self.progress_bar.update(1)

        cap.release()

        df = pd.DataFrame(timestamps)

        df.to_csv(
            os.path.join(self.out_dir, "video_frame_timestamps.csv"),
            index=False
        )

        print(f"\n[OK] Saved {saved_idx} frames → {self.out_dir}")
        print("[OK] Saved video_frame_timestamps.csv")

        return df