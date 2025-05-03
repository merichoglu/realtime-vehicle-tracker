# src/speed_estimator.py

from typing import List


class SpeedEstimator:
    def __init__(
        self,
        pix2m_a: float,
        pix2m_b: float,
        fps: float = 30.0,
        alpha: float = 0.2,
        min_pix: float = 0.2,
        pix2m_min: float = 0.113,  # = D/h2  (near-camera)
        pix2m_max: float = 0.228,  # = D/h1  (far)
    ):
        self.a = pix2m_a
        self.b = pix2m_b
        self.fps = fps
        self.alpha = alpha
        self.min_pix = min_pix
        self.min_m = pix2m_min
        self.max_m = pix2m_max
        self._history = {}  # last (cx,cy)
        self._smooth = {}  # last speed

    def estimate(self, objs: List[dict]) -> List[dict]:
        out = []
        for obj in objs:
            tid = obj["track_id"]
            x1, y1, x2, y2 = obj["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # default
            speed = 0.0

            if tid in self._history:
                px, py = self._history[tid]
                dy = cy - py
                dist_pix = abs(dy)  # vertical only

                if dist_pix >= self.min_pix:
                    # compute raw pix→m
                    raw = self.a * cy + self.b
                    # clamp to measured range
                    pix2m = max(min(raw, self.max_m), self.min_m)
                    # metres moved this frame
                    dist_m = dist_pix * pix2m
                    speed_mps = dist_m * self.fps
                    raw_kmh = speed_mps * 3.6

                    # low-pass filter
                    prev = self._smooth.get(tid, raw_kmh)
                    speed = self.alpha * raw_kmh + (1 - self.alpha) * prev
                else:
                    speed = self._smooth.get(tid, 0.0)

            # update history & smooth
            self._history[tid] = (cx, cy)
            self._smooth[tid] = speed
            obj["speed_kmh"] = speed
            out.append(obj)
        return out
