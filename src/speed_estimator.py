# src/speed_estimator.py

from typing import Dict, List, Tuple


class SpeedEstimator:
    def __init__(self, pixel_to_meter: float, fps: float):
        self.pixel_to_meter = pixel_to_meter
        self.fps = fps
        # store last center position per track_id
        self._history: Dict[int, Tuple[float, float]] = {}

    def estimate(self, tracked_objects: List[dict]) -> List[dict]:
        results: List[dict] = []
        for obj in tracked_objects:
            tid = obj["track_id"]
            x1, y1, x2, y2 = obj["bbox"]
            # compute center
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if tid in self._history:
                px, py = self._history[tid]
                # pixel displacement
                dx = cx - px
                dy = cy - py
                dist_pix = (dx**2 + dy**2) ** 0.5
                # convert to meters
                dist_m = dist_pix * self.pixel_to_meter
                # speed in m/s = dist_m * fps
                speed_mps = dist_m * self.fps
                # convert to km/h
                speed_kmh = speed_mps * 3.6
            else:
                speed_kmh = 0.0

            # update history
            self._history[tid] = (cx, cy)
            # annotate object
            obj["speed_kmh"] = speed_kmh
            results.append(obj)

        return results
