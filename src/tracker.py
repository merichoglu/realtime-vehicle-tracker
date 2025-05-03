# src/tracker.py

from typing import Dict, List, Tuple

from deep_sort_realtime.deepsort_tracker import DeepSort


def iou(
    boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]
) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / (unionArea + 1e-6)


class ObjectTracker:
    def __init__(self, max_age=60, n_init=5, max_cosine_distance=0.4):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
        )

    def update(
        self,
        detections: List[Tuple[int, float, Tuple[float, float, float, float]]],
        frame,
    ) -> List[Dict]:
        formatted = [
            ([x1, y1, x2, y2], conf, cls) for cls, conf, (x1, y1, x2, y2) in detections
        ]
        tracks = self.tracker.update_tracks(formatted, frame=frame)
        out, seen = [], set()

        for t in tracks:
            # filter unconfirmed or stale
            if not t.is_confirmed() or t.time_since_update > 0:
                continue
            tid = t.track_id
            if tid in seen:
                continue
            seen.add(tid)

            pred = t.to_ltrb()  # use kalman-predicted bbox
            best_box, best_iou = None, 0.0
            for cls, _, det_box in detections:
                if cls != t.det_class:
                    continue
                score = iou(pred, det_box)
                if score > best_iou:
                    best_iou, best_box = score, det_box

            x1, y1, x2, y2 = best_box if best_box is not None else pred
            out.append(
                {
                    "track_id": tid,
                    "class_id": t.det_class,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )

        return out
