# src/tracker.py

from typing import Any, Dict, List, Tuple

from deep_sort_realtime.deepsort_tracker import DeepSort


def iou(
    boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]
) -> float:
    """
    Compute Intersection over Union between two [x1,y1,x2,y2] boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea + 1e-6
    return interArea / unionArea


class ObjectTracker:
    def __init__(
        self, max_age: int = 30, n_init: int = 3, max_cosine_distance: float = 0.4
    ):
        """
        Wraps Deep SORT.
        """
        self.tracker = DeepSort(
            max_age=max_age, n_init=n_init, max_cosine_distance=max_cosine_distance
        )

    def update(self, detections: List[Tuple[int, float, Any]], frame) -> List[Dict]:
        """
        Args:
          detections: list of (cls_id, conf, [x1,y1,x2,y2]) from YOLO
          frame: current BGR image

        Returns:
          list of dicts: {
            "track_id": int,
            "class_id": int,
            "bbox": [x1,y1,x2,y2]
          }
        """
        # 1) format for Deep SORT
        formatted = []
        for cls_id, conf, bbox in detections:
            x1, y1, x2, y2 = bbox
            formatted.append(([x1, y1, x2, y2], conf, cls_id))

        # 2) run tracker
        tracks = self.tracker.update_tracks(formatted, frame=frame)

        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            cls = track.det_class

            # 3) get the Kalman‐predicted box for this track (for IoU matching)
            pred_box = track.to_ltrb()

            # 4) find which YOLO det it came from
            best_iou = 0.0
            best_bbox = pred_box  # fallback
            for det_cls, _, det_bbox in detections:
                if det_cls != cls:
                    continue
                i = iou(pred_box, det_bbox)
                if i > best_iou:
                    best_iou = i
                    best_bbox = det_bbox

            x1, y1, x2, y2 = best_bbox
            results.append(
                {
                    "track_id": tid,
                    "class_id": cls,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )

        return results
