import argparse
import logging
import os

import cv2

from detector import VehicleDetector
from speed_estimator import SpeedEstimator
from tracker import ObjectTracker


def setup_logger() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="real-time vehicle detection, tracking, and speed estimation."
    )
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--model", type=str, default="models/yolov8s.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="results/output.mp4")
    return parser.parse_args()


def run_detection(
    video_path: str, model_path: str, conf_threshold: float, output_path: str
) -> None:
    source = 0 if video_path is None else video_path
    detector = VehicleDetector(model_path, conf=conf_threshold)
    tracker = ObjectTracker()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error("failed to open video source: %s", source)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # pix2m calibration constants measured
    pix2m_a = -0.000269
    pix2m_b = 0.278620

    # init speed estimator with measured constants
    speed_estimator = SpeedEstimator(
        pix2m_a=pix2m_a,
        pix2m_b=pix2m_b,
        fps=fps,
        alpha=0.2,
        min_pix=0.2,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    logging.info("processing... press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # detect vehicles
        dets = detector.detect(frame)
        VEHICLE_CLASS_IDS = {2, 3, 5, 7}
        dets = [d for d in dets if d[0] in VEHICLE_CLASS_IDS]

        # track vehicles
        tracks = tracker.update(dets, frame)

        # estimate speed using perspective calibration
        tracks = speed_estimator.estimate(tracks)

        # draw results
        for obj in tracks:
            x1, y1, x2, y2 = obj["bbox"]
            tid = obj["track_id"]
            speed = obj["speed_kmh"]
            label = f"ID {tid} | {speed:.1f} km/h"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        out.write(frame)
        cv2.imshow("Vehicle Tracking + Speed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logging.info("saved output to: %s", output_path)


if __name__ == "__main__":
    setup_logger()
    args = parse_args()
    run_detection(args.path, args.model, args.conf, args.out)
