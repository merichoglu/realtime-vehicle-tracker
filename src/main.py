# src/main.py

import argparse
import logging
import os
from typing import Optional

import cv2

from detector import VehicleDetector
from speed_estimator import SpeedEstimator
from tracker import ObjectTracker


def setup_logger() -> None:
    """Configure logging format."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time vehicle detection, tracking, and speed estimation using YOLOv8 and Deep SORT."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to input video file. If omitted, webcam will be used.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov8n.pt",
        help="Path to YOLOv8 .pt model file.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold for YOLO detection.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output.mp4",
        help="Output video path (optional)",
    )
    return parser.parse_args()


def run_detection(
    video_path: Optional[str], model_path: str, conf_threshold: float, output_path: str
) -> None:
    source = 0 if video_path is None else video_path

    detector = VehicleDetector(model_path=model_path, conf=conf_threshold)
    tracker = ObjectTracker()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error("Failed to open video source: %s", source)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pixel_to_meter = 0.05
    speed_estimator = SpeedEstimator(pixel_to_meter, fps)

    logging.info("Processing... Press 'q' to quit preview window.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        yolo_detections = detector.detect(frame)
        VEHICLE_CLASS_IDS = {2, 3, 5, 7}
        yolo_detections = [
            det for det in yolo_detections if det[0] in VEHICLE_CLASS_IDS
        ]

        tracked_objects = tracker.update(yolo_detections, frame)
        tracked_objects = speed_estimator.estimate(tracked_objects)

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["bbox"]
            tid = obj["track_id"]
            speed = obj.get("speed_kmh", 0.0)
            label = f"ID {tid} | {speed:.1f} km/h"

            h = max(20, y2 - y1)
            font_scale = min(max(h / 300, 0.3), 0.8)
            thickness = max(1, int(h / 200))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        out.write(frame)
        cv2.imshow("Vehicle Detection + Tracking + Speed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logging.info("Saved output video to: %s", output_path)


if __name__ == "__main__":
    setup_logger()
    args = parse_args()
    run_detection(
        video_path=args.path,
        model_path=args.model,
        conf_threshold=args.conf,
        output_path=args.out,
    )
