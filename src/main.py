# src/main.py

import argparse
import logging
from typing import Optional

import cv2

from detector import VehicleDetector


def setup_logger() -> None:
    """Configure logging format."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time vehicle detection using YOLOv11."
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
        default="yolov8n.pt",
        help="Path to YOLOv11 .pt model file.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold for YOLO detection.",
    )
    return parser.parse_args()


def run_detection(
    video_path: Optional[str], model_path: str, conf_threshold: float
) -> None:
    """
    Run YOLOv11-based vehicle detection on input video or webcam.

    Args:
        video_path (Optional[str]): Path to video file or None for webcam.
        model_path (str): Path to YOLOv11 model weights.
        conf_threshold (float): Confidence threshold.
    """
    source = 0 if video_path is None else video_path
    detector = VehicleDetector(model_path=model_path, conf=conf_threshold)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error("Failed to open video source: %s", source)
        return

    logging.info("Starting video stream... Press 'q' to quit.")

    while True:
        ret: bool
        frame: Optional[cv2.typing.MatLike]
        ret, frame = cap.read()
        if not ret or frame is None:
            logging.info("End of video or failed to read frame.")
            break

        detections = detector.detect(frame)

        for cls_id, conf, xyxy in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"{cls_id} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow("YOLOv11 Vehicle Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            logging.info("Quitting detection.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    setup_logger()
    args = parse_args()
    run_detection(video_path=args.path, model_path=args.model, conf_threshold=args.conf)
