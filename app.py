import os
import tempfile

import cv2
import streamlit as st

from src.detector import VehicleDetector
from src.speed_estimator import SpeedEstimator
from src.tracker import ObjectTracker

st.set_page_config(page_title="Vehicle Detection & Speed Estimation", layout="wide")
st.title("🚗 Vehicle Detection, Tracking, and Speed Estimation")

uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pixel_to_meter = 0.05

    detector = VehicleDetector(model_path="models/yolov8n.pt", conf=0.4)
    tracker = ObjectTracker()
    speed_estimator = SpeedEstimator(pixel_to_meter, fps)

    stframe = st.empty()
    st.info("Processing... Press Stop to end early.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        VEHICLES = {2, 3, 5, 7}
        detections = [d for d in detections if d[0] in VEHICLES]

        tracked = tracker.update(detections, frame)
        tracked = speed_estimator.estimate(tracked)

        for obj in tracked:
            x1, y1, x2, y2 = obj["bbox"]
            tid = obj["track_id"]
            speed = obj["speed_kmh"]
            label = f"ID {tid} | {speed:.1f} km/h"

            # Improve readability: scale font and thickness with box height
            h = max(20, y2 - y1)
            font_scale = min(max(h / 300, 0.3), 0.8)
            thickness = max(1, int(h / 200))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)

            # Outline + fill text
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

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

    cap.release()
    os.remove(video_path)
    st.success("Done!")
