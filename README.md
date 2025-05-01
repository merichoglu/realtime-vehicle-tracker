# 🚗 Real-Time Vehicle Tracker

A real-time vehicle detection, tracking, and speed estimation app powered by YOLOv8 and Deep SORT.

![demo](results/annotated.gif)

---

## 🔍 Features

- YOLOv8-based object detection
- Deep SORT for persistent tracking
- Speed estimation in km/h (via pixel-to-meter ratio & FPS)
- Streamlit UI for easy video uploads
- CLI tool for batch processing & video export

---

## 🧰 Installation

```bash
git clone https://github.com/merichoglu/realtime-vehicle-tracker.git
cd realtime-vehicle-tracker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
