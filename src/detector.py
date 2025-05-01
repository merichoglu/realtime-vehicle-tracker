from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf

    def detect(self, image):
        results = self.model.predict(image, conf=self.conf_threshold, verbose=True)[0]
        detections = []

        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            detections.append((cls_id, confidence, xyxy))

        return detections
