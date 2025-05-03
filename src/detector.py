from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, model_path="yolov8s.pt", conf=0.5, iou=0.3, imgsz=640):
        self.model = YOLO(model_path)
        self.conf_threshold = conf
        self.iou_threshold = iou
        self.imgsz = imgsz

    def detect(self, image):
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            imgsz=self.imgsz,
        )[0]
        detections = []

        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            detections.append((cls_id, confidence, xyxy))

        return detections
