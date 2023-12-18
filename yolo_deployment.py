from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/last.pt")

model.predict(source="0", show=True, conf=0.4)

