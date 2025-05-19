from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='datasets-LISA-traffic-light/data.yaml', epochs=100, imgsz=640, batch=16, name='yolov8n-tiny-LISA-traffic-light')

