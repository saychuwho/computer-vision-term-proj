import cv2
import os
from ultralytics import YOLO
from tqdm import tqdm

import json

annotation_file = 'datasets/BDD-X-Annotations-finetune-val.json'
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

video_path_list = [annotation["video"][0] for annotation in annotations]
# video_path_list = glob.glob(os.path.join(video_dir, '*.mp4'))


output_labels_dir = './YOLO-labels/traffic-light'
model_path = 'runs/detect/yolov8n-tiny-LISA-traffic-light5/weights/best.pt'


# Load YOLO model
model = YOLO(model_path)

# Open video
for video_path in tqdm(video_path_list):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    # Get original video FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        original_fps = 30  # fallback if FPS can't be read

    # Prepare label file (one per video)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    label_path = os.path.join(output_labels_dir, f"{video_basename}.txt")

    with open(label_path, 'w') as f:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Only process frames at 1 FPS
            if frame_idx % int(original_fps) == 0:
                # Run YOLO inference
                results = model(frame, verbose=False)

                # Write all detections for this frame
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    xywhn = boxes.xywhn.cpu().numpy()
                    class_ids = boxes.cls.cpu().numpy().astype(int)
                    for box, class_id in zip(xywhn, class_ids):
                        x_center, y_center, width, height = box[:4]
                        # Format: frame_idx class_id x_center y_center width height
                        f.write(f"{frame_idx} {class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            frame_idx += 1

    cap.release()
