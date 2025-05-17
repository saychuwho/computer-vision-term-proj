import os
import pandas as pd
import cv2

# 1. Configuration
csv_path = 'annotations.csv'
images_dir = 'images'
labels_dir = 'labels'
class_map = {'traffic_light': 0, 'traffic_sign': 1}  # adjust as needed

os.makedirs(labels_dir, exist_ok=True)

# 2. Read CSV
df = pd.read_csv(csv_path)

# 3. Process each row
for _, row in df.iterrows():
    img_path = os.path.join(images_dir, row['filename'])
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # Convert to YOLO format
    x_center = ((row['x_min'] + row['x_max']) / 2) / w
    y_center = ((row['y_min'] + row['y_max']) / 2) / h
    box_width = (row['x_max'] - row['x_min']) / w
    box_height = (row['y_max'] - row['y_min']) / h
    class_id = class_map[row['class']]
    
    # Write to label file
    label_path = os.path.join(labels_dir, os.path.splitext(row['filename'])[0] + '.txt')
    with open(label_path, 'a') as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
