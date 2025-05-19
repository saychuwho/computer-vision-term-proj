import os
import pandas as pd
import cv2
import shutil
from tqdm import tqdm

# 1. Configuration
csv_path = './tiny-LISA-traffic-sign/db_lisa_tiny/annotations.csv'

origin_images_dir = './tiny-LISA-traffic-sign/db_lisa_tiny'

dataset_images_dir = './datasets-LISA-traffic-sign/images'
dataset_labels_dir = './datasets-LISA-traffic-sign/labels'

class_map = {
    'stop': 0, 
    'yield': 1,
    'yieldAhead': 2,
    'merge': 3,
    'signalAhead': 4,
    'pedestrianCrossing': 5,
    'keepRight': 6,
    'speedLimit35': 7,
    'speedLimit25': 8
    }  # adjust as needed


# 2. Read CSV
df = pd.read_csv(csv_path)

def convert_bbox(x1, y1, x2, y2, img_w, img_h):
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = abs(x2 - x1) / img_w
    height = abs(y2 - y1) / img_h
    return x_center, y_center, width, height

class_info = {"stop": [],
              "yield": [],
              "yieldAhead": [],
              "merge": [],
              "signalAhead": [],
              "pedestrianCrossing": [],
              "keepRight": [],
              "speedLimit35": [],
              "speedLimit25": []}

# 3. Process each row
for i, row in tqdm(df.iterrows()):
    img_path = os.path.join(origin_images_dir, row['filename'])
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # Convert to YOLO format
    x_center, y_center, box_width, box_height = convert_bbox(row['x1'], row['y1'], row['x2'], row['y2'], w, h)
    class_id = class_map[row['class']]
    
    class_info[row['class']].append((row['filename'], f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"))

# 4. write to files spliting train and val
for class_name in class_info:
    class_len = len(class_info[class_name])
    val_len = int(class_len * 0.2)
    train_len = class_len - val_len

    for annotate_data in class_info[class_name][:train_len]:
        filename, label = annotate_data
        image_path = dataset_images_dir + '/train/' + filename
        label_path = dataset_labels_dir + '/train/' + os.path.splitext(filename)[0] + '.txt'
        
        shutil.copy2(img_path, image_path)
        with open(label_path, 'w') as f:
            f.write(label)
    
    for annotate_data in class_info[class_name][train_len:]:
        filename, label = annotate_data
        image_path = dataset_images_dir + '/val/' + filename
        label_path = dataset_labels_dir + '/val/' + os.path.splitext(filename)[0] + '.txt'
        
        shutil.copy2(img_path, image_path)
        with open(label_path, 'w') as f:
            f.write(label)