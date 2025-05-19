import os
import pandas as pd
import cv2
import shutil
from tqdm import tqdm

# # 1. Configuration
# csv_path_train_list = ['./LISA-traffic-light/Annotations/Annotations/dayTrain/dayClip1/',
#                        './LISA-traffic-light/Annotations/Annotations/dayTrain/dayClip2/',
#                        './LISA-traffic-light/Annotations/Annotations/dayTrain/dayClip3/',
#                        './LISA-traffic-light/Annotations/Annotations/dayTrain/dayClip4/',
#                        './LISA-traffic-light/Annotations/Annotations/dayTrain/dayClip5/',
#                        './LISA-traffic-light/Annotations/Annotations/nightTrain/nightClip1/',
#                        './LISA-traffic-light/Annotations/Annotations/nightTrain/nightClip2/',
#                        './LISA-traffic-light/Annotations/Annotations/nightTrain/nightClip3/',
#                        './LISA-traffic-light/Annotations/Annotations/nightTrain/nightClip4/',
#                        './LISA-traffic-light/Annotations/Annotations/nightTrain/nightClip5/']
# csv_path_val_list = ['./LISA-traffic-light/Annotations/Annotations/daySequence1/',
#                      './LISA-traffic-light/Annotations/Annotations/daySequence2/',
#                      './LISA-traffic-light/Annotations/Annotations/nightSequence1/',
#                      './LISA-traffic-light/Annotations/Annotations/nightSequence2/']

# origin_images_dir_train_list = [./LISA-traffic-light/dayTrain/dayTrain/dayClip1/frames/]
# origin_images_dir_val_list = [./LISA-traffic-light/daySequence1/daySequence1/frames/]


dataset_images_dir = './datasets-LISA-traffic-light/images'
dataset_labels_dir = './datasets-LISA-traffic-light/labels'

class_map = { }  # adjust as needed


def get_class_id(class_map, class_name):
    if class_name not in class_map:
        class_map[class_name] = len(class_map)
    return class_map[class_name]


def convert_bbox(x1, y1, x2, y2, img_w, img_h):
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = abs(x2 - x1) / img_w
    height = abs(y2 - y1) / img_h
    return x_center, y_center, width, height


def make_labels(csv_path, origin_images_dir, dataset_images_dir, dataset_labels_dir, type):
    csv_path += "frameAnnotationsBOX.csv"
    df = pd.read_csv(csv_path, sep=';')

    for i, row in tqdm(df.iterrows()):
        img_path_from_csv = os.path.basename(row['Filename'])
        img_path = origin_images_dir + img_path_from_csv
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # Convert to YOLO format
        x_center, y_center, box_width, box_height = convert_bbox(row['Upper left corner X'], 
                                                                 row['Upper left corner Y'], 
                                                                 row['Lower right corner X'], 
                                                                 row['Lower right corner Y'], w, h)
        class_id = get_class_id(class_map, row['Annotation tag'])
        
        # Write to label file
        image_path = dataset_images_dir + f'/{type}/' + img_path_from_csv
        label_path = dataset_labels_dir + f'/{type}/' + os.path.splitext(img_path_from_csv)[0] + '.txt'
        
        shutil.copy2(img_path, image_path)
        with open(label_path, 'a') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

# val set 
print("Processing val set...")
# for i in range(1, 3):
#     csv_path_day = f'./LISA-traffic-light/Annotations/Annotations/daySequence{i}/'
#     csv_path_night = f'./LISA-traffic-light/Annotations/Annotations/nightSequence{i}/'
#     origin_images_dir_day = f'./LISA-traffic-light/daySequence{i}/daySequence{i}/frames/'
#     origin_images_dir_night = f'./LISA-traffic-light/nightSequence{i}/nightSequence{i}/frames/'

#     make_labels(csv_path_day, origin_images_dir_day, dataset_images_dir, dataset_labels_dir, 'val')
#     make_labels(csv_path_night, origin_images_dir_night, dataset_images_dir, dataset_labels_dir, 'val')

csv_path_day = f'./LISA-traffic-light/Annotations/Annotations/dayTrain/dayClip1/'
origin_images_dir_day = f'./LISA-traffic-light/dayTrain/dayTrain/dayClip1/frames/'
make_labels(csv_path_day, origin_images_dir_day, dataset_images_dir, dataset_labels_dir, 'train')

csv_path_night = f'./LISA-traffic-light/Annotations/Annotations/nightTrain/nightClip1/'
origin_images_dir_night = f'./LISA-traffic-light/nightTrain/nightTrain/nightClip1/frames/'
make_labels(csv_path_night, origin_images_dir_night, dataset_images_dir, dataset_labels_dir, 'train')

# train dayTrain
print("Processing train set day...")
for i in range(2, 14):
    csv_path_day = f'./LISA-traffic-light/Annotations/Annotations/dayTrain/dayClip{i}/'
    origin_images_dir_day = f'./LISA-traffic-light/dayTrain/dayTrain/dayClip{i}/frames/'
    make_labels(csv_path_day, origin_images_dir_day, dataset_images_dir, dataset_labels_dir, 'train')

# train nightTrain
print("Processing train set night...")
for i in range(2, 6):
    csv_path_night = f'./LISA-traffic-light/Annotations/Annotations/nightTrain/nightClip{i}/'
    origin_images_dir_night = f'./LISA-traffic-light/nightTrain/nightTrain/nightClip{i}/frames/'
    make_labels(csv_path_night, origin_images_dir_night, dataset_images_dir, dataset_labels_dir, 'train')

print("class_map: ", class_map)
