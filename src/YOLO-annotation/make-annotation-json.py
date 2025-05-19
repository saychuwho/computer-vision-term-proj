import json
import os
from tqdm import tqdm

traffic_light_classes = {
    0: 'stop',
    1: 'go',
    2: 'warning',
    3: 'warningLeft',
    4: 'stopLeft',
    5: 'goLeft',
    6: 'goForward'
}

traffic_sign_classes = {
    0: 'stop',
    1: 'yield',
    2: 'yieldAhead',
    3: 'merge',
    4: 'signalAhead',
    5: 'pedestrianCrossing',
    6: 'keepRight',
    7: 'speedLimit35',
    8: 'speedLimit25'
}


original_annotation_file = 'datasets/BDD-X-Annotations-finetune-val.json'
save_annotation_file = 'datasets/BDD-X-Annotations-finetune-val-traffic.json'

with open(original_annotation_file, 'r') as f:
    original_annotations = json.load(f)

save_annotation_list = []

for annotation in tqdm(original_annotations):

    save_video_dict = {}

    video_path = annotation["video"][0]
    
    save_video_dict["video"] = [video_path]
    save_video_dict["conversations"] = []

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    video_traffic_light = {}
    video_traffic_sign = {}

    video_traffic_light_result = "YOLO-labels/traffic-light/" + video_name + ".txt"
    video_traffic_sign_result = "YOLO-labels/traffic-sign/" + video_name + ".txt"

    # get traffic light annotations
    with open(video_traffic_light_result, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            frame_idx = int(parts[0])
            class_id = int(parts[1])
            x_center = float(parts[2])
            y_center = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])

            if class_id not in video_traffic_light:
                video_traffic_light[class_id] = []
            
            video_traffic_light[class_id].append({
                "frame": frame_idx,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height
            })
        
    # get traffic sign annotations
    with open(video_traffic_sign_result, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            frame_idx = int(parts[0])
            class_id = int(parts[1])
            x_center = float(parts[2])
            y_center = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])

            if class_id not in video_traffic_sign:
                video_traffic_sign[class_id] = []
            
            video_traffic_sign[class_id].append({
                "frame": frame_idx,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height
            })

    # add traffic light conversations
    save_video_dict["conversations"].append({
        "from": "human",
        "value": "Can you see the traffic light in the video?"
    })

    if video_traffic_light:
        gpt_response = "Yes, I can see the traffic light. I can see a "
        for i, class_id in enumerate(video_traffic_light):
            if i < len(video_traffic_light) - 1:
                gpt_response += traffic_light_classes[class_id] + ", "
            else:
                gpt_response += traffic_light_classes[class_id] + "."
    else:
        gpt_response = "No, I can't see the traffic light."

    save_video_dict["conversations"].append({
        "from": "gpt",
        "value": gpt_response
    })

    # add traffic sign conversations
    save_video_dict["conversations"].append({
        "from": "human",
        "value": "Can you see the traffic sign in the video?"
    })

    if video_traffic_sign:
        gpt_response = "Yes, I can see the traffic sign. I can see a "
        for i, class_id in enumerate(video_traffic_sign):
            if i < len(video_traffic_sign) - 1:
                gpt_response += traffic_sign_classes[class_id] + ", "
            else:
                gpt_response += traffic_sign_classes[class_id] + "."
    else:
        gpt_response = "No, I can't see the traffic sign."

    save_video_dict["conversations"].append({
        "from": "gpt",
        "value": gpt_response
    })

    save_annotation_list.append(save_video_dict)

# save to json file
with open(save_annotation_file, 'w') as f:
    json.dump(save_annotation_list, f, indent=4)

print(f"Annotations saved to {save_annotation_file}")