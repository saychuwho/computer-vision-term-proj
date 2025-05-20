
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import json
from tqdm import tqdm
import subprocess

device = "cuda:0"
model_path = "finetuned-model/videollama3-finetuned-traffic-16856"
preproc_path = "DAMO-NLP-SG/VideoLLaMA3-2B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(preproc_path, trust_remote_code=True)

def get_response(text_prompt, video_path):
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 180}},
                {"type": "text", "text": text_prompt},
            ]
        },
    ]

    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    output_ids = model.generate(**inputs, max_new_tokens=256)

    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return response


# def is_video_valid(video_path):
#     """Returns True if ffprobe can read the video file, False otherwise."""
#     try:
#         result = subprocess.run(
#             [
#                 "ffprobe",
#                 "-v", "error",
#                 "-select_streams", "v:0",
#                 "-show_entries", "stream=codec_name",
#                 "-of", "default=noprint_wrappers=1:nokey=1",
#                 video_path
#             ],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             timeout=10
#         )
#         return result.returncode == 0 and result.stdout.strip() != b""
#     except Exception as e:
#         print(f"ffprobe failed for {video_path}: {e}")
#         return False


annotation_val_json_traffic = 'datasets/BDD-X-Annotations-finetune-val-traffic.json'
annotation_val_json = 'datasets/BDD-X-Annotations-finetune-val.json'

save_json = 'datasets/BDD-X-Annotations-finetune-val-output-videoLLaMA3-traffic.json'

with open(annotation_val_json, 'r') as f:
    annotations_normal = json.load(f)
with open(annotation_val_json_traffic, 'r') as f:
    annotations_traffic = json.load(f)

qa_results = []

print(f"save file: {save_json}")

for normal_conv, traffic_conv in tqdm(zip(annotations_normal, annotations_traffic),
                                      total=len(annotations_normal),
                                      desc=f"Processing videos "):
    video_path = normal_conv["video"][0]
    

    # if not is_video_valid(video_path):
    #     print(f"Invalid video file: {video_path}")
    #     continue

    responses = {}
    responses["video"] = [video_path]
    conversations = []
    for i in range(0, 3, 2):
        normal_text_prompt = normal_conv["conversations"][i]["value"]
        traffic_text_prompt = traffic_conv["conversations"][i]["value"]

        normal_response = get_response(normal_text_prompt, video_path)
        traffic_response = get_response(traffic_text_prompt, video_path)

        conversations.append({
            "question": normal_text_prompt,
            "answer": normal_response,
            "ground_truth": normal_conv["conversations"][i+1]["value"]
        })
        conversations.append({
            "question": traffic_text_prompt,
            "answer": traffic_response,
            "ground_truth": traffic_conv["conversations"][i+1]["value"]
        })
    
    responses["conversations"] = conversations

    qa_results.append(responses)

with open(save_json, 'w') as f:
    json.dump(qa_results, f, indent=4)
    print(f"Saved results to {save_json}")
