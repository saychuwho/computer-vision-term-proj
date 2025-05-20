from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import json
from tqdm import tqdm

device="cuda:0"
model_path = "./finetuned-model/smollm2-sample"
preproc_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2"
).to("cuda")
processor = AutoProcessor.from_pretrained(preproc_path)


def get_response(text_prompt, video_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": text_prompt}
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return generated_texts[0]


annotation_val_json_traffic = 'datasets/BDD-X-Annotations-finetune-val-traffic.json'
annotation_val_json = 'datasets/BDD-X-Annotations-finetune-val.json'

save_json = 'datasets/BDD-X-Annotations-finetune-val-output-SmolLM2-sample-after800.json'

print("save file: ", save_json)

with open(annotation_val_json, 'r') as f:
    annotations_normal = json.load(f)
with open(annotation_val_json_traffic, 'r') as f:
    annotations_traffic = json.load(f)

# partial save due to kill
qa_results = []
counter = 0

annotations_normal_part = annotations_normal[801:]
annotations_traffic_part = annotations_traffic[801:]

# for normal_conv, traffic_conv in tqdm(zip(annotations_normal, annotations_traffic),
#                                       total=len(annotations_normal),
#                                       desc="Processing videos"):
for normal_conv, traffic_conv in tqdm(zip(annotations_normal_part, annotations_traffic_part),
                                      total=len(annotations_normal_part),
                                      desc="Processing videos"):
    
    video_path = normal_conv["video"][0]
    

    # if not is_video_valid(video_path):
    #     print(f"Invalid video file: {video_path}")
    #     continue

    responses = {}
    responses["video"] = [video_path]
    conversations = []
    for i in [0, 2]:
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


    # if counter != 0 and counter % 200 == 0:
    #     save_json_tmp = f'datasets/BDD-X-Annotations-finetune-val-output-SmolLM2-sample-{counter}.json'
        
    #     with open(save_json_tmp, 'w') as f:
    #         json.dump(qa_results, f, indent=4)

    # counter += 1

with open(save_json, 'w') as f:
    json.dump(qa_results, f, indent=4)
    print(f"Saved results to {save_json}")