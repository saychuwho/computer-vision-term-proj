from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import json

result_json = "datasets/BDD-X-Annotations-finetune-val-output-SmolLM2-sample-800.json"
result_file_name = "collected-data/SmolLM2-sample.json"

with open(result_json, "r") as f:
    results = json.load(f)

# video_path_list = []
video_path_list = []
model_output_list_sample = []
ground_truth_list_sample = []
model_output_list_traffic = []
ground_truth_list_traffic = []

result_dict = {
    "sample": {},
    "traffic": {}
}

# extra code for sample and traffic
result_json_2 = "datasets/BDD-X-Annotations-finetune-val-output-SmolLM2-sample-after800.json"
with open(result_json_2, "r") as f:
    results_2 = json.load(f)

results += results_2

print(f"Result file name: {result_file_name}")
print(f"Number of results: {len(results)}")
print("Processing results...")
for result in results:
    # video_path = result["video"][0]
    for j in [0, 2]:
        model_output_sample = result["conversations"][j]["answer"]
        ground_truth_sample = result["conversations"][j]["ground_truth"]

        model_output_sample_assist_idx = model_output_sample.split().index("Assistant:")
        model_output_sample = ' '.join(model_output_sample.split()[model_output_sample_assist_idx+1:])

        model_output_traffic = result["conversations"][j+1]["answer"]
        ground_truth_traffic = result["conversations"][j+1]["ground_truth"]
        
        model_output_traffic_assist_idx = model_output_traffic.split().index("Assistant:")
        model_output_traffic = ' '.join(model_output_traffic.split()[model_output_traffic_assist_idx+1:])

        model_output_list_sample.append(model_output_sample)
        ground_truth_list_sample.append(ground_truth_sample)
        model_output_list_traffic.append(model_output_traffic)
        ground_truth_list_traffic.append(ground_truth_traffic)
        video_path_list.append(result["video"][0])



# rouge-n 
print("Calculating ROUGE scores...")

for k, sample in enumerate(model_output_list_sample):
    if len(sample) == 0:
        print("Empty sample found")
        print(video_path_list[k])
        print(sample)


rouge = Rouge()
rouge_scores = rouge.get_scores(model_output_list_sample, ground_truth_list_sample, avg=True)
result_dict["sample"]["rouge-1"] = rouge_scores["rouge-1"]
result_dict["sample"]["rouge-2"] = rouge_scores["rouge-2"]
result_dict["sample"]["rouge-l"] = rouge_scores["rouge-l"]

rouge = Rouge()
rouge_scores = rouge.get_scores(model_output_list_traffic, ground_truth_list_traffic, avg=True)
result_dict["traffic"]["rouge-1"] = rouge_scores["rouge-1"]
result_dict["traffic"]["rouge-2"] = rouge_scores["rouge-2"]
result_dict["traffic"]["rouge-l"] = rouge_scores["rouge-l"]

# bleu
print("Calculating BLEU scores...")
bleu_scores_sample = []
bleu_scores_traffic = []

for model_output, ground_truth in zip(model_output_list_sample, ground_truth_list_sample):
    model_output_tokens = model_output.split()
    ground_truth_tokens = ground_truth.split()

    bleu_score = sentence_bleu([ground_truth_tokens], model_output_tokens)
    bleu_scores_sample.append(bleu_score)

result_dict["sample"]["bleu"] = sum(bleu_scores_sample) / len(bleu_scores_sample)

for model_output, ground_truth in zip(model_output_list_traffic, ground_truth_list_traffic):
    model_output_tokens = model_output.split()
    ground_truth_tokens = ground_truth.split()

    bleu_score = sentence_bleu([ground_truth_tokens], model_output_tokens)
    bleu_scores_traffic.append(bleu_score)

result_dict["traffic"]["bleu"] = sum(bleu_scores_traffic) / len(bleu_scores_traffic)

with open(result_file_name, "w") as f:
    json.dump(result_dict, f, indent=4)
print(f"Results saved to {result_file_name}")
