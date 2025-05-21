from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import json
import torch
from transformers import BertTokenizer, BertModel


result_json = "datasets/BDD-X-Annotations-finetune-val-output-SmolLM2-vanila-500.json"
result_file_name = "collected-data/SmolLM2-vanila.json"

with open(result_json, "r") as f:
    results = json.load(f)

# video_path_list = []
video_path_list = []
model_output_list_sample = {
    0: [],
    2: []
}
ground_truth_list_sample = {
    0: [],
    2: []
}
model_output_list_traffic = {
    1: [],
    3: []
}
ground_truth_list_traffic = {
    1: [],
    3: []
}

result_dict = {
    "sample-0": {},
    "sample-2": {},
    "traffic-1": {},
    "traffic-3": {}
}

# # extra code for sample and traffic
# result_json_2 = "datasets/BDD-X-Annotations-finetune-val-output-SmolLM2-sample-after800.json"
# with open(result_json_2, "r") as f:
#     results_2 = json.load(f)

# results += results_2

print(f"Result file name: {result_file_name}")
results = results[:500]
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

        model_output_list_sample[j].append(model_output_sample)
        ground_truth_list_sample[j].append(ground_truth_sample)
        model_output_list_traffic[j+1].append(model_output_traffic)
        ground_truth_list_traffic[j+1].append(ground_truth_traffic)
        video_path_list.append(result["video"][0])



# rouge-n 
print("Calculating ROUGE scores...")

for k in [0, 2]:
    rouge = Rouge()
    rouge_scores = rouge.get_scores(model_output_list_sample[0], ground_truth_list_sample[0], avg=True)
    result_dict[f"sample-{k}"]["rouge-1"] = rouge_scores["rouge-1"]
    result_dict[f"sample-{k}"]["rouge-2"] = rouge_scores["rouge-2"]
    result_dict[f"sample-{k}"]["rouge-l"] = rouge_scores["rouge-l"]


for k in [1, 3]:
    rouge = Rouge()
    rouge_scores = rouge.get_scores(model_output_list_traffic[1], ground_truth_list_traffic[1], avg=True)
    result_dict[f"traffic-{k}"]["rouge-1"] = rouge_scores["rouge-1"]
    result_dict[f"traffic-{k}"]["rouge-2"] = rouge_scores["rouge-2"]
    result_dict[f"traffic-{k}"]["rouge-l"] = rouge_scores["rouge-l"]


# bleu
print("Calculating BLEU scores...")
bleu_scores_sample = {
    0: [],
    2: []
}
bleu_scores_traffic = {
    1: [],
    3: []
}

for k in [0, 2]:
    for model_output, ground_truth in zip(model_output_list_sample[k], ground_truth_list_sample[k]):
        model_output_tokens = model_output.split()
        ground_truth_tokens = ground_truth.split()

        bleu_score = sentence_bleu([ground_truth_tokens], model_output_tokens)
        bleu_scores_sample[k].append(bleu_score)

    result_dict[f"sample-{k}"]["bleu"] = sum(bleu_scores_sample[k]) / len(bleu_scores_sample[k])

for k in [1, 3]:
    for model_output, ground_truth in zip(model_output_list_traffic[k], ground_truth_list_traffic[k]):
        model_output_tokens = model_output.split()
        ground_truth_tokens = ground_truth.split()

        bleu_score = sentence_bleu([ground_truth_tokens], model_output_tokens)
        bleu_scores_traffic[k].append(bleu_score)

    result_dict[f"traffic-{k}"]["bleu"] = sum(bleu_scores_traffic[k]) / len(bleu_scores_traffic[k])


# BERTScore
# code from https://www.comet.com/site/blog/bertscore-for-llm-evaluation/?utm_source=chatgpt.com

print("Calculating BERTScore...")

# Load BERT model and tokenizer
MODEL_NAME = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME, device_map="auto")

def get_embeddings(text):
    """
    Generate token embeddings for the input text using BERT.

    Args:
        text (str): Input text or batch of sentences.

    Returns:
        torch.Tensor: Token embeddings with shape (batch_size, seq_len, hidden_dim).
    """
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Move inputs to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)

    # Compute embeddings without gradient calculation
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Return last hidden states (token-level embeddings)
    return outputs.last_hidden_state

def cosine_similarity(generated_embeddings, reference_embeddings):
    """
    Compute cosine similarity between two sets of embeddings.

    Args:
        generated_embeddings (torch.Tensor): Embeddings of candidate tokens with shape (batch_size, seq_len, hidden_dim).
        reference_embeddings (torch.Tensor): Embeddings of reference tokens with shape (batch_size, seq_len, hidden_dim).

    Returns:
        torch.Tensor: Cosine similarity matrix with shape (seq_len_generated, seq_len_reference).
    """
    # Normalize embeddings along the hidden dimension
    generated_embeddings = torch.nn.functional.normalize(generated_embeddings, dim=-1)
    reference_embeddings = torch.nn.functional.normalize(reference_embeddings, dim=-1)

    # Compute similarity using batched matrix multiplication
    return torch.bmm(generated_embeddings, reference_embeddings.transpose(1, 2))

def get_precision(similarity_matrix):
    """
    Calculate BERT precision as the mean of the maximum similarity scores from the candidate to the reference.

    Args:
        similarity_matrix (torch.Tensor): Cosine similarity matrix.

    Returns:
        torch.Tensor: Precision score.
    """
    return similarity_matrix.max(dim=2)[0].mean()

def get_recall(similarity_matrix):
    """
    Calculate BERT recall as the mean of the maximum similarity scores from the reference to the candidate.

    Args:
        similarity_matrix (torch.Tensor): Cosine similarity matrix.

    Returns:
        torch.Tensor: Recall score.
    """
    return similarity_matrix.max(dim=1)[0].mean()

def get_f1_score(precision, recall):
    """
    Compute the F1 score given precision and recall.

    Args:
        precision (torch.Tensor): Precision score.
        recall (torch.Tensor): Recall score.

    Returns:
        torch.Tensor: F1 score.
    """
    return 2 * (precision * recall) / (precision + recall)

def bert_score(candidate, reference):
    """
    Compute BERTScore (Precision, Recall, F1) between a candidate and a reference sentence.

    Args:
        candidate (str): Candidate sentence.
        reference (str): Reference sentence.

    Returns:
        dict: Dictionary containing precision, recall, and F1 scores.
    """
    # Get token embeddings for candidate and reference
    candidate_embeddings = get_embeddings(candidate)
    reference_embeddings = get_embeddings(reference)

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(candidate_embeddings, reference_embeddings)

    # Calculate precision, recall, and F1 scores
    precision = get_precision(similarity_matrix)
    recall = get_recall(similarity_matrix)
    f1_score = get_f1_score(precision, recall)

    # Return scores as a dictionary
    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1_score": f1_score.item(),
    }

bert_score_precisions_sample = {
    0: [],
    2: []
}
bert_score_recalls_sample = {
    0: [],
    2: []
}
bert_score_f1_scores_sample = {
    0: [],
    2: []
}

for k in [0, 2]:
    for model_output, ground_truth in zip(model_output_list_sample[k], ground_truth_list_sample[k]):
        result = bert_score(model_output, ground_truth)
        bert_score_precisions_sample[k].append(result["precision"])
        bert_score_recalls_sample[k].append(result["recall"])
        bert_score_f1_scores_sample[k].append(result["f1_score"])

    result_dict[f"sample-{k}"]["bert_score"] = {
        "precision": sum(bert_score_precisions_sample[k]) / len(bert_score_precisions_sample[k]),
        "recall": sum(bert_score_recalls_sample[k]) / len(bert_score_recalls_sample[k]),
        "f1_score": sum(bert_score_f1_scores_sample[k]) / len(bert_score_f1_scores_sample[k]),
    }

bert_score_precisions_traffic = {
    1: [],
    3: []
}
bert_score_recalls_traffic = {
    1: [],
    3: []
}
bert_score_f1_scores_traffic = {
    1: [],
    3: []
}

for k in [1, 3]:
    for model_output, ground_truth in zip(model_output_list_traffic[k], ground_truth_list_traffic[k]):
        result = bert_score(model_output, ground_truth)
        bert_score_precisions_traffic[k].append(result["precision"])
        bert_score_recalls_traffic[k].append(result["recall"])
        bert_score_f1_scores_traffic[k].append(result["f1_score"])

    result_dict[f"traffic-{k}"]["bert_score"] = {
        "precision": sum(bert_score_precisions_traffic[k]) / len(bert_score_precisions_traffic[k]),
        "recall": sum(bert_score_recalls_traffic[k]) / len(bert_score_recalls_traffic[k]),
        "f1_score": sum(bert_score_f1_scores_traffic[k]) / len(bert_score_f1_scores_traffic[k]),
    }



with open(result_file_name, "w") as f:
    json.dump(result_dict, f, indent=4)
print(f"Results saved to {result_file_name}")
