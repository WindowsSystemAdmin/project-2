//Prerequisites:
pip install transformers torch tqdm


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Load BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Get word embedding (average over tokens)
def get_word_embedding(word):
    tokens = tokenizer(word, return_tensors='pt', add_special_tokens=False)
    with torch.no_grad():
        output = model(**tokens).last_hidden_state.squeeze(0)
    return output.mean(dim=0)

# Cosine similarity
def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# L2 distance
def l2_dist(a, b):
    return torch.norm(a - b).item()

# Load dataset
def load_dataset(filepath):
    groups = defaultdict(list)
    with open(filepath, 'r') as f:
        current_group = None
        for line in f:
            if line.startswith(':'):
                current_group = line[2:].strip().lower()
            else:
                words = line.strip().lower().split()
                if len(words) == 4 and current_group:
                    groups[current_group].append(tuple(words))
    return groups

# Perform analogy evaluation
def evaluate_group(name, data):
    vocab = set()
    for a, b, c, d in data:
        vocab.update([b, d])
    vocab_embeddings = {w: get_word_embedding(w) for w in vocab}

    results_cosine = {1: 0, 2: 0, 5: 0, 10: 0, 20: 0}
    results_l2 = {1: 0, 2: 0, 5: 0, 10: 0, 20: 0}

    for a, b, c, d in tqdm(data, desc=f'Evaluating {name}'):
        try:
            vec_a = get_word_embedding(a)
            vec_b = vocab_embeddings[b]
            vec_c = get_word_embedding(c)
            target_vec = vocab_embeddings[d]

            vec_ab = vec_a - vec_b
            scores_cosine = []
            scores_l2 = []

            for candidate, vec_d in vocab_embeddings.items():
                if candidate in [b, d]:
                    continue
                vec_cd = vec_c - vec_d
                scores_cosine.append((candidate, cosine_sim(vec_ab, vec_cd)))
                scores_l2.append((candidate, l2_dist(vec_ab, vec_cd)))

            sorted_cosine = sorted(scores_cosine, key=lambda x: -x[1])
            sorted_l2 = sorted(scores_l2, key=lambda x: x[1])

            def is_correct(sorted_list, k):
                return any(candidate == d for candidate, _ in sorted_list[:k])

            for k in results_cosine.keys():
                if is_correct(sorted_cosine, k):
                    results_cosine[k] += 1
                if is_correct(sorted_l2, k):
                    results_l2[k] += 1
        except:
            continue

    total = len(data)
    print(f"\n=== Results for group: {name.upper()} ===")
    print(f"{'k':<5}{'Accuracy Using Cosine Similarity':<35}{'Accuracy Using L2 Distance'}")
    for k in results_cosine:
        acc_cosine = 100 * results_cosine[k] / total
        acc_l2 = 100 * results_l2[k] / total
        print(f"{k:<5}{acc_cosine:<35.2f}{acc_l2:.2f}")

# === Run the full evaluation ===
if __name__ == "__main__":
    filepath = 'word-test.v1.txt'  # Replace with full path if needed
    all_groups = load_dataset(filepath)

    for group in ['family', 'city-in-state', 'currency']:
        if group in all_groups:
            evaluate_group(group, all_groups[group])
        else:
            print(f"Group {group} not found in the dataset.")
