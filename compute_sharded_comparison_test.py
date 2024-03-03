import os
import math
import random
import numpy as np
from scipy.stats import t as tdist

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm.auto import tqdm

import json
import fire

os.environ['TOKENIZERS_PARALLELISM'] = "True"

flatten = lambda l : [x for s in l for x in s]
shuffle = lambda l : random.sample(l, k=len(l))


def load_dataset(dataset_path):
    # For loading a JSON-serialized list of examples.
    if dataset_path.endswith(".json"):
        print("loading from json...")
        with open(dataset_path, "r") as f:
            data = f.read()
            examples = json.loads(data)
            return examples

    # For loading a dataset where each example is on its own line.
    with open(dataset_path, "r") as f:
        lines = f.readlines()
    return lines


def compute_logprob_of_token_sequence(tokens, model, context_len=2048, stride=1024, device=0):
    inputs = tokens[:-1]
    targets = tokens[1:]

    logp = torch.zeros((1, 1), dtype=torch.float32).to(device)

    # compute the smallest multiple k of s so that t <= ks + c.
    t = len(inputs)
    c = context_len
    s = stride
    k = math.ceil(max(0, t - c) / s)
    for j in range(k + 1):
        start = s * j
        end = min(s * j + c, t)
        rel_offs = max(0, c - s) if j > 0 else 0

        w_inp = inputs[start:end]
        w_inp = torch.tensor(w_inp).to(device)
        w_trg = targets[start:end]
        w_trg = torch.tensor(w_trg).to(device)

        model.eval()
        with torch.no_grad():
            out = model(torch.unsqueeze(w_inp, 0))
            logps = torch.nn.functional.log_softmax(out.logits[0], dim=-1)
            logps = logps.gather(-1, w_trg.unsqueeze(-1)).squeeze(-1)
            logp += logps[rel_offs:].sum()

        del w_inp
        del w_trg
        torch.cuda.empty_cache()

    return logp.item()


def main(model_name_or_path, dataset_path, context_len=2048, stride=1024, num_shards=50,
         permutations_per_shard=250, random_seed=0, max_examples=5000, device="cuda"):
    # Set random seed(s).
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Load the dataset.
    examples = load_dataset(dataset_path)
    examples = examples[:max_examples]
    num_examples = len(examples)
    print(f"Loaded {num_examples} examples from {dataset_path}")
    
    # Load tokenizer and tokenize the examples.
    t = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenized_examples = [t.encode(ex) for ex in examples]
    
    # Compute the number of examples for each shard.
    shard_idx = enumerate([num_examples // num_shards] * num_shards)
    shard_counts = [(x + 1 if i < num_examples % num_shards else x) for i, x in shard_idx]
    shard_bounds = [0] + np.cumsum(np.asarray(shard_counts)).tolist()
    
    canon, shuffled = [], []
    
    m = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    m.to(device)
    
    for start, end in tqdm(zip(shard_bounds, shard_bounds[1:])):
        cur_tokens = flatten(tokenized_examples[start:end])
        canon.append(compute_logprob_of_token_sequence(cur_tokens, m, context_len, stride, device))
        for _ in range(permutations_per_shard):
            shuffled.append(compute_logprob_of_token_sequence(shuffle(cur_tokens), m, context_len, stride, device))
    
    return canon, shuffled


def t_test(canon, shuffled):
    diffs = canon - shuffled.mean(axis=1)
    z = np.mean(diffs) / np.std(diffs) * np.sqrt(len(diffs))
    pval = 1 - tdist.cdf(z, df=len(diffs)-1)
    return pval
