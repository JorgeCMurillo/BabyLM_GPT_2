#!/usr/bin/env python3
import torch
from typing import List, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch

def per_token_log_likelihood(model, tokenizer, input_texts, device="cuda"):
    # Tokenize the batch with padding
    # return_tensors="pt" creates a rectangular tensor (Batch, Max_Seq_Len)
    inputs = tokenizer(input_texts, add_special_tokens=False, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(device)
    attn_mask = inputs.attention_mask.to(device)
    
    # Get batch size
    batch_size = input_ids.shape[0]

    # Prepend BOS token (Batch-wise)
    bos_token_id = tokenizer.bos_token_id
    bos_tensor = torch.full((batch_size, 1), bos_token_id, device=device)
    input_ids = torch.cat([bos_tensor, input_ids], dim=1)

    # Prepend Attention Mask (Batch-wise)
    ones_tensor = torch.ones((batch_size, 1), device=device)
    attn_mask = torch.cat([ones_tensor, attn_mask], dim=1)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs['logits']

    # Remove last logit (shift left)
    logits = logits[:, :-1, :]
    log_probs = torch.log_softmax(logits, dim=-1)

    # Remove first input_id (BOS) for aligning labels
    input_ids = input_ids[:, 1:]

    # Gather log probs at the specific input indices
    token_logprobs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

    # Note: token_logprobs now contains values for PADDING tokens at the end too.
    # We return the whole thing and let the caller slice it.
    return token_logprobs, inputs.attention_mask

def per_token_conditional_log_likelihood(model, tokenizer, contexts, targets, device="cuda"):
    # 1. Prepare batch of texts exactly as you did
    texts = [c + " " + t for c, t in zip(contexts, targets)]
    
    # 2. Calculate context lengths for each item in the batch
    # We do this in a loop because they can be different lengths
    context_token_lengths = [len(tokenizer.encode(c, add_special_tokens=False)) for c in contexts]

    # 3. Run the model on the batch (High Performance)
    batch_log_probs, batch_attn_masks = per_token_log_likelihood(model, tokenizer, texts, device)

    # 4. Slice the results row-by-row to match your original logic
    results = []
    for i in range(len(contexts)):
        # Get the start index for this specific row
        start_idx = context_token_lengths[i]
        
        # Get the total length of valid tokens (excluding padding)
        # sum() gives the count of real tokens (non-padded)
        valid_length = batch_attn_masks[i].sum().item()
        
        # Slice: from context_end up to the actual end of the sentence (ignoring padding)
        # We use valid_length because batching adds padding zeros at the end which we don't want
        row_result = batch_log_probs[i, start_idx:valid_length]
        
        results.append(row_result)

    return results
def per_token_conditional_log_likelihood(model, tokenizer, contexts, targets, device="cuda", batch_size=8):
    all_results = []
    
    # Process data in chunks of batch_size
    for i in range(0, len(contexts), batch_size):
        # 1. Slice the current batch
        batch_contexts = contexts[i : i + batch_size]
        batch_targets = targets[i : i + batch_size]
        
        # 2. Prepare batch of texts (Context + " " + Target)
        batch_texts = [c + " " + t for c, t in zip(batch_contexts, batch_targets)]
        
        # 3. Calculate context lengths for this specific batch
        #    (We need this to know where to start slicing the probabilities)
        batch_context_lengths = [len(tokenizer.encode(c, add_special_tokens=False)) for c in batch_contexts]

        # 4. Run the model on the current batch
        #    per_token_log_likelihood handles padding and tokenization internally
        batch_log_probs, batch_attn_masks = per_token_log_likelihood(model, tokenizer, batch_texts, device)

        # 5. Process results for this batch
        for j in range(len(batch_contexts)):
            # Get the start index for this specific row
            start_idx = batch_context_lengths[j]
            
            # Get the total length of valid tokens (excluding padding)
            valid_length = batch_attn_masks[j].sum().item()
            
            # Slice: from context_end up to the actual end of the sentence
            # We assume batch_log_probs[j] corresponds to batch_texts[j]
            row_result = batch_log_probs[j, start_idx:valid_length]
            
            all_results.append(row_result)

    return all_results
import pandas as pd
from pathlib import Path

SRC = Path("/home/jorge/tokenPred/babylm_10m/test_eval/evaluation-pipeline-2025/evaluation_data/fast_eval/ewok_fast")          # adjust path if needed

ewok_df = pd.concat(
    [
        pd.read_json(fp, lines=True, encoding="utf-8")
        #   .assign(file=fp.name, category=fp.stem)  # metadata
        for fp in SRC.glob("*.jsonl")              # use rglob("*.jsonl") if nested
    ],
    ignore_index=True,
    sort=False,    # keep union of columns
)

# optional niceties
ewok_df = ewok_df.convert_dtypes()
def evaluate(model,tokenizer):
    domains = ewok_df['Domain'].unique()
    # domain = 'spatial-relations'
    domain_scores_official = {}
    domain_scores_full = {}
    for domain in domains:
        print(domain)
        bs = 2
        df = ewok_df[ewok_df['Domain'] == domain]
        # print(len(df))
        #we apply the per_token_conditional_log_likelihood function to the ewok dataset for each row
        context1 = df["Context1"].tolist()
        target1 = df["Target1"].tolist()
        context2 = df["Context2"].tolist()
        target2 = df["Target2"].tolist()
        #we check to make sure the model assigns higher likelihood P(T_1|C_1) > P(T_1|C_2) and P(T_2|C_2) > P(T_2|C_1)
        results_1_1 = per_token_conditional_log_likelihood(model, tokenizer, context1, target1, batch_size=bs)
        results_1_2 = per_token_conditional_log_likelihood(model, tokenizer, context1, target2, batch_size=bs)
        results_2_2 = per_token_conditional_log_likelihood(model, tokenizer, context2, target2, batch_size=bs)
        results_2_1 = per_token_conditional_log_likelihood(model, tokenizer, context2, target1, batch_size=bs)
        import numpy as np
        correct_1 = []
        for r1, r2 in zip(results_1_1, results_1_2):
            sum_r1 = r1.sum().item()
            sum_r2 = r2.sum().item()
            correct_1.append(sum_r1 > sum_r2)
        correct_2 = []
        for r1, r2 in zip(results_2_2, results_2_1):
            sum_r1 = r1.sum().item()
            sum_r2 = r2.sum().item()
            correct_2.append(sum_r1 > sum_r2)
        accuracy_1 = np.mean(correct_1)
        accuracy_2 = np.mean(correct_2)
        print(f"Accuracy for Target1: {accuracy_1*100:.2f}%")
        print(f"Accuracy for Target2: {accuracy_2*100:.2f}%")
        domain_scores_full[domain] = (accuracy_1.item(), accuracy_2.item())
        # we store only the accuracy for Target1 as official score,
        #as that matches the original ewok evaluation protocol
        domain_scores_official[domain] = accuracy_1.item() 
    return domain_scores_official, domain_scores_full