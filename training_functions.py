#!/usr/bin/env python3
import numpy as np 
import torch 
from itertools import chain
import transformers
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)
from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('/home/jorge/tokenPred/babylm/notebooks/tokenizer')
tokenizer_path = '/home/jorge/tokenPred/babylm_10m/train_files/CustomLlama/models/tokenizer16000.json'
tokenizer = Tokenizer.from_file(str(tokenizer_path))

def get_batch(train_data, block_size, batch_size, next_token=True,dim=1):
    # Generate a set of random indices into the training data.
    ix = torch.randint(low=block_size, high=len(train_data) - block_size, size=(batch_size,)).numpy()
    prev_token_indicator = tokenizer.encode(' <pr>')
    prev_token_len = len(prev_token_indicator)
    if next_token:
        # Next token prediction: inputs are slices of length block_size,
        # targets are the same slices shifted by one position.
        x = torch.stack([train_data[i : i + block_size] for i in ix])
        y = torch.stack([train_data[i + 1 : i + block_size + 1] for i in ix])
    else:
        # Previous token prediction with special '@' token:
        # For x, extract tokens starting at i+1 for block_size-1 tokens,
        # then append the encoded '@' token. Finally, reverse the sequence.
        
        x = torch.stack([
            torch.tensor(train_data[i + 1 : i + block_size +1 - prev_token_len ].tolist() +prev_token_indicator)
            for i in ix
        ])
    
        x = torch.flip(x, dims=[dim])
        
        # For y, extract tokens starting at i for block_size-1 tokens,
        # then append the encoded '@' token. Reverse the sequence as well.
        y = torch.stack([
            torch.tensor(train_data[i : i + block_size - prev_token_len].tolist() + prev_token_indicator)
            for i in ix
        ])
        y = torch.flip(y, dims=[dim])
    
    return x, y

def get_data_batch(tokenized_files, block_size, batch_size, next_token=True,dim=1):
    # we construct an array of random indices that we'll use to sample
    key_list = list(tokenized_files.keys())
    # we sample a random file from the training data    
    r_idx = np.random.randint(0, len(key_list))
    r_file = key_list[r_idx]
    # if there is more than one dataset, we sample a random file
    if len(key_list) > 1:
        train_data = tokenized_files[r_file]['input_ids'][0]
    else:
        #otherwise we just use the file itself
        train_data = tokenized_files['input_ids'][0]
    #we get a batch of data using the get_batch function
    return get_batch(train_data, block_size, batch_size, next_token=next_token,dim=dim)

def combine_tokenized_files(chunks):
    """
    chunks: Sequence of dicts, each with at least
            'input_ids': Tensor of shape [1, L_i] (or [L_i])
            optionally 'attention_mask' of the same shape.
    returns: one dict with
             'input_ids': Tensor [1, sum L_i]
             'attention_mask': Tensor [1, sum L_i]  (only if present)
    """
    # pull out all the input_id tensors
    seqs = [chunks[c]['input_ids'] for c in chunks]
    # decide which dim to concat on (1 if shape [1, L], else 0)
    concat_dim = 1 if seqs[0].dim() == 2 else 0
    out = {'input_ids': torch.cat(seqs, dim=concat_dim)}

    # # if they had masks, merge them too
    # if 'attention_mask' in chunks[0]:
    #     masks = [chunks[c]['attention_mask'] for c in chunks]
    #     out['attention_mask'] = torch.cat(masks, dim=concat_dim)

    return out