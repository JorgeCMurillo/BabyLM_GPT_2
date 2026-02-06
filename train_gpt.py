# #!/usr/bin/env python

import os, pickle, random

import time
import argparse
import json
from itertools import islice  # <--- Added for efficient slicing
import morfessor
from transformers import PreTrainedTokenizer
import ewok_eval
from ewok_eval import evaluate
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from contextlib import nullcontext

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from transformers import (
    AutoTokenizer, GPT2Config, GPT2LMHeadModel,
    get_cosine_schedule_with_warmup
)
from accelerate import Accelerator
from huggingface_hub import login

# Login to Hugging Face
HF_TOKEN_PATH = os.path.join(os.path.dirname(__file__), "hf_token.txt")
with open(HF_TOKEN_PATH, "r", encoding="utf-8") as f:
    HF_TOKEN = f.read().strip()
login(token=HF_TOKEN)

# import os, json
from datetime import datetime

def to_jsonable(x):
    """Convert tensors / numpy / scalars inside dicts to JSON-safe Python types."""
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
    except Exception:
        pass
    if torch.is_tensor(x):
        return x.detach().cpu().tolist() if x.ndim > 0 else x.item()
    return x

def save_epoch_metrics(metrics_list, out_path):
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    os.replace(tmp_path, out_path)  # atomic write

class ChunkedDataset(Dataset):
    """
    A simple PyTorch Dataset that wraps a list of token chunks.
    """
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return torch.tensor(self.chunks[idx], dtype=torch.long)


def set_all_seeds(seed_value):
    """Set seed for reproducibility."""    
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    random.seed(seed_value)
import random
import numpy as np




def main(seed, batch_size, num_epochs,tokenizer_name) -> None:
    import re
    # ---------- house-keeping ----------
    set_all_seeds(seed)
    print('seed number:', seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    accelerator = Accelerator(gradient_accumulation_steps=1)
    device = accelerator.device

    # ---------- data paths ----------
    # Using the last path from your snippets
    if tokenizer_name != 'gpt2':
        chunked_path = '/home/jorge/tokenPred/babylm_10m/train_files/CustomLlama/data/babylm_dev_clean/tokenized_morf_data.pkl'
        val_chunked_path = "/home/jorge/tokenPred/babylm_10m/train_files/CustomLlama/data/babylm_dev_clean/tokenized_morf_data_val.pkl"

    else:
        chunked_path = "/home/jorge/tokenPred/babylm_10m/train_files/CustomLlama/data/babylm_10M_clean/bbylm_agentic_mixed.pkl"
        val_chunked_path = "/home/jorge/tokenPred/babylm_10m/train_files/CustomLlama/data/babylm_dev_clean/babylm_dev_data.pkl"
    
    print('chunked_path:', chunked_path)
    
    out_dir = 'babygpt-10m-trial_run_Dec_'+'CustomAG_' + str(batch_size) +'_' +str(seed)+ '_epochs' + str(num_epochs)

    if accelerator.is_main_process:
        os.makedirs(out_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    print("Loading token-chunks …")
    with open(chunked_path, "rb") as f:
        chunks = pickle.load(f)
    
    with open(val_chunked_path, "rb") as f:
        val_chunks = pickle.load(f)
        
    print(f"Loaded {len(chunks):,} training chunks")
    print(f"Loaded {len(val_chunks):,} validation chunks")

    # ---------- hyper-parameters ----------
    NUM_EPOCHS         = num_epochs
    BASE_BATCH_SIZE    = batch_size 
    BASE_LEARNING_RATE = 5e-5
    BATCH_SIZE         = BASE_BATCH_SIZE
    WEIGHT_DECAY       = 0.0
    WARMUP_RATIO       = 0.1
    print('BATCH_SIZE:', BATCH_SIZE)
    print('NUM_EPOCHS:', NUM_EPOCHS)

    # ---------- tokenizer ----------
    if tokenizer_name != 'gpt2':
        # ---------- Pretokenization helpers ----------

        TOKEN_RE = re.compile(
            r"[^\W\d_]+(?:'[^\W\d_]+)?|"   # unicode letters + simple contractions (no underscores)
            r"\d+|"                        # numbers
            r"\.\.\.|…|—|–|"               # multi-char punctuation
            r"[^\s]"                       # any other single non-space char (punct/symbol)
        )

        def pretokenize(text: str):
            # Collapse Gutenberg italics markers: _word_ -> word
            text = re.sub(r"_([^_]+)_", r"\1", text)
            return TOKEN_RE.findall(text)

        def is_wordlike(tok: str) -> bool:
            # Treat anything with at least one alphabetic char as a "word"
            return any(c.isalpha() for c in tok)


        # ---------- Morfessor Tokenizer ----------

        class MorfessorTokenizer(PreTrainedTokenizer):
            vocab_files_names = {
                "vocab_file": "vocab.json",
                "model_file": "morfessor.bin",
            }

            def __init__(self, vocab_file=None, model_file=None, **kwargs):
                # ✅ Prevent duplicate special tokens passed by from_pretrained()
                pad_token = kwargs.pop("pad_token", "<pad>")
                unk_token = kwargs.pop("unk_token", "<unk>")
                bos_token = kwargs.pop("bos_token", "<bos>")
                eos_token = kwargs.pop("eos_token", "<eos>")

                # 1) Default vocab (stable IDs)
                self.vocab = {pad_token: 0, unk_token: 1, bos_token: 2, eos_token: 3}
                self.inv_vocab = {i: t for t, i in self.vocab.items()}

                # 2) Load vocab.json if provided
                if vocab_file and os.path.exists(vocab_file):
                    with open(vocab_file, "r", encoding="utf-8") as f:
                        self.vocab = json.load(f)
                    self.inv_vocab = {int(idx): tok for tok, idx in self.vocab.items()}

                # 3) Morfessor model
                self.io = morfessor.MorfessorIO()
                self.model = (
                    self.io.read_binary_model_file(model_file)
                    if (model_file and os.path.exists(model_file))
                    else morfessor.BaselineModel()
                )
                self.model_file = model_file

                # 4) HF init
                super().__init__(
                    pad_token=pad_token,
                    unk_token=unk_token,
                    bos_token=bos_token,
                    eos_token=eos_token,
                    **kwargs
                )

            def train_from_texts(self, texts, model_save_path=None):
                # ---- Build word counts (TRAIN morfessor only on word-like tokens) ----
                word_counts = {}
                for text in texts:
                    for tok in pretokenize(text):
                        if is_wordlike(tok):
                            word_counts[tok] = word_counts.get(tok, 0) + 1

                data = [(count, word) for word, count in word_counts.items()]

                # ---- Train Morfessor ----
                self.model.load_data(data)
                self.model.train_batch()

                # ---- Optionally save morfessor model ----
                if model_save_path:
                    self.io.write_binary_model_file(model_save_path, self.model)
                    self.model_file = model_save_path

                # ---- Build vocab from: special tokens + punctuation tokens + morphemes ----
                specials = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
                all_tokens = set(specials)

                for text in texts:
                    # This returns punctuation as tokens + morphemes for words
                    all_tokens.update(self._tokenize(text))

                # ✅ Keep special IDs stable, put everything else after
                rest = sorted(all_tokens - set(specials))
                self.vocab = {tok: i for i, tok in enumerate(specials + rest)}
                self.inv_vocab = {i: tok for tok, i in self.vocab.items()}

            def _tokenize(self, text):
                tokens = []
                for tok in pretokenize(text):
                    if is_wordlike(tok):
                        segs, _ = self.model.viterbi_segment(tok)
                        tokens.extend(segs)
                    else:
                        # punctuation/symbols stay as their own token (".", "!", "—", etc.)
                        tokens.append(tok)
                return tokens

            def _convert_token_to_id(self, token):
                return self.vocab.get(token, self.vocab[self.unk_token])

            def _convert_id_to_token(self, idx):
                return self.inv_vocab.get(int(idx), self.unk_token)

            def get_vocab(self):
                return dict(self.vocab)

            @property
            def vocab_size(self):
                return len(self.vocab)

            def save_vocabulary(self, save_directory, filename_prefix=None):
                os.makedirs(save_directory, exist_ok=True)

                vocab_path = os.path.join(save_directory, "vocab.json")
                with open(vocab_path, "w", encoding="utf-8") as f:
                    json.dump(self.vocab, f, ensure_ascii=False, indent=2)

                model_path = os.path.join(save_directory, "morfessor.bin")
                self.io.write_binary_model_file(model_path, self.model)

                return (vocab_path, model_path)

        tokenizer = MorfessorTokenizer.from_pretrained('/home/jorge/tokenPred/babylm_10m/train_files/dataOpt/Notebooks/tok_test')
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token       # pad ≡ EOS
    EOS_ID = tokenizer.pad_token_id                 # 50256
    print(tokenizer)


    def collate_fn(batch, PAD_ID):
        lengths = torch.tensor([x.numel() for x in batch], dtype=torch.long)
        tokens = pad_sequence(batch, batch_first=True, padding_value=PAD_ID) 
        B, T = tokens.shape

        # length-based attention mask
        ar = torch.arange(T).unsqueeze(0) 
        attn_mask = (ar < lengths.unsqueeze(1)).to(tokens.device)

        input_ids = tokens[:, :-1]
        labels    = tokens[:, 1:].clone()
        
        # loss mask: valid targets are those whose label position is not padding
        loss_mask = attn_mask[:, 1:]

        return input_ids, labels, loss_mask, attn_mask[:, :-1]

    # ---------- dataloaders ----------
    dataset = ChunkedDataset(chunks)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, PAD_ID=EOS_ID)
    )

    val_dataset = ChunkedDataset(val_chunks)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, PAD_ID=EOS_ID)
    )
    if tokenizer_name != 'gpt2':
            # ---------- model ----------
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=EOS_ID,
            eos_token_id=EOS_ID,
            unk_token_id=EOS_ID,
            n_ctx=1024, n_positions=1024,
            n_embd=768, n_head=12, n_layer=12
        )
    else:

        # ---------- model ----------
        config = GPT2Config(
            vocab_size=50257,
            bos_token_id=EOS_ID,
            eos_token_id=EOS_ID,
            unk_token_id=EOS_ID,
            n_ctx=1024, n_positions=1024,
            n_embd=768, n_head=12, n_layer=12
        )
    model = GPT2LMHeadModel(config)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---------- optimizer & scheduler ----------
    num_gpus = accelerator.num_processes
    LEARNING_RATE = BASE_LEARNING_RATE * num_gpus
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE,
                      betas=(0.9, 0.999), eps=1e-8,
                      weight_decay=WEIGHT_DECAY)

    total_steps = NUM_EPOCHS * len(loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps
    )

    # ---------- prepare for DDP ----------
    # IMPORTANT: Pass val_loader here
    model, optimizer, loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, loader, val_loader, scheduler
    )

    # Lists to store history for plotting
    train_loss_history = []
    val_loss_history = []

    # --------------------------------------------------
    #  Train Epoch Function (Nested to access tokenizer/device)
    # --------------------------------------------------
    def train_epoch(
            model, optimizer, scheduler, dataloader, accelerator,
            epoch, val_dataloader=None, train_history=None, val_history=None,
            out_dir=None, gradient_clip_norm=-1.0, log_every=200
    ):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        win_loss = 0.0
        win_tokens = 0
        global_step = 0
        
        # Validate 10 times per epoch
        eval_interval = max(1, len(dataloader) // 10)
        
        # Limit validation to 50 batches to prevent hanging/slowness
        VAL_BATCH_LIMIT = 50

        autocast_ctx = accelerator.autocast if hasattr(accelerator, "autocast") else nullcontext

        for step, (input_ids, labels, loss_mask, attention) in enumerate(tqdm(dataloader,
                                                                desc=f"Epoch {epoch}",
                                                                disable=not accelerator.is_local_main_process)):

            # --- Forward ---
            with autocast_ctx():
                logits = model(input_ids=input_ids, attention_mask=attention)['logits']
                log_probs = torch.log_softmax(logits, dim=-1)
                
                safe_labels = labels.clamp(min=0)
                token_log_probs = torch.gather(log_probs, -1, safe_labels.unsqueeze(-1)).squeeze(-1)
                
                tgt_mask = loss_mask.float()
                # prevent div by zero
                denom = tgt_mask.sum()
                if denom > 0:
                    loss = -(token_log_probs * tgt_mask).sum() / denom
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

            # --- Backward ---
            accelerator.backward(loss)
            if gradient_clip_norm > 0:
                clip_grad_norm_(model.parameters(), gradient_clip_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # --- Tracking ---
            bs_tokens = int(tgt_mask.sum().item())
            loss_val = loss.item()
            total_loss += loss_val * bs_tokens
            total_tokens += bs_tokens
            win_loss += loss_val * bs_tokens
            win_tokens += bs_tokens
            global_step += 1
            
            # Store Train History
            if train_history is not None:
                curr_idx = (epoch - 1) * len(dataloader) + step
                train_history.append((curr_idx, loss_val))

            if global_step % log_every == 0 and accelerator.is_local_main_process:
                avg = win_loss / win_tokens if win_tokens > 0 else 0
                print(f"[Ep {epoch:02d} | Stp {global_step:05d}] last={loss_val:.4f} avg_token={avg:.4f}")
                win_loss = 0.0
                win_tokens = 0

            # --- Validation Logic (Fast Subset) ---
            if val_dataloader is not None and step % eval_interval == 0:
                model.eval()
                val_loss_accum = 0.0
                val_tokens_accum = 0
                
                with torch.no_grad():
                    # Check only first 50 batches!
                    for v_batch in islice(val_dataloader, VAL_BATCH_LIMIT):
                        v_ids, v_labels, v_loss_mask, v_attn = v_batch
                        
                        v_logits = model(input_ids=v_ids, attention_mask=v_attn)['logits']
                        v_log_probs = torch.log_softmax(v_logits, dim=-1)
                        v_safe_labels = v_labels.clamp(min=0)
                        v_token_log_probs = torch.gather(v_log_probs, -1, v_safe_labels.unsqueeze(-1)).squeeze(-1)
                        v_tgt_mask = v_loss_mask.float()
                        
                        v_loss = -(v_token_log_probs * v_tgt_mask).sum()
                        val_loss_accum += v_loss.item()
                        val_tokens_accum += v_tgt_mask.sum().item()
                
                current_val_loss = val_loss_accum / max(1, val_tokens_accum)
                
                if accelerator.is_local_main_process:
                    print(f" --> Validation Loss (first {VAL_BATCH_LIMIT} batches): {current_val_loss:.4f}")
                    if val_history is not None:
                        curr_idx = (epoch - 1) * len(dataloader) + step
                        val_history.append((curr_idx, current_val_loss))
                
                model.train()

            # --- Generation Check ---
            if global_step % 199 == 0 and accelerator.is_local_main_process:
                random_text = [
                                # Existing prompts
                                "Sunlight filtered through the ancient oak’s twisting branches.",
                                "A curious cat perched on the windowsill.",
                                "In the dim control room, tiny lights blinked like a field of artificial stars.",
                                
                                # NEW: Agentic / EWoK Probes
                                
                                # 1. False Belief (Theory of Mind)
                                # Does the model understand John acts on his *belief*, not reality?
                                "John believes his keys are in his pocket, but they are actually on the table. He reaches into his pocket to find ",
                                
                                # 2. Preference / Contrast
                                # Does the model track that distinct agents have distinct tastes?
                                "Alice loves spicy food, while Bob hates it. When the waiter brought the extra-hot curry, Alice smiled, but Bob ",
                                
                                # 3. Doubt / Caution
                                # Does the model understand how uncertainty affects action?
                                "The explorer doubted the old rope bridge was safe. Before stepping onto it, she carefully ",
                                
                                # 4. Goal-Directed Action (Obstacle)
                                # Does the model infer the logical next step to achieve a desire?
                                "The dog desperately wanted the steak on the high counter, but it was too short to reach. To get the food, it started to ",
                                
                                # 5. Social Intent / Regret
                                # Does the model understand social repair mechanisms?
                                "Mark didn't mean to bump into the stranger. Feeling bad about the accident, he quickly turned around to say "
                            ]
                model.eval()
                with torch.no_grad():
                    txt = random.choice(random_text)
                    inp = tokenizer.encode(txt, return_tensors="pt").to(device)
                    out = model.generate(inp, max_length=100, num_return_sequences=1, do_sample=True)
                    gen_txt = tokenizer.decode(out[0], skip_special_tokens=True)
                    print(f"Generated: {gen_txt}")
                model.train()


        if accelerator.is_local_main_process:
            avg_epoch_loss = total_loss / max(1, total_tokens)
            print(f"⇨ End of epoch {epoch}: token-avg loss = {avg_epoch_loss:.4f}")

            # --- Plotting ---
            if out_dir:
                plt.figure(figsize=(10, 6))
                if train_history:
                    tx, ty = zip(*train_history)
                    plt.plot(tx, ty, label='Train Loss', alpha=0.3)
                if val_history:
                    vx, vy = zip(*val_history)
                    plt.plot(vx, vy, label='Val Loss', linewidth=2, color='red', marker='o')
                
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title(f'Training Progress (Epoch {epoch})')
                plt.legend()
                plt.grid(True)
                plot_path = os.path.join(out_dir, 'loss_curve.png')
                plt.savefig(plot_path)
                plt.close()
                print(f"Saved loss plot to {plot_path}")

        # Save Checkpoint
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(out_dir, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"Model + tokenizer saved to {out_dir}/")

        return {"loss": total_loss / max(1, total_tokens)}
    epoch_metrics = []  # <- list of dicts (1 per epoch)
    metrics_path = os.path.join(out_dir, "epoch_metrics.json")

    # ---------- Run Training ----------
    for i in range(NUM_EPOCHS):
        print(f"Training epoch {i+1}/{NUM_EPOCHS} ...")

        train_epoch(
            model, optimizer, scheduler, loader, accelerator,
            epoch=i+1,
            val_dataloader=val_loader,
            train_history=train_loss_history,
            val_history=val_loss_history,
            out_dir=out_dir
        )

        with torch.no_grad():
            eval_dict_1, eval_dict_2 = evaluate(model, tokenizer)

        # --- Store per-epoch record ---
        record = {
            "epoch": i,
            "timestamp": datetime.now().isoformat(),
            # "train": to_jsonable(train_stats),
            "eval_1": to_jsonable(eval_dict_1),
            "eval_2": to_jsonable(eval_dict_2),
        }
        epoch_metrics.append(record)
        # --- Save to disk (main process only) ---
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(out_dir, exist_ok=True)
            save_epoch_metrics(epoch_metrics, metrics_path)
            print(f"Saved epoch metrics to {metrics_path}")

    # Final Push
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        tokenizer.push_to_hub(out_dir)
        model.push_to_hub(out_dir)
        print(f"pushed model to hub. {out_dir}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT-architecture benchmark")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--tokenizer_name', type=str, default='gpt2')
    args = parser.parse_args()
    main(seed=args.seed, batch_size=args.batch_size, num_epochs=args.num_epochs, tokenizer_name=args.tokenizer_name)
