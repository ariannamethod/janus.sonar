"""Encode fineweb_raw.txt into uint16 shards using the BPE trained above.
Uses the HF tokenizer for speed (notorch-compatible because we derived
notorch merges from it directly). One shard for 30MB corpus.
"""
import numpy as np, os, time
from tokenizers import ByteLevelBPETokenizer

CORPUS = "data/fineweb_raw.txt"
SHARD_OUT = "data/train.bin"
VAL_OUT = "data/val.bin"
VAL_FRAC = 0.05

tok = ByteLevelBPETokenizer("data/hf-vocab.json", "data/hf-merges.txt")

t0 = time.time()
with open(CORPUS, "rb") as f:
    raw = f.read()

# Tokenize in one go — HF handles bytes-to-unicode internally
text = raw.decode("utf-8", errors="replace")
enc = tok.encode(text)
ids = np.array(enc.ids, dtype=np.uint16)
print(f"tokenized {len(raw)/1024/1024:.2f} MB → {len(ids):,} tokens in {time.time()-t0:.1f}s")
print(f"compression: {len(raw)/len(ids):.2f} chars/tok")

# Train / val split
n_val = int(len(ids) * VAL_FRAC)
val_ids = ids[:n_val]
train_ids = ids[n_val:]
print(f"train: {len(train_ids):,} tokens, val: {len(val_ids):,} tokens")

train_ids.tofile(SHARD_OUT)
val_ids.tofile(VAL_OUT)
print(f"wrote {SHARD_OUT} ({os.path.getsize(SHARD_OUT):,} bytes)")
print(f"wrote {VAL_OUT} ({os.path.getsize(VAL_OUT):,} bytes)")
