"""HF ByteLevelBPE train + export to notorch integer-pair format.
notorch format: one line per merge, "<left_id> <right_id>" where ids
refer to base bytes (0-255) or previously-merged tokens (256+).
"""
import json, time, sys
from tokenizers import ByteLevelBPETokenizer
from tokenizers.models import BPE
from tokenizers import Tokenizer, pre_tokenizers, decoders

CORPUS = "data/fineweb_raw.txt"
VOCAB = 4096
OUT_MERGES = "data/janus15m_bpe_merges.txt"

t0 = time.time()

# Train byte-level BPE
tok = ByteLevelBPETokenizer()
tok.train(files=[CORPUS], vocab_size=VOCAB, min_frequency=2, special_tokens=[])
print(f"trained in {time.time()-t0:.1f}s")

# Save raw HF artifacts to inspect
tok.save_model("data", "hf")
# data/hf-vocab.json, data/hf-merges.txt

# Read them back.
with open("data/hf-vocab.json") as f:
    hf_vocab = json.load(f)   # {str_token: hf_id}
with open("data/hf-merges.txt") as f:
    lines = f.read().splitlines()
# Skip HF version header (first line starts with #)
merges_lines = [l for l in lines if l and not l.startswith("#")]

print(f"vocab size: {len(hf_vocab)}")
print(f"merges: {len(merges_lines)}")

# Build bytes_to_unicode table (GPT-2 trick reverse).
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {bytes([b]): chr(c) for b, c in zip(bs, cs)}

byte2uni = bytes_to_unicode()      # {b'A': 'A', b' ': 'Ġ', ...}
uni2byte = {v: k[0] for k, v in byte2uni.items()}   # {'A': 65, 'Ġ': 32, ...}

# Map HF token-string → notorch integer id.
# notorch base ids: 0..255 = literal bytes.
# notorch merged ids: 256, 257, ... in merge-application order.
str2ntid = {}
for uni_char, byte_val in uni2byte.items():
    str2ntid[uni_char] = byte_val   # base byte tokens

# Process merges in order; each produces one new notorch id.
out_pairs = []
next_nt_id = 256
for line in merges_lines:
    parts = line.split(" ")
    if len(parts) != 2:
        print(f"skipping weird line: {line!r}")
        continue
    left_s, right_s = parts[0], parts[1]
    if left_s not in str2ntid or right_s not in str2ntid:
        print(f"unresolved merge: {left_s!r}+{right_s!r}")
        continue
    L = str2ntid[left_s]
    R = str2ntid[right_s]
    out_pairs.append((L, R))
    merged_s = left_s + right_s
    str2ntid[merged_s] = next_nt_id
    next_nt_id += 1

print(f"notorch pairs: {len(out_pairs)}")

with open(OUT_MERGES, "w") as f:
    for L, R in out_pairs:
        f.write(f"{L} {R}\n")
print(f"wrote {OUT_MERGES}")
print(f"wall: {time.time()-t0:.1f}s")
