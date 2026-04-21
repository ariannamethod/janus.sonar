"""Stream FineWeb-Edu, write ~30MB raw text."""
from datasets import load_dataset
import sys, time

TARGET_BYTES = 30 * 1024 * 1024   # 30 MB
OUT_PATH = "data/fineweb_raw.txt"

t0 = time.time()
ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

total = 0; n = 0
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for ex in ds:
        txt = ex.get("text", "")
        if not txt: continue
        b = txt.encode("utf-8")
        if total + len(b) + 2 > TARGET_BYTES:
            break
        f.write(txt); f.write("\n\n")
        total += len(b) + 2; n += 1
        if n % 200 == 0:
            print(f"  {n} docs, {total/1024/1024:.1f} MB, {time.time()-t0:.0f}s", flush=True)

print(f"done: {n} docs, {total/1024/1024:.2f} MB, {OUT_PATH}")
