# =============================================================================
#  download_data.py  --  the ONLY Python in the pipeline (permitted role:
#  download raw text and dump it to our binary format).
#
#  Produces in /kaggle/working:
#    train.bin       raw uint16 token ids, headerless  <- what train_gpu mmaps
#    val.bin         1% holdout, same format
#    tokenizer.json  the trained byte-level BPE (vocab locked to 2048)
#    vocab_c.bin     id -> raw-bytes table ("CVB1") for the local C decoder
#
#  Default corpus: WikiText-103 (~500 MB text -> ~150M tokens). For the 5B-token
#  campaign, switch to the streaming FineWeb block at the bottom.
# =============================================================================
import os, struct
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

VOCAB_SIZE = 2048
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

# ---- 1. download raw text --------------------------------------------------
print("downloading WikiText-103 ...")
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
corpus_path = os.path.join(OUT, "corpus.txt")
with open(corpus_path, "w", encoding="utf-8") as f:
    for row in ds:
        t = row["text"]
        if t.strip():
            f.write(t)
print(f"corpus: {os.path.getsize(corpus_path)/1e6:.0f} MB")

# ---- 2. byte-level BPE, vocab locked to 2048 (matches the C engine) --------
tok = Tokenizer(models.BPE(unk_token=None))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE, special_tokens=[],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
# train the vocab on a 20 MB sample (plenty for 2048 merges), encode everything
sample_path = os.path.join(OUT, "sample.txt")
with open(corpus_path, "rb") as f, open(sample_path, "wb") as g:
    g.write(f.read(20_000_000))
tok.train([sample_path], trainer)
assert tok.get_vocab_size() == VOCAB_SIZE, f"vocab {tok.get_vocab_size()} != {VOCAB_SIZE}"
tok.save(os.path.join(OUT, "tokenizer.json"))

# ---- 3. encode -> uint16 -> train.bin / val.bin (headerless) ---------------
print("encoding corpus (rust tokenizer, chunked) ...")
ids_all = []
with open(corpus_path, "r", encoding="utf-8") as f:
    while True:
        chunk = f.read(10_000_000)
        if not chunk:
            break
        ids_all.append(np.asarray(tok.encode(chunk).ids, dtype=np.uint16))
ids = np.concatenate(ids_all)
assert ids.max() < VOCAB_SIZE
n_val = max(1, len(ids) // 100)
ids[:-n_val].tofile(os.path.join(OUT, "train.bin"))
ids[-n_val:].tofile(os.path.join(OUT, "val.bin"))
print(f"train.bin: {len(ids)-n_val:,} tokens ({(len(ids)-n_val)*2/1e9:.2f} GB)")

# ---- 4. id -> raw-bytes table for the local C decoder ("CVB1") -------------
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}
u2b = bytes_to_unicode()
with open(os.path.join(OUT, "vocab_c.bin"), "wb") as f:
    f.write(b"CVB1")
    f.write(struct.pack("<i", VOCAB_SIZE))
    for tid in range(VOCAB_SIZE):
        raw = bytes(u2b[ch] for ch in tok.id_to_token(tid))
        f.write(struct.pack("<i", len(raw)))
        f.write(raw)
print("wrote vocab_c.bin (for local C decoding)")

# ---- 5B-token campaign (later): stream FineWeb instead ----------------------
#   from datasets import load_dataset
#   ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
#                     split="train", streaming=True)
#   ...iterate rows, encode chunks, append uint16 to train.bin until 5e9 tokens.
#   (10 GB file; Kaggle working dir holds ~73 GB, so it fits.)
