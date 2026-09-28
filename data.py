"""Data pipeline: BPE corpus + frozen n-gram importance model.

Builds (data/):
  bpe.json            BPE tokenizer (vocab 4096, trained on the TRAIN split only)
  train.bin/val.bin   uint16 token ids (enwik8, 95/5 byte split)
  ood.bin             out-of-domain ids (Lovecraft, "The Tomb" collection)
  ood_austen.bin      out-of-domain ids (Pride and Prejudice, Gutenberg #1342)
  ood_darwin.bin      out-of-domain ids (Darwin, "On the Origin of Species", #1228)
  *.sur.npy           float16 per-position surprisal (nats) under the FROZEN
                      interpolated trigram/bigram/unigram model below
  uni/bi/tri.npz      the frozen n-gram counts (train split only)
  stats.json          vocab, sizes, thresholds

Design invariant: "importance" is defined by an n-gram model counted on TRAIN
ONLY, before any neural training run exists. Selection of low/high-importance
tokens is therefore never circular with the model under test.

Usage:  python data.py            # builds everything
        python data.py --ood-only # rebuild only the extra OOD sets
"""
import argparse
import json
import os
import sys

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

VOCAB = 4096
LOW_PCT = 30  # "low importance" = below this train-surprisal percentile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# Local corpus paths. Override with env vars, or drop the files at these
# relative defaults. Neither file is fetched automatically:
#   ENWIK8_PATH     - enwik8 (Wikipedia dump), http://mattmahoney.net/dc/enwik8.zip
#   LOVECRAFT_PATH  - any public-domain Lovecraft text file (plain .txt), e.g.
#                     from https://www.gutenberg.org (works pre-1928 are PD in
#                     the US) or https://www.hplovecraft.com/writings/texts/
ENWIK8_PATH = os.environ.get("ENWIK8_PATH", os.path.join(HERE, "enwik8"))
LOVECRAFT_PATH = os.environ.get("LOVECRAFT_PATH", os.path.join(HERE, "lovecraft.txt"))
GUTENBERG = {
    # extra OOD domains, downloaded automatically on first run
    "ood_austen": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "ood_darwin": "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
}


# ---------------------------------------------------------------- n-gram model
def _lookup(npz, keys):
    """Count lookup with 0 for unseen keys (searchsorted over sorted unique keys)."""
    idx = np.searchsorted(npz["keys"], keys)
    idx = np.clip(idx, 0, len(npz["keys"]) - 1)
    hit = npz["keys"][idx] == keys
    return np.where(hit, npz["cnt"][idx], 0).astype(np.float64)


class FrozenNGram:
    """Interpolated trigram/bigram/unigram over BPE ids, counted on TRAIN only.

    P(x) = 0.2*P1(x) + 0.3*P2(x|a) + 0.5*P3(x|ab), add-1/2 backoff smoothing.
    Frozen: fit once, never updated during any training run.
    """

    LAM = (0.2, 0.3, 0.5)

    def __init__(self, train_ids, vocab):
        self.V = vocab
        self.uni = os.path.join(DATA, "uni.npz")
        self.bi = os.path.join(DATA, "bi.npz")
        self.tri = os.path.join(DATA, "tri.npz")
        if not os.path.exists(self.tri):
            self._build_counts(train_ids)
        uni, bi, tri = np.load(self.uni), np.load(self.bi), np.load(self.tri)
        self.uni_p = np.full(vocab, 0.5, dtype=np.float64)
        self.uni_p[uni["keys"]] += uni["cnt"]
        self.uni_p /= self.uni_p.sum()
        self.l_uni = -np.log(self.uni_p)
        self.ctx_map = np.zeros(vocab, dtype=np.float64)
        self.ctx_map[uni["keys"]] = uni["cnt"].astype(np.float64)
        self._bi, self._tri = bi, tri

    def _build_counts(self, train_ids):
        arr = train_ids.astype(np.int64)
        V = self.V
        for path, order in [(self.tri, 3), (self.bi, 2)]:
            n = len(arr) - order + 1
            keys = np.zeros(n, dtype=np.int64)
            for j in range(order):
                keys = keys * V + arr[j: j + n]
            uq, cnt = np.unique(keys, return_counts=True)
            np.savez_compressed(path, keys=uq, cnt=cnt.astype(np.int32))
        uq, cnt = np.unique(arr, return_counts=True)
        np.savez_compressed(self.uni, keys=uq, cnt=cnt.astype(np.int32))

    def surprisal(self, ids):
        arr = np.asarray(ids, dtype=np.int64)
        V, N = self.V, len(arr)
        lam = self.LAM
        sur = lam[0] * self.uni_p[arr]
        bcnt = _lookup(self._bi, arr[:-1] * V + arr[1:])
        p_bi = np.zeros(N)
        p_bi[1:] = (bcnt + 0.5) / (self.ctx_map[arr[:-1]] + 0.5 * V)
        sur[1:] += lam[1] * p_bi[1:]
        sur[0] += lam[1] * self.uni_p[arr[0]]
        tcnt = _lookup(self._tri, arr[:-2] * V * V + arr[1:-1] * V + arr[2:])
        ab_ctx = _lookup(self._bi, arr[:-2] * V + arr[1:-1])
        p_tri = np.zeros(N)
        p_tri[2:] = (tcnt + 0.5 * p_bi[1: N - 1]) / (ab_ctx + 0.5)
        sur[2:] += lam[2] * p_tri[2:]
        return (-np.log(np.maximum(sur, 1e-12))).astype(np.float16)


# ------------------------------------------------------------------- pipeline
def _strip_gutenberg(path):
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8", errors="replace")
    a = raw.find("*** START OF")
    b = raw.find("*** END OF")
    if a != -1:
        raw = raw[raw.find("\n", a) + 1:]
    if b != -1:
        raw = raw[:b]
    return raw


def _download(url, dest):
    import urllib.request
    urllib.request.urlretrieve(url, dest)


def build(ood_only=False):
    os.makedirs(DATA, exist_ok=True)
    tok_path = os.path.join(DATA, "bpe.json")

    if ood_only:
        tok = Tokenizer.from_file(tok_path)
        stats = json.load(open(os.path.join(DATA, "stats.json")))
        ngram = None
        train_sur = np.load(os.path.join(DATA, "train.sur.npy")).astype(np.float32)
    else:
        if not os.path.exists(ENWIK8_PATH):
            sys.exit(f"enwik8 not found at {ENWIK8_PATH}. Download from "
                      "http://mattmahoney.net/dc/enwik8.zip, unzip, and set "
                      "ENWIK8_PATH (env var) or place it at that path.")
        with open(ENWIK8_PATH, "rb") as f:
            raw = f.read()
        n = len(raw)
        train_raw, val_raw = raw[: int(n * 0.95)], raw[int(n * 0.95):]

        if os.path.exists(tok_path):
            tok = Tokenizer.from_file(tok_path)
        else:
            tmp = os.path.join(DATA, "_train.txt")
            with open(tmp, "wb") as f:
                f.write(train_raw)
            tok = Tokenizer(BPE(byte_fallback=True))
            tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
            tok.train([tmp], BpeTrainer(vocab_size=VOCAB, special_tokens=["<pad>"]))
            tok.save(tok_path)
            os.remove(tmp)

        def enc(b):
            return np.array(tok.encode(b.decode("utf-8", errors="replace")).ids,
                            dtype=np.uint16)

        train, val = enc(train_raw), enc(val_raw)
        train.tofile(os.path.join(DATA, "train.bin"))
        val.tofile(os.path.join(DATA, "val.bin"))
        print(f"train {len(train):,} val {len(val):,} tokens, vocab {tok.get_vocab_size()}")

        ngram = FrozenNGram(train, tok.get_vocab_size())
        sur_tr = ngram.surprisal(train)
        np.save(os.path.join(DATA, "train.sur.npy"), sur_tr)
        sur_va = ngram.surprisal(val)
        np.save(os.path.join(DATA, "val.sur.npy"), sur_va)

        stats = {
            "vocab": tok.get_vocab_size(),
            "train_tokens": int(len(train)),
            "val_tokens": int(len(val)),
            "low_sur_threshold_nats": float(np.percentile(sur_tr.astype(np.float32), LOW_PCT)),
            "sur_mean_train": float(sur_tr.astype(np.float32).mean()),
            "sur_mean_val": float(sur_va.astype(np.float32).mean()),
        }

    V = stats["vocab"]
    for name, url in GUTENBERG.items():
        dest = os.path.join(DATA, f"{name}.txt")
        if not os.path.exists(dest):
            print("downloading", url)
            _download(url, dest)
        ids = np.array(tok.encode(_strip_gutenberg(dest)).ids, dtype=np.uint16)
        sur = (ngram or FrozenNGram(
            np.fromfile(os.path.join(DATA, "train.bin"), dtype=np.uint16), V)).surprisal(ids)
        ids.tofile(os.path.join(DATA, f"{name}.bin"))
        np.save(os.path.join(DATA, f"{name}.sur.npy"), sur)
        stats[f"{name}_tokens"] = int(len(ids))
        stats[f"{name}_sur_mean"] = float(sur.astype(np.float32).mean())
        print(name, f"{len(ids):,} tokens, mean frozen surprisal {stats[f'{name}_sur_mean']:.3f}")

    # primary OOD (Lovecraft) if the configured file exists
    if os.path.exists(LOVECRAFT_PATH):
        ids = np.array(tok.encode(
            open(LOVECRAFT_PATH, "rb").read().decode("utf-8", errors="replace")).ids,
            dtype=np.uint16)
        ids.tofile(os.path.join(DATA, "ood.bin"))
        np.save(os.path.join(DATA, "ood.sur.npy"),
                (ngram or FrozenNGram(
                    np.fromfile(os.path.join(DATA, "train.bin"), dtype=np.uint16), V)).surprisal(ids))
        stats["ood_tokens"] = int(len(ids))
        print("ood", f"{len(ids):,} tokens")
    else:
        print(f"note: LOVECRAFT_PATH ({LOVECRAFT_PATH}) not found, skipping "
              "primary OOD set (ood.bin). Any public-domain Lovecraft .txt "
              "works; austen/darwin OOD sets are unaffected.")

    with open(os.path.join(DATA, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("stats:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-only", action="store_true")
    build(ap.parse_args().ood_only)
