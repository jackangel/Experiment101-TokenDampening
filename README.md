# Do Language Models Learn to Dampen Low-Importance Tokens?

A controlled ablation study of **token-level "noise dampening"** in small
transformer LMs: if we force a model to suppress predictable ("low-importance")
tokens during training, does it generalize better? And if we *don't* force it —
does it choose to dampen them anyway?

**TL;DR of findings** (4-layer / 256d / BPE-4096 / enwik8, seed-noise 0.009 nats):

1. **Forced dampening hurts, monotonically.** Down-weighting low-surprisal
   tokens at training time (input scale × and loss weight × (1−α)) worsens clean
   validation loss at every dose (α = 0.25/0.5/0.75 → +0.02/+0.12/+0.32 nats).
   Dampening *high*-surprisal tokens also hurts — the harm is the intervention,
   not the selection. Predictable tokens are scaffolding the model learns from,
   not noise.
2. **A *learned* input gate dampens them anyway — and wins.** A zero-init
   residual input gate (context → per-dimension add-back) improves validation
   (−0.026), OOD transfer (−0.054) and both extra domains, and its activity
   **anti-correlates with frozen n-gram surprisal (r ≈ −0.20)**. Given the
   freedom to choose what to transform, the model dampens exactly the
   predictable tokens whose forced suppression was harmful.
3. **Causal closure: the direction is the active ingredient.** Clamping the same
   gate to act *only* on predictable tokens beats free choice (val −0.041, OOD
   −0.082 vs control); restricting it to surprising tokens gives roughly half
   the free gate's gain; strict ordering G_low > G > G_high > C on both val and
   OOD, every step > 2× seed noise. What the loss cannot do (D_low hurts), the
   gate does better when *told* where to look — the value is in the mechanism
   and the placement (input embeddings, pre-attention), not in freeing the
   choice.
4. **Noise exposure is a different mechanism.** Training with 25% random-input
   corruption buys large corruption robustness (−0.41 nats) at in-domain cost;
   50% is past the optimum. Robustness comes from *seeing* corrupted contexts,
   not from down-weighting uninformative tokens.
5. **No strong emergent dampening in vanilla training.** Attention mass on
   low-importance keys sits 3–5 pp below key prevalence but barely moves from
   the untrained baseline — the mild suppression exists at init, and training
   does not deepen it.
6. **The OOD edge of forced dampening survives the matched-loss control.** At
   3000 steps, D_low_50 stays ~0.11–0.15 nats *better* on OOD than C at every
   matched in-domain loss level, and wins on both extra domains — a real
   fit-vs-transfer tradeoff, priced at +0.114 in-domain.

## Why "importance" is not circular

The recurring trap in this kind of study: define importance using the training
model's own losses and you get a curriculum on easy tokens. Here importance is
**frozen before any neural training exists**: per-token surprisal (nats) under
an interpolated trigram/bigram/unigram model counted on the *train split only*.
"Low-importance" = below the train 30th percentile (≈1.78 nats). The reference
never sees model weights, and the same frozen scores define every arm's
selection, the low/high loss decomposition, and the attention probe.

## The arms

| arm | mechanism | question |
|---|---|---|
| `C`, `C_seed2` | nothing | control + run-to-run noise margin |
| `D_low_{25,50,75}` | damp low-importance tokens at dose α | H1: does forced dampening help? |
| `D_high_50` | damp high-importance tokens | is the effect about "low" or about dampening? |
| `N_{25,35,50}` | random input corruption | is robustness about noise exposure instead? |
| `G` | learned residual input gate | what does the model dampen *by choice*? |
| `G_kill` | gate replaced by constant-zero map | pathway kill-switch: is any G win input-conditioned? |
| `G_frozen` | gate params frozen at zero | capacity-matched control |
| `C_long`, `D_low_50_long` | 3000 steps + trajectory | does D_low's OOD edge survive at matched in-domain loss? |

Methodological commitments (all pre-registered in `EXPERIMENTS.md`):

- one factor per arm, shared harness, identical data order seeds
- BPE only (vocab 4096, tokenizer trained on the train split)
- evaluation always on clean canonical data, same seeded eval windows per arm
- verdicts adjudicated against falsification criteria written *before* the runs
- identity-at-init and causality unit probes must pass before any queue

## Repository layout

```
data.py        corpus + BPE + frozen n-gram importance model
harness.py     model, mechanisms, training, eval (one JSON line per run)
probes.py      unit probes (run before any queue)
run_queue.py   VRAM-gated sequential arm runner (resume-safe)
analysis.py    comparison table + pre-registered adjudication
results/       results.jsonl (committed runs), EXPERIMENTS.md (full log)
```

## Running

Requires: Python 3.10+, PyTorch (CUDA optional but recommended), `tokenizers`,
`numpy`. A single arm trains in ~12–20 min on an RTX 3060; CPU works but is slow.

Two corpora are **not** auto-downloaded and must be supplied yourself:
- `enwik8` — download from http://mattmahoney.net/dc/enwik8.zip, unzip, and
  either place it at `<repo>/enwik8` or point `ENWIK8_PATH` at it.
- a Lovecraft text file (primary OOD set) — Lovecraft's works are public
  domain in the US (published pre-1928); any plain-text copy works, e.g. from
  Project Gutenberg or https://www.hplovecraft.com/writings/texts/. Place it
  at `<repo>/lovecraft.txt` or point `LOVECRAFT_PATH` at it. This set is
  optional — `data.py` skips it and continues with the austen/darwin OOD sets
  if the file isn't found.

The Austen and Darwin OOD sets ARE fetched automatically from Project
Gutenberg by `data.py`.

```bash
pip install torch tokenizers numpy

# 1. build the data (needs enwik8 + optional lovecraft.txt supplied above;
#    auto-downloads the austen/darwin Gutenberg OOD texts)
set ENWIK8_PATH=C:\path\to\enwik8
set LOVECRAFT_PATH=C:\path\to\lovecraft.txt
python data.py

# 2. unit probes — all must print PASS
python probes.py

# 3. a smoke arm (tiny, throwaway), then the real queue
python harness.py C smoke
python run_queue.py 2500 C C_seed2 D_low_25 D_low_50 D_low_75 D_high_50 N_25 N_35 N_50

# 4. round 2 (learned gate + matched-loss controls)
python run_queue.py 2500 G G_kill G_frozen C_long D_low_50_long

# 5. table + verdicts
python analysis.py
```

Results append to `results/results.jsonl` (resume-safe — arms already present
are skipped).

## Results snapshot

See `results/EXPERIMENTS.md` for the full log, and `results/results.jsonl` for
raw numbers. Headline table (1500-step arms, nats/token):

| arm | val | Δval | OOD | robust(25%) |
|---|---|---|---|---|
| C | 4.0930 | — | 5.3660 | 5.2919 |
| D_low_25 | 4.1161 | +0.023 | 5.3525 | 5.2901 |
| D_low_50 | 4.2087 | +0.116 | 5.3464 | 5.3298 |
| D_low_75 | 4.4155 | +0.323 | 5.4462 | 5.4396 |
| D_high_50 | 4.1873 | +0.094 | 5.5070 | 5.3516 |
| N_25 | 4.3267 | +0.234 | 5.5511 | **4.8803** |
| **G** | **4.0675** | **−0.026** | **5.3125** | 5.3041 |
| G_kill / G_frozen | 4.0930 | 0 | 5.3660 | 5.2919 |

## Interpretation & limits

The combined picture: **dampening is a real and beneficial operation, but it
must be learned, positioned, and dosed by the model itself.** Where the
transformation sits matters — pre-attention input-space gating wins where
target-loss down-weighting loses — and *which* tokens get dampened flips between
harm and benefit depending on who does the selecting: the frozen reference
(wrong tokens, or wrong dose) or the model (its own tradeoff).

Limits: small scale (4L/256d/1500–3000 steps), one primary corpus + three OOD
documents, importance from a trigram-class reference. **Every mechanism arm is
a single seed** — the only variance estimate is the 0.009-nat gap between the
two control seeds (`C`, `C_seed2`), which is then reused as the significance
bar (2× margin) for every other arm and metric. That's a cheap heuristic, not
a proper variance estimate; a skeptical reader should treat effect sizes near
the margin (e.g. `D_low_25` at +0.023) as weak evidence, and the larger gaps
(`D_low_75`, `G`, the Round 3 ordering) as the reliable part of the result.
No additional seeds are planned for this study. The gate↔surprisal correlation
is evidence of *what the model learns to dampen*, not yet of causality — the
next step would be forcing the correlation's sign and checking whether the win
follows.

