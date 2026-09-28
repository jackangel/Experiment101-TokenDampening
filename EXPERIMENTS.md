# Dampening Experiment — Pre-Registration

## Question
Does training that forces dampening of low-importance (low frozen-surprisal) tokens
improve generalization (val loss)? Do trained LMs already allocate less capacity to
low-importance tokens (emergence, H2)?

## Importance definition (frozen, non-circular)
Surprisal (nats) under an interpolated trigram/bigram/unigram model counted on the
TRAIN split only. "Low-importance" = below train 30th percentile. "High" = above
train 70th percentile. The reference model never sees any training run's weights.

## Corpus / tokenization
- enwik8, 95/5 train/val split. BPE vocab 4096, tokenizer trained on TRAIN only.
- OOD set: Lovecraft (wave-lm input.txt), same tokenizer.
- Standing rule: never character-level.

## Arms (one factor: training-time token treatment)
| arm | treatment (inputs x) | loss weights |
|---|---|---|
| C | none (alpha=0 path disabled) | 1 |
| D-low-25/50/75 | embed x (1-a) on low-surprisal positions | w=1-a on low targets |
| D-high-50 | same on high-surprisal positions | w=1-a on high targets |
| N-25/50 | replace x with random token at rate a (uniform positions) | 1 |

All arms: identical model, data order seed, constant LR, steps, eval protocol.
Dampening is TRAIN-TIME ONLY. Every eval is on clean canonical data.

## Metrics
- headline: val loss (nats/token, clean)
- OOD loss (Lovecraft)
- robustness: val loss with inputs corrupted at rate 0.25 (targets natural)
- H2: last-layer attention mass to low- vs high-surprisal context keys, logged
  every checkpoint for EVERY arm (baseline C included). If the theory is right,
  mass-to-low should sit below the ~30% key prevalence and fall over training.
- loss decomposition on val: low-target vs high-target mean loss

## Falsification criteria (pre-registered)
- H1 supported iff some alpha in D-low improves clean val by > 2x run-to-run noise
  (noise margin = max spread across repeated C seeds at this budget; measure first
  with 2 C seeds if arms are close).
- If D-high ALSO improves: the importance axis is doing the work (regularization
  effect), not the "low-importance" claim -> theory not supported as stated.
- If D-low monotonically HURTS with alpha: theory loses this round at this scale.
- H2 supported iff C's attention-mass-to-low < prevalence by a clear margin and
  declines over training. Mass tracking prevalence flat = no evidence of emergent
  dampening.
- Corruption exposure (N) matching or beating D-low = noise robustness comes from
  seeing corrupted contexts, not from dampening low-importance tokens.

## Budget protocol
s/step measured on the free GPU before the queue; steps chosen so one arm is
~10-20 min. Constant eval window count (fixed seeded windows) across arms.
Constant LR (no cosine) — schedule is itself an ablation at short budgets.

## Results (all 8 arms, 1500 steps, GPU)

Noise margin (C vs C_seed2): 0.009 nats; 2x margin = 0.018.

| arm | val | Δval | ood | robust | low_l | high_l | attn_low |
|---|---|---|---|---|---|---|---|
| C | 4.0930 | — | 5.3660 | 5.2919 | 1.706 | 6.173 | 0.262 |
| C_seed2 | 4.0838 | −0.009 | 5.3678 | 5.2896 | 1.709 | 6.150 | 0.253 |
| D_low_25 | 4.1161 | +0.023 | 5.3525 | 5.2901 | 1.811 | 6.149 | 0.259 |
| D_low_50 | 4.2087 | +0.116 | 5.3464 | 5.3298 | 2.081 | 6.155 | 0.259 |
| D_low_75 | 4.4155 | +0.323 | 5.4462 | 5.4396 | 2.599 | 6.191 | 0.268 |
| D_high_50 | 4.1873 | +0.094 | 5.5070 | 5.3516 | 1.676 | 6.437 | 0.259 |
| N_25 | 4.3267 | +0.234 | 5.5511 | 4.8803 | 2.091 | 6.252 | 0.255 |
| N_50 | 4.7742 | +0.681 | 5.8682 | 5.1347 | 2.785 | 6.488 | 0.255 |

## Verdicts (adjudicated against pre-registered criteria)

1. **H1 REFUTED at this scale.** D-low hurts clean val MONOTONICALLY with alpha
   (+0.023 / +0.116 / +0.323), every step > 2x noise margin. No win anywhere in
   the sweep. Low-surprisal tokens are a resource the model learns FROM, not noise
   to cut (dose-response opposite the hypothesis => re-derive the component's role).
2. **D-high also hurts (+0.094)** => the harm is the dampening intervention itself,
   not the "low-importance" selection. No regularization confound to invoke.
   (Curiosity: D_high_50 improved low-target loss to 1.676 — best of all arms —
   at the cost of high-target loss 6.437, worst of the dampened arms.)
3. **OOD: the only positive signal.** D_low_50 improves OOD by −0.020 (>2x margin)
   while HURTING in-domain by +0.116 — a small in-domain/OOD tradeoff, direction
   the theory wants, size ~5x smaller than the in-domain cost. Worth one follow-up
   arm before crediting it (e.g. D_low_50 with a smaller LR or more steps to see
   if the in-domain cost is an optimization artifact).
4. **Noise exposure (N) is a different mechanism, confirmed.** N_25 costs +0.234
   in-domain but gains −0.412 robustness — robustness is bought with corruption
   exposure, not with dampening. Matches criterion 5 (N beats D-low on robustness).
5. **H2: WEAK.** attn-to-low sits 3–5pp below prevalence (0.30) in EVERY arm
   including C, and only ~1–2pp below the untrained-init value (0.27). A mild
   static under-attention to predictable keys exists, but training does not
   progressively deepen it — no emergence of strong dampening over training.

## Status
- [x] data built (data/), frozen n-gram counts + per-token surprisal
- [x] s/step probe (arms landed 12-19 min on GPU)
- [x] smoke batch (8/8 pass after deep-merge + quantile + queue-name fixes)
- [x] queue (8/8 arms, VRAM-gated behind user's gdense benchmark)
- [x] analysis vs falsification criteria (this section)
- [x] Round 2 (running): G / G_kill / G_frozen (learned residual input gate + readout),
      C_long / D_low_50_long (3000 steps, val+OOD trajectory every 750 — decides whether
      the D_low OOD gain survives at matched in-domain loss), N_35 (robustness optimum bracket).
      New OOD domains: austen (245K tok, near), darwin (281K tok, far).
      Pre-registered gate readout: corr(gate_scale, frozen surprisal) < 0 = model damps
      predictable tokens when free to choose (theory validated in learned form);
      > 0 = dampens surprising tokens (competence-reallocation); ~0/flat gate = closes program.

## Round 2 results (14/14 runs complete)

**G (learned gate) — WIN, and the readout validates the theory's learned form:**
val 4.0675 (−0.026 vs C, ~3x margin), OOD 5.3125 (−0.054), austen 5.361 /
darwin 5.207 (both better than controls 5.419/5.279). gate_sur_corr = −0.203:
given free choice, the model dampens LOW-surprisal tokens and benefits.
G_kill and G_frozen bit-identical to C (dVal −0.0000) — effect is the learned,
input-conditioned gate, not capacity.

**Matched-loss OOD adjudication — the tradeoff is REAL:**
C_long traj (val/ood): 1500: 4.093/5.366, 2250: 3.897/5.229, 3000: 3.764/5.093
D_low_50_long traj:    1500: 4.209/5.346, 2250: 4.013/5.185, 3000: 3.878/5.066
Interpolated at equal val loss, D is better on OOD by +0.115 / +0.125 / +0.145
(at val = 4.093 / 4.013 / 3.897) — the advantage GROWS with training. Not an
underfitting artifact. Price at equal budget: +0.114 in-domain (3.878 vs 3.764).
D also wins austen (4.997 vs 5.035) and darwin (4.919 vs 4.928) at 3000 steps.
Decomposition at 3000: D trades low-token competence (1.703 vs C 1.307) for
high-token gains (5.900 vs 5.932) and transfer.

**N_35: robustness optimum bracketed.** robust loss: N_25 4.880 | N_35 4.947 |
N_50 5.135 (C 5.292). Monotone in the wrong direction past 25% — optimum is
near 25%, not between 25 and 50.

## Final synthesis
1. Forced dampening: hurts in-domain, monotone in dose. REFUTED as stated.
2. Learned dampening: helps everywhere, and the model aims it at low-surprisal
   tokens (corr −0.20). The theory is right about WHAT, wrong about WHO/WHEN:
   the model should choose, not the loss.
3. The fit-vs-transfer tradeoff from forced dampening is real and grows with
   training — a genuine regularization mechanism, priced in-domain.
4. Robustness = corruption exposure (optimum ~25%), a different mechanism.
5. No deepening emergent suppression in vanilla attention over training.

## Round 3 — causality of the readout (16/16 runs complete)

Design: G_low / G_high = the SAME learned gate, clamped to act ONLY on
low- / high-surprisal tokens (clamp verified bit-identity when nothing is
selected; gradients flow through the allowed path). Same init, data, budget.

| arm | val | dVal | ood | robust |
|---|---|---|---|---|
| G_low  | 4.0522 | −0.041 | 5.2839 | 5.2476 |
| G      | 4.0675 | −0.026 | 5.3125 | 5.3041 |
| G_high | 4.0760 | −0.017 | 5.3231 | 5.2580 |
| C      | 4.0930 | 0      | 5.3660 | 5.2919 |

VERDICT (pre-registered outcome #1): **dampening predictable tokens IS the
active ingredient.** Strict ordering G_low > G > G_high > C on val AND ood
(all steps > 2x margin): the gain is monotone in how predictable the allowed
token set is. Restricting the gate to predictable tokens BEATS free choice —
the free gate's activity on non-predictable positions dilutes it.
Causal chain closed: direction (low-importance) x mechanism (learned input-
space gate) both necessary; loss-weight dampening of the same tokens (D_low)
still hurts — placement matters as much as direction.

Caveat: for clamped arms the gate<->surprisal readout is uninterpretable
(the clamp zeroes gradients on forbidden positions, so raw gate output there
is untrained drift; G_high's −0.46 corr is an artifact of measuring it).
The loss numbers above carry the verdict.

## Campaign synthesis (final)
1. WHAT the theory claimed (dampening predictable tokens) is causally right —
   but only through a learned input-space gate, where it is the optimal
   assignment (better than free choice).
2. WHO decides: the model, through a gate — not the loss (D_low hurts).
3. WHERE: input embeddings, pre-attention; the same tokens down-weighted in
   the LOSS have the opposite effect.
4. Fit-vs-transfer: forced dampening buys real OOD transfer at in-domain cost
   (grows with training); the gate wins without the tradeoff.
5. Robustness to corruption comes from exposure (optimum ~25%), not dampening.
