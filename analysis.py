"""Analysis: print the comparison table and adjudicate pre-registered criteria.

Usage: python analysis.py [results.jsonl]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "results.jsonl")
rows = [json.loads(l) for l in open(path)]
by = {r["run"]: r for r in rows}
prevalence = 0.30  # fraction of low-importance keys

C = by.get("C")
if C is None:
    sys.exit("control arm C not found")
margin = (abs(by["C"]["val"]["loss"] - by["C_seed2"]["val"]["loss"])
          if "C_seed2" in by else None)

print(f"{'arm':<15}{'val':>8}{'dVal':>8}{'ood':>8}{'robust':>8}"
      f"{'low':>8}{'high':>8}{'attnL':>7}{'gCorr':>7}")
for r in rows:
    v, o = r["val"], r.get("ood", {}).get("ood", r.get("ood", {}))
    ood_l = v if not isinstance(o, dict) else o.get("loss")
    gs = r.get("gate_stats")
    print(f"{r['run']:<15}{v['loss']:>8.4f}"
          f"{v['loss'] - C['val']['loss']:>+8.4f}"
          f"{ood_l:>8.4f}{r['robust']['loss']:>8.4f}"
          f"{v['low_loss']:>8.4f}{v['high_loss']:>8.4f}"
          f"{(v['attn_mass_low'] or float('nan')):>7.3f}"
          f"{(gs['gate_sur_corr'] if gs else float('nan')):>7.3f}")

print(f"\nnoise margin (C vs C_seed2): {margin:.4f} nats" if margin else "")
print("extra OOD domains:")
for r in rows:
    od = r.get("ood") or {}
    doms = {k: round(v["loss"], 4) for k, v in od.items() if isinstance(v, dict)}
    for k, v in (r.get("ood_domains") or {}).items():  # legacy float format
        doms.setdefault(k, round(v, 4))
    if doms:
        print(f"  {r['run']:<15}{doms}")

# ------------------------------------------------------------------ verdicts
print("\n--- pre-registered adjudication ---")
def delta(arm, key="val"):
    return by[arm][key]["loss"] - C[key]["loss"] if arm in by else None

if margin:
    sig = 2 * margin
    dlow = [f"D_low_{a}" for a in (25, 50, 75) if f"D_low_{a}" in by]
    effects = [delta(a) for a in dlow]
    if effects and all(e > sig for e in effects):
        print("H1 (forced low-importance dampening helps): REFUTED — monotone harm")
    elif any(e < -sig for e in effects):
        print("H1: SUPPORTED — some alpha improves clean val beyond 2x noise margin")
    else:
        print("H1: inconclusive at this budget")

    if "D_high_50" in by and delta("D_high_50") > sig:
        print("D_high also hurts => harm is the dampening intervention itself, "
              "not the low-importance selection")

if "G" in by:
    g, k = by["G"], by.get("G_kill", C)
    dv = g["val"]["loss"] - k["val"]["loss"]
    print(f"\nlearned gate vs kill-switch: dVal {dv:+.4f}")
    if g.get("gate_stats"):
        c = g["gate_stats"]["gate_sur_corr"]
        if c < -0.05:
            print(f"gate readout corr {c:+.3f} < 0: the model DAMPS predictable tokens "
                  "when free to choose — dampening is learned, and beneficial")
        elif c > 0.05:
            print(f"gate readout corr {c:+.3f} > 0: the model damps SURPRISING tokens "
                  "(competence reallocation, not noise-cutting)")
        else:
            print("gate readout corr ~0: no surprisal-aligned dampening")

if "N_25" in by:
    print(f"\nnoise exposure robustness: N_25 {delta('N_25', 'robust'):+.4f} "
          "(negative = more robust under 25% input corruption)")

for r in rows:
    if r.get("traj"):
        print(f"\n{r['run']} trajectory (matched-loss check):")
        for p in r["traj"]:
            print(f"  step {p['step']:>5} val {p['val']:.4f} ood {p['ood']}")
