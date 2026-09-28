"""Unit probes — run before any training queue (each should print PASS).

P1 rewrite-future  : changing future tokens must not move past-position logits
                     (catches inverted causal masks).
P2 control identity: damp/gate disabled forward == plain hand-rolled computation.
P3 gate identity   : zero-init residual gate == no gate, bit-for-bit; and a
                     constant-zero gate (kill-switch) == no gate.
P4 gate dampens    : with alpha=1 selection semantics, scaling embeddings to 0
                     makes token identity irrelevant at those positions.
P5 attention sanity: rows sum to 1 and carry zero mass on future keys.
"""
import json
import os

import numpy as np
import torch

import harness as H

stats = json.load(open(os.path.join(H.DATA, "stats.json")))
V = stats["vocab"]
T, B, d = 64, 4, 128
model = H.TinyLM(V, T, d, 2, 4, gate=True).to(H.DEV)
rng = np.random.default_rng(0)
x = torch.from_numpy(rng.integers(1, V, size=(B, T))).to(H.DEV)

# P1 causality
model.eval()
with torch.no_grad():
    l1, _ = model(x)
    x2 = x.clone()
    x2[:, T // 2:] = torch.randint(1, V, (B, T - T // 2), device=H.DEV)
    l2, _ = model(x2)
d_past = (l1[:, : T // 2 - 1] - l2[:, : T // 2 - 1]).abs().max().item()
print(f"P1 rewrite-future: {d_past:.2e}", "PASS" if d_past < 1e-6 else "FAIL")

# P2 zero-init gate == plain model
def plain(mm, idx):
    h = mm.emb(idx) + mm.pos(torch.arange(idx.shape[1], device=idx.device))
    for b in mm.blocks:
        h, _ = b(h, mm.mask_f)
    return mm.head(mm.lnf(h))

model.gate[-1].weight.data.zero_()
model.gate[-1].bias.data.zero_()
with torch.no_grad():
    dd = (model(x)[0] - plain(model, x)).abs().max().item()
print(f"P2 zero-init gate identity: {dd:.2e}", "PASS" if dd < 1e-6 else "FAIL")

# P3 kill-switch (constant zero gate) == plain model
class _ZeroGate(torch.nn.Module):
    def forward(self, emb):
        return torch.zeros_like(emb)
model.gate = _ZeroGate()
with torch.no_grad():
    dd = (model(x)[0] - plain(model, x)).abs().max().item()
print(f"P3 kill-switch identity: {dd:.2e}", "PASS" if dd < 1e-6 else "FAIL")

# P4 zero-scaled embeddings hide token identity
model.gate = None
sel = torch.zeros(B, T, dtype=torch.bool, device=H.DEV)
sel[:, ::2] = True
scale = torch.where(sel, 0.0, torch.ones_like(x.float()))
with torch.no_grad():
    la, _ = H.model_forward_scaled(model, x, scale)
    xr = x.clone()
    xr[sel] = torch.randint(1, V, (int(sel.sum()),), device=H.DEV)
    lb, _ = H.model_forward_scaled(model, xr, scale)
dd = (la - lb).abs().max().item()
print(f"P4 zero-scale hides tokens: {dd:.2e}", "PASS" if dd < 1e-6 else "FAIL")

# P5 attention sanity
model.collect_attn = True
with torch.no_grad():
    _, ws = model(x)
model.collect_attn = False
w = ws[-1]
s = (w.sum(-1) - 1).abs().max().item()
tril = torch.tril(torch.ones(T, T, device=H.DEV)).unsqueeze(0)
future = (w * (1 - tril)).sum().item()
print(f"P5 attn rows sum to 1: {s:.2e}", "PASS" if s < 1e-4 else "FAIL")
print(f"P5 future-key mass: {future:.2e}", "PASS" if future < 1e-6 else "FAIL")
