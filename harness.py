"""Training harness for the token-dampening study.

Mechanisms (train-time only, all switchable by config):
  damp  {enabled, which: low|high, alpha, high_q}
        On frozen-ngram-selected positions: scale input embeddings by (1-alpha)
        and weight the target CE loss by (1-alpha). Selection uses the FROZEN
        n-gram surprisal (data/*.sur.npy) -- never the model under test.
  noise {enabled, rate}
        Replace input tokens with random ids at `rate`; targets untouched.
  gate  {enabled, freeze, kill}
        Learned residual input gate x + f(x), f zero-init => exact identity at
        start. kill = replace f with a constant zero map (pathway kill-switch).
        freeze = keep f's parameters fixed at zero (capacity-matched control).

Evaluation (identical for every arm): clean validation, clean OOD domains,
corrupted-input robustness, low/high-loss decomposition, attention-mass probe
(mass on low-importance keys), optional gate<->surprisal readout, and optional
val/OOD trajectory snapshots during long runs.

Results append as one JSON line per run to results.jsonl (resume-safe: runs
already present are skipped).

Usage:
  python harness.py <arm>            # full run
  python harness.py <arm> smoke      # tiny throwaway sanity run (smoke.jsonl)
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

BASE = dict(
    d_model=256, n_layer=4, n_head=4, block_size=256,
    batch_size=64, lr=3e-4, train_steps=1500, eval_windows=32,
    log_every=250, seed=1234,
    traj_at=(1500, 2250, 3000),          # snapshot steps for long runs
    damp=dict(enabled=False, which="low", alpha=0.0, high_q=0.70),
    noise=dict(enabled=False, rate=0.0),
    gate=dict(enabled=False, freeze=False, kill=False),
)

RUNS = {}


def register(name, **ovr):
    RUNS[name] = ovr


# round 1: forced dampening / noise exposure
register("C")
register("C_seed2", seed=777)
register("D_low_25", damp=dict(enabled=True, which="low", alpha=0.25))
register("D_low_50", damp=dict(enabled=True, which="low", alpha=0.50))
register("D_low_75", damp=dict(enabled=True, which="low", alpha=0.75))
register("D_high_50", damp=dict(enabled=True, which="high", alpha=0.50))
register("N_25", noise=dict(enabled=True, rate=0.25))
register("N_35", noise=dict(enabled=True, rate=0.35))
register("N_50", noise=dict(enabled=True, rate=0.50))
# round 2: learned gate (meta-dampening) + matched-loss controls
register("G", gate=dict(enabled=True))
register("G_kill", gate=dict(enabled=True, kill=True))
register("G_frozen", gate=dict(enabled=True, freeze=True))
register("C_long", train_steps=3000)
register("D_low_50_long", damp=dict(enabled=True, which="low", alpha=0.50),
         train_steps=3000)


# ---------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask, need_weights=False):
        h = self.ln1(x)
        a, w = self.attn(h, h, h, attn_mask=mask,
                         need_weights=need_weights, average_attn_weights=True)
        x = x + a
        return x + self.mlp(self.ln2(x)), w


class TinyLM(nn.Module):
    """Pre-LN transformer with optional residual input gate."""

    def __init__(self, V, T, d, nl, nh, gate=False):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        self.collect_attn = False
        self.gate = None
        if gate:
            self.gate = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            nn.init.zeros_(self.gate[-1].weight)   # zero-init => exact identity
            nn.init.zeros_(self.gate[-1].bias)
        tril = torch.tril(torch.ones(T, T, dtype=torch.bool))
        self.register_buffer("mask_f",
                             torch.zeros(T, T).masked_fill(~tril, float("-inf")))

    def forward(self, idx):
        x = self.emb(idx)
        if self.gate is not None:
            x = x + self.gate(x)      # residual; identity at init
        x = x + self.pos(torch.arange(idx.shape[1], device=idx.device))
        ws = []
        for b in self.blocks:
            x, w = b(x, self.mask_f, need_weights=self.collect_attn)
            if self.collect_attn:
                ws.append(w)
        return self.head(self.lnf(x)), ws


# ------------------------------------------------------------------ utilities
def resolve(name, smoke=False):
    def deep_merge(base, ovr):
        out = dict(base)
        for k, v in ovr.items():
            out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        return out
    cfg = deep_merge(BASE, RUNS[name])
    cfg["run_name"] = name
    cfg["vocab"] = json.load(open(os.path.join(DATA, "stats.json")))["vocab"]
    if smoke:
        cfg.update(train_steps=20, eval_windows=2, batch_size=8, block_size=128,
                   traj_at=(15,))
    return cfg


def run(cfg, results_path=os.path.join("results", "results.jsonl")):
    t0 = time.time()
    stats = json.load(open(os.path.join(DATA, "stats.json")))
    V = cfg["vocab"]
    thr = stats["low_sur_threshold_nats"]

    def load(name):
        ids = np.fromfile(os.path.join(DATA, f"{name}.bin"), dtype=np.uint16)
        sur = np.load(os.path.join(DATA, f"{name}.sur.npy")).astype(np.float32)
        return ids, sur

    train_ids, train_sur = load("train")
    val_ids, val_sur = load("val")
    ood_sets = {n: load(n) for n in
                ("ood", "ood_austen", "ood_darwin")
                if os.path.exists(os.path.join(DATA, f"{n}.bin"))}
    hi_thr = float(np.quantile(train_sur[:4_000_000], cfg["damp"]["high_q"]))
    T, B = cfg["block_size"], cfg["batch_size"]
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    model = TinyLM(V, T, cfg["d_model"], cfg["n_layer"], cfg["n_head"],
                   gate=cfg["gate"]["enabled"]).to(DEV)
    if cfg["gate"]["enabled"] and cfg["gate"]["kill"]:
        class _ZeroGate(nn.Module):
            def forward(self, x):
                return torch.zeros_like(x)
        model.gate = _ZeroGate().to(DEV)  # kill-switch: constant zero
    if cfg["gate"]["enabled"] and cfg["gate"]["freeze"]:
        for p in model.gate.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["lr"])

    def make_batch():
        ix = rng.integers(0, len(train_ids) - T - 1, size=B)
        x = np.stack([train_ids[i:i + T + 1] for i in ix]).astype(np.int64)
        s = np.stack([train_sur[i:i + T + 1] for i in ix])
        xt = torch.from_numpy(x[:, :-1]).to(DEV)
        yt = torch.from_numpy(x[:, 1:]).to(DEV)
        xsur = torch.from_numpy(s[:, :-1]).to(DEV)
        ysur = torch.from_numpy(s[:, 1:]).to(DEV)
        if cfg["noise"]["enabled"]:
            m = torch.rand_like(xt.float()) < cfg["noise"]["rate"]
            xt = torch.where(m, torch.randint(1, V, xt.shape, device=DEV), xt)
        return xt, yt, xsur, ysur

    @torch.no_grad()
    def evaluate(ids, sur, windows, collect=False, corrupt_rate=0.0):
        """Clean canonical eval; same seeded windows for every arm."""
        model.eval()
        ix = np.random.default_rng(999).integers(0, len(ids) - T - 1, size=windows * B)
        tot = ntok = 0
        low_l, low_n, high_l, high_n = 0.0, 0, 0.0, 0
        a_low = a_high = a_all = 0.0
        for wi in range(windows):
            xs = np.stack([ids[i:i + T + 1] for i in ix[wi * B:(wi + 1) * B]]).astype(np.int64)
            ss = np.stack([sur[i:i + T + 1] for i in ix[wi * B:(wi + 1) * B]])
            xt = torch.from_numpy(xs[:, :-1]).to(DEV)
            yt = torch.from_numpy(xs[:, 1:]).to(DEV)
            ysur = torch.from_numpy(ss[:, 1:]).to(DEV)
            if corrupt_rate > 0:
                m = torch.rand_like(xt.float()) < corrupt_rate
                xt = torch.where(m, torch.randint(1, V, xt.shape, device=DEV), xt)
            model.collect_attn = collect
            logits, ws = model(xt)
            model.collect_attn = False
            ce = F.cross_entropy(logits.reshape(-1, V), yt.reshape(-1),
                                 reduction="none").reshape(yt.shape)
            tot += ce.sum().item()
            ntok += ce.numel()
            lowm, highm = ysur < thr, ysur > hi_thr
            low_l += ce[lowm].sum().item()
            low_n += int(lowm.sum())
            high_l += ce[highm].sum().item()
            high_n += int(highm.sum())
            if collect and ws:
                klow = (ysur < thr).float().unsqueeze(1)
                khigh = (ysur > hi_thr).float().unsqueeze(1)
                for w in ws:  # [B, Tq, Tk], heads averaged
                    a_low += (w * klow).sum().item()
                    a_high += (w * khigh).sum().item()
                    a_all += w.sum().item()
        model.train()
        return dict(loss=tot / ntok,
                    low_loss=low_l / max(low_n, 1),
                    high_loss=high_l / max(high_n, 1),
                    attn_mass_low=(a_low / a_all if a_all else None),
                    attn_mass_high=(a_high / a_all if a_all else None))

    # ---------------- training ----------------
    traj_at = set(cfg["traj_at"]) & set(range(1, cfg["train_steps"] + 1))
    traj = []
    model.train()
    for step in range(cfg["train_steps"]):
        xt, yt, xsur, ysur = make_batch()
        emb_scale = w = None
        if cfg["damp"]["enabled"] and cfg["damp"]["alpha"] > 0:
            a = cfg["damp"]["alpha"]
            low = cfg["damp"]["which"] == "low"
            sel_in = (xsur < thr) if low else (xsur > hi_thr)
            sel_tgt = (ysur < thr) if low else (ysur > hi_thr)
            emb_scale = torch.where(sel_in, 1.0 - a, torch.ones_like(xsur))
            w = torch.where(sel_tgt.reshape(-1), 1.0 - a,
                            torch.ones(B * T, device=DEV))
        if emb_scale is not None:
            logits, _ = model_forward_scaled(model, xt, emb_scale)
        else:
            logits, _ = model(xt)
        ce = F.cross_entropy(logits.reshape(-1, V), yt.reshape(-1), reduction="none")
        loss = (ce * w).sum() / w.sum() if w is not None else ce.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg["log_every"] == 0 or step == cfg["train_steps"] - 1:
            print(f"step {step} loss {loss.item():.4f}", flush=True)
        if (step + 1) in traj_at:
            t_v = evaluate(val_ids, val_sur, cfg["eval_windows"])["loss"]
            t_o = {n: evaluate(i, s, cfg["eval_windows"])["loss"]
                   for n, (i, s) in ood_sets.items()}
            traj.append(dict(step=step + 1, val=t_v, ood=t_o))
            print(f"TRAJ step {step+1} val {t_v:.4f} ood {t_o}", flush=True)

    # ---------------- final evals ----------------
    final = evaluate(val_ids, val_sur, cfg["eval_windows"], collect=True)
    ood = {n: evaluate(i, s, cfg["eval_windows"]) for n, (i, s) in ood_sets.items()}
    robust = evaluate(val_ids, val_sur, cfg["eval_windows"], corrupt_rate=0.25)

    gate_stats = None
    if cfg["gate"]["enabled"] and not cfg["gate"]["kill"]:
        model.eval()
        gs, ss = [], []
        with torch.no_grad():
            ix = np.random.default_rng(999).integers(0, len(val_ids) - T, size=4 * B)
            for wi in range(4):
                xs = np.stack([val_ids[i:i + T] for i in ix[wi * B:(wi + 1) * B]]).astype(np.int64)
                s2 = np.stack([val_sur[i:i + T] for i in ix[wi * B:(wi + 1) * B]])
                xt = torch.from_numpy(xs).to(DEV)
                emb = model.emb(xt)
                g = model.gate(emb)
                gs.append((g.norm(dim=-1) / (emb.norm(dim=-1) + 1e-8)).reshape(-1).cpu().numpy())
                ss.append(torch.from_numpy(s2).reshape(-1).numpy())
        g = np.concatenate(gs)
        s = np.concatenate(ss)
        corr = float(np.corrcoef(g, s)[0, 1]) if np.std(g) > 1e-8 and np.std(s) > 1e-8 else 0.0
        gate_stats = dict(gate_scale_mean=float(g.mean()), gate_sur_corr=corr)

    rec = dict(run=cfg["run_name"], cfg=json.loads(json.dumps(cfg)),
               params=sum(p.numel() for p in model.parameters()),
               steps=cfg["train_steps"], val=final, ood=ood, robust=robust,
               gate_stats=gate_stats, traj=traj or None,
               elapsed_s=round(time.time() - t0, 1))
    out_path = os.path.join(HERE, results_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("RESULT", json.dumps(dict(
        run=rec["run"], val=final["loss"],
        ood={n: v["loss"] for n, v in ood.items()},
        robust=robust["loss"], gate=gate_stats)), flush=True)
    return rec


def model_forward_scaled(model, idx, emb_scale):
    """Forward pass with input embeddings scaled (damp arm)."""
    x = model.emb(idx) * emb_scale.unsqueeze(-1)
    if model.gate is not None:
        x = x + model.gate(x)
    x = x + model.pos(torch.arange(idx.shape[1], device=idx.device))
    ws = []
    for b in model.blocks:
        x, _ = b(x, model.mask_f)
    return model.head(model.lnf(x)), ws


if __name__ == "__main__":
    name = sys.argv[1]
    smoke = len(sys.argv) > 2 and sys.argv[2] == "smoke"
    results_path = "smoke.jsonl" if smoke else os.path.join("results", "results.jsonl")
    if not smoke and os.path.exists(os.path.join(HERE, results_path)):
        done = {json.loads(l)["run"] for l in open(os.path.join(HERE, results_path))}
        if name in done:
            print(f"{name} already in results.jsonl, skipping")
            sys.exit(0)
    run(resolve(name, smoke), results_path)
