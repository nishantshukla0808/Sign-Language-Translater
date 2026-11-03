"""
Train a small phrase classifier (1D-CNN + BiGRU) on your recorded clips and compare it with
the DTW few-shot matcher on the SAME cross-validation folds.

    /home/vlmpi/venv-export/bin/python train_phrases.py --person nishu

Needs PyTorch (the export venv has it; the runtime .venv does not). Writes
models/phrases.onnx + models/phrases_labels.json (runtime needs only onnxruntime).

Honesty note: folds are stratified random splits of clips that were recorded in the same
sessions, so the CV score is OPTIMISTIC. A new day / new person will score lower.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fewshot
from fewshot import T_LEN, features, score, slug
from landmarks import LHAND_OFF, RHAND_OFF

p = argparse.ArgumentParser()
p.add_argument("--person", required=True, help="one or several comma-separated people under data/, e.g. nishu,phone")
p.add_argument("--data", default="data")
p.add_argument("--epochs", type=int, default=80)
p.add_argument("--folds", type=int, default=5)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="models")
p.add_argument("--no-export", action="store_true")
p.add_argument("--no-cv", action="store_true", help="skip cross-validation (fast retrain)")
args = p.parse_args()
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
torch.set_num_threads(4)


# ---- data ------------------------------------------------------------------------------
def hand_flags(win: np.ndarray) -> np.ndarray:
    idx = np.round(np.linspace(0, len(win) - 1, T_LEN)).astype(int)
    return np.stack([np.isfinite(win[idx, LHAND_OFF, 0]), np.isfinite(win[idx, RHAND_OFF, 0])], 1).astype(np.float32)


people = [slug(x) for x in args.person.split(",") if x.strip()]
phrase_dirs: dict[str, list[Path]] = {}
for person in people:
    for d in sorted(q for q in (Path(args.data) / person).iterdir() if q.is_dir()):
        phrase_dirs.setdefault(d.name, []).extend(sorted(d.glob("*.npy")))
labels, X, M, names = [], [], [], []
for name in sorted(phrase_dirs):
    if not phrase_dirs[name]:
        continue
    labels.append(name)
    for c in phrase_dirs[name]:
        w = np.load(c)
        X.append(features(w).reshape(T_LEN, 49, 2).astype(np.float32))
        M.append(hand_flags(w))
        names.append((len(labels) - 1, c))
X, M = np.stack(X), np.stack(M)
y = np.array([n[0] for n in names])
C = len(labels)
print(f"[data] {len(y)} clips, {C} phrases: " + ", ".join(f"{labels[i]}={int((y == i).sum())}" for i in range(C)), flush=True)


def build_input(base: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """base (B,T,49,2), mask (B,T,2) -> (B,T,98+98+2): positions, velocities, hand-present flags."""
    flat = base.reshape(base.shape[0], base.shape[1], -1)
    vel = torch.cat([torch.zeros_like(flat[:, :1]), flat[:, 1:] - flat[:, :-1]], 1) * 4.0
    return torch.cat([flat, vel, mask], -1)


def augment(base: torch.Tensor, mask: torch.Tensor):
    B = base.shape[0]
    present = (base.abs().sum(-1, keepdim=True) > 0).float()
    ang = (torch.rand(B) * 2 - 1) * 8 * np.pi / 180
    sc = 0.9 + 0.2 * torch.rand(B)
    cs, sn = torch.cos(ang) * sc, torch.sin(ang) * sc
    R = torch.stack([torch.stack([cs, -sn], -1), torch.stack([sn, cs], -1)], -2)      # (B,2,2)
    shift = (torch.rand(B, 1, 1, 2) * 2 - 1) * 0.08
    out = torch.einsum("btnc,bcd->btnd", base, R) + shift
    out = out + torch.randn_like(out) * 0.01
    out = out * present                                                                  # missing stays missing
    L = rng.integers(24, T_LEN + 1, size=B)
    o2, m2 = torch.zeros_like(out), torch.zeros_like(mask)
    for i in range(B):                                                                   # random time crop + stretch
        s0 = int(rng.integers(0, T_LEN - L[i] + 1))
        seg = out[i, s0:s0 + L[i]].reshape(1, L[i], -1).permute(0, 2, 1)
        o2[i] = F.interpolate(seg, size=T_LEN, mode="linear", align_corners=True).permute(0, 2, 1).reshape(T_LEN, 49, 2)
        idx = torch.linspace(0, L[i] - 1, T_LEN).round().long() + s0
        m2[i] = mask[i, idx]
        if rng.random() < 0.2:                                                           # brief hand dropout
            f0 = int(rng.integers(0, T_LEN - 3))
            o2[i, f0:f0 + 3, 7:] = 0
            m2[i, f0:f0 + 3] = 0
    return o2, m2


class Net(nn.Module):
    def __init__(self, n_in, n_cls, h=64):
        super().__init__()
        self.c1 = nn.Conv1d(n_in, h, 5, padding=2)
        self.b1 = nn.BatchNorm1d(h)
        self.c2 = nn.Conv1d(h, h, 5, padding=2)
        self.b2 = nn.BatchNorm1d(h)
        self.gru = nn.GRU(h, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(4 * h, n_cls)

    def forward(self, x):                     # (B,T,F)
        z = x.transpose(1, 2)
        z = F.relu(self.b1(self.c1(z)))
        z = F.relu(self.b2(self.c2(z)))
        z, _ = self.gru(z.transpose(1, 2))
        z = torch.cat([z.mean(1), z.amax(1)], -1)
        return self.fc(self.drop(z))


def train(idx_tr, epochs, verbose=False):
    Xb, Mb = torch.from_numpy(X[idx_tr]), torch.from_numpy(M[idx_tr])
    yb = torch.from_numpy(y[idx_tr])
    net = Net(198, C)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=epochs * ((len(idx_tr) + 31) // 32))
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(idx_tr))
        tot = 0.0
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            xa, ma = augment(Xb[b], Mb[b])
            loss = F.cross_entropy(net(build_input(xa, ma)), yb[b], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        if verbose and (ep + 1) % 20 == 0:
            print(f"   epoch {ep + 1}/{epochs}  loss {tot / len(perm):.3f}", flush=True)
    return net.eval()


def predict(net, idx):
    with torch.no_grad():
        return F.softmax(net(build_input(torch.from_numpy(X[idx]), torch.from_numpy(M[idx]))), -1).numpy()


# ---- cross-validation: network vs DTW on the same folds -----------------------------------
fold_of = np.zeros(len(y), int)
for c in range(C):
    ids = rng.permutation(np.flatnonzero(y == c))
    fold_of[ids] = np.arange(len(ids)) % args.folds

tpl_all = {labels[c]: [(names[i][1], features(np.load(names[i][1])))
                       for i in np.flatnonzero(y == c)] for c in range(C)} if not args.no_cv else {}
net_correct = dtw_correct = 0
conf_net = np.zeros((C, C), int)
conf_dtw = np.zeros((C, C), int)
p_right, p_wrong = [], []
t0 = time.time()
for k in range(0 if args.no_cv else args.folds):
    tr, te = np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)
    net = train(tr, args.epochs)
    pr = predict(net, te)
    for j, i in enumerate(te):
        pn = int(pr[j].argmax())
        conf_net[y[i], pn] += 1
        (p_right if pn == y[i] else p_wrong).append(float(pr[j].max()))
        test_names = {names[t][1] for t in te}
        tpl = {lab: [(n, f) for n, f in items if n not in test_names] for lab, items in tpl_all.items()}
        sc = score(features(np.load(names[i][1])), tpl)
        pd = labels.index(min(sc, key=sc.get))
        conf_dtw[y[i], pd] += 1
    a_net = float((pr.argmax(1) == y[te]).mean())
    print(f"[cv] fold {k + 1}/{args.folds}: network {a_net:.0%}   ({time.time() - t0:.0f}s elapsed)", flush=True)

n = len(y)
if not args.no_cv:
    print(f"\n[cv] OVERALL  network {np.trace(conf_net)}/{n} = {np.trace(conf_net) / n:.1%}   "
          f"DTW matcher {np.trace(conf_dtw)}/{n} = {np.trace(conf_dtw) / n:.1%}")
    print("[cv] per phrase (network right / total   |   DTW right / total):")
    for c in range(C):
        print(f"   {labels[c]:<16} {conf_net[c, c]:>2}/{conf_net[c].sum():<2}   |   {conf_dtw[c, c]:>2}/{conf_dtw[c].sum():<2}")
    print("[cv] network confusions (true -> predicted):", ", ".join(
        f"{labels[a]}->{labels[b]} x{conf_net[a, b]}" for a in range(C) for b in range(C) if a != b and conf_net[a, b]) or "none")
    if p_wrong:
        print(f"[cv] confidence: mean top-prob when right {np.mean(p_right):.2f}, when wrong {np.mean(p_wrong):.2f}")

# ---- final model on all data + ONNX export ------------------------------------------------
if not args.no_export:
    t1 = time.time()
    net = train(np.arange(len(y)), args.epochs, verbose=True)
    print(f"[final] trained on all {len(y)} clips in {time.time() - t1:.0f}s", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, T_LEN, 198)
    torch.onnx.export(net, dummy, str(out / "phrases.onnx"), input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    (out / "phrases_labels.json").write_text(json.dumps(labels, indent=1))
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out / "phrases.onnx"), providers=["CPUExecutionProvider"])
    xs = build_input(torch.from_numpy(X[:8]), torch.from_numpy(M[:8]))
    diff = float(np.abs(sess.run(None, {"x": xs.numpy()})[0] - net(xs).detach().numpy()).max())
    print(f"[export] wrote {out / 'phrases.onnx'} ({(out / 'phrases.onnx').stat().st_size / 1024:.0f} KB), "
          f"labels -> {out / 'phrases_labels.json'};  ONNX vs torch max diff {diff:.1e}")
    print("[final] NOTE: the training-set accuracy of this model is not an accuracy estimate; use the CV numbers above.")
