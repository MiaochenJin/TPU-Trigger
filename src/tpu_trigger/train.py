"""Training loop for TriggerNet variants on mock data."""

import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn

from .mockdata import make_dataset
from .models import count_params, make_model


def auc_score(scores, labels):
    """Mann-Whitney AUC with tie handling (int8 scores are discrete)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return (r[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _to_input(x):
    """(N, C, T) -> (N, C, 1, T) float32 torch tensor."""
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(2)


@torch.no_grad()
def evaluate(model, x, y, device, batch=256):
    model.eval()
    logits = []
    for i in range(0, len(x), batch):
        xb = _to_input(x[i:i + batch]).to(device)
        logits.append(model(xb).cpu().numpy())
    logits = np.concatenate(logits)
    scores = logits[:, 1] - logits[:, 0]
    acc = float(((scores > 0).astype(np.int64) == y).mean())
    return acc, auc_score(scores, y), logits


def train(variant, T=256, n_train=20000, n_val=4000, epochs=5, batch=128,
          lr=1e-3, snr=2.0, seed=0, outdir="runs", device=None,
          data=None, n_ch=16):
    """data: optional (x_tr, y_tr, x_va, y_va) arrays, x of shape (N, n_ch, T);
    if omitted, mock data is generated (n_ch=16)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(outdir) / variant
    out.mkdir(parents=True, exist_ok=True)

    if data is None:
        print(f"[{variant}] generating mock data (T={T}, snr={snr})...")
        x_tr, y_tr = make_dataset(n_train, T=T, snr=snr, seed=seed)
        x_va, y_va = make_dataset(n_val, T=T, snr=snr, seed=seed + 1)
    else:
        x_tr, y_tr, x_va, y_va = data
    n_train = len(x_tr)

    model = make_model(variant, T=T, n_ch=n_ch).to(device)
    print(f"[{variant}] {count_params(model):,} params, device={device}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best = {"val_acc": 0.0}
    ckpt = out / "best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        perm = np.random.default_rng(seed + epoch).permutation(n_train)
        losses = []
        for i in range(0, n_train, batch):
            idx = perm[i:i + batch]
            xb = _to_input(x_tr[idx]).to(device)
            yb = torch.from_numpy(y_tr[idx]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val_acc, val_auc, _ = evaluate(model, x_va, y_va, device)
        print(f"[{variant}] epoch {epoch}/{epochs}: "
              f"loss={np.mean(losses):.4f} val_acc={val_acc:.4f} "
              f"val_auc={val_auc:.4f} ({time.time() - t0:.0f}s)")
        if val_acc >= best["val_acc"]:
            best = {"val_acc": val_acc, "val_auc": val_auc, "epoch": epoch}
            torch.save(model.state_dict(), ckpt)

    print(f"[{variant}] best: {best}")
    return {"variant": variant, "ckpt": str(ckpt),
            "params": count_params(model), **best}
