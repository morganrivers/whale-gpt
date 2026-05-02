"""
Held-out next-Token prediction with progressively bigger models.

Single target = the full coda Token (V≈207). Single metric = bits/token
cross-entropy on the held-out 20 % of sequences (log2 nats / log(2)).
Sequence-level 5-fold CV. Context K = 8 past codas (truncated/padded).

Models:

  M0  Majority             constant prediction = train mode
  M1  Markov-1             P(y | last_token), Laplace-smoothed (alpha=0.5)
  M2  Markov-2             P(y | last 2 tokens), Laplace-smoothed
  M3  MLP-S    (128,64)    sklearn, one-hot last-K context
  M4  MLP-M    (256,128)   sklearn, one-hot last-K context
  M5  MLP-L    (512,256,128) sklearn, one-hot last-K context
  M6  EmbMLP                pytorch, 32-d learned token embeddings + MLP
  M7  MiniTfm  (2L, 4h)    pytorch, embedding + 2x self-attention block

Outputs go to outputs/grammar/predict_models.{md,json}.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ROOT / "data" / "whale_tokens.csv"
OUT_JSON = ROOT / "outputs" / "grammar" / "predict_models.json"
OUT_MD = ROOT / "outputs" / "grammar" / "predict_models.md"

CONTEXT_K = 8
N_FOLDS = 5
TARGET = "Token"
PAD = "<PAD>"


def per_seq(tokens: pd.DataFrame, col: str) -> dict[int, list[str]]:
    return {
        int(seq_id): [str(v) for v in g.sort_values("itemPosition")[col].tolist()]
        for seq_id, g in tokens.groupby("sequenceId")
    }


def build_windows(seq_dict, seq_ids, k):
    X, y = [], []
    for sid in seq_ids:
        s = seq_dict[sid]
        for i in range(1, len(s)):
            ctx = s[max(0, i - k) : i]
            ctx = [PAD] * (k - len(ctx)) + ctx
            X.append(ctx)
            y.append(s[i])
    return np.array(X, dtype=object), np.array(y, dtype=object)


def bits_per_token(p_true: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.mean(np.log2(np.maximum(p_true, eps))))


def evaluate_baselines(y_train, X_train, y_test, X_test, vocab_size, classes):
    cls_idx = {c: i for i, c in enumerate(classes)}
    out = {}

    # majority
    train_counts = Counter(y_train.tolist())
    majority = train_counts.most_common(1)[0][0]
    n_train = len(y_train)
    smoothing = 0.5
    p_uniform_smoothed = np.array(
        [
            (train_counts.get(t, 0) + smoothing) / (n_train + smoothing * vocab_size)
            for t in y_test
        ]
    )
    out["M0_majority"] = dict(
        accuracy=float(np.mean(y_test == majority)),
        bits_per_token=bits_per_token(p_uniform_smoothed),
    )

    # Markov-k
    def markov_k(k):
        trans: dict[tuple, Counter] = {}
        for i, ctx in enumerate(X_train):
            key = tuple(ctx[-k:])
            trans.setdefault(key, Counter())[y_train[i]] += 1
        p_true = np.empty(len(y_test))
        preds = []
        for i, ctx in enumerate(X_test):
            key = tuple(ctx[-k:])
            c = trans.get(key)
            if c is None:
                p_true[i] = (train_counts.get(y_test[i], 0) + smoothing) / (
                    n_train + smoothing * vocab_size
                )
                preds.append(majority)
            else:
                denom = sum(c.values()) + smoothing * vocab_size
                p_true[i] = (c.get(y_test[i], 0) + smoothing) / denom
                preds.append(c.most_common(1)[0][0])
        return dict(
            accuracy=float(np.mean(np.array(preds) == y_test)),
            bits_per_token=bits_per_token(p_true),
        )

    out["M1_markov1"] = markov_k(1)
    out["M2_markov2"] = markov_k(2)
    return out


def evaluate_sklearn_mlp(name, hidden, X_train_oh, y_train, X_test_oh, y_test, classes):
    le = LabelEncoder().fit(np.concatenate([y_train, y_test]))
    yti = le.transform(y_train)
    yvi = le.transform(y_test)
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=0,
        n_iter_no_change=15,
    )
    mlp.fit(X_train_oh, yti)
    pred_int = mlp.predict(X_test_oh)
    probs = mlp.predict_proba(X_test_oh)
    cls_to_col = {c: i for i, c in enumerate(le.inverse_transform(mlp.classes_))}
    p_true = np.array(
        [probs[i, cls_to_col[t]] if t in cls_to_col else 1e-12 for i, t in enumerate(y_test)]
    )
    return dict(
        accuracy=float(np.mean(le.inverse_transform(pred_int) == y_test)),
        bits_per_token=bits_per_token(p_true),
        n_params=int(sum(p.size for p in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)),
    )


# ------------------ pytorch models -----------------------------------------

class EmbMLP(nn.Module):
    def __init__(self, V, k, d=32, hidden=(256, 128)):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        layers = []
        in_dim = d * k
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = h
        layers.append(nn.Linear(in_dim, V))
        self.head = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, k) long
        e = self.emb(x).flatten(1)
        return self.head(e)


class MiniTransformer(nn.Module):
    def __init__(self, V, k, d=64, n_layers=2, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(k, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4 * d, dropout=0.1,
            batch_first=True, activation="gelu",
        )
        self.tfm = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d, V)
        self.k = k

    def forward(self, x):  # x: (B, k)
        B, K = x.shape
        pos_ids = torch.arange(K, device=x.device).expand(B, K)
        h = self.emb(x) + self.pos(pos_ids)
        h = self.tfm(h)
        return self.head(h[:, -1])  # predict from last context position


def train_torch(model, X_train, y_train, X_val, y_val, epochs=80, lr=1e-3, bs=128, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    n = X_train.shape[0]
    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    patience, since = 10, 0
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i : i + bs]
            xb, yb = X_train[b], y_train[b]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(X_val), y_val).item()
        if val < best_val - 1e-4:
            best_val, since = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    return model


def evaluate_torch_model(name, model_factory, X_train, y_train, X_test, y_test, vocab_size, classes, seed=0):
    le = LabelEncoder().fit(np.concatenate([y_train, y_test]))
    pad_id = le.transform([PAD])[0] if PAD in le.classes_ else None
    if pad_id is None:
        # PAD wasn't in train; extend by adding a dummy
        # Easier: make a separate vocab including PAD explicitly
        all_classes = list(le.classes_) + [PAD]
        le2 = LabelEncoder().fit(np.array(all_classes))
        pad_id = le2.transform([PAD])[0]
        Xtr_int = np.array([[le2.transform([c])[0] if c in le2.classes_ else pad_id for c in row] for row in X_train])
        Xte_int = np.array([[le2.transform([c])[0] if c in le2.classes_ else pad_id for c in row] for row in X_test])
        ytr_int = le2.transform(y_train)
        yte_int = le2.transform(y_test)
        V_full = len(le2.classes_)
        used_le = le2
    else:
        Xtr_int = np.array([[le.transform([c])[0] if c in le.classes_ else pad_id for c in row] for row in X_train])
        Xte_int = np.array([[le.transform([c])[0] if c in le.classes_ else pad_id for c in row] for row in X_test])
        ytr_int = le.transform(y_train)
        yte_int = le.transform(y_test)
        V_full = len(le.classes_)
        used_le = le

    Xtr_t = torch.from_numpy(Xtr_int).long()
    Xte_t = torch.from_numpy(Xte_int).long()
    ytr_t = torch.from_numpy(ytr_int).long()
    yte_t = torch.from_numpy(yte_int).long()

    # split a small val set out of train for early stopping
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr_t))
    n_val = max(64, int(0.1 * len(Xtr_t)))
    val_idx = torch.from_numpy(perm[:n_val])
    tr_idx = torch.from_numpy(perm[n_val:])

    model = model_factory(V_full)
    n_params = sum(p.numel() for p in model.parameters())

    train_torch(
        model, Xtr_t[tr_idx], ytr_t[tr_idx], Xtr_t[val_idx], ytr_t[val_idx], seed=seed
    )

    model.eval()
    with torch.no_grad():
        logits = model(Xte_t)
        log_probs = torch.log_softmax(logits, dim=-1)
        # bits/token = -mean log2 P(true)
        ll = log_probs.gather(1, yte_t.unsqueeze(1)).squeeze(1)  # nats per token
        bits = -ll.mean().item() / math.log(2)
        pred = logits.argmax(-1)
        acc = (pred == yte_t).float().mean().item()
    return dict(accuracy=float(acc), bits_per_token=float(bits), n_params=int(n_params))


def evaluate_one_fold(seq_dict, train_ids, test_ids, k):
    X_train, y_train = build_windows(seq_dict, train_ids, k)
    X_test, y_test = build_windows(seq_dict, test_ids, k)
    classes = sorted(set(y_train.tolist()) | set(y_test.tolist()) | {PAD})
    V = len(classes)

    # one-hot encoding for sklearn models
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(np.array(X_train, dtype=object))
    Xtr_oh = enc.transform(X_train)
    Xte_oh = enc.transform(X_test)

    res = evaluate_baselines(y_train, X_train, y_test, X_test, V, classes)
    res["M3_MLP_S"] = evaluate_sklearn_mlp("MLP-S", (128, 64), Xtr_oh, y_train, Xte_oh, y_test, classes)
    res["M4_MLP_M"] = evaluate_sklearn_mlp("MLP-M", (256, 128), Xtr_oh, y_train, Xte_oh, y_test, classes)
    res["M5_MLP_L"] = evaluate_sklearn_mlp("MLP-L", (512, 256, 128), Xtr_oh, y_train, Xte_oh, y_test, classes)

    res["M6_EmbMLP"] = evaluate_torch_model(
        "EmbMLP", lambda V: EmbMLP(V, k, d=32, hidden=(256, 128)),
        X_train, y_train, X_test, y_test, V, classes,
    )
    res["M7_MiniTfm"] = evaluate_torch_model(
        "MiniTfm", lambda V: MiniTransformer(V, k, d=64, n_layers=2, n_heads=4),
        X_train, y_train, X_test, y_test, V, classes,
    )
    res["_meta"] = dict(
        n_train=int(len(X_train)), n_test=int(len(X_test)), V=V
    )
    return res


def main() -> None:
    tokens = pd.read_csv(TOKENS)
    seq_dict = per_seq(tokens, TARGET)
    seq_ids = sorted(seq_dict.keys())
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    for fi, (tr_idx, te_idx) in enumerate(kf.split(seq_ids)):
        train_ids = [seq_ids[i] for i in tr_idx]
        test_ids = [seq_ids[i] for i in te_idx]
        t0 = time.time()
        res = evaluate_one_fold(seq_dict, train_ids, test_ids, CONTEXT_K)
        res["_meta"]["fold"] = fi
        res["_meta"]["seconds"] = round(time.time() - t0, 1)
        fold_results.append(res)
        models = [k for k in res if k.startswith("M")]
        print(
            f"fold {fi}: "
            + "  ".join(
                f"{m}: bpt={res[m]['bits_per_token']:.3f} acc={res[m]['accuracy']:.3f}"
                for m in models
            )
            + f"  ({res['_meta']['seconds']}s)"
        )

    # aggregate
    summary = {}
    model_keys = [k for k in fold_results[0] if k.startswith("M")]
    for m in model_keys:
        bits = [f[m]["bits_per_token"] for f in fold_results]
        accs = [f[m]["accuracy"] for f in fold_results]
        n_params = fold_results[0][m].get("n_params")
        summary[m] = dict(
            bits_per_token_mean=float(np.mean(bits)),
            bits_per_token_std=float(np.std(bits)),
            perplexity_mean=float(2 ** np.mean(bits)),
            accuracy_mean=float(np.mean(accs)),
            accuracy_std=float(np.std(accs)),
            n_params=n_params,
        )

    out = dict(folds=fold_results, summary=summary, k=CONTEXT_K, target=TARGET)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))

    L = []
    L.append("# 5-fold next-Token prediction: bigger models, log-loss")
    L.append("")
    L.append(
        f"Sequence-level KFold ({N_FOLDS} folds, no within-sequence leakage). "
        f"Target = `{TARGET}` (V ≈ 207). Context K = {CONTEXT_K} past codas. "
        "Metric = held-out cross-entropy in **bits/token** (log₂); lower is better. "
        "Perplexity = 2^(bits/token)."
    )
    L.append("")
    L.append("| # | model | params | bits/token (↓) | perplexity (↓) | accuracy |")
    L.append("|---|-------|-------:|---------------:|---------------:|---------:|")
    label_for = {
        "M0_majority": "majority (smoothed unigram)",
        "M1_markov1": "Markov-1",
        "M2_markov2": "Markov-2",
        "M3_MLP_S": "MLP-S (128, 64)",
        "M4_MLP_M": "MLP-M (256, 128)",
        "M5_MLP_L": "MLP-L (512, 256, 128)",
        "M6_EmbMLP": "Embedding-MLP (d=32, 256·128)",
        "M7_MiniTfm": "MiniTransformer (2L, 4h, d=64)",
    }
    for m in ["M0_majority", "M1_markov1", "M2_markov2", "M3_MLP_S", "M4_MLP_M", "M5_MLP_L", "M6_EmbMLP", "M7_MiniTfm"]:
        s = summary[m]
        params = f"{s['n_params']:,}" if s.get("n_params") else "—"
        L.append(
            f"| {m[1]} | {label_for[m]} | {params} | "
            f"{s['bits_per_token_mean']:.3f} ± {s['bits_per_token_std']:.3f} | "
            f"{s['perplexity_mean']:.2f} | "
            f"{s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f} |"
        )
    L.append("")
    L.append("## Per-fold bits/token")
    L.append("")
    L.append("| fold |" + "|".join(f" {label_for[m]} " for m in label_for) + "|")
    L.append("|------|" + "|".join(["---:"] * len(label_for)) + "|")
    for f in fold_results:
        L.append(
            f"| {f['_meta']['fold']} |"
            + "|".join(f" {f[m]['bits_per_token']:.3f} " for m in label_for)
            + "|"
        )
    L.append("")
    best_m = min(summary, key=lambda m: summary[m]["bits_per_token_mean"])
    L.append("## Takeaway")
    L.append("")
    L.append(
        f"- The best held-out model is **{label_for[best_m]}** at "
        f"{summary[best_m]['bits_per_token_mean']:.3f} bits/token "
        f"(perplexity ≈ {summary[best_m]['perplexity_mean']:.2f})."
    )
    L.append(
        f"- The unigram majority baseline scores {summary['M0_majority']['bits_per_token_mean']:.3f} "
        f"bits/token; the best model saves "
        f"**{summary['M0_majority']['bits_per_token_mean'] - summary[best_m]['bits_per_token_mean']:.2f} bits/token** "
        f"(perplexity drops {summary['M0_majority']['perplexity_mean']:.1f} → "
        f"{summary[best_m]['perplexity_mean']:.1f})."
    )
    L.append(
        f"- Markov-1 alone is already strong ({summary['M1_markov1']['bits_per_token_mean']:.3f} bpt). "
        f"Larger models help further; the MiniTransformer / Emb-MLP can attend to "
        f"the full {CONTEXT_K}-coda context."
    )
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
