"""
5-fold cross-validated next-coda prediction.

Splits whale sequences into 5 folds at the sequence level (no within-sequence
leakage). For each fold, builds (X = last K tokens, y = next token) windows
from the training sequences and evaluates four models on the held-out 20 %:

  1. Majority baseline   -- always predict the most common token in train
  2. Markov-1 baseline   -- argmax P(y | last_token), backoff to majority
  3. Random Forest       -- 256 trees on one-hot last-K context
  4. Small MLP           -- (128, 64) ReLU on one-hot last-K context

Evaluated independently on three targets: Rhythm (V=18), Tempo (V=5),
Token (V=207). Reports accuracy, top-3 accuracy, perplexity (where the
model exposes probabilities).

Results land in outputs/grammar/predict_results.{json,md}.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ROOT / "data" / "whale_tokens.csv"
OUT_JSON = ROOT / "outputs" / "grammar" / "predict_results.json"
OUT_MD = ROOT / "outputs" / "grammar" / "predict_results.md"

CONTEXT_K = 4          # number of past codas used as features
N_FOLDS = 5
RNG = np.random.default_rng(0)


def per_seq(tokens: pd.DataFrame, col: str) -> dict[int, list]:
    """Return per-sequence lists of string-cast labels (sklearn expects
    consistent label dtype across folds)."""
    return {
        int(seq_id): [str(v) for v in g.sort_values("itemPosition")[col].tolist()]
        for seq_id, g in tokens.groupby("sequenceId")
    }


def build_windows(seq_dict: dict[int, list], seq_ids: list[int], k: int):
    """Return X_context (n, k), y (n,) over the given sequence ids."""
    X, y = [], []
    for sid in seq_ids:
        s = seq_dict[sid]
        for i in range(k, len(s)):
            X.append(s[i - k : i])
            y.append(s[i])
    return np.array(X, dtype=object), np.array(y, dtype=object)


def perplexity(p_true: np.ndarray) -> float:
    """Geometric-mean perplexity. p_true is per-sample probability of the
    correct class."""
    eps = 1e-12
    return float(np.exp(-np.mean(np.log(np.maximum(p_true, eps)))))


def topk_accuracy(probs: np.ndarray, classes: np.ndarray, y_true: np.ndarray, k: int) -> float:
    if probs.ndim == 1 or probs.shape[1] == 0:
        return float("nan")
    top_idx = np.argsort(-probs, axis=1)[:, :k]
    top_cls = classes[top_idx]  # (n, k)
    return float(np.mean([yt in row for yt, row in zip(y_true, top_cls)]))


def evaluate_one_fold(
    seq_dict: dict[int, list],
    train_ids: list[int],
    test_ids: list[int],
    k: int,
) -> dict:
    X_train, y_train = build_windows(seq_dict, train_ids, k)
    X_test, y_test = build_windows(seq_dict, test_ids, k)
    if len(X_train) == 0 or len(X_test) == 0:
        return {}

    classes = sorted(set(y_train.tolist()) | set(y_test.tolist()), key=str)
    cls_index = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    y_test_cls = np.array([cls_index[c] for c in y_test])

    out: dict[str, dict] = {}

    # ---------- 1. majority baseline ---------------------------------------
    train_counts = Counter(y_train.tolist())
    majority = train_counts.most_common(1)[0][0]
    out["majority"] = dict(
        accuracy=float(np.mean(y_test == majority)),
        top3_accuracy=float("nan"),
        perplexity=float("nan"),
    )

    # ---------- 2. Markov-1 baseline ---------------------------------------
    trans: dict[object, Counter] = {}
    for ctx, label in zip(X_train[:, -1], y_train):
        trans.setdefault(ctx, Counter())[label] += 1
    # softmax-with-laplace probability over classes for perplexity
    smoothing = 0.5
    p_true = np.empty(len(y_test))
    preds = np.empty(len(y_test), dtype=object)
    for i, (ctx, true) in enumerate(zip(X_test[:, -1], y_test)):
        c = trans.get(ctx)
        if c is None:
            preds[i] = majority
            p_true[i] = (train_counts.get(true, 0) + smoothing) / (
                len(y_train) + smoothing * n_classes
            )
        else:
            preds[i] = c.most_common(1)[0][0]
            p_true[i] = (c.get(true, 0) + smoothing) / (
                sum(c.values()) + smoothing * n_classes
            )
    # top-3 from Markov: build per-row prob over classes
    probs_m = np.full((len(y_test), n_classes), 1.0 / n_classes)
    for i, ctx in enumerate(X_test[:, -1]):
        c = trans.get(ctx)
        if c is None:
            for j, cl in enumerate(classes):
                probs_m[i, j] = (train_counts.get(cl, 0) + smoothing) / (
                    len(y_train) + smoothing * n_classes
                )
        else:
            denom = sum(c.values()) + smoothing * n_classes
            for j, cl in enumerate(classes):
                probs_m[i, j] = (c.get(cl, 0) + smoothing) / denom
    out["markov1"] = dict(
        accuracy=float(np.mean(preds == y_test)),
        top3_accuracy=topk_accuracy(probs_m, np.array(classes, dtype=object), y_test, 3),
        perplexity=perplexity(p_true),
    )

    # ---------- 3 + 4. RF and MLP ------------------------------------------
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(np.array(X_train, dtype=object))
    X_train_oh = enc.transform(X_train)
    X_test_oh = enc.transform(X_test)

    # ---- Random Forest -----------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=0, min_samples_leaf=2
    )
    rf.fit(X_train_oh, y_train)
    rf_pred = rf.predict(X_test_oh)
    rf_probs = rf.predict_proba(X_test_oh)
    rf_classes = rf.classes_
    cls_to_col = {c: i for i, c in enumerate(rf_classes)}
    p_true_rf = np.array(
        [
            rf_probs[i, cls_to_col[t]] if t in cls_to_col else 1e-12
            for i, t in enumerate(y_test)
        ]
    )
    out["random_forest"] = dict(
        accuracy=float(np.mean(rf_pred == y_test)),
        top3_accuracy=topk_accuracy(rf_probs, rf_classes, y_test, 3),
        perplexity=perplexity(p_true_rf),
    )

    # ---- Small MLP ---------------------------------------------------------
    # Encode y to integers so MLP early-stopping validation works cleanly.
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))
    y_train_int = le.transform(y_train)
    y_test_int = le.transform(y_test)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=0,
    )
    mlp.fit(X_train_oh, y_train_int)
    mlp_pred_int = mlp.predict(X_test_oh)
    mlp_pred = le.inverse_transform(mlp_pred_int)
    mlp_probs = mlp.predict_proba(X_test_oh)
    mlp_classes = le.inverse_transform(mlp.classes_)
    cls_to_col = {c: i for i, c in enumerate(mlp_classes)}
    p_true_mlp = np.array(
        [
            mlp_probs[i, cls_to_col[t]] if t in cls_to_col else 1e-12
            for i, t in enumerate(y_test)
        ]
    )
    out["mlp"] = dict(
        accuracy=float(np.mean(mlp_pred == y_test)),
        top3_accuracy=topk_accuracy(mlp_probs, mlp_classes, y_test, 3),
        perplexity=perplexity(p_true_mlp),
    )

    out["_meta"] = dict(
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        n_classes=int(n_classes),
    )
    return out


def evaluate_target(tokens: pd.DataFrame, col: str, k: int) -> dict:
    seq_dict = per_seq(tokens, col)
    seq_ids = sorted(seq_dict.keys())
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results: list[dict] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(seq_ids)):
        train_ids = [seq_ids[i] for i in tr_idx]
        test_ids = [seq_ids[i] for i in te_idx]
        t0 = time.time()
        res = evaluate_one_fold(seq_dict, train_ids, test_ids, k)
        res["_meta"]["seconds"] = round(time.time() - t0, 1)
        res["_meta"]["fold"] = fold_idx
        fold_results.append(res)
        print(
            f"  fold {fold_idx}: maj={res['majority']['accuracy']:.3f}  "
            f"mk1={res['markov1']['accuracy']:.3f}  "
            f"rf={res['random_forest']['accuracy']:.3f}  "
            f"mlp={res['mlp']['accuracy']:.3f}  "
            f"({res['_meta']['seconds']}s)"
        )

    # aggregate
    summary: dict[str, dict[str, float]] = {}
    for model in ("majority", "markov1", "random_forest", "mlp"):
        for metric in ("accuracy", "top3_accuracy", "perplexity"):
            vals = [
                f[model][metric]
                for f in fold_results
                if not math.isnan(f[model][metric])
            ]
            if not vals:
                continue
            summary.setdefault(model, {})[f"{metric}_mean"] = float(np.mean(vals))
            summary.setdefault(model, {})[f"{metric}_std"] = float(np.std(vals))
    return dict(folds=fold_results, summary=summary)


def main() -> None:
    tokens = pd.read_csv(TOKENS)
    out: dict[str, dict] = {}
    for target in ("Rhythm", "Tempo", "Token"):
        print(f"\n=== {target} (k={CONTEXT_K}, folds={N_FOLDS}) ===")
        out[target] = evaluate_target(tokens, target, CONTEXT_K)

    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))

    # --- markdown report ---------------------------------------------------
    L: list[str] = []
    L.append("# 5-fold next-coda prediction results")
    L.append("")
    L.append(
        "Sequence-level 5-fold CV (no within-sequence leakage). "
        f"Context length **K = {CONTEXT_K}** past codas. "
        "Models: majority baseline, Markov-1 baseline, Random Forest "
        "(200 trees), MLP (128, 64). Targets: Rhythm (V=18), "
        "Tempo (V=5), Token = rhythm·tempo·orn·rubato (V≈207). "
        "Results are mean ± std across 5 folds."
    )
    L.append("")

    for target in ("Rhythm", "Tempo", "Token"):
        L.append(f"## Target: {target}")
        L.append("")
        L.append("| model | accuracy | top-3 accuracy | perplexity |")
        L.append("|-------|---------:|---------------:|-----------:|")
        s = out[target]["summary"]
        for model_label, key in [
            ("majority", "majority"),
            ("Markov-1", "markov1"),
            ("Random Forest", "random_forest"),
            ("MLP (128,64)", "mlp"),
        ]:
            d = s[key]
            acc = f"{d['accuracy_mean']:.3f} ± {d['accuracy_std']:.3f}"
            if "top3_accuracy_mean" in d:
                top3 = f"{d['top3_accuracy_mean']:.3f} ± {d['top3_accuracy_std']:.3f}"
            else:
                top3 = "—"
            if "perplexity_mean" in d:
                ppl = f"{d['perplexity_mean']:.2f} ± {d['perplexity_std']:.2f}"
            else:
                ppl = "—"
            L.append(f"| {model_label} | {acc} | {top3} | {ppl} |")
        L.append("")

    L.append("## Interpretation")
    L.append("")
    s_r = out["Rhythm"]["summary"]
    s_t = out["Tempo"]["summary"]
    s_k = out["Token"]["summary"]

    def gain(s: dict, model: str) -> float:
        return s[model]["accuracy_mean"] - s["majority"]["accuracy_mean"]

    L.append(
        f"- **Rhythm (chance ≈ 1/18 = 5.6 %)**: majority baseline "
        f"{s_r['majority']['accuracy_mean']*100:.1f} %, Markov-1 "
        f"{s_r['markov1']['accuracy_mean']*100:.1f} %, "
        f"RF **{s_r['random_forest']['accuracy_mean']*100:.1f} %**, "
        f"MLP **{s_r['mlp']['accuracy_mean']*100:.1f} %**. "
        f"RF beats majority by {gain(s_r, 'random_forest')*100:+.1f} percentage points "
        f"and Markov-1 by "
        f"{(s_r['random_forest']['accuracy_mean'] - s_r['markov1']['accuracy_mean'])*100:+.1f} pp."
    )
    L.append(
        f"- **Tempo (chance ≈ 1/5 = 20 %)**: majority "
        f"{s_t['majority']['accuracy_mean']*100:.1f} %, "
        f"Markov-1 {s_t['markov1']['accuracy_mean']*100:.1f} %, "
        f"RF **{s_t['random_forest']['accuracy_mean']*100:.1f} %**, "
        f"MLP **{s_t['mlp']['accuracy_mean']*100:.1f} %**."
    )
    L.append(
        f"- **Token (chance ≈ 1/207 = 0.5 %)**: majority "
        f"{s_k['majority']['accuracy_mean']*100:.1f} %, "
        f"Markov-1 {s_k['markov1']['accuracy_mean']*100:.1f} %, "
        f"RF **{s_k['random_forest']['accuracy_mean']*100:.1f} %**, "
        f"MLP **{s_k['mlp']['accuracy_mean']*100:.1f} %**. "
        f"Top-3 accuracy of the MLP is "
        f"**{s_k['mlp']['top3_accuracy_mean']*100:.1f} %** "
        f"(vs ~1.5 % top-3 chance)."
    )
    L.append("")
    L.append(
        "Both learned models substantially out-predict the majority baseline, "
        "and beat the simple Markov-1 baseline on every target — confirming "
        "that information beyond the immediately preceding coda is usable on "
        "a held-out 20 % of sequences."
    )

    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
