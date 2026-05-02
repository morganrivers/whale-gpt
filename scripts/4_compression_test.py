"""
Held-out compression tests:

1. gzip / lzma / bz2 the readable corpus vs shuffled controls. The hint is
   literal: better understanding ↔ better compression. We pick the simplest
   universal compressors, not models we tuned.

2. Held-out cross-entropy of order-k Markov models trained on N-1 sequences
   and tested on the held-out one. This rules out in-sample overfitting and
   matches how a real LM would be evaluated.

3. Per-feature held-out test: rhythm, tempo, etc. independently.

Outputs to outputs/grammar/compression.json and a short text report.
"""
from __future__ import annotations

import bz2
import gzip
import json
import lzma
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ROOT / "data" / "whale_tokens.csv"
CORPUS_TXT = ROOT / "data" / "whale_corpus.txt"
OUT_DIR = ROOT / "outputs" / "grammar"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- universal compressor experiment ---------------------------------

def compress_sizes(text: bytes) -> dict[str, int]:
    return dict(
        raw=len(text),
        gzip=len(gzip.compress(text, compresslevel=9)),
        bz2=len(bz2.compress(text, compresslevel=9)),
        lzma=len(lzma.compress(text, preset=9)),
    )


def shuffle_lines(text: str, seed: int) -> str:
    rng = random.Random(seed)
    lines = text.split("\n")
    rng.shuffle(lines)
    return "\n".join(lines)


def shuffle_tokens(text: str, seed: int) -> str:
    """Shuffle every whitespace-separated token; preserves alphabet only."""
    rng = random.Random(seed)
    tokens = text.split()
    rng.shuffle(tokens)
    return " ".join(tokens)


def universal_compression_test() -> dict:
    text = CORPUS_TXT.read_text()
    real = compress_sizes(text.encode())

    # destroy higher-order structure two ways
    line_shuffled = [
        compress_sizes(shuffle_lines(text, seed).encode()) for seed in range(5)
    ]
    token_shuffled = [
        compress_sizes(shuffle_tokens(text, seed).encode()) for seed in range(5)
    ]

    def stats(runs: list[dict[str, int]], k: str) -> dict:
        sizes = [r[k] for r in runs]
        return dict(mean=float(np.mean(sizes)), std=float(np.std(sizes)))

    return dict(
        real=real,
        line_shuffled={k: stats(line_shuffled, k) for k in real},
        token_shuffled={k: stats(token_shuffled, k) for k in real},
    )


# ---------- held-out Markov cross-entropy -----------------------------------

def held_out_ce(
    seqs: list[list[str]], context_len: int, smoothing: float = 0.5
) -> float:
    """Leave-one-sequence-out cross-entropy in bits/token."""
    total_bits = 0.0
    total_tokens = 0
    V = len({t for s in seqs for t in s})
    for held_idx in range(len(seqs)):
        train = [t for i, s in enumerate(seqs) if i != held_idx for t in s]
        test = seqs[held_idx]
        if len(test) <= context_len:
            continue
        if context_len == 0:
            counts = Counter(train)
            denom = len(train) + smoothing * V
            for tok in test:
                p = (counts[tok] + smoothing) / denom
                total_bits -= math.log2(p)
                total_tokens += 1
        else:
            ctx_counts: dict[tuple, Counter] = defaultdict(Counter)
            for i in range(context_len, len(train)):
                ctx_counts[tuple(train[i - context_len : i])][train[i]] += 1
            for i in range(context_len, len(test)):
                ctx = tuple(test[i - context_len : i])
                c = ctx_counts.get(ctx)
                if c is None:
                    # back off to unigram from training
                    counts = Counter(train)
                    p = (counts[test[i]] + smoothing) / (
                        len(train) + smoothing * V
                    )
                else:
                    p = (c[test[i]] + smoothing) / (sum(c.values()) + smoothing * V)
                total_bits -= math.log2(p)
                total_tokens += 1
    return total_bits / total_tokens if total_tokens else float("nan")


def held_out_test(tokens: pd.DataFrame) -> dict:
    out: dict[str, dict[str, float]] = {}
    for col in ("Token", "Rhythm", "Tempo", "Rubato", "Ornamentation"):
        seqs = []
        for _, g in tokens.groupby("sequenceId"):
            seqs.append(list(g.sort_values("itemPosition")[col].astype(str)))
        if col == "Token":
            # Subsample 60 sequences for speed; LOO is O(N^2 V)
            rng = random.Random(0)
            idx = rng.sample(range(len(seqs)), min(60, len(seqs)))
            seqs_sub = [seqs[i] for i in idx]
        else:
            seqs_sub = seqs
        d: dict[str, float] = {}
        for k in (0, 1, 2):
            d[f"H_{k}"] = held_out_ce(seqs_sub, k)
        # shuffled control at order 1
        rng = random.Random(7)
        seqs_sh = [list(s) for s in seqs_sub]
        for s in seqs_sh:
            rng.shuffle(s)
        d["H_1_shuffled"] = held_out_ce(seqs_sh, 1)
        d["compression_gain_vs_shuffled_bits"] = d["H_1_shuffled"] - d["H_1"]
        d["unigram_vs_order1_gain_bits"] = d["H_0"] - d["H_1"]
        out[col] = d
    return out


def main() -> None:
    print("running universal-compression test ...")
    uni = universal_compression_test()

    print("running held-out Markov CE ...")
    tokens = pd.read_csv(TOKENS)
    ho = held_out_test(tokens)

    out = dict(universal_compression=uni, held_out_markov=ho)
    (OUT_DIR / "compression.json").write_text(json.dumps(out, indent=2))

    # human-readable summary
    lines = ["=== Universal compressors on whale_corpus.txt ==="]
    raw = uni["real"]["raw"]
    for codec in ("gzip", "bz2", "lzma"):
        r = uni["real"][codec]
        ls = uni["line_shuffled"][codec]["mean"]
        ts = uni["token_shuffled"][codec]["mean"]
        lines.append(
            f"  {codec:6s}  real={r:>7d} bytes  ({r / raw * 100:5.1f}% of raw)  "
            f"line_shuffle={ls:>7.0f}  token_shuffle={ts:>7.0f}  "
            f"savings_vs_token_shuffle={(ts - r) / ts * 100:+.2f}%"
        )

    lines.append("")
    lines.append("=== Held-out Markov cross-entropy (bits/token) ===")
    for col, d in ho.items():
        lines.append(
            f"  {col:14s} H0={d['H_0']:.3f}  H1={d['H_1']:.3f}  H2={d['H_2']:.3f}  "
            f"H1_shuffled={d['H_1_shuffled']:.3f}  "
            f"H1_gain_vs_shuffle={d['compression_gain_vs_shuffled_bits']:.3f}  "
            f"H0_minus_H1={d['unigram_vs_order1_gain_bits']:.3f}"
        )

    (OUT_DIR / "compression_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
