"""
Compression / structure tests on the whale corpus.

Hypothesis: if whale calls have grammar, structured n-gram models compress
the corpus better than shuffled controls AND better than a unigram model.
We measure cross-entropy in bits/token, mutual information per feature at
lag k, n-gram surprisal, and turn-taking conditional structure.

Outputs go to outputs/grammar/.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ROOT / "data" / "whale_tokens.csv"
OUT_DIR = ROOT / "outputs" / "grammar"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = random.Random(0)


# ---------- helpers ----------------------------------------------------------

def cross_entropy(stream: list, context_len: int, smoothing: float = 0.5) -> float:
    """Bits/token for an order-`context_len` Markov model with add-`smoothing`.

    Trained AND scored on `stream` itself (in-sample) so the comparison
    against shuffled streams measures "how much can structure be exploited".
    """
    if context_len == 0:
        counts = Counter(stream)
        V = len(counts)
        N = len(stream)
        denom = N + smoothing * V
        bits = 0.0
        for tok in stream:
            p = (counts[tok] + smoothing) / denom
            bits -= math.log2(p)
        return bits / N

    # higher-order: P(x_t | x_{t-k:t})
    ctx_counts: dict[tuple, Counter] = defaultdict(Counter)
    for i in range(context_len, len(stream)):
        ctx = tuple(stream[i - context_len : i])
        ctx_counts[ctx][stream[i]] += 1

    V = len(set(stream))
    bits = 0.0
    for i in range(context_len, len(stream)):
        ctx = tuple(stream[i - context_len : i])
        c = ctx_counts[ctx]
        denom = sum(c.values()) + smoothing * V
        p = (c[stream[i]] + smoothing) / denom
        bits -= math.log2(p)
    return bits / (len(stream) - context_len)


def shuffled(stream: list, seed: int = 0) -> list:
    rng = random.Random(seed)
    s = list(stream)
    rng.shuffle(s)
    return s


def mutual_information(xs: list, ys: list) -> float:
    """MI(X;Y) in bits, MLE estimator with no smoothing."""
    assert len(xs) == len(ys)
    n = len(xs)
    if n == 0:
        return 0.0
    pxy = Counter(zip(xs, ys))
    px = Counter(xs)
    py = Counter(ys)
    mi = 0.0
    for (x, y), c in pxy.items():
        p_xy = c / n
        p_x = px[x] / n
        p_y = py[y] / n
        mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return mi


def shannon_entropy(xs: list) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    counts = Counter(xs)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


# ---------- experiments -----------------------------------------------------

def experiment_compression(tokens: pd.DataFrame) -> dict:
    """Bits/token for n-gram models on full token stream and individual features.

    We chunk by sequence so context never crosses dialogue boundaries.
    Bits are reported macro-averaged over sequences (weighted by length).
    """

    def stream_per_seq(col: str) -> list[list]:
        out = []
        for _, g in tokens.groupby("sequenceId", sort=True):
            out.append(list(g.sort_values("itemPosition")[col].astype(str)))
        return out

    feats = {
        "Token": stream_per_seq("Token"),
        "Rhythm": stream_per_seq("Rhythm"),
        "Tempo": stream_per_seq("Tempo"),
        "Ornamentation": stream_per_seq("Ornamentation"),
        "Rubato": stream_per_seq("Rubato"),
    }

    results: dict[str, dict] = {}
    for fname, seqs in feats.items():
        flat = [t for s in seqs for t in s]
        baseline = cross_entropy(flat, 0)
        per_order: dict[str, float] = {"order_0": baseline}
        for k in (1, 2, 3):
            per_seq_bits = []
            for s in seqs:
                if len(s) > k:
                    per_seq_bits.append(
                        (len(s) - k, cross_entropy(s, k, smoothing=0.5))
                    )
            total_w = sum(w for w, _ in per_seq_bits)
            avg = sum(w * b for w, b in per_seq_bits) / total_w if total_w else float("nan")
            per_order[f"order_{k}"] = avg

        # null distribution: shuffle within each sequence
        shuffled_bits = []
        for trial in range(20):
            shuffled_seqs = [shuffled(s, seed=trial * 100 + hash(fname) % 1000) for s in seqs]
            per_seq_bits = []
            for s in shuffled_seqs:
                if len(s) > 1:
                    per_seq_bits.append((len(s) - 1, cross_entropy(s, 1, smoothing=0.5)))
            total_w = sum(w for w, _ in per_seq_bits)
            avg = sum(w * b for w, b in per_seq_bits) / total_w if total_w else float("nan")
            shuffled_bits.append(avg)

        per_order["order_1_shuffled_mean"] = float(np.mean(shuffled_bits))
        per_order["order_1_shuffled_std"] = float(np.std(shuffled_bits))
        per_order["compression_gain_bits"] = per_order["order_0"] - per_order["order_1"]
        per_order["beats_shuffled_by_bits"] = per_order["order_1_shuffled_mean"] - per_order["order_1"]
        per_order["alphabet_size"] = len({t for s in seqs for t in s})
        per_order["num_tokens"] = sum(len(s) for s in seqs)
        results[fname] = per_order

    return results


def experiment_mi_lag(tokens: pd.DataFrame, max_lag: int = 10) -> dict:
    """MI between feature[t] and feature[t+k] within each sequence.

    Result tells us how far into the past a feature carries information.
    """
    feats = ["Rhythm", "Tempo", "Ornamentation", "Rubato"]
    out: dict[str, dict] = {}
    for f in feats:
        per_lag: dict[int, float] = {}
        for k in range(1, max_lag + 1):
            xs, ys = [], []
            for _, g in tokens.groupby("sequenceId"):
                arr = list(g.sort_values("itemPosition")[f].astype(str))
                if len(arr) <= k:
                    continue
                xs.extend(arr[:-k])
                ys.extend(arr[k:])
            per_lag[k] = mutual_information(xs, ys)
        out[f] = per_lag

        # shuffled control at lag 1
        bits = []
        for trial in range(50):
            xs_, ys_ = [], []
            for _, g in tokens.groupby("sequenceId"):
                arr = list(g.sort_values("itemPosition")[f].astype(str))
                if len(arr) <= 1:
                    continue
                arr2 = shuffled(arr, seed=trial * 7 + hash(f) % 1000)
                xs_.extend(arr2[:-1])
                ys_.extend(arr2[1:])
            bits.append(mutual_information(xs_, ys_))
        out[f]["shuffled_lag1_mean"] = float(np.mean(bits))
        out[f]["shuffled_lag1_std"] = float(np.std(bits))
    return out


def experiment_turn_taking(tokens: pd.DataFrame) -> dict:
    """Conditional structure: given a coda from whale W, can we predict the
    immediately-following coda from a *different* whale?

    We measure H(R_resp | R_init) - H(R_resp) and the same for tempo.
    A negative number means the responder is structurally constrained by
    the initiator.
    """
    pairs_rhythm: list[tuple[str, str]] = []
    pairs_tempo: list[tuple[str, str]] = []
    pairs_token: list[tuple[str, str]] = []
    for _, g in tokens.groupby("sequenceId"):
        g = g.sort_values("itemPosition").reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            if a.Whale == b.Whale:
                continue
            # they must be temporally adjacent (skip across long pauses)
            if (b.TsToAbs - a.TsToAbs) > 5.0:
                continue
            pairs_rhythm.append((str(a.Rhythm), str(b.Rhythm)))
            pairs_tempo.append((str(a.Tempo), str(b.Tempo)))
            pairs_token.append((a.Token, b.Token))

    def conditional_entropy(pairs: list[tuple[str, str]]) -> tuple[float, float, float]:
        if not pairs:
            return float("nan"), float("nan"), float("nan")
        x, y = zip(*pairs)
        h_y = shannon_entropy(list(y))
        h_yx = h_y - mutual_information(list(x), list(y))
        # baseline: shuffle x against y
        ctrl = []
        for trial in range(50):
            x2 = shuffled(list(x), seed=trial)
            ctrl.append(shannon_entropy(list(y)) - mutual_information(x2, list(y)))
        return h_y, h_yx, float(np.mean(ctrl))

    h_r, hc_r, ctrl_r = conditional_entropy(pairs_rhythm)
    h_t, hc_t, ctrl_t = conditional_entropy(pairs_tempo)
    h_k, hc_k, ctrl_k = conditional_entropy(pairs_token)
    return dict(
        n_response_pairs=len(pairs_rhythm),
        rhythm=dict(H_resp=h_r, H_resp_given_init=hc_r, shuffled_baseline=ctrl_r),
        tempo=dict(H_resp=h_t, H_resp_given_init=hc_t, shuffled_baseline=ctrl_t),
        token=dict(H_resp=h_k, H_resp_given_init=hc_k, shuffled_baseline=ctrl_k),
    )


def experiment_ngrams(tokens: pd.DataFrame, n: int = 3, top_k: int = 30) -> dict:
    """Find n-grams that occur far more often than independent product of
    unigram probabilities would predict (PMI-style score)."""
    seqs = []
    for _, g in tokens.groupby("sequenceId"):
        # collapse tokens to (Rhythm, Tempo) ignoring rubato/ornament so
        # we focus on the categorical syllable backbone
        seqs.append(
            list(g.sort_values("itemPosition").apply(
                lambda r: f"{r.Rhythm}.{r.Tempo}", axis=1
            ))
        )
    flat = [t for s in seqs for t in s]
    unigram = Counter(flat)
    n_total = len(flat)

    ngram_counts: Counter = Counter()
    for s in seqs:
        for i in range(len(s) - n + 1):
            ngram_counts[tuple(s[i : i + n])] += 1

    def expected(ng: tuple) -> float:
        p = 1.0
        for tok in ng:
            p *= unigram[tok] / n_total
        return p * (n_total - n + 1)

    rows = []
    for ng, c in ngram_counts.items():
        if c < 5:
            continue
        e = expected(ng)
        if e == 0:
            continue
        ratio = c / e
        # log-likelihood-ratio approximation: 2*c*log(c/e)
        llr = 2 * c * math.log(c / e)
        rows.append((ng, c, e, ratio, llr))
    rows.sort(key=lambda r: -r[4])

    return dict(
        n=n,
        total_ngrams=len(ngram_counts),
        top=[
            dict(ngram=" ".join(ng), count=c, expected=e, ratio=ratio, llr=llr)
            for ng, c, e, ratio, llr in rows[:top_k]
        ],
    )


def experiment_boundaries(tokens: pd.DataFrame) -> dict:
    """Which (rhythm, tempo) tokens are over-represented at the *first*
    or *last* position of an exchange (a contiguous block by one whale
    bounded by pauses > 5s or by speaker change)?"""
    starts: Counter = Counter()
    ends: Counter = Counter()
    middles: Counter = Counter()
    for _, g in tokens.groupby("sequenceId"):
        g = g.sort_values("itemPosition").reset_index(drop=True)
        # break into runs by whale and 5s pauses
        runs: list[list[int]] = []
        cur: list[int] = []
        prev = None
        for i, r in g.iterrows():
            if prev is None:
                cur.append(i)
            else:
                p = g.iloc[prev]
                if r.Whale != p.Whale or (r.TsToAbs - p.TsToAbs) > 5.0:
                    if cur:
                        runs.append(cur)
                    cur = [i]
                else:
                    cur.append(i)
            prev = i
        if cur:
            runs.append(cur)
        for run in runs:
            if len(run) < 2:
                continue
            starts[f"{g.iloc[run[0]].Rhythm}.{g.iloc[run[0]].Tempo}"] += 1
            ends[f"{g.iloc[run[-1]].Rhythm}.{g.iloc[run[-1]].Tempo}"] += 1
            for i in run[1:-1]:
                middles[f"{g.iloc[i].Rhythm}.{g.iloc[i].Tempo}"] += 1

    def normalize(c: Counter) -> dict[str, float]:
        n = sum(c.values()) or 1
        return {k: v / n for k, v in c.items()}

    s, e, m = normalize(starts), normalize(ends), normalize(middles)
    keys = set(s) | set(e) | set(m)
    rows = []
    for k in keys:
        sp, ep, mp = s.get(k, 0.0), e.get(k, 0.0), m.get(k, 0.0)
        rows.append(dict(
            token=k,
            p_start=sp, p_middle=mp, p_end=ep,
            start_over_middle=(sp / mp) if mp > 0 else float("inf"),
            end_over_middle=(ep / mp) if mp > 0 else float("inf"),
            n_start=starts[k], n_middle=middles[k], n_end=ends[k],
        ))
    rows.sort(key=lambda r: -r["start_over_middle"] if r["n_start"] >= 5 else 0.0)
    return dict(
        n_runs=sum(starts.values()),
        starts_top=[r for r in rows if r["n_start"] >= 5][:20],
        rows_by_end=sorted(
            [r for r in rows if r["n_end"] >= 5],
            key=lambda r: -r["end_over_middle"],
        )[:20],
    )


def experiment_rubato_coupling(tokens: pd.DataFrame) -> dict:
    """Does the rubato direction of whale B at time t+1 depend on the
    rubato direction of whale A at time t (when they overlap or alternate
    within a few seconds)?"""
    pairs = []
    for _, g in tokens.groupby("sequenceId"):
        g = g.sort_values("itemPosition").reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            if a.Whale == b.Whale:
                continue
            if (b.TsToAbs - a.TsToAbs) > 5.0:
                continue
            if a.Rubato == "?" or b.Rubato == "?":
                continue
            pairs.append((a.Rubato, b.Rubato))
    if not pairs:
        return dict(error="no usable pairs")
    x, y = zip(*pairs)
    h_y = shannon_entropy(list(y))
    mi = mutual_information(list(x), list(y))
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a, b in pairs:
        table[a][b] += 1
    ctrl = []
    for trial in range(200):
        x2 = shuffled(list(x), seed=trial)
        ctrl.append(mutual_information(x2, list(y)))
    return dict(
        n_pairs=len(pairs),
        H_response=h_y,
        MI=mi,
        MI_shuffled_mean=float(np.mean(ctrl)),
        MI_shuffled_std=float(np.std(ctrl)),
        contingency={a: dict(t) for a, t in table.items()},
    )


def main() -> None:
    tokens = pd.read_csv(TOKENS)

    out: dict = {}
    out["compression"] = experiment_compression(tokens)
    out["mi_by_lag"] = experiment_mi_lag(tokens)
    out["turn_taking"] = experiment_turn_taking(tokens)
    out["ngrams_n2"] = experiment_ngrams(tokens, n=2)
    out["ngrams_n3"] = experiment_ngrams(tokens, n=3)
    out["ngrams_n4"] = experiment_ngrams(tokens, n=4)
    out["boundaries"] = experiment_boundaries(tokens)
    out["rubato_coupling"] = experiment_rubato_coupling(tokens)

    out_json = OUT_DIR / "grammar_results.json"
    with out_json.open("w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote {out_json}")

    # tiny human-readable summary
    summary = OUT_DIR / "summary.txt"
    with summary.open("w") as f:
        c = out["compression"]
        f.write("=== Compression (bits/token; lower = more structure) ===\n")
        for feat, r in c.items():
            f.write(
                f"  {feat:14s}  V={r['alphabet_size']:>4d}  N={r['num_tokens']:>5d}  "
                f"H0={r['order_0']:.3f}  H1={r['order_1']:.3f}  "
                f"H2={r['order_2']:.3f}  H3={r['order_3']:.3f}  "
                f"H1_shuffled={r['order_1_shuffled_mean']:.3f}  "
                f"gain_vs_shuffle={r['beats_shuffled_by_bits']:.3f}\n"
            )

        f.write("\n=== Mutual information at lag k (bits) ===\n")
        for feat, r in out["mi_by_lag"].items():
            row = "  ".join(f"k={k}:{r[k]:.3f}" for k in range(1, 11))
            f.write(f"  {feat:14s}  {row}  shuffled_lag1={r['shuffled_lag1_mean']:.3f}\n")

        f.write("\n=== Turn-taking (responder predictability) ===\n")
        tt = out["turn_taking"]
        f.write(f"  n_pairs = {tt['n_response_pairs']}\n")
        for feat in ("rhythm", "tempo", "token"):
            d = tt[feat]
            f.write(
                f"  {feat:8s}  H(resp)={d['H_resp']:.3f}  "
                f"H(resp|init)={d['H_resp_given_init']:.3f}  "
                f"shuffled={d['shuffled_baseline']:.3f}\n"
            )

        f.write("\n=== Top n-grams by log-likelihood-ratio over independence ===\n")
        for k in (2, 3, 4):
            f.write(f"  -- {k}-grams --\n")
            for r in out[f"ngrams_n{k}"]["top"][:15]:
                f.write(
                    f"    {r['ngram']:30s}  count={r['count']:>4d}  "
                    f"expected={r['expected']:>6.2f}  ratio={r['ratio']:>6.2f}  llr={r['llr']:>7.1f}\n"
                )

        f.write("\n=== Tokens over-represented at run START (over middle) ===\n")
        for r in out["boundaries"]["starts_top"][:15]:
            f.write(
                f"  {r['token']:8s}  n_start={r['n_start']:>3d}  "
                f"p_start/p_mid={r['start_over_middle']:.2f}\n"
            )

        f.write("\n=== Tokens over-represented at run END (over middle) ===\n")
        for r in out["boundaries"]["rows_by_end"][:15]:
            f.write(
                f"  {r['token']:8s}  n_end={r['n_end']:>3d}  "
                f"p_end/p_mid={r['end_over_middle']:.2f}\n"
            )

        f.write("\n=== Cross-whale rubato coupling ===\n")
        rc = out["rubato_coupling"]
        f.write(
            f"  n_pairs={rc['n_pairs']}  H(resp)={rc['H_response']:.3f}  "
            f"MI={rc['MI']:.4f}  "
            f"MI_shuffled={rc['MI_shuffled_mean']:.4f} +/- {rc['MI_shuffled_std']:.4f}\n"
        )
        f.write("  contingency (rows = initiator rubato, cols = responder):\n")
        cats = ("=", "/", "\\")
        f.write(f"            {' '.join(f'{c:>5s}' for c in cats)}\n")
        for a in cats:
            row = rc["contingency"].get(a, {})
            f.write(f"    {a:>3s}    {' '.join(f'{row.get(c, 0):>5d}' for c in cats)}\n")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
