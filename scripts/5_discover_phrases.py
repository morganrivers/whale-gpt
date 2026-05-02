"""
Discover candidate "phrases" / chunks in the whale corpus and check whether
the language exhibits *duality of patterning* (the same rhythm sequence
recurring across multiple tempo realizations and vice versa).

Concrete findings produced:

  A. The 30 strongest multi-coda phrases (by significance over independence
     ratio AND by how often each one is followed deterministically).
  B. Cross-feature factorization: do rhythm-bigrams and tempo-bigrams pair
     independently? (a strong hint of "phonemes" vs "tempo lines")
  C. A finite-state machine view: top-3 successors per (rhythm, tempo) state
     with their probabilities; states whose successor distribution is nearly
     deterministic.
  D. Long-range information signature: MI(rhythm[t], rhythm[t+k]) for k up
     to 30, compared to a 1st-order Markov sample (which would give MI≈0
     past lag 1 if the corpus were truly Markov-1).
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


def mutual_information(xs: list, ys: list) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    pxy = Counter(zip(xs, ys))
    px = Counter(xs)
    py = Counter(ys)
    mi = 0.0
    for (x, y), c in pxy.items():
        p_xy = c / n
        mi += p_xy * math.log2(p_xy / ((px[x] / n) * (py[y] / n)))
    return mi


def shannon_entropy(xs: list) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    counts = Counter(xs)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def per_seq(tokens: pd.DataFrame, col: str) -> list[list[str]]:
    return [
        list(g.sort_values("itemPosition")[col].astype(str))
        for _, g in tokens.groupby("sequenceId", sort=True)
    ]


# ---------- A. phrase discovery -------------------------------------------

def discover_phrases(
    seqs: list[list[str]], n: int, min_count: int = 5, top: int = 25
) -> list[dict]:
    """Find n-grams that strongly violate the unigram independence hypothesis
    AND are highly *predictive* of their last token (P(last | prefix) is
    concentrated)."""
    flat = [t for s in seqs for t in s]
    unigram = Counter(flat)
    n_total = len(flat)

    grams = Counter()
    prefix_to_last = defaultdict(Counter)
    for s in seqs:
        for i in range(len(s) - n + 1):
            ng = tuple(s[i : i + n])
            grams[ng] += 1
            prefix_to_last[ng[:-1]][ng[-1]] += 1

    def expected(ng: tuple) -> float:
        p = 1.0
        for tok in ng:
            p *= unigram[tok] / n_total
        return p * (n_total - n + 1)

    rows = []
    for ng, c in grams.items():
        if c < min_count:
            continue
        e = expected(ng)
        if e == 0:
            continue
        prefix = ng[:-1]
        prefix_total = sum(prefix_to_last[prefix].values())
        # how much of the prefix's continuations does this n-gram cover?
        determinism = c / prefix_total if prefix_total else 0.0
        llr = 2 * c * math.log(c / e)
        rows.append(
            dict(
                ngram=" ".join(ng),
                count=c,
                expected=round(e, 2),
                ratio=round(c / e, 2),
                llr=round(llr, 1),
                p_last_given_prefix=round(determinism, 3),
                prefix_total=prefix_total,
            )
        )
    rows.sort(key=lambda r: -r["llr"])
    return rows[:top]


# ---------- B. duality of patterning --------------------------------------

def duality_of_patterning(tokens: pd.DataFrame) -> dict:
    """Do rhythm-sequences recur across multiple tempo profiles?

    For every rhythm bigram (R_t, R_{t+1}) we collect the multiset of
    realized tempo bigrams. If the tempo distribution is *broad* (high
    entropy) for many rhythm bigrams, that's evidence the rhythm 'word'
    is independently meaningful from the tempo 'tone' (a key signature
    of duality of patterning).

    We report:
      - mean entropy of tempo-bigrams given rhythm-bigram (high = decoupled)
      - mean entropy of rhythm-bigrams given tempo-bigram
      - the count of rhythm bigrams that appear with >= 2 distinct tempos
      - several illustrative examples
    """
    pairs = []
    for _, g in tokens.groupby("sequenceId"):
        g = g.sort_values("itemPosition").reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            pairs.append(((int(a.Rhythm), int(b.Rhythm)), (int(a.Tempo), int(b.Tempo))))

    # group tempo bigrams by rhythm bigram
    rhythm_to_tempos: dict[tuple, list[tuple]] = defaultdict(list)
    tempo_to_rhythms: dict[tuple, list[tuple]] = defaultdict(list)
    for r, t in pairs:
        rhythm_to_tempos[r].append(t)
        tempo_to_rhythms[t].append(r)

    examples_rhythm = []
    h_t_given_r = []
    for r, ts in rhythm_to_tempos.items():
        if len(ts) < 5:
            continue
        h = shannon_entropy(ts)
        h_t_given_r.append((r, len(ts), h, len(set(ts))))
    h_t_given_r.sort(key=lambda x: -x[2])
    for r, n, h, distinct in h_t_given_r[:10]:
        examples_rhythm.append(
            dict(
                rhythm_bigram=str(r),
                count=n,
                tempo_distinct_realizations=distinct,
                tempo_entropy_bits=round(h, 3),
                top_realizations=Counter(rhythm_to_tempos[r]).most_common(5),
            )
        )

    examples_tempo = []
    h_r_given_t = []
    for t, rs in tempo_to_rhythms.items():
        if len(rs) < 5:
            continue
        h = shannon_entropy(rs)
        h_r_given_t.append((t, len(rs), h, len(set(rs))))
    h_r_given_t.sort(key=lambda x: -x[2])
    for t, n, h, distinct in h_r_given_t[:10]:
        examples_tempo.append(
            dict(
                tempo_bigram=str(t),
                count=n,
                rhythm_distinct_realizations=distinct,
                rhythm_entropy_bits=round(h, 3),
                top_realizations=Counter(tempo_to_rhythms[t]).most_common(5),
            )
        )

    avg_h_t_given_r = float(np.mean([h for _, _, h, _ in h_t_given_r])) if h_t_given_r else float("nan")
    avg_h_r_given_t = float(np.mean([h for _, _, h, _ in h_r_given_t])) if h_r_given_t else float("nan")

    return dict(
        n_rhythm_bigrams_with_count_ge_5=len(h_t_given_r),
        avg_tempo_entropy_given_rhythm_bits=round(avg_h_t_given_r, 3),
        n_tempo_bigrams_with_count_ge_5=len(h_r_given_t),
        avg_rhythm_entropy_given_tempo_bits=round(avg_h_r_given_t, 3),
        rhythm_bigram_with_most_diverse_tempos=examples_rhythm,
        tempo_bigram_with_most_diverse_rhythms=examples_tempo,
    )


# ---------- C. finite-state-machine view ----------------------------------

def fsa_view(tokens: pd.DataFrame, top_per_state: int = 3) -> dict:
    """Top successors per (rhythm, tempo) state. Highlight near-deterministic
    transitions (P(top successor) > 0.7)."""
    transitions: dict[tuple, Counter] = defaultdict(Counter)
    for _, g in tokens.groupby("sequenceId"):
        g = g.sort_values("itemPosition").reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            transitions[(int(a.Rhythm), int(a.Tempo))][(int(b.Rhythm), int(b.Tempo))] += 1

    rows = []
    for state, succs in transitions.items():
        n = sum(succs.values())
        if n < 10:
            continue
        top = succs.most_common(top_per_state)
        p_top = top[0][1] / n
        rows.append(
            dict(
                state=str(state),
                n=n,
                top_successors=[(str(k), v, round(v / n, 3)) for k, v in top],
                p_top=round(p_top, 3),
                successor_entropy_bits=round(
                    -sum((c / n) * math.log2(c / n) for c in succs.values()), 3
                ),
            )
        )
    rows.sort(key=lambda r: -r["p_top"])
    return dict(
        n_states=len(rows),
        most_deterministic=rows[:20],
        most_uncertain=sorted(rows, key=lambda r: -r["successor_entropy_bits"])[:20],
    )


# ---------- D. long-range MI signature ------------------------------------

def long_range_mi(seqs: list[list[str]], max_lag: int = 30, n_simulated: int = 5) -> dict:
    """Compare rhythm MI(t, t+k) to a *simulated* order-1 Markov chain
    sharing the same first-order statistics. If real MI exceeds the Markov
    sample MI past lag 1, we have higher-order grammar."""

    def mi_curve(seqs_: list[list[str]]) -> dict[int, float]:
        out = {}
        for k in range(1, max_lag + 1):
            xs, ys = [], []
            for s in seqs_:
                if len(s) <= k:
                    continue
                xs.extend(s[:-k])
                ys.extend(s[k:])
            out[k] = mutual_information(xs, ys)
        return out

    real = mi_curve(seqs)

    # simulate a 1st-order Markov chain with the same transition matrix
    flat = [t for s in seqs for t in s]
    starts = Counter(s[0] for s in seqs if s)
    trans: dict[str, Counter] = defaultdict(Counter)
    for s in seqs:
        for i in range(len(s) - 1):
            trans[s[i]][s[i + 1]] += 1

    rng = random.Random(42)
    simulated_curves = []
    for _ in range(n_simulated):
        sim_seqs = []
        for s in seqs:
            if not s:
                continue
            seq = [random.choices(list(starts), weights=list(starts.values()))[0]]
            while len(seq) < len(s):
                cur = seq[-1]
                nxt_dist = trans.get(cur)
                if not nxt_dist:
                    seq.append(rng.choice(flat))
                else:
                    seq.append(
                        random.choices(list(nxt_dist), weights=list(nxt_dist.values()))[0]
                    )
            sim_seqs.append(seq)
        simulated_curves.append(mi_curve(sim_seqs))

    sim_mean = {k: float(np.mean([c[k] for c in simulated_curves])) for k in real}
    sim_std = {k: float(np.std([c[k] for c in simulated_curves])) for k in real}
    return dict(
        real={k: round(v, 4) for k, v in real.items()},
        markov1_simulated_mean={k: round(v, 4) for k, v in sim_mean.items()},
        markov1_simulated_std={k: round(v, 4) for k, v in sim_std.items()},
        excess_over_markov1={k: round(real[k] - sim_mean[k], 4) for k in real},
    )


def main() -> None:
    tokens = pd.read_csv(TOKENS)
    rhythm_seqs = per_seq(tokens, "Rhythm")
    rhythm_tempo_seqs = []
    for _, g in tokens.groupby("sequenceId"):
        g = g.sort_values("itemPosition")
        rhythm_tempo_seqs.append([f"{r.Rhythm}.{r.Tempo}" for _, r in g.iterrows()])

    print("discovering phrases ...")
    phrases = {
        f"n{n}": discover_phrases(rhythm_tempo_seqs, n) for n in (3, 4, 5)
    }

    print("checking duality of patterning ...")
    dop = duality_of_patterning(tokens)

    print("building FSA view ...")
    fsa = fsa_view(tokens)

    print("computing long-range MI signature ...")
    lrmi = long_range_mi(rhythm_seqs, max_lag=30, n_simulated=5)

    out = dict(phrases=phrases, duality_of_patterning=dop, fsa=fsa, long_range_mi=lrmi)
    (OUT_DIR / "discovery.json").write_text(json.dumps(out, indent=2, default=str))

    # human-readable
    L = []
    L.append("=== Strong recurring phrases (rhythm.tempo, n>=3) ===")
    for k in ("n3", "n4", "n5"):
        L.append(f"  -- {k} --")
        for p in phrases[k][:10]:
            L.append(
                f"    {p['ngram']:30s}  count={p['count']:>4d}  "
                f"ratio={p['ratio']:>6.2f}  P(last|prefix)={p['p_last_given_prefix']:.2f}"
            )

    L.append("")
    L.append("=== Duality of patterning ===")
    L.append(
        f"  rhythm bigrams w/ count>=5: {dop['n_rhythm_bigrams_with_count_ge_5']}; "
        f"avg H(tempo|rhythm) = {dop['avg_tempo_entropy_given_rhythm_bits']} bits"
    )
    L.append(
        f"  tempo bigrams  w/ count>=5: {dop['n_tempo_bigrams_with_count_ge_5']}; "
        f"avg H(rhythm|tempo) = {dop['avg_rhythm_entropy_given_tempo_bits']} bits"
    )
    L.append("  rhythm bigrams realized with the most distinct tempo profiles:")
    for ex in dop["rhythm_bigram_with_most_diverse_tempos"][:5]:
        L.append(
            f"    R={ex['rhythm_bigram']:>10s}  n={ex['count']:>4d}  "
            f"distinct_tempo_pairs={ex['tempo_distinct_realizations']:>3d}  "
            f"H={ex['tempo_entropy_bits']:.2f}  "
            f"top={ex['top_realizations'][:3]}"
        )

    L.append("")
    L.append("=== FSA view: most deterministic (rhythm, tempo) transitions ===")
    for r in fsa["most_deterministic"][:15]:
        succs = ", ".join(f"{s}:{p:.2f}" for s, _, p in r["top_successors"])
        L.append(
            f"  state {r['state']:>10s}  n={r['n']:>4d}  H_succ={r['successor_entropy_bits']:.2f}  "
            f"top: {succs}"
        )

    L.append("")
    L.append("=== Long-range rhythm MI: real vs simulated 1st-order Markov ===")
    L.append("  k    real   markov1   excess")
    for k in (1, 2, 3, 5, 8, 12, 16, 20, 25, 30):
        if k in lrmi["real"]:
            L.append(
                f"  {k:>3d}  {lrmi['real'][k]:.3f}  {lrmi['markov1_simulated_mean'][k]:.3f}    "
                f"{lrmi['excess_over_markov1'][k]:+.3f}"
            )

    (OUT_DIR / "discovery_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
