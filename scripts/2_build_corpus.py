"""
Build a compact, human-readable token corpus from `whale-dialogue-script.csv`.

Each coda becomes a token of the form  R T O Ru  where
    R  = rhythm letter   a..r          (18 rhythm classes from Sharma et al.)
    T  = tempo digit     1..5          (5 tempo modes)
    O  = '.' or '*'                    ('*' = ornamented)
    Ru = '=', '/', '\\', '?'           (rubato direction; '?' if undefined)

Lines in the corpus look like:
    SEQ 0
    W1 i1.= i1.= b1.= B1.= ...
    CHO 1,2 f3.= d3.=
    PAUSE 25s
    ...

This format is used everywhere downstream so we always have a single source
of truth that is also readable as plain text.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_CSV = ROOT / "data" / "whale-dialogue-script.csv"
OUT_TXT = ROOT / "data" / "whale_corpus.txt"
OUT_TOKENS = ROOT / "data" / "whale_tokens.csv"

RHYTHM_LETTERS = "abcdefghijklmnopqr"  # 18 rhythms (matches Sharma)


def tempo_class(d: float) -> int:
    # cutoffs replicate scripts/1c_generate_readable_text.py
    if d < 0.45:
        return 0
    if d < 0.61:
        return 1
    if d < 0.93:
        return 2
    if d < 1.08:
        return 3
    return 4


# Rubato cutoffs replicate scripts/1c_generate_readable_text.py
RUBATO_LOW, RUBATO_HIGH = -0.02142, 0.01846


def rubato_dir(prev_dur: float, dur: float, same_rt: bool) -> str:
    if not same_rt:
        return "?"
    delta = dur - prev_dur
    if delta < RUBATO_LOW:
        return "\\"
    if delta < RUBATO_HIGH:
        return "="
    return "/"


def coda_token(rhythm: int, tempo: int, orn: int, rub: str) -> str:
    return f"{RHYTHM_LETTERS[rhythm]}{tempo + 1}{'*' if orn else '.'}{rub}"


def main() -> None:
    df = pd.read_csv(SCRIPT_CSV)

    # tempo + token per row
    df["Tempo"] = df["Duration"].apply(tempo_class)

    rows = []
    last_token_per_seq_per_whale: dict[tuple[int, int], dict] = {}
    for _, r in df.iterrows():
        seq = int(r.sequenceId)
        whale = int(r.Whale)
        rhythm = int(r.Coda)
        tempo = int(r.Tempo)
        orn = int(r.Ornamentation)
        dur = float(r.Duration)

        prev = last_token_per_seq_per_whale.get((seq, whale))
        if prev is None:
            rub = "?"
        else:
            same_rt = (prev["rhythm"] == rhythm) and (prev["tempo"] == tempo)
            rub = rubato_dir(prev["dur"], dur, same_rt)

        token = coda_token(rhythm, tempo, orn, rub)
        rows.append(
            dict(
                sequenceId=seq,
                itemPosition=int(r.itemPosition),
                Whale=whale,
                Rhythm=rhythm,
                Tempo=tempo,
                Ornamentation=orn,
                Synchrony=int(r.Synchrony),
                Duration=dur,
                TsToAbs=float(r.TsToAbs),
                Token=token,
                Rubato=rub,
            )
        )
        last_token_per_seq_per_whale[(seq, whale)] = dict(
            rhythm=rhythm, tempo=tempo, dur=dur
        )

    tokens = pd.DataFrame(rows)
    tokens.to_csv(OUT_TOKENS, index=False)

    # Also dump a readable text view (chorus grouping, pauses, sequence headers).
    with OUT_TXT.open("w") as f:
        for seq_id, seq in tokens.groupby("sequenceId", sort=True):
            seq = seq.sort_values(["itemPosition"]).reset_index(drop=True)
            f.write(f"SEQ {seq_id}\n")
            i = 0
            prev_t = None
            while i < len(seq):
                row = seq.iloc[i]
                if prev_t is not None:
                    gap = row.TsToAbs - prev_t
                    if gap > 10:
                        f.write(f"PAUSE {gap:.1f}s\n")
                # accumulate chorus block
                if row.Synchrony == 1:
                    block = [row]
                    j = i + 1
                    while j < len(seq) and seq.iloc[j].Synchrony == 1 and (
                        seq.iloc[j].TsToAbs - row.TsToAbs < 2.0
                    ):
                        block.append(seq.iloc[j])
                        j += 1
                    whales = sorted({int(b.Whale) for b in block})
                    toks = " ".join(b.Token for b in block)
                    f.write(f"CHO {','.join(map(str, whales))} {toks}\n")
                    prev_t = block[-1].TsToAbs
                    i = j
                    continue
                # accumulate single-whale run
                same_whale = [row]
                j = i + 1
                while (
                    j < len(seq)
                    and seq.iloc[j].Synchrony == 0
                    and seq.iloc[j].Whale == row.Whale
                    and (seq.iloc[j].TsToAbs - same_whale[-1].TsToAbs) < 10
                ):
                    same_whale.append(seq.iloc[j])
                    j += 1
                toks = " ".join(b.Token for b in same_whale)
                f.write(f"W{int(row.Whale)} {toks}\n")
                prev_t = same_whale[-1].TsToAbs
                i = j
            f.write("\n")

    # Tiny stats summary printed to stdout.
    n_codas = len(tokens)
    n_seq = tokens.sequenceId.nunique()
    n_unique = tokens.Token.nunique()
    print(f"wrote {OUT_TOKENS} ({n_codas} codas, {n_seq} sequences, {n_unique} unique tokens)")
    print(f"wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
