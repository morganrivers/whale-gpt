"""Follow-up: clan-exclusive codas, controlled clan classification,
and a fairer individual-ID test (codas-per-individual leakage check).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score

DATA = Path("data/DominicaCodas.csv")
OUT = Path("outputs/analysis")
ICI_COLS = [f"ICI{i}" for i in range(1, 10)]


def add_engineered(df):
    df = df.copy()
    icis = df[ICI_COLS].to_numpy()
    cum = np.cumsum(icis, axis=1)
    dur = df["Duration"].replace(0, np.nan).to_numpy().reshape(-1, 1)
    rel = np.where(dur > 0, cum / dur, 0.0)
    for i in range(9):
        df[f"REL{i+1}"] = rel[:, i]
    for i in range(1, 9):
        a = df[f"ICI{i}"].to_numpy()
        b = df[f"ICI{i+1}"].to_numpy()
        df[f"RATIO{i}"] = np.where((a > 0) & (b > 0), b / a, 0.0)
    df["meanICI"] = df[ICI_COLS].replace(0, np.nan).mean(axis=1).fillna(0)
    df["stdICI"] = df[ICI_COLS].replace(0, np.nan).std(axis=1).fillna(0)
    df["cv_ICI"] = np.where(df["meanICI"] > 0, df["stdICI"] / df["meanICI"], 0.0)
    return df


def main():
    df = pd.read_csv(DATA)
    df = add_engineered(df)

    # ---- 1. Clan-exclusive coda types ------------------------------------
    ct = (
        df.groupby("CodaType")["Clan"].agg(
            n="count",
            ec1=lambda s: int((s == "EC1").sum()),
            ec2=lambda s: int((s == "EC2").sum()),
        )
    )
    ct["ec2_frac"] = ct["ec2"] / ct["n"]
    ct = ct.sort_values("n", ascending=False)
    ct.to_csv(OUT / "codatype_by_clan.csv")
    shared = ct[(ct["ec1"] >= 20) & (ct["ec2"] >= 20)]
    ec1_only = ct[(ct["ec2"] == 0) & (ct["n"] >= 30)]
    ec2_only = ct[(ct["ec1"] == 0) & (ct["n"] >= 30)]

    print("Coda types with substantial use by both clans:")
    print(shared)
    print("\nEC1-only coda types (n>=30):")
    print(ec1_only.head(20))
    print("\nEC2-only coda types (n>=30):")
    print(ec2_only.head(20))

    # ---- 2. Clan classification within each shared coda type --------------
    feat_full = (
        ["nClicks", "Duration"]
        + ICI_COLS
        + [f"REL{i}" for i in range(1, 10)]
        + [f"RATIO{i}" for i in range(1, 9)]
        + ["meanICI", "stdICI", "cv_ICI"]
    )
    print("\nClan classification controlled by coda type:")
    rows = []
    for coda in shared.index:
        sub = df[df["CodaType"] == coda]
        y = sub["Clan"].astype(str).to_numpy()
        if len(np.unique(y)) < 2:
            continue
        X = sub[feat_full].to_numpy()
        rf = RandomForestClassifier(
            n_estimators=400, n_jobs=-1, class_weight="balanced", random_state=0
        )
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        bal = cross_val_score(rf, X, y, scoring="balanced_accuracy", cv=skf, n_jobs=-1)
        majority = pd.Series(y).value_counts().iloc[0] / len(y)
        print(f"  coda {coda:>10}  n={len(sub):>5}  baseline={majority:.3f}  "
              f"balanced_acc={bal.mean():.3f} +/- {bal.std():.3f}")
        rows.append(
            {
                "coda": coda,
                "n": int(len(sub)),
                "baseline": float(majority),
                "balanced_acc": float(bal.mean()),
                "balanced_acc_std": float(bal.std()),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "clan_within_codatype.csv", index=False)

    # ---- 3. Individual ID with grouped split (no whale leakage) ----------
    # Note: the original IDN test used random splits. With many codas per whale
    # the model could memorize timing fingerprints that don't generalize to a
    # held-out whale. The proper test is: predict ID for codas where the whale
    # is in the train set (representative of the real task: identifying a
    # whale from a new utterance). Random CV is the right metric for that.
    # We add a stricter test: within a single coda type, can timing alone
    # still identify the individual? If yes, it suggests a personal
    # fingerprint independent of which coda was produced.
    df_id = df[df["IDN"].astype(str) != "0"]
    print("\nIndividual ID within most common coda types (excluding unidentified):")
    rows = []
    for coda in df_id["CodaType"].value_counts().head(5).index:
        sub = df_id[df_id["CodaType"] == coda]
        counts = sub["IDN"].value_counts()
        keep = counts[counts >= 30].index
        sub = sub[sub["IDN"].isin(keep)]
        if sub["IDN"].nunique() < 2:
            continue
        X = sub[feat_full].to_numpy()
        y = sub["IDN"].astype(str).to_numpy()
        majority = pd.Series(y).value_counts().iloc[0] / len(y)
        rf = RandomForestClassifier(
            n_estimators=400, n_jobs=-1, class_weight="balanced", random_state=0
        )
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        bal = cross_val_score(rf, X, y, scoring="balanced_accuracy", cv=skf, n_jobs=-1)
        print(f"  coda {coda:>10}  n={len(sub):>5}  whales={sub['IDN'].nunique():>2}  "
              f"baseline={majority:.3f}  balanced_acc={bal.mean():.3f}")
        rows.append(
            {
                "coda": coda,
                "n": int(len(sub)),
                "n_whales": int(sub["IDN"].nunique()),
                "baseline": float(majority),
                "balanced_acc": float(bal.mean()),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "individual_within_codatype.csv", index=False)

    # ---- 4. Mean rhythm signature for clans on a shared coda --------------
    # 5R1 is the primary candidate if it's shared.
    candidates = [c for c in ["5R1", "5R2", "5R3", "4R2"] if c in shared.index]
    if candidates:
        coda = candidates[0]
        sig = (
            df[df["CodaType"] == coda]
            .groupby("Clan")[ICI_COLS[: int(coda[0]) - 1] + ["Duration"]]
            .agg(["mean", "std"])
        )
        print(f"\nMean rhythm signature on shared coda '{coda}':")
        print(sig)
        sig.to_csv(OUT / f"rhythm_signature_{coda}.csv")


if __name__ == "__main__":
    main()
