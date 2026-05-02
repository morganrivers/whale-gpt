"""Random forest analyses on Dominica sperm whale codas.

Investigates whether click-timing patterns alone encode:
  1. Clan identity (EC1 vs EC2)
  2. Social unit identity (within-clan dialect)
  3. Individual whale identity
  4. Coda type (sanity check + structural insight)

Also reports per-feature importance and per-clan rhythm signatures.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

DATA = Path("data/DominicaCodas.csv")
OUT = Path("outputs/analysis")
OUT.mkdir(parents=True, exist_ok=True)

ICI_COLS = [f"ICI{i}" for i in range(1, 10)]
RAW_FEATS = ["nClicks", "Duration"] + ICI_COLS


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Add features that don't depend on padding-zero conventions."""
    df = df.copy()
    # cumulative click times normalized by total duration (relative rhythm)
    icis = df[ICI_COLS].to_numpy()
    cum = np.cumsum(icis, axis=1)
    # avoid divide-by-zero
    dur = df["Duration"].replace(0, np.nan).to_numpy().reshape(-1, 1)
    rel = np.where(dur > 0, cum / dur, 0.0)
    for i in range(9):
        df[f"REL{i+1}"] = rel[:, i]
    # ratios between successive non-zero ICIs (rhythm contour)
    for i in range(1, 9):
        a = df[f"ICI{i}"].to_numpy()
        b = df[f"ICI{i+1}"].to_numpy()
        df[f"RATIO{i}"] = np.where((a > 0) & (b > 0), b / a, 0.0)
    df["meanICI"] = df[ICI_COLS].replace(0, np.nan).mean(axis=1).fillna(0)
    df["stdICI"] = df[ICI_COLS].replace(0, np.nan).std(axis=1).fillna(0)
    df["cv_ICI"] = np.where(df["meanICI"] > 0, df["stdICI"] / df["meanICI"], 0.0)
    return df


def evaluate(df: pd.DataFrame, target: str, feature_cols: list[str], note: str = "") -> dict:
    sub = df.dropna(subset=[target]).copy()
    counts = sub[target].value_counts()
    keep = counts[counts >= 30].index
    sub = sub[sub[target].isin(keep)]
    if sub[target].nunique() < 2:
        return {"target": target, "skip": "too few classes"}
    X = sub[feature_cols].to_numpy()
    y = sub[target].astype(str).to_numpy()

    # baseline = predict majority class
    majority = pd.Series(y).value_counts().iloc[0] / len(y)
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced",
        random_state=0,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    acc = cross_val_score(rf, X, y, scoring="accuracy", cv=skf, n_jobs=-1)
    bal = cross_val_score(rf, X, y, scoring="balanced_accuracy", cv=skf, n_jobs=-1)
    rf.fit(X, y)
    importances = sorted(
        zip(feature_cols, rf.feature_importances_), key=lambda t: -t[1]
    )
    result = {
        "target": target,
        "note": note,
        "n_samples": int(len(sub)),
        "n_classes": int(sub[target].nunique()),
        "majority_baseline": float(majority),
        "cv_accuracy_mean": float(acc.mean()),
        "cv_accuracy_std": float(acc.std()),
        "cv_balanced_accuracy_mean": float(bal.mean()),
        "cv_balanced_accuracy_std": float(bal.std()),
        "top_features": [(f, float(v)) for f, v in importances[:10]],
        "class_distribution": {str(k): int(v) for k, v in counts.items()},
    }
    return result


def main() -> None:
    df = pd.read_csv(DATA)
    df = add_engineered(df)
    feat_basic = RAW_FEATS
    feat_full = RAW_FEATS + [f"REL{i}" for i in range(1, 10)] + [
        f"RATIO{i}" for i in range(1, 9)
    ] + ["meanICI", "stdICI", "cv_ICI"]

    results = []

    # --- Experiment 1: Clan identity (EC1 vs EC2) ----------------------------
    results.append(evaluate(df, "Clan", feat_full, "click timing -> clan dialect"))

    # also restrict to a single shared coda type to control for structural difference
    common = df["CodaType"].value_counts().head(1).index[0]
    df_common = df[df["CodaType"] == common]
    res_common = evaluate(
        df_common,
        "Clan",
        feat_full,
        f"clan within single coda type '{common}' (controls for type)",
    )
    results.append(res_common)

    # --- Experiment 2: Social unit -----------------------------------------
    results.append(evaluate(df, "Unit", feat_full, "click timing -> social unit"))

    # within EC1 only (since EC2 has just 1 unit basically)
    df_ec1 = df[df["Clan"] == "EC1"]
    results.append(evaluate(df_ec1, "Unit", feat_full, "unit within EC1 only"))

    # --- Experiment 3: Individual whale ------------------------------------
    df_id = df[df["IDN"].astype(str) != "0"]  # 0 means unidentified
    results.append(evaluate(df_id, "IDN", feat_full, "individual ID (unidentified excluded)"))

    # --- Experiment 4: Coda type (structural sanity check) -----------------
    results.append(evaluate(df, "CodaType", feat_full, "raw timing -> coda type"))

    # --- Per-clan signature on the most common coda type --------------------
    sig = (
        df[df["CodaType"] == common]
        .groupby("Clan")[ICI_COLS + ["Duration"]]
        .agg(["mean", "std"])
    )
    sig.to_csv(OUT / f"clan_signature_{common.replace('+','p')}.csv")

    with open(OUT / "rf_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # human-readable summary
    lines = []
    for r in results:
        if "skip" in r:
            continue
        lines.append("=" * 72)
        lines.append(f"target: {r['target']}  ({r['note']})")
        lines.append(f"  n={r['n_samples']}  classes={r['n_classes']}  "
                     f"majority-baseline={r['majority_baseline']:.3f}")
        lines.append(f"  CV accuracy:           {r['cv_accuracy_mean']:.3f} "
                     f"+/- {r['cv_accuracy_std']:.3f}")
        lines.append(f"  CV balanced accuracy:  {r['cv_balanced_accuracy_mean']:.3f} "
                     f"+/- {r['cv_balanced_accuracy_std']:.3f}")
        lines.append("  top features:")
        for f, v in r["top_features"]:
            lines.append(f"    {f:<10} {v:.4f}")
    summary = "\n".join(lines)
    (OUT / "rf_summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
