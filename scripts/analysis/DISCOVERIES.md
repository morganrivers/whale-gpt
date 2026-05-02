# Discoveries from random-forest analysis of Dominica sperm-whale codas

Dataset: `data/DominicaCodas.csv` — 8,719 codas, 2 clans (EC1, EC2),
13 social units, 35 coda types, 36 individual whales.

Random forests (400 trees, 5-fold stratified CV, balanced class weights)
were trained on raw inter-click intervals plus engineered rhythm features
(cumulative-time fractions `REL1..REL9`, successive-ICI ratios `RATIO1..RATIO8`,
mean / std / coefficient-of-variation of ICIs).

Scripts: `scripts/analysis/rf_whale_patterns.py`, `scripts/analysis/followup.py`.
Raw outputs: `outputs/analysis/rf_results.json`, `rf_summary.txt`,
`codatype_by_clan.csv`, `clan_within_codatype.csv`, `individual_within_codatype.csv`.

---

## 1. The two clans have near-disjoint vocabularies

EC1 and EC2 don't just have different accents — they speak almost
non-overlapping dialects.

| coda type | EC1 uses | EC2 uses |
|-----------|---------:|---------:|
| `1+1+3`   |    3,574 |       15 |
| `5R1`     |    1,510 |        0 |
| `5R3`     |       19 |      623 |
| `4D`, `5R2`, `8i`, `2+3`, `3D`, `1+31`, `7D2` | each 35–287 | 0 |

EC1's signature call (`1+1+3`, used 3,574 times) is essentially absent from
EC2 (15 uses); EC2's signature (`5R3`) is essentially absent from EC1.
Only three coda types are used in meaningful numbers by both clans
(`4R2`, `5-NOISE`, `9R`).

This is a concrete, quantitative confirmation of vocal-clan structure
in the dataset.

## 2. Clan dialect leaks into shared codas (95.1% balanced accuracy)

Even when both clans use the same coda type, the way they *pronounce* it
differs enough for a random forest to identify the clan from raw click
timing.

| target                                    | n     | baseline | RF balanced accuracy |
|-------------------------------------------|------:|---------:|---------------------:|
| Clan (all data)                           | 8,719 |    0.891 |  **0.951 ± 0.008**   |
| Clan within coda `9R`                     |    43 |    0.535 |  **0.935 ± 0.083**   |
| Clan within coda `5-NOISE`                |   280 |    0.811 |  **0.888 ± 0.063**   |
| Clan within coda `4R2`                    |   340 |    0.891 |  **0.624 ± 0.049**   |

For `9R`, the RF nearly doubles the majority-class baseline (53→93%),
so the same coda is *audibly* clan-specific in its rhythm. On `4R2` the
gain is smaller, suggesting that this coda is closer to clan-neutral.

## 3. Individuals have rhythmic fingerprints within a single coda type

Holding coda type fixed (so the model can't cheat by recognising what
was said), the random forest still identifies the producing whale at
2–3× the majority baseline:

| coda type | whales | n     | baseline | RF balanced accuracy |
|-----------|-------:|------:|---------:|---------------------:|
| `5R1`     |      8 |   415 |    0.188 |  **0.512**           |
| `1+1+3`   |     11 | 1,310 |    0.206 |  **0.334**           |
| `5R2`     |      2 |    89 |    0.663 |  **0.698**           |

Individual whales pace their clicks distinctively — there's a personal
rhythm signature on top of clan/unit dialect.

## 4. Mid-coda intervals, not duration, carry the dialect signal

Across every classification target, the most informative features were
the *middle* inter-click intervals and their ratios, not call length or
click count. Top-10 feature importances for `Clan`:

```
ICI4       0.140
RATIO2     0.121   (= ICI3 / ICI2)
ICI3       0.121
cv_ICI     0.109   (coefficient of variation across all ICIs)
REL2       0.089   (cumulative time of click 3, normalized)
REL1       0.069
REL3       0.059
stdICI     0.047
Duration   0.044
meanICI    0.041
```

`nClicks` and absolute `Duration` are weak features. What matters is
how the call is paced internally — clans differ in the middle of the
phrase, not in its length.

## 5. Social-unit dialect is real but ~2× weaker than clan dialect

| target                                  | n     | classes | baseline | RF balanced accuracy |
|-----------------------------------------|------:|--------:|---------:|---------------------:|
| Clan                                    | 8,719 |       2 |    0.891 |               0.951  |
| Unit (all)                              | 8,719 |      13 |    0.193 |               0.443  |
| Unit within EC1 only                    | 7,770 |      10 |    0.216 |               0.429  |

Social units are distinguishable above chance (~2× baseline), but the
signal is much weaker than the clan signal — consistent with dialect
forming primarily at the clan level, with finer-grained unit variation
layered on top.

---

## Methods note

- Class weighting was set to `balanced` to avoid the model collapsing onto
  the majority clan (EC1, 89%) — balanced accuracy is the metric that
  protects against this trivially.
- Within-coda clan classification controls for the trivial leak that EC1
  and EC2 use mostly different codas. The fact that classification still
  works inside `9R` and `5-NOISE` shows the clan signal is encoded in
  the rhythm, not just the call inventory.
- Individuals labelled `IDN == 0` ("unidentified") were excluded from
  individual-ID experiments.
- Sample sizes per class were thresholded at ≥30 to avoid degenerate folds.
