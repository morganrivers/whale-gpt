# 5-fold next-coda prediction results

Sequence-level 5-fold CV (no within-sequence leakage). Context length **K = 4** past codas. Models: majority baseline, Markov-1 baseline, Random Forest (200 trees), MLP (128, 64). Targets: Rhythm (V=18), Tempo (V=5), Token = rhythm·tempo·orn·rubato (V≈207). Results are mean ± std across 5 folds.

## Target: Rhythm

| model | accuracy | top-3 accuracy | perplexity |
|-------|---------:|---------------:|-----------:|
| majority | 0.398 ± 0.060 | 0.689 ± 0.019 | — |
| Markov-1 | 0.506 ± 0.073 | 0.774 ± 0.020 | 4.89 ± 0.71 |
| Random Forest | 0.554 ± 0.056 | 0.827 ± 0.022 | 4.24 ± 0.58 |
| MLP (128,64) | 0.536 ± 0.063 | 0.803 ± 0.048 | 4.41 ± 0.88 |

## Target: Tempo

| model | accuracy | top-3 accuracy | perplexity |
|-------|---------:|---------------:|-----------:|
| majority | 0.416 ± 0.044 | 0.863 ± 0.025 | — |
| Markov-1 | 0.647 ± 0.045 | 0.895 ± 0.022 | 2.85 ± 0.33 |
| Random Forest | 0.675 ± 0.042 | 0.922 ± 0.021 | 2.53 ± 0.32 |
| MLP (128,64) | 0.681 ± 0.046 | 0.913 ± 0.027 | 2.53 ± 0.30 |

## Target: Token

| model | accuracy | top-3 accuracy | perplexity |
|-------|---------:|---------------:|-----------:|
| majority | 0.076 ± 0.015 | 0.228 ± 0.051 | — |
| Markov-1 | 0.239 ± 0.034 | 0.445 ± 0.067 | 45.51 ± 12.73 |
| Random Forest | 0.253 ± 0.035 | 0.505 ± 0.082 | 48.85 ± 17.11 |
| MLP (128,64) | 0.260 ± 0.035 | 0.484 ± 0.062 | 36.85 ± 12.72 |

## Interpretation

- **Rhythm (chance ≈ 1/18 = 5.6 %)**: majority baseline 39.8 %, Markov-1 50.6 %, RF **55.4 %**, MLP **53.6 %**. RF beats majority by +15.6 percentage points and Markov-1 by +4.9 pp.
- **Tempo (chance ≈ 1/5 = 20 %)**: majority 41.6 %, Markov-1 64.7 %, RF **67.5 %**, MLP **68.1 %**.
- **Token (chance ≈ 1/207 = 0.5 %)**: majority 7.6 %, Markov-1 23.9 %, RF **25.3 %**, MLP **26.0 %**. Top-3 accuracy of the MLP is **48.4 %** (vs ~1.5 % top-3 chance).

Both learned models substantially out-predict the majority baseline, and beat the simple Markov-1 baseline on every target — confirming that information beyond the immediately preceding coda is usable on a held-out 20 % of sequences.
