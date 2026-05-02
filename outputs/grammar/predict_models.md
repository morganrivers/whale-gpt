# 5-fold next-Token prediction: bigger models, log-loss

Sequence-level KFold (5 folds, no within-sequence leakage). Target = `Token` (V ≈ 207). Context K = 8 past codas. Metric = held-out cross-entropy in **bits/token** (log₂); lower is better. Perplexity = 2^(bits/token).

| # | model | params | bits/token (↓) | perplexity (↓) | accuracy |
|---|-------|-------:|---------------:|---------------:|---------:|
| 0 | majority (smoothed unigram) | — | 5.993 ± 0.259 | 63.69 | 0.081 ± 0.016 |
| 1 | Markov-1 | — | 5.418 ± 0.371 | 42.75 | 0.242 ± 0.031 |
| 2 | Markov-2 | — | 6.066 ± 0.293 | 67.00 | 0.236 ± 0.049 |
| 3 | MLP-S (128, 64) | 204,863 | 5.157 ± 0.514 | 35.67 | 0.249 ± 0.032 |
| 4 | MLP-M (256, 128) | 425,919 | 5.596 ± 0.995 | 48.36 | 0.256 ± 0.046 |
| 5 | MLP-L (512, 256, 128) | 925,631 | 5.973 ± 1.653 | 62.80 | 0.251 ± 0.037 |
| 6 | Embedding-MLP (d=32, 256·128) | 131,049 | 4.843 ± 0.413 | 28.69 | 0.232 ± 0.024 |
| 7 | MiniTransformer (2L, 4h, d=64) | 126,409 | 4.625 ± 0.433 | 24.68 | 0.251 ± 0.036 |

## Per-fold bits/token

| fold | majority (smoothed unigram) | Markov-1 | Markov-2 | MLP-S (128, 64) | MLP-M (256, 128) | MLP-L (512, 256, 128) | Embedding-MLP (d=32, 256·128) | MiniTransformer (2L, 4h, d=64) |
|------|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.019 | 5.429 | 6.116 | 5.419 | 5.681 | 5.181 | 4.935 | 4.602 |
| 1 | 5.817 | 5.142 | 5.865 | 4.784 | 4.740 | 5.531 | 4.567 | 4.297 |
| 2 | 6.217 | 5.609 | 6.220 | 5.498 | 7.147 | 5.543 | 4.977 | 4.908 |
| 3 | 5.601 | 4.920 | 5.639 | 4.345 | 4.338 | 4.430 | 4.256 | 4.048 |
| 4 | 6.310 | 5.991 | 6.491 | 5.737 | 6.073 | 9.178 | 5.478 | 5.271 |

## Takeaway

- The best held-out model is **MiniTransformer (2L, 4h, d=64)** at 4.625 bits/token (perplexity ≈ 24.68).
- The unigram majority baseline scores 5.993 bits/token; the best model saves **1.37 bits/token** (perplexity drops 63.7 → 24.7).
- Markov-1 alone is already strong (5.418 bpt). Larger models help further; the MiniTransformer / Emb-MLP can attend to the full 8-coda context.
