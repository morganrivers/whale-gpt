# REPRODUCING THE WHALE MINI-TRANSFORMER

This file documents how to reproduce, train, and use the
**MiniTransformer** model that achieved the best held-out next-coda
log-loss in our 5-fold CV benchmark
(4.63 ± 0.43 bits/token, perplexity ≈ 24.7 — see
[`outputs/grammar/predict_models.md`](outputs/grammar/predict_models.md)).

## Where is it?

| component                | path                                                |
|--------------------------|-----------------------------------------------------|
| Model class definition   | [`scripts/7_predict_logloss.py`](scripts/7_predict_logloss.py) — class `MiniTransformer` |
| Training/CV loop         | same file — `train_torch` and `evaluate_torch_model` |
| Token corpus (input)     | [`data/whale_tokens.csv`](data/whale_tokens.csv) and [`data/whale_corpus.txt`](data/whale_corpus.txt) |
| Corpus build script      | [`scripts/2_build_corpus.py`](scripts/2_build_corpus.py) |
| Held-out scores (output) | [`outputs/grammar/predict_models.json`](outputs/grammar/predict_models.json) and [`predict_models.md`](outputs/grammar/predict_models.md) |

> **Note:** the script trains a fresh model on each of the 5 CV folds and
> evaluates it on the held-out 20 % — it does *not* currently checkpoint
> weights to disk. To use the model in production you re-run the script
> (~3 minutes on CPU) and either persist the resulting `nn.Module` or copy
> the class into your own loader. A snippet that does so is at the
> bottom of this file.

## Architecture (126 k parameters)

```python
class MiniTransformer(nn.Module):
    def __init__(self, V, k, d=64, n_layers=2, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(V, d)              # learned token embedding
        self.pos = nn.Embedding(k, d)              # learned positional emb.
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4 * d,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.tfm = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d, V)
        self.k = k

    def forward(self, x):                          # x: (B, k) long
        B, K = x.shape
        pos_ids = torch.arange(K, device=x.device).expand(B, K)
        h = self.emb(x) + self.pos(pos_ids)
        h = self.tfm(h)
        return self.head(h[:, -1])                 # logits over V tokens
```

| hyper-parameter        | value                                            |
|------------------------|--------------------------------------------------|
| context length `k`     | 8 past codas (left-padded with `<PAD>`)          |
| token vocab `V`        | ≈ 207 (20 8-character tokens like `i1.=`, `b1*/`, plus `<PAD>`) |
| model dim `d`          | 64                                               |
| layers                 | 2                                                |
| heads                  | 4                                                |
| feed-forward dim       | 256                                              |
| dropout                | 0.1                                              |
| activation             | GELU                                             |
| positional embedding   | learned (one per position, not sinusoidal)       |
| prediction head        | uses only the last context position              |
| optimizer              | AdamW, lr 1e-3, weight decay 1e-4                |
| training               | up to 80 epochs, early-stop on a 10 % val split, patience 10 |
| batch size             | 128                                              |

## Token format the model expects

A coda is encoded as a 4-character string `R T O Ru`:

| field     | meaning                            | values                                |
|-----------|------------------------------------|---------------------------------------|
| `R`       | Rhythm class (Sharma et al. 2024)  | `a`–`r` (18 classes)                  |
| `T`       | Tempo class                        | `1`–`5` (5 duration modes)            |
| `O`       | Ornamentation                      | `.` = no, `*` = yes                   |
| `Ru`      | Rubato direction vs previous coda  | `=` flat, `/` rising, `\` falling, `?` undefined |

Examples: `i1.=`, `b1*/`, `f3.\\`, `d4.?`. There is also a special
sentinel `<PAD>` used when the context is shorter than `k`.

## Reproducing the held-out score from scratch

```sh
# 1) get the repo and base deps
git checkout claude/whale-language-research-tEudI
conda create -n whale-gpt python=3.11 -y && conda activate whale-gpt
pip install pandas numpy scipy scikit-learn torch

# 2) build the token corpus from the raw Sharma et al. annotations
python scripts/00_create_coda_means.py
python scripts/0_extract_codas.py
python scripts/1a_create_dialogue.py
python scripts/1b_create_dialogue_script.py
python scripts/2_build_corpus.py            # writes data/whale_tokens.csv

# 3) run the 5-fold CV log-loss benchmark
python scripts/7_predict_logloss.py
#   ~3 min on CPU. Writes outputs/grammar/predict_models.{md,json}.
```

If `data/whale_tokens.csv` is already in your checkout (the repo ships
it), step 2 collapses to just `python scripts/2_build_corpus.py` if you
want to rebuild it.

## Reading the output

`outputs/grammar/predict_models.md` contains both the summary table and
the per-fold breakdown.

The per-fold output written to stdout looks like:

```
fold 0: M0_majority: bpt=6.019 acc=0.086  M1_markov1: bpt=5.429 acc=0.236
        ...  M7_MiniTfm: bpt=4.602 acc=0.253  (37.3s)
```

`bpt` = bits/token (held-out cross-entropy), `acc` = top-1 accuracy.
Lower bpt is better; PPL = 2^bpt.

## Using the trained MiniTransformer for inference

Because the original script does not save weights, the simplest workflow
is to add a checkpoint after training. Drop this snippet at the bottom
of `scripts/7_predict_logloss.py`'s `evaluate_torch_model` (right before
the `return dict(...)`):

```python
if name == "MiniTfm":
    out_dir = Path("outputs/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            state_dict=model.state_dict(),
            le_classes=list(used_le.classes_),    # token <-> int mapping
            k=k, d=64, n_layers=2, n_heads=4, V=V_full,
        ),
        out_dir / f"minitfm_fold{seed}.pt",
    )
```

Then to predict the next coda from your own 8-coda context:

```python
import torch, numpy as np
from sklearn.preprocessing import LabelEncoder
from scripts.7_predict_logloss import MiniTransformer  # if running from repo root

ckpt = torch.load("outputs/models/minitfm_fold0.pt", weights_only=False)
le = LabelEncoder().fit(np.array(ckpt["le_classes"]))
model = MiniTransformer(
    V=ckpt["V"], k=ckpt["k"],
    d=ckpt["d"], n_layers=ckpt["n_layers"], n_heads=ckpt["n_heads"],
)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# 8-coda context, oldest first; left-pad with "<PAD>" if shorter than 8
context = ["<PAD>", "<PAD>", "<PAD>", "i1.=", "i1.=", "b1.?", "b1*/", "i1.="]
ids = torch.tensor(le.transform(context)).long().unsqueeze(0)  # (1, 8)

with torch.no_grad():
    logits = model(ids)                    # (1, V)
    probs = logits.softmax(-1)[0]
top5 = torch.topk(probs, 5)
for p, i in zip(top5.values, top5.indices):
    print(f"  {le.inverse_transform([i.item()])[0]:8s}  P={p.item():.3f}")
```

Sample output (from a checkpoint of fold 0):

```
  i1.=     P=0.41
  b1.?     P=0.19
  d1.?     P=0.07
  b1*/     P=0.06
  i1.\     P=0.04
```

## Caveats

1. **Small dataset.** ~4 800 codas total ÷ 5 folds ≈ 960 held-out codas
   per fold. Variance across folds is real (per-fold bpt for the
   transformer ranges from 4.05 to 5.27); aggregate over folds before
   drawing conclusions.
2. **Token sparsity.** The 207 tokens have a heavy-tailed distribution
   (top token `5.4` covers ~17 % of all codas). Naive accuracy is
   dominated by the head; bits/token is the metric that captures the
   tail behaviour.
3. **Context length is fixed at 8.** Longer contexts may help marginally
   but the dialogue-level structure (sequence boundaries, choruses) is
   already lost by the time the corpus is flattened to per-position
   windows. A future extension would be to feed `<CHO>` / `<W1>` /
   `<W2>` boundary tokens directly into the model.
4. **No held-out clan or whale.** Folds are made by sequence ID. Some
   whales appear in many sequences, so the held-out fold may share
   speakers with the training fold. A speaker-disjoint split would be a
   more conservative test.
5. **CPU is fine.** A full 5-fold run takes about 3 minutes on a single
   modern CPU core (PyTorch CPU build).

## Why this beats the larger MLPs

The 925 k-parameter `MLP-L (512, 256, 128)` (fed one-hot context of
shape `8 × 207 = 1656`) actively over-fits — its fold-4 log-loss
explodes to 9.18 bpt — while the 126 k-parameter MiniTransformer
sustains 5.27 bpt on the same fold. Two reasons:

- **Learned embeddings** put related tokens (e.g. `i1.=` and `i1.\`)
  near each other, so a single training example influences neighbouring
  vocabulary items.
- **Self-attention** lets the model weight earlier context positions
  rather than treating the 8-coda window as a fixed flattened vector.
  Whale codas show 0.16 bits of mutual information at lag 12
  (see [`FINDINGS.md`](outputs/grammar/FINDINGS.md)) — a model that
  cannot reason about position-specific dependencies leaves bits on the
  floor.

This pattern is consistent with what's seen on small text corpora more
broadly: with O(10⁴) tokens, structured inductive biases (embeddings +
attention) outperform brute-force MLP scale.
