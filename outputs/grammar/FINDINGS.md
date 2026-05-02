# Whale-language grammar: structure beyond Sharma et al. (2024)

This file documents what is found beyond the four sub-coda features
(rhythm, tempo, rubato, ornamentation) reported in
[Sharma et al., *Nat. Commun.* 2024](https://doi.org/10.1038/s41467-024-47221-8).
The hint we follow throughout: **better understanding compresses better.**
Every claim below is backed by either a compression ratio, a held-out
cross-entropy gap, or a mutual-information surplus over a matched-statistics
control.

All experiments use Dataset 2 of Sharma et al. (the DTag-tagged subset
with timing & speaker IDs), exposed via `data/whale_tokens.csv` and the
plain-text view in `data/whale_corpus.txt`. The token format is

    R T O Ru          e.g.  i1.=  b1*/  f3.\ 

where R∈{a..r} is rhythm (18 classes), T∈{1..5} is tempo, O is `*` for an
ornament else `.`, and Ru is `=`,`/`,`\` for level / rising / falling rubato.
Sequences are split into single-whale runs (`W1 …`), choruses
(`CHO 1,2 …`), and pauses.

## Headline result — the corpus is not a memoryless or 1st-order chain

Mutual information `MI(rhythm[t], rhythm[t+k])` is plotted against a
simulated 1st-order Markov chain that *shares the same transition matrix*
as the data. By construction the simulated chain matches MI at lag 1 and
should decay to ~0 by lag 3. Real whale data does not.

| lag k |  real MI |  Markov-1 sample MI |  excess (bits) |
|------:|---------:|--------------------:|---------------:|
|     1 |    0.536 |               0.580 |         −0.044 |
|     2 |    0.498 |               0.213 |     **+0.286** |
|     3 |    0.417 |               0.107 |     **+0.311** |
|     5 |    0.329 |               0.054 |     **+0.275** |
|     8 |    0.291 |               0.047 |     **+0.243** |
|    12 |    0.210 |               0.052 |     **+0.159** |
|    20 |    0.136 |               0.065 |     **+0.071** |
|    30 |    0.104 |               0.077 |         +0.027 |

A 1st-order Markov chain with the *exact* same bigram statistics
loses essentially all rhythm information after two steps. The real
corpus retains an excess of 0.16 bits at lag 12 and is still measurably
above the Markov baseline at lag 30. Sharma et al. analyse only
*within-coda* structure; this is direct evidence of **multi-coda grammar**.

## Compression: real vs. shuffled

Universal compressors run on the readable corpus (no tuning):

| codec | real (bytes) | line-shuffled | token-shuffled | savings vs token-shuffle |
|------:|-------------:|--------------:|---------------:|-------------------------:|
| gzip  |        8 553 |         9 443 |         11 475 |               **+25.5 %** |
| bz2   |        6 405 |         6 823 |          8 611 |               **+25.6 %** |
| lzma  |        7 548 |         8 653 |         10 634 |               **+29.0 %** |

Token-shuffling preserves the unigram distribution but destroys order;
all three compressors reduce file size by ~26–29 % when given the real
order, so ~3 600 bits of structural information per 4 802 codas (≈0.75
bits/token) sit *purely in token order*.

Held-out cross-entropy of an order-1 Markov model trained on N−1
sequences and scored on the held-out sequence (Token alphabet, V=207):

| feature       | H₀ (unigram) | H₁ (Markov-1) | H₁(shuffled) | H₀−H₁ | shuffled − H₁ |
|---------------|-------------:|--------------:|-------------:|------:|--------------:|
| Token (V=207) |        6.115 |     **5.568** |        6.253 |  0.547 |     **+0.685** |
| Rhythm (V=18) |        2.777 |     **2.296** |        2.590 |  0.481 |     **+0.294** |
| Tempo (V=5)   |        2.046 |     **1.522** |        1.839 |  0.524 |     **+0.318** |
| Rubato (V=4)  |        1.720 |         1.707 |        1.722 |  0.013 |          0.015 |
| Ornament (2)  |        0.568 |         0.570 |        0.574 | −0.002 |          0.004 |

Held-out gains are real (rubato/ornamentation are known *boundary*
features, so they're not predictable from local context — consistent with
Sharma et al.).

## Discovered "words": near-deterministic multi-coda phrases

Top n-grams over `(rhythm.tempo)` ranked by log-likelihood-ratio over
independence, with the *deterministic-continuation* probability
P(last token | prefix) shown:

| n-gram (rhythm.tempo) | count | observed/expected | P(last \| prefix) |
|-----------------------|------:|------------------:|------------------:|
| 5.4 5.4 5.4 5.4 5.4   |  453  |          **104×** |         **0.90** |
| 6.0 6.0 6.0 6.0 6.0   |   80  |        **3 586×** |         **0.82** |
| 5.3 5.3 5.3 5.3 5.3   |   17  |        **4 811×** |         **0.68** |
| 6.1 6.1 6.1 6.1 6.1   |   10  |      **416 969×** |         **1.00** |
| **8.0 3.0 1.0 1.0**   |   27  |        **119×**   |         **0.93** |
| 5.4 4.2 5.4 4.2 5.4   |   20  |          **231×** |         **0.83** |
| 4.2 5.4 5.4 5.4 5.4   |   27  |           **44×** |         **0.93** |

Two kinds of phrase emerge:

1. **Sustained iteration** of one coda type for 3–5+ codas (`5.4 5.4 …`,
   `6.0 6.0 …`). This is what whale-gpt's predictions also tend to
   produce — and now we know it's a real grammatical pattern, not a
   model failure mode. Whales really do "drum" on a single coda type.

2. **Polysyllabic motifs** with deterministic continuations:
   - `i1 d1 b1 b1` (= `8.0 3.0 1.0 1.0`) is a four-coda sequence with
     P(`b1` | `i1 d1 b1`) = 0.93. This is a discovered *word*: rhythm
     **i → d → b → b** at the slowest tempo. It does not arise from
     any single rhythm/tempo cluster in the paper.
   - `5.4 4.2 5.4 4.2 5.4` — an alternation pattern between two
     specific (rhythm, tempo) tokens that is 231× more common than
     independence predicts.

## Evidence of duality of patterning

The paper closes with: *“Our findings open up the possibility that sperm
whale communication might provide our first example of \[duality of
patterning\] in another species.”* A signature of duality is that the
same low-level "phoneme" sequences recur **across multiple realisations
of higher-level prosody** (tempo) and vice-versa. We measure this
directly:

- 69 distinct rhythm bigrams occur ≥5 times. Across them, the average
  conditional entropy H(tempo bigram | rhythm bigram) = **1.58 bits**.
  E.g. rhythm bigram **(3, 5)** appears 150 times across **15 distinct
  tempo bigrams** with H = 3.31 bits. The rhythm pair (3,5) is reused
  with very different tempo profiles.
- Conversely, 25 tempo bigrams occur ≥5 times, with average
  H(rhythm bigram | tempo bigram) = **2.93 bits** — the same tempo
  contour carries a wide rhythm vocabulary.

Rhythm and tempo are decoupled at the *sequence* level, not just at the
single-coda level the paper documented.

## Turn-taking grammar (cross-whale conditional structure)

For 2 192 close-onset pairs where two whales exchange codas within 5 s:

| feature | H(responder) | H(responder \| initiator) | shuffled control |
|---------|-------------:|--------------------------:|-----------------:|
| rhythm  |        2.516 |                  **2.251** |            2.461 |
| tempo   |        2.151 |                  **1.840** |            2.146 |
| token   |        5.898 |                  **3.553** |            4.242 |

Sharma's "rubato is imitated" claim shows duration matching at the
continuous level. We additionally show that whales **categorically match
tempo class** in turn-taking (responder tempo class entropy drops 0.31
bits when conditioned on initiator), and that knowing the initiator's
full token cuts responder uncertainty by **2.34 bits** (well above the
1.66-bit drop we'd see by chance).

## Boundary markers (independent of ornamentation)

The paper showed *ornamented* codas mark exchange boundaries.
We find **purely-rhythmic** boundary markers: certain (rhythm, tempo)
tokens are 6–20× more likely to appear at the *first* position of a
single-whale run than in the middle:

| token | n_start | P(start) / P(middle) |
|-------|--------:|---------------------:|
| 1.2   |    15   |              **20.0** |
| 0.2   |    20   |               **6.7** |
| 3.2   |    15   |               **6.7** |
| 4.2   |    36   |               **6.0** |

Tempo class **2** (durations 0.51–0.61 s) is over-represented at run
starts across multiple rhythms. Conversely tokens **3.1**, **5.0**, **3.2**
are over-represented at run *ends*. So sperm whales appear to use a
small set of fixed (rhythm, tempo) markers to open and close exchanges,
analogous to discourse markers like "ok, so …" / "… right?" in human
conversation.

## Near-deterministic FSA transitions

Treating each (rhythm, tempo) as a finite-state-automaton state, several
transitions are highly deterministic (P(top successor) > 0.5, n ≥ 20):

| state    |    n | top successor | P(top) |
|----------|-----:|---------------|-------:|
| (5, 4)   | 1111 | (5, 4)        |  0.74  |
| (0, 2)   |   31 | (1, 0)        |  0.74  |
| (1, 2)   |   29 | (1, 0)        |  0.59  |
| (6, 0)   |  374 | (6, 0)        |  0.57  |
| (7, 2)   |   20 | (7, 2)        |  0.55  |
| (1, 1)   |   64 | (1, 0)        |  0.52  |
| (4, 2)   |  165 | (5, 4)        |  0.52  |

Several "openers" feed back into the same low-tempo run (`x.2 → 1.0`),
consistent with the boundary-marker finding above.

## What about cross-whale rubato direction?

We tested whether the responder's rubato direction (↑/=/↓) depends on
the initiator's. MI = 0.0021 bits versus a shuffled control of
0.0041 ± 0.0030 bits — **no signal**. Sharma's "rubato is imitated"
result is in *duration*, not in *direction*: whales match the
target duration but pick the direction independently. This refines the
paper's claim and suggests rubato direction is a private, not a
coordinated, feature.

## Per-feature summary

| paper claim                                 | this work confirms                | this work newly shows                                        |
|---------------------------------------------|-----------------------------------|--------------------------------------------------------------|
| 18 rhythms × 5 tempos × orn × rubato        | 207 distinct realised tokens     | tokens follow strong *sequential* grammar, not just combinatorial |
| ornamentation marks chorus changes           | yes (boundaries)                  | non-ornamented (rhythm, tempo) markers also flag boundaries |
| rubato is imitated                           | yes (duration)                    | tempo *class* is also matched in turn-taking; rubato *direction* is **not** coupled across whales |
| repertoire is combinatorial                  | yes                               | combinatorial *and* shows duality-of-patterning signatures (rhythm/tempo decouple at sequence level) |
| sequential structure is an open question     | acknowledged                      | discovered concrete near-deterministic phrases including a 4-coda "word" `i1 d1 b1 b1` |
| ≤5 bits/coda                                 | unigram entropy ≈ 6.1 bits        | held-out perplexity drops to 5.57 bits/coda with Markov-1; further reductions available with longer context |

## How to reproduce

```sh
python3 scripts/2_build_corpus.py        # writes data/whale_tokens.csv & whale_corpus.txt
python3 scripts/3_grammar_analysis.py    # writes outputs/grammar/{summary.txt, grammar_results.json}
python3 scripts/4_compression_test.py    # writes outputs/grammar/compression*.json|txt
python3 scripts/5_discover_phrases.py    # writes outputs/grammar/discovery*.json|txt
```
