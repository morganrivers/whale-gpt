<img src="./assets/Sperm_whale_pod.jpg">


Hello!

This repo is a first attempt to build WhaleGPT, a transformer 'language' model for whale language. The data and codas used in this model come from [Sharma et al.](https://github.com/pratyushasharma/sw-combinatoriality) At this point the data is likely insufficient for a model that is directly useful, so it should be considered experimental.

The approach is the following:

1. The data used in Sharma et al. contains click sequences with a length of up to 28 inter click intervals, i.e. 29 clicks, but the codas being used to encode these click sequences have up to 9 inter click intervals, i.e. up to 10 clicks. Longer click sequences can not be encoded as a single coda, as throwign away surplus click intervals would be rash. We have therefore developed an algorithm to code click sequences into codas, contained in [`scripts/0_extract_codas.py`](https://github.com/0xideas/whale-gpt/blob/main/scripts/0_extract_codas.py). This algorithm is used to decode all click sequences, now named 'vocalizations', into sequences of codas. This applies to all vocalizations, including those consisting of 10 clicks or less. The algorithm takes the mean relative interval sequence for each coda (i.e. the mean time percentile of each click in the coda), and recursively divides the vocalization into codas or 'surplus' clicks (that can usually be interpreted as ornamentation). The sequences of these subdivisions are then scored by the manhattan distance, and the sequence of codas with the minimum total distance is then taken as the vocalization encoding into codas.
2. These coda sequences are then encoded into dialogue form using [`scripts/1_create_dialogue.py`](https://github.com/0xideas/whale-gpt/blob/main/scripts/1_create_dialogue.py). The main decisions on how to represent these overlapping coda sequences by multiple whales in discrete form are (1) Codas that begin sufficiently close in time are considered simultaneous and encoded in a single row. The threshold used to making this determination is somewhat arbitrary and currently set to 0.3 seconds. (2) Some recordings contain two or more whales, and this can be represented in multiple ways. Here, we adopt a 'me' vs 'other' encoding, where for each whale contained in a recording, the codas emitted by the whale is encoded in the columns "Coda1", "Ornamentation1" and "Duration1", while the codas emitted by any of the other whales are encoded in the columns "Coda2", "Ornamentation2" and "Duration2". When either the primary whale or the other whales are silent, this is encoded with an additional token '98'. The data used for modelling can be found at [`data/whale-dialogues.csv`](https://github.com/0xideas/whale-gpt/blob/main/data/whale-dialogues.csv).
3. The language model itself is a decoder only transformer with 126k parameters that autoregressively models "Coda1", "Ornamentation1", "Duration1", "Coda2", "Ornamentation2" and "Duration2". Each incremental output of these variables is generated from the previous 25 values of all of these variables. We use the package [sequifier](https://github.com/0xideas/sequifier) that enables the easy configuration, training and inference for models of this type.

Several generated "Whale dialogues" can be found in [`outputs/predictions/sequifier-dialogue-best-5000-predictions.csv`](https://github.com/0xideas/whale-gpt/blob/main/outputs/predictions/sequifier-dialogue-best-5000-predictions.csv), where each sequenceId value is a single "dialogue". Currently most of them end up with both whales repeatedly emitting the coda "5", either while the other is silent or together. 

The roughly 9k observations contained in this dataset are clearly insufficient to create a useful model, but this modelling work should serve as an encouragement for additional data collection and a basis for future model development.

The development of this model can be reproduced in the following steps, using Mac (or likely most Linux distributions). Since all the artefacts are also contained in this repository, all the steps after can also be executed individually (after executing 1., 2., 3. and 7.).

1. `conda create --name whale-gpt python=3.11 -y`
2. `conda activate whale-gpt`
3. `pip install sequifier==0.4.0.0 scikit-learn`
4. `python scripts/00_create_coda_means.py`
5. `python scripts/0_extract_codas.py`
6. `python scripts/1_create_dialogue.py`
7. `sequifier preprocess`
8. `sequifier train`
9. `sequifier infer`

## Grammar / structure analysis (beyond Sharma et al. 2024)

A separate analysis pipeline lives under `scripts/2_build_corpus.py` →
`scripts/3_grammar_analysis.py` → `scripts/4_compression_test.py` →
`scripts/5_discover_phrases.py`. It builds a compact human-readable
token corpus at `data/whale_corpus.txt` and runs compression / mutual-
information / phrase-discovery experiments. Results and a written
write-up land in `outputs/grammar/`; the synthesis is in
[`outputs/grammar/FINDINGS.md`](outputs/grammar/FINDINGS.md).

Headline: a 1st-order Markov chain with the same transition matrix as
the data loses essentially all rhythm mutual-information after lag 2,
yet the real corpus retains 0.16 bits at lag 12 — direct evidence of
multi-coda grammar that Sharma et al. left as an open question. The
corpus also compresses ~26 % smaller with bz2 than a token-shuffled
control, contains near-deterministic 4-coda "words" (e.g. `i1 d1 b1 b1`,
P=0.93), and exhibits duality-of-patterning signatures (rhythm bigrams
realised across many tempo profiles and vice-versa).