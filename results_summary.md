# Yelp reproduction results (Elliot, run 2026-10-30)

Temporal hold-out split (test 20%, validation 20%), relevance threshold 1. Hyperparameters are picked by nDCG@10 on validation. Gini comes from Elliot, where a higher value means recommendations are spread more evenly across items.

## Best configuration per model, cutoff @10

| Model | Best config | nDCG | Precision | ItemCoverage | EPC | Gini |
|---|---|---:|---:|---:|---:|---:|
| Random | seed=42 | 0.0005 | 0.0001 | 9866 | 0.0001 | **0.8997** |
| MostPop | – | 0.0111 | 0.0031 | 15 | 0.0032 | 0.0009 |
| UserKNN | nn=40, cosine | 0.0308 | 0.0074 | 8166 | 0.0084 | 0.2473 |
| ItemKNN | nn=100, cosine | 0.0146 | 0.0042 | 8668 | 0.0043 | 0.3693 |
| AttributeItemKNN | nn=70, braycurtis | 0.0022 | 0.0006 | 9727 | 0.0007 | 0.4499 |
| MF | f=10, best epoch 6 | 0.0080 | 0.0021 | 795 | 0.0023 | 0.0210 |
| PureSVD | f=10 | **0.0358** | **0.0090** | 173 | **0.0103** | 0.0053 |
| BPRMF | f=10, best epoch 5 | 0.0130 | 0.0032 | 25 | 0.0035 | 0.0011 |
| NonNegMF | f=10, best epoch 2 | 0.0016 | 0.0004 | 746 | 0.0005 | 0.0231 |
| PMF | f=50, best epoch 10 | 0.0084 | 0.0023 | 1277 | 0.0025 | 0.0287 |

## Best configuration per model, cutoff @5

| Model | nDCG | Precision | ItemCoverage | EPC | Gini |
|---|---:|---:|---:|---:|---:|
| Random | 0.0003 | 0.0001 | 9866 | 0.0001 | **0.8591** |
| MostPop | 0.0078 | 0.0034 | 9 | 0.0035 | 0.0004 |
| UserKNN | 0.0236 | 0.0091 | 7124 | 0.0098 | 0.2152 |
| ItemKNN | 0.0097 | 0.0043 | 8183 | 0.0044 | 0.3262 |
| AttributeItemKNN | 0.0016 | 0.0007 | 9381 | 0.0007 | 0.4182 |
| MF | 0.0059 | 0.0024 | 536 | 0.0025 | 0.0139 |
| PureSVD | **0.0283** | **0.0115** | 99 | **0.0123** | 0.0029 |
| BPRMF | 0.0095 | 0.0037 | 17 | 0.0039 | 0.0005 |
| NonNegMF | 0.0011 | 0.0005 | 734 | 0.0005 | 0.0204 |
| PMF | 0.0060 | 0.0026 | 949 | 0.0028 | 0.0204 |

## Grid search, cutoff @10

| Model | Config | nDCG | Precision | ItemCoverage | Gini |
|---|---|---:|---:|---:|---:|
| ItemKNN | nn=50, cosine | 0.0142 | 0.0040 | 8707 | 0.3784 |
| ItemKNN | nn=70, cosine | 0.0143 | 0.0040 | 8683 | 0.3736 |
| ItemKNN | **nn=100, cosine** | **0.0146** | 0.0042 | 8668 | 0.3693 |
| ItemKNN | nn=50, euclidean | 0.0016 | 0.0006 | 4765 | 0.0166 |
| ItemKNN | nn=70, euclidean | 0.0016 | 0.0006 | 4762 | 0.0167 |
| ItemKNN | nn=100, euclidean | 0.0017 | 0.0006 | 4770 | 0.0170 |
| AttributeItemKNN | nn=50, braycurtis | 0.0021 | 0.0006 | 9721 | 0.4446 |
| AttributeItemKNN | **nn=70, braycurtis** | **0.0022** | 0.0006 | 9727 | 0.4499 |
| AttributeItemKNN | nn=100, braycurtis | 0.0021 | 0.0006 | 9720 | 0.4556 |
| AttributeItemKNN | nn=50, manhattan | 0.0016 | 0.0004 | 9506 | 0.4025 |
| AttributeItemKNN | nn=70, manhattan | 0.0017 | 0.0005 | 9484 | 0.4072 |
| AttributeItemKNN | nn=100, manhattan | 0.0017 | 0.0004 | 9428 | 0.4082 |

## Observations

- **Most accurate model:** PureSVD (nDCG@10 0.036), with UserKNN close behind (0.031). PureSVD only recommends 173 distinct items, so most of its accuracy comes from popular items.
- **Best balance of accuracy and diversity:** UserKNN. It is almost as accurate as PureSVD, covers more than 8k items and has a Gini of 0.25.
- **MostPop is a strong baseline.** MF, PMF, NonNegMF and AttributeItemKNN all score below it. BPRMF scores about the same as MostPop and only covers 25 items, so it is probably just learning popularity.
- **The learned MF models look undertrained.** They ran 10 epochs at lr=0.001, PMF was still improving at epoch 10, and NonNegMF peaked at epoch 2. More epochs or a higher learning rate would likely help.
- **Cosine is far better than euclidean for ItemKNN** (about 9× the nDCG). The number of neighbors makes little difference.
- **Content-based AttributeItemKNN** has the highest coverage and Gini of any model except Random, but is only about 4× as accurate as Random.
- **All scores are low,** which is expected with a sparse dataset and a temporal split.

## Runtime (about 5 h 13 min total)

| Model | Time |
|---|---:|
| NonNegMF | 2 h 13 min |
| BPRMF (batch size 1) | 58 min |
| ItemKNN (6 configs) | 51 min |
| AttributeItemKNN (6 configs) | 45 min |
| UserKNN | 7 min |
| PMF | 7 min |
| PureSVD | 6 min |
| MF | 4 min |
| Random + MostPop | < 1 min |
