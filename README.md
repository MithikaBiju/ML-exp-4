# ML-exp-4
# MLE vs. MAP Naive Bayes — 20 Newsgroups Text Classification

A from-scratch implementation of Multinomial Naive Bayes comparing **Maximum Likelihood Estimation (MLE)** against **Maximum a Posteriori (MAP)** estimation under several Dirichlet priors, applied to text classification on the 20 Newsgroups dataset.

## Overview

Plain MLE for Naive Bayes assigns zero probability to any word never seen in a class during training, which breaks the classifier on unseen test data (the "zero-frequency problem"). This notebook implements Naive Bayes manually with NumPy, then shows how Dirichlet priors (Lidstone/Laplace smoothing) fix this and improve accuracy — framed explicitly as MAP estimation with different prior strengths.

## Dataset

[`sklearn.datasets.fetch_20newsgroups`](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset), restricted to 4 categories:

- `alt.atheism`
- `soc.religion.christian`
- `comp.graphics`
- `sci.med`

Headers, footers, and quotes stripped to avoid leakage. Text vectorized with `CountVectorizer` (English stop words removed, top 5000 features by frequency).

- Train shape: `(2257, 5000)`
- Test shape: `(1502, 5000)`

## Methodology

1. Vectorize documents into word-count vectors
2. Compute class priors `P(c)` and aggregate per-class word counts `N_c`
3. Implement a shared `predict_naive_bayes` function that classifies using log-probabilities (to avoid numerical underflow), with `-1e9` substituted for `log(0)`
4. **MLE estimate:** `θ = N_c / N_c.sum()` — pure relative frequency, no smoothing
5. **MAP estimates:** four different Dirichlet priors added to the word counts before normalizing:
   - Lidstone smoothing (α = 1.01)
   - Laplace smoothing (α = 2.0)
   - Strong Dirichlet prior (α = 10.0)
   - Non-uniform empirical prior (based on overall corpus word frequency)

## Results

| Model | Test Accuracy |
|---|---|
| MLE (no smoothing) | 0.7710 |
| MAP — Lidstone (α = 1.01) | 0.8103 |
| MAP — Laplace (α = 2.0) | **0.8182** |
| MAP — Strong Dirichlet (α = 10.0) | 0.7863 |
| MAP — Non-Uniform (empirical) | 0.8063 |

**Key takeaway:** even a very light Dirichlet prior (α = 1.01) improves accuracy by ~4 points over raw MLE by eliminating zero-probability words. Laplace smoothing (α = 2.0) gives the best result here. Pushing the prior too strong (α = 10.0) starts to hurt performance by over-flattening the class-conditional word distributions — showing that MAP estimation is a bias-variance trade-off, not a free lunch.

## How to run

```bash
pip install numpy scikit-learn
```

Open and run the notebook top to bottom in Jupyter or Google Colab. The dataset downloads automatically via scikit-learn on first run (internet connection required for that step).

## Requirements

- Python 3.x
- numpy, scikit-learn
