# Hate Speech Detection via Sentiment Knowledge Sharing

A multi-task learning model for hate speech detection that simultaneously learns hate speech detection and sentiment analysis, sharing knowledge between the two tasks via a **Mixture-of-Experts (MMoE)** architecture.

---

## Overview

Most hate speech detection models pattern-match on surface-level words — they flag a slur regardless of context. This model takes a different approach: since hate speech is strongly correlated with negative sentiment, training jointly on sentiment analysis teaches the model emotional tone and context, not just word patterns.

**Core idea:** A model that understands *how* something is said, not just *what* words appear.

---

## Architecture

```
Input Tweet
    ↓
GloVe Embeddings (300d) + Category Embeddings (100d)
    ↓  [128 × 400 per tweet]
MMoE Layer — 4 Expert Networks in parallel
    Each expert: Multi-head Attention (4 heads) → FFN → Dual Pooling → 800d
    ↓
Task-specific Gating
    Gate learns which experts to trust per task per tweet
    ↓
Two Task Heads
    Hate Speech Head  → sigmoid → probability
    Sentiment Head    → sigmoid → probability
```

**Three key components:**

**1. Category Embeddings** — each token is checked against a derogatory word dictionary. Toxic tokens get a different learned embedding than neutral tokens. This injects explicit lexical knowledge into the input.

**2. Mixture of Experts (MMoE)** — instead of one shared hidden layer between tasks (which causes negative transfer), multiple expert networks run in parallel. Each task has its own gate that learns to weight the experts differently — hate speech and sentiment can specialise without interfering.

**3. Multi-task Learning** — sentiment analysis is an auxiliary task. Its gradients flow back through the shared experts, teaching them emotional context that pure hate speech labels alone cannot provide.

---

## Results

| Dataset | Accuracy | Weighted F1 |
|---------|----------|-------------|
| SemEval-2019 Task 5 | 60.11% | 60.36% |

Trained on: SemEval-2019 Task 5 (English) + Davidson hate speech dataset  
Auxiliary task: Kaggle Twitter Sentiment dataset

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/isha152002/HateSpeechDetection_SKS.git
cd HateSpeechDetection_SKS
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download data

Place the following files in `data/raw/`:

| File | Source |
|------|--------|
| `glove.840B.300d.txt` | [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) |
| `semeval_train.parquet` | [HuggingFace — valeriobasile/HatEval](https://huggingface.co/datasets/valeriobasile/HatEval) |
| `semeval_val.parquet` | Same as above |
| `semeval_test.parquet` | Same as above |
| `davidson.csv` | [Davidson et al. GitHub](https://github.com/t-davidson/hate-speech-and-offensive-language) |
| `kaggle_sentiment.csv` | [Kaggle Twitter Sentiment](https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech) |
| `derogatory_dictionary.csv` | Davidson lexicon folder (same repo as davidson.csv) |

### 4. Run training

```bash
python main.py
```

---

## Hyperparameters

All hyperparameters are centralised in `config.py`.

---


