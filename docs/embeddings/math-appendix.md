---
layout: module
title: "Mathematical Foundations"
subtitle: "Full derivations for word embedding algorithms"
---

This appendix provides complete mathematical derivations for the embedding algorithms covered in the main module. Prerequisites include calculus, linear algebra, and basic probability theory.

---

## A.1 Skip-gram Objective Function

### The Training Objective

Given a corpus of words $w_1, w_2, ..., w_T$, the Skip-gram model maximizes:

$$J(\theta) = \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t; \theta)$$

Where:
- $T$ = total number of words in corpus
- $c$ = context window size (typically 5-10)
- $\theta$ = model parameters (embedding matrices)

### Probability Model

For center word $w_c$ and context word $w_o$:

$$P(w_o | w_c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}$$

Where:
- $v_c \in \mathbb{R}^d$ = center word embedding
- $u_o \in \mathbb{R}^d$ = context word embedding
- $V$ = vocabulary

### Computational Problem

The denominator requires summing over the entire vocabulary:

$$\sum_{w \in V} \exp(u_w^T v_c)$$

For $|V| = 50,000$, this is prohibitively expensive!

---

## A.2 Negative Sampling

### The Solution

Instead of computing the full softmax, we reformulate as binary classification:

**Positive example:** $(w_c, w_o)$ from actual corpus → label = 1
**Negative examples:** $(w_c, w_k)$ with random $w_k$ → label = 0

### New Objective

$$J(\theta) = \log \sigma(u_o^T v_c) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)} [\log \sigma(-u_k^T v_c)]$$

Where:
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function
- $K$ = number of negative samples (typically 5-20)
- $P_n(w)$ = noise distribution

### Noise Distribution

Mikolov et al. found that a modified unigram distribution works best:

$$P_n(w) = \frac{f(w)^{3/4}}{\sum_{w'} f(w')^{3/4}}$$

The $3/4$ power slightly smooths the distribution, giving rare words more chance of being sampled.

### Gradient Derivation

For the positive term:

$$\frac{\partial}{\partial v_c} \log \sigma(u_o^T v_c) = (1 - \sigma(u_o^T v_c)) \cdot u_o$$

For negative terms:

$$\frac{\partial}{\partial v_c} \log \sigma(-u_k^T v_c) = -\sigma(u_k^T v_c) \cdot u_k$$

---

## A.3 GloVe: Global Vectors

### Co-occurrence Matrix

Define $X_{ij}$ as the count of word $j$ appearing in the context of word $i$:

$$X_i = \sum_k X_{ik}$$ (total count for word $i$)

$$P_{ij} = P(j|i) = \frac{X_{ij}}{X_i}$$ (co-occurrence probability)

### The GloVe Insight

Consider the ratio of co-occurrence probabilities:

$$\frac{P(k|i)}{P(k|j)}$$

| Relationship | ice | steam |
|--------------|-----|-------|
| $P(k\|ice)$ | large | small |
| $P(k\|steam)$ | small | large |
| Ratio | large | small |

**Key observation:** The ratio distinguishes relevant from irrelevant words!

### Objective Function

GloVe seeks word vectors such that:

$$w_i^T w_j + b_i + b_j = \log X_{ij}$$

With weighted least squares objective:

$$J = \sum_{i,j=1}^{V} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

### Weighting Function

$$f(x) = \begin{cases}
(x/x_{max})^{\alpha} & \text{if } x < x_{max} \\
1 & \text{otherwise}
\end{cases}$$

With $\alpha = 0.75$ and $x_{max} = 100$ (empirically chosen).

This prevents:
- Rare co-occurrences from being overweighted
- Common co-occurrences from dominating

---

## A.4 Cosine Similarity

### Definition

$$\cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

### Properties

| Value | Interpretation |
|-------|----------------|
| 1 | Identical direction |
| 0 | Orthogonal (no relationship) |
| -1 | Opposite direction |

### Why Cosine Over Euclidean?

**Euclidean distance:** $\|A - B\| = \sqrt{\sum_i (A_i - B_i)^2}$

Problem: Sensitive to vector magnitude!

```
A = [1, 0]      # Unit vector
B = [100, 0]   # Same direction, different magnitude

Euclidean: ||A - B|| = 99  # Seems very different
Cosine: cos(A, B) = 1      # Correctly identifies same direction
```

For text, we care about direction (meaning) not magnitude (frequency).

---

## A.5 Attention Mechanism Preview

Modern contextual embeddings use attention. The basic self-attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ = queries (what we're looking for)
- $K$ = keys (what we match against)
- $V$ = values (what we retrieve)
- $d_k$ = dimension of keys

### Why $\sqrt{d_k}$?

For large $d_k$, dot products can become very large:

$$\mathbb{E}[q \cdot k] = 0, \quad \text{Var}[q \cdot k] = d_k$$

Scaling by $\sqrt{d_k}$ keeps gradients stable.

---

## A.6 BERT Pre-training Objectives

### Masked Language Modeling (MLM)

Given a sequence with 15% of tokens masked:

$$\mathcal{L}_{MLM} = -\sum_{i \in M} \log P(w_i | \mathbf{w}_{\backslash M})$$

Where $M$ is the set of masked positions.

### Next Sentence Prediction (NSP)

Binary classification for sentence pairs:

$$\mathcal{L}_{NSP} = -\log P(y | [CLS], A, [SEP], B, [SEP])$$

Where $y \in \{IsNext, NotNext\}$.

### Combined Objective

$$\mathcal{L} = \mathcal{L}_{MLM} + \mathcal{L}_{NSP}$$

---

## A.7 Dimensionality and the Curse

### The Volume Concentration Problem

In high dimensions, most volume is near the surface of a hypersphere:

$$\lim_{d \to \infty} \frac{V_d(r - \epsilon)}{V_d(r)} = \lim_{d \to \infty} \left(1 - \frac{\epsilon}{r}\right)^d = 0$$

For any $\epsilon > 0$, almost all volume is in a thin shell!

### Implications for Embeddings

1. **Distances concentrate:** All points become approximately equidistant
2. **Nearest neighbor degrades:** Distinction between nearest and random neighbors vanishes
3. **Need careful dimensionality:** Too high is as bad as too low

### Practical Guidance

| Task | Recommended Dimension |
|------|----------------------|
| Small corpus (<1M words) | 50-100 |
| Medium corpus (1M-100M) | 100-300 |
| Large corpus (>100M) | 300-500 |

---

## References

1. Mikolov, T., et al. (2013). *Efficient Estimation of Word Representations in Vector Space*. ICLR Workshop. [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)

2. Mikolov, T., et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS. [arXiv:1310.4546](https://arxiv.org/abs/1310.4546)

3. Pennington, J., Socher, R., & Manning, C. (2014). *GloVe: Global Vectors for Word Representation*. EMNLP. [PDF](https://aclanthology.org/D14-1162.pdf)

4. Bojanowski, P., et al. (2017). *Enriching Word Vectors with Subword Information*. TACL. [arXiv:1607.04606](https://arxiv.org/abs/1607.04606)

5. Peters, M. E., et al. (2018). *Deep Contextualized Word Representations*. NAACL. [arXiv:1802.05365](https://arxiv.org/abs/1802.05365)

6. Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*. NAACL. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
