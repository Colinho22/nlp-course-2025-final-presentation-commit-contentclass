---
layout: module
title: "Word Embeddings"
subtitle: "From One-Hot Encoding to Dense Vector Representations"
hero: true
hero_actions:
  - text: "View Charts"
    url: "/embeddings/gallery/"
    style: "secondary"
  - text: "Math Appendix"
    url: "/embeddings/math-appendix/"
    style: "secondary"
slides: "/assets/slides/embeddings/embeddings_enhanced.pdf"
slides_title: "Embeddings Presentation"
slides_thumbnail: "/assets/slides/embeddings/thumbnail.png"
notebooks:
  - title: "Discovery Notebook"
    description: "Pre-class activities and guided exploration"
    path: "embeddings/handouts/discovery_notebook.ipynb"
    html: "/assets/notebooks/embeddings/discovery_notebook.html"
  - title: "Word Embeddings Lab"
    description: "Hands-on Word2Vec implementation"
    path: "NLP_slides/week02_neural_lm/lab/week02_word_embeddings_lab.ipynb"
    html: "/assets/notebooks/embeddings/word_embeddings_lab.html"
objectives:
  - "Explain why one-hot encoding fails to capture semantic relationships"
  - "Describe how dense embeddings represent meaning as vectors"
  - "Understand the Word2Vec training objectives (Skip-gram and CBOW)"
  - "Apply vector arithmetic to solve word analogies"
  - "Calculate word similarity using cosine similarity"
  - "Explain the distributional hypothesis and its implications"
bibliography:
  - key: mikolov2013
    authors: "Mikolov, T., Chen, K., Corrado, G., & Dean, J."
    year: 2013
    title: "Efficient Estimation of Word Representations in Vector Space"
    venue: "ICLR Workshop"
    url: "https://arxiv.org/abs/1301.3781"
  - key: pennington2014
    authors: "Pennington, J., Socher, R., & Manning, C."
    year: 2014
    title: "GloVe: Global Vectors for Word Representation"
    venue: "EMNLP"
    url: "https://aclanthology.org/D14-1162/"
  - key: peters2018
    authors: "Peters, M. E., et al."
    year: 2018
    title: "Deep Contextualized Word Representations"
    venue: "NAACL"
    url: "https://arxiv.org/abs/1802.05365"
related:
  - title: "Chart Gallery"
    description: "33 visualizations covering all embedding concepts"
    url: "/embeddings/gallery/"
    action: "Browse Charts"
  - title: "Math Appendix"
    description: "Full mathematical derivations and proofs"
    url: "/embeddings/math-appendix/"
    action: "See Derivations"
---

## The Fundamental Problem

How do we represent meaning mathematically? This is the central challenge of computational linguistics.

**Human Understanding** involves rich semantic connections - when we think of "cat," we immediately associate it with "animal," "kitten," "feline," "pet," and "meow." These connections form a complex web of meaning.

**Computer's Dilemma:**
- Words are just strings: "cat" = ['c','a','t']
- No inherent meaning
- No similarity measure
- Can't do math on strings!

**What We Need:**

> Convert: "cat" → [0.2, -0.4, 0.7, ...]
> Such that: similar words → similar vectors
{: .callout .callout-insight}

**Goal:** Capture meaning in numbers so computers can process language.

---

## One-Hot Encoding: The Starting Point

The simplest approach to representing words numerically - but fundamentally flawed.

### How One-Hot Works

| Word | Vector |
|------|--------|
| cat  | [1, 0, 0, 0, 0] |
| dog  | [0, 1, 0, 0, 0] |
| mat  | [0, 0, 1, 0, 0] |
| sat  | [0, 0, 0, 1, 0] |
| hat  | [0, 0, 0, 0, 1] |

Each word gets exactly one "1" in its vector, with all other positions being "0".

### Critical Problems

**1. No Similarity:**

$$\text{similarity}(\text{cat}, \text{kitten}) = 0$$
$$\text{similarity}(\text{cat}, \text{computer}) = 0$$

Both are equally dissimilar! One-hot encoding treats all words as orthogonal.

**2. Huge Dimensions:**
- English: 170,000+ words
- Each word = 170,000-dimensional vector
- 99.999% zeros (extremely sparse!)

**3. No Relationships:**

$$\text{cat} + \text{kitten} = [1,0,0...] + [0,1,0...] = [1,1,0...]$$

The result is meaningless - we can't combine word meanings.

> **Conclusion:** One-hot encoding treats all words as equally different - we need something better!
{: .callout .callout-warning}

---

## Dense Embeddings: The Solution

From sparse to dense - capturing meaning in vectors.

### The Transformation

| Property | One-Hot | Dense |
|----------|---------|-------|
| Dimensions | 50,000+ | 100-300 |
| Sparsity | 99.998% zeros | All values meaningful |
| Size | Huge | 100x smaller |
| Semantics | None | Captured |

**Example Dense Vector:**

$$\text{cat} = [0.21, -0.43, 0.67, 0.15, -0.22, ...]$$

Each dimension captures some aspect of meaning learned from data.

### Benefits of Dense Embeddings

1. **100x smaller** - dramatically reduced memory and computation
2. **Captures semantics** - similar words have similar vectors
3. **Enables arithmetic** - meaningful operations on word vectors
4. **Learned from data** - patterns emerge automatically

---

## The Embedding Space

Where words live as vectors.

### Key Properties

**1. Clustering:** Similar words group together in the vector space.
- Animals cluster together: cat, dog, bird, fish
- Food words cluster together: apple, bread, milk
- Action words cluster together: run, jump, walk

**2. Distance = Similarity:**
- cat ↔ dog: close (high similarity)
- cat ↔ run: far (low similarity)

**3. Directions = Relations:**
The most remarkable property - relationships between words correspond to consistent directions in the space.

| Relationship | Examples |
|--------------|----------|
| Gender | man→woman, king→queen |
| Country-Capital | France→Paris, Japan→Tokyo |
| Tense | walk→walked, run→ran |

> **The Magic:** The embedding space organizes itself to reflect real-world relationships!
{: .callout .callout-insight}

---

## Word2Vec: Learning Embeddings

How do we learn these meaningful vectors?

### The Distributional Hypothesis

> "You shall know a word by the company it keeps"
> — J.R. Firth (1957)

Words that appear in similar contexts tend to have similar meanings. Word2Vec exploits this insight.

### Two Training Approaches

**1. Skip-gram:** Given a word, predict its context

```
Input: "cat"
Predict: "the", "sat", "on", "mat"
```

The model learns: what words typically appear near "cat"?

**2. CBOW (Continuous Bag of Words):** Given context, predict the word

```
Input: "the", "___", "sat"
Predict: "cat"
```

The model learns: what word fits this context?

### The Training Objective

$$\max \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)$$

Where:
- $T$ = total words in corpus
- $c$ = context window size
- $w_t$ = center word
- $w_{t+j}$ = context word

We maximize the probability of seeing actual context words given each center word.

---

## Vector Arithmetic: The Surprising Discovery

Embeddings capture analogies through simple arithmetic!

### Famous Examples

**King - Man + Woman = Queen**

This works because:
- "king" and "queen" share a "royalty" component
- "man" and "woman" differ by a "gender" component
- Subtracting "man" removes male, adding "woman" adds female

**More Analogies:**

| Analogy | Equation |
|---------|----------|
| Capitals | Paris - France + Germany = Berlin |
| Comparatives | bigger - big + small = smaller |
| Tense | walked - walk + run = ran |

### Why Does This Work?

Relationships are encoded as **directions** in the vector space:
- Same relationship = same direction
- Linear structure emerges naturally from training

> **Remarkable:** These patterns were never explicitly programmed - they emerge from the data!
{: .callout .callout-insight}

---

## Measuring Word Similarity

How similar are two words?

### Cosine Similarity

$$\text{similarity}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|} = \cos(\theta)$$

The cosine of the angle between two vectors:
- **1.0** = identical direction (maximum similarity)
- **0.0** = perpendicular (no relationship)
- **-1.0** = opposite direction (antonyms, sometimes)

### Example Similarity Matrix

|       | cat  | dog  | car  | truck |
|-------|------|------|------|-------|
| cat   | 1.00 | 0.80 | 0.10 | 0.05  |
| dog   | 0.80 | 1.00 | 0.15 | 0.10  |
| car   | 0.10 | 0.15 | 1.00 | 0.85  |
| truck | 0.05 | 0.10 | 0.85 | 1.00  |

**Pattern:** Animals cluster together, vehicles cluster together!

---

## Beyond Word2Vec

### GloVe: Global Vectors

While Word2Vec uses local context windows, GloVe [(Pennington et al., 2014)](#ref-pennington2014) leverages global co-occurrence statistics:

$$J = \sum_{i,j=1}^{V} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

Where $X_{ij}$ counts how often words $i$ and $j$ co-occur in the corpus.

### FastText: Subword Embeddings

FastText extends Word2Vec by representing words as bags of character n-grams:

- "where" → {"<wh", "whe", "her", "ere", "re>"}

**Benefits:**
- Handles out-of-vocabulary words
- Captures morphological patterns
- Better for morphologically rich languages

### Contextual Embeddings

Static embeddings give each word a single vector. But words have different meanings in context:

- "bank" (financial) vs. "bank" (river)
- "apple" (fruit) vs. "Apple" (company)

**ELMo** [(Peters et al., 2018)](#ref-peters2018) and **BERT** provide context-dependent representations where the same word gets different vectors based on surrounding context.

---

## Practical Considerations

### Choosing Embedding Dimensions

| Dimension | Trade-off |
|-----------|-----------|
| 50-100 | Fast, less expressive |
| 200-300 | Good balance (most common) |
| 500+ | More expressive, slower, risk of overfitting |

### Pre-trained vs. Training Your Own

**Use Pre-trained When:**
- Limited domain-specific data
- General NLP tasks
- Quick prototyping

**Train Your Own When:**
- Large domain-specific corpus
- Specialized vocabulary
- Need task-specific semantics

### Common Pitfalls

1. **Not normalizing vectors** before computing similarity
2. **Ignoring out-of-vocabulary words** - use subword methods
3. **Assuming all relationships are linear** - some aren't
4. **Using embeddings without fine-tuning** for specific tasks

---

## Summary

### Key Takeaways

1. **One-hot encoding fails** because it treats all words as equally different
2. **Dense embeddings capture semantics** by learning from co-occurrence patterns
3. **Word2Vec** learns embeddings by predicting context (Skip-gram) or words (CBOW)
4. **Vector arithmetic** reveals that relationships are directions in embedding space
5. **Cosine similarity** measures semantic relatedness between word vectors
6. **Contextual embeddings** (ELMo, BERT) provide word-in-context representations

### What's Next?

- **Math Appendix**: Full derivations of Skip-gram objective, negative sampling, and GloVe
- **Chart Gallery**: 33 visualizations covering all concepts
- **Lab Notebooks**: Hands-on implementation with Python

---

*For mathematical details including the full Skip-gram derivation, negative sampling optimization, and GloVe matrix factorization, see the [Math Appendix]({{ '/embeddings/math-appendix/' | relative_url }}).*
