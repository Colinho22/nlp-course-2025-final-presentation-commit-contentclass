# NLP Course 2025 - Module Scope Boundaries

## Purpose

This document clarifies what each module SHOULD and SHOULD NOT cover to prevent content redundancy and ensure prerequisite flow.

## Module Hierarchy

```
Week 0: First Day Lecture
    ↓
Neural Network Primer (prerequisite for Week 2+)
    ↓
Week 1: Statistical Language Models
Week 2: Neural Language Models & Embeddings
    ↓
Week 3: RNN/LSTM/GRU (with optional LSTM Primer)
    ↓
Week 4-12: Advanced Topics
```

## Detailed Module Boundaries

### Week 0: First Day Lecture
**SHOULD Cover:**
- Course motivation (ChatGPT, real applications)
- 12-week journey overview
- Assessment structure
- Prerequisites self-check
- Installation and setup

**SHOULD NOT Cover:**
- Any technical NLP content
- Neural network theory
- Programming

### Neural Network Primer (Prerequisite Module)
**SHOULD Cover:**
- What is a neural network (biological → mathematical)
- Single neuron computation (weights, biases, activation)
- Activation functions (sigmoid, tanh, ReLU, softmax)
- Matrix organization and multiplication
- Feedforward networks
- Gradient descent intuition
- Backpropagation basics
- Universal Approximation Theorem
- XOR problem solution

**SHOULD NOT Cover:**
- RNN-specific concepts
- LSTM gates
- Sequence modeling
- NLP applications
- Language modeling

**When to Assign:** Before Week 2 for students without NN background

### Week 1: Foundations & Statistical Language Models
**SHOULD Cover:**
- Probability basics (joint, conditional, Bayes)
- N-gram models (unigram, bigram, trigram)
- Markov assumption
- Maximum Likelihood Estimation
- Smoothing techniques (Laplace, Kneser-Ney)
- Perplexity evaluation
- Data sparsity problem

**SHOULD NOT Cover:**
- Neural networks
- Word embeddings
- Deep learning

### Week 2: Neural Language Models & Word Embeddings
**SHOULD Cover:**
- Neural language model architecture
- Word2Vec (CBOW, Skip-gram)
- Negative sampling
- GloVe (global co-occurrence)
- Embedding properties (analogy, similarity)
- Evaluation methods

**SHOULD NOT Cover:**
- Basic neural network theory (→ NN Primer)
- Backpropagation derivation (→ NN Primer)
- Sequential processing (→ Week 3)

**Prerequisites:** Neural Network Primer

### Week 3: RNN/LSTM/GRU
**SHOULD Cover:**
- Why sequential processing matters
- Recurrent connections and parameter sharing
- Hidden state as memory
- RNN equations and forward pass
- Vanishing/exploding gradient problem
- LSTM overview (3 gates, cell state, why it works)
- GRU simplified alternative
- BPTT intuition
- When to use RNN vs LSTM vs GRU

**SHOULD NOT Cover:**
- Basic neural network fundamentals (→ NN Primer)
- Activation function theory (→ NN Primer)
- Matrix multiplication review (→ NN Primer)
- Generic gradient descent (→ NN Primer)
- Detailed LSTM gate mechanics (→ LSTM Primer for deep dive)

**Prerequisites:** Neural Network Primer (REQUIRED)

**Canonical Version:** 19 slides (`20250922_1242_week03_rnn_optimal.pdf`)

### LSTM Primer (Optional Deep Dive)
**SHOULD Cover:**
- Autocomplete challenge motivation
- RNN baseline and limitations
- Memory problem in detail
- Vanishing gradient mathematical proof
- LSTM architecture deep dive (32 slides)
- Each gate mechanism in detail
- Cell state computation
- BPTT for LSTM
- Training tips and debugging
- Model comparison
- PyTorch implementation details

**SHOULD NOT Cover:**
- Basic NN theory (→ NN Primer)
- RNN introduction (→ Week 3)

**When to Assign:** Optional for students wanting deep LSTM understanding beyond Week 3's high-level overview

### Week 4: Sequence-to-Sequence
**SHOULD Cover:**
- Encoder-decoder architecture
- Attention mechanism
- Beam search
- Teacher forcing
- Applications (translation, summarization)

**SHOULD NOT Cover:**
- RNN basics (→ Week 3)
- Basic attention theory (detailed in Week 5)

**Prerequisites:** Week 3 RNN/LSTM

### Week 5: Transformers & Attention
**SHOULD Cover:**
- Self-attention mechanism
- Multi-head attention
- Positional encoding
- Transformer architecture
- Scaled dot-product attention

**SHOULD NOT Cover:**
- RNN architecture (→ Week 3)
- Basic attention from Week 4 (reference it)

**Prerequisites:** Week 4 Seq2Seq

### Weeks 6-12: Advanced Topics
Follow similar principle: Build on prerequisites, don't repeat them.

## Common Violations to Avoid

### ❌ BAD: Week 3 Re-teaching NN Basics
```
Week 3 Slide: "What is a Neural Network?"
- Biological neuron → Mathematical neuron
- Weighted sums and activations
- Backpropagation intuition
```
**Why Wrong:** Students should have learned this in NN Primer

**Fix:** Start Week 3 assuming NN knowledge, add "Prerequisites: NN Primer" note

### ❌ BAD: Week 5 Re-teaching RNN
```
Week 5 Slide: "RNN Architecture Review"
- Recurrent connections
- Hidden state equations
- BPTT explanation
```
**Why Wrong:** Wastes time, students did this in Week 3

**Fix:** Brief 1-slide "Recall: RNNs have sequential processing" then move on

### ❌ BAD: Multiple Modules Teaching Same Content
```
NN Primer: "Backpropagation"
Week 2: "Backpropagation for Word2Vec" (repeats basics)
Week 3: "Backpropagation Through Time" (repeats chain rule)
```
**Why Wrong:** Redundant, confusing

**Fix:** NN Primer teaches backprop once, later weeks reference and extend

## Enforcement Checklist

When creating/reviewing a module:

1. **List Prerequisites:** What should students know before this module?
2. **Check Redundancy:** Does any slide re-teach prerequisite content?
3. **Scope Boundaries:** Does content stay focused on module's unique contribution?
4. **Reference Pattern:** Do you reference (not repeat) prior modules?
5. **Length Check:** Is lecture appropriate length (20-25 slides for 90 min)?

## Version Control

When multiple versions exist:
1. Designate ONE canonical version in module README
2. Document why other versions are deprecated
3. Update COURSE_INDEX.md with canonical path
4. Add DEPRECATED.md explaining migration

## Last Updated

2025-09-28 - Initial module boundary documentation based on Week 3 review