# Week 3: Recurrent Neural Networks (RNN/LSTM/GRU)

## Canonical Version

**RECOMMENDED PRESENTATION:** `presentations/20250929_1027_week03_rnn_template.pdf` (28 slides, 599KB)

This is the recommended template-based version for teaching Week 3. Uses the Madrid theme with lavender color scheme and follows the template_beamer_final.tex layout conventions. Assumes students have completed the Neural Network Primer prerequisite.

## Version Guide

### Primary Version (Use This)
- **File:** `20250929_1027_week03_rnn_template.pdf` (28 slides, 599KB)
- **Source:** `20250929_1027_week03_rnn_template.tex` (38KB)
- **Created:** 2025-09-29
- **Duration:** 60-90 minutes
- **Scope:** RNN-specific concepts only
- **Style:** Template-based with Madrid theme and lavender color scheme
- **Prerequisites:** Neural Network Primer or equivalent knowledge
- **Coverage:**
  - Why sequential processing matters
  - RNN architecture and equations
  - Vanishing gradient problem
  - LSTM architecture overview (high-level)
  - GRU comparison
  - Backpropagation Through Time (BPTT)
  - Code examples (NumPy + PyTorch)

### Previous Versions (Reference Only - in previous/ folder)
- **20250922_1242_week03_rnn_optimal.pdf (19 slides):** Earlier optimal version
- **20250922_1458_week03_rnn_with_foundations.pdf (34 slides):** DEPRECATED - Contains redundant NN basics that duplicate NN Primer

## Prerequisites

**Required Knowledge Before Week 3:**
1. Complete **Neural Network Primer** module (`NLP_slides/nn_primer/`)
2. Understand:
   - What is a neural network (biological → mathematical neurons)
   - Single neuron computation and weighted sums
   - Activation functions (sigmoid, tanh, ReLU, softmax)
   - Matrix multiplication and organization
   - Gradient descent intuition
   - Backpropagation basics

**If students lack prerequisites:** Direct them to NN Primer before starting Week 3.

## Module Scope

**Week 3 SHOULD Cover:**
- Sequential data processing challenges
- Recurrent connections and parameter sharing
- Hidden state as memory
- RNN equations and forward pass
- Vanishing/exploding gradient problem
- LSTM architecture overview (3 gates, cell state)
- GRU simplified alternative
- Backpropagation Through Time (BPTT) intuition
- When to use RNN vs LSTM vs GRU
- Practical implementation (PyTorch)

**Week 3 SHOULD NOT Cover:**
- Basic neural network fundamentals (→ NN Primer)
- Detailed backpropagation derivations (→ NN Primer)
- Deep LSTM gate mechanics (→ LSTM Primer for 32-slide deep dive)
- Matrix multiplication review (→ NN Primer)
- Generic gradient descent (→ NN Primer)

## Related Modules

- **Neural Network Primer** (`NLP_slides/nn_primer/`): Prerequisite foundational knowledge
- **LSTM Primer** (`NLP_slides/lstm_primer/`): Optional 32-slide deep dive for students wanting detailed LSTM understanding
- **Week 4 Seq2seq** (`NLP_slides/week04_seq2seq/`): Next topic building on RNNs

## Lab Materials

- **Main Lab:** `lab/week03_rnn_lab.ipynb` (28KB)
- **Enhanced Lab:** `lab/week03_rnn_lab_enhanced.ipynb` (698KB) - includes additional exercises and visualizations

## Handouts

- **Student Version:** `presentations/handouts/week03_rnn_handout_student.pdf`
- **Instructor Version:** `presentations/handouts/week03_rnn_handout_instructor.pdf`

## Figures

All 24 required figures are generated and located in `figures/`:
- RNN architecture diagrams
- Gradient flow visualizations
- LSTM gate mechanisms
- Comparison charts
- Code examples

## Teaching Notes

**Lecture Structure (90 minutes):**
1. Motivation (10 min): Why sequences matter
2. RNN Basics (20 min): Architecture, equations, examples
3. Vanishing Gradient (15 min): Problem statement with telephone game analogy
4. LSTM Solution (25 min): Gates, cell state, why it works
5. Practical Implementation (15 min): Code walkthrough
6. Summary & Next Steps (5 min): Seq2seq preview

**Common Student Questions:**
- "Why can't we just use feedforward networks?" → Fixed input size, no memory
- "What's the difference between LSTM and GRU?" → GRU has 2 gates instead of 3, simpler but similar performance
- "When do we use attention instead?" → Week 4 and 5 will cover this evolution

**Pedagogical Notes:**
- Use concrete analogies (telephone game for vanishing gradients)
- Show actual code early (students learn by doing)
- Reference LSTM Primer for students wanting deeper mathematics
- Connect to Week 2 embeddings (what goes into RNNs)
- Preview Week 4 seq2seq (why we need encoder-decoder)

## Last Updated

2025-09-30 - Updated canonical version to template-based presentation, cleaned up previous versions