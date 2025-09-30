# Week 4: Sequence-to-Sequence Models

## Recommended Version

**CANONICAL VERSION:** `20250928_1510_week04_seq2seq_journey_complete.tex` (28 slides)

This version uses the "Compression Journey" narrative with complete pedagogical arc and assumes ZERO NLP pre-knowledge.

## Version Guide

### Primary Version (Use This)
- **File:** `20250928_1510_week04_seq2seq_journey_complete.tex` (28 slides, 286KB)
- **Duration:** 90 minutes
- **Approach:** Zero pre-knowledge, unified narrative with full dramatic arc
- **Structure:** Four-act journey with hope → disappointment → breakthrough pattern
- **Pedagogy:** Every technical term motivated before use, all narrative beats present
- **Coverage:**
  - Why embeddings (from bytes to meaning)
  - What hidden states are (evolving understanding)
  - Encoder-decoder architecture (reading vs writing)
  - Information bottleneck (quantified with ratios)
  - Attention mechanism (from human behavior to mathematics)
  - Dot product similarity (geometric intuition first)
  - Connection to transformers and modern LLMs

### Alternative Version (Reference)
- **File:** `week04_seq2seq_bsc_enhanced_final.tex` (29 slides)
- **Status:** Original BSc version with checkpoint pedagogy
- **Differences:** Linear problem-solution structure, assumes some NLP background
- **Use case:** Reference for comparison, checkpoint quiz examples

## Key Pedagogical Principles

**Zero Pre-Knowledge Implementation:**
1. **Embeddings:** Motivated from "computers see bytes, need meaning"
2. **Hidden states:** Built from human reading comprehension analogy
3. **Context vectors:** Introduced as compression naturally before naming
4. **Dot product:** Explained with 2D geometric visualization first
5. **Attention:** Derived from observing human translation behavior

**Narrative Structure:**
- Unified "compression journey" metaphor throughout
- Not disconnected problem-solution cycles
- Clear causal flow: fundamental challenge → solutions → limitations → breakthrough

**Concrete-to-Abstract Progression:**
- Start with bytes and characters
- Build to embeddings (numerical meaning)
- Progress to vectors and operations
- Culminate in attention mechanism
- Connect to modern AI (ChatGPT, Claude)

## Teaching Notes

**Lecture Structure (90 minutes):**
1. Act 1: Compression Challenge (15 min) - Build all foundational concepts
2. Act 2: Encoder-Decoder (15 min) - Introduce architecture and quantify bottleneck
3. Act 3: Attention Revolution (30 min) - From human insight to implementation
4. Act 4: Synthesis (15 min) - Modern connections and evolution
5. Q&A and Discussion (15 min)

**Critical Slides:**
- Slide 4: First principles of embeddings (computer's dilemma)
- Slide 5: Hidden state as "evolving understanding"
- Slide 7: Quantifying compression (information theory)
- Slide 13: Dot product geometric intuition (2D before 256D)
- Slide 16: Human translation insight (selective focus)
- Slide 20: Evolution timeline (Seq2seq → Transformers → ChatGPT)

**Common Student Questions:**
- "Why can't we just use bigger vectors?" → Information capacity limits, computational cost
- "How does the model learn attention weights?" → Backpropagation trains scoring function
- "Is this still used today?" → Foundation of all modern LLMs, attention is universal

## Prerequisites

**Required from Week 3:**
- Basic understanding that neural networks process numbers
- Concept of sequential processing (RNN idea)
- Backpropagation intuition

**NOT Required:**
- Deep RNN/LSTM knowledge (we motivate everything fresh)
- Mathematical background in information theory
- Prior NLP experience

## Related Materials

- **Lab:** `lab/week04_seq2seq_lab.ipynb` - Implement seq2seq with attention
- **Figures:** All in `figures/` directory (attention heatmap, architecture diagrams)
- **Next Week:** Week 5 Transformers (remove RNNs, keep only attention)

## Last Updated

2025-09-28 - New canonical version created with zero pre-knowledge approach