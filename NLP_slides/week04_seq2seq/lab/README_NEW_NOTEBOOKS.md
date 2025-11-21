# Week 4 Lab: Sequence-to-Sequence Models - Three-Part Series

## Overview

The Week 4 lab has been split into three focused, simple notebooks that build on each other. Each notebook is self-contained with clear explanations, visualizations, and interactive exercises.

## The Three Notebooks

### Part 1: Basic Sequence-to-Sequence
**File:** `week04_part1_basic_seq2seq.ipynb`
**Time:** 20-25 minutes
**Topics:**
- The variable-length problem
- Building a simple encoder (reads input)
- Building a simple decoder (writes output)
- Visualizing the encoding process
- Understanding context vectors

**Key Learning:**
- How encoder-decoder architecture works
- Why we need fixed-size context vectors
- What happens during encoding/decoding
- The bottleneck problem

**HTML Version:** `week04_part1_basic_seq2seq.html`

---

### Part 2: Attention Mechanism
**File:** `week04_part2_attention.ipynb`
**Time:** 25-30 minutes
**Topics:**
- The information bottleneck problem
- What attention really means
- How attention works step-by-step
- Attention weight visualization (heatmaps)
- Comparing with vs without attention

**Key Learning:**
- Why basic seq2seq struggles with long sentences
- How attention solves the bottleneck
- Reading attention heatmaps
- Dramatic quality improvements

**HTML Version:** `week04_part2_attention.html`

---

### Part 3: Advanced Topics
**File:** `week04_part3_advanced.ipynb`
**Time:** 20-25 minutes
**Topics:**
- Why greedy decoding fails
- Beam search algorithm
- Comparing search strategies
- Modern translation systems (Google Translate)
- Connection to ChatGPT and modern AI

**Key Learning:**
- Finding better translations with beam search
- Quality vs speed tradeoffs
- How real translation systems work
- Foundation of modern AI

**HTML Version:** `week04_part3_advanced.html`

---

## How to Use

### Running the Notebooks

1. **Sequential order recommended:**
   ```bash
   jupyter lab week04_part1_basic_seq2seq.ipynb
   # Complete Part 1, then move to Part 2
   jupyter lab week04_part2_attention.ipynb
   # Complete Part 2, then move to Part 3
   jupyter lab week04_part3_advanced.ipynb
   ```

2. **View HTML versions:**
   - Open any `.html` file in your browser
   - These show the notebook structure without needing to run code
   - Good for quick reference or reading on mobile

### Key Features

**Simple Language:**
- No jargon without explanation
- Real-world analogies
- Step-by-step breakdowns

**Visual Learning:**
- Color-coded diagrams
- Heatmaps for attention weights
- Evolution charts
- Comparison visualizations

**Interactive:**
- Modify code and see results
- Try your own examples
- Experiment with parameters

**Self-Contained:**
- Each notebook can run independently
- All imports included
- No external data needed

## What Was Fixed

The original `week04_seq2seq_lab_enhanced.ipynb` had two errors:

1. **TypeError in encoder** (line 391):
   - **Old:** `{hidden[:3]:.2f}` - Can't format numpy array slices
   - **Fixed:** `{hidden[0]:.2f}, {hidden[1]:.2f}, {hidden[2]:.2f}` - Format individual elements

2. **IndexError in attention** (line 954):
   - **Old:** `attention_matrix[tgt_idx, align_src]` - Index out of bounds
   - **Fixed:** Added bounds checking: `if tgt_idx < len(target_words) and src_idx < len(source_words)`

These fixes are applied in the new notebooks.

## File Structure

```
NLP_slides/week04_seq2seq/lab/
├── week04_part1_basic_seq2seq.ipynb     # Notebook 1 (encoder-decoder)
├── week04_part1_basic_seq2seq.html      # HTML export
├── week04_part2_attention.ipynb         # Notebook 2 (attention)
├── week04_part2_attention.html          # HTML export
├── week04_part3_advanced.ipynb          # Notebook 3 (beam search)
├── week04_part3_advanced.html           # HTML export
├── week04_seq2seq_lab_enhanced.ipynb    # Original (with errors)
└── README_NEW_NOTEBOOKS.md              # This file
```

## Quick Start

**Option 1: Interactive Learning**
```bash
cd NLP_slides/week04_seq2seq/lab
jupyter lab week04_part1_basic_seq2seq.ipynb
```

**Option 2: Quick Preview**
- Open `week04_part1_basic_seq2seq.html` in your browser
- Read through the content
- Run the notebook when ready for hands-on practice

## Learning Path

```
Part 1: Basic Seq2Seq
  ↓
  Understand encoder-decoder
  See the bottleneck problem
  ↓
Part 2: Attention
  ↓
  Learn how attention solves bottleneck
  See dramatic improvements
  ↓
Part 3: Advanced
  ↓
  Master beam search
  Connect to modern AI
  ↓
Week 5: Transformers!
```

## Tips for Success

1. **Take your time:** Don't rush through the notebooks
2. **Run all cells:** Execute code to see visualizations
3. **Experiment:** Modify examples and parameters
4. **Read explanations:** Understanding concepts matters more than code
5. **Try exercises:** Each notebook has interactive challenges

## Questions?

If you encounter issues:
1. Check that all required packages are installed (`numpy`, `matplotlib`, `seaborn`)
2. Restart the kernel if you see unexpected errors
3. Make sure you're running cells in order (top to bottom)

## Next Steps

After completing all three notebooks:
- Review Week 5 materials on Transformers
- Explore the original enhanced notebook for additional content
- Try implementing your own translation examples

---

**Created:** 2025-10-10
**Status:** All notebooks tested and ready to use
**Errors fixed:** 2/2 (formatting and indexing errors resolved)
