# NLP Course 2025 - Complete Course Index

## Course Overview
Comprehensive 12-week Natural Language Processing course covering statistical foundations through modern transformer architectures. Includes presentations, lab notebooks, handouts, and supplementary modules.

## Course Structure

### Week 1: Foundations & Statistical Language Models
**Location:** `NLP_slides/week01_foundations/`
- **Main Presentation:** `Foundations_and_Statistical_Language_Modelling.pdf`
- **Topics:** N-grams, Markov models, probability theory, perplexity
- **Lab:** No lab (introductory week)
- **Handouts:** Available in `presentations/handouts/`

### Week 2: Neural Language Models & Word Embeddings
**Location:** `NLP_slides/week02_neural_lm/`
- **Main Presentation:** `20250120_0901_week02_compiled.pdf`
- **Topics:** Word2Vec (CBOW, Skip-gram), GloVe, neural LM architectures
- **Lab:** `lab/week02_word_embeddings_lab.ipynb` - Implement and visualize embeddings
- **Handouts:** Student/instructor versions for Word2Vec exercises

### Week 3: RNN/LSTM/GRU
**Location:** `NLP_slides/week03_rnn/`
- **Main Presentation:** `presentations/20250922_1242_week03_rnn_optimal.pdf` (19 slides, CANONICAL)
- **Topics:** Sequential models, vanishing gradients, gating mechanisms, BPTT
- **Lab:** `lab/week03_rnn_lab.ipynb` - RNN and LSTM implementation
- **PREREQUISITES (REQUIRED):**
  - **Must complete Neural Network Primer** (`nn_primer/`) OR have equivalent knowledge
  - Required topics: neurons, activation functions, backpropagation, gradient descent, matrix operations
- **Optional Deep Dive:** LSTM Primer (32 slides) for detailed gate mechanics

### Week 4: Sequence-to-Sequence Models
**Location:** `NLP_slides/week04_seq2seq/`
- **Main Presentation:** Multiple versions in `presentations/` (BSc, enhanced)
- **Topics:** Encoder-decoder architecture, attention mechanism, beam search
- **Lab:** `lab/week04_seq2seq_lab.ipynb` - Machine translation with attention
- **Key Concepts:** Teacher forcing, attention alignment, decoder strategies

### Week 5: Transformers & Attention
**Location:** `NLP_slides/week05_transformers/`
- **Main Presentation:** `presentations/20250924_1540_week05_transformers_comprehensive.pdf`
- **Topics:** Self-attention, multi-head attention, positional encoding, architecture
- **Lab:** `lab/week05_transformer_lab.ipynb` - Build transformer from scratch
- **Milestone:** Foundation for all modern NLP models

### Week 6: Pre-trained Models (BERT, GPT)
**Location:** `NLP_slides/week06_pretrained/`
- **Main Presentation:** Available in `presentations/`
- **Topics:** Transfer learning, masked language modeling, autoregressive models
- **Lab:** `lab/week06_bert_finetuning.ipynb` - Fine-tune BERT for classification
- **Key Models:** BERT, GPT-1/2, RoBERTa, ELECTRA

### Week 7: Advanced Transformers
**Location:** `NLP_slides/week07_advanced/`
- **Main Presentation:** `presentations/20250921_1729_week07_optimal_template.pdf`
- **Topics:** T5, GPT-3, mixture of experts, scaling laws, emergent abilities
- **Lab:** `lab/week07_advanced_transformers_lab.ipynb`
- **Focus:** Modern large language models and their properties

### Week 8: Tokenization & Vocabulary
**Location:** `NLP_slides/week08_tokenization/`
- **Main Presentation:** `presentations/20250923_2110_week08_tokenization_optimal.pdf`
- **Topics:** BPE, WordPiece, SentencePiece, vocabulary design
- **Lab:** `lab/week08_tokenization_lab.ipynb` - Implement BPE tokenizer
- **Practical:** Critical for deployment and cross-lingual models

### Week 9: Decoding Strategies
**Location:** `NLP_slides/week09_decoding/`
- **Main Presentation:** `presentations/20250923_2110_week09_decoding_optimal.pdf`
- **Topics:** Greedy search, beam search, sampling methods, nucleus sampling
- **Lab:** `lab/week09_decoding_lab.ipynb` - Compare decoding strategies
- **Applications:** Text generation quality and diversity control

### Week 10: Fine-tuning & Prompt Engineering
**Location:** `NLP_slides/week10_finetuning/`
- **Main Presentation:** `presentations/20250923_2110_week10_finetuning_optimal.pdf`
- **Topics:** Adaptation strategies, parameter-efficient tuning, in-context learning
- **Lab:** `lab/week10_finetuning_lab.ipynb` - LoRA and prompt engineering
- **Methods:** Full fine-tuning, LoRA, prefix tuning, prompt design

### Week 11: Efficiency & Optimization
**Location:** `NLP_slides/week11_efficiency/`
- **Main Presentation:** `presentations/20250923_2110_week11_efficiency_optimal.tex`
- **Topics:** Model compression, quantization, distillation, inference optimization
- **Lab:** `lab/week11_efficiency_lab.ipynb` - Quantize and compress models
- **Practical:** Deployment at scale, edge devices

### Week 12: Ethics & Fairness
**Location:** `NLP_slides/week12_ethics/`
- **Main Presentation:** `presentations/20250923_2110_week12_ethics_optimal.pdf`
- **Topics:** Bias detection, fairness metrics, safety, responsible AI
- **Lab:** `lab/week12_ethics_lab.ipynb` - Measure and mitigate bias
- **Critical:** Responsible deployment considerations

## Supplementary Modules

### Neural Network Primer (Prerequisite Module)
**Location:** `NLP_slides/nn_primer/`
**Purpose:** Zero pre-knowledge introduction to neural networks

**When to Use:** Before Week 2 if students lack neural network background

**Materials:**
- 3 comprehensive presentations (48+ pages)
- Discovery-based handouts (classification and function approximation)
- 8 custom figures with concrete analogies

**Topics:**
- Perceptrons and basic neurons
- Activation functions (sigmoid, ReLU, tanh)
- XOR problem and hidden layers
- Backpropagation intuition
- Universal Approximation Theorem

**Pedagogy:** Concrete-to-abstract progression using party decisions and temperature prediction

**Handouts:**
- `handouts/20250926_0800_nn_discovery_handout.pdf` (6 pages, classification focus)
- `handouts/20250926_0801_function_approximation_handout.pdf` (9 pages, regression focus)

### LSTM Primer (Deep Dive Module)
**Location:** `NLP_slides/lstm_primer/`
**Purpose:** Comprehensive LSTM understanding with checkpoint pedagogy

**When to Use:** Between Weeks 3-4 for deeper LSTM understanding

**Materials:**
- 11 presentation versions (10-32 slides)
- Modular architecture with 9 section files
- 20 custom-generated figures
- BSc-level checkpoint quiz slides

**Topics:**
1. Autocomplete challenge (motivation)
2. N-gram baseline and limitations
3. Memory problem visualization
4. RNN architecture with numerical example
5. Vanishing gradient problem (Paris example)
6. LSTM architecture overview (three gates)
7. Deep dive into gate mechanisms (6 slides)
8. BPTT and training progression
9. Variants, applications, PyTorch implementation

**Recommended Version:** `presentations/20250926_2100_lstm_primer_comprehensive_enhanced.pdf` (32 slides)

**Structure:** Modular sections in `presentations/sections/` directory

## Additional Resources

### Embeddings Module
**Location:** `embeddings/`
- Standalone module with PowerShell build system
- 48-slide comprehensive presentation on word embeddings
- 3D visualization notebook (MSc level)
- Build: `cd embeddings && .\build.ps1`

### Exercise Notebooks
**Shakespeare Sonnets:**
- `exercises/shakespeare/shakespeare_sonnets_simple_bsc.ipynb` (BSc level)
- Poetry generation using character-level models

**N-gram Exercises:**
- `exercises/ngrams_Alice_in_Wonderland.ipynb`
- Text generation with Alice in Wonderland corpus

### Visualization Series
**Location:** `notebooks/visualizations/`
Progressive learning through interactive notebooks:
1. `1_simple_ngrams.ipynb` - N-gram basics
2. `2_word_embeddings.ipynb` - Embedding exploration
3. `3_simple_neural_net.ipynb` - Neural fundamentals
4. `4_compare_NLP_methods.ipynb` - Method comparison
5. `5_Tokens Journey Through a Transformer.ipynb` - Token flow
6. `6_Transformers in 3D A Visual Journey.ipynb` - 3D visualization
7. `7_Transformers_in_3d_simplified.ipynb` - Simplified 3D
8. `8_How_Transformers_Learn_Training_in_3D.ipynb` - Training process

## Quick Reference

### Compilation Commands
```bash
# Compile presentation
cd NLP_slides/weekXX_topic/presentations
pdflatex filename.tex
pdflatex filename.tex  # Run twice for references

# Generate figures
cd NLP_slides/weekXX_topic/python
python generate_weekXX_optimal_charts.py

# Clean auxiliary files
mkdir temp 2>$null; mv *.aux,*.log,*.nav,*.out,*.snm,*.toc,*.vrb temp/
```

### Running Lab Notebooks
```bash
# Standard Jupyter
jupyter notebook

# JupyterLab interface
jupyter lab

# Navigate to specific week
cd NLP_slides/weekXX_topic/lab
```

### Figure Generation
```bash
# Week-specific figures
cd NLP_slides/weekXX_topic/python
python generate_weekXX_optimal_charts.py

# Neural Network Primer handout figures
cd NLP_slides/nn_primer/python
python generate_handout_figures.py

# LSTM Primer comprehensive figures
cd NLP_slides/lstm_primer/python
python generate_all_lstm_figures.py

# Common/shared figures
cd NLP_slides/common
python generate_softmax_2d.py
python generate_softmax_function.py
```

## Suggested Course Flow

### Standard 12-Week Course
1. **Week 0 (Optional):** Neural Network Primer for students without background
2. **Weeks 1-2:** Foundations and neural language models
3. **Week 3:** RNN/LSTM (supplement with LSTM Primer as needed)
4. **Weeks 4-6:** Seq2seq, transformers, pre-trained models
5. **Weeks 7-9:** Advanced architectures, tokenization, decoding
6. **Weeks 10-12:** Fine-tuning, efficiency, ethics

### Intensive 8-Week Course
- Week 1: Foundations + Neural LM (combine Weeks 1-2)
- Week 2: RNN/LSTM (Week 3 + LSTM Primer)
- Week 3: Seq2seq (Week 4)
- Week 4: Transformers (Week 5)
- Week 5: Pre-trained models (Week 6)
- Week 6: Advanced topics (Week 7)
- Week 7: Practical deployment (Weeks 8-11 highlights)
- Week 8: Ethics (Week 12)

### Prerequisites by Week
- **Weeks 1-2:** Basic probability, linear algebra
- **Week 3:** Neural network basics (or complete NN Primer)
- **Week 4:** Understanding of RNN architecture
- **Week 5:** Seq2seq models and attention concept
- **Weeks 6+:** Transformer architecture fundamentals
- **Week 11:** Basic knowledge of model deployment
- **Week 12:** No specific prerequisites

## Key Learning Milestones

1. **After Week 2:** Understand embeddings and neural language models
2. **After Week 3:** Implement RNN and LSTM from scratch
3. **After Week 5:** Comprehend transformer architecture completely
4. **After Week 6:** Fine-tune pre-trained models for tasks
5. **After Week 9:** Control generation quality and diversity
6. **After Week 12:** Deploy models responsibly with ethical considerations

## Documentation Files

- **status.md** - Current project status and completion tracking
- **changelog.md** - Detailed change history and updates
- **CLAUDE.md** - Development instructions and conventions
- **COURSE_INDEX.md** - This file (student-facing course navigation)

## Contact & Contribution

For issues, improvements, or questions about course materials, refer to the repository documentation or contact the course instructor.

---

**Last Updated:** 2025-09-27
**Course Version:** 2025 NLP Lectures
**Total Weeks:** 12 core weeks + 2 supplementary modules
**Total Lab Notebooks:** 12
**Total Presentations:** 60+ (including versions and supplements)