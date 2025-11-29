# NLP Course 2025: From N-grams to Transformers

<p align="center">
  <a href="https://quantlet.de">
    <img src="logo/quantlet.png" alt="QuantLet Logo" width="120"/>
  </a>
</p>

<p align="center">
  <strong>QuantLet-Compatible Course Materials</strong>
</p>

![Course Status](https://img.shields.io/badge/weeks-12%2F12%20complete-brightgreen)
![Framework](https://img.shields.io/badge/framework-100%25%20applied-success)
![Labs](https://img.shields.io/badge/labs-12%20notebooks-blue)
![Charts](https://img.shields.io/badge/charts-168%20visualizations-purple)
![License](https://img.shields.io/badge/license-MIT-orange)

> A comprehensive Natural Language Processing course covering statistical foundations through modern transformer architectures. Build ChatGPT from scratch!

## Quick Start (3 Steps)

```bash
# 1. Clone the repository
git clone https://github.com/josterri/2025_NLP_Lectures.git
cd 2025_NLP_Lectures

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start learning!
jupyter lab NLP_slides/week02_neural_lm/lab/week02_word_embeddings_lab.ipynb
```

## What You'll Learn

This course takes you from foundational statistical methods to state-of-the-art neural architectures:

- **Weeks 1-2:** Statistical language models and word embeddings (Word2Vec, GloVe)
- **Weeks 3-4:** Sequential models (RNN/LSTM) and sequence-to-sequence with attention
- **Weeks 5-7:** Transformers, BERT, GPT, and advanced architectures
- **Weeks 8-10:** Tokenization, decoding strategies, and fine-tuning
- **Weeks 11-12:** Efficiency optimization and ethical AI deployment

By the end, you'll build a working transformer from scratch and understand the architecture behind ChatGPT and Claude.

## Course Structure

### Core Materials (12 Weeks)
Each week includes:
- **Presentation:** LaTeX/Beamer slides with optimal readability
- **Lab Notebook:** Interactive Jupyter notebook with hands-on exercises
- **Handouts:** Pre-class discovery exercises and post-class technical practice

### Supplementary Modules
- **Neural Network Primer:** Zero pre-knowledge intro to neural networks
- **LSTM Primer:** Comprehensive deep dive into LSTM architecture (32 slides)
- **Embeddings Module:** Standalone word embedding module with 3D visualizations

### Total Content
- 60+ presentations (including versions and supplements)
- 12 interactive lab notebooks
- 40+ handout documents
- 100+ Python-generated figures
- 8 progressive visualization notebooks

## Prerequisites

- **Required:**
  - Python 3.8 or higher
  - Basic linear algebra (vectors, matrices)
  - Basic probability theory
  - Comfortable with Python programming

- **Helpful but not required:**
  - PyTorch experience
  - Understanding of backpropagation
  - Machine learning fundamentals

**New to neural networks?** Start with our Neural Network Primer module before Week 2.

## Installation

### Option 1: pip (Recommended)
```bash
pip install -r requirements.txt
```

### Option 2: conda
```bash
conda env create -f environment.yml
conda activate nlp2025
```

### GPU Support
For GPU acceleration (recommended for Weeks 5+):
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions and troubleshooting.

## Course Navigation

### Week-by-Week Guide
Full navigation with topics, prerequisites, and learning objectives: [COURSE_INDEX.md](COURSE_INDEX.md)

### Week Highlights

| Week | Topic | Key Concepts | Lab |
|------|-------|--------------|-----|
| 1 | Foundations | N-grams, perplexity, statistical LM | - |
| 2 | Word Embeddings | Word2Vec, GloVe, neural LM | Implement embeddings |
| 3 | RNN/LSTM | Sequential models, BPTT | Build LSTM from scratch |
| 4 | Seq2Seq | Attention mechanism, translation | Machine translation |
| 5 | Transformers | Self-attention, multi-head | Build transformer |
| 6 | Pre-trained | BERT, GPT, transfer learning | Fine-tune BERT |
| 7 | Advanced | T5, GPT-3, scaling laws | Experiment with GPT |
| 8 | Tokenization | BPE, WordPiece, SentencePiece | Implement tokenizer |
| 9 | Decoding | Beam, sampling, nucleus, **contrastive** | Compare 6 methods |
| 10 | Fine-tuning | LoRA, prompt engineering | Adapt models |
| 11 | Efficiency | Quantization, distillation | Optimize models |
| 12 | Ethics | Bias, fairness, safety | Measure bias |

## Quantlet Charts

All Python-generated visualizations follow the [Quantlet](https://quantlet.de) standard format with:
- Numbered folders (`01_chart_name/`, `02_chart_name/`, etc.)
- Self-contained Python scripts
- Standard `metainfo.txt` with description, keywords, and usage

### Final Lecture Charts
See [FinalLecture/](FinalLecture/) for 8 Quantlet-formatted visualizations covering:
- Vector database architecture
- HNSW nearest neighbor search
- RAG conditional probabilities
- Hybrid search flow

## Project Structure

```
├── FinalLecture/               # Quantlet-formatted charts (Final Lecture)
├── logo/                       # Quantlet branding
├── NLP_slides/
│   ├── week01_foundations/      # Week 1: Statistical LM
│   ├── week02_neural_lm/        # Week 2: Word embeddings
│   ├── week03_rnn/              # Week 3: RNN/LSTM/GRU
│   ├── ...                      # Weeks 4-12
│   ├── nn_primer/               # Neural network primer
│   ├── lstm_primer/             # LSTM deep dive
│   └── common/                  # Shared templates and utils
├── embeddings/                  # Standalone embeddings module
├── exercises/                   # Additional practice
├── figures/                     # Shared visualizations
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
└── COURSE_INDEX.md              # Full course navigation
```

## Key Learning Milestones

- ✅ **After Week 2:** Understand and implement word embeddings
- ✅ **After Week 3:** Build RNN and LSTM from scratch
- ✅ **After Week 5:** Comprehend transformer architecture completely
- ✅ **After Week 6:** Fine-tune pre-trained models (BERT, GPT)
- ✅ **After Week 9:** Control text generation quality and diversity
- ✅ **After Week 12:** Deploy models responsibly with ethical considerations

## Usage Examples

### Run a Lab Notebook
```bash
# Start Jupyter Lab
jupyter lab

# Navigate to a week's lab folder
cd NLP_slides/week05_transformers/lab
jupyter notebook week05_transformer_lab.ipynb
```

### Compile a Presentation
```bash
cd NLP_slides/week02_neural_lm/presentations
pdflatex week02_neural_lm.tex
```

### Generate Figures
```bash
cd NLP_slides/week05_transformers/python
python generate_week05_optimal_charts.py
```

## Testing the Course

Test all lab notebooks for execution:
```bash
python test_notebooks.py
```

This validates that all 12 lab notebooks execute correctly in your environment.

## Course Delivery Options

### Standard 12-Week Semester
- One week per topic
- Weekly labs and assignments
- Suitable for undergraduate/graduate courses

### Intensive 8-Week Course
- Combine Weeks 1-2, skip some advanced topics
- Accelerated pace for bootcamps
- Focus on core transformer concepts

### Self-Paced Learning
- Progress at your own speed
- Complete prerequisite modules first
- Focus on labs and hands-on practice

## Documentation

- **[COURSE_INDEX.md](COURSE_INDEX.md)** - Complete week-by-week navigation
- **[INSTALLATION.md](INSTALLATION.md)** - Detailed setup instructions
- **[CLAUDE.md](CLAUDE.md)** - Development guide and conventions
- **[status.md](status.md)** - Project status and completion tracking
- **[changelog.md](changelog.md)** - Change history

## Support and Resources

- **Issues:** Report problems at [GitHub Issues](https://github.com/josterri/2025_NLP_Lectures/issues)
- **Prerequisites:** Check the Neural Network Primer if you're new to deep learning
- **GPU Requirements:** Most labs work on CPU; Weeks 5+ benefit from GPU

## Contributing

Contributions are welcome! Areas for contribution:
- Additional exercises and examples
- Translations to other languages
- MSc-level challenge problems
- Bug fixes and improvements

## License

This course is released under the MIT License. See LICENSE for details.

## Acknowledgments

Course materials developed with pedagogical focus on:
- Discovery-based learning
- Concrete-to-abstract progression
- Hands-on implementation
- Real-world applications

Built with LaTeX/Beamer, Python, PyTorch, and Jupyter.

## Citation

If you use these materials in your course or research, please cite:

```bibtex
@misc{nlp2025course,
  title={NLP Course 2025: From N-grams to Transformers},
  author={Joerg Osterrieder},
  year={2025},
  url={https://github.com/josterri/2025_NLP_Lectures}
}
```

---

**Ready to start?** Check [INSTALLATION.md](INSTALLATION.md) for setup, then dive into Week 2's word embeddings lab!

**Questions?** See [COURSE_INDEX.md](COURSE_INDEX.md) for complete navigation and prerequisites.