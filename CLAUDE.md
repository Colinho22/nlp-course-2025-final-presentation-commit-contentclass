# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

NLP course presentation system with LaTeX/Beamer slides covering 12 weeks of Natural Language Processing topics. The course progresses from statistical language models through neural networks to modern transformer architectures and ethical considerations. Contains over 100 LaTeX presentations, 25+ Jupyter notebooks, 80+ Python figure generation scripts, and comprehensive lab materials.

## Quick Start Commands

### Most Common Tasks
```bash
# Verify installation (30 seconds)
python verify_installation.py

# Test all lab notebooks
python test_notebooks.py

# Compile a presentation with timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmm"
cd NLP_slides/weekXX_topic/presentations
pdflatex -jobname="${timestamp}_presentation" main.tex
mkdir temp 2>$null; mv *.aux,*.log,*.nav,*.out,*.snm,*.toc,*.vrb temp/

# Generate figures for a week
cd NLP_slides/weekXX_topic/python
python generate_weekXX_optimal_charts.py

# Build embeddings module
cd embeddings
.\build.ps1
```

## Key Architecture

### Week Module Structure
Each week follows a consistent pattern:
```
NLP_slides/weekXX_topic/
├── presentations/
│   ├── main_presentation.tex      # Primary lecture slides
│   ├── handouts/                  # Student/instructor versions
│   └── previous/                  # Timestamped backups
├── python/                        # Figure generation scripts
├── figures/                       # Week-specific figures
├── lab/                          # Jupyter lab notebook
└── overview/                      # Week overview and learning objectives
```

### Shared Infrastructure
- **NLP_slides/common/slide_layouts.tex**: Centralized LaTeX macros and slide templates used across all presentations
- **NLP_slides/common/master_template.tex**: Master template with optimal readability settings (referenced by all weeks)
- **NLP_slides/common/chart_utils.py**: Shared utilities for consistent chart styling across all figure generation
- **NLP_slides/common/shared_figures/**: Cross-week figures referenced by multiple presentations
- **NLP_slides/common/*.py**: Shared Python scripts for figure generation and validation
- **NLP_slides/figures/**: Legacy central figures directory (being phased out in favor of week-specific folders)
- **NLP_slides/python_scripts/**: Legacy scripts directory (being migrated to week-specific python/ folders)

### Special Modules
- **week00_introduction/**: First-day lecture materials
  - `presentations/20250927_1400_week00_first_day_lecture.pdf` - 20-slide course introduction
  - Course motivation, 12-week journey, assessment overview
  - Getting started guide and prerequisites check
- **embeddings/**: Standalone module with its own build system
  - Has PowerShell build script (`build.ps1`) and Makefile
  - Contains modular presentation with 48 slides on word embeddings
  - Separate from main week structure with own styles/, content/, appendix/ folders
- **exercises/**: Additional exercise materials
  - Shakespeare sonnet generation notebooks (BSc and MSc levels)
  - Alice in Wonderland n-gram exercises
- **NLP_slides/nn_primer/**: Neural network fundamentals module
  - Complete structure: presentations/, figures/, python/, handouts/
  - Foundational content for students with ZERO pre-knowledge
  - Discovery-based handouts that build from concrete analogies (party decision → neuron computation)
  - All figure generation uses proper sigmoid activation functions (no fake approximations)
- **NLP_slides/lstm_primer/**: LSTM comprehensive introduction module (Reference Example)
  - Modular architecture with 9 section files in `sections/` subdirectory
  - 32-slide comprehensive version: `20250926_2100_lstm_primer_comprehensive_enhanced.pdf`
  - Demonstrates BSc-level checkpoint pedagogy (2 quiz slides with answers)
  - Master file uses `\input{sections/XX_name.tex}` pattern for maintainability
  - Complete section structure:
    - `01_introduction.tex` - Autocomplete challenge and real-world impact
    - `02_baselines.tex` - N-gram baseline models and limitations
    - `03_memory_problem.tex` - Memory problem motivation
    - `04_rnn.tex` - RNN architecture with concrete numerical example
    - `05_vanishing_gradient.tex` - Vanishing gradient problem with Paris example
    - `06_lstm_architecture.tex` - LSTM overview with three gates and first checkpoint
    - `07_lstm_details.tex` - Deep dive into gate mechanisms (6 slides)
    - `08_training.tex` - BPTT, training progression, gradient highway, second checkpoint, training tips, debugging
    - `09_applications.tex` - Variants, model comparison, attention bridge, PyTorch implementation
  - All 7 generated figures integrated: autocomplete_screenshot, context_window_comparison, lstm_architecture, gradient_flow_comparison, gate_activation_heatmap, training_progression, model_comparison_table
  - Uses `[fragile]` frames for code listings to avoid compilation errors

## Key Commands

### Full Course Generation Script
```bash
# Generate all remaining materials for weeks 8-12
python generate_all_remaining_materials.py

# Creates presentations, lab notebooks, and handouts with proper structure
```

### Building LaTeX/PDF Documents
```bash
# Standard compilation (Windows PowerShell)
pdflatex filename.tex
pdflatex filename.tex  # Run twice for references

# Alternative with timestamp naming (following user preferences)
$timestamp = Get-Date -Format "yyyyMMdd_HHmm"
pdflatex -jobname="${timestamp}_presentation" filename.tex

# Clean auxiliary files after compilation
mkdir temp 2>$null; mv *.aux,*.log,*.nav,*.out,*.snm,*.toc,*.vrb temp/ 2>$null

# Avoid file lock issues
pdflatex -jobname=output_name filename.tex

# Compile and immediately move auxiliary files
pdflatex presentation.tex; mkdir temp 2>$null; mv *.aux,*.log,*.nav,*.out,*.snm,*.toc,*.vrb temp/
```

### Embeddings Module Build
```powershell
# PowerShell build script for embeddings presentation
cd embeddings
.\build.ps1                    # Full compilation (3 passes)
.\build.ps1 -Fast             # Single pass for quick preview
.\build.ps1 -Clean            # Remove temporary files
.\build.ps1 -Output custom    # Custom output name
```

### Figure Generation for Specific Weeks
```bash
# Navigate to week's python folder
cd NLP_slides/weekXX_topic/python

# Main figure generation scripts (naming pattern)
python generate_weekXX_optimal_charts.py     # Optimal readability version (preferred)
python generate_weekXX_enhanced_charts.py    # Enhanced visuals
python generate_weekXX_bsc_charts.py         # BSc-level simplified
python generate_weekXX_minimalist_charts.py  # Minimalist monochromatic style

# Neural Network Primer handout figures
cd NLP_slides/nn_primer/python
python generate_handout_figures.py           # Generates all 8 handout figures (party analogy, neurons, XOR, etc.)

# LSTM Primer comprehensive figures
cd NLP_slides/lstm_primer/python
python generate_all_lstm_figures.py          # Generates all 7 LSTM figures (architecture, gates, training, comparison)

# Common figure generation from shared scripts
cd NLP_slides/common
python generate_softmax_2d.py                # 2D softmax visualization
python generate_softmax_function.py          # Softmax function plots
python check_missing_figures.py              # Validate all figures exist
python generate_nlp_charts.py                # General NLP visualizations
python chart_utils.py                        # Shared utilities (imported by other scripts)

# Week overview figure generation
cd NLP_slides/weekXX_topic/overview
python generate_weekXX_overview_charts.py    # Week overview visualizations

# Week-specific generation scripts (examples)
python generate_week2_evolution_charts.py    # Week 2 evolution timeline
python generate_week4_minimalist_charts.py   # Week 4 minimalist seq2seq figures
python generate_week5_intro_visuals.py       # Week 5 intro animations
python generate_week6_paradigm_shift.py      # Week 6 paradigm visualization

# Showcase charts for overview presentations
cd scripts
python generate_showcase_charts.py           # Course-wide showcase visualizations
```

### Running Jupyter Notebooks
```bash
jupyter notebook              # Standard interface
jupyter lab                   # JupyterLab interface
```

## Lab Notebook Locations

### Core Lab Sessions
- `NLP_slides/week03_rnn/lab/week03_rnn_lab.ipynb` - RNN/LSTM implementation
- `NLP_slides/week04_seq2seq/lab/week04_seq2seq_lab.ipynb` - Seq2seq with attention
- `NLP_slides/week05_transformers/lab/week05_transformer_lab.ipynb` - Transformer from scratch
- `NLP_slides/week06_pretrained/lab/week06_bert_finetuning.ipynb` - BERT fine-tuning

### Exercise Notebooks
- `exercises/shakespeare/shakespeare_sonnets_simple_bsc.ipynb` - Simplified poetry generation
- `exercises/ngrams_Alice_in_Wonderland.ipynb` - N-gram text generation
- `embeddings/word_embeddings_3d_msc.ipynb` - Advanced 3D embeddings (MSc level)

### Visualization Series (Progressive Learning)
In `notebooks/visualizations/`:
1. `1_simple_ngrams.ipynb` - N-gram basics
2. `2_word_embeddings.ipynb` - Embedding exploration
3. `3_simple_neural_net.ipynb` - Neural network fundamentals
4. `4_compare_NLP_methods.ipynb` - Method comparison
5. `5_Tokens Journey Through a Transformer.ipynb` - Token flow
6. `6_Transformers in 3D A Visual Journey.ipynb` - 3D visualization
7. `7_Transformers_in_3d_simplified.ipynb` - Simplified 3D
8. `8_How_Transformers_Learn_Training_in_3D.ipynb` - Training process

## LaTeX/Beamer Critical Requirements

### Document Class Configuration
```latex
\documentclass[8pt,aspectratio=169]{beamer}
\usetheme{Madrid}  % Primary theme (Frankfurt also used)
\usecolortheme{seahorse}
\setbeamertemplate{navigation symbols}{}

% For minimalist style (preferred)
\definecolor{mainGray}{RGB}{64,64,64}
\definecolor{annotGray}{RGB}{180,180,180}
\definecolor{backGray}{RGB}{240,240,240}
```

### Math and Probability Notation
- Conditional probability: `$P(A\given B)$` (requires `\newcommand{\given}{\mid}`)
- Times symbol: `$\times$` not bare `\times`
- All math in math mode: `$...$` or `\[...\]`

### Common LaTeX Pitfalls
- Add `[fragile]` to frames with lstlisting code
- Use LaTeX quotes `` `` not straight quotes " "
- Lists: `\begin{itemize}...\end{itemize}` (never HTML-style `</end{itemize}`)
- Graphics paths: `../figures/filename.pdf` (relative to presentation location)

## Slide Layout System (from common/slide_layouts.tex)

### Standard Slide Templates
- `\conceptslide{title}{content}{figure}` - Theory and concepts
- `\codeslide{title}{code}{explanation}` - Code examples
- `\resultslide{title}{visualization}{insights}` - Results

### Mathematical Commands
- `\given` - Conditional probability bar
- `\prob{X}` - Probability notation
- `\argmax`, `\softmax` - ML operators
- `\highlight{text}` - Emphasis
- `\eqbox{equation}` - Boxed equations

### BSc Pedagogical Elements
- `\checkpoint{content}` - Yellow understanding checks
- `\prereq{content}` - Blue prerequisites
- `\misconception{content}` - Red common errors
- `\intuition{content}` - Purple insights
- `\realworld{content}` - Orange applications

**Checkpoint Slide Pattern** (from LSTM primer):
```latex
\begin{frame}{Checkpoint: Understanding LSTM Architecture}
\begin{center}
\textbf{Test Your Understanding}
\end{center}
\vspace{5mm}

\begin{columns}
\column{0.48\textwidth}
\textbf{Quick Quiz:}

\vspace{3mm}
\textbf{Question 1:} What's the key difference between RNN and LSTM?

\begin{itemize}
\item[A)] More parameters
\item[B)] Addition instead of multiplication
\item[C)] Faster training
\item[D)] Different activations
\end{itemize}

\column{0.48\textwidth}
\textbf{Answers:}

\vspace{3mm}
\textbf{Answer 1:} B - Addition for cell state

\begin{itemize}
\item Cell state: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
\item Creates direct gradient path
\item No repeated matrix multiplication
\item This is the key innovation!
\end{itemize}
\end{columns}
\end{frame}
```

## Python Figure Generation Standards

### Setup Template
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Educational color scheme
COLOR_CURRENT = '#FF6B6B'  # Red - focus/current
COLOR_CONTEXT = '#4ECDC4'  # Teal - context
COLOR_PREDICT = '#95E77E'  # Green - output
COLOR_NEUTRAL = '#E0E0E0'  # Gray - neutral

# Minimalist monochromatic palette (preferred)
COLOR_MAIN = '#404040'     # RGB(64,64,64)
COLOR_ACCENT = '#B4B4B4'   # RGB(180,180,180)
COLOR_LIGHT = '#F0F0F0'    # RGB(240,240,240)

plt.figure(figsize=(10, 6))
# ... plotting code ...
plt.savefig('../figures/name.pdf', dpi=300, bbox_inches='tight')
```

### Critical: Use Real Neural Network Math
When creating function approximation visualizations, ALWAYS use proper activation functions:

```python
# CORRECT: Real sigmoid activation
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Each neuron: a_i * sigmoid(w_i*x + b_i)
y_approx = sum(a * sigmoid(w * x + b) for w, a, b in neurons)

# WRONG: Fake triangular approximations or arbitrary scaling
# y_approx += np.maximum(0, -abs(x - pos) + 0.8)  # ❌ NOT how neural networks work
```

## Directory Naming Patterns

### Week Directories
- Main directories: `NLP_slides/weekXX_topic/` (e.g., `week02_neural_lm`, `week05_transformers`)
- Legacy asterisk directories: `week08_*`, `week09_*`, etc. (older versions, being phased out)
- Standalone presentations: `NLP_slides/NLP_Course_standalone_presentations/` (individual .tex/.pdf files)

## Common Problem Solutions

### N-gram Repetitive Text
1. Filter punctuation-heavy contexts
2. Add diverse starter words
3. Use power 0.7 for weighted selection
4. Shuffle starter word order

### LaTeX Compilation Issues
- **PDF locked**: Use `-jobname=newname` or close viewer
- **Fragile frame**: Add `[fragile]` with code listings
- **Math errors**: Check for unescaped `$`, `_`, `^`
- **Conditional probability**: Always in math mode with `\given`

### Windows-Specific
- Use forward slashes or escaped backslashes in paths
- PowerShell for embeddings build script
- Move auxiliary files to temp folder after compilation to avoid clutter

## Course Week Topics Reference
1. **Foundations & Statistical Language Models** - N-grams, Markov models, basic probability
2. **Neural Language Models & Word Embeddings** - Word2Vec, GloVe, neural architectures
3. **RNN/LSTM/GRU** - Sequential models, vanishing gradients, gating mechanisms
4. **Sequence-to-Sequence Models** - Encoder-decoder, attention mechanism, beam search
5. **Transformers & Attention** - Self-attention, multi-head attention, positional encoding
6. **Pre-trained Models (BERT, GPT)** - Transfer learning, masked LM, autoregressive models
7. **Advanced Transformers** - T5, GPT-3, modern architectures
8. **Tokenization & Vocabulary** - BPE, WordPiece, SentencePiece
9. **Decoding Strategies** - Greedy, beam search, sampling methods
10. **Fine-tuning & Prompt Engineering** - Adaptation strategies, in-context learning
11. **Efficiency & Optimization** - Model compression, quantization, distillation
12. **Ethics & Fairness** - Bias, safety, responsible AI

## Presentation Workflow

### Creating New Presentations
1. Use timestamp prefix: `YYYYMMDD_HHMM_filename.tex`
2. Move previous versions to `previous/` folder before saving
3. Apply minimalist monochromatic style by default
4. Generate figures with Python scripts (no TikZ)
5. Clean auxiliary files to `temp/` after compilation

### Presentation Naming Conventions
- Main presentations: `weekXX_topic.tex` (e.g., `week05_transformers.tex`)
- Optimal versions: `YYYYMMDD_HHMM_weekXX_topic_optimal.tex`
- BSc level: `weekXX_topic_bsc.tex`
- Enhanced versions: `weekXX_topic_enhanced.tex`
- Handouts: `weekXX_handout_student.tex`, `weekXX_handout_instructor.tex`
- Previous versions: Store in `previous/` subdirectory with timestamps

### Quality Checklist
- Two-column layout (0.48\textwidth each)
- Madrid theme, 8pt font, aspectratio=169
- Monochromatic gray palette (RGB values as specified)
- Python-generated PDF figures only
- Proper LaTeX quotes and math mode
- `[fragile]` for code listings

## Important Files and Utilities

### Templates and Configuration
- `NLP_slides/common/master_template.tex` - Master template used by all presentations
- `NLP_slides/common/slide_layouts.tex` - LaTeX macros for all slide templates and commands
- `NLP_slides/common/chart_utils.py` - Python utilities for consistent figure styling
- `template_layout.tex` / `template_layout.pdf` - Root level template examples

### Status Tracking and Infrastructure
- `status.md` - Current project status (12/12 weeks complete, all infrastructure ready)
- `changelog.md` - Detailed change history
- `README.md` - Student-facing course overview with 3-step quick start
- `COURSE_INDEX.md` - Complete week-by-week navigation
- `INSTALLATION.md` - Comprehensive setup guide with troubleshooting
- `requirements.txt` - Python dependencies (30+ packages)
- `environment.yml` - Conda environment specification
- `verify_installation.py` - 30-second environment verification script
- `test_notebooks.py` - Automated testing for all lab notebooks

### Archive Directory
- `archive/NLP_slides_deutsch/` - German version backups
- `archive/presentations/` - Previous presentation versions

## Handout Design Philosophy

### Neural Network Primer Handouts
The `nn_primer/handouts/` demonstrate pedagogical best practices for zero pre-knowledge learners:

**Two Complete Discovery Handouts:**
1. **Classification Handout** (`20250926_0800_nn_discovery_handout_v3.pdf`) - 6 pages
   - Focus: Decision boundaries and classification
   - Analogy: Party decision problem (discrete choices)
   - Key problem: XOR (requires multiple neurons)
   - 8 figures covering neuron basics, activation functions, XOR solution

2. **Function Approximation Handout** (`20250926_0801_function_approximation_handout.pdf`) - 9 pages
   - Focus: Curve fitting and regression
   - Analogy: Temperature prediction (continuous values)
   - Key problem: Smooth curves (requires sigmoid combinations)
   - 10 figures covering sigmoid parameters, bump creation, parabola approximation, Universal Approximation Theorem

**Pedagogical Principles:**
1. **Concrete-to-Abstract Progression**: Start with relatable scenarios (party decisions, temperature) before mathematical notation
2. **Discovery-Based Learning**: Students compute examples BY HAND before seeing formal definitions
3. **"This is a Neuron!" Reveal**: Only name concepts AFTER students have experienced them
4. **Visual Scaffolding**: Figures appear after concept introduction, not before
5. **Fill-in-the-Blank Consolidation**: Active learning through completion exercises
6. **Progressive Complexity**: 1 neuron → 2 neurons → 3 neurons → many neurons (Universal Approximation)

### Handout Document Structure
```latex
\documentclass[10pt,a4paper]{article}  % Compact font for more content
\usepackage[margin=0.75in]{geometry}   % Tighter margins
\setlist{nosep, leftmargin=*, after=\vspace{-2pt}}  % Compact spacing

% Exercise boxes with minimal padding
\newtcolorbox{exercise}[1][]{
    colback=blue!5!white,
    colframe=blue!75!black,
    title=#1,
    fonttitle=\bfseries,
    left=3pt, right=3pt, top=3pt, bottom=3pt
}

% Use compact header instead of \maketitle
\begin{center}
\textbf{\Large Title}\\[2pt]
\textit{Subtitle}\\[2pt]
\small Context | Time: XX minutes
\end{center}
```

## Student Onboarding Flow

New students should follow this sequence:
1. **Clone repository** → `git clone https://github.com/josterri/2025_NLP_Lectures.git`
2. **Install dependencies** → `pip install -r requirements.txt` (15 minutes)
3. **Verify environment** → `python verify_installation.py` (30 seconds)
4. **Review Week 0 lecture** → `NLP_slides/week00_introduction/presentations/20250927_1400_week00_first_day_lecture.pdf`
5. **Complete NN Primer** (if needed) → `NLP_slides/nn_primer/` for zero pre-knowledge students
6. **Start Week 2 lab** → `jupyter lab NLP_slides/week02_neural_lm/lab/week02_word_embeddings_lab.ipynb`

## Course Delivery Infrastructure

### Testing and Validation
```bash
# Quick environment check (30 seconds)
python verify_installation.py

# Test all lab notebooks (generates JSON and Markdown reports)
python test_notebooks.py

# Test specific week notebook
cd NLP_slides/weekXX_topic/lab
jupyter nbconvert --to notebook --execute notebook_name.ipynb
```

### Documentation Hierarchy
- **README.md**: First impression, 3-step quick start, course overview
- **COURSE_INDEX.md**: Complete navigation, prerequisites, week-by-week breakdown
- **INSTALLATION.md**: Detailed setup, GPU config, platform-specific troubleshooting
- **CLAUDE.md**: This file - development guidance
- **status.md**: Project completion tracking
- **changelog.md**: Historical changes and milestones