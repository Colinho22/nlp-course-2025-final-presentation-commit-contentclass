# Word Embeddings: A Visual Deep Dive - Modular Presentation

## Overview

This is a fully modular LaTeX/Beamer presentation about word embeddings, covering everything from basic one-hot encoding to advanced topics like contextual embeddings and the mathematical foundations of modern NLP.

**Author**: Joerg R. Osterrieder  
**Website**: www.joergosterrieder.com  
**Total Slides**: 48  
**Format**: Beamer presentation, Madrid theme, 16:9 aspect ratio

## Directory Structure

```
embeddings_modular/
├── main.tex                    # Main orchestrator file
├── preamble.tex                # Package imports and setup
├── metadata.tex                # Title, author, date
├── build.ps1                   # Windows build script
├── README.md                   # This file
│
├── styles/                     # Styling and formatting
│   ├── colors.tex             # Color definitions
│   ├── commands.tex           # Custom commands
│   └── listings.tex           # Code listing styles
│
├── content/                    # Main presentation content
│   ├── 00_frontmatter.tex    # Title, TOC, objectives
│   ├── 01_introduction.tex   # The fundamental problem
│   ├── 02_basic_concepts.tex # One-hot and dense embeddings
│   ├── 03_word2vec.tex       # Word2Vec and vector arithmetic
│   ├── 04_contextual.tex     # Evolution to contextual
│   ├── 05_advanced_topics.tex # Advanced representations
│   ├── 06_dimensionality.tex # Curse of dimensionality
│   ├── 07_training.tex       # Training dynamics
│   └── 08_summary.tex        # Summary and conclusions
│
└── appendix/                   # Technical appendices
    ├── A_mathematics.tex      # Mathematical foundations
    └── B_skipgram.tex         # Skip-gram visual guide
```

## Building the Presentation

### Prerequisites

- LaTeX distribution (MiKTeX or TeX Live)
- PowerShell (for Windows build script)
- Python 3.x (for generating figures - optional)

### Quick Build

#### Windows (PowerShell):
```powershell
# Standard build
.\build.ps1

# Fast build (single pass)
.\build.ps1 -Fast

# Clean and build
.\build.ps1 -Clean

# Custom output name
.\build.ps1 -Output my_presentation
```

#### Manual compilation:
```bash
# Full compilation (3 passes for references)
pdflatex main.tex
pdflatex main.tex
pdflatex main.tex
```

#### Using latexmk (if installed):
```bash
latexmk -pdf main.tex
```

## Content Modules

### Part I: Fundamentals
1. **Introduction** (01_introduction.tex)
   - The fundamental problem of representing words
   - Human vs computer understanding

2. **Basic Concepts** (02_basic_concepts.tex)
   - One-hot encoding and its limitations
   - Dense embeddings as the solution
   - Embedding space visualization

3. **Word2Vec** (03_word2vec.tex)
   - Skip-gram and CBOW models
   - Vector arithmetic and analogies
   - Measuring similarity

4. **Contextual Embeddings** (04_contextual.tex)
   - Static vs contextual embeddings
   - Evolution from Word2Vec to BERT

### Part II: Advanced Topics
5. **Advanced Representations** (05_advanced_topics.tex)
   - Sparsity analysis
   - Cosine similarity geometry
   - Context windows
   - Vector arithmetic proof

6. **Curse of Dimensionality** (06_dimensionality.tex)
   - Distance concentration
   - Volume paradox
   - Surface concentration
   - Optimal dimensions

7. **Training Dynamics** (07_training.tex)
   - Three phases of training
   - Gradient dynamics
   - Convergence patterns

### Appendices
- **Mathematical Foundations** (A_mathematics.tex)
  - Skip-gram objective
  - Negative sampling
  - GloVe framework
  - Attention mechanisms
  - BERT training

- **Skip-gram Visual Guide** (B_skipgram.tex)
  - Architecture overview
  - Training pairs extraction
  - Forward/backward passes
  - Evolution visualization

## Customization

### Adding New Sections

1. Create a new `.tex` file in `content/` or `appendix/`
2. Add `\input{content/your_new_file}` to `main.tex`
3. Rebuild the presentation

### Modifying Styles

- **Colors**: Edit `styles/colors.tex`
- **Commands**: Add macros to `styles/commands.tex`
- **Code listings**: Modify `styles/listings.tex`

### Changing Theme

Edit `preamble.tex`:
```latex
\usetheme{Madrid}  % Change to Warsaw, Berlin, etc.
\usecolortheme{seahorse}  % Change color theme
```

## Dependencies

### Required LaTeX Packages
- beamer
- amsmath, amssymb
- tikz
- graphicx
- algorithm2e
- listings
- hyperref
- array, colortbl, booktabs
- tcolorbox

### Figure Generation (Optional)

Python scripts for generating charts:
- `generate_skipgram_training_charts.py`
- `generate_verified_charts.py`
- `generate_dimensionality_charts.py`

Install Python dependencies:
```bash
pip install numpy matplotlib seaborn scipy
```

## Tips for Presenting

1. **Navigation**: Use PDF viewer's presentation mode
2. **Timing**: ~1 minute per slide (48 slides = ~50 minutes)
3. **Interactivity**: Python demos available in `../python_scripts/`
4. **Handouts**: Generate with `\documentclass[handout]{beamer}`

## Compilation Issues

### Common Problems and Solutions

1. **Missing figures**: Ensure `../figures/embeddings/` directory exists
2. **Font issues**: Install required fonts or use `\usepackage{lmodern}`
3. **Memory errors**: Increase LaTeX memory or reduce image quality
4. **Unicode errors**: Ensure UTF-8 encoding in all `.tex` files

### Checking Logs

```powershell
# View compilation errors
Select-String -Pattern "Error" build\main.log

# View warnings
Select-String -Pattern "Warning" build\main.log
```

## Version Control

This modular structure is optimized for version control:
- Each content section in separate file
- Easy to track changes
- Merge-friendly structure
- Can compile individual sections for testing

## License

Educational content - free to use and modify with attribution.

## Contact

**Author**: Joerg R. Osterrieder  
**Website**: www.joergosterrieder.com

---

*Last Updated: 2024*