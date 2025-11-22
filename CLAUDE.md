# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

NLP course presentation system with LaTeX/Beamer slides covering 12 weeks of Natural Language Processing topics. The course progresses from statistical language models through neural networks to modern transformer architectures and ethical considerations.

**Course Status**: 🎉 **100% COMPLETE** - All 12 weeks restructured with Educational Framework (Nov 18, 2025)

**Content Summary**:
- **500+ total slides** across 12 weeks + specialized modules (summarization, embeddings, LSTM primer, ML intro)
- **250+ professional charts** - all Python-generated PDFs with consistent BSc Discovery color scheme
- **26+ Jupyter notebooks** with hands-on exercises
- **90+ figure generation scripts** including enhanced summarization and decoding charts
- **100% framework coverage** - Every week uses discovery-based pedagogy

**Last Major Update**: 2025-11-19 (Week 9 Decoding: Final Input→Process→Output redesign)
**Platform**: Windows (PowerShell commands)
**LaTeX**: Beamer presentations with Madrid theme, 8pt font
**Python**: 3.8+ with PyTorch, matplotlib, seaborn, mpl_toolkits.mplot3d, graphviz
**Repository**: https://github.com/josterri/2025_NLP_Lectures

## Quick Reference Commands

### Generate Charts and Compile Presentation (Most Common Workflow)
```powershell
# 1. Generate charts first
cd NLP_slides/weekXX_topic/python
python generate_weekXX_bsc_discovery_charts.py

# 2. Compile with timestamp
cd ../presentations
$timestamp = Get-Date -Format "yyyyMMdd_HHmm"
pdflatex -jobname="${timestamp}_weekXX_topic_bsc_discovery" presentation.tex
mkdir temp 2>$null; mv *.aux,*.log,*.nav,*.out,*.snm,*.toc,*.vrb temp/

# 3. Open the PDF
start "${timestamp}_weekXX_topic_bsc_discovery.pdf"
```

### Fix Common LaTeX Compilation Issues
```powershell
# PDF locked by viewer
pdflatex -jobname=temp_output presentation.tex

# Missing figures
cd ../python && python generate_weekXX_charts.py

# Overflow warnings - fix in .tex file:
# \includegraphics[width=0.60\textwidth]{...}  # Reduce from 0.75
# \vspace{-3mm}  # Add negative space before figure
```

### Run Lab Notebooks
```bash
# Start Jupyter Lab
jupyter lab

# Open specific week's lab
jupyter lab NLP_slides/weekXX_topic/lab/weekXX_lab.ipynb

# Test notebook execution
jupyter nbconvert --to notebook --execute notebook.ipynb --output test.ipynb
```

### Build Special Modules
```powershell
# Embeddings module (has its own build system)
cd embeddings
.\build.ps1                    # Full compilation (3 passes)
.\build.ps1 -Fast             # Single pass preview
.\build.ps1 -Clean            # Remove temp files
# Alternative: make full (Unix/Mac/WSL)

# Summarization module (enhanced Nov 2025)
cd NLP_slides/summarization_module/python
python generate_new_2025_charts.py               # 5 new charts
python generate_all_enhanced_charts_complete.py  # All 87 charts
cd ../presentations
pdflatex 20251115_0943_llm_summarization_bsc_discovery.tex

# Week 9 Decoding (final version Nov 19, 2025)
cd NLP_slides/week09_decoding/python
python generate_week09_enhanced_charts.py        # Input→Process→Output charts
python generate_full_exploration_graphviz.py     # Exponential explosion tree
cd ../presentations
pdflatex 20251119_0943_week09_decoding_final.tex

# Test all notebooks
python test_notebooks.py
```

## High-Level Architecture

### Week Module Structure Pattern
Every week follows this consistent structure:
```
NLP_slides/weekXX_topic/
├── presentations/
│   ├── YYYYMMDD_HHMM_canonical.tex  # Current version only
│   ├── YYYYMMDD_HHMM_canonical.pdf  # Current PDF
│   ├── figures/                      # Local copy of required figures (optional)
│   ├── previous/                     # All older versions
│   └── temp/                         # Compilation artifacts
├── python/                           # Chart generation scripts
├── figures/                          # Generated PDF charts
├── lab/                             # Jupyter notebook(s)
└── overview/                        # Learning objectives
```

### Special Modules (Different Structure)

**Summarization Module** (Enhanced Nov 15, 2025):
- Location: `NLP_slides/summarization_module/`
- Latest: `20251115_0943_llm_summarization_bsc_discovery.pdf` (40 pages)
- Contains: 87 charts total (82 existing + 5 new)
- Four-part structure: Problem Space (8) + Technical (10) + Advanced (10) + Evaluation (7) + End (1)
- Generation: `generate_new_2025_charts.py` for newest visualizations

**Embeddings Module**:
- Location: `embeddings/`
- Has PowerShell build script (`build.ps1`) and Makefile
- Modular structure with styles/, content/, appendix/ folders
- 48 slides on word embeddings with interactive 3D visualizations
- Build variants: presentation, handout, notes, print

**LSTM Primer** (Reference Implementation):
- Location: `NLP_slides/lstm_primer/`
- Modular with 9 section files in `sections/` subdirectory
- Uses `\input{sections/XX_name.tex}` pattern
- 32-slide comprehensive version with checkpoint quizzes

**Neural Network Primer**:
- Location: `NLP_slides/nn_primer/`
- Zero pre-knowledge content with discovery handouts
- Uses proper sigmoid activation (no fake approximations)

**ML Introduction Module** (New):
- Location: `NLP_slides/ML_Intro/`
- Latest: `20251015_1030_ml_paradigms.pdf`
- Machine learning paradigms overview

### Shared Infrastructure
- `NLP_slides/common/slide_layouts.tex` - Centralized LaTeX macros
- `NLP_slides/common/chart_utils.py` - Shared chart styling utilities
- `NLP_slides/template_beamer_final.tex` - Master template (DO NOT use \input)
- `test_notebooks.py` - Validates all lab notebooks execute correctly

## Critical LaTeX/Beamer Requirements

### MANDATORY Preamble Structure
Always copy this complete preamble from `template_beamer_final.tex`:
```latex
\documentclass[8pt,aspectratio=169]{beamer}
\usetheme{Madrid}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{multicol}
\usepackage{amsmath}
\usepackage{tikz}
\usepackage{listings}
\usepackage{algorithm2e}

% Complete color palette (ALL 12 REQUIRED)
\definecolor{mlblue}{RGB}{0,102,204}
\definecolor{mlpurple}{RGB}{51,51,178}
\definecolor{mllavender}{RGB}{173,173,224}
\definecolor{mllavender2}{RGB}{193,193,232}
\definecolor{mllavender3}{RGB}{204,204,235}
\definecolor{mllavender4}{RGB}{214,214,239}
\definecolor{mlorange}{RGB}{255, 127, 14}
\definecolor{mlgreen}{RGB}{44, 160, 44}
\definecolor{mlred}{RGB}{214, 39, 40}
\definecolor{mlgray}{RGB}{127, 127, 127}
\definecolor{lightgray}{RGB}{240, 240, 240}
\definecolor{midgray}{RGB}{180, 180, 180}

% Beamer customization (ALL REQUIRED)
\setbeamercolor{palette primary}{bg=mllavender3,fg=mlpurple}
\setbeamercolor{palette secondary}{bg=mllavender2,fg=mlpurple}
\setbeamercolor{palette tertiary}{bg=mllavender,fg=white}
\setbeamercolor{palette quaternary}{bg=mlpurple,fg=white}
\setbeamercolor{structure}{fg=mlpurple}
\setbeamercolor{section in toc}{fg=mlpurple}
\setbeamercolor{subsection in toc}{fg=mlblue}
\setbeamercolor{title}{fg=mlpurple}
\setbeamercolor{frametitle}{fg=mlpurple,bg=mllavender3}
\setbeamercolor{block title}{bg=mllavender2,fg=mlpurple}
\setbeamercolor{block body}{bg=mllavender4,fg=black}

% Bottom note command (REQUIRED)
\newcommand{\bottomnote}[1]{%
\vfill
\vspace{-2mm}
\textcolor{mllavender2}{\rule{\textwidth}{0.4pt}}
\vspace{1mm}
\footnotesize
\textbf{#1}
}

% Conditional probability (REQUIRED)
\newcommand{\given}{\mid}
```

### Common LaTeX Fixes
- **Columns**: Always use `\column{0.48\textwidth}` (never 0.49)
- **Code frames**: Add `[fragile]` option for lstlisting
- **Quotes**: Use `` `` and '' '' (not straight quotes)
- **Math mode**: `$P(A \given B)$` not bare P(A|B)
- **Graphics**: Relative paths `../figures/name.pdf` or local `figures/name.pdf`
- **Unicode**: Replace with LaTeX commands (✓ → \checkmark, ✗ → $\times$)

## Python Chart Generation Standards

### BSc Discovery Color Scheme (USE FOR ALL NEW CHARTS)
```python
# Standard setup
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

# BSc Discovery colors (STANDARD)
COLOR_MAIN = '#404040'      # Main elements (dark gray)
COLOR_ACCENT = '#3333B2'    # Key concepts (purple)
COLOR_GRAY = '#B4B4B4'      # Secondary elements
COLOR_LIGHT = '#F0F0F0'     # Backgrounds
COLOR_GREEN = '#2CA02C'     # Success/positive
COLOR_RED = '#D62728'       # Error/negative
COLOR_ORANGE = '#FF7F0E'    # Warning/medium
COLOR_BLUE = '#0066CC'      # Information

# Chart generation pattern
fig, ax = plt.subplots(figsize=(10, 6))
# ... plotting code ...

# Standard styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.2, linestyle='--')

# Save with standard naming
plt.savefig('../figures/concept_name_bsc.pdf', dpi=300, bbox_inches='tight')
plt.close()  # CRITICAL: Never use plt.show()
```

### Chart Count Guidelines
| Presentation Type | Slides | Expected Charts | Ratio |
|-------------------|--------|-----------------|-------|
| Two-tier (20+15) | 35 | 12-15 | 0.34-0.43 |
| Comprehensive | 42-52 | 18-36 | 0.43-0.69 |
| Focused | 20-28 | 8-12 | 0.40-0.43 |
| Discovery-based | 36-40 | 15-20 | 0.42-0.50 |
| Enhanced (with quizzes) | 60-65 | 50+ | 0.80+ |
| Reorganized | 64 | 30+ | 0.47+ |

### Graphviz Requirements
```python
import graphviz

# Check installation: where dot
# Should show: C:\Program Files\Graphviz\bin\dot.exe

dot = graphviz.Digraph(format='pdf', engine='dot')
dot.attr(dpi='300', rankdir='LR')
# ... add nodes and edges ...
dot.render('../figures/pipeline_graphviz', cleanup=True)
```

## File Naming Conventions

### Timestamp Format (REQUIRED)
All .tex and .pdf files: `YYYYMMDD_HHMM_descriptive_name.extension`

### Version Management Workflow
```powershell
cd NLP_slides/weekXX_topic/presentations

# 1. Create previous folder if needed
mkdir previous 2>$null

# 2. Move current version to previous
mv 20251020_1200_current.tex previous/
mv 20251020_1200_current.pdf previous/

# 3. Save new with timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmm"
# Save as: ${timestamp}_weekXX_new.tex

# 4. Clean after compilation
mkdir temp 2>$null
mv *.aux,*.log,*.nav,*.out,*.snm,*.toc,*.vrb temp/
```

### Directory Organization
Keep ONLY canonical version in main folder:
```
presentations/
├── YYYYMMDD_HHMM_current.tex  ← CANONICAL
├── YYYYMMDD_HHMM_current.pdf  ← CANONICAL
├── figures/                    ← Optional: localized figures
├── previous/                   ← All older versions
└── temp/                      ← Auxiliary files
```

### Self-Contained Presentations
For portability, create local figures folder:
```powershell
# Copy required figures to presentations/figures/
mkdir presentations/figures
# Update paths in .tex from ../figures/ to figures/
```

## Educational Framework Patterns

### Discovery-Based Pedagogy (USE FOR ALL NEW CONTENT)
1. **Problem before solution** (slides 2-3)
2. **Concrete before abstract**
3. **Worked examples with actual numbers**
4. **Dual-slide pattern** (visual + detail)
5. **Checkpoint quizzes** at key points (every ~10 slides)

### Improved Pedagogical Structures

**Toolbox-First Approach** (Week 9 Example):
1. Introduce all methods/tools upfront (with mechanisms)
2. Show concrete examples immediately after each tool
3. Then explain WHY each tool is needed (problems)
4. Quiz to reinforce understanding
5. Integration and tradeoffs

**Benefits**:
- No forward references to unknown concepts
- Immediate reinforcement through examples
- Students understand available options before seeing problems

### Two-Tier vs Single-Tier Decision
**Use Two-Tier (20 main + 15 appendix)** when:
- Multiple algorithms to cover
- Deep mathematics needed
- Mixed audience levels

**Use Single-Tier (36-40 slides)** when:
- Single narrative arc
- Discovery pedagogy flow
- Heavy prerequisites needed

## Testing and Validation

```bash
# Environment check (30 seconds)
python verify_installation.py

# Test all notebooks
python test_notebooks.py

# Test specific notebook
cd NLP_slides/weekXX_topic/lab
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 notebook.ipynb
```

## Canonical Week Versions

All weeks have been restructured with discovery-based pedagogy:

1. **Foundations**: `20251026_1610_week01_foundations_bsc_discovery.pdf` (42 slides, dice analogy)
2. **Embeddings**: `20251027_0648_week02_embeddings_bsc_discovery.pdf` (35 slides, quad-hook)
3. **RNN/LSTM**: `20251002_1300_week03_lstm_bsc_21slides.pdf` (21 slides, water tank)
4. **Seq2Seq**: `20251003_1200_week04_improved.pdf` (journey structure)
5. **Transformers**: `20251006_1003_bsc_transformers_final_fixed.pdf` (3D visualizations)
6. **Pre-trained**: `20251026_1653_week06_pretrained_bsc_discovery.pdf` (52 slides, $1M cost)
7. **Advanced**: `20251027_1240_week07_advanced_bsc_discovery.pdf` (35 slides, scaling laws)
8. **Tokenization**: `20251027_1713_week08_tokenization_bsc_discovery.pdf` (35 slides, BPE)
9. **Decoding**: `20251119_0943_week09_decoding_final.pdf` (65 slides, Input→Process→Output design)
10. **Fine-tuning**: `20251007_2052_week10_finetuning_bsc_discovery.pdf` (38 slides, LoRA)
11. **Efficiency**: `20251008_0757_week11_efficiency_bsc_discovery.pdf` (41 slides, quantization)
12. **Ethics**: `20251008_0818_week12_ethics_bsc_discovery.pdf` (26 slides, bias detection)

**Special Modules:**
- **Summarization**: `20251115_0943_llm_summarization_bsc_discovery.pdf` (40 slides, discovery-based)
- **Embeddings**: `embeddings/embeddings_presentation.pdf` (48 slides, modular)
- **LSTM Primer**: `lstm_primer/lstm_primer_complete.pdf` (32 slides, modular sections)
- **ML Introduction**: `ML_Intro/20251015_1030_ml_paradigms.pdf` (paradigms overview)

## Common Pitfalls to Avoid

### LaTeX
- ❌ Figures at 0.95\textwidth → ✅ Use 0.60-0.75\textwidth
- ❌ Missing [fragile] → ✅ Add to code frames with lstlisting
- ❌ Unicode symbols → ✅ Use LaTeX commands (\checkmark, $\times$)
- ❌ Math outside mode → ✅ Use $...$
- ❌ Straight quotes → ✅ Use `` `` and '' ''

### Python
- ❌ plt.show() → ✅ plt.close()
- ❌ No backend → ✅ matplotlib.use('Agg')
- ❌ Low DPI → ✅ dpi=300
- ❌ Hardcoded paths → ✅ Relative '../figures/'
- ❌ Wrong colors → ✅ Use BSc Discovery palette

### Workflow
- ❌ Overwriting canonical → ✅ Move to previous/ first
- ❌ No timestamp → ✅ YYYYMMDD_HHMM_ prefix
- ❌ Charts after LaTeX → ✅ Generate charts FIRST
- ❌ Single pdflatex → ✅ Run twice for references
- ❌ Auxiliary files in main → ✅ Move to temp/

## Recent Improvements (November 2025)

### Week 9 Final Updates (Nov 19, 2025)
**LATEST VERSION**: Major improvements to readability and design

**Latest Changes**:
1. **Extreme Case 1 COMPLETELY REDESIGNED** (Input→Process→Output flow):
   - LEFT: Input prompt box with task description
   - CENTER: Step-by-step greedy selection with options and argmax
   - RIGHT: Output result with statistics
   - ALL FONTS INCREASED 50% for better readability
   - Removed duplicate titles and "Left-to-Right" subtitle

2. **Greedy's Fatal Flaw Chart ENHANCED** (larger fonts, better layout):
   - Figure size increased to 16×9 for more space
   - All font sizes increased 30-50% (title: 22pt, headers: 16pt, text: 13-15pt)
   - Step 2 now has LOW probability (0.15) making greedy total higher (0.076 vs 0.062)
   - Added follow-up sentence: "It was a mouse, and the hunt began."
   - Key insight prominently displayed: "Higher probability ≠ Better text quality"

3. **Full Exploration Slide SIMPLIFIED** (single tree focus):
   - Single centered tree chart at 0.65 width (reduced from 0.80)
   - Clear focus on exponential explosion visualization
   - Overflow issues fixed

4. **All Overflow Issues RESOLVED**:
   - Fixed 9 major overflows (up to 83pt reduced to under 17pt)
   - Adjusted figure widths from 0.70-0.95 to 0.60-0.70\textwidth
   - Added negative spacing where needed

**Files**:
- `20251119_1135_week09_improved_readability.pdf` - 66 slides with all improvements
- `generate_week09_enhanced_charts.py` - Updated with 50% larger fonts
- `generate_greedy_suboptimal_comparison.py` - Enhanced readability version
- `generate_full_exploration_graphviz.py` - Tree visualization (to be updated)

### Week 9 Decoding Complete Reorganization (Nov 18, 2025, morning)
**NEW PEDAGOGICAL STRUCTURE**: Toolbox+Examples → Problems → Integration

**Key Changes**:
1. **Reorganized from 70 to 64 slides** - removed redundant solution slides
2. **Methods with immediate examples** (Slides 9-26):
   - Each method introduction immediately followed by worked examples
   - No 20-slide gap between concept and application
3. **Self-contained presentation**:
   - Created local `presentations/figures/` with all 30 required PDFs
   - Updated all paths from `../figures/` to `figures/`
   - Presentation folder now completely portable
4. **Improved learning flow**:
   - Concrete examples right after abstract concepts
   - Students see HOW methods work before learning WHY needed
   - Quiz after all methods+examples for reinforcement

**Files**:
- `20251118_1100_week09_reorganized.tex` - Reorganized source
- `20251118_1100_week09_reorganized.pdf` - Final 64-slide version
- `presentations/figures/` - 30 localized charts (2.0 MB)

### Week 9 Decoding Pedagogical Restructuring (Nov 18, 2025, morning)
**COMPLETE REDESIGN** from "Problems→Solutions" to "Toolbox→Why→How" pedagogy:

**STRUCTURE** (70 slides):
1. **Title Slide**: Purple gradient beamercolorbox (template-style)
2. **Extreme Cases**: Show spectrum from greedy to full search
3. **NEW Transition Slide**: "What If We Explored More Paths?"
4. **The Toolbox**: 6 method introductions with mechanisms
5. **Quiz 1**: Match methods to mechanisms
6. **Problems Reframed**: "Why X? Problem Y" format
7. **Solutions with Examples**: Deep dive into each method

**PEDAGOGICAL RATIONALE**:
- Methods introduced first → Then see WHY needed → Then learn HOW
- Solves forward reference issues
- Discovery-based WITH scaffolding

### Week 9 Decoding Module Major Redesign (Nov 16, 2025)
- **NEW**: 4 extreme case slides showing greedy vs full search spectrum
- **REDESIGNED**: Quality-Diversity Tradeoff as scatter plot
- **REDESIGNED**: Problem 4 with 4 comprehensive graphviz charts
- **NEW**: Multiple visualization types for each problem
- **NEW**: `beam_search_tree_graphviz.pdf` - Complete beam search visualization
- **REMOVED**: Appendix B (7 slides) - consolidated into main flow

### Summarization Module Complete Redesign (Nov 15, 2025)
- 40-slide discovery-based structure
- 87 total charts (5 newly created)
- Four-part organization: Problem Space → Technical Architecture → Advanced Techniques → Evaluation
- Includes RAG enhancement and chain-of-thought reasoning