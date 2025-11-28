# Week 5: Transformers and Attention

## Canonical Presentation

**PRIMARY VERSION FOR TEACHING:**
- **File:** `presentations/20250928_1648_week05_transformers_speed_revolution.pdf`
- **Size:** 28 slides, 199KB
- **Status:** PEDAGOGICALLY EXCELLENT
- **Last Updated:** 2025-09-28

### Why This Version

This presentation was substantially restructured (2025-09-28) using the didactic framework extracted from Week 4's pedagogical success. It transforms the transformer architecture from a technical reference into a compelling narrative journey.

**Key Pedagogical Features:**
- **Four-Act Dramatic Structure:** Hope → Disappointment → Breakthrough → Impact
- **Zero Pre-Knowledge:** Builds speed problem from concrete GPU calculations (3.9 years training time)
- **All 8 Critical Pedagogical Beats Present:**
  1. First Success (pure attention works on short sentences)
  2. Failure Pattern Emerges (90% quality drop on long sequences)
  3. Diagnosing Root Cause (permutation invariance traced with examples)
  4. Human Introspection ("How do YOU know word order?")
  5. Hypothesis (conceptual comparison: compression vs selection)
  6. Zero-Jargon Explanation (actual numbers: [0.3,0.2] + [0.1,0.0])
  7. Full Numerical Walkthrough (every calculation shown for "The cat sat")
  8. Experimental Validation (real Vaswani et al. 2017 results)

**Unified Narrative:**
- "The Waiting Game" metaphor throughout
- Speed problem quantified: 90 days (RNN) → 45 days (RNN+Attention) → 1 day (Transformer)
- Connects to modern applications: ChatGPT only possible due to 100x speedup
- GPU utilization improvement: 2% → 92%

**Concrete-to-Abstract Progression:**
- Starts with GPU wastage ($10,000 hardware doing $96 work)
- Builds attention from dot product similarity with 2D visualization
- Shows full numerical example before mathematical formulas
- Connects transformer architecture to sequential processing bottleneck

**Pedagogical Gap (Addressed 2025-09-28):**
- **Previous State:** Zero charts in canonical presentation despite 9 available figures
- **Impact:** Visual learners underserved; complex concepts (GPU utilization, positional encoding, attention mechanism) lacked visual reinforcement
- **Solution:** Integrated 12 pedagogically-aligned charts at critical narrative moments:
  - Act 1: Training time comparison, GPU utilization waste, sequential bottleneck
  - Act 2: Attention success heatmap, quality degradation curve, permutation invariance
  - Act 3: Positional encoding waves, vector addition, self-attention algorithm, numerical walkthrough
  - Act 4: Speed vs quality scatter, modern applications timeline
- **Result:** Enhanced visual learning without disrupting narrative flow; each critical pedagogical beat gets visual anchor

## Alternative Versions

### Comprehensive Technical Reference
- **File:** `presentations/20250924_1540_week05_transformers_comprehensive.pdf`
- **Size:** 46 slides, 604KB
- **Purpose:** Detailed technical documentation
- **Use Case:** Reference material, grad students, comprehensive coverage
- **Note:** Linear structure without narrative arc

### BSc Final Version
- **File:** `presentations/week05_transformers_bsc_restructured_final.pdf`
- **Size:** 4.9MB (includes many figures)
- **Purpose:** BSc-level comprehensive reference
- **Use Case:** Self-contained teaching with all visualizations embedded

### Legacy Compiled Version
- **File:** `presentations/20250120_0857_week05_compiled.pdf`
- **Size:** 323KB
- **Purpose:** Previous version before framework application
- **Status:** Superseded by Speed Revolution version

## Course Materials Structure

### Presentations
```
presentations/
├── 20250928_1648_week05_transformers_speed_revolution.pdf [CANONICAL]
├── 20250928_1648_week05_transformers_speed_revolution.tex
├── 20250924_1540_week05_transformers_comprehensive.pdf [REFERENCE]
└── week05_transformers_bsc_restructured_final.pdf [BSC VERSION]
```

### Lab Materials
```
lab/
└── week05_transformer_lab.ipynb - Hands-on transformer implementation
```

### Figures
```
figures/
├── attention_heatmap.pdf
├── encoder_decoder.pdf
├── model_comparison.pdf
├── sequence_analysis.pdf
└── training_dashboard.pdf
```

### Python Scripts
```
python/
├── generate_week5_optimal_figures.py
└── generate_week5_transformer_charts.py
```

## Teaching Recommendations

### For 90-Minute Lecture
Use the **Speed Revolution** version (28 slides):
- Act 1 (Slides 1-5): Establish speed problem with concrete numbers
- Act 2 (Slides 6-11): Show pure attention success and failure pattern
- Act 3 (Slides 12-21): Positional encoding breakthrough with full beats
- Act 4 (Slides 22-28): Connect to modern AI and implementation

**Timing Guidance:**
- Act 1: 10 minutes (establish crisis)
- Act 2: 15 minutes (build hope then disappointment)
- Act 3: 40 minutes (detailed breakthrough explanation)
- Act 4: 15 minutes (synthesis and modern context)
- Questions: 10 minutes

### For 2-Hour Workshop
Combine Speed Revolution presentation with transformer lab notebook:
- First hour: Complete narrative presentation
- Second hour: Hands-on implementation in `lab/week05_transformer_lab.ipynb`

### For Graduate Course
Use Comprehensive Technical Reference for detailed coverage, then Speed Revolution for conceptual clarity.

## Learning Objectives

After this week, students should be able to:
1. Explain why RNNs are slow (sequential dependency, low GPU utilization)
2. Understand self-attention mechanism with concrete numerical examples
3. Explain positional encoding and why it's necessary
4. Trace full transformer forward pass with calculations
5. Compare training efficiency: RNN vs Transformer (100x speedup)
6. Connect transformer architecture to modern LLMs (GPT, BERT)

## Prerequisites

Students should understand before this week:
1. Word embeddings (Week 2)
2. RNNs and sequential processing (Week 3)
3. Sequence-to-sequence models and attention (Week 4)
4. Dot product and vector similarity
5. Basic matrix operations

## Key Papers Referenced

1. Vaswani et al. (2017) - "Attention Is All You Need"
   - Original transformer paper
   - WMT English-German results cited in Slide 21
   - Training time comparisons: 90 days → 1 day

2. Bahdanau et al. (2015) - Neural Machine Translation
   - RNN+Attention baseline for comparison
   - 45-day training time reference

## Building PDFs

### Compile Speed Revolution Version
```bash
cd presentations
pdflatex 20250928_1648_week05_transformers_speed_revolution.tex
pdflatex 20250928_1648_week05_transformers_speed_revolution.tex  # Run twice for references
mkdir -p temp && mv *.aux *.log *.nav *.out *.snm *.toc *.vrb temp/
```

### Generate Figures
```bash
cd python
python generate_week5_optimal_figures.py
```

## Version History

### 2025-09-28: Speed Revolution [CANONICAL]
- Complete restructure using didactic framework
- Four-act dramatic structure with all 8 critical beats
- Zero pre-knowledge approach from GPU calculations
- 28 slides, 199KB
- Pedagogically excellent status

### 2025-09-24: Comprehensive Reference
- 46-slide technical documentation
- Linear problem-solution structure
- Complete coverage but no narrative arc

### 2025-01-20: Legacy Version
- Previous canonical version
- Replaced by Speed Revolution

## Didactic Framework Application

This presentation serves as the second reference implementation (after Week 4 Seq2seq) of the didactic framework documented in:
- `DIDACTIC_PRESENTATION_FRAMEWORK.md` (project root)
- `DIDACTIC_PRESENTATION_FRAMEWORK.html` (styled version)

The framework has been successfully applied to transform technical content into pedagogically excellent narrative journeys. Week 5 demonstrates the framework's applicability to different topics while maintaining the same structural principles.

## Contact and Contributions

For questions about pedagogical approach or framework application, refer to project-level documentation in:
- `status.md` - Current project status
- `changelog.md` - Detailed change history
- `DIDACTIC_PRESENTATION_FRAMEWORK.md` - Reusable pedagogical template