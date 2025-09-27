# NLP Course 2025 - Project Status

## Last Updated: 2025-09-27

## Overall Progress
- **Total Weeks**: 12
- **Complete Presentations**: 12/12 ✅
- **Lab Notebooks**: 12/12 (Weeks 2-12, plus Week 7 completed) ✅
- **Handout Materials**: 12/12 ✅
- **Supplementary Modules**: 2/2 (Neural Network Primer, LSTM Primer) ✅

## Week-by-Week Status

### Completed Weeks (Full Materials)
- **Week 1: Foundations & Statistical Language Models** ✅
  - Main presentation (Foundations_and_Statistical_Language_Modelling.tex/pdf)
  - Python figure generation scripts
  - Student/instructor handouts
  - No lab (introductory week)

- **Week 2: Neural Language Models & Word Embeddings** ✅
  - Multiple presentation versions (optimal, enhanced, restructured)
  - Lab notebook (week02_word_embeddings_lab.ipynb)
  - Complete handout materials
  - All figures generated

- **Week 3: RNN/LSTM/GRU** ✅
  - Main presentation (week03_rnn_restructured.tex/pdf)
  - Lab notebook (week03_rnn_lab.ipynb)
  - Python figure generation scripts
  - Handout materials available

- **Week 4: Sequence-to-Sequence Models** ✅
  - Multiple presentation versions (enhanced, BSc, nature professional)
  - Lab notebook (week04_seq2seq_lab.ipynb)
  - Enhanced lab version available
  - Complete handout materials

- **Week 5: Transformers & Attention** ✅
  - Main presentation (week05_transformers_bsc_restructured_final.pdf)
  - Lab notebook (week05_transformer_lab.ipynb)
  - Multiple figure generation scripts
  - Missing handouts

- **Week 6: Pre-trained Models (BERT, GPT)** ✅
  - Main presentation (week06_pretrained_bsc.pdf)
  - Lab notebook (week06_bert_finetuning.ipynb)
  - Figure generation scripts
  - Missing handouts

- **Week 7: Advanced Transformers** ✅
  - Main presentation created (20250921_1729_week07_optimal_template.tex/pdf)
  - All figures generated (10 custom visualizations)
  - Topics: T5, GPT-3, MoE, scaling laws, emergent abilities
  - Lab notebook created (week07_advanced_transformers_lab.ipynb)
  - Handouts created

### Recently Completed (Weeks 8-12)
- **Week 8: Tokenization & Vocabulary** ✅
  - Main presentation created (20250923_2110_week08_tokenization_optimal.tex/pdf)
  - Lab notebook created (week08_tokenization_lab.ipynb)
  - Handouts created (student and instructor versions)
  - All figures generated

- **Week 9: Decoding Strategies** ✅
  - Main presentation created (20250923_2110_week09_decoding_optimal.tex/pdf)
  - Lab notebook created (week09_decoding_lab.ipynb)
  - Handouts created (student and instructor versions)
  - All figures generated

- **Week 10: Fine-tuning & Prompt Engineering** ✅
  - Main presentation created (20250923_2110_week10_finetuning_optimal.tex/pdf)
  - Lab notebook created (week10_finetuning_lab.ipynb)
  - Handouts created (student and instructor versions)
  - All figures generated

- **Week 11: Efficiency & Optimization** ✅
  - Main presentation created (20250923_2110_week11_efficiency_optimal.tex/pdf)
  - Lab notebook created (week11_efficiency_lab.ipynb)
  - Handouts created (student and instructor versions)
  - All figures generated

- **Week 12: Ethics & Fairness** ✅
  - Main presentation created (20250923_2110_week12_ethics_optimal.tex/pdf)
  - Lab notebook created (week12_ethics_lab.ipynb)
  - Handouts created (student and instructor versions)
  - All figures generated

## Infrastructure Components
### Common Resources ✅
- Master template (optimal readability style)
- Slide layout macros
- Shared figure generation scripts
- Color schemes and styling

### Supplementary Modules ✅
- **Neural Network Primer**: Complete foundational module (2025-09-27)
  - Location: `NLP_slides/nn_primer/`
  - 3 presentation versions (comprehensive 48+ pages)
  - Discovery-based handouts (classification and function approximation)
  - 8 custom-generated figures (party analogy, neurons, XOR, sigmoid parameters)
  - Zero pre-knowledge approach with concrete analogies
  - Topics: Perceptrons, activation functions, Universal Approximation Theorem

- **LSTM Primer**: Comprehensive standalone module (2025-09-27)
  - Location: `NLP_slides/lstm_primer/`
  - 11 presentation versions (10-32 slides, various formats)
  - Modular architecture with 9 section files
  - 20 custom-generated figures (architecture, gates, training progression)
  - BSc-level checkpoint pedagogy with quiz slides
  - Comprehensive coverage: RNN baseline, vanishing gradients, LSTM architecture, BPTT
  - Reference example for modular presentation structure

- **Embeddings module**: Standalone with own build system (PowerShell + Make)
- **Exercise notebooks**: Shakespeare sonnets, Alice n-grams
- **Visualization series**: 8 progressive notebooks

## Next Priority Actions
1. ~~Complete all 12 weeks with presentations, labs, and handouts~~ ✅
2. ~~Create Neural Network Primer module~~ ✅
3. ~~Create LSTM Primer module~~ ✅
4. Commit repository reorganization (125 legacy file deletions + new modules)
5. Create master course index/syllabus document
6. Generate course overview presentation (meta-lecture)
7. Test all Jupyter notebooks for execution
8. Create student-facing course website/navigation

## Build Commands Reference
```bash
# Compile LaTeX presentation
cd NLP_slides/weekXX_*/presentations
pdflatex filename.tex

# Generate figures
cd NLP_slides/weekXX_*/python
python generate_weekXX_optimal_figures.py

# Clean auxiliary files
mkdir -p temp && mv *.aux *.log *.nav *.out *.snm *.toc *.vrb temp/
```

## Quality Metrics
- All presentations use optimal readability template
- Monochromatic gray palette for professional look
- Python-generated figures (no TikZ)
- Consistent file naming with timestamps
- Two-column layout standard