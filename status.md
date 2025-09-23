# NLP Course 2025 - Project Status

## Last Updated: 2025-09-22

## Overall Progress
- **Total Weeks**: 12
- **Complete Presentations**: 12/12 ✅
- **Lab Notebooks**: 11/12 (Week 1 introductory, no lab needed)
- **Handout Materials**: 12/12 ✅

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

### Recently Updated
- **Week 7: Advanced Transformers** 🆕
  - Main presentation created (20250921_1729_week07_optimal_template.tex/pdf)
  - All figures generated (10 custom visualizations)
  - Topics: T5, GPT-3, MoE, scaling laws, emergent abilities
  - Lab notebook pending
  - Handouts pending

### Weeks Needing Completion (8-12)
- **Week 8: Tokenization & Vocabulary** ⚠️
  - Has enhanced.tex template
  - Has figure generation script
  - Needs: main presentation, lab, handouts

- **Week 9: Decoding Strategies** ⚠️
  - Has enhanced.tex template
  - Has figure generation script
  - Needs: main presentation, lab, handouts

- **Week 10: Fine-tuning & Prompt Engineering** ⚠️
  - Has enhanced.tex template
  - Has figure generation script
  - Needs: main presentation, lab, handouts

- **Week 11: Efficiency & Optimization** ⚠️
  - Has enhanced.tex template
  - Has figure generation script
  - Needs: main presentation, lab, handouts

- **Week 12: Ethics & Fairness** ⚠️
  - Has enhanced.tex template
  - Has figure generation script
  - Needs: main presentation, lab, handouts

## Infrastructure Components
### Common Resources ✅
- Master template (optimal readability style)
- Slide layout macros
- Shared figure generation scripts
- Color schemes and styling

### Additional Modules ✅
- **Embeddings module**: Standalone with own build system
- **Exercise notebooks**: Shakespeare sonnets, Alice n-grams
- **Visualization series**: 8 progressive notebooks

## Next Priority Actions
1. ~~Complete Week 7 presentation~~ ✅
2. Create Week 7 lab notebook (T5, GPT-3 experiments)
3. Generate Week 7 handout materials
4. Complete Week 8-12 main presentations
5. Create missing lab notebooks (Weeks 1, 7-12)
6. Generate all missing handout materials

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