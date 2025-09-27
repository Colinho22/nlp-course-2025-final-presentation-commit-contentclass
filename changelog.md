# NLP Course 2025 - Changelog

## 2025-09-27
### MAJOR MILESTONE: Supplementary Modules Complete ✅
- **LSTM Primer Module** - Comprehensive standalone module
  - Created 11 presentation versions (10-32 slides)
  - Modular architecture with 9 section files in sections/ subdirectory
  - Generated 20 custom figures (architecture diagrams, gate visualizations, training progression)
  - BSc-level checkpoint pedagogy with quiz slides and answers
  - Complete coverage: autocomplete challenge, RNN baseline, vanishing gradients, LSTM architecture, BPTT, applications
  - Reference example for modular presentation structure
  - Location: NLP_slides/lstm_primer/

- **Neural Network Primer Module** - Foundational zero-knowledge module
  - 3 comprehensive presentation versions (48+ pages)
  - Discovery-based handouts: classification (6 pages) and function approximation (9 pages)
  - Generated 8 custom figures (party analogy, neuron computation, XOR solution, sigmoid parameters)
  - Concrete-to-abstract pedagogy: party decisions to neural networks
  - Topics: perceptrons, activation functions, backpropagation, Universal Approximation Theorem
  - Location: NLP_slides/nn_primer/

### Repository Status
- **All 12 core weeks**: Complete with presentations, labs, and handouts
- **Supplementary modules**: 2/2 complete (NN Primer, LSTM Primer)
- **Lab notebooks**: 12/12 (including Week 7 advanced transformers lab)
- **Documentation**: Updated status.md and CLAUDE.md

### Pending Commit
- 125 legacy file deletions (cleanup of standalone presentations directory)
- New untracked modules: lstm_primer/, nn_primer/, addendum/, handouts/
- New presentation PDFs: Week 5 comprehensive, Week 10, Week 12 optimal versions
- New figures: 6 new visualization PDFs in figures/ directory

## 2025-09-23
### Completed and Tested
- **Fixed LaTeX compilation issues** for all week 8-12 presentations
  - Escaped ampersands in titles (Tokenization \& Vocabulary, etc.)
  - Successfully compiled Week 8 presentation as validation
  - Cleaned auxiliary files to temp/ directories
- **Generated all figures** for weeks 8-12 using Python scripts
- **Repository organization** completed
  - Committed 479 files in major reorganization
  - Migrated from standalone presentations to week-based structure
  - Added common templates and shared infrastructure

### Documentation Updates
- **Enhanced CLAUDE.md** with:
  - Full course generation script documentation
  - Directory naming patterns clarification
  - Important files and utilities section
  - Shared infrastructure details (master_template.tex, chart_utils.py)
- **Updated status.md** to reflect completion of weeks 8-12
- **Updated changelog.md** with recent changes

### Verified
- All 12 weeks now have complete materials (presentations, labs, handouts)
- Lab notebooks exist for weeks 2-12 (Week 1 excluded as introductory)
- All figure generation scripts have been executed successfully

## 2025-09-22
### MAJOR COMPLETION MILESTONE 🎯
- **ALL 12 WEEKS NOW COMPLETE**
- Generated all remaining course materials using automated script

### Added - Weeks 8-12
- **Week 8: Tokenization & Vocabulary**
  - Main presentation (optimal template)
  - Lab notebook for tokenization experiments
  - Student and instructor handouts
  - Figure generation script

- **Week 9: Decoding Strategies**
  - Main presentation (optimal template)
  - Lab notebook for decoding methods
  - Student and instructor handouts
  - Figure generation script

- **Week 10: Fine-tuning & Prompt Engineering**
  - Main presentation (optimal template)
  - Lab notebook for fine-tuning techniques
  - Student and instructor handouts
  - Figure generation script

- **Week 11: Efficiency & Optimization**
  - Main presentation (optimal template)
  - Lab notebook for model optimization
  - Student and instructor handouts
  - Figure generation script

- **Week 12: Ethics & Fairness**
  - Main presentation (optimal template)
  - Lab notebook for bias detection
  - Student and instructor handouts
  - Figure generation script

### Added - Missing Handouts
- Created handouts for Weeks 5-6
- All weeks now have complete instructor/student handout materials

### Infrastructure
- Created automated generation script for rapid material creation
- Ensured consistent structure across all weeks
- All materials follow optimal readability template standard

## 2025-09-21
### Added
- **Week 7: Advanced Transformers** complete presentation
  - Created main presentation using optimal readability template
  - Generated 10 custom figures:
    - Emergent abilities chart
    - T5 span corruption visualization
    - GPT-3 performance metrics
    - Architecture comparison (encoder/decoder/encoder-decoder)
    - Mixture of Experts (MoE) architecture
    - 3D parallelism visualization
    - Compute scaling timeline
    - GPT-3 applications ecosystem
    - Alignment challenge diagram
    - Model explosion timeline
  - Topics covered: T5, GPT-3, scaling laws, emergent abilities, MoE, training infrastructure
  - Successfully compiled to PDF

### Infrastructure
- Created project status.md for tracking progress
- Created changelog.md for version history
- Fixed Python script Unicode issues for Windows compatibility
- Fixed LaTeX compilation issues (fragile frames with lstlisting)

### Repository Organization
- Established clear week-by-week structure
- Identified gaps in materials (Weeks 8-12 need completion)
- Documented build commands and workflows

## 2025-09-20
### Modified
- Week 2 Neural Language Models presentations
- Updated figure generation scripts

## 2025-09-12
### Initial Setup
- Repository structure created
- 12-week course framework established
- Common infrastructure:
  - Master template with optimal readability
  - Slide layout macros
  - Color schemes
- Weeks 1-6 initial materials imported
- Python figure generation scripts for all weeks
- Lab notebooks for weeks 2-6

## 2025-07-25
### Legacy Materials
- Original enhanced presentations created
- Initial figure generation scripts
- Basic course structure

## Key Improvements Made
1. **Standardization**: All presentations now use consistent optimal readability template
2. **Visual Quality**: Python-generated figures replace TikZ for better quality
3. **Organization**: Clear directory structure with presentations/, python/, figures/, lab/ folders
4. **Documentation**: Comprehensive status tracking and changelog
5. **Build Process**: Streamlined compilation with auxiliary file management

## Known Issues
1. Weeks 8-12 missing main presentations (have enhanced templates)
2. Lab notebooks needed for weeks 1, 7-12
3. Handout materials missing for weeks 5-12
4. Some figures referenced but not yet generated for later weeks

## Upcoming Priorities
1. Week 7 lab notebook creation
2. Week 7 handout materials
3. Week 8 Tokenization complete presentation
4. Systematic completion of weeks 8-12
5. Creation of all missing lab notebooks

## Technical Notes
- Using pdfLaTeX for compilation
- Python 3.x with matplotlib, seaborn for figures
- Madrid beamer theme with custom color scheme
- 8pt font size, 16:9 aspect ratio
- Monochromatic palette: RGB(64,64,64), RGB(180,180,180), RGB(240,240,240)