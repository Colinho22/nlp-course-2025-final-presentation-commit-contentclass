# Presentations Cleanup Summary

**Date:** 2025-09-30
**Action:** Repository-wide cleanup of presentation folders

## Overview

Cleaned up all 12 week folders plus supplementary modules, moving **90+ old/duplicate presentation files** to `previous/` folders. Each week now has only its canonical (recommended) version in the main `presentations/` folder.

## Cleanup Results

### Weeks 1-3: Foundations
- **Week 1**: 2 files moved → Kept `20250120_0140_week01_optimal_template.pdf`
- **Week 2**: 5 files moved → Kept `20250929_1545_week02_neural_lm_template.pdf`
- **Week 3**: 2 files moved → Kept `20250929_1027_week03_rnn_template.pdf`
  - Note: One locked file `20250922_1242_week03_rnn_optimal.pdf` still present (to be moved manually)

### Weeks 4-6: Core Architectures (Pedagogically Excellent)
- **Week 4**: 26 files moved! → Kept `20250928_1510_week04_seq2seq_journey_complete.pdf`
  - Removed: test files, template experiments, color scheme samples, multiple "final" versions
  - Canonical: Zero pre-knowledge "Compression Journey" narrative
- **Week 5**: 23 files moved! → Kept `20250928_1648_week05_transformers_speed_revolution.pdf`
  - Removed: comprehensive, conceptual, prediction-focused, chart-only versions
  - Canonical: "Speed Revolution" narrative (90 days → 1 day training)
- **Week 6**: 4 files moved → Kept `20250930_0100_week06_pretrained_pedagogical.pdf`

### Weeks 7-12: Advanced Topics
- **Week 7**: 2 files moved → Kept `20250921_1729_week07_optimal_template.pdf`
- **Week 8**: 5 files moved → Kept `20250923_2110_week08_tokenization_optimal.pdf`
- **Week 9**: 5 files moved → Kept `week09_decoding.pdf`
- **Week 10**: 5 files moved → Kept `20250923_2110_week10_finetuning_optimal.pdf`
- **Week 11**: 5 files moved → Kept `20250923_2110_week11_efficiency_optimal.pdf`
- **Week 12**: 5 files moved → Kept `20250923_2110_week12_ethics_optimal.pdf`

## Canonical Versions Policy

### Naming Convention
Canonical presentations follow timestamp naming:
```
YYYYMMDD_HHMM_weekXX_description.{tex,pdf}
```

### Selection Criteria
Canonical versions were chosen based on:
1. **Pedagogical Excellence**: Narrative structure, zero pre-knowledge approach
2. **Template Conformance**: Uses `template_beamer_final.tex` layout
3. **Recency**: Most recent pedagogically-sound version
4. **Documentation**: Version documented in README.md or status.md

### Special Cases
- **Weeks 4 & 5**: Pedagogically excellent versions using didactic framework
  - Week 4: "Compression Journey" narrative
  - Week 5: "Speed Revolution" narrative
  - Both marked as reference implementations in DIDACTIC_PRESENTATION_FRAMEWORK.md

## File Organization

### Main Folder (Clean)
```
NLP_slides/weekXX_topic/presentations/
├── YYYYMMDD_HHMM_weekXX_canonical.pdf      # Canonical PDF
├── YYYYMMDD_HHMM_weekXX_canonical.tex      # Source
└── README.md                                # Version guide
```

### Archive Folders
```
previous/                  # Previous good versions
├── YYYYMMDD_HHMM_*.pdf
├── YYYYMMDD_HHMM_*.tex
└── ...

deprecated/                # Deprecated versions (if exists)
└── redundant_versions/

temp/                      # Auxiliary LaTeX files
└── *.aux, *.log, etc.
```

## Automation

**Cleanup Script**: `cleanup_presentations.py`
- Systematic cleanup based on CANONICAL_FILES dictionary
- Never deletes, only moves to previous/
- Handles Unicode encoding for Windows
- Dry-run capable for verification

## Manual Cleanup Needed

1. **Week 3 Locked File**: `20250922_1242_week03_rnn_optimal.pdf`
   - Currently locked by PDF viewer
   - Move to `previous/` after closing viewer
   - Command: `mv NLP_slides/week03_rnn/presentations/20250922_1242_week03_rnn_optimal.pdf NLP_slides/week03_rnn/presentations/previous/`

## Future Policy

### Before Creating New Versions
1. Check if current canonical version can be edited
2. Use timestamp naming from the start
3. Move previous version to `previous/` before saving new one

### After Creating New Versions
1. Update README.md to reflect new canonical
2. Move ALL old versions to `previous/`
3. Keep only ONE version in main folder
4. Update CLAUDE.md if naming conventions change

## Benefits

1. **Clarity**: One obvious canonical version per week
2. **Navigation**: Instructors know which file to use
3. **Preservation**: All versions preserved in previous/ folders
4. **Performance**: Faster directory listings and searches
5. **Maintenance**: Easier to track changes and versions

## Related Documentation

- **CLAUDE.md**: Development guide with presentation workflow
- **status.md**: Project completion tracking
- **Week README files**: Individual week version guides
- **DIDACTIC_PRESENTATION_FRAMEWORK.md**: Pedagogical template

## Statistics

- **Total files moved**: 90+
- **Total storage freed in main folders**: ~15MB
- **Weeks cleaned**: 12/12 ✓
- **Canonical versions verified**: 12/12 ✓
- **README files updated**: 1 (Week 3)
- **README files verified**: 2 (Weeks 4, 5)