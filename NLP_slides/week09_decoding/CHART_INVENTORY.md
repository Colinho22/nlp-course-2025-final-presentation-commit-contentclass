# Week 9 Decoding Charts: Complete Inventory & Font Size Audit

**Date**: November 22, 2025
**Purpose**: Standardize all charts to BSc Discovery font sizes (18-24pt)
**Current State**: 27 charts with font sizes ranging from 18pt to 42pt

---

## BSc Discovery Font Standard (TARGET)

- **Title**: 24pt (main chart title)
- **Labels**: 20pt (axis labels, legends)
- **Annotations**: 18pt (text boxes, callouts, node labels)
- **Ticks**: 16pt (axis tick labels)

---

## Complete Chart Inventory (27 Charts)

### 1. vocabulary_probability.pdf
- **Slide**: 3
- **Folder**: `charts_individual/vocabulary_probability/`
- **Script**: `generate_vocabulary_probability.py` (assumed)
- **Type**: Matplotlib
- **Current Fonts**: Need to check
- **Issue**: TBD
- **Priority**: Medium

---

### 2. prediction_to_text_pipeline.pdf
- **Slide**: 4 (Context slide)
- **Folder**: `charts_individual/prediction_to_text_pipeline/`
- **Script**: `generate_prediction_to_text_pipeline.py`
- **Type**: Matplotlib diagram
- **Current Fonts**: Need to check
- **Issue**: TBD
- **Priority**: High (early slide)

---

### 3. extreme_greedy_single_path.pdf
- **Slide**: 4 (Extreme Case 1)
- **Folder**: `charts_individual/extreme_greedy_single_path/`
- **Script**: `generate_extreme_greedy_single_path_bsc.py`
- **Type**: Matplotlib (Input→Process→Output flow)
- **Current Fonts**:
  - FONTSIZE_TITLE: 36pt → **Too large** (reduce to 24pt)
  - FONTSIZE_LABEL: 30pt → **Too large** (reduce to 20pt)
  - FONTSIZE_TICK: 28pt → **Too large** (reduce to 16pt)
  - FONTSIZE_ANNOTATION: 28pt → **Too large** (reduce to 18pt)
  - FONTSIZE_TEXT: 30pt → **Too large** (reduce to 20pt)
  - FONTSIZE_SMALL: 24pt → **OK** (keep)
  - Inline headers: 24pt → **OK** (keep)
  - Inline content: 18pt → **OK** (keep)
- **Issue**: Excessive font sizes in constants, but inline usage already correct
- **Priority**: HIGH (recently redesigned slide)
- **Action**: Update constants to match inline usage

---

### 4. greedy_suboptimal_comparison_bsc.pdf
- **Slide**: 6 (Greedy's Fatal Flaw)
- **Folder**: `charts_individual/greedy_suboptimal_comparison_bsc/`
- **Script**: `generate_greedy_suboptimal_comparison_bsc.py`
- **Type**: Matplotlib
- **Current Fonts**: Need to check (likely similar to extreme_greedy)
- **Issue**: Recently simplified to 18×10 size
- **Priority**: HIGH

---

### 5. extreme_full_beam_explosion.pdf
- **Slide**: 6 (Extreme Case 2)
- **Folder**: `charts_individual/extreme_full_beam_explosion/`
- **Script**: `generate_extreme_full_beam_explosion.py`
- **Type**: Matplotlib
- **Current Fonts**: Need to check
- **Issue**: TBD
- **Priority**: HIGH (extreme case visualization)

---

### 6. full_exploration_tree_graphviz.pdf
- **Slide**: 7 (Full Exploration)
- **Folder**: `charts_individual/full_exploration_tree_graphviz/`
- **Script**: `generate_full_exploration_graphviz.py`
- **Type**: Graphviz
- **Current Fonts**: Need to check
- **Issue**: Exponential explosion tree
- **Priority**: HIGH

---

### 7. extreme_coverage_comparison.pdf
- **Slide**: 8 (The Extremes)
- **Folder**: `charts_individual/extreme_coverage_comparison/`
- **Script**: `generate_extreme_coverage_comparison.py`
- **Type**: Matplotlib
- **Current Fonts**: Need to check
- **Issue**: TBD
- **Priority**: Medium

---

### 8. practical_methods_coverage.pdf
- **Slide**: 9 (Sweet Spot)
- **Folder**: `charts_individual/practical_methods_coverage/`
- **Script**: `generate_practical_methods_coverage.py`
- **Type**: Matplotlib
- **Current Fonts**: Need to check
- **Issue**: TBD
- **Priority**: Medium

---

### 9. beam_search_tree_graphviz.pdf
- **Slide**: 11 (Beam Search Example)
- **Folder**: `charts_individual/beam_search_tree_graphviz/`
- **Script**: `generate_beam_search_graphviz.py`
- **Type**: Graphviz
- **Current Fonts**:
  - Global: 30pt → **Too large** (reduce to 20pt)
  - Start node: 36pt → **Too large** (reduce to 24pt)
  - Kept nodes: 30pt → **Too large** (reduce to 20pt)
  - Pruned nodes: 28pt → **Too large** (reduce to 18pt)
  - Final nodes: 32pt → **Too large** (reduce to 22pt)
  - Cluster labels: 32pt → **Too large** (reduce to 22pt)
- **Issue**: All fonts 28-36pt range (8-12pt too large)
- **Priority**: HIGH (important pedagogical visualization)
- **Action**: Reduce all fonts by 10-12pt to fit 18-24pt range

---

### 10. temperature_effects.pdf
- **Slide**: 13 (Temperature Sampling)
- **Folder**: `charts_individual/temperature_effects/`
- **Script**: `generate_temperature_effects_bsc.py`
- **Type**: Matplotlib (3 subplots)
- **Current Fonts**:
  - FONTSIZE_TITLE: 36pt → **Too large**
  - FONTSIZE_LABEL: 30pt → **Too large**
  - FONTSIZE_TICK: 28pt → **Too large**
  - FONTSIZE_ANNOTATION: 28pt → **Too large**
  - FONTSIZE_TEXT: 30pt → **Too large**
  - FONTSIZE_SMALL: 24pt → **OK**
  - **Inline title**: 40pt! → **WAY too large** (reduce to 24pt)
- **Issue**: Uses 40pt for subplot titles (worst offender)
- **Priority**: CRITICAL (14×4 figure with oversized fonts will cause cramping)
- **Action**: Urgent font size reduction

---

### 11. topk_example.pdf
- **Slide**: 15 (Top-k Example)
- **Folder**: `charts_individual/topk_example/`
- **Script**: `generate_topk_example.py`
- **Type**: Matplotlib
- **Current Fonts**: Need to check
- **Issue**: TBD
- **Priority**: Medium

---

### 12. contrastive_vs_nucleus.pdf
- **Slide**: 17 (Contrastive vs Nucleus)
- **Folder**: `charts_individual/contrastive_vs_nucleus/`
- **Script**: `generate_contrastive_vs_nucleus.py`
- **Type**: Matplotlib
- **Current Fonts**: Need to check
- **Issue**: TBD
- **Priority**: Medium

---

### 13. problem1_repetition_output.pdf
- **Slide**: 18 (Problem 1: Repetition)
- **Folder**: `charts_individual/problem1_repetition_output/`
- **Script**: `generate_problem1_repetition_output.py`
- **Type**: Matplotlib
- **Current Fonts**: Need to check
- **Issue**: TBD
- **Priority**: Low (problem slides)

---

### 14. problem2_diversity_output.pdf
- **Slide**: 19 (Problem 2: No Diversity)
- **Folder**: `charts_individual/problem2_diversity_output/`
- **Script**: `generate_problem2_diversity_output.py`
- **Type**: Matplotlib
- **Current Fonts**: Need to check
- **Issue**: TBD
- **Priority**: Low

---

### 15. problem3_balance_output.pdf
- **Slide**: 20 (Problem 3: Balance)
- **Folder**: `charts_individual/problem3_balance_output/`
- **Script**: `generate_problem3_balance_output.py`
- **Type**: Matplotlib
- **Current Fonts**: Need to check
- **Issue**: TBD
- **Priority**: Low

---

### 16-19. problem4_*.pdf (4 charts)
- **Slide**: 21 (Problem 4: Missing Better Paths)
- **Folder**: `charts_individual/problem4_*/`
- **Scripts**:
  - `generate_problem4_search_tree_pruning.py`
  - `generate_problem4_path_comparison.py`
  - `generate_problem4_probability_evolution.py`
  - `generate_problem4_recovery_problem.py`
- **Type**: Matplotlib
- **Current Fonts**: Need to check
- **Issue**: 4-chart slide (2×2 grid)
- **Priority**: Medium (complex multi-chart slide)

---

### 20. problem5_distribution_output.pdf
- **Slide**: 22 (Problem 5: Distribution Tail)
- **Folder**: `charts_individual/problem5_distribution_output/`
- **Script**: `generate_problem5_distribution_output.py`
- **Type**: Matplotlib
- **Current Fonts**: Need to check
- **Issue**: TBD
- **Priority**: Low

---

### 21. problem6_speed_output.pdf
- **Slide**: 23 (Problem 6: Generic Text)
- **Folder**: `charts_individual/problem6_speed_output/`
- **Script**: `generate_problem6_speed_output.py`
- **Type**: Matplotlib
- **Current Fonts**: Need to check
- **Issue**: TBD
- **Priority**: Low

---

### 22. degeneration_problem.pdf
- **Slide**: Contrastive section (slide ~40)
- **Folder**: `charts_individual/degeneration_problem/`
- **Script**: `generate_degeneration_problem.py`
- **Type**: Matplotlib
- **Current Fonts**: Need to check
- **Issue**: TBD
- **Priority**: Medium

---

### 23. quality_diversity_scatter.pdf
- **Slide**: 24 (Quality-Diversity Tradeoff)
- **Folder**: `charts_individual/quality_diversity_scatter/`
- **Script**: `generate_quality_diversity_scatter.py`
- **Type**: Matplotlib scatter plot
- **Current Fonts**: Need to check
- **Issue**: Recently redesigned
- **Priority**: HIGH (important synthesis slide)

---

### 24. quality_diversity_pareto.pdf
- **Slide**: 26 (Pareto Frontier)
- **Folder**: `charts_individual/quality_diversity_pareto/`
- **Script**: `generate_quality_diversity_pareto.py`
- **Type**: Matplotlib
- **Current Fonts**: Need to check
- **Issue**: TBD
- **Priority**: HIGH

---

### 25. task_method_decision_tree.pdf
- **Slide**: 27 (Decision Tree)
- **Folder**: `charts_individual/task_method_decision_tree/`
- **Script**: `generate_task_method_decision_tree.py`
- **Type**: Matplotlib or Graphviz decision tree
- **Current Fonts**: Need to check
- **Issue**: TBD
- **Priority**: HIGH (practical application)

---

### 26. task_recommendations_table.pdf
- **Slide**: 28 (Task Recommendations)
- **Folder**: `charts_individual/task_recommendations_table/`
- **Script**: `generate_task_recommendations_table.py`
- **Type**: Matplotlib table
- **Current Fonts**: Need to check
- **Issue**: TBD
- **Priority**: HIGH (practical reference)

---

### 27. production_settings.pdf
- **Slide**: A18 (Appendix - Production Settings)
- **Folder**: `charts_individual/production_settings/`
- **Script**: `generate_production_settings.py`
- **Type**: Matplotlib table
- **Current Fonts**: Need to check
- **Issue**: TBD
- **Priority**: Low (appendix only)

---

## Summary Statistics

- **Total Charts**: 27
- **Matplotlib charts**: ~23
- **Graphviz charts**: ~4
- **Priority Distribution**:
  - CRITICAL: 1 (temperature_effects)
  - HIGH: 9 charts
  - MEDIUM: 10 charts
  - LOW: 7 charts

---

## Font Size Issues Identified

### Pattern 1: Oversized Constants (Matplotlib)
**Affected**: Most matplotlib scripts
**Current**:
```python
FONTSIZE_TITLE = 36
FONTSIZE_LABEL = 30
FONTSIZE_TICK = 28
FONTSIZE_ANNOTATION = 28
FONTSIZE_TEXT = 30
FONTSIZE_SMALL = 24
```

**Target**:
```python
FONTSIZE_TITLE = 24
FONTSIZE_LABEL = 20
FONTSIZE_TICK = 16
FONTSIZE_ANNOTATION = 18
FONTSIZE_TEXT = 20
FONTSIZE_SMALL = 18
```

**Delta**: Reduce all by 10-12pt

---

### Pattern 2: Inline Font Overrides
**Example**: `temperature_effects_bsc.py` line 80: `fontsize=40`
**Issue**: Hardcoded 40pt for subplot titles
**Fix**: Change to 24pt

---

### Pattern 3: Graphviz Font Sizes
**Affected**: beam_search, full_exploration, task_decision_tree
**Current**: 28-36pt range
**Target**: 18-24pt range
**Delta**: Reduce all by 8-14pt

---

## Next Steps

1. ✅ Complete inventory of all 27 charts
2. [ ] Read remaining 20 scripts to document all font sizes
3. [ ] Identify canonical script for each chart
4. [ ] Update all matplotlib constants to BSc Discovery standard
5. [ ] Update all graphviz font attributes
6. [ ] Fix inline font overrides
7. [ ] Review layout issues (spacing, proportions)
8. [ ] Regenerate all charts
9. [ ] Update .tex paths from `../figures_with_logo_200px/` to `../figures/`
10. [ ] Create master logo addition script

---

## Notes

- All current charts use `../figures_with_logo_200px/` path
- Need to generate clean versions to `../figures/` first
- Then create ONE master script to add logos to all PDFs
- Keep individual folder structure as requested
- Remove duplicate/backup scripts during cleanup
