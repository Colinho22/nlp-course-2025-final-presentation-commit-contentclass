# Week 9 Decoding Charts: Canonical Scripts Mapping

**Date**: November 22, 2025
**Purpose**: Identify the official generation script for each chart

---

## Canonical Scripts (27 Charts)

| # | Chart PDF | Folder | Canonical Script | Type | Size |
|---|-----------|--------|------------------|------|------|
| 1 | vocabulary_probability.pdf | vocabulary_probability | `generate_vocabulary_probability_bsc.py` | Matplotlib | 3,432 B |
| 2 | prediction_to_text_pipeline.pdf | prediction_to_text_pipeline | `generate_prediction_to_text_pipeline_bsc.py` | Matplotlib | 4,297 B |
| 3 | extreme_greedy_single_path.pdf | extreme_greedy_single_path | `generate_extreme_greedy_single_path_bsc.py` | Matplotlib | 7,874 B |
| 4 | greedy_suboptimal_comparison_bsc.pdf | greedy_suboptimal_comparison_bsc | `generate_greedy_suboptimal_comparison.py` | Matplotlib | 8,601 B |
| 5 | extreme_full_beam_explosion.pdf | extreme_full_beam_explosion | `generate_extreme_full_beam_explosion_bsc.py` | Matplotlib | 4,697 B |
| 6 | full_exploration_tree_graphviz.pdf | full_exploration_tree_graphviz | `generate_full_exploration_graphviz.py` | Graphviz | 6,728 B |
| 7 | extreme_coverage_comparison.pdf | extreme_coverage_comparison | `generate_extreme_coverage_comparison_bsc.py` | Matplotlib | 3,912 B |
| 8 | practical_methods_coverage.pdf | practical_methods_coverage | `generate_practical_methods_coverage_bsc.py` | Matplotlib | 5,609 B |
| 9 | beam_search_tree_graphviz.pdf | beam_search_tree_graphviz | `generate_beam_search_graphviz.py` | Graphviz | 6,746 B |
| 10 | temperature_effects.pdf | temperature_effects | `generate_temperature_effects_bsc.py` | Matplotlib | 3,171 B |
| 11 | topk_example.pdf | topk_example | `generate_topk_example_bsc.py` | Matplotlib | 4,072 B |
| 12 | contrastive_vs_nucleus.pdf | contrastive_vs_nucleus | `generate_contrastive_vs_nucleus_bsc.py` | Matplotlib | 4,023 B |
| 13 | problem1_repetition_output.pdf | problem1_repetition_output | `generate_problem1_repetition_output_bsc.py` | Matplotlib | 3,517 B |
| 14 | problem2_diversity_output.pdf | problem2_diversity_output | `generate_problem2_diversity_output_bsc.py` | Matplotlib | 3,527 B |
| 15 | problem3_balance_output.pdf | problem3_balance_output | `generate_problem3_balance_output_bsc.py` | Matplotlib | 3,356 B |
| 16 | problem4_search_tree_pruning.pdf | problem4_search_tree_pruning | `generate_problem4_search_tree_pruning_bsc.py` | Matplotlib | 2,131 B |
| 17 | problem4_path_comparison.pdf | problem4_path_comparison | `generate_problem4_path_comparison_bsc.py` | Matplotlib | 2,230 B |
| 18 | problem4_probability_evolution.pdf | problem4_probability_evolution | `generate_problem4_probability_evolution_bsc.py` | Matplotlib | 1,695 B |
| 19 | problem4_recovery_problem.pdf | problem4_recovery_problem | `generate_problem4_recovery_problem_bsc.py` | Matplotlib | 1,968 B |
| 20 | problem5_distribution_output.pdf | problem5_distribution_output | `generate_problem5_distribution_output_bsc.py` | Matplotlib | 3,924 B |
| 21 | problem6_speed_output.pdf | problem6_speed_output | `generate_problem6_speed_output_bsc.py` | Matplotlib | 3,524 B |
| 22 | degeneration_problem.pdf | degeneration_problem | `generate_degeneration_problem_bsc.py` | Matplotlib | 3,504 B |
| 23 | quality_diversity_scatter.pdf | quality_diversity_scatter | `generate_quality_diversity_scatter_bsc.py` | Matplotlib | 5,167 B |
| 24 | quality_diversity_pareto.pdf | quality_diversity_pareto | `generate_quality_diversity_pareto_bsc.py` | Matplotlib | 4,120 B |
| 25 | task_method_decision_tree.pdf | task_method_decision_tree | `generate_task_method_decision_tree_bsc.py` | Matplotlib | 4,646 B |
| 26 | task_recommendations_table.pdf | task_recommendations_table | `generate_task_recommendations_table_bsc.py` | Matplotlib | 4,469 B |
| 27 | production_settings.pdf | production_settings | `generate_production_settings_bsc.py` | Matplotlib | 3,390 B |

---

## Type Distribution

- **Matplotlib**: 25 charts
- **Graphviz**: 2 charts

---

## Duplicate Scripts to Remove (Cleanup Phase)

### Bundled Generator Scripts (in multiple folders)

1. **`generate_week09_enhanced_charts.py`** (31,796 bytes)
   - Found in 8 folders:
     - extreme_greedy_single_path
     - extreme_full_beam_explosion
     - extreme_coverage_comparison
     - practical_methods_coverage
     - problem4_search_tree_pruning
     - problem4_path_comparison
     - problem4_probability_evolution
     - problem4_recovery_problem
     - quality_diversity_scatter
   - **Action**: DELETE from all folders (duplicates canonical scripts)

2. **`generate_week09_charts_bsc_enhanced.py`** (57,352 bytes)
   - Found in 7 folders:
     - prediction_to_text_pipeline
     - temperature_effects
     - topk_example
     - contrastive_vs_nucleus
     - degeneration_problem
     - quality_diversity_pareto
     - task_method_decision_tree
     - task_recommendations_table
     - production_settings
   - **Action**: DELETE from all folders (duplicates canonical scripts)

3. **`fix_week09_final_charts.py`** (21,164 bytes)
   - Found in 6 folders:
     - problem1_repetition_output
     - problem2_diversity_output
     - problem3_balance_output
     - problem5_distribution_output
     - problem6_speed_output
   - **Action**: DELETE from all folders (old fix script, not needed)

4. **`fix_charts_redesign.py`** (8,859 bytes)
   - Found in 1 folder:
     - vocabulary_probability
   - **Action**: DELETE (old fix script)

---

## Cleanup Summary

- **Total scripts found**: 54 Python files across 27 folders
- **Canonical scripts**: 27 (one per chart)
- **Duplicate scripts to delete**: 27 (54 - 27)
- **Space to reclaim**: ~1.5 MB

---

## Pattern Observations

### Naming Convention
- Canonical scripts follow pattern: `generate_<chart_name>_bsc.py`
- "bsc" indicates BSc Discovery color scheme
- Exception: `generate_beam_search_graphviz.py` (graphviz doesn't use BSc colors)

### Script Sizes
- **Graphviz scripts**: 6-7 KB (simple node/edge definitions)
- **Simple matplotlib**: 2-4 KB (single chart generation)
- **Complex matplotlib**: 7-8 KB (multi-panel or detailed layout)

---

## Next Steps

1. ✅ Identified all 27 canonical scripts
2. [ ] Document font sizes for remaining 24 scripts (3 already documented)
3. [ ] Standardize fonts in all scripts
4. [ ] Remove 27 duplicate scripts
5. [ ] Test regeneration of all charts
