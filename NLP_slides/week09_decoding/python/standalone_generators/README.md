# Week 9 Standalone Chart Generators

This folder contains **32 standalone Python scripts** - one script per chart used in `NLP_Decoding_Strategies.tex`.

## Overview

**32 figures** in presentation → **32 Python scripts** (one-to-one mapping)

Each script:
- Generates exactly ONE chart
- Outputs to current directory (`./`)
- Has all imports and BSc Discovery color scheme
- Can run independently

## Generated Charts

### Test Results: 28/32 Working

| Chart | Script | Status |
|-------|--------|--------|
| vocabulary_probability_bsc.pdf | generate_vocabulary_probability_bsc.py | ✓ PASS |
| prediction_to_text_pipeline_bsc.pdf | generate_prediction_to_text_pipeline_bsc.py | ✓ PASS |
| extreme_greedy_single_path_bsc.pdf | generate_extreme_greedy_single_path_bsc.py | ✓ PASS |
| greedy_suboptimal_comparison_bsc.pdf | generate_greedy_suboptimal_comparison.py | ✓ PASS |
| extreme_full_beam_explosion_bsc.pdf | generate_extreme_full_beam_explosion_bsc.py | ⚠ TIMEOUT |
| full_exploration_tree_graphviz.pdf | generate_full_exploration_graphviz.py | ✓ PASS |
| extreme_coverage_comparison_bsc.pdf | generate_extreme_coverage_comparison_bsc.py | ✓ PASS |
| practical_methods_coverage_bsc.pdf | generate_practical_methods_coverage_bsc.py | ✓ PASS |
| beam_search_tree_graphviz.pdf | generate_beam_search_graphviz.py | ✓ PASS |
| temperature_effects_bsc.pdf | generate_temperature_effects_bsc.py | ✓ PASS |
| temperature_calculation_bsc.pdf | generate_temperature_calculation_bsc.py | ✓ PASS |
| topk_filtering_bsc.pdf | generate_topk_filtering_bsc.py | ✓ PASS |
| topk_example_bsc.pdf | generate_topk_example_bsc.py | ✓ PASS |
| nucleus_process_bsc.pdf | generate_nucleus_process_bsc.py | ✓ PASS |
| nucleus_cumulative_bsc.pdf | generate_nucleus_cumulative_bsc.py | ✓ PASS |
| degeneration_problem_bsc.pdf | generate_degeneration_problem_bsc.py | ✓ PASS |
| contrastive_mechanism_bsc.pdf | generate_contrastive_mechanism_bsc.py | ✓ PASS |
| contrastive_vs_nucleus_bsc.pdf | generate_contrastive_vs_nucleus_bsc.py | ✓ PASS |
| problem1_repetition_output_bsc.pdf | generate_problem1_repetition_output_bsc.py | ✓ PASS |
| problem2_diversity_output_bsc.pdf | generate_problem2_diversity_output_bsc.py | ✓ PASS |
| problem3_balance_output_bsc.pdf | generate_problem3_balance_output_bsc.py | ✓ PASS |
| problem4_search_tree_pruning_bsc.pdf | ❌ MISSING | Need to create |
| problem4_path_comparison_bsc.pdf | ❌ MISSING | Need to create |
| problem4_probability_evolution_bsc.pdf | ❌ MISSING | Need to create |
| problem4_recovery_problem_bsc.pdf | ❌ MISSING | Need to create |
| problem5_distribution_output_bsc.pdf | generate_problem5_distribution_output_bsc.py | ✓ PASS |
| problem6_speed_output_bsc.pdf | generate_problem6_speed_output_bsc.py | ✓ PASS |
| quality_diversity_scatter_bsc.pdf | generate_quality_diversity_scatter_bsc.py | ✓ PASS |
| quality_diversity_pareto_bsc.pdf | generate_quality_diversity_pareto_bsc.py | ✓ PASS |
| task_method_decision_tree_bsc.pdf | generate_task_method_decision_tree_bsc.py | ✓ PASS |
| task_recommendations_table_bsc.pdf | generate_task_recommendations_table_bsc.py | ✓ PASS |
| production_settings_bsc.pdf | generate_production_settings_bsc.py | ✓ PASS |

## Usage

### Regenerate All Figures
```bash
cd NLP_slides/week09_decoding
python regenerate_all_tex_figures.py
```

### Regenerate Individual Figure
```bash
cd NLP_slides/week09_decoding/python/standalone_generators
python generate_vocabulary_probability_bsc.py
```

### Test All Scripts
```bash
cd NLP_slides/week09_decoding
python test_standalone_generators.py
```

## How Charts Were Created

Charts were extracted from these comprehensive scripts:
- `generate_week09_charts_bsc_enhanced.py` (18 functions)
- `generate_week09_enhanced_charts.py` (7 functions)
- `fix_week09_final_charts.py` (6 functions)
- `fix_charts_redesign.py` (1 function)
- Plus 3 already-standalone scripts

**Extraction tool**: `extract_individual_charts.py` (uses AST parsing)

## Dependencies

- matplotlib
- numpy
- seaborn
- graphviz (for graphviz charts)
- scipy (for some interpolation charts)
