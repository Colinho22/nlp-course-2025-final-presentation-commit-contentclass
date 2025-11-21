"""
Extract ALL 32 Charts as Standalone Scripts
============================================
Creates standalone_generators/ folder with one .py file per chart.

Each standalone script:
- Generates exactly ONE PDF
- Has all imports and color scheme
- Outputs to current directory
- Can be run independently
"""

import re
import ast
from pathlib import Path
import shutil

# Complete mapping: figure_name -> (source_script, function_name)
CHART_SOURCES = {
    # From generate_week09_charts_bsc_enhanced.py (18 charts)
    'quality_diversity_tradeoff_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_quality_diversity_tradeoff'),
    'temperature_effects_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_temperature_effects'),
    'temperature_calculation_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_temperature_calculation'),
    'topk_filtering_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_topk_filtering'),
    'topk_example_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_topk_example'),
    'nucleus_process_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_nucleus_process'),
    'nucleus_cumulative_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_nucleus_cumulative'),
    'degeneration_problem_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_degeneration_problem'),
    'contrastive_mechanism_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_contrastive_mechanism'),
    'contrastive_vs_nucleus_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_contrastive_vs_nucleus'),
    'quality_diversity_pareto_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_quality_diversity_pareto'),
    'task_method_decision_tree_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_task_method_decision_tree'),
    'task_recommendations_table_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_task_recommendations_table'),
    'computational_cost_comparison_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_computational_cost_comparison'),
    'production_settings_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_production_settings'),
    'prediction_to_text_pipeline_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_prediction_pipeline'),
    'vocabulary_probability_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_vocab_probability'),  # May need to verify
    'beam_example_tree_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_beam_example_tree'),

    # From generate_week09_enhanced_charts.py (10 charts)
    'quality_diversity_scatter_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_quality_diversity_scatter'),
    'problem4_search_tree_pruning_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_problem4_graphviz_charts'),
    'problem4_path_comparison_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_problem4_graphviz_charts'),
    'problem4_probability_evolution_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_problem4_graphviz_charts'),
    'problem4_recovery_problem_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_problem4_graphviz_charts'),
    'extreme_greedy_single_path_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_extreme_case_1_greedy'),
    'extreme_full_beam_explosion_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_extreme_case_2_full_beam'),
    'extreme_computational_cost_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_extreme_case_3_computational_cost'),
    'extreme_coverage_comparison_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_extreme_coverage_comparison'),
    'practical_methods_coverage_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_practical_methods_coverage'),

    # From fix_week09_final_charts.py (6 problem outputs)
    'problem1_repetition_output_bsc.pdf': ('fix_week09_final_charts.py', 'generate_problem1_repetition'),
    'problem2_diversity_output_bsc.pdf': ('fix_week09_final_charts.py', 'generate_problem2_diversity'),
    'problem3_balance_output_bsc.pdf': ('fix_week09_final_charts.py', 'generate_problem3_balance'),
    'problem5_distribution_output_bsc.pdf': ('fix_week09_final_charts.py', 'generate_problem5_distribution'),
    'problem6_speed_output_bsc.pdf': ('fix_week09_final_charts.py', 'generate_problem6_speed'),

    # Standalone scripts (just copy them)
    'beam_search_tree_graphviz.pdf': ('generate_beam_search_graphviz.py', None),
    'full_exploration_tree_graphviz.pdf': ('generate_full_exploration_graphviz.py', None),
    'greedy_suboptimal_comparison_bsc.pdf': ('generate_greedy_suboptimal_comparison.py', None),
}

print(f"Total charts mapped: {len(CHART_SOURCES)}")
print("Creating standalone_generators/ folder...")

output_dir = Path("python/standalone_generators")
output_dir.mkdir(parents=True, exist_ok=True)

# Now I'll create the actual extraction in the next step
print(f"\nReady to extract {len(CHART_SOURCES)} standalone scripts")
print(f"Output: {output_dir.absolute()}")
