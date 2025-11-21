"""
Copy the ORIGINAL working scripts to individual folders
========================================================
Ensures each folder has the EXACT script that created the original figure.
"""

from pathlib import Path
import shutil

# Definitive mapping: figure -> source script that created it
FIGURE_TO_SCRIPT = {
    # From generate_beam_search_graphviz.py (standalone)
    'beam_search_tree_graphviz.pdf': 'generate_beam_search_graphviz.py',

    # From generate_full_exploration_graphviz.py (standalone)
    'full_exploration_tree_graphviz.pdf': 'generate_full_exploration_graphviz.py',

    # From generate_greedy_suboptimal_comparison.py (standalone)
    'greedy_suboptimal_comparison_bsc.pdf': 'generate_greedy_suboptimal_comparison.py',

    # From generate_week09_enhanced_charts.py
    'quality_diversity_scatter_bsc.pdf': 'generate_week09_enhanced_charts.py',
    'extreme_greedy_single_path_bsc.pdf': 'generate_week09_enhanced_charts.py',
    'extreme_full_beam_explosion_bsc.pdf': 'generate_week09_enhanced_charts.py',
    'extreme_computational_cost_bsc.pdf': 'generate_week09_enhanced_charts.py',
    'extreme_coverage_comparison_bsc.pdf': 'generate_week09_enhanced_charts.py',
    'practical_methods_coverage_bsc.pdf': 'generate_week09_enhanced_charts.py',
    'problem4_search_tree_pruning_bsc.pdf': 'generate_week09_enhanced_charts.py',
    'problem4_path_comparison_bsc.pdf': 'generate_week09_enhanced_charts.py',
    'problem4_probability_evolution_bsc.pdf': 'generate_week09_enhanced_charts.py',
    'problem4_recovery_problem_bsc.pdf': 'generate_week09_enhanced_charts.py',

    # From generate_week09_charts_bsc_enhanced.py (comprehensive script with 23 functions)
    'vocabulary_probability_bsc.pdf': 'fix_charts_redesign.py',  # Actually from fix script
    'prediction_to_text_pipeline_bsc.pdf': 'generate_week09_charts_bsc_enhanced.py',
    'temperature_effects_bsc.pdf': 'generate_week09_charts_bsc_enhanced.py',
    'temperature_calculation_bsc.pdf': 'generate_week09_charts_bsc_enhanced.py',
    'topk_filtering_bsc.pdf': 'generate_week09_charts_bsc_enhanced.py',
    'topk_example_bsc.pdf': 'generate_week09_charts_bsc_enhanced.py',
    'nucleus_process_bsc.pdf': 'generate_week09_charts_bsc_enhanced.py',
    'nucleus_cumulative_bsc.pdf': 'generate_week09_charts_bsc_enhanced.py',
    'degeneration_problem_bsc.pdf': 'generate_week09_charts_bsc_enhanced.py',
    'contrastive_mechanism_bsc.pdf': 'generate_week09_charts_bsc_enhanced.py',
    'contrastive_vs_nucleus_bsc.pdf': 'generate_week09_charts_bsc_enhanced.py',
    'quality_diversity_pareto_bsc.pdf': 'generate_week09_charts_bsc_enhanced.py',
    'task_method_decision_tree_bsc.pdf': 'generate_week09_charts_bsc_enhanced.py',
    'task_recommendations_table_bsc.pdf': 'generate_week09_charts_bsc_enhanced.py',
    'production_settings_bsc.pdf': 'generate_week09_charts_bsc_enhanced.py',

    # From fix_week09_final_charts.py
    'problem1_repetition_output_bsc.pdf': 'fix_week09_final_charts.py',
    'problem2_diversity_output_bsc.pdf': 'fix_week09_final_charts.py',
    'problem3_balance_output_bsc.pdf': 'fix_week09_final_charts.py',
    'problem5_distribution_output_bsc.pdf': 'fix_week09_final_charts.py',
    'problem6_speed_output_bsc.pdf': 'fix_week09_final_charts.py',
}

charts_dir = Path("python/charts_individual")
python_dir = Path("python")

print("=" * 70)
print("Copying Original Working Scripts to Individual Folders")
print("=" * 70)

for fig, script in FIGURE_TO_SCRIPT.items():
    folder_name = fig.replace('.pdf', '')
    folder = charts_dir / folder_name

    if not folder.exists():
        print(f"  SKIP {folder_name} - folder doesn't exist")
        continue

    # Copy the original script
    source_script = python_dir / script
    if source_script.exists():
        dest_script = folder / script
        shutil.copy2(source_script, dest_script)
        print(f"  OK {folder_name}/ <- {script}")
    else:
        print(f"  MISSING {folder_name} - script {script} not found")

print("\n" + "=" * 70)
print("Note: Multi-function scripts need manual extraction")
print("  - generate_week09_enhanced_charts.py has 7 functions")
print("  - generate_week09_charts_bsc_enhanced.py has 23 functions")
print("  - fix_week09_final_charts.py has 6 functions")
print("=" * 70)
