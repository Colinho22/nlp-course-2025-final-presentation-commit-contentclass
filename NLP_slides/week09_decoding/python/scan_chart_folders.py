"""
Scan all chart folders to identify canonical generation scripts.
"""
import os
from pathlib import Path

# List of all 27 charts from the .tex file
charts_needed = [
    'vocabulary_probability',
    'prediction_to_text_pipeline',
    'extreme_greedy_single_path',
    'greedy_suboptimal_comparison_bsc',
    'extreme_full_beam_explosion',
    'full_exploration_tree_graphviz',
    'extreme_coverage_comparison',
    'practical_methods_coverage',
    'beam_search_tree_graphviz',
    'temperature_effects',
    'topk_example',
    'contrastive_vs_nucleus',
    'problem1_repetition_output',
    'problem2_diversity_output',
    'problem3_balance_output',
    'problem4_search_tree_pruning',
    'problem4_path_comparison',
    'problem4_probability_evolution',
    'problem4_recovery_problem',
    'problem5_distribution_output',
    'problem6_speed_output',
    'degeneration_problem',
    'quality_diversity_scatter',
    'quality_diversity_pareto',
    'task_method_decision_tree',
    'task_recommendations_table',
    'production_settings'
]

base_path = Path(r'D:\Joerg\Research\slides\2025_NLP_16\NLP_slides\week09_decoding\python\charts_individual')

print("=" * 80)
print("WEEK 9 CHART FOLDER SCAN")
print("=" * 80)

for chart_name in charts_needed:
    folder_path = base_path / chart_name

    print(f"\n{chart_name}:")
    print("-" * 60)

    if not folder_path.exists():
        print(f"  [X] FOLDER NOT FOUND: {folder_path}")
        continue

    # Find all Python scripts
    py_files = list(folder_path.glob('*.py'))

    if not py_files:
        print(f"  [!] NO PYTHON SCRIPTS FOUND")
    else:
        print(f"  [OK] Found {len(py_files)} script(s):")
        for py_file in py_files:
            file_size = py_file.stat().st_size
            print(f"    - {py_file.name} ({file_size:,} bytes)")

            # Try to identify the canonical script
            if len(py_files) == 1:
                print(f"      >> CANONICAL (only script)")
            elif 'generate_' in py_file.name.lower() and chart_name in py_file.name.lower():
                print(f"      >> LIKELY CANONICAL (name matches)")
            elif 'bsc' in py_file.name.lower():
                print(f"      >> LIKELY CANONICAL (BSc Discovery version)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total charts needed: {len(charts_needed)}")
print(f"Folders scanned: {len([c for c in charts_needed if (base_path / c).exists()])}")
