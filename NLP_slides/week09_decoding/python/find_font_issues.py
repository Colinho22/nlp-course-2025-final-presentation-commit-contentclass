"""
Scan all canonical chart scripts for hardcoded fontsize values.
Find charts that still have non-standard font sizes.
"""

import re
from pathlib import Path

# Canonical scripts
SCRIPTS = [
    'vocabulary_probability/generate_vocabulary_probability_bsc.py',
    'prediction_to_text_pipeline/generate_prediction_to_text_pipeline_bsc.py',
    'extreme_greedy_single_path/generate_extreme_greedy_single_path_bsc.py',
    'greedy_suboptimal_comparison_bsc/generate_greedy_suboptimal_comparison.py',
    'extreme_full_beam_explosion/generate_extreme_full_beam_explosion_bsc.py',
    'extreme_coverage_comparison/generate_extreme_coverage_comparison_bsc.py',
    'practical_methods_coverage/generate_practical_methods_coverage_bsc.py',
    'temperature_effects/generate_temperature_effects_bsc.py',
    'topk_example/generate_topk_example_bsc.py',
    'contrastive_vs_nucleus/generate_contrastive_vs_nucleus_bsc.py',
    'problem1_repetition_output/generate_problem1_repetition_output_bsc.py',
    'problem2_diversity_output/generate_problem2_diversity_output_bsc.py',
    'problem3_balance_output/generate_problem3_balance_output_bsc.py',
    'problem4_search_tree_pruning/generate_problem4_search_tree_pruning_bsc.py',
    'problem4_path_comparison/generate_problem4_path_comparison_bsc.py',
    'problem4_probability_evolution/generate_problem4_probability_evolution_bsc.py',
    'problem4_recovery_problem/generate_problem4_recovery_problem_bsc.py',
    'problem5_distribution_output/generate_problem5_distribution_output_bsc.py',
    'problem6_speed_output/generate_problem6_speed_output_bsc.py',
    'degeneration_problem/generate_degeneration_problem_bsc.py',
    'quality_diversity_scatter/generate_quality_diversity_scatter_bsc.py',
    'quality_diversity_pareto/generate_quality_diversity_pareto_bsc.py',
    'task_method_decision_tree/generate_task_method_decision_tree_bsc.py',
    'task_recommendations_table/generate_task_recommendations_table_bsc.py',
    'production_settings/generate_production_settings_bsc.py',
]

GRAPHVIZ_SCRIPTS = [
    'full_exploration_tree_graphviz/generate_full_exploration_graphviz.py',
    'beam_search_tree_graphviz/generate_beam_search_graphviz.py',
]

def find_font_issues(script_path):
    """Find hardcoded fontsize values in a script."""

    content = script_path.read_text(encoding='utf-8')

    # Pattern for hardcoded fontsize values
    pattern = r'fontsize\s*=\s*[\'"]?(\d+)[\'"]?'

    matches = re.findall(pattern, content, re.IGNORECASE)

    if matches:
        # Convert to integers and find non-standard sizes
        sizes = [int(m) for m in matches]
        non_standard = [s for s in sizes if s not in [16, 18, 20, 22, 24]]

        return sizes, non_standard

    return [], []

def main():
    base_path = Path(__file__).parent / 'charts_individual'

    print("=" * 80)
    print("SCANNING FOR FONT SIZE ISSUES")
    print("=" * 80)
    print("\nBSc Discovery Standard: 16pt (ticks), 18pt (annotations), 20pt (labels), 24pt (titles)")
    print("\n" + "=" * 80)

    issues_found = []

    print("\nMATLOTLIB SCRIPTS:")
    print("-" * 80)

    for script_rel in SCRIPTS:
        script_path = base_path / script_rel
        if not script_path.exists():
            continue

        all_sizes, non_standard = find_font_issues(script_path)

        if non_standard:
            folder = script_path.parent.name
            print(f"\n[ISSUE] {folder}")
            print(f"        Script: {script_path.name}")
            print(f"        Non-standard sizes found: {sorted(set(non_standard))}")
            print(f"        All sizes in script: {sorted(set(all_sizes))}")
            issues_found.append((folder, script_path.name, non_standard))

    print("\n\nGRAPHVIZ SCRIPTS:")
    print("-" * 80)

    for script_rel in GRAPHVIZ_SCRIPTS:
        script_path = base_path / script_rel
        if not script_path.exists():
            continue

        all_sizes, non_standard = find_font_issues(script_path)

        if non_standard:
            folder = script_path.parent.name
            print(f"\n[ISSUE] {folder}")
            print(f"        Script: {script_path.name}")
            print(f"        Non-standard sizes found: {sorted(set(non_standard))}")
            print(f"        All sizes in script: {sorted(set(all_sizes))}")
            issues_found.append((folder, script_path.name, non_standard))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if issues_found:
        print(f"[FOUND] {len(issues_found)} chart(s) with font size issues:")
        for folder, script, sizes in issues_found:
            print(f"  - {folder}: sizes {sorted(set(sizes))}")
    else:
        print("[OK] No font size issues found! All charts use BSc Discovery standard.")

    print("=" * 80)

if __name__ == '__main__':
    main()
