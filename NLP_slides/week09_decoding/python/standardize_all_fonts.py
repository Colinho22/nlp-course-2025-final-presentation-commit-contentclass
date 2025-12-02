"""
Standardize all Week 9 chart fonts to BSc Discovery standard (18-24pt).

This script updates all 27 canonical chart generation scripts to use
consistent font sizes across the presentation.

BSc Discovery Font Standard:
- FONTSIZE_TITLE: 24pt (chart titles)
- FONTSIZE_LABEL: 20pt (axis labels, legends)
- FONTSIZE_ANNOTATION: 18pt (text boxes, callouts)
- FONTSIZE_TICK: 16pt (axis ticks)
- FONTSIZE_TEXT: 20pt (general text)
- FONTSIZE_SMALL: 18pt (smaller annotations)
"""

import re
from pathlib import Path
import shutil

# BSc Discovery Font Standard
NEW_FONT_CONSTANTS = """
FONTSIZE_TITLE = 24
FONTSIZE_LABEL = 20
FONTSIZE_TICK = 16
FONTSIZE_ANNOTATION = 18
FONTSIZE_LEGEND = 18
FONTSIZE_TEXT = 20
FONTSIZE_SMALL = 18
""".strip()

# Canonical scripts to update (27 matplotlib charts)
MATPLOTLIB_SCRIPTS = [
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

# Graphviz scripts (handled separately)
GRAPHVIZ_SCRIPTS = [
    'full_exploration_tree_graphviz/generate_full_exploration_graphviz.py',
    'beam_search_tree_graphviz/generate_beam_search_graphviz.py',
]

# Common inline font size replacements (for hardcoded values)
INLINE_REPLACEMENTS = [
    (r'fontsize\s*=\s*42', 'fontsize=24'),  # Worst offender
    (r'fontsize\s*=\s*40', 'fontsize=24'),  # Temperature subplot titles
    (r'fontsize\s*=\s*36', 'fontsize=24'),  # Titles
    (r'fontsize\s*=\s*32', 'fontsize=22'),  # Large labels
    (r'fontsize\s*=\s*30', 'fontsize=20'),  # Labels
    (r'fontsize\s*=\s*28', 'fontsize=18'),  # Annotations
    (r'fontsize\s*=\s*26', 'fontsize=18'),  # Small annotations
    (r"fontsize\s*=\s*'42'", "fontsize='24'"),
    (r"fontsize\s*=\s*'40'", "fontsize='24'"),
    (r"fontsize\s*=\s*'36'", "fontsize='24'"),
    (r"fontsize\s*=\s*'32'", "fontsize='22'"),
    (r"fontsize\s*=\s*'30'", "fontsize='20'"),
    (r"fontsize\s*=\s*'28'", "fontsize='18'"),
]

# Graphviz font replacements
GRAPHVIZ_REPLACEMENTS = [
    (r"fontsize\s*=\s*'42'", "fontsize='24'"),
    (r"fontsize\s*=\s*'40'", "fontsize='24'"),
    (r"fontsize\s*=\s*'36'", "fontsize='24'"),
    (r"fontsize\s*=\s*'32'", "fontsize='22'"),
    (r"fontsize\s*=\s*'30'", "fontsize='20'"),
    (r"fontsize\s*=\s*'28'", "fontsize='18'"),
]


def update_matplotlib_script(script_path):
    """Update a matplotlib script with new font constants and inline replacements."""

    print(f"\n  Processing: {script_path.name}")

    # Read the script
    content = script_path.read_text(encoding='utf-8')

    # Create backup
    backup_path = script_path.with_suffix('.py.backup')
    shutil.copy2(script_path, backup_path)
    print(f"    Backup: {backup_path.name}")

    # Replace font size constants block
    # Pattern: Match the entire FONTSIZE_XXX = YY block
    pattern = r'FONTSIZE_TITLE\s*=\s*\d+\s*\nFONTSIZE_LABEL\s*=\s*\d+\s*\nFONTSIZE_TICK\s*=\s*\d+\s*\nFONTSIZE_ANNOTATION\s*=\s*\d+.*?(?=\n\ndef|\n\nif|\Z)'

    match = re.search(pattern, content, re.DOTALL)

    if match:
        old_block = match.group(0)
        content = content.replace(old_block, NEW_FONT_CONSTANTS + '\n')
        print(f"    [OK] Updated font constants block")
    else:
        print(f"    [WARN] Font constants block not found (might need manual update)")

    # Replace inline font size overrides
    changes = 0
    for old_pattern, new_value in INLINE_REPLACEMENTS:
        matches = len(re.findall(old_pattern, content))
        if matches > 0:
            content = re.sub(old_pattern, new_value, content)
            changes += matches

    if changes > 0:
        print(f"    [OK] Replaced {changes} inline font size(s)")

    # Write updated content
    script_path.write_text(content, encoding='utf-8')
    print(f"    [OK] Saved updated script")

    return True


def update_graphviz_script(script_path):
    """Update a graphviz script with new font sizes."""

    print(f"\n  Processing: {script_path.name}")

    # Read the script
    content = script_path.read_text(encoding='utf-8')

    # Create backup
    backup_path = script_path.with_suffix('.py.backup')
    shutil.copy2(script_path, backup_path)
    print(f"    Backup: {backup_path.name}")

    # Replace inline graphviz font sizes
    changes = 0
    for old_pattern, new_value in GRAPHVIZ_REPLACEMENTS:
        matches = len(re.findall(old_pattern, content))
        if matches > 0:
            content = re.sub(old_pattern, new_value, content)
            changes += matches

    if changes > 0:
        print(f"    [OK] Replaced {changes} graphviz font size(s)")
    else:
        print(f"    [WARN] No graphviz font sizes found to replace")

    # Write updated content
    script_path.write_text(content, encoding='utf-8')
    print(f"    [OK] Saved updated script")

    return True


def main():
    base_path = Path(__file__).parent / 'charts_individual'

    print("=" * 80)
    print("WEEK 9 FONT STANDARDIZATION")
    print("=" * 80)
    print(f"\nBSc Discovery Font Standard:")
    print(f"  Title: 24pt | Labels: 20pt | Annotations: 18pt | Ticks: 16pt")
    print("=" * 80)

    # Update Matplotlib scripts
    print("\n[1/2] UPDATING MATPLOTLIB SCRIPTS (25 charts)")
    print("-" * 80)

    success_count = 0
    for script_rel_path in MATPLOTLIB_SCRIPTS:
        script_path = base_path / script_rel_path

        if not script_path.exists():
            print(f"\n  [ERROR] NOT FOUND: {script_path}")
            continue

        try:
            if update_matplotlib_script(script_path):
                success_count += 1
        except Exception as e:
            print(f"    [ERROR] {e}")

    print(f"\n[OK] Updated {success_count}/{len(MATPLOTLIB_SCRIPTS)} matplotlib scripts")

    # Update Graphviz scripts
    print("\n[2/2] UPDATING GRAPHVIZ SCRIPTS (2 charts)")
    print("-" * 80)

    graphviz_success = 0
    for script_rel_path in GRAPHVIZ_SCRIPTS:
        script_path = base_path / script_rel_path

        if not script_path.exists():
            print(f"\n  [ERROR] NOT FOUND: {script_path}")
            continue

        try:
            if update_graphviz_script(script_path):
                graphviz_success += 1
        except Exception as e:
            print(f"    [ERROR] {e}")

    print(f"\n[OK] Updated {graphviz_success}/{len(GRAPHVIZ_SCRIPTS)} graphviz scripts")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"[OK] Matplotlib scripts: {success_count}/{len(MATPLOTLIB_SCRIPTS)}")
    print(f"[OK] Graphviz scripts: {graphviz_success}/{len(GRAPHVIZ_SCRIPTS)}")
    print(f"[OK] Total updated: {success_count + graphviz_success}/27")
    print("\nBackup files created with .py.backup extension")
    print("=" * 80)


if __name__ == '__main__':
    main()
