"""
Regenerate all 27 charts with standardized BSc Discovery fonts.

Runs each canonical generator script to produce clean PDFs (no logos)
in the ../figures/ directory.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# All 27 canonical scripts
CANONICAL_SCRIPTS = [
    'vocabulary_probability/generate_vocabulary_probability_bsc.py',
    'prediction_to_text_pipeline/generate_prediction_to_text_pipeline_bsc.py',
    'extreme_greedy_single_path/generate_extreme_greedy_single_path_bsc.py',
    'greedy_suboptimal_comparison_bsc/generate_greedy_suboptimal_comparison.py',
    'extreme_full_beam_explosion/generate_extreme_full_beam_explosion_bsc.py',
    'full_exploration_tree_graphviz/generate_full_exploration_graphviz.py',
    'extreme_coverage_comparison/generate_extreme_coverage_comparison_bsc.py',
    'practical_methods_coverage/generate_practical_methods_coverage_bsc.py',
    'beam_search_tree_graphviz/generate_beam_search_graphviz.py',
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

def regenerate_all():
    base_path = Path(__file__).parent / 'charts_individual'
    start_time = datetime.now()

    print("=" * 80)
    print("REGENERATING ALL 27 CHARTS WITH STANDARDIZED FONTS")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%H:%M:%S')}")
    print(f"Output: ../figures/ (clean, no logos)")
    print("=" * 80)

    success_count = 0
    failed = []

    for i, script_rel_path in enumerate(CANONICAL_SCRIPTS, 1):
        script_path = base_path / script_rel_path

        if not script_path.exists():
            print(f"\n[{i}/27] [ERROR] NOT FOUND: {script_rel_path}")
            failed.append((script_rel_path, "Script not found"))
            continue

        print(f"\n[{i}/27] {script_path.parent.name}")
        print(f"        Running: {script_path.name}")

        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(script_path.parent),
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print(f"        [OK] Chart generated")
                success_count += 1
            else:
                print(f"        [ERROR] Exit code {result.returncode}")
                if result.stderr:
                    print(f"        Error: {result.stderr[:200]}")
                failed.append((script_rel_path, f"Exit code {result.returncode}"))

        except subprocess.TimeoutExpired:
            print(f"        [ERROR] Timeout (>30s)")
            failed.append((script_rel_path, "Timeout"))
        except Exception as e:
            print(f"        [ERROR] {e}")
            failed.append((script_rel_path, str(e)))

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"[OK] Successfully generated: {success_count}/27 charts")
    if failed:
        print(f"[WARN] Failed: {len(failed)} charts")
        for script, reason in failed:
            print(f"       - {script}: {reason}")
    print(f"\nDuration: {duration:.1f} seconds")
    print(f"End time: {end_time.strftime('%H:%M:%S')}")
    print("=" * 80)

    # Count PDFs in figures folder
    figures_path = base_path.parent / 'figures'
    if figures_path.exists():
        pdf_count = len(list(figures_path.glob('*.pdf')))
        print(f"\n[INFO] ../figures/ now contains {pdf_count} PDFs")

if __name__ == '__main__':
    regenerate_all()
