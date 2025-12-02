"""
Collect all generated chart PDFs from individual folders to central ../figures/ directory.

Each chart generation script saves to its local folder. This script copies all PDFs
to the central figures/ directory for use in the presentation.
"""

from pathlib import Path
import shutil

# Expected PDF outputs from each folder
EXPECTED_PDFS = {
    'vocabulary_probability': 'vocabulary_probability_bsc.pdf',
    'prediction_to_text_pipeline': 'prediction_to_text_pipeline_bsc.pdf',
    'extreme_greedy_single_path': 'extreme_greedy_single_path_bsc.pdf',
    'greedy_suboptimal_comparison_bsc': 'greedy_suboptimal_comparison_bsc.pdf',
    'extreme_full_beam_explosion': 'extreme_full_beam_explosion_bsc.pdf',
    'full_exploration_tree_graphviz': 'full_exploration_tree_graphviz.pdf',
    'extreme_coverage_comparison': 'extreme_coverage_comparison_bsc.pdf',
    'practical_methods_coverage': 'practical_methods_coverage_bsc.pdf',
    'beam_search_tree_graphviz': 'beam_search_tree_graphviz.pdf',
    'temperature_effects': 'temperature_effects_bsc.pdf',
    'topk_example': 'topk_example_bsc.pdf',
    'contrastive_vs_nucleus': 'contrastive_vs_nucleus_bsc.pdf',
    'problem1_repetition_output': 'problem1_repetition_output_bsc.pdf',
    'problem2_diversity_output': 'problem2_diversity_output_bsc.pdf',
    'problem3_balance_output': 'problem3_balance_output_bsc.pdf',
    'problem4_search_tree_pruning': 'problem4_search_tree_pruning_bsc.pdf',
    'problem4_path_comparison': 'problem4_path_comparison_bsc.pdf',
    'problem4_probability_evolution': 'problem4_probability_evolution_bsc.pdf',
    'problem4_recovery_problem': 'problem4_recovery_problem_bsc.pdf',
    'problem5_distribution_output': 'problem5_distribution_output_bsc.pdf',
    'problem6_speed_output': 'problem6_speed_output_bsc.pdf',
    'degeneration_problem': 'degeneration_problem_bsc.pdf',
    'quality_diversity_scatter': 'quality_diversity_scatter_bsc.pdf',
    'quality_diversity_pareto': 'quality_diversity_pareto_bsc.pdf',
    'task_method_decision_tree': 'task_method_decision_tree_bsc.pdf',
    'task_recommendations_table': 'task_recommendations_table_bsc.pdf',
    'production_settings': 'production_settings_bsc.pdf',
}

def collect_all_charts():
    base_path = Path(__file__).parent / 'charts_individual'
    dest_path = base_path.parent / 'figures'

    # Ensure destination exists
    dest_path.mkdir(exist_ok=True)

    print("=" * 80)
    print("COLLECTING ALL CHART PDFs TO CENTRAL FIGURES/ DIRECTORY")
    print("=" * 80)
    print(f"Destination: {dest_path}")
    print("=" * 80)

    collected = 0
    missing = []

    for folder_name, pdf_name in EXPECTED_PDFS.items():
        source_path = base_path / folder_name / pdf_name
        dest_file = dest_path / pdf_name

        print(f"\n[{collected + 1 + len(missing)}/27] {pdf_name}")

        if not source_path.exists():
            print(f"        [MISSING] {source_path}")
            missing.append((folder_name, pdf_name))
            continue

        # Copy to destination
        shutil.copy2(source_path, dest_file)
        file_size = dest_file.stat().st_size
        print(f"        [OK] Copied ({file_size:,} bytes)")
        collected += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"[OK] Collected: {collected}/27 PDFs")

    if missing:
        print(f"[WARN] Missing: {len(missing)} PDFs")
        for folder, pdf in missing:
            print(f"       - {folder}/{pdf}")

    # Count final PDFs
    final_count = len(list(dest_path.glob('*.pdf')))
    print(f"\n[INFO] ../figures/ now contains {final_count} PDFs total")
    print("=" * 80)

if __name__ == '__main__':
    collect_all_charts()
