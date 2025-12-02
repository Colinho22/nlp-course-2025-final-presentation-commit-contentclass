"""
Regenerate the 11 charts that were just fixed (10 + temperature).
"""

import subprocess
import sys
from pathlib import Path

FIXED_CHARTS = [
    'temperature_effects/generate_temperature_effects_bsc.py',
    'prediction_to_text_pipeline/generate_prediction_to_text_pipeline_bsc.py',
    'greedy_suboptimal_comparison_bsc/generate_greedy_suboptimal_comparison.py',
    'extreme_full_beam_explosion/generate_extreme_full_beam_explosion_bsc.py',
    'extreme_coverage_comparison/generate_extreme_coverage_comparison_bsc.py',
    'practical_methods_coverage/generate_practical_methods_coverage_bsc.py',
    'topk_example/generate_topk_example_bsc.py',
    'contrastive_vs_nucleus/generate_contrastive_vs_nucleus_bsc.py',
    'problem6_speed_output/generate_problem6_speed_output_bsc.py',
    'degeneration_problem/generate_degeneration_problem_bsc.py',
    'quality_diversity_scatter/generate_quality_diversity_scatter_bsc.py',
]

def main():
    base_path = Path(__file__).parent / 'charts_individual'

    print("=" * 80)
    print("REGENERATING 11 FIXED CHARTS")
    print("=" * 80)

    success = 0
    failed = []

    for i, script_rel in enumerate(FIXED_CHARTS, 1):
        script_path = base_path / script_rel

        if not script_path.exists():
            print(f"\n[{i}/11] [ERROR] Not found: {script_rel}")
            failed.append(script_rel)
            continue

        folder = script_path.parent.name
        print(f"\n[{i}/11] {folder}")
        print(f"        Running: {script_path.name}")

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(script_path.parent),
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print(f"        [OK] Generated")
                success += 1
            else:
                print(f"        [ERROR] Exit code {result.returncode}")
                failed.append(script_rel)

        except Exception as e:
            print(f"        [ERROR] {e}")
            failed.append(script_rel)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"[OK] Successfully generated: {success}/11 charts")
    if failed:
        print(f"[WARN] Failed: {len(failed)}")
        for script in failed:
            print(f"       - {script}")
    print("=" * 80)

if __name__ == '__main__':
    main()
