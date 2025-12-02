"""
Fix all non-standard font sizes in chart scripts.

Converts:
- 38pt → 24pt (likely titles/large text)
- 14pt → 18pt (likely annotations)
- 12pt → 18pt (annotations)
- 13pt → 18pt (annotations)
- 11pt → 18pt (annotations)
- 10pt → 16pt (ticks)
- 9pt → 16pt (ticks)
- 8pt → 16pt (ticks)
"""

import re
from pathlib import Path
import shutil

# Charts with known font issues
CHARTS_TO_FIX = {
    'prediction_to_text_pipeline/generate_prediction_to_text_pipeline_bsc.py': {
        38: 24,
    },
    'greedy_suboptimal_comparison_bsc/generate_greedy_suboptimal_comparison.py': {
        14: 18,
    },
    'extreme_full_beam_explosion/generate_extreme_full_beam_explosion_bsc.py': {
        11: 18,
        10: 16,
        9: 16,
    },
    'extreme_coverage_comparison/generate_extreme_coverage_comparison_bsc.py': {
        14: 18,
        12: 18,
        10: 16,
        9: 16,
    },
    'practical_methods_coverage/generate_practical_methods_coverage_bsc.py': {
        14: 18,
        11: 18,
        9: 16,
        8: 16,
    },
    'topk_example/generate_topk_example_bsc.py': {
        38: 24,
    },
    'contrastive_vs_nucleus/generate_contrastive_vs_nucleus_bsc.py': {
        38: 24,
    },
    'problem6_speed_output/generate_problem6_speed_output_bsc.py': {
        12: 18,
    },
    'degeneration_problem/generate_degeneration_problem_bsc.py': {
        38: 24,
    },
    'quality_diversity_scatter/generate_quality_diversity_scatter_bsc.py': {
        13: 18,
        11: 18,
        10: 16,
        9: 16,
    },
}

def fix_font_sizes(script_path, size_map):
    """Replace font sizes in script according to size_map."""

    print(f"\n  Processing: {script_path.name}")

    # Read content
    content = script_path.read_text(encoding='utf-8')

    # Create backup
    backup_path = script_path.with_suffix('.py.backup2')
    shutil.copy2(script_path, backup_path)

    # Replace each size
    changes = 0
    for old_size, new_size in sorted(size_map.items(), reverse=True):
        # Pattern for fontsize=XX or fontsize='XX' or fontsize="XX"
        patterns = [
            (rf'fontsize\s*=\s*{old_size}(?!\d)', f'fontsize={new_size}'),
            (rf"fontsize\s*=\s*'{old_size}'", f"fontsize='{new_size}'"),
            (rf'fontsize\s*=\s*"{old_size}"', f'fontsize="{new_size}"'),
        ]

        for pattern, replacement in patterns:
            matches = len(re.findall(pattern, content))
            if matches > 0:
                content = re.sub(pattern, replacement, content)
                changes += matches
                print(f"    {old_size}pt -> {new_size}pt: {matches} replacement(s)")

    if changes > 0:
        # Write updated content
        script_path.write_text(content, encoding='utf-8')
        print(f"    [OK] Total: {changes} font size(s) fixed")
        return True
    else:
        print(f"    [SKIP] No changes needed")
        return False

def main():
    base_path = Path(__file__).parent / 'charts_individual'

    print("=" * 80)
    print("FIXING ALL FONT SIZE ISSUES")
    print("=" * 80)

    fixed_count = 0
    total_charts = len(CHARTS_TO_FIX)

    for script_rel, size_map in CHARTS_TO_FIX.items():
        script_path = base_path / script_rel

        if not script_path.exists():
            print(f"\n  [ERROR] Not found: {script_rel}")
            continue

        folder = script_path.parent.name
        print(f"\n[{fixed_count + 1}/{total_charts}] {folder}")

        if fix_font_sizes(script_path, size_map):
            fixed_count += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"[OK] Fixed {fixed_count}/{total_charts} charts")
    print("Backup files created with .py.backup2 extension")
    print("=" * 80)

if __name__ == '__main__':
    main()
