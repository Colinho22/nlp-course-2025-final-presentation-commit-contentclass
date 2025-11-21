"""
Extract Standalone Chart Scripts from Multi-Function Scripts
============================================================
Creates ONE Python script per chart by extracting individual functions.

For each chart in the presentation:
1. Find the function that generates it
2. Extract function + dependencies
3. Create standalone script with proper imports and color scheme
4. Output to standalone_generators/
"""

import re
import ast
from pathlib import Path

# Source scripts to extract from
SOURCE_SCRIPTS = {
    'generate_week09_charts_bsc_enhanced.py': [
        ('generate_quality_diversity_tradeoff', 'quality_diversity_tradeoff_bsc.pdf'),
        ('generate_beam_search_visual', 'beam_search_visual_bsc.pdf'),
        ('generate_beam_example_tree', 'beam_example_tree_bsc.pdf'),
        ('generate_temperature_effects', 'temperature_effects_bsc.pdf'),
        ('generate_temperature_calculation', 'temperature_calculation_bsc.pdf'),
        ('generate_topk_filtering', 'topk_filtering_bsc.pdf'),
        ('generate_topk_example', 'topk_example_bsc.pdf'),
        ('generate_nucleus_process', 'nucleus_process_bsc.pdf'),
        ('generate_nucleus_cumulative', 'nucleus_cumulative_bsc.pdf'),
        ('generate_degeneration_problem', 'degeneration_problem_bsc.pdf'),
        ('generate_contrastive_mechanism', 'contrastive_mechanism_bsc.pdf'),
        ('generate_contrastive_vs_nucleus', 'contrastive_vs_nucleus_bsc.pdf'),
        ('generate_quality_diversity_pareto', 'quality_diversity_pareto_bsc.pdf'),
        ('generate_task_method_decision_tree', 'task_method_decision_tree_bsc.pdf'),
        ('generate_task_recommendations_table', 'task_recommendations_table_bsc.pdf'),
        ('generate_computational_cost_comparison', 'computational_cost_comparison_bsc.pdf'),
        ('generate_production_settings', 'production_settings_bsc.pdf'),
        ('generate_prediction_pipeline', 'prediction_to_text_pipeline_bsc.pdf'),
    ],
    'generate_week09_enhanced_charts.py': [
        ('generate_quality_diversity_scatter', 'quality_diversity_scatter_bsc.pdf'),
        ('generate_extreme_case_1_greedy', 'extreme_greedy_single_path_bsc.pdf'),
        ('generate_extreme_case_2_full_beam', 'extreme_full_beam_explosion_bsc.pdf'),
        ('generate_extreme_case_3_computational_cost', 'extreme_computational_cost_bsc.pdf'),
        ('generate_extreme_coverage_comparison', 'extreme_coverage_comparison_bsc.pdf'),
        ('generate_practical_methods_coverage', 'practical_methods_coverage_bsc.pdf'),
    ],
}

# Additional single-chart scripts (already standalone)
ALREADY_STANDALONE = [
    ('generate_beam_search_graphviz.py', 'beam_search_tree_graphviz.pdf'),
    ('generate_full_exploration_graphviz.py', 'full_exploration_tree_graphviz.pdf'),
    ('generate_greedy_suboptimal_comparison.py', 'greedy_suboptimal_comparison_bsc.pdf'),
]

# Common header template
HEADER_TEMPLATE = '''"""
Generate {chart_name}

Part of Week 9: NLP Decoding Strategies
Output: {output_file}
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

# BSc Discovery Color Scheme
COLOR_MAIN = '#333333'
COLOR_ACCENT = '#3333B2'
COLOR_LAVENDER = '#ADADC0'
COLOR_LAVENDER2 = '#C1C1E8'
COLOR_BLUE = '#0066CC'
COLOR_GRAY = '#7F7F7F'
COLOR_LIGHT = '#F0F0F0'
COLOR_RED = '#D62728'
COLOR_GREEN = '#2CA02C'
COLOR_ORANGE = '#FF7F0E'

plt.style.use('seaborn-v0_8-whitegrid')

# Font sizes
FONTSIZE_TITLE = 36
FONTSIZE_LABEL = 30
FONTSIZE_TICK = 28
FONTSIZE_ANNOTATION = 28
FONTSIZE_LEGEND = 26
FONTSIZE_TEXT = 30

def set_minimalist_style(ax):
    """Apply minimalist styling"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_ACCENT)
    ax.spines['bottom'].set_color(COLOR_ACCENT)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(colors=COLOR_MAIN, which='both', labelsize=FONTSIZE_TICK, width=2, length=6)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=3, color=COLOR_LAVENDER)
    ax.set_facecolor('white')

'''

def extract_function(source_code, function_name):
    """Extract a function and its code from source."""
    # Find function definition
    pattern = rf'(def {function_name}\([^)]*\):.*?)(?=\n(?:def |if __name__|$))'
    match = re.search(pattern, source_code, re.DOTALL)

    if match:
        return match.group(1).strip()
    return None


def create_standalone_script(source_file, function_name, output_file, output_dir):
    """Create standalone script for one chart."""
    # Read source
    source_path = Path(f"python/{source_file}")
    with open(source_path, 'r', encoding='utf-8') as f:
        source_code = f.read()

    # Extract function
    function_code = extract_function(source_code, function_name)
    if not function_code:
        print(f"  ⚠️  Could not find function {function_name} in {source_file}")
        return False

    # Create chart name
    chart_name = output_file.replace('.pdf', '').replace('_bsc', '').replace('_', ' ').title()

    # Build standalone script
    header = HEADER_TEMPLATE.format(chart_name=chart_name, output_file=output_file)

    # Fix output path in function (change ../figures/ to ./)
    function_code_fixed = function_code.replace("'../figures/", "'./")
    function_code_fixed = function_code_fixed.replace('"../figures/', '"./')

    standalone = header + '\n' + function_code_fixed + '\n\nif __name__ == "__main__":\n    ' + function_name + '()\n'

    # Write to file
    script_name = f"generate_{output_file.replace('.pdf', '')}.py"
    output_path = output_dir / script_name
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(standalone)

    print(f"  ✓ {script_name}")
    return True


def main():
    print("=" * 70)
    print("Extract Standalone Chart Scripts")
    print("Week 9: Decoding Strategies")
    print("=" * 70)

    output_dir = Path("python/standalone_generators")
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    success = 0

    # Extract from multi-function scripts
    for source_file, functions in SOURCE_SCRIPTS.items():
        print(f"\n{source_file}:")
        for func_name, output_file in functions:
            if create_standalone_script(source_file, func_name, output_file, output_dir):
                success += 1
            total += 1

    # Copy already-standalone scripts
    print(f"\nAlready standalone scripts:")
    for source_file, output_file in ALREADY_STANDALONE:
        import shutil
        src = Path(f"python/{source_file}")
        dest = output_dir / source_file
        shutil.copy2(src, dest)
        print(f"  ✓ {source_file}")
        success += 1
        total += 1

    print("\n" + "=" * 70)
    print(f"Created {success}/{total} standalone chart scripts")
    print(f"Output: {output_dir.absolute()}")
    print("=" * 70)
    print("\nEach script:")
    print("  - Generates exactly ONE chart")
    print("  - Outputs to current directory (./)")
    print("  - Has all imports and color scheme")
    print("  - Can be run independently")


if __name__ == '__main__':
    main()
