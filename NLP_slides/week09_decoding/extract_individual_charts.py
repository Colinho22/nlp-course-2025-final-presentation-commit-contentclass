"""
Extract Individual Chart Generation Scripts
============================================
Creates 32+ standalone Python scripts in standalone_generators/

Each script generates exactly ONE chart.
"""

import re
import ast
from pathlib import Path
import shutil

output_dir = Path("python/standalone_generators")
output_dir.mkdir(parents=True, exist_ok=True)

# Common header for matplotlib charts
MATPLOTLIB_HEADER = '''"""
{description}

Generated chart: {output_file}
Part of Week 9: NLP Decoding Strategies
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np
import seaborn as sns
try:
    from scipy import interpolate
except ImportError:
    pass  # Not all scripts need scipy

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

FONTSIZE_TITLE = 36
FONTSIZE_LABEL = 30
FONTSIZE_TICK = 28
FONTSIZE_ANNOTATION = 28
FONTSIZE_LEGEND = 26
FONTSIZE_TEXT = 30
FONTSIZE_SMALL = 24

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

# Graphviz header
GRAPHVIZ_HEADER = '''"""
{description}

Generated chart: {output_file}
Part of Week 9: NLP Decoding Strategies
"""

import graphviz
import subprocess
import os

'''

def extract_function_code(source_file, function_name):
    """Extract a complete function from source file using AST."""
    try:
        with open(f"python/{source_file}", 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Get the source code for this function
                lines = content.split('\n')
                start_line = node.lineno - 1
                end_line = node.end_lineno

                func_lines = lines[start_line:end_line]
                return '\n'.join(func_lines)

        return None
    except Exception as e:
        print(f"    Error parsing {source_file}: {e}")
        return None


def create_standalone_from_function(source_file, function_name, output_file, description):
    """Extract function and create standalone script."""
    func_code = extract_function_code(source_file, function_name)

    if not func_code:
        print(f"  X Could not extract {function_name} from {source_file}")
        return False

    # Choose header based on whether it uses graphviz
    if 'graphviz' in func_code.lower():
        header = GRAPHVIZ_HEADER.format(description=description, output_file=output_file)
        # May need additional imports
        if 'patches' in func_code:
            header = header.replace('import graphviz', 'import graphviz\nimport matplotlib.pyplot as plt\nimport matplotlib.patches as patches')
    else:
        header = MATPLOTLIB_HEADER.format(description=description, output_file=output_file)

    # Fix output paths
    func_code_fixed = re.sub(r"['\"]\.\.\/figures\/[^'\"]*['\"]", f"'./{output_file}'", func_code)

    # Build complete script
    script_content = header + '\n' + func_code_fixed + f'\n\nif __name__ == "__main__":\n    {function_name}()\n    print(f"Generated {output_file}")\n'

    # Write to file
    script_name = f"generate_{output_file.replace('.pdf', '')}.py"
    script_path = output_dir / script_name

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    print(f"  OK {script_name}")
    return True


# Define all extractions
extractions = [
    # From generate_week09_charts_bsc_enhanced.py
    ('generate_week09_charts_bsc_enhanced.py', 'generate_quality_diversity_tradeoff', 'quality_diversity_tradeoff_bsc.pdf', 'Quality vs diversity tradeoff scatter plot'),
    ('generate_week09_charts_bsc_enhanced.py', 'generate_temperature_effects', 'temperature_effects_bsc.pdf', 'Temperature effects on probability distribution'),
    ('generate_week09_charts_bsc_enhanced.py', 'generate_temperature_calculation', 'temperature_calculation_bsc.pdf', 'Temperature calculation formula visualization'),
    ('generate_week09_charts_bsc_enhanced.py', 'generate_topk_filtering', 'topk_filtering_bsc.pdf', 'Top-k filtering process'),
    ('generate_week09_charts_bsc_enhanced.py', 'generate_topk_example', 'topk_example_bsc.pdf', 'Top-k numerical example'),
    ('generate_week09_charts_bsc_enhanced.py', 'generate_nucleus_process', 'nucleus_process_bsc.pdf', 'Nucleus sampling process'),
    ('generate_week09_charts_bsc_enhanced.py', 'generate_nucleus_cumulative', 'nucleus_cumulative_bsc.pdf', 'Nucleus cumulative distribution'),
    ('generate_week09_charts_bsc_enhanced.py', 'generate_degeneration_problem', 'degeneration_problem_bsc.pdf', 'Text degeneration problem'),
    ('generate_week09_charts_bsc_enhanced.py', 'generate_contrastive_mechanism', 'contrastive_mechanism_bsc.pdf', 'Contrastive search mechanism'),
    ('generate_week09_charts_bsc_enhanced.py', 'generate_contrastive_vs_nucleus', 'contrastive_vs_nucleus_bsc.pdf', 'Contrastive vs nucleus comparison'),
    ('generate_week09_charts_bsc_enhanced.py', 'generate_quality_diversity_pareto', 'quality_diversity_pareto_bsc.pdf', 'Pareto frontier visualization'),
    ('generate_week09_charts_bsc_enhanced.py', 'generate_task_method_decision_tree', 'task_method_decision_tree_bsc.pdf', 'Task to method decision tree'),
    ('generate_week09_charts_bsc_enhanced.py', 'generate_task_recommendations_table', 'task_recommendations_table_bsc.pdf', 'Task recommendations table'),
    ('generate_week09_charts_bsc_enhanced.py', 'generate_computational_cost_comparison', 'computational_cost_comparison_bsc.pdf', 'Computational cost comparison'),
    ('generate_week09_charts_bsc_enhanced.py', 'generate_production_settings', 'production_settings_bsc.pdf', 'Production API settings'),
    ('generate_week09_charts_bsc_enhanced.py', 'generate_prediction_pipeline', 'prediction_to_text_pipeline_bsc.pdf', 'Prediction to text pipeline'),
    ('generate_week09_charts_bsc_enhanced.py', 'generate_beam_example_tree', 'beam_example_tree_bsc.pdf', 'Beam search example tree'),

    # From generate_week09_enhanced_charts.py
    ('generate_week09_enhanced_charts.py', 'generate_quality_diversity_scatter', 'quality_diversity_scatter_bsc.pdf', 'Quality-diversity scatter with Pareto'),
    ('generate_week09_enhanced_charts.py', 'generate_extreme_case_1_greedy', 'extreme_greedy_single_path_bsc.pdf', 'Extreme case greedy single path'),
    ('generate_week09_enhanced_charts.py', 'generate_extreme_case_2_full_beam', 'extreme_full_beam_explosion_bsc.pdf', 'Extreme case full beam explosion'),
    ('generate_week09_enhanced_charts.py', 'generate_extreme_case_3_computational_cost', 'extreme_computational_cost_bsc.pdf', 'Extreme case computational cost'),
    ('generate_week09_enhanced_charts.py', 'generate_extreme_coverage_comparison', 'extreme_coverage_comparison_bsc.pdf', 'Extreme case coverage comparison'),
    ('generate_week09_enhanced_charts.py', 'generate_practical_methods_coverage', 'practical_methods_coverage_bsc.pdf', 'Practical methods coverage heatmaps'),

    # From fix_week09_final_charts.py
    ('fix_week09_final_charts.py', 'generate_problem1_repetition', 'problem1_repetition_output_bsc.pdf', 'Problem 1 repetition output'),
    ('fix_week09_final_charts.py', 'generate_problem2_diversity', 'problem2_diversity_output_bsc.pdf', 'Problem 2 diversity output'),
    ('fix_week09_final_charts.py', 'generate_problem3_balance', 'problem3_balance_output_bsc.pdf', 'Problem 3 balance output'),
    ('fix_week09_final_charts.py', 'generate_problem5_distribution', 'problem5_distribution_output_bsc.pdf', 'Problem 5 distribution output'),
    ('fix_week09_final_charts.py', 'generate_problem6_speed', 'problem6_speed_output_bsc.pdf', 'Problem 6 speed output'),

    # From fix_charts_redesign.py
    ('fix_charts_redesign.py', 'create_vocabulary_probability_chart', 'vocabulary_probability_bsc.pdf', 'Vocabulary probability distribution'),
]

print("=" * 70)
print("Extracting Individual Chart Scripts")
print("=" * 70)

success_count = 0
for source, func, output, desc in extractions:
    if create_standalone_from_function(source, func, output, desc):
        success_count += 1

# Copy already-standalone scripts
standalone_scripts = [
    'generate_beam_search_graphviz.py',
    'generate_full_exploration_graphviz.py',
    'generate_greedy_suboptimal_comparison.py',
]

print("\nCopying standalone scripts:")
for script in standalone_scripts:
    src = Path(f"python/{script}")
    if src.exists():
        dest = output_dir / script
        shutil.copy2(src, dest)
        print(f"  OK {script}")
        success_count += 1

# Handle problem4 special case (one function generates 4 charts)
print("\nNote: problem4_graphviz_charts generates 4 PDFs - needs manual split")

print("\n" + "=" * 70)
print(f"Created {success_count} standalone chart scripts")
print(f"Output: {output_dir.absolute()}")
print("=" * 70)
print("\nEach script:")
print("  - Generates ONE PDF in current directory")
print("  - Has all imports and styling")
print("  - Can run independently")
