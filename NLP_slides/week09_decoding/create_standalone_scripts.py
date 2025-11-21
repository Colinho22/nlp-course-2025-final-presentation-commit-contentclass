"""
Create standalone Python scripts - one per chart.

Extracts individual chart generation functions from multi-chart scripts
and creates standalone executable scripts in standalone_generators/ folder.
"""

from pathlib import Path
import re

# Create output directory
output_dir = Path("python/standalone_generators")
output_dir.mkdir(parents=True, exist_ok=True)

# Figures used in presentation (from \includegraphics analysis)
FIGURES_NEEDED = [
    'vocabulary_probability_bsc.pdf',
    'prediction_to_text_pipeline_bsc.pdf',
    'extreme_greedy_single_path_bsc.pdf',
    'greedy_suboptimal_comparison_bsc.pdf',
    'extreme_full_beam_explosion_bsc.pdf',
    'full_exploration_tree_graphviz.pdf',
    'extreme_coverage_comparison_bsc.pdf',
    'practical_methods_coverage_bsc.pdf',
    'beam_search_tree_graphviz.pdf',
    'temperature_effects_bsc.pdf',
    'temperature_calculation_bsc.pdf',
    'topk_filtering_bsc.pdf',
    'topk_example_bsc.pdf',
    'nucleus_process_bsc.pdf',
    'nucleus_cumulative_bsc.pdf',
    'degeneration_problem_bsc.pdf',
    'contrastive_mechanism_bsc.pdf',
    'contrastive_vs_nucleus_bsc.pdf',
    'problem1_repetition_output_bsc.pdf',
    'problem2_diversity_output_bsc.pdf',
    'problem3_balance_output_bsc.pdf',
    'problem4_search_tree_pruning_bsc.pdf',
    'problem4_path_comparison_bsc.pdf',
    'problem4_probability_evolution_bsc.pdf',
    'problem4_recovery_problem_bsc.pdf',
    'problem5_distribution_output_bsc.pdf',
    'problem6_speed_output_bsc.pdf',
    'quality_diversity_scatter_bsc.pdf',
    'quality_diversity_pareto_bsc.pdf',
    'task_method_decision_tree_bsc.pdf',
    'task_recommendations_table_bsc.pdf',
    'production_settings_bsc.pdf',
]

# Common header for all scripts
COMMON_HEADER = '''"""
{description}

Part of Week 9 NLP Decoding Strategies presentation.
Generated chart: {output_file}
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

# BSc Discovery Color Scheme
COLOR_MAIN = '#404040'      # Main elements (dark gray)
COLOR_ACCENT = '#3333B2'    # Key concepts (purple)
COLOR_GRAY = '#B4B4B4'      # Secondary elements
COLOR_LIGHT = '#F0F0F0'     # Backgrounds
COLOR_GREEN = '#2CA02C'     # Success/positive
COLOR_RED = '#D62728'       # Error/negative
COLOR_ORANGE = '#FF7F0E'    # Warning/medium
COLOR_BLUE = '#0066CC'      # Information

plt.style.use('seaborn-v0_8-whitegrid')
'''

# Mapping of figures to their source and description
FIGURE_INFO = {
    'vocabulary_probability_bsc.pdf': ('generate_week09_charts_CLEAN.py', 'Simple bar chart showing vocabulary probability distribution'),
    'prediction_to_text_pipeline_bsc.pdf': ('generate_week09_charts_CLEAN.py', 'Pipeline from model output to final text'),
    'beam_search_tree_graphviz.pdf': ('generate_beam_search_graphviz.py', 'Beam search tree with pruning using graphviz'),
    'full_exploration_tree_graphviz.pdf': ('generate_full_exploration_graphviz.py', 'Full exploration exponential explosion tree'),
    'greedy_suboptimal_comparison_bsc.pdf': ('generate_greedy_suboptimal_comparison.py', 'Greedy vs optimal path comparison'),
    # Enhanced charts
    'quality_diversity_scatter_bsc.pdf': ('generate_week09_enhanced_charts.py', 'Quality-diversity scatter plot with Pareto frontier'),
    'problem4_search_tree_pruning_bsc.pdf': ('generate_week09_enhanced_charts.py', 'Beam search pruning graphviz tree'),
    'problem4_path_comparison_bsc.pdf': ('generate_week09_enhanced_charts.py', 'Path comparison graphviz'),
    'problem4_probability_evolution_bsc.pdf': ('generate_week09_enhanced_charts.py', 'Probability evolution graphviz'),
    'problem4_recovery_problem_bsc.pdf': ('generate_week09_enhanced_charts.py', 'Recovery problem graphviz'),
    'extreme_greedy_single_path_bsc.pdf': ('generate_week09_enhanced_charts.py', 'Greedy single path extreme case'),
    'extreme_full_beam_explosion_bsc.pdf': ('generate_week09_enhanced_charts.py', 'Full beam explosion extreme case'),
    'extreme_computational_cost_bsc.pdf': ('generate_week09_enhanced_charts.py', 'Computational cost comparison'),
    'extreme_coverage_comparison_bsc.pdf': ('generate_week09_enhanced_charts.py', 'Coverage comparison heatmaps'),
    'practical_methods_coverage_bsc.pdf': ('generate_week09_enhanced_charts.py', 'Practical methods coverage'),
    # CLEAN script charts
    'temperature_effects_bsc.pdf': ('generate_week09_charts_CLEAN.py', 'Temperature effects 3-panel comparison'),
    'topk_filtering_bsc.pdf': ('generate_week09_charts_CLEAN.py', 'Top-k filtering with cutoff'),
    'nucleus_process_bsc.pdf': ('generate_week09_charts_CLEAN.py', 'Nucleus sampling process'),
    'nucleus_cumulative_bsc.pdf': ('generate_week09_charts_CLEAN.py', 'Nucleus cumulative distribution'),
    'degeneration_problem_bsc.pdf': ('generate_week09_charts_CLEAN.py', 'Degeneration problem visualization'),
    'quality_diversity_pareto_bsc.pdf': ('generate_week09_charts_CLEAN.py', 'Pareto frontier version'),
    'computational_cost_comparison_bsc.pdf': ('generate_week09_charts_CLEAN.py', 'Cost comparison bars'),
    # Missing/additional charts
    'topk_example_bsc.pdf': ('generate_missing_charts.py', 'Top-k numerical example'),
    'contrastive_mechanism_bsc.pdf': ('generate_missing_charts.py', 'Contrastive search mechanism'),
    'contrastive_vs_nucleus_bsc.pdf': ('generate_missing_charts.py', 'Contrastive vs nucleus comparison'),
    'production_settings_bsc.pdf': ('generate_missing_charts.py', 'Production API settings'),
}

print(f"Found {len(FIGURES_NEEDED)} figures in presentation")
print(f"Mapped {len(FIGURE_INFO)} to source scripts")
print(f"Need to locate {len(FIGURES_NEEDED) - len(FIGURE_INFO)} additional figures\n")

# List unmapped figures
unmapped = [f for f in FIGURES_NEEDED if f not in FIGURE_INFO]
if unmapped:
    print("Unmapped figures (need to find generation code):")
    for f in unmapped:
        print(f"  - {f}")

print(f"\nNext: Extract individual functions and create {len(FIGURE_INFO)} standalone scripts")
print(f"Output: {output_dir}/")
