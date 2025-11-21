"""
Extreme case computational cost

Generated chart: extreme_computational_cost_bsc.pdf
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


def generate_extreme_case_3_computational_cost():
    """
    Extreme Case 3: Computational cost comparison
    Bar chart showing 1 vs 100 vs 10,000 vs 1,000,000 vs 10,000,000,000 paths.
    """
    print("Generating extreme_computational_cost_bsc.pdf...")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Data for different search strategies
    strategies = ['Greedy\n(1 path)',
                 'Beam k=5\n(500 paths)',
                 'Beam k=50\n(50K paths)',
                 'Top-k=100\n(10M paths)',
                 'Full Search\n(10B paths)']

    paths = [1, 500, 50000, 10000000, 10000000000]

    # Time assuming 1 microsecond per path evaluation
    times_us = np.array(paths)  # in microseconds
    times_sec = times_us / 1e6  # convert to seconds

    # Use log scale
    log_times = np.log10(times_sec + 1e-10)  # Add small constant to avoid log(0)

    # Create bars with color gradient
    colors_list = [COLOR_GREEN, COLOR_BLUE, COLOR_ORANGE, COLOR_ORANGE, COLOR_RED]

    bars = ax.barh(strategies, log_times, color=colors_list,
                   edgecolor=COLOR_MAIN, linewidth=1.5, alpha=0.8)

    # Add actual time labels
    for i, (strategy, time_sec) in enumerate(zip(strategies, times_sec)):
        if time_sec < 0.001:
            time_label = f'{time_sec*1e6:.1f} μs'
        elif time_sec < 1:
            time_label = f'{time_sec*1e3:.1f} ms'
        elif time_sec < 60:
            time_label = f'{time_sec:.1f} sec'
        elif time_sec < 3600:
            time_label = f'{time_sec/60:.1f} min'
        elif time_sec < 86400:
            time_label = f'{time_sec/3600:.1f} hours'
        else:
            time_label = f'{time_sec/86400:.1f} days'

        ax.text(log_times[i] + 0.3, i, time_label,
               va='center', fontsize=11, fontweight='bold', color=COLOR_MAIN)

    # Formatting
    ax.set_xlabel('Computation Time (log scale)',
                  fontsize=13, fontweight='bold', color=COLOR_MAIN)
    ax.set_title('Computational Cost vs Search Breadth\n(Assuming 1 μs per path, sequence length=5)',
                fontsize=14, fontweight='bold', color=COLOR_MAIN, pad=20)

    # Add vertical lines for reference
    ax.axvline(x=np.log10(0.001), color=COLOR_GRAY, linestyle='--',
              alpha=0.5, linewidth=1)
    ax.text(np.log10(0.001), len(strategies), '1 ms',
           ha='center', va='bottom', fontsize=9, color=COLOR_GRAY)

    ax.axvline(x=np.log10(1), color=COLOR_GRAY, linestyle='--',
              alpha=0.5, linewidth=1)
    ax.text(np.log10(1), len(strategies), '1 sec',
           ha='center', va='bottom', fontsize=9, color=COLOR_GRAY)

    # Grid and spines
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Custom x-ticks
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.set_xticklabels(['1 μs', '1 ms', '1 sec', '16 min', '11 days'])

    plt.tight_layout()
    # Add QuantLet attribution
    ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_Extreme_Computational_Cost',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#888888',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))
    plt.savefig('./extreme_computational_cost_bsc.pdf',
               dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] extreme_computational_cost_bsc.pdf created")

if __name__ == "__main__":
    generate_extreme_case_3_computational_cost()
    print(f"Generated extreme_computational_cost_bsc.pdf")
