"""
Computational cost comparison

Generated chart: computational_cost_comparison_bsc.pdf
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


def generate_computational_cost_comparison():
    """Speed vs quality tradeoff for all methods"""
    fig, ax = plt.subplots(figsize=(11, 6))

    # Methods with relative costs (greedy = 1.0 baseline)
    methods = ['Greedy', 'Temperature', 'Top-k', 'Nucleus', 'Beam\n(w=3)', 'Contrastive']
    relative_time = [1.0, 1.1, 1.2, 1.3, 4.5, 12.0]
    quality_score = [65, 55, 68, 80, 88, 85]
    colors_cost = [COLOR_GREEN, COLOR_GREEN, COLOR_GREEN, COLOR_GREEN, COLOR_GRAY, COLOR_RED]

    x_pos = np.arange(len(methods))

    # Dual axis: time (bars) and quality (line)
    bars = ax.bar(x_pos, relative_time, color=colors_cost, alpha=0.7,
                 edgecolor=COLOR_MAIN, linewidth=3)

    # Add time labels on bars
    for i, (x, t) in enumerate(zip(x_pos, relative_time)):
        ax.text(x, t + 0.3, f'{t:.1f}x', ha='center', fontsize=42, weight='bold')

    # Quality line on secondary axis
    ax2 = ax.twinx()
    ax2.plot(x_pos, quality_score, color=COLOR_ACCENT, linewidth=3, marker='o',
            markersize=20, label='Quality Score')

    # Annotations
    ax.axhline(5, color=COLOR_GRAY, linestyle='--', alpha=0.4, linewidth=1)
    ax.text(len(methods) - 0.5, 5.5, '5x slower than greedy',
           fontsize=42, style='italic', color=COLOR_GRAY)

    # Labels
    ax.set_xlabel('Decoding Method', fontsize=40, weight='bold')
    ax.set_ylabel('Relative Computation Time (vs Greedy)', fontsize=38, weight='bold', color=COLOR_MAIN)
    ax2.set_ylabel('Quality Score (0-100)', fontsize=38, weight='bold', color=COLOR_ACCENT)
    ax.set_title('Computational Cost vs Quality Tradeoff', fontsize=42, weight='bold', pad=15)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, max(relative_time) * 1.2)
    ax2.set_ylim(0, 100)

    # Legend
    ax2.legend(loc='upper left', fontsize=42)

    set_minimalist_style(ax)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color(COLOR_ACCENT)
    ax2.tick_params(colors=COLOR_ACCENT)

    # Insight box
    ax.text(0.5, -0.22, 'Insight: Contrastive gives best quality-diversity but 12x slower. Nucleus is best balanced choice.',
           ha='center', transform=ax.transAxes, fontsize=42, style='italic',
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN))

    plt.tight_layout()
    plt.savefig('./computational_cost_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: computational_cost_comparison_bsc.pdf")

if __name__ == "__main__":
    generate_computational_cost_comparison()
    print(f"Generated computational_cost_comparison_bsc.pdf")
