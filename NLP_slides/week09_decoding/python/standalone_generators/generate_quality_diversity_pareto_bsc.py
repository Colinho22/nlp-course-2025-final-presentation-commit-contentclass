"""
Pareto frontier visualization

Generated chart: quality_diversity_pareto_bsc.pdf
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


def generate_quality_diversity_pareto():
    """All 6 methods on quality-diversity plot"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # All 6 methods positioned
    methods_data = [
        ('Greedy', 85, 15, COLOR_RED, 'x'),
        ('Beam\nSearch', 88, 20, COLOR_RED, 's'),
        ('Temp\n(T=0.5)', 82, 35, COLOR_GRAY, 'o'),
        ('Temp\n(T=1.5)', 50, 85, COLOR_RED, 'o'),
        ('Top-k\n(k=40)', 70, 65, COLOR_GRAY, '^'),
        ('Nucleus\n(p=0.9)', 82, 78, COLOR_GREEN, 'D'),
        ('Contrastive\n(α=0.6)', 85, 82, COLOR_GREEN, '*')
    ]

    for method, quality, diversity, color, marker in methods_data:
        ax.scatter(diversity, quality, s=1000, c=color, marker=marker, alpha=0.7,
                  edgecolors=COLOR_MAIN, linewidths=3, zorder=3)
        # Offset label slightly
        offset_x = 3 if diversity < 50 else -3
        offset_y = 3 if quality < 70 else -3
        ax.text(diversity + offset_x, quality + offset_y, method,
               fontsize=FONTSIZE_ANNOTATION, ha='center', weight='bold')

    # Draw Pareto frontier (approximate)
    frontier_x = [15, 20, 78, 82, 85]
    frontier_y = [85, 88, 82, 85, 50]
    ax.plot(frontier_x, frontier_y, 'k--', alpha=0.3, linewidth=3, label='Pareto Frontier (approx)')

    # Optimal region
    optimal = patches.Rectangle((65, 75), 25, 20, linewidth=3,
                               edgecolor=COLOR_GREEN, facecolor=COLOR_GREEN,
                               alpha=0.1, linestyle='--')
    ax.add_patch(optimal)
    ax.text(77, 87, 'Optimal\nZone', ha='center', fontsize=42,
           weight='bold', color=COLOR_GREEN)

    # Problem zones
    ax.text(20, 20, 'Boring\nRepetitive', ha='center', fontsize=42,
           style='italic', color=COLOR_RED, alpha=0.6)
    ax.text(85, 30, 'Random\nNonsense', ha='center', fontsize=42,
           style='italic', color=COLOR_RED, alpha=0.6)

    ax.set_xlabel('Diversity (Creativity, Novelty) →', fontsize=40, weight='bold')
    ax.set_ylabel('Quality (Coherence, Correctness) →', fontsize=40, weight='bold')
    ax.set_title('Quality-Diversity Tradeoff: All Methods', fontsize=42, weight='bold', pad=15)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.legend(loc='lower left', fontsize=42)
    set_minimalist_style(ax)

    plt.tight_layout()
    # Add QuantLet attribution
    ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_Quality_Diversity_Pareto',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#888888',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))
    plt.savefig('./quality_diversity_pareto_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: quality_diversity_pareto_bsc.pdf")

if __name__ == "__main__":
    generate_quality_diversity_pareto()
    print(f"Generated quality_diversity_pareto_bsc.pdf")
