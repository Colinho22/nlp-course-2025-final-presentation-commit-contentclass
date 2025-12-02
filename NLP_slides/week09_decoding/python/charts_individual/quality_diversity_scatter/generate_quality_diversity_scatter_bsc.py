"""
Quality-diversity scatter with Pareto

Generated chart: quality_diversity_scatter_bsc.pdf
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

FONTSIZE_TITLE = 24
FONTSIZE_LABEL = 20
FONTSIZE_TICK = 16
FONTSIZE_ANNOTATION = 18
FONTSIZE_LEGEND = 18
FONTSIZE_TEXT = 20
FONTSIZE_SMALL = 18


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


def generate_quality_diversity_scatter():
    """
    Generate scatter plot showing quality-diversity tradeoff for 4 core methods.
    Shows: Greedy, Beam, Top-k, Nucleus on coherence vs diversity axes.
    """
    print("Generating quality_diversity_scatter_bsc.pdf...")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Method data: (diversity, coherence, label)
    # Based on typical behavior patterns
    methods = {
        'Greedy': (0.15, 0.85, COLOR_MAIN),
        'Beam (k=5)': (0.25, 0.80, COLOR_ACCENT),
        'Top-k (k=50)': (0.60, 0.65, COLOR_ORANGE),
        'Nucleus (p=0.9)': (0.70, 0.75, COLOR_GREEN)
    }

    # Plot methods
    for method, (div, coh, color) in methods.items():
        ax.scatter(div, coh, s=400, c=color, alpha=0.7,
                  edgecolors=COLOR_MAIN, linewidth=2, zorder=3)

        # Add labels with offset
        offset_x = 0.05 if div < 0.5 else -0.05
        offset_y = 0.03 if coh < 0.75 else -0.03
        ha = 'left' if div < 0.5 else 'right'

        ax.annotate(method, xy=(div, coh),
                   xytext=(div + offset_x, coh + offset_y),
                   fontsize=18, fontweight='bold', color=COLOR_MAIN,
                   ha=ha, va='center')

    # Draw Pareto frontier curve
    # Approximate frontier through the "good" methods
    frontier_x = np.array([0.15, 0.25, 0.60, 0.70, 0.75])
    frontier_y = np.array([0.85, 0.80, 0.65, 0.75, 0.70])

    # Fit smooth curve
    from scipy.interpolate import make_interp_spline
    x_smooth = np.linspace(frontier_x.min(), frontier_x.max(), 100)
    spl = make_interp_spline(frontier_x, frontier_y, k=2)
    y_smooth = spl(x_smooth)

    ax.plot(x_smooth, y_smooth, '--', color=COLOR_GRAY,
           linewidth=2, alpha=0.6, label='Pareto Frontier', zorder=1)

    # Add "Sweet Spot" region
    sweet_x = [0.55, 0.75]
    sweet_y = [0.65, 0.80]
    rect = Rectangle((sweet_x[0], sweet_y[0]),
                     sweet_x[1] - sweet_x[0],
                     sweet_y[1] - sweet_y[0],
                     facecolor=COLOR_GREEN, alpha=0.1,
                     edgecolor=COLOR_GREEN, linewidth=2,
                     linestyle='--', zorder=0)
    ax.add_patch(rect)
    ax.text(0.65, 0.62, 'Sweet Spot', fontsize=16,
           color=COLOR_GREEN, fontweight='bold', ha='center')

    # Add region labels
    ax.text(0.15, 0.15, 'Too Deterministic\n(Repetitive)',
           fontsize=16, color=COLOR_RED, ha='center', alpha=0.7,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_LIGHT,
                    edgecolor=COLOR_RED, alpha=0.3))

    ax.text(0.85, 0.15, 'Too Random\n(Incoherent)',
           fontsize=16, color=COLOR_RED, ha='center', alpha=0.7,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_LIGHT,
                    edgecolor=COLOR_RED, alpha=0.3))

    # Formatting
    ax.set_xlabel('Diversity (Entropy)', fontsize=18, fontweight='bold', color=COLOR_MAIN)
    ax.set_ylabel('Coherence (Quality)', fontsize=18, fontweight='bold', color=COLOR_MAIN)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    ax.legend(loc='lower left', fontsize=16, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('./quality_diversity_scatter_bsc.pdf',
               dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] quality_diversity_scatter_bsc.pdf created")

if __name__ == "__main__":
    generate_quality_diversity_scatter()
    print(f"Generated quality_diversity_scatter_bsc.pdf")
