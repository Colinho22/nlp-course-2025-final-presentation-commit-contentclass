"""
Extreme case coverage comparison

Generated chart: extreme_coverage_comparison_bsc.pdf
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


def generate_extreme_coverage_comparison():
    """
    The Extremes: Shows greedy vs full search coverage.
    Split from original search coverage - Part 1 of 2.
    """
    print("Generating extreme_coverage_comparison_bsc.pdf...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Create 2D representation of search space
    # X-axis: First word choice (0-100)
    # Y-axis: Second word choice (0-100)

    methods = [
        ('Greedy Decoding\n(1 path)', 1, COLOR_RED, 'Reds'),
        ('Full Search\n(All 10,000 paths)', 'full', COLOR_GREEN, 'Greens')
    ]

    for idx, (method_name, beam_width, color, cmap) in enumerate(methods):
        ax = axes[idx]

        # Create heat map
        grid = np.zeros((100, 100))

        if beam_width == 1:
            # Greedy: single point
            grid[50, 50] = 1.0
        else:
            # Full search: everything explored
            grid = np.ones((100, 100))

        # Plot heat map
        im = ax.imshow(grid, cmap=cmap,
                      origin='lower', extent=[0, 100, 0, 100],
                      vmin=0, vmax=1, alpha=0.8)

        # Calculate coverage
        coverage = (grid > 0).sum() / (100 * 100) * 100

        # Title
        ax.set_title(f'{method_name}\nCoverage: {coverage:.2f}%',
                    fontsize=18, fontweight='bold', color=COLOR_MAIN, pad=10)

        ax.set_xlabel('First Word (100 options)', fontsize=16, color=COLOR_GRAY)
        ax.set_ylabel('Second Word (100 options)', fontsize=16, color=COLOR_GRAY)

        # Add grid
        ax.grid(True, alpha=0.2, color='white', linewidth=0.5)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Explored', fontsize=16, color=COLOR_GRAY)

    plt.suptitle('The Extremes: Coverage Comparison\n(Vocabulary=100, showing first 2 words only)',
                fontsize=18, fontweight='bold', color=COLOR_MAIN, y=1.02)

    plt.tight_layout()
    plt.savefig('./extreme_coverage_comparison_bsc.pdf',
               dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] extreme_coverage_comparison_bsc.pdf created")

if __name__ == "__main__":
    generate_extreme_coverage_comparison()
    print(f"Generated extreme_coverage_comparison_bsc.pdf")
