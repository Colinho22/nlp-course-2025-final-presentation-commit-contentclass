"""
Practical methods coverage heatmaps

Generated chart: practical_methods_coverage_bsc.pdf
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


def generate_practical_methods_coverage():
    """
    Practical Solutions: Shows beam search, top-k, and nucleus coverage.
    Split from original search coverage - Part 2 of 2.
    """
    print("Generating practical_methods_coverage_bsc.pdf...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Seed for reproducibility
    np.random.seed(42)

    # Create 2D representation of search space
    methods = [
        ('Beam Search (k=5)\n25 paths', 5, COLOR_BLUE, 'Blues'),
        ('Top-k Sampling (k=20)\n~400 paths', 20, COLOR_ORANGE, 'Oranges'),
        ('Nucleus (p=0.9)\n~1000 paths', 35, COLOR_GREEN, 'Greens')
    ]

    for idx, (method_name, beam_width, color, cmap) in enumerate(methods):
        ax = axes[idx]

        # Create heat map
        grid = np.zeros((100, 100))

        if beam_width == 5:
            # Beam k=5: small cluster
            centers = [(50, 50), (48, 52), (52, 48), (51, 51), (49, 49)]
            for cx, cy in centers:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if 0 <= cx+dx < 100 and 0 <= cy+dy < 100:
                            grid[cx+dx, cy+dy] = max(grid[cx+dx, cy+dy],
                                                    1.0 - (abs(dx) + abs(dy))/5)
        elif beam_width == 20:
            # Top-k: medium spread
            for _ in range(400):
                cx, cy = np.random.randint(20, 80), np.random.randint(20, 80)
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if 0 <= cx+dx < 100 and 0 <= cy+dy < 100:
                            dist = np.sqrt(dx**2 + dy**2)
                            grid[cx+dx, cy+dy] = max(grid[cx+dx, cy+dy],
                                                    max(0, 1.0 - dist/5))
        else:
            # Nucleus: adaptive coverage
            for _ in range(1000):
                cx, cy = np.random.randint(15, 85), np.random.randint(15, 85)
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if 0 <= cx+dx < 100 and 0 <= cy+dy < 100:
                            dist = np.sqrt(dx**2 + dy**2)
                            grid[cx+dx, cy+dy] = max(grid[cx+dx, cy+dy],
                                                    max(0, 1.0 - dist/3))

        # Plot heat map
        im = ax.imshow(grid, cmap=cmap,
                      origin='lower', extent=[0, 100, 0, 100],
                      vmin=0, vmax=1, alpha=0.8)

        # Calculate coverage
        coverage = (grid > 0).sum() / (100 * 100) * 100

        # Title
        ax.set_title(f'{method_name}\nCoverage: {coverage:.2f}%',
                    fontsize=11, fontweight='bold', color=COLOR_MAIN, pad=10)

        ax.set_xlabel('First Word', fontsize=9, color=COLOR_GRAY)
        if idx == 0:
            ax.set_ylabel('Second Word', fontsize=9, color=COLOR_GRAY)

        # Add grid
        ax.grid(True, alpha=0.2, color='white', linewidth=0.5)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 50, 100])

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if idx == 2:
            cbar.set_label('Exploration\nIntensity', fontsize=8, color=COLOR_GRAY)

    # Add Sweet Spot indicator
    axes[2].text(85, 85, '← Sweet\n    Spot', fontsize=9,
               color=COLOR_GREEN, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=COLOR_GREEN, alpha=0.8))

    plt.suptitle('Practical Solutions: Balancing Coverage and Computation',
                fontsize=14, fontweight='bold', color=COLOR_MAIN, y=1.02)

    plt.tight_layout()
    # Add QuantLet attribution
    ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_Practical_Methods_Coverage',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#888888',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))
    plt.savefig('./practical_methods_coverage_bsc.pdf',
               dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] practical_methods_coverage_bsc.pdf created")

if __name__ == "__main__":
    generate_practical_methods_coverage()
    print(f"Generated practical_methods_coverage_bsc.pdf")
