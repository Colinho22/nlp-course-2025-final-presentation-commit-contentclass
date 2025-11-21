"""
Quality vs diversity tradeoff scatter plot

Generated chart: quality_diversity_tradeoff_bsc.pdf
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


def generate_quality_diversity_tradeoff():
    """Discovery hook: The paradox of text generation"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create scatter plot showing the tradeoff
    # Greedy: High quality, low diversity
    # Beam: High quality, low diversity
    # Temp high: Low quality, high diversity
    # Nucleus: Balanced

    methods = ['Greedy', 'Beam\nSearch', 'Temp\n(low)', 'Temp\n(high)',
               'Top-k', 'Nucleus\n(Top-p)']
    quality = [85, 88, 70, 45, 65, 80]
    diversity = [15, 20, 60, 95, 70, 75]
    colors_map = [COLOR_RED, COLOR_RED, COLOR_GRAY, COLOR_RED,
                  COLOR_GRAY, COLOR_GREEN]

    for i, (method, q, d, c) in enumerate(zip(methods, quality, diversity, colors_map)):
        ax.scatter(d, q, s=1000, c=c, alpha=0.7, edgecolors=COLOR_MAIN, linewidths=3)
        ax.annotate(method, (d, q), fontsize=42, ha='center', va='center',
                   weight='bold', color='white')

    # Add "Sweet Spot" annotation
    ax.annotate('Sweet Spot', xy=(75, 80), xytext=(85, 90),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_GREEN),
               fontsize=40, weight='bold', color=COLOR_GREEN)

    ax.set_xlabel('Diversity (Creativity)', fontsize=FONTSIZE_LABEL, weight='bold')
    ax.set_ylabel('Quality (Coherence)', fontsize=FONTSIZE_LABEL, weight='bold')
    ax.set_title('The Decoding Paradox: Quality vs Diversity Tradeoff',
                fontsize=FONTSIZE_TITLE, weight='bold', pad=20)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Add problem zones
    ax.axhspan(0, 50, alpha=0.05, color=COLOR_RED, label='Too Random')
    ax.axvspan(0, 30, alpha=0.05, color=COLOR_RED, label='Too Boring')

    set_minimalist_style(ax)

    plt.tight_layout()
    plt.savefig('./quality_diversity_tradeoff_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: quality_diversity_tradeoff_bsc.pdf")

if __name__ == "__main__":
    generate_quality_diversity_tradeoff()
    print(f"Generated quality_diversity_tradeoff_bsc.pdf")
