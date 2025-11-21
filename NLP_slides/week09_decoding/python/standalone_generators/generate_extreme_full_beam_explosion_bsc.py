"""
Extreme case full beam explosion

Generated chart: extreme_full_beam_explosion_bsc.pdf
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


def generate_extreme_case_2_full_beam():
    """
    Extreme Case 2: Full beam search (exponential explosion)
    Shows how vocabulary size 100 leads to exponential growth.
    """
    print("Generating extreme_full_beam_explosion_bsc.pdf...")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Draw exponentially growing tree
    vocab_size = 100
    levels = 5

    for level in range(levels):
        y = 5 - level
        num_nodes = vocab_size ** level if level < 3 else 200  # Cap visual

        # Calculate actual count
        actual_count = vocab_size ** level

        if level < 3:
            # Draw all nodes
            x_span = min(16, 2 * np.sqrt(num_nodes))
            x_positions = np.linspace(-x_span/2, x_span/2, num_nodes)

            for x in x_positions:
                size = 0.3 if level == 0 else max(0.05, 0.3 - level * 0.05)
                circle = Circle((x, y), size,
                              facecolor=COLOR_ORANGE,
                              edgecolor=COLOR_RED,
                              linewidth=1, alpha=0.6)
                ax.add_patch(circle)
        else:
            # Just show representative cluster
            x_positions = np.linspace(-8, 8, 200)
            for x in x_positions:
                circle = Circle((x, y), 0.02,
                              facecolor=COLOR_RED,
                              edgecolor=COLOR_RED,
                              linewidth=0.5, alpha=0.4)
                ax.add_patch(circle)

        # Label with count
        label = f'Step {level}: {actual_count:,} nodes'
        if actual_count >= 1000000:
            label = f'Step {level}: {actual_count/1e6:.0f}M nodes'
        if actual_count >= 1000000000:
            label = f'Step {level}: {actual_count/1e9:.1f}B nodes'

        ax.text(-9, y, label, fontsize=10, ha='right', va='center',
               fontweight='bold', color=COLOR_MAIN,
               bbox=dict(boxstyle='round,pad=0.4',
                        facecolor='white', alpha=0.9))

    # Add title
    ax.text(0, 6.5, 'Extreme Case 2: Full Search Space',
           fontsize=16, fontweight='bold', ha='center', color=COLOR_MAIN)

    ax.text(0, 6.1, 'Vocabulary size = 100, explore ALL paths',
           fontsize=11, ha='center', color=COLOR_GRAY, style='italic')

    # Computation box
    comp_text = 'Total paths: 100^5 = 10 billion\n\nIf 1 μs per path:\n10 billion × 1 μs = 2.8 hours\n\nIf 1 ms per path:\n10 billion × 1 ms = 115 days!'
    ax.text(0, -0.5, comp_text, fontsize=9, ha='center',
           bbox=dict(boxstyle='round,pad=0.8',
                    facecolor=COLOR_LIGHT,
                    edgecolor=COLOR_RED, linewidth=2),
           color=COLOR_MAIN, fontweight='bold')

    ax.set_xlim(-11, 10)
    ax.set_ylim(-2, 7)
    ax.axis('off')

    plt.tight_layout()
    # Add QuantLet attribution
    ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_Extreme_Full_Beam_Explosion',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#888888',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))
    plt.savefig('./extreme_full_beam_explosion_bsc.pdf',
               dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] extreme_full_beam_explosion_bsc.pdf created")

if __name__ == "__main__":
    generate_extreme_case_2_full_beam()
    print(f"Generated extreme_full_beam_explosion_bsc.pdf")
