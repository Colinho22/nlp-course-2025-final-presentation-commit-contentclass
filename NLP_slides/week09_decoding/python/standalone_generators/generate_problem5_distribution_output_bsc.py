"""
Problem 5 distribution output

Generated chart: problem5_distribution_output_bsc.pdf
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


def generate_problem5_distribution():
    """Problem 5: Wrong probability distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Model's distribution (wrong)
    words1 = ['the', 'a', 'cat', 'dog', 'ran', 'jumped', 'quickly', '...']
    probs1 = [0.40, 0.30, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01]

    bars1 = ax1.bar(range(len(words1)), probs1, color=COLOR_RED, alpha=0.7)
    bars1[0].set_color(COLOR_RED)
    bars1[0].set_alpha(0.9)

    ax1.set_xticks(range(len(words1)))
    ax1.set_xticklabels(words1, rotation=45, ha='right')
    ax1.set_ylabel('Probability', fontsize=FONTSIZE_LABEL)
    ax1.set_title('Model Distribution\n(Wrong for Context)', fontsize=FONTSIZE_LABEL, weight='bold')
    ax1.set_ylim(0, 0.5)

    # Add problem indicator
    ax1.text(3.5, 0.35, 'Too much\nmass on\ncommon words!', ha='center',
            fontsize=FONTSIZE_SMALL, color=COLOR_RED, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=COLOR_RED))

    set_minimalist_style(ax1)

    # Right: True distribution (correct)
    words2 = ['quickly', 'gracefully', 'frantically', 'slowly', 'the', 'a', 'cat', '...']
    probs2 = [0.25, 0.20, 0.18, 0.15, 0.08, 0.06, 0.05, 0.03]

    bars2 = ax2.bar(range(len(words2)), probs2, color=COLOR_GREEN, alpha=0.7)
    for i in range(4):
        bars2[i].set_alpha(0.9)

    ax2.set_xticks(range(len(words2)))
    ax2.set_xticklabels(words2, rotation=45, ha='right')
    ax2.set_ylabel('Probability', fontsize=FONTSIZE_LABEL)
    ax2.set_title('True Distribution\n(Correct for Context)', fontsize=FONTSIZE_LABEL, weight='bold')
    ax2.set_ylim(0, 0.5)

    # Add correct indicator
    ax2.text(3.5, 0.35, 'Adverbs\nshould be\nmore likely!', ha='center',
            fontsize=FONTSIZE_SMALL, color=COLOR_GREEN, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=COLOR_GREEN))

    set_minimalist_style(ax2)

    # Overall title
    fig.suptitle('Context: "The cat ran ___"', fontsize=FONTSIZE_TITLE, weight='bold', y=1.02)

    plt.tight_layout()
    # Add QuantLet attribution
    ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_Problem5_Distribution_Output',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#888888',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))
    plt.savefig('./problem5_distribution_output_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[CREATED] problem5_distribution_output_bsc.pdf")

if __name__ == "__main__":
    generate_problem5_distribution()
    print(f"Generated problem5_distribution_output_bsc.pdf")
