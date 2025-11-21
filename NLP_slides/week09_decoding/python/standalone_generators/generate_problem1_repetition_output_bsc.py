"""
Problem 1 repetition output

Generated chart: problem1_repetition_output_bsc.pdf
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


def generate_problem1_repetition():
    """Problem 1: Repetition loops with full output text"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Main output text box
    output_text = "The city is a major city in the city in the city..."

    # Create visual representation of repetition
    words = ["The", "city", "is", "a", "major", "city", "in", "the", "city", "in", "the", "city", "..."]
    x_positions = np.arange(len(words))

    # Highlight repeated phrase
    for i in range(len(words)):
        if words[i] == "city":
            ax.bar(i, 1, color=COLOR_RED, alpha=0.7, width=0.8)
        elif words[i] in ["in", "the"]:
            ax.bar(i, 1, color=COLOR_ORANGE, alpha=0.5, width=0.8)
        else:
            ax.bar(i, 1, color=COLOR_GRAY, alpha=0.3, width=0.8)

    # Add word labels
    for i, word in enumerate(words):
        ax.text(i, 0.5, word, ha='center', va='center',
                fontsize=FONTSIZE_TEXT, weight='bold', color='white')

    # Add loop indicator arrows
    ax.annotate('', xy=(10.5, 1.2), xytext=(5.5, 1.2),
                arrowprops=dict(arrowstyle='<->', lw=3, color=COLOR_RED))
    ax.text(8, 1.35, 'LOOP', ha='center', fontsize=FONTSIZE_LABEL,
            weight='bold', color=COLOR_RED)

    # Add full output text at bottom
    ax.text(6, -0.5, f'Output: "{output_text}"', ha='center', va='center',
            fontsize=FONTSIZE_ANNOTATION, style='italic', color=COLOR_MAIN)

    # Add problem label
    ax.text(6, 1.8, 'Greedy Decoding Gets Stuck', ha='center',
            fontsize=FONTSIZE_TITLE, weight='bold', color=COLOR_MAIN)

    ax.set_xlim(-0.5, len(words) - 0.5)
    ax.set_ylim(-0.7, 2)
    ax.axis('off')

    plt.tight_layout()
    # Add QuantLet attribution
    ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_Problem1_Repetition_Output',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#888888',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))
    plt.savefig('./problem1_repetition_output_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[CREATED] problem1_repetition_output_bsc.pdf")

if __name__ == "__main__":
    generate_problem1_repetition()
    print(f"Generated problem1_repetition_output_bsc.pdf")
