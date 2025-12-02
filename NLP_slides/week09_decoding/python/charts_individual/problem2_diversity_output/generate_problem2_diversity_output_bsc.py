"""
Problem 2 diversity output

Generated chart: problem2_diversity_output_bsc.pdf
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


def generate_problem2_diversity():
    """Problem 2: No diversity/nonsense words"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Show multiple outputs with no diversity
    outputs = [
        "The weather is zxqp today",
        "I went to the flurb yesterday",
        "She likes to eat qwerty food",
        "The glorp is very blorptastic"
    ]

    y_pos = np.arange(len(outputs))

    # Create boxes for each output
    for i, output in enumerate(outputs):
        # Background box
        rect = patches.FancyBboxPatch((0.1, i - 0.35), 10, 0.7,
                                      boxstyle="round,pad=0.05",
                                      facecolor=COLOR_LIGHT,
                                      edgecolor=COLOR_RED, linewidth=2)
        ax.add_patch(rect)

        # Highlight nonsense words in red
        words = output.split()
        x_offset = 0.3
        for word in words:
            if word in ['zxqp', 'flurb', 'qwerty', 'glorp', 'blorptastic']:
                ax.text(x_offset, i, word, fontsize=FONTSIZE_TEXT,
                       color=COLOR_RED, weight='bold', va='center')
            else:
                ax.text(x_offset, i, word, fontsize=FONTSIZE_TEXT,
                       color=COLOR_MAIN, va='center')
            x_offset += 1.2

    # Add title
    ax.text(5, 4.5, 'High Temperature Creates Nonsense', ha='center',
            fontsize=FONTSIZE_TITLE, weight='bold', color=COLOR_MAIN)

    # Add problem indicator
    ax.text(5, -1, 'Generated words not in vocabulary!', ha='center',
            fontsize=FONTSIZE_ANNOTATION, style='italic', color=COLOR_RED)

    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('./problem2_diversity_output_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[CREATED] problem2_diversity_output_bsc.pdf")

if __name__ == "__main__":
    generate_problem2_diversity()
    print(f"Generated problem2_diversity_output_bsc.pdf")
