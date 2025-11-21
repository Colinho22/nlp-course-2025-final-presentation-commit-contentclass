"""
Vocabulary probability distribution

Generated chart: vocabulary_probability_bsc.pdf
Part of Week 9: NLP Decoding Strategies
"""

# QuantLet Metadata
QUANTLET_URL = "https://quantlet.com/NLPDecoding_Vocabularyprobability"


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


def create_vocabulary_probability_chart():
    """NEW: Bar chart showing probability distribution over vocabulary"""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Example: "The cat " probabilities
    words = ['sat', 'is', 'jumped', 'ran', 'slept', 'played', '...', '(49,994 more)']
    probs = [0.45, 0.30, 0.25, 0.18, 0.12, 0.08, 0.05, 0.02]

    # Create horizontal bars
    y_positions = np.arange(len(words))
    bars = ax.barh(y_positions, probs, color=COLOR_ACCENT, edgecolor=COLOR_MAIN, linewidth=2, alpha=0.8)

    # Highlight top 3
    for i in range(3):
        bars[i].set_color(COLOR_GREEN)
        bars[i].set_alpha(0.9)

    # Fade last two (tail)
    bars[-2].set_alpha(0.3)
    bars[-1].set_alpha(0.1)

    # Add probability values
    for i, (prob, word) in enumerate(zip(probs, words)):
        if i < len(words) - 1:
            ax.text(prob + 0.02, i, f'{prob:.2f}', va='center',
                    fontsize=FONTSIZE_ANNOTATION, weight='bold', color=COLOR_MAIN)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(words, fontsize=FONTSIZE_LABEL)
    ax.set_xlabel('Probability P(word | "The cat ")', fontsize=FONTSIZE_LABEL, weight='bold')
    ax.set_title('The Decoding Challenge: Choose From 50,000 Words',
                 fontsize=FONTSIZE_TITLE, weight='bold', pad=20)

    ax.set_xlim(0, 0.55)
    ax.invert_yaxis()  # Highest probability at top

    # Add annotation
    ax.text(0.52, 7, '← Tail (99.9% of vocab)', va='center',
            fontsize=FONTSIZE_ANNOTATION, style='italic', color=COLOR_GRAY)

    set_minimalist_style(ax)

    plt.tight_layout()
    # Add QuantLet attribution
    ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_Vocabulary_Probability',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#888888',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))
    plt.savefig('./vocabulary_probability_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[CREATED] vocabulary_probability_bsc.pdf (slide 2 chart)")

if __name__ == "__main__":
    create_vocabulary_probability_chart()
    print(f"Generated vocabulary_probability_bsc.pdf")
