"""
Beam search example tree

Generated chart: beam_example_tree_bsc.pdf
Part of Week 9: NLP Decoding Strategies
"""

# QuantLet Metadata
QUANTLET_URL = "https://quantlet.com/NLPDecoding_Beamexampletree"


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


def generate_beam_example_tree():
    """Concrete numerical example: 'The cat...' with beam width=3"""
    fig, ax = plt.subplots(figsize=(11, 7))

    # Tree structure with actual words and scores
    ax.text(0.5, 0.95, 'START: "The cat"', ha='center', fontsize=38,
           weight='bold', bbox=dict(boxstyle='round', facecolor=COLOR_ACCENT, alpha=0.8))

    # Step 1: Top 3 candidates
    step1_words = ['sat', 'jumped', 'walked']
    step1_scores = [0.45, 0.30, 0.25]
    for i, (word, score) in enumerate(zip(step1_words, step1_scores)):
        x = 0.2 + i * 0.3
        ax.text(x, 0.75, f'{word}\n{score:.2f}', ha='center', fontsize=42,
               bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN))
        ax.plot([0.5, x], [0.93, 0.77], 'k-', alpha=0.4, linewidth=1)

    # Step 2: Expand best 3 (only showing expansion from 'sat')
    ax.annotate('Expand each\nof top 3', xy=(0.2, 0.73), xytext=(0.05, 0.60),
               arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_ACCENT),
               fontsize=42, style='italic', color=COLOR_ACCENT)

    step2_words = ['on', 'down', 'still']
    step2_scores = [0.50, 0.30, 0.20]
    for i, (word, score) in enumerate(zip(step2_words, step2_scores)):
        x = 0.05 + i * 0.12
        cumulative = step1_scores[0] * score  # Multiply probabilities
        ax.text(x, 0.50, f'{word}\n{cumulative:.3f}', ha='center', fontsize=42,
               bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT if i < 3 else '#FFE6E6'))
        ax.plot([0.2, x], [0.73, 0.52], 'k-', alpha=0.3, linewidth=1)

    # Best path highlighted
    ax.text(0.05, 0.35, '"The cat sat on"\nScore: 0.225', ha='center', fontsize=42,
           weight='bold', bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, alpha=0.3))
    ax.plot([0.05, 0.05], [0.48, 0.37], color=COLOR_ACCENT, linewidth=3)

    # Key insight box
    ax.text(0.75, 0.50, 'Key: At each step\n• Score all candidates\n• Keep top 3\n• Prune rest\n• Repeat',
           ha='center', va='center', fontsize=42,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_LIGHT,
                    edgecolor=COLOR_ACCENT, linewidth=3))

    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(0.2, 1.0)
    ax.axis('off')
    ax.set_title('Beam Search Example: "The cat..." (Width=3)', fontsize=40, weight='bold')

    plt.tight_layout()
    # Add QuantLet attribution
    ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_Beam_Example_Tree',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#888888',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))
    plt.savefig('./beam_example_tree_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: beam_example_tree_bsc.pdf")

if __name__ == "__main__":
    generate_beam_example_tree()
    print(f"Generated beam_example_tree_bsc.pdf")
