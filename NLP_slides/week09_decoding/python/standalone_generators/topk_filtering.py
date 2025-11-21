"""
Top-k filtering process

Generated chart: topk_filtering_bsc.pdf
Part of Week 9: NLP Decoding Strategies
"""

# QuantLet Metadata
QUANTLET_URL = "https://quantlet.com/NLPDecoding_Topkfiltering"


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


def generate_topk_filtering():
    """Top-k selection process visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create probability distribution
    tokens = [f'tok{i}' for i in range(10)]
    probs = np.array([0.30, 0.18, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.01])

    # k=5 cutoff
    k = 5

    # Draw bars individually with different alphas
    for i, (token, prob) in enumerate(zip(tokens, probs)):
        color = COLOR_GREEN if i < k else COLOR_RED
        alpha = 0.8 if i < k else 0.3
        ax.bar(i, prob, color=color, alpha=alpha, edgecolor=COLOR_MAIN, linewidth=3)

    # Add line showing cutoff
    ax.axvline(k - 0.5, color=COLOR_ACCENT, linestyle='--', linewidth=3, label=f'k={k} cutoff')

    # Add annotations
    ax.text(2, 0.35, f'Top-{k}\nSample from these', ha='center',
           fontsize=38, weight='bold', color=COLOR_GREEN,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLOR_GREEN, linewidth=3))

    ax.text(7, 0.08, 'Ignored\n(too unlikely)', ha='center',
           fontsize=42, style='italic', color=COLOR_RED)

    # Labels
    ax.set_xlabel('Tokens (ranked by probability)', fontsize=38, weight='bold')
    ax.set_ylabel('Probability', fontsize=38, weight='bold')
    ax.set_title('Top-k Filtering: Fixed Vocabulary Size', fontsize=40, weight='bold')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.legend(loc='upper right', fontsize=42)

    set_minimalist_style(ax)

    plt.tight_layout()
    # Add QuantLet attribution
    ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_Topk_Filtering',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#888888',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))
    plt.savefig('./topk_filtering_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: topk_filtering_bsc.pdf")

if __name__ == "__main__":
    generate_topk_filtering()
    print(f"Generated topk_filtering_bsc.pdf")
