"""
Nucleus cumulative distribution

Generated chart: nucleus_cumulative_bsc.pdf
Part of Week 9: NLP Decoding Strategies
"""

# QuantLet Metadata
QUANTLET_URL = "https://quantlet.com/NLPDecoding_Nucleuscumulative"


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


def generate_nucleus_cumulative():
    """Dynamic cutoff visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Scenario 1: Peaked distribution (nucleus is small)
    probs1 = [0.60, 0.20, 0.10, 0.05, 0.03, 0.02]
    tokens1 = ['the', 'a', 'an', 'this', 'that', 'my']
    cumsum1 = np.cumsum(probs1)
    p_threshold = 0.85
    nucleus1 = np.searchsorted(cumsum1, p_threshold) + 1

    colors1 = [COLOR_GREEN if i < nucleus1 else COLOR_GRAY for i in range(len(probs1))]
    ax1.bar(tokens1, probs1, color=colors1, alpha=0.7, edgecolor=COLOR_MAIN, linewidth=3)
    ax1.set_title('Peaked Distribution\n(Nucleus = 2 tokens)', fontsize=38, weight='bold')
    ax1.set_ylabel('Probability', fontsize=42, weight='bold')
    ax1.text(0.5, 0.55, f'Top 2 tokens\n= {cumsum1[1]:.0%} mass',
            ha='center', fontsize=42, bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, alpha=0.3))
    set_minimalist_style(ax1)

    # Scenario 2: Flat distribution (nucleus is large)
    probs2 = [0.18, 0.16, 0.14, 0.13, 0.12, 0.10, 0.09, 0.08]
    tokens2 = ['cat', 'dog', 'bird', 'fish', 'frog', 'deer', 'bear', 'wolf']
    cumsum2 = np.cumsum(probs2)
    nucleus2 = np.searchsorted(cumsum2, p_threshold) + 1

    colors2 = [COLOR_GREEN if i < nucleus2 else COLOR_GRAY for i in range(len(probs2))]
    ax2.bar(tokens2, probs2, color=colors2, alpha=0.7, edgecolor=COLOR_MAIN, linewidth=3)
    ax2.set_title('Flat Distribution\n(Nucleus = 6 tokens)', fontsize=38, weight='bold')
    ax2.set_ylabel('Probability', fontsize=42, weight='bold')
    ax2.text(2.5, 0.17, f'Top 6 tokens\n= {cumsum2[5]:.0%} mass',
            ha='center', fontsize=42, bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, alpha=0.3))
    set_minimalist_style(ax2)

    fig.suptitle(f'Nucleus Adapts to Distribution Shape (p={p_threshold})',
                fontsize=40, weight='bold', y=1.02)

    plt.tight_layout()
    # Add QuantLet attribution
    ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_Nucleus_Cumulative',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#888888',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))
    plt.savefig('./nucleus_cumulative_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: nucleus_cumulative_bsc.pdf")

if __name__ == "__main__":
    generate_nucleus_cumulative()
    print(f"Generated nucleus_cumulative_bsc.pdf")
