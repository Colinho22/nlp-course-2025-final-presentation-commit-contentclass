"""
Nucleus sampling process

Generated chart: nucleus_process_bsc.pdf
Part of Week 9: NLP Decoding Strategies
"""

# QuantLet Metadata
QUANTLET_URL = "https://quantlet.com/NLPDecoding_Nucleusprocess"


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


def generate_nucleus_process():
    """Cumulative probability threshold visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Probability distribution
    tokens = ['cat', 'dog', 'bird', 'fish', 'mouse', 'rat', 'frog', 'deer']
    probs = np.array([0.40, 0.20, 0.15, 0.10, 0.06, 0.04, 0.03, 0.02])
    cumulative = np.cumsum(probs)

    # p=0.85 threshold
    p_threshold = 0.85
    nucleus_size = np.searchsorted(cumulative, p_threshold) + 1

    x_pos = np.arange(len(tokens))

    # Bar chart with cumulative line
    colors = [COLOR_GREEN if cumulative[i] <= p_threshold else COLOR_RED
             for i in range(len(tokens))]
    bars = ax.bar(x_pos, probs, color=colors, alpha=0.7, edgecolor=COLOR_MAIN, linewidth=3)

    # Cumulative line
    ax2 = ax.twinx()
    ax2.plot(x_pos, cumulative, color=COLOR_ACCENT, linewidth=3, marker='o',
            markersize=18, label='Cumulative Probability')
    ax2.axhline(p_threshold, color=COLOR_ACCENT, linestyle='--', linewidth=3,
               label=f'p={p_threshold} threshold')

    # Nucleus annotation
    ax.text(nucleus_size/2 - 0.5, 0.45, f'Nucleus\n({nucleus_size} tokens)',
           ha='center', fontsize=38, weight='bold', color=COLOR_GREEN,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLOR_GREEN, linewidth=3))

    ax.set_xlabel('Tokens (ranked by probability)', fontsize=38, weight='bold')
    ax.set_ylabel('Individual Probability', fontsize=38, weight='bold', color=COLOR_MAIN)
    ax2.set_ylabel('Cumulative Probability', fontsize=38, weight='bold', color=COLOR_ACCENT)
    ax.set_title('Nucleus (Top-p) Sampling: Dynamic Vocabulary', fontsize=40, weight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(tokens, rotation=0)
    ax2.legend(loc='center right', fontsize=42)

    set_minimalist_style(ax)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color(COLOR_ACCENT)
    ax2.tick_params(colors=COLOR_ACCENT, which='both')

    plt.tight_layout()
    # Add QuantLet attribution
    ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_Nucleus_Process',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#888888',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))
    plt.savefig('./nucleus_process_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: nucleus_process_bsc.pdf")

if __name__ == "__main__":
    generate_nucleus_process()
    print(f"Generated nucleus_process_bsc.pdf")
