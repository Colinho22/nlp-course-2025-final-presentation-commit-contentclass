"""
Temperature effects on probability distribution

Generated chart: temperature_effects_bsc.pdf
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


def generate_temperature_effects():
    """How temperature reshapes probability distributions"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    # Original logits
    tokens = ['the', 'a', 'on', 'at', 'in']
    logits = np.array([2.0, 1.0, 0.5, 0.3, 0.1])

    # Three temperatures
    temps = [0.5, 1.0, 2.0]
    axes = [ax1, ax2, ax3]
    titles = ['T=0.5\n(Focused)', 'T=1.0\n(Balanced)', 'T=2.0\n(Flat)']

    for temp, ax, title in zip(temps, axes, titles):
        # Apply temperature
        scaled = logits / temp
        probs = np.exp(scaled) / np.sum(np.exp(scaled))

        bars = ax.bar(tokens, probs, color=COLOR_ACCENT, alpha=0.7, edgecolor=COLOR_MAIN, linewidth=3)

        # Highlight max
        max_idx = np.argmax(probs)
        bars[max_idx].set_color(COLOR_GREEN)
        bars[max_idx].set_alpha(0.9)

        ax.set_title(title, fontsize=40, weight='bold')
        ax.set_ylabel('Probability' if ax == ax1 else '', fontsize=42)
        ax.set_ylim(0, max(probs) * 1.2)

        # Add probability labels
        for i, (token, prob) in enumerate(zip(tokens, probs)):
            ax.text(i, prob + 0.02, f'{prob:.2f}', ha='center', fontsize=42, weight='bold')

        set_minimalist_style(ax)

    fig.suptitle('Temperature Effects on Probability Distribution', fontsize=42, weight='bold', y=1.02)
    plt.tight_layout()
    # Add QuantLet attribution
    ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_Temperature_Effects',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#888888',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))
    plt.savefig('./temperature_effects_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: temperature_effects_bsc.pdf")

if __name__ == "__main__":
    generate_temperature_effects()
    print(f"Generated temperature_effects_bsc.pdf")
