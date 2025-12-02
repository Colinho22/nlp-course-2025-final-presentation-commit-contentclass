"""
Top-k numerical example

Generated chart: topk_example_bsc.pdf
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


def generate_topk_example():
    """Worked numerical example with k=3"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.text(0.5, 0.95, 'Top-k Example: k=3', ha='center', fontsize=24, weight='bold')

    # Original distribution
    ax.text(0.05, 0.85, '1. Original Probabilities:', fontsize=24, weight='bold', color=COLOR_ACCENT)
    data1 = [
        ('cat', 0.45),
        ('dog', 0.18),
        ('bird', 0.15),
        ('fish', 0.10),
        ('mouse', 0.08),
        ('...', 0.04)
    ]
    y = 0.78
    for token, prob in data1:
        ax.text(0.08, y, f'{token}: {prob:.2f}', fontsize=24)
        y -= 0.05

    # After filtering
    ax.text(0.40, 0.85, '2. Keep Top-3:', fontsize=24, weight='bold', color=COLOR_ACCENT)
    ax.text(0.42, 0.78, 'cat: 0.45', fontsize=24, weight='bold', color=COLOR_GREEN)
    ax.text(0.42, 0.73, 'dog: 0.18', fontsize=24, weight='bold', color=COLOR_GREEN)
    ax.text(0.42, 0.68, 'bird: 0.15', fontsize=24, weight='bold', color=COLOR_GREEN)
    ax.text(0.42, 0.63, '(discard rest)', fontsize=24, style='italic', color=COLOR_GRAY)

    # Renormalize
    ax.text(0.70, 0.85, '3. Renormalize:', fontsize=24, weight='bold', color=COLOR_ACCENT)
    ax.text(0.72, 0.78, 'Sum = 0.45 + 0.18 + 0.15 = 0.78', fontsize=24)
    ax.text(0.72, 0.72, 'cat: 0.45/0.78 = 0.58', fontsize=24, color=COLOR_MAIN)
    ax.text(0.72, 0.67, 'dog: 0.18/0.78 = 0.23', fontsize=24, color=COLOR_MAIN)
    ax.text(0.72, 0.62, 'bird: 0.15/0.78 = 0.19', fontsize=24, color=COLOR_MAIN)

    # Arrow showing process
    ax.annotate('', xy=(0.38, 0.75), xytext=(0.35, 0.75),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_ACCENT))
    ax.annotate('', xy=(0.68, 0.75), xytext=(0.65, 0.75),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_ACCENT))

    # Result
    ax.text(0.5, 0.25, 'Result: Sample from {cat: 58%, dog: 23%, bird: 19%}',
           ha='center', fontsize=24, weight='bold',
           bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, alpha=0.2,
                    edgecolor=COLOR_GREEN, linewidth=3))

    ax.text(0.5, 0.10, 'Prevents sampling from long tail ("mouse" eliminated)',
           ha='center', fontsize=24, style='italic', color=COLOR_GRAY)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('./topk_example_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: topk_example_bsc.pdf")

if __name__ == "__main__":
    generate_topk_example()
    print(f"Generated topk_example_bsc.pdf")
