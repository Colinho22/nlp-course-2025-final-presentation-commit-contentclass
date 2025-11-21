"""
Problem 3 balance output

Generated chart: problem3_balance_output_bsc.pdf
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


def generate_problem3_balance():
    """Problem 3: Boring/same outputs"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Show 100 requests with same output
    num_requests = 10
    same_output = "The weather is nice today."

    # Visual grid of identical outputs
    for i in range(num_requests):
        y = i // 2
        x = (i % 2) * 5

        # Box for each output
        rect = patches.FancyBboxPatch((x + 0.2, y * 0.8 - 0.3), 4.5, 0.6,
                                      boxstyle="round,pad=0.02",
                                      facecolor=COLOR_LAVENDER2,
                                      edgecolor=COLOR_GRAY, linewidth=1)
        ax.add_patch(rect)

        # Request number
        ax.text(x + 0.5, y * 0.8, f'#{i+1}:', fontsize=FONTSIZE_SMALL,
                color=COLOR_GRAY, va='center', weight='bold')

        # Same output
        ax.text(x + 2.7, y * 0.8, same_output, fontsize=FONTSIZE_TEXT,
                color=COLOR_MAIN, va='center')

    # Add large "100x" indicator
    ax.text(10.5, 2, '100x', fontsize=36, weight='bold',
            color=COLOR_RED, rotation=15, alpha=0.7)

    # Title
    ax.text(5, 5, 'Zero Creativity: Always Same Output', ha='center',
            fontsize=FONTSIZE_TITLE, weight='bold', color=COLOR_MAIN)

    # Problem indicator
    ax.text(5, -1, f'Asked 100 times → Always: "{same_output}"', ha='center',
            fontsize=FONTSIZE_ANNOTATION, style='italic', color=COLOR_RED)

    ax.set_xlim(0, 11)
    ax.set_ylim(-1.5, 5.5)
    ax.axis('off')

    plt.tight_layout()
    # Add QuantLet attribution
    ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_Problem3_Balance_Output',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#888888',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))
    plt.savefig('./problem3_balance_output_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[CREATED] problem3_balance_output_bsc.pdf")

if __name__ == "__main__":
    generate_problem3_balance()
    print(f"Generated problem3_balance_output_bsc.pdf")
