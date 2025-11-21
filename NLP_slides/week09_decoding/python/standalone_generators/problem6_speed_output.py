"""
Problem 6 speed output

Generated chart: problem6_speed_output_bsc.pdf
Part of Week 9: NLP Decoding Strategies
"""

# QuantLet Metadata
QUANTLET_URL = "https://quantlet.com/NLPDecoding_Problem6Speedoutput"


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


def generate_problem6_speed():
    """Problem 6: Too slow for real-time"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Speed comparison bars
    methods = ['Greedy\n(Baseline)', 'Beam\n(k=5)', 'Beam\n(k=10)', 'Beam\n(k=50)']
    speeds = [10, 50, 100, 500]  # milliseconds
    colors = [COLOR_GREEN, COLOR_ORANGE, COLOR_RED, COLOR_RED]

    bars = ax.barh(range(len(methods)), speeds, color=colors, alpha=0.8)

    # Add speed values
    for i, (method, speed) in enumerate(zip(methods, speeds)):
        ax.text(speed + 20, i, f'{speed}ms', va='center',
                fontsize=FONTSIZE_TEXT, weight='bold')

        # Add latency indicator
        if speed > 100:
            ax.text(speed/2, i, 'TOO SLOW!', ha='center', va='center',
                    fontsize=12, color='white', weight='bold')

    # Add threshold line
    ax.axvline(x=100, color=COLOR_RED, linestyle='--', linewidth=2, alpha=0.7)
    ax.text(100, 3.5, 'Real-time threshold (100ms)',
            fontsize=FONTSIZE_SMALL, color=COLOR_RED, ha='center')

    # Add user experience indicators
    ax.text(600, 3, 'User gives up!', fontsize=FONTSIZE_LABEL,
            color=COLOR_RED, weight='bold', style='italic')
    ax.text(600, 2, '😕 Bad UX', fontsize=FONTSIZE_TEXT, color=COLOR_RED)

    ax.text(600, 0, '😊 Good UX', fontsize=FONTSIZE_TEXT, color=COLOR_GREEN)

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=FONTSIZE_LABEL)
    ax.set_xlabel('Generation Time per Token', fontsize=FONTSIZE_LABEL, weight='bold')
    ax.set_title('Beam Search Makes Typing Assistants Unusable',
                 fontsize=FONTSIZE_TITLE, weight='bold')

    ax.set_xlim(0, 700)
    set_minimalist_style(ax)

    plt.tight_layout()
    # Add QuantLet attribution
    ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_Problem6_Speed_Output',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#888888',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))
    plt.savefig('./problem6_speed_output_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[CREATED] problem6_speed_output_bsc.pdf")

if __name__ == "__main__":
    generate_problem6_speed()
    print(f"Generated problem6_speed_output_bsc.pdf")
