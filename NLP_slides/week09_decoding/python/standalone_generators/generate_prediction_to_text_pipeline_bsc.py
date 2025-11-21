"""
Prediction to text pipeline

Generated chart: prediction_to_text_pipeline_bsc.pdf
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


def generate_prediction_pipeline():
    """Journey: How we got here - from model to text"""
    fig, ax = plt.subplots(figsize=(13, 6))

    # Pipeline stages
    stages = ['Input\nText', 'MODEL\n(Transformer)', 'Probability\nDistribution', 'DECODING\n[TODAY]', 'Output\nText']
    x_positions = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Draw boxes
    for i, (stage, x) in enumerate(zip(stages, x_positions)):
        if i == 3:  # TODAY box
            color = COLOR_ACCENT
            text_color = 'white'
            linewidth = 3
        elif i in [0, 4]:  # Input/Output
            color = COLOR_LIGHT
            text_color = COLOR_MAIN
            linewidth = 1.5
        else:
            color = COLOR_GRAY + '30'
            text_color = COLOR_MAIN
            linewidth = 2

        bbox = dict(boxstyle='round,pad=0.8', facecolor=color,
                   edgecolor=COLOR_MAIN, linewidth=linewidth)
        ax.text(x, 0.75, stage, ha='center', va='center', fontsize=38,
               weight='bold', color=text_color, bbox=bbox)

    # Draw arrows
    for i in range(len(x_positions) - 1):
        ax.annotate('', xy=(x_positions[i+1] - 0.05, 0.75),
                   xytext=(x_positions[i] + 0.05, 0.75),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color=COLOR_MAIN))

    # Add examples below each stage
    examples = [
        '"The cat"',
        'Weeks 3-7:\nRNN→Transformer\n→BERT/GPT',
        'P(sat)=0.45\nP(is)=0.30\nP(jumped)=0.25\n...\n50,000 words!',
        '6 Strategies:\nGreedy, Beam\nTemp, Top-k\nNucleus,\nContrastive',
        '"The cat sat\non the mat"'
    ]

    for x, example in zip(x_positions, examples):
        ax.text(x, 0.35, example, ha='center', va='center', fontsize=42,
               style='italic', color=COLOR_GRAY,
               bbox=dict(boxstyle='round', facecolor='white',
                        edgecolor=COLOR_GRAY, linewidth=1, alpha=0.7))

    # Key question
    ax.text(0.5, 0.10, 'Key Question: Model gives 50,000 probabilities - which word do we choose?',
           ha='center', fontsize=38, weight='bold', color=COLOR_ACCENT,
           bbox=dict(boxstyle='round', facecolor=COLOR_ACCENT, alpha=0.15,
                    edgecolor=COLOR_ACCENT, linewidth=3))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('From Prediction to Generation: The Decoding Challenge',
                fontsize=42, weight='bold', pad=20)

    plt.tight_layout()
    # Add QuantLet attribution
    ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_Prediction_To_Text_Pipeline',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#888888',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))
    plt.savefig('./prediction_to_text_pipeline_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: prediction_to_text_pipeline_bsc.pdf")

if __name__ == "__main__":
    generate_prediction_pipeline()
    print(f"Generated prediction_to_text_pipeline_bsc.pdf")
