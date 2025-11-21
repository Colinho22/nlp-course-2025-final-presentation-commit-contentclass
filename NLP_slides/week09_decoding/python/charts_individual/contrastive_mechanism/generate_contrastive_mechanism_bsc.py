"""
Contrastive search mechanism

Generated chart: contrastive_mechanism_bsc.pdf
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


def generate_contrastive_mechanism():
    """How contrastive search balances probability and diversity"""
    fig, ax = plt.subplots(figsize=(11, 7))

    ax.text(0.5, 0.95, 'Contrastive Search: How It Works', ha='center',
           fontsize=42, weight='bold')

    # Step 1: Model probability
    ax.text(0.05, 0.82, 'Step 1: Get top-k candidates by probability',
           fontsize=42, weight='bold', color=COLOR_ACCENT)
    candidates = [('city', 0.45), ('town', 0.18), ('area', 0.15), ('place', 0.12)]
    y = 0.75
    for token, prob in candidates:
        ax.text(0.08, y, f'• {token}: {prob:.2f}', fontsize=42)
        y -= 0.04

    # Step 2: Context similarity
    ax.text(0.05, 0.55, 'Step 2: Compute similarity to recent context',
           fontsize=42, weight='bold', color=COLOR_ACCENT)
    ax.text(0.08, 0.48, 'Context: "...the city has..."', fontsize=42, style='italic')
    similarities = [('city', 0.92), ('town', 0.75), ('area', 0.65), ('place', 0.60)]
    y = 0.42
    for token, sim in similarities:
        ax.text(0.08, y, f'• {token}: {sim:.2f} (cosine similarity)', fontsize=42)
        y -= 0.04

    # Step 3: Penalty
    ax.text(0.55, 0.82, 'Step 3: Apply diversity penalty (α=0.6)',
           fontsize=42, weight='bold', color=COLOR_ACCENT)
    ax.text(0.57, 0.75, 'score = (1-α)×P(token) - α×similarity', fontsize=42,
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT))

    scores = []
    y = 0.67
    alpha = 0.6
    for i, ((token, prob), (_, sim)) in enumerate(zip(candidates, similarities)):
        score = (1 - alpha) * prob - alpha * sim
        scores.append((token, score))
        color = COLOR_GREEN if i == 1 else COLOR_MAIN  # 'town' wins
        ax.text(0.57, y, f'{token}: 0.4×{prob:.2f} - 0.6×{sim:.2f} = {score:.3f}',
               fontsize=42, weight='bold' if i == 1 else 'normal', color=color)
        y -= 0.04

    # Result
    ax.text(0.55, 0.48, 'Winner: "town" (high prob, lower similarity)',
           fontsize=42, weight='bold', color=COLOR_GREEN,
           bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, alpha=0.2,
                    edgecolor=COLOR_GREEN, linewidth=3))

    # Formula box
    ax.text(0.5, 0.15, 'Key Insight: Balance probability (coherence) with diversity (novelty)',
           ha='center', fontsize=42, style='italic',
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN, linewidth=1))

    ax.text(0.5, 0.05, 'α=0: Pure greedy | α=0.6: Balanced | α=1.0: Maximum diversity',
           ha='center', fontsize=42, color=COLOR_GRAY)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('./contrastive_mechanism_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: contrastive_mechanism_bsc.pdf")

if __name__ == "__main__":
    generate_contrastive_mechanism()
    print(f"Generated contrastive_mechanism_bsc.pdf")
