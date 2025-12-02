"""
Contrastive vs nucleus comparison

Generated chart: contrastive_vs_nucleus_bsc.pdf
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


def generate_contrastive_vs_nucleus():
    """Side-by-side output comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.text(0.5, 0.95, 'Same Prompt, Different Methods', ha='center',
           fontsize=24, weight='bold')

    # Prompt
    ax.text(0.5, 0.88, 'Prompt: "The future of artificial intelligence is"',
           ha='center', fontsize=24, style='italic',
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT, linewidth=3))

    # Nucleus output
    ax.text(0.25, 0.75, 'Nucleus (p=0.9)', ha='center', fontsize=24,
           weight='bold', color=COLOR_ACCENT)
    nucleus_text = ('"...is promising and will transform\n'
                   'many industries. We expect to see\n'
                   'significant advances in healthcare,\n'
                   'education, and research in the\n'
                   'coming years."')
    ax.text(0.25, 0.52, nucleus_text, ha='center', va='center', fontsize=24,
           bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor=COLOR_GREEN, linewidth=3))
    ax.text(0.25, 0.28, '+ Diverse\n+ Creative\n- Some repetition',
           ha='center', fontsize=24, color=COLOR_MAIN)

    # Contrastive output
    ax.text(0.75, 0.75, 'Contrastive (α=0.6)', ha='center', fontsize=24,
           weight='bold', color=COLOR_ACCENT)
    contrastive_text = ('"...is rapidly evolving, bringing\n'
                       'unprecedented opportunities across\n'
                       'sectors ranging from medicine to\n'
                       'climate science, while raising\n'
                       'important ethical questions."')
    ax.text(0.75, 0.52, contrastive_text, ha='center', va='center', fontsize=24,
           bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor=COLOR_GREEN, linewidth=3))
    ax.text(0.75, 0.28, '+ Diverse\n+ Creative\n+ No repetition',
           ha='center', fontsize=24, weight='bold', color=COLOR_GREEN)

    # Comparison
    ax.text(0.5, 0.08, 'Contrastive Search explicitly penalizes copying recent context',
           ha='center', fontsize=24, style='italic', color=COLOR_GREEN)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('./contrastive_vs_nucleus_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: contrastive_vs_nucleus_bsc.pdf")

if __name__ == "__main__":
    generate_contrastive_vs_nucleus()
    print(f"Generated contrastive_vs_nucleus_bsc.pdf")
