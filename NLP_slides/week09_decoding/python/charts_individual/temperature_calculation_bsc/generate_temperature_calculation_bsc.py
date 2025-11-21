"""
Temperature calculation formula visualization

Generated chart: temperature_calculation_bsc.pdf
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


def generate_temperature_calculation():
    """Step-by-step numerical example"""
    fig, ax = plt.subplots(figsize=(11, 7))

    # Setup
    ax.text(0.5, 0.95, 'Temperature Calculation: Step-by-Step',
           ha='center', fontsize=42, weight='bold')

    # Given
    ax.text(0.05, 0.85, 'Given: Logits = [2.0, 1.0, 0.5, 0.2]',
           fontsize=38, weight='bold', color=COLOR_MAIN)
    ax.text(0.05, 0.80, 'Tokens = ["cat", "dog", "bird", "fish"]',
           fontsize=38, color=COLOR_MAIN)

    # Step 1: T=0.5
    ax.text(0.05, 0.70, 'Step 1: Scale by T=0.5 (MORE PEAKED)',
           fontsize=38, weight='bold', color=COLOR_ACCENT)
    ax.text(0.05, 0.65, 'Scaled = [2.0/0.5, 1.0/0.5, 0.5/0.5, 0.2/0.5]', fontsize=42)
    ax.text(0.05, 0.60, '       = [4.0, 2.0, 1.0, 0.4]', fontsize=42, weight='bold', color=COLOR_ACCENT)
    ax.text(0.05, 0.55, 'Softmax → [0.73, 0.18, 0.07, 0.02]', fontsize=42)
    ax.text(0.05, 0.50, '→ 73% on "cat" (VERY FOCUSED)', fontsize=42,
           weight='bold', color=COLOR_GREEN)

    # Step 2: T=1.0
    ax.text(0.55, 0.70, 'Step 2: Scale by T=1.0 (UNCHANGED)',
           fontsize=38, weight='bold', color=COLOR_ACCENT)
    ax.text(0.55, 0.65, 'Scaled = [2.0, 1.0, 0.5, 0.2]', fontsize=42)
    ax.text(0.55, 0.60, 'Softmax → [0.53, 0.19, 0.12, 0.10]', fontsize=42)
    ax.text(0.55, 0.55, '→ 53% on "cat" (BALANCED)', fontsize=42, weight='bold')

    # Step 3: T=2.0
    ax.text(0.05, 0.38, 'Step 3: Scale by T=2.0 (FLATTER)',
           fontsize=38, weight='bold', color=COLOR_ACCENT)
    ax.text(0.05, 0.33, 'Scaled = [1.0, 0.5, 0.25, 0.1]', fontsize=42)
    ax.text(0.05, 0.28, 'Softmax → [0.38, 0.23, 0.17, 0.15]', fontsize=42)
    ax.text(0.05, 0.23, '→ 38% on "cat" (MUCH FLATTER)', fontsize=42,
           weight='bold', color=COLOR_RED)

    # Formula
    ax.text(0.55, 0.35, 'General Formula:', fontsize=38, weight='bold')
    ax.text(0.55, 0.28, r'$p_i = \frac{\exp(\text{logit}_i / T)}{\sum_j \exp(\text{logit}_j / T)}$',
           fontsize=40, bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT, linewidth=3))

    # Insight box
    ax.text(0.3, 0.08, 'Lower T → More confident (peaky)\nHigher T → More random (flat)',
           ha='center', fontsize=42, style='italic',
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('./temperature_calculation_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: temperature_calculation_bsc.pdf")

if __name__ == "__main__":
    generate_temperature_calculation()
    print(f"Generated temperature_calculation_bsc.pdf")
