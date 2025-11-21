"""
Production API settings

Generated chart: production_settings_bsc.pdf
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


def generate_production_settings():
    """Real-world parameter settings from production systems"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Production settings table
    systems = [
        ('GPT-3 API (2024)', 'Nucleus', 'T=0.7, p=1.0', 'Balanced default'),
        ('ChatGPT', 'Nucleus + Temp', 'T=0.8, p=0.95', 'Creative but controlled'),
        ('Google Translate', 'Beam Search', 'width=4', 'Quality critical'),
        ('GitHub Copilot', 'Greedy', 'T=0', 'Code correctness'),
        ('Claude', 'Nucleus', 'T=1.0, p=0.9', 'High quality generation'),
        ('Hugging Face Default', 'Greedy', 'T=1.0', 'Deterministic baseline')
    ]

    table_data = [[s[0], s[1], s[2], s[3]] for s in systems]
    headers = ['System (2024-2025)', 'Method', 'Parameters', 'Goal']

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='left', loc='center',
                    colWidths=[0.25, 0.20, 0.25, 0.30])

    table.auto_set_font_size(False)
    table.set_fontsize(22)
    table.scale(1, 2.8)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor(COLOR_ACCENT)
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=42)

    # Highlight rows
    for i in range(1, len(systems) + 1):
        if 'Nucleus' in systems[i-1][1]:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor(COLOR_GREEN + '15')

    ax.axis('off')
    ax.set_title('Production Decoding Settings (Real Systems 2024-2025)',
                fontsize=42, weight='bold', pad=20)

    plt.tight_layout()
    # Add QuantLet attribution
    ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_Production_Settings',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#888888',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))
    plt.savefig('./production_settings_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: production_settings_bsc.pdf")

if __name__ == "__main__":
    generate_production_settings()
    print(f"Generated production_settings_bsc.pdf")
