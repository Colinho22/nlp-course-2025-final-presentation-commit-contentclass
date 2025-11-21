"""
Task recommendations table

Generated chart: task_recommendations_table_bsc.pdf
Part of Week 9: NLP Decoding Strategies
"""

# QuantLet Metadata
QUANTLET_URL = "https://quantlet.com/NLPDecoding_Taskrecommendationstable"


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


def generate_task_recommendations_table():
    """Comprehensive task-method mapping"""
    fig, ax = plt.subplots(figsize=(13, 8))

    # Extended task-method table
    tasks = [
        'Machine Translation',
        'Factual Q&A',
        'Summarization',
        'Code Generation',
        'Creative Writing',
        'Dialogue Systems',
        'Story Generation',
        'Long-form Articles'
    ]

    recommended = [
        'Beam Search',
        'Greedy / Low Temp',
        'Beam Search',
        'Greedy',
        'Nucleus / Contrastive',
        'Nucleus',
        'Contrastive',
        'Contrastive'
    ]

    params = [
        'width=3-5',
        'T=0.1-0.3',
        'width=4',
        'T=0',
        'p=0.9, α=0.6',
        'p=0.85-0.95',
        'α=0.5-0.7',
        'α=0.6, p=0.9'
    ]

    rationale = [
        'Deterministic, quality critical',
        'Single correct answer needed',
        'Balance coverage + conciseness',
        'Syntax errors costly',
        'Diverse but coherent',
        'Natural variation needed',
        'Avoid repetition in long text',
        'Degeneration prevention'
    ]

    # Create table
    table_data = []
    for i in range(len(tasks)):
        table_data.append([tasks[i], recommended[i], params[i], rationale[i]])

    headers = ['Task', 'Recommended Method', 'Parameters', 'Why?']

    # Color code by category
    cell_colors = []
    for i, task in enumerate(tasks):
        if i < 4:  # Deterministic tasks
            cell_colors.append([COLOR_LIGHT, COLOR_RED + '30', 'white', 'white'])
        else:  # Creative tasks
            cell_colors.append([COLOR_LIGHT, COLOR_GREEN + '30', 'white', 'white'])

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='left', loc='center',
                    cellColours=cell_colors,
                    colWidths=[0.22, 0.22, 0.18, 0.38])

    table.auto_set_font_size(False)
    table.set_fontsize(20)
    table.scale(1, 2.2)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor(COLOR_ACCENT)
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=42)

    # Add category labels
    ax.text(0.12, 0.73, 'Deterministic Tasks', fontsize=42, weight='bold',
           color=COLOR_RED, rotation=90, va='center')
    ax.text(0.12, 0.35, 'Creative Tasks', fontsize=42, weight='bold',
           color=COLOR_GREEN, rotation=90, va='center')

    ax.axis('off')
    ax.set_title('Task-Specific Decoding Recommendations (2025)', fontsize=42, weight='bold', pad=20)

    plt.tight_layout()
    # Add QuantLet attribution
    ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_Task_Recommendations_Table',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#888888',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))
    plt.savefig('./task_recommendations_table_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: task_recommendations_table_bsc.pdf")

if __name__ == "__main__":
    generate_task_recommendations_table()
    print(f"Generated task_recommendations_table_bsc.pdf")
