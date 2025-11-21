"""
Task to method decision tree

Generated chart: task_method_decision_tree_bsc.pdf
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


def generate_task_method_decision_tree():
    """Flowchart from task characteristics to recommended method"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Decision tree structure
    ax.text(0.5, 0.95, 'START: What kind of task?', ha='center', fontsize=40, weight='bold',
           bbox=dict(boxstyle='round', facecolor=COLOR_ACCENT, edgecolor=COLOR_MAIN, linewidth=3))

    # First split: Deterministic vs Creative
    ax.text(0.25, 0.82, 'Need exact/factual?', ha='center', fontsize=42, weight='bold')
    ax.text(0.75, 0.82, 'Need creative/diverse?', ha='center', fontsize=42, weight='bold')

    # Left branch: Deterministic tasks
    ax.text(0.25, 0.68, 'Translation\nQ&A\nSummarization', ha='center', fontsize=42,
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT))
    ax.text(0.25, 0.55, '→ BEAM SEARCH\n(width=3-5)', ha='center', fontsize=42, weight='bold',
           color='white', bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, linewidth=3))
    ax.text(0.25, 0.42, 'Or Temp=0.3-0.5\nfor consistency', ha='center', fontsize=42, style='italic')

    # Right branch: Creative tasks
    ax.text(0.75, 0.68, 'Creative Writing\nDialogue\nStorytelling', ha='center', fontsize=42,
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT))

    # Second split: With/without repetition risk
    ax.text(0.65, 0.52, 'Short responses?\n(low repetition risk)', ha='center', fontsize=42)
    ax.text(0.65, 0.40, '→ NUCLEUS\n(p=0.9-0.95)', ha='center', fontsize=42, weight='bold',
           color='white', bbox=dict(boxstyle='round', facecolor=COLOR_GREEN))

    ax.text(0.85, 0.52, 'Long generation?\n(repetition risk)', ha='center', fontsize=42)
    ax.text(0.85, 0.40, '→ CONTRASTIVE\n(α=0.5-0.7)', ha='center', fontsize=42, weight='bold',
           color='white', bbox=dict(boxstyle='round', facecolor=COLOR_GREEN))

    # Special case: Code generation
    ax.text(0.5, 0.25, 'Special: Code Generation', ha='center', fontsize=42, weight='bold')
    ax.text(0.5, 0.18, '→ Greedy or Beam (correctness critical)', ha='center', fontsize=42)
    ax.text(0.5, 0.12, '→ Then verify syntax/semantics', ha='center', fontsize=42, style='italic', color=COLOR_GRAY)

    # Draw connecting lines
    ax.plot([0.5, 0.25], [0.93, 0.84], 'k-', alpha=0.3, linewidth=3)
    ax.plot([0.5, 0.75], [0.93, 0.84], 'k-', alpha=0.3, linewidth=3)
    ax.plot([0.25, 0.25], [0.80, 0.70], 'k-', alpha=0.3, linewidth=3)
    ax.plot([0.75, 0.65], [0.80, 0.70], 'k-', alpha=0.3, linewidth=3)
    ax.plot([0.75, 0.65], [0.80, 0.54], 'k-', alpha=0.3, linewidth=3)
    ax.plot([0.75, 0.85], [0.80, 0.54], 'k-', alpha=0.3, linewidth=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Choosing the Right Decoding Method', fontsize=42, weight='bold')

    plt.tight_layout()
    # Add QuantLet attribution
    ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_Task_Method_Decision_Tree',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#888888',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))
    plt.savefig('./task_method_decision_tree_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: task_method_decision_tree_bsc.pdf")

if __name__ == "__main__":
    generate_task_method_decision_tree()
    print(f"Generated task_method_decision_tree_bsc.pdf")
