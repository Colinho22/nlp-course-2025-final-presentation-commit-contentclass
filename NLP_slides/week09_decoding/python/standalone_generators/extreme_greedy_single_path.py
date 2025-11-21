"""
Extreme case greedy single path

Generated chart: extreme_greedy_single_path_bsc.pdf
Part of Week 9: NLP Decoding Strategies
"""

# QuantLet Metadata
QUANTLET_URL = "https://quantlet.com/NLPDecoding_Extremegreedysinglepath"


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


def generate_extreme_case_1_greedy():
    """
    Extreme Case 1: Greedy decoding (single deterministic path)
    Completely new design with Input→Process→Output flow.
    Bold 2-tier font hierarchy (24pt headers, 18pt content) for maximum readability.
    """
    print("Generating extreme_greedy_single_path_bsc.pdf...")

    fig, ax = plt.subplots(figsize=(16, 8))

    # LEFT: INPUT - Show initial text prompt
    input_x = -5
    input_y = 3.5

    # Draw input box
    input_rect = FancyBboxPatch((input_x - 1.5, input_y - 1.5), 3, 3,
                                boxstyle="round,pad=0.1",
                                facecolor=COLOR_LIGHT,
                                edgecolor=COLOR_ACCENT,
                                linewidth=2, zorder=2)
    ax.add_patch(input_rect)

    ax.text(input_x, input_y + 2, 'INPUT', fontsize=24, fontweight='bold',
           ha='center', color=COLOR_ACCENT)

    ax.text(input_x, input_y + 0.5, 'Prompt:', fontsize=18,
           ha='center', color=COLOR_MAIN, fontweight='bold')
    ax.text(input_x, input_y - 0.2, '"The cat"', fontsize=18,
           ha='center', color=COLOR_MAIN, style='italic')
    ax.text(input_x, input_y - 0.9, 'Task: Generate\nnext 3 words', fontsize=18,
           ha='center', color=COLOR_GRAY)

    # Arrow from INPUT to PROCESS
    ax.annotate('', xy=(-2.5, input_y), xytext=(input_x + 1.5, input_y),
               arrowprops=dict(arrowstyle='->', lw=3, color=COLOR_ACCENT))

    # CENTER: PROCESS - Show greedy selection at each step
    process_x = 0

    ax.text(process_x, 6, 'PROCESS: Greedy Selection', fontsize=24, fontweight='bold',
           ha='center', color=COLOR_ACCENT)

    # Show step-by-step greedy process
    steps = [
        ('Step 1', 'sat', 0.71, ['sat: 0.71', 'is: 0.15', 'ran: 0.08', '...']),
        ('Step 2', 'on', 0.69, ['on: 0.69', 'near: 0.12', 'by: 0.10', '...']),
        ('Step 3', 'the', 0.72, ['the: 0.72', 'a: 0.18', 'my: 0.05', '...'])
    ]

    for i, (step_label, word, prob, options) in enumerate(steps):
        y_pos = 4.5 - i * 1.8

        # Draw options box
        opts_rect = FancyBboxPatch((process_x - 2, y_pos - 0.7), 1.8, 1.4,
                                   boxstyle="round,pad=0.05",
                                   facecolor='white',
                                   edgecolor=COLOR_GRAY,
                                   linewidth=1, alpha=0.5)
        ax.add_patch(opts_rect)

        # Show top options
        ax.text(process_x - 1.1, y_pos + 0.3, step_label, fontsize=18,
               ha='center', color=COLOR_GRAY, fontweight='bold')
        for j, opt in enumerate(options[:3]):
            y_opt = y_pos - 0.1 - j * 0.2
            color = COLOR_GREEN if j == 0 else COLOR_GRAY
            weight = 'bold' if j == 0 else 'normal'
            ax.text(process_x - 1.1, y_opt, opt, fontsize=18,
                   ha='center', color=color, fontweight=weight)

        # Arrow showing selection
        ax.annotate('argmax', xy=(process_x + 0.3, y_pos), xytext=(process_x - 0.2, y_pos),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_GREEN),
                   fontsize=18, ha='center', color=COLOR_GREEN)

        # Selected word box
        word_rect = FancyBboxPatch((process_x + 0.3, y_pos - 0.4), 1.5, 0.8,
                                   boxstyle="round,pad=0.05",
                                   facecolor=COLOR_GREEN,
                                   edgecolor=COLOR_ACCENT,
                                   linewidth=2, alpha=0.8)
        ax.add_patch(word_rect)

        ax.text(process_x + 1.05, y_pos, f'"{word}"', fontsize=18,
               ha='center', va='center', color='white', fontweight='bold')

        # Connect to next step
        if i < len(steps) - 1:
            ax.plot([process_x + 1.05, process_x - 1.1],
                   [y_pos - 0.5, y_pos - 1.3],
                   '--', color=COLOR_GRAY, linewidth=1.5, alpha=0.5)

    # Arrow from PROCESS to OUTPUT
    ax.annotate('', xy=(3.5, input_y), xytext=(process_x + 2, input_y),
               arrowprops=dict(arrowstyle='->', lw=3, color=COLOR_ACCENT))

    # RIGHT: OUTPUT - Show final result and statistics
    output_x = 5.5

    ax.text(output_x, input_y + 2, 'OUTPUT', fontsize=24, fontweight='bold',
           ha='center', color=COLOR_ACCENT)

    # Generated text box
    output_rect = FancyBboxPatch((output_x - 1.8, input_y + 0.3), 3.6, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor=COLOR_GREEN,
                                 edgecolor=COLOR_ACCENT,
                                 linewidth=2, alpha=0.1)
    ax.add_patch(output_rect)

    ax.text(output_x, input_y + 0.9, '"The cat sat on the"', fontsize=18,
           ha='center', color=COLOR_MAIN, fontweight='bold')

    # Statistics box
    stats_rect = FancyBboxPatch((output_x - 1.8, input_y - 2.5), 3.6, 2,
                                boxstyle="round,pad=0.1",
                                facecolor=COLOR_LIGHT,
                                edgecolor=COLOR_RED,
                                linewidth=2)
    ax.add_patch(stats_rect)

    ax.text(output_x, input_y - 0.8, 'Statistics:', fontsize=18,
           ha='center', color=COLOR_RED, fontweight='bold')

    stats_lines = [
        'Paths explored: 1',
        'Paths possible: 100³',
        '= 1,000,000',
        'Coverage: 0.0001%'
    ]

    for i, line in enumerate(stats_lines):
        ax.text(output_x, input_y - 1.4 - i*0.3, line, fontsize=18,
               ha='center', color=COLOR_MAIN)

    # Add arrow pointing to stats
    ax.annotate('', xy=(4.5, 3.5), xytext=(2, 3),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_RED, alpha=0.5))

    # Removed duplicate title and subtitle as requested

    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    plt.tight_layout()
    # Add QuantLet attribution
    ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_Extreme_Greedy_Single_Path',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, color='#888888',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.7, linewidth=0.5))
    plt.savefig('./extreme_greedy_single_path_bsc.pdf',
               dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] extreme_greedy_single_path_bsc.pdf created (reordered left-to-right)")

if __name__ == "__main__":
    generate_extreme_case_1_greedy()
    print(f"Generated extreme_greedy_single_path_bsc.pdf")
