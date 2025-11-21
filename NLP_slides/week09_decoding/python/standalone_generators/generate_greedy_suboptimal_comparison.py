"""
Generate Greedy Suboptimal Comparison Chart
Shows side-by-side paths where greedy misses a better sequence.
Created: November 19, 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# BSc Discovery color scheme
COLOR_MAIN = '#404040'       # Main elements (dark gray)
COLOR_ACCENT = '#3333B2'     # Key concepts (purple)
COLOR_GRAY = '#B4B4B4'       # Secondary elements
COLOR_LIGHT = '#F0F0F0'      # Backgrounds
COLOR_GREEN = '#2CA02C'      # Success/positive
COLOR_RED = '#D62728'        # Error/negative
COLOR_ORANGE = '#FF7F0E'     # Warning/medium
COLOR_BLUE = '#0066CC'       # Information

def generate_greedy_suboptimal_comparison():
    """Create simplified side-by-side comparison showing greedy missing better path.
    Simplified: Removed KEY INSIGHT box, probability explanations, and elaborate context.
    Focus: Two paths + probabilities + one simple takeaway message."""

    fig, ax = plt.subplots(figsize=(18, 10))

    # Title
    ax.text(0.5, 0.96, 'Two Paths: Greedy Choice vs. Better Alternative',
            fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes,
            color=COLOR_MAIN)

    # LEFT SIDE: Greedy Path (what it chooses)
    left_x = 0.25

    # Left header box
    greedy_box = FancyBboxPatch((0.05, 0.75), 0.4, 0.12,
                                boxstyle="round,pad=0.02",
                                transform=ax.transAxes,
                                facecolor=COLOR_GREEN, alpha=0.1,
                                edgecolor=COLOR_GREEN, linewidth=2)
    ax.add_patch(greedy_box)

    ax.text(left_x, 0.84, 'GREEDY CHOOSES', fontsize=24, fontweight='bold',
           ha='center', transform=ax.transAxes, color=COLOR_GREEN)

    ax.text(left_x, 0.79, '"The cat sat on the"', fontsize=18,
           ha='center', transform=ax.transAxes, color=COLOR_MAIN,
           style='italic')

    # Greedy path steps
    steps_y = 0.65
    step_height = 0.14

    greedy_steps = [
        ('Input:', '"The cat"', None, None),
        ('Step 1:', 'sat', 0.71, 'HIGH'),
        ('Step 2:', 'on', 0.15, 'LOW'),  # Low probability here!
        ('Step 3:', 'the', 0.72, 'HIGH'),
    ]

    for i, (label, word, prob, rating) in enumerate(greedy_steps):
        y_pos = steps_y - i * step_height

        # Step label
        ax.text(0.08, y_pos, label, fontsize=18, fontweight='bold',
               transform=ax.transAxes, color=COLOR_MAIN)

        # Word box
        if prob:
            word_box = FancyBboxPatch((0.13, y_pos - 0.025), 0.09, 0.05,
                                      boxstyle="round,pad=0.01",
                                      transform=ax.transAxes,
                                      facecolor=COLOR_LIGHT,
                                      edgecolor=COLOR_GREEN, linewidth=2)
            ax.add_patch(word_box)

        # Word text
        ax.text(0.175, y_pos, f'"{word}"', fontsize=18,
               ha='center', transform=ax.transAxes,
               color=COLOR_MAIN, style='italic')

        # Probability
        if prob:
            # Color based on probability level for greedy path
            if i == 2:  # Step 2 with low probability
                prob_color = COLOR_ORANGE
            else:
                prob_color = COLOR_GREEN

            ax.text(0.26, y_pos, f'P={prob:.2f}', fontsize=18,
                   ha='center', transform=ax.transAxes, color=prob_color,
                   fontweight='bold')

            # Text indicator instead of Unicode checkmark
            ax.text(0.35, y_pos, 'CHOSEN', fontsize=14,
                   ha='center', transform=ax.transAxes, color=COLOR_GREEN,
                   fontweight='bold')

    # Greedy total probability
    total_greedy = 0.71 * 0.15 * 0.72  # Updated with new step 2 probability
    ax.text(left_x, 0.17, f'Total: {total_greedy:.3f}', fontsize=18,
           ha='center', transform=ax.transAxes, color=COLOR_GREEN,
           fontweight='bold')

    # Greedy result box
    result_box1 = FancyBboxPatch((0.05, 0.06), 0.4, 0.09,
                                 boxstyle="round,pad=0.02",
                                 transform=ax.transAxes,
                                 facecolor=COLOR_LIGHT,
                                 edgecolor=COLOR_GRAY, linewidth=2)
    ax.add_patch(result_box1)

    ax.text(left_x, 0.105, 'Result: Generic, predictable', fontsize=18,
           ha='center', transform=ax.transAxes, color=COLOR_MAIN)

    # RIGHT SIDE: Better Path (what greedy misses)
    right_x = 0.75

    # Right header box
    missed_box = FancyBboxPatch((0.55, 0.75), 0.4, 0.12,
                                boxstyle="round,pad=0.02",
                                transform=ax.transAxes,
                                facecolor=COLOR_ORANGE, alpha=0.1,
                                edgecolor=COLOR_ORANGE, linewidth=2)
    ax.add_patch(missed_box)

    ax.text(right_x, 0.84, 'GREEDY MISSES', fontsize=24, fontweight='bold',
           ha='center', transform=ax.transAxes, color=COLOR_ORANGE)

    ax.text(right_x, 0.79, '"The cat spotted something"', fontsize=18,
           ha='center', transform=ax.transAxes, color=COLOR_MAIN,
           style='italic')

    # Better path steps
    better_steps = [
        ('Input:', '"The cat"', None, None),
        ('Step 1:', 'spotted', 0.08, 'LOW'),
        ('Step 2:', 'something', 0.85, 'HIGH'),
        ('Step 3:', 'moving', 0.91, 'HIGH'),
    ]

    for i, (label, word, prob, rating) in enumerate(better_steps):
        y_pos = steps_y - i * step_height

        # Step label
        ax.text(0.58, y_pos, label, fontsize=18, fontweight='bold',
               transform=ax.transAxes, color=COLOR_MAIN)

        # Word box
        if prob:
            # Special coloring for first step (low probability)
            if i == 1:
                edge_color = COLOR_RED
            else:
                edge_color = COLOR_ORANGE

            word_box = FancyBboxPatch((0.63, y_pos - 0.025), 0.13, 0.05,
                                      boxstyle="round,pad=0.01",
                                      transform=ax.transAxes,
                                      facecolor=COLOR_LIGHT,
                                      edgecolor=edge_color, linewidth=2)
            ax.add_patch(word_box)

        # Word text
        word_x = 0.695
        ax.text(word_x, y_pos, f'"{word}"', fontsize=18,
               ha='center', transform=ax.transAxes,
               color=COLOR_MAIN, style='italic')

        # Probability
        if prob:
            # Color based on probability value
            prob_color = COLOR_RED if prob < 0.1 else COLOR_ORANGE
            ax.text(0.79, y_pos, f'P={prob:.2f}', fontsize=18,
                   ha='center', transform=ax.transAxes, color=prob_color,
                   fontweight='bold')

            # Text indicator instead of Unicode X mark
            if i == 1:
                ax.text(0.88, y_pos, 'IGNORED', fontsize=14,
                       ha='center', transform=ax.transAxes, color=COLOR_RED,
                       fontweight='bold')

    # Better path total probability (now lower than greedy!)
    total_better = 0.08 * 0.85 * 0.91
    ax.text(right_x, 0.17, f'Total: {total_better:.3f}', fontsize=18,
           ha='center', transform=ax.transAxes, color=COLOR_ORANGE,
           fontweight='bold')

    # Better result box
    result_box2 = FancyBboxPatch((0.55, 0.06), 0.4, 0.09,
                                 boxstyle="round,pad=0.02",
                                 transform=ax.transAxes,
                                 facecolor=COLOR_LIGHT,
                                 edgecolor=COLOR_ORANGE, linewidth=2)
    ax.add_patch(result_box2)

    ax.text(right_x, 0.105, 'Result: Engaging, creates tension', fontsize=18,
           ha='center', transform=ax.transAxes, color=COLOR_MAIN)

    # BOTTOM: Simple key insight
    ax.text(0.5, 0.02, 'Greedy chooses the safer path (P=0.076) but misses better narratives (P=0.062)',
           fontsize=20, ha='center', transform=ax.transAxes,
           color=COLOR_ACCENT, fontweight='bold', style='italic')

    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Save figure
    plt.tight_layout()
    plt.savefig('./greedy_suboptimal_comparison_bsc.pdf',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("Generated greedy_suboptimal_comparison_bsc.pdf")

if __name__ == "__main__":
    generate_greedy_suboptimal_comparison()