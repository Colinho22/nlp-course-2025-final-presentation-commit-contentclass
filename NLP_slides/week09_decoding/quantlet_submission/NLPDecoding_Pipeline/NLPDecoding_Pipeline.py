"""
Generate pipeline visualization showing Weeks 1-8 all produce probability distributions,
and Week 9 is about choosing from those distributions.

This is the "big picture" chart showing how all NLP methods converge to the same
output format (probability distribution over vocabulary), and decoding is the final step.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def create_weeks_pipeline_chart():
    """Create big picture pipeline showing Weeks 1-8 → Week 9."""
    fig, ax = plt.subplots(figsize=(16, 10))

    # Color scheme
    COLOR_WEEK = '#4A90E2'      # Blue for week boxes
    COLOR_MODEL = '#E74C3C'     # Red for model output
    COLOR_PROB = '#F39C12'      # Orange for probabilities
    COLOR_DECODE = '#27AE60'    # Green for decoding
    COLOR_LIGHT = '#ECF0F1'     # Light gray for background

    # Title
    ax.text(0.5, 0.95, 'The Big Picture: From Models to Text',
            ha='center', fontsize=20, fontweight='bold',
            transform=ax.transAxes)

    # Section 1: Weeks 1-8 Timeline (Top)
    y_top = 0.85
    week_labels = [
        'Week 1\nN-grams',
        'Week 2\nNeural LM',
        'Week 3\nRNN/LSTM',
        'Week 4\nSeq2seq',
        'Week 5\nTransformers',
        'Week 6\nBERT/GPT',
        'Week 7\nAdvanced',
        'Week 8\nTokenization'
    ]

    week_x_positions = np.linspace(0.08, 0.92, len(week_labels))

    for i, (x, label) in enumerate(zip(week_x_positions, week_labels)):
        # Week box
        box = FancyBboxPatch((x - 0.045, y_top - 0.04), 0.09, 0.06,
                            boxstyle="round,pad=0.005",
                            facecolor=COLOR_WEEK,
                            edgecolor='black',
                            linewidth=2,
                            alpha=0.8,
                            transform=ax.transAxes)
        ax.add_patch(box)

        ax.text(x, y_top, label, ha='center', va='center', fontsize=8,
                fontweight='bold', color='white', transform=ax.transAxes)

    # Arrow connecting weeks
    arrow = FancyArrowPatch((0.08, y_top - 0.055), (0.92, y_top - 0.055),
                           arrowstyle='->', mutation_scale=30,
                           linewidth=3, color='gray', alpha=0.5,
                           transform=ax.transAxes)
    ax.add_patch(arrow)

    # Section 2: Convergence - ALL produce probabilities
    y_converge = 0.65

    # Text box: "ALL methods output the same thing"
    convergence_box = FancyBboxPatch((0.15, y_converge - 0.03), 0.7, 0.05,
                                    boxstyle="round,pad=0.01",
                                    facecolor=COLOR_MODEL,
                                    edgecolor='black',
                                    linewidth=3,
                                    transform=ax.transAxes)
    ax.add_patch(convergence_box)

    ax.text(0.5, y_converge,
            'ALL methods output: P(word | context) for ENTIRE vocabulary',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='white', transform=ax.transAxes)

    # Arrows from weeks to convergence box
    for x in [0.08, 0.22, 0.36, 0.5, 0.64, 0.78, 0.92]:
        arrow_down = FancyArrowPatch((x, y_top - 0.045), (x, y_converge + 0.03),
                                    arrowstyle='->', mutation_scale=20,
                                    linewidth=2, color=COLOR_MODEL,
                                    alpha=0.6, transform=ax.transAxes)
        ax.add_patch(arrow_down)

    # Section 3: Probability Distribution Visualization
    y_prob = 0.42

    # Background box for probability section
    prob_bg = Rectangle((0.1, y_prob - 0.12), 0.8, 0.15,
                        facecolor=COLOR_LIGHT, edgecolor='black',
                        linewidth=2, transform=ax.transAxes)
    ax.add_patch(prob_bg)

    # Example: "The weather is __"
    ax.text(0.15, y_prob + 0.08, 'Example: "The weather is __"',
            ha='left', fontsize=12, fontweight='bold',
            transform=ax.transAxes)

    # Show probability distribution as bars
    words = ['nice', 'beautiful', 'perfect', 'gorgeous', 'awful', 'mild', 'unpredictable', '...']
    probs = [0.60, 0.20, 0.10, 0.05, 0.02, 0.02, 0.01, 0.0]

    # Mini bar chart
    bar_x_start = 0.15
    bar_spacing = 0.08

    for i, (word, prob) in enumerate(zip(words, probs)):
        x_pos = bar_x_start + i * bar_spacing

        if i < len(words) - 1:  # Don't draw bar for "..."
            # Bar
            bar_height = prob * 0.15  # Scale for display
            bar = Rectangle((x_pos - 0.02, y_prob - 0.09), 0.04, bar_height,
                           facecolor=COLOR_PROB, edgecolor='black',
                           linewidth=1, transform=ax.transAxes)
            ax.add_patch(bar)

            # Probability label
            ax.text(x_pos, y_prob - 0.095, f'{prob:.2f}',
                   ha='center', va='top', fontsize=7,
                   transform=ax.transAxes)

        # Word label
        ax.text(x_pos, y_prob - 0.105, word,
               ha='center', va='top', fontsize=8,
               rotation=45, transform=ax.transAxes)

    # Vocabulary size note
    ax.text(0.85, y_prob + 0.05, 'Vocabulary size:\n~50,000 words!',
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3),
            transform=ax.transAxes)

    # Section 4: The Question
    y_question = 0.20

    # Question box with big arrow
    question_arrow = FancyArrowPatch((0.5, y_prob - 0.13), (0.5, y_question + 0.08),
                                    arrowstyle='->', mutation_scale=40,
                                    linewidth=5, color='red',
                                    transform=ax.transAxes)
    ax.add_patch(question_arrow)

    question_box = FancyBboxPatch((0.2, y_question - 0.02), 0.6, 0.07,
                                 boxstyle="round,pad=0.01",
                                 facecolor='yellow',
                                 edgecolor='red',
                                 linewidth=3,
                                 transform=ax.transAxes)
    ax.add_patch(question_box)

    ax.text(0.5, y_question + 0.02,
            'WEEK 9 QUESTION: Which word should we actually pick?',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='red', transform=ax.transAxes)

    # Section 5: Decoding Strategies (Bottom)
    y_decode = 0.08

    strategies = ['Greedy', 'Beam\nSearch', 'Temperature', 'Top-k', 'Top-p', 'Task-\nSpecific']
    strategy_x = np.linspace(0.12, 0.88, len(strategies))

    for x, strategy in zip(strategy_x, strategies):
        # Strategy box
        box = FancyBboxPatch((x - 0.04, y_decode - 0.025), 0.08, 0.045,
                            boxstyle="round,pad=0.005",
                            facecolor=COLOR_DECODE,
                            edgecolor='black',
                            linewidth=2,
                            transform=ax.transAxes)
        ax.add_patch(box)

        ax.text(x, y_decode, strategy, ha='center', va='center', fontsize=8,
                fontweight='bold', color='white', transform=ax.transAxes)

    # Arrow from question to strategies
    arrow_to_decode = FancyArrowPatch((0.5, y_question - 0.025), (0.5, y_decode + 0.025),
                                     arrowstyle='->', mutation_scale=30,
                                     linewidth=3, color=COLOR_DECODE,
                                     transform=ax.transAxes)
    ax.add_patch(arrow_to_decode)

    # Final output
    ax.text(0.5, 0.01, 'Final Output: "The weather is nice"',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLOR_DECODE, alpha=0.3),
            transform=ax.transAxes)

    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('./weeks_pipeline_to_decoding.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated weeks_pipeline_to_decoding.pdf")


if __name__ == "__main__":
    create_weeks_pipeline_chart()
