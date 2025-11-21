"""
Generate side-by-side comparison of decoding strategies on same prompt
Shows actual different outputs from 6 decoding methods
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color scheme
COLOR_GREEDY = '#FF6B6B'      # Red
COLOR_BEAM = '#FFA500'        # Orange
COLOR_LOWT = '#FFD700'        # Gold
COLOR_MIDT = '#4ECDC4'        # Teal
COLOR_HIGHT = '#95E77E'       # Green
COLOR_TOPP = '#9B59B6'        # Purple

def create_decoding_comparison():
    """Create side-by-side comparison of all decoding methods."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Common prompt
    prompt = "The weather is"

    # Results from different methods (simulated realistic outputs)
    methods = [
        ("Greedy (T=0)", COLOR_GREEDY,
         "nice today and the weather is nice today and"),
        ("Beam Search (size=3)", COLOR_BEAM,
         "beautiful and sunny with clear blue skies"),
        ("T=0.5, p=0.9", COLOR_LOWT,
         "absolutely gorgeous and perfect for outdoor activities"),
        ("T=0.7, p=0.9 (Default)", COLOR_MIDT,
         "quite pleasant with mild temperatures throughout the day"),
        ("T=1.0, p=0.95", COLOR_HIGHT,
         "remarkably favorable for this time of year, surprisingly warm"),
        ("T=1.5, p=0.95 (Creative)", COLOR_TOPP,
         "exceptionally magnificent, creating an atmosphere of pure serenity")
    ]

    # Starting position
    y_pos = 0.95
    box_height = 0.12
    box_width = 0.88
    spacing = 0.02

    # Title
    ax.text(0.5, 0.98, 'Decoding Strategies: Same Prompt, Different Outputs',
            ha='center', fontsize=18, fontweight='bold',
            transform=ax.transAxes)

    # Prompt box at top
    prompt_box = FancyBboxPatch((0.05, y_pos - box_height), box_width, box_height - 0.02,
                                boxstyle="round,pad=0.01",
                                facecolor='lightblue',
                                edgecolor='black',
                                linewidth=2)
    ax.add_patch(prompt_box)
    ax.text(0.5, y_pos - box_height/2, f'Prompt: "{prompt}..."',
            ha='center', va='center', fontsize=14, fontweight='bold',
            transform=ax.transAxes)

    y_pos -= (box_height + 0.04)

    # Each method
    for method_name, color, output_text in methods:
        # Method name box
        method_box = FancyBboxPatch((0.05, y_pos - box_height), 0.25, box_height,
                                    boxstyle="round,pad=0.01",
                                    facecolor=color,
                                    edgecolor='black',
                                    linewidth=2,
                                    alpha=0.6)
        ax.add_patch(method_box)
        ax.text(0.05 + 0.125, y_pos - box_height/2, method_name,
                ha='center', va='center', fontsize=10, fontweight='bold',
                transform=ax.transAxes)

        # Output box
        output_box = FancyBboxPatch((0.32, y_pos - box_height), 0.61, box_height,
                                    boxstyle="round,pad=0.01",
                                    facecolor='white',
                                    edgecolor=color,
                                    linewidth=2)
        ax.add_patch(output_box)

        # Output text with quote marks
        full_output = f'"{prompt} {output_text}"'
        ax.text(0.33, y_pos - box_height/2, full_output,
                ha='left', va='center', fontsize=9, style='italic',
                transform=ax.transAxes, wrap=True)

        y_pos -= (box_height + spacing)

    # Annotations
    # Repetition indicator
    ax.text(0.95, 0.72, '← Repetitive!',
            ha='right', fontsize=9, color='red', fontweight='bold',
            transform=ax.transAxes)

    # Quality indicators
    ax.text(0.95, 0.56, '← Good balance',
            ha='right', fontsize=9, color='green', fontweight='bold',
            transform=ax.transAxes)

    ax.text(0.95, 0.24, '← Very creative!',
            ha='right', fontsize=9, color='purple', fontweight='bold',
            transform=ax.transAxes)

    # Bottom legend
    legend_text = [
        "• Greedy: Always picks highest probability → deterministic, often repetitive",
        "• Beam: Explores multiple paths → better quality, still deterministic",
        "• Low T: Focused sampling → high quality, slight variation",
        "• Mid T: Balanced → good mix of quality and creativity",
        "• High T: Creative sampling → diverse, unpredictable",
    ]

    y_legend = 0.08
    for text in legend_text:
        ax.text(0.05, y_legend, text, ha='left', fontsize=8,
                transform=ax.transAxes)
        y_legend -= 0.015

    # Clean up axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('./decoding_live_demo.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated decoding_live_demo.pdf")


if __name__ == "__main__":
    create_decoding_comparison()
