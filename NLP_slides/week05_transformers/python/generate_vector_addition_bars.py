"""
Generate Vector Addition Visual with Bars
Week 5 Transformers - For Slide 14 (Zero-Jargon Explanation)
Shows embedding + position = combined vector as stacked bars
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# Template colors
COLOR_MAIN = '#333366'
COLOR_BLUE = '#4472C4'
COLOR_PURPLE = '#8B5A9B'
COLOR_GREEN = '#44A044'
COLOR_ORANGE = '#FF7F0E'
COLOR_LIGHT_BG = '#F0F0F0'

# Example vectors
embedding = np.array([0.3, 0.2, 0.5, 0.1])
position = np.array([0.1, 0.0, 0.05, 0.02])
combined = embedding + position

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.5, 'Vector Addition: Embedding + Position = Combined', ha='center', va='center',
        fontsize=14, fontweight='bold', color=COLOR_MAIN)

# SECTION 1: Word Embedding (meaning of "cat")
ax.text(2, 8.5, 'Word Embedding', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_BLUE)
ax.text(2, 8, '(Meaning of "cat")', ha='center', va='center',
        fontsize=9, color=COLOR_BLUE)

# Bar chart for embedding
bar_width = 0.4
x_positions = np.arange(4)
for i, val in enumerate(embedding):
    bar = Rectangle((0.7 + i * 0.8, 5), bar_width, val * 4,
                    facecolor=COLOR_BLUE, edgecolor=COLOR_MAIN, linewidth=2, alpha=0.6)
    ax.add_patch(bar)
    ax.text(0.9 + i * 0.8, 4.5, f'{val:.1f}', ha='center', va='center',
            fontsize=9, fontweight='bold')

# Dimension labels
for i in range(4):
    ax.text(0.9 + i * 0.8, 4, f'D{i+1}', ha='center', va='center',
            fontsize=8)

# PLUS SIGN
ax.text(4.2, 6.5, '+', ha='center', va='center',
        fontsize=24, fontweight='bold', color=COLOR_MAIN)

# SECTION 2: Position Encoding
ax.text(6.5, 8.5, 'Position Encoding', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_ORANGE)
ax.text(6.5, 8, '(Position #2 pattern)', ha='center', va='center',
        fontsize=9, color=COLOR_ORANGE)

# Bar chart for position
for i, val in enumerate(position):
    bar = Rectangle((5.2 + i * 0.8, 5), bar_width, val * 4,
                    facecolor=COLOR_ORANGE, edgecolor=COLOR_MAIN, linewidth=2, alpha=0.6)
    ax.add_patch(bar)
    ax.text(5.4 + i * 0.8, 4.5, f'{val:.2f}', ha='center', va='center',
            fontsize=9, fontweight='bold')

# Dimension labels
for i in range(4):
    ax.text(5.4 + i * 0.8, 4, f'D{i+1}', ha='center', va='center',
            fontsize=8)

# EQUALS SIGN
ax.text(8.5, 6.5, '=', ha='center', va='center',
        fontsize=24, fontweight='bold', color=COLOR_MAIN)

# SECTION 3: Combined Vector
ax.text(11, 8.5, 'Combined Vector', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_GREEN)
ax.text(11, 8, '(Meaning + Position)', ha='center', va='center',
        fontsize=9, color=COLOR_GREEN)

# Stacked bar chart (showing both components)
for i in range(4):
    # Bottom part (embedding - blue)
    bar_embed = Rectangle((9.7 + i * 0.8, 5), bar_width, embedding[i] * 4,
                          facecolor=COLOR_BLUE, edgecolor=COLOR_MAIN,
                          linewidth=1.5, alpha=0.5)
    ax.add_patch(bar_embed)

    # Top part (position - orange)
    bar_pos = Rectangle((9.7 + i * 0.8, 5 + embedding[i] * 4), bar_width, position[i] * 4,
                        facecolor=COLOR_ORANGE, edgecolor=COLOR_MAIN,
                        linewidth=1.5, alpha=0.5)
    ax.add_patch(bar_pos)

    # Total value label
    ax.text(9.9 + i * 0.8, 4.5, f'{combined[i]:.2f}', ha='center', va='center',
            fontsize=9, fontweight='bold', color=COLOR_GREEN)

# Dimension labels
for i in range(4):
    ax.text(9.9 + i * 0.8, 4, f'D{i+1}', ha='center', va='center',
            fontsize=8)

# Legend for stacked bars
legend_y = 2.5
legend_box1 = Rectangle((9, legend_y), 0.4, 0.3, facecolor=COLOR_BLUE, alpha=0.5)
ax.add_patch(legend_box1)
ax.text(9.6, legend_y + 0.15, 'Embedding (meaning)', ha='left', va='center',
        fontsize=9)

legend_box2 = Rectangle((9, legend_y - 0.6), 0.4, 0.3, facecolor=COLOR_ORANGE, alpha=0.5)
ax.add_patch(legend_box2)
ax.text(9.6, legend_y - 0.45, 'Position encoding', ha='left', va='center',
        fontsize=9)

# KEY INSIGHT BOX
insight_box = FancyBboxPatch((1, 0.5), 12, 1.2,
                             boxstyle="round,pad=0.1",
                             facecolor='#E6FFE6', edgecolor=COLOR_GREEN, linewidth=2)
ax.add_patch(insight_box)
ax.text(7, 1.3, 'Key Insight: Both meaning AND position are now in the same vector!', ha='center', va='center',
        fontsize=11, fontweight='bold', color=COLOR_GREEN)
ax.text(7, 0.8, 'The model sees a single combined representation with all information', ha='center', va='center',
        fontsize=9, color=COLOR_GREEN)

plt.tight_layout()

# Save figure
output_path = '../figures/sr_20_vector_addition_bars.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')

plt.close()
