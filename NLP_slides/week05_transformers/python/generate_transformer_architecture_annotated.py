"""
Generate Annotated Transformer Architecture Diagram
Week 5 Transformers - Shows three key innovations highlighted
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
import numpy as np

# Template colors
COLOR_MAIN = '#333366'
COLOR_BLUE = '#4472C4'  # Positional Encoding
COLOR_PURPLE = '#8B5A9B'  # Self-Attention
COLOR_GREEN = '#44A044'  # Parallelization
COLOR_LIGHT_BG = '#F0F0F0'
COLOR_LAVENDER = '#ADADD8'

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11.5, 'Transformer Architecture: Three Key Innovations',
        ha='center', va='center', fontsize=14, fontweight='bold', color=COLOR_MAIN)

# BOTTOM: Input + Positional Encoding (Innovation 1)
# Input embeddings
input_box = FancyBboxPatch((1, 0.5), 3, 0.8,
                           boxstyle="round,pad=0.05",
                           facecolor=COLOR_LIGHT_BG, edgecolor=COLOR_MAIN, linewidth=2)
ax.add_patch(input_box)
ax.text(2.5, 0.9, 'Input\nEmbeddings', ha='center', va='center',
        fontsize=9, fontweight='bold')

# Plus sign
ax.text(4.5, 0.9, '+', ha='center', va='center', fontsize=16, fontweight='bold')

# Positional encoding (HIGHLIGHTED - Innovation 1)
pos_box = FancyBboxPatch((5, 0.5), 3, 0.8,
                         boxstyle="round,pad=0.05",
                         facecolor=COLOR_BLUE, edgecolor=COLOR_BLUE, linewidth=3, alpha=0.3)
ax.add_patch(pos_box)
pos_border = Rectangle((4.9, 0.4), 3.2, 1.0, fill=False, edgecolor=COLOR_BLUE, linewidth=3)
ax.add_patch(pos_border)
ax.text(6.5, 0.9, 'Positional\nEncoding', ha='center', va='center',
        fontsize=9, fontweight='bold', color=COLOR_BLUE)

# Annotation for Innovation 1
ax.text(8.8, 0.9, '❶', ha='center', va='center',
        fontsize=14, fontweight='bold', color=COLOR_BLUE,
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor=COLOR_BLUE, linewidth=2))

# Arrow up
arrow1 = FancyArrowPatch((5, 1.5), (5, 2.5),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color=COLOR_MAIN)
ax.add_patch(arrow1)

# MIDDLE: Multi-Head Self-Attention Layers (Innovation 2)
# First attention layer (HIGHLIGHTED - Innovation 2)
attention_box1 = FancyBboxPatch((1.5, 2.8), 7, 1.2,
                                boxstyle="round,pad=0.05",
                                facecolor=COLOR_PURPLE, edgecolor=COLOR_PURPLE,
                                linewidth=3, alpha=0.2)
ax.add_patch(attention_box1)

# Multiple heads visualization
head_positions = [2.5, 4, 5.5, 7]
for i, x in enumerate(head_positions):
    head_circle = Circle((x, 3.4), 0.35, facecolor=COLOR_LAVENDER,
                         edgecolor=COLOR_PURPLE, linewidth=2)
    ax.add_patch(head_circle)
    ax.text(x, 3.4, f'H{i+1}', ha='center', va='center',
            fontsize=8, fontweight='bold')

# Connections between heads
for i in range(len(head_positions) - 1):
    x1, x2 = head_positions[i], head_positions[i+1]
    ax.plot([x1+0.35, x2-0.35], [3.4, 3.4], '-', color=COLOR_PURPLE, linewidth=1, alpha=0.5)

ax.text(5, 2.3, 'Multi-Head Self-Attention', ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLOR_PURPLE)

# Annotation for Innovation 2
ax.text(9.2, 3.4, '❷', ha='center', va='center',
        fontsize=14, fontweight='bold', color=COLOR_PURPLE,
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor=COLOR_PURPLE, linewidth=2))

# Arrow up
arrow2 = FancyArrowPatch((5, 4.2), (5, 4.8),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color=COLOR_MAIN)
ax.add_patch(arrow2)

# Feed-forward layer
ff_box = FancyBboxPatch((2, 5), 6, 0.7,
                        boxstyle="round,pad=0.05",
                        facecolor=COLOR_LIGHT_BG, edgecolor=COLOR_MAIN, linewidth=2)
ax.add_patch(ff_box)
ax.text(5, 5.35, 'Feed-Forward Network', ha='center', va='center',
        fontsize=9, fontweight='bold')

# Arrow up
arrow3 = FancyArrowPatch((5, 5.8), (5, 6.5),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color=COLOR_MAIN)
ax.add_patch(arrow3)

# Second attention layer
attention_box2 = FancyBboxPatch((1.5, 6.7), 7, 1.2,
                                boxstyle="round,pad=0.05",
                                facecolor=COLOR_PURPLE, edgecolor=COLOR_PURPLE,
                                linewidth=2, alpha=0.15)
ax.add_patch(attention_box2)

# Multiple heads (layer 2)
for i, x in enumerate(head_positions):
    head_circle = Circle((x, 7.3), 0.35, facecolor=COLOR_LAVENDER,
                         edgecolor=COLOR_PURPLE, linewidth=2)
    ax.add_patch(head_circle)
    ax.text(x, 7.3, f'H{i+1}', ha='center', va='center',
            fontsize=8, fontweight='bold')

ax.text(5, 6.2, 'Multi-Head Self-Attention', ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLOR_PURPLE)

# Arrow up
arrow4 = FancyArrowPatch((5, 8.1), (5, 8.7),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color=COLOR_MAIN)
ax.add_patch(arrow4)

# Stacking indicator
ax.text(0.5, 5.5, '×N\nlayers', ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLOR_MAIN)

# Output layer
output_box = FancyBboxPatch((2, 9), 6, 0.7,
                            boxstyle="round,pad=0.05",
                            facecolor=COLOR_LIGHT_BG, edgecolor=COLOR_MAIN, linewidth=2)
ax.add_patch(output_box)
ax.text(5, 9.35, 'Output Layer', ha='center', va='center',
        fontsize=9, fontweight='bold')

# RIGHT SIDE: Parallelization annotation (Innovation 3)
# Parallel processing arrows
parallel_y_positions = [1, 3.4, 5.35, 7.3, 9.35]
for y in parallel_y_positions:
    parallel_arrow = FancyArrowPatch((9.5, y), (10.5, y),
                                    arrowstyle='->', mutation_scale=15,
                                    linewidth=2, color=COLOR_GREEN, alpha=0.6)
    ax.add_patch(parallel_arrow)

# Parallelization box
parallel_box = FancyBboxPatch((9.3, 4), 1.5, 2.5,
                              boxstyle="round,pad=0.1",
                              facecolor=COLOR_GREEN, edgecolor=COLOR_GREEN,
                              linewidth=3, alpha=0.1)
ax.add_patch(parallel_box)

ax.text(10, 5.25, 'Parallel\nProcessing', ha='center', va='center',
        fontsize=9, fontweight='bold', color=COLOR_GREEN, rotation=270)

# Annotation for Innovation 3
ax.text(10.8, 5.25, '❸', ha='center', va='center',
        fontsize=14, fontweight='bold', color=COLOR_GREEN,
        bbox=dict(boxstyle='circle', facecolor='white', edgecolor=COLOR_GREEN, linewidth=2))

# LEGEND: Three Innovations
legend_y = 10.5
legends = [
    (1.5, COLOR_BLUE, '❶ Positional Encoding: Adds order without sequential processing'),
    (4.5, COLOR_PURPLE, '❷ Self-Attention: All words attend to all words simultaneously'),
    (7.5, COLOR_GREEN, '❸ Parallelization: 100x speedup by using all GPU cores'),
]

for x, color, text in legends:
    ax.text(x, legend_y, text, ha='left', va='center',
            fontsize=9, fontweight='bold', color=color)

plt.tight_layout()

# Save figure
output_path = '../figures/sr_16_transformer_architecture_annotated.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')

plt.close()
