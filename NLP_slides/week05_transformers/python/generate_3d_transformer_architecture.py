"""
Generate 3D transformer architecture visualization
Shows the complete transformer block in 3D
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def draw_box(ax, corner, width, height, depth, color, alpha=0.3, label=''):
    """Draw a 3D box"""
    # Define the vertices of the box
    vertices = [
        [corner[0], corner[1], corner[2]],
        [corner[0] + width, corner[1], corner[2]],
        [corner[0] + width, corner[1] + height, corner[2]],
        [corner[0], corner[1] + height, corner[2]],
        [corner[0], corner[1], corner[2] + depth],
        [corner[0] + width, corner[1], corner[2] + depth],
        [corner[0] + width, corner[1] + height, corner[2] + depth],
        [corner[0], corner[1] + height, corner[2] + depth]
    ]

    # Define the 6 faces of the box
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
    ]

    # Create the 3D polygon collection
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black', linewidth=1)
    ax.add_collection3d(poly)

    # Add label
    if label:
        center = [corner[0] + width/2, corner[1] + height/2, corner[2] + depth/2]
        ax.text(center[0], center[1], center[2], label, fontsize=10, fontweight='bold',
               ha='center', va='center')

    return vertices

def draw_arrow(ax, start, end, color='black', linewidth=2):
    """Draw arrow between two points"""
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
           color=color, linewidth=linewidth)
    # Add arrowhead using scatter
    ax.scatter(end[0], end[1], end[2], s=100, c=color, marker='^')

# Create figure
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# Layer heights
z_input = 0
z_embedding = 1
z_positional = 2
z_attention = 3.5
z_multihead = 5
z_concat = 6
z_feedforward = 7.5
z_output = 9

# Colors for different components
COLOR_INPUT = '#E8E8E8'
COLOR_EMBEDDING = '#4472C4'
COLOR_POSITIONAL = '#44A044'
COLOR_ATTENTION = '#FF7F0E'
COLOR_FEEDFORWARD = '#D62728'
COLOR_OUTPUT = '#8B5A9B'

# Draw input layer (words)
n_words = 6
for i in range(n_words):
    x = i * 1.5 - 4
    draw_box(ax, [x, -0.5, z_input], 1, 1, 0.3, COLOR_INPUT, alpha=0.5, label=f'W{i+1}')

# Draw embedding layer
draw_box(ax, [-4.5, -1, z_embedding], 8, 2, 0.5, COLOR_EMBEDDING, alpha=0.4, label='Embeddings')

# Draw positional encoding
draw_box(ax, [-4.5, -1, z_positional], 8, 2, 0.5, COLOR_POSITIONAL, alpha=0.4, label='+ Position')

# Draw multi-head attention (4 heads)
head_colors = ['#4472C4', '#44A044', '#D62728', '#8B5A9B']
for i in range(4):
    x = i * 2 - 3
    draw_box(ax, [x, -1, z_attention], 1.5, 2, 1, head_colors[i], alpha=0.5, label=f'Head {i+1}')

# Draw concatenation layer
draw_box(ax, [-4.5, -1, z_concat], 8, 2, 0.5, COLOR_ATTENTION, alpha=0.4, label='Concatenate')

# Draw feed-forward network
draw_box(ax, [-4.5, -1, z_feedforward], 8, 2, 1, COLOR_FEEDFORWARD, alpha=0.4, label='Feed Forward')

# Draw output layer
draw_box(ax, [-4.5, -1, z_output], 8, 2, 0.5, COLOR_OUTPUT, alpha=0.4, label='Output')

# Draw connections (arrows)
# Input to embeddings
for i in range(n_words):
    x = i * 1.5 - 4 + 0.5
    draw_arrow(ax, [x, 0, z_input + 0.3], [x, 0, z_embedding], color='gray', linewidth=1)

# Embeddings to positional
draw_arrow(ax, [0, 0, z_embedding + 0.5], [0, 0, z_positional], color='blue', linewidth=2)

# Positional to attention heads
for i in range(4):
    x = i * 2 - 3 + 0.75
    draw_arrow(ax, [0, 0, z_positional + 0.5], [x, 0, z_attention], color='green', linewidth=1)

# Attention heads to concatenation
for i in range(4):
    x = i * 2 - 3 + 0.75
    draw_arrow(ax, [x, 0, z_attention + 1], [0, 0, z_concat], color='orange', linewidth=1)

# Concatenation to feed-forward
draw_arrow(ax, [0, 0, z_concat + 0.5], [0, 0, z_feedforward], color='orange', linewidth=2)

# Feed-forward to output
draw_arrow(ax, [0, 0, z_feedforward + 1], [0, 0, z_output], color='red', linewidth=2)

# Draw residual connections (skip connections)
# From positional to concat
ax.plot([-5, -5, -5], [0, 0, 0], [z_positional + 0.25, z_concat - 0.5, z_concat + 0.25],
       'g--', linewidth=2, alpha=0.5)
# From concat to output
ax.plot([5, 5, 5], [0, 0, 0], [z_concat + 0.25, z_output - 0.5, z_output + 0.25],
       'r--', linewidth=2, alpha=0.5)

# Add labels for key concepts
ax.text(-6, 0, z_attention + 0.5, 'Parallel\nAttention', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax.text(6, 0, z_concat, 'Residual\nConnection', fontsize=9, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Set labels and title
ax.set_xlabel('Sequence Position', fontsize=11, fontweight='bold')
ax.set_ylabel('Feature Dimension', fontsize=11, fontweight='bold')
ax.set_zlabel('Layer Depth', fontsize=11, fontweight='bold')
ax.set_title('Complete Transformer Architecture in 3D\nAll Processing Happens in Parallel!',
            fontsize=14, fontweight='bold')

# Set viewing angle
ax.view_init(elev=20, azim=-45)

# Set limits
ax.set_xlim(-7, 7)
ax.set_ylim(-3, 3)
ax.set_zlim(-0.5, 10)

# Add legend
legend_elements = [
    mpatches.Patch(color=COLOR_EMBEDDING, label='Embedding Layer'),
    mpatches.Patch(color=COLOR_POSITIONAL, label='Positional Encoding'),
    mpatches.Patch(color=COLOR_ATTENTION, label='Multi-Head Attention'),
    mpatches.Patch(color=COLOR_FEEDFORWARD, label='Feed-Forward Network'),
    mpatches.Patch(color=COLOR_OUTPUT, label='Output Layer')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

# Add annotation
ax.text2D(0.5, 0.02,
         'Information flows upward through parallel layers - no sequential bottleneck!',
         transform=ax.transAxes, fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3, pad=0.5))

plt.tight_layout()
plt.savefig('../figures/3d_transformer_architecture.pdf', dpi=300, bbox_inches='tight')
print("Generated: 3d_transformer_architecture.pdf")
plt.close()