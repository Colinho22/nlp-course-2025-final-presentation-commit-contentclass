"""
Generate 3D Word-to-Vector Transformation
Shows how text "cat" becomes a numerical vector
INPUT: text → PROCESS: lookup → OUTPUT: numbers
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Template colors
COLOR_BLUE = '#4472C4'
COLOR_PURPLE = '#8B5A9B'
COLOR_MAIN = '#333366'

# Create figure
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Set view angle
ax.view_init(elev=20, azim=45)

# LEFT SIDE: Text word "cat"
ax.text(0, 2, 3, 'cat', fontsize=60, ha='center', va='center',
        fontweight='bold', color=COLOR_BLUE,
        bbox=dict(boxstyle='round,pad=1', facecolor='#E6F2FF',
                 edgecolor=COLOR_BLUE, linewidth=3))

ax.text(0, 2, 1.5, 'INPUT\n(text)', fontsize=12, ha='center',
        color=COLOR_BLUE, fontweight='bold')

# MIDDLE: Transformation arrow and process
ax.plot([1.5, 4.5], [2, 2], [3, 3], color=COLOR_PURPLE, linewidth=4)
ax.scatter([4.5], [2], [3], c=COLOR_PURPLE, s=200, marker='>')

ax.text(3, 2, 4, 'Dictionary\nLookup', fontsize=11, ha='center',
        color=COLOR_PURPLE, fontweight='bold', style='italic')

# RIGHT SIDE: Vector representation
vector_values = [0.2, 0.5, -0.1, 0.3, 0.7, -0.4]
vector_labels = ['dim 1', 'dim 2', 'dim 3', 'dim 4', 'dim 5', '...']

# Draw 3D vector as arrows
origin = [6, 2, 3]
for i, (val, label) in enumerate(zip(vector_values, vector_labels)):
    # Scale for visibility
    dx = val * 2
    dy = 0
    dz = -i * 0.5

    # Arrow
    ax.quiver(origin[0], origin[1], origin[2],
             dx, dy, dz,
             arrow_length_ratio=0.3, color=COLOR_PURPLE,
             linewidth=3, alpha=0.8)

    # Value label
    ax.text(origin[0] + dx + 0.3, origin[1], origin[2] + dz,
           f'{val:.1f}', fontsize=10, ha='left',
           color=COLOR_PURPLE, fontweight='bold')

    # Dimension label
    ax.text(origin[0] - 0.5, origin[1], origin[2] + dz,
           label, fontsize=9, ha='right',
           color=COLOR_MAIN)

# Vector bracket
ax.text(origin[0] - 1.2, origin[1], origin[2], '[', fontsize=80,
        ha='center', va='center', color=COLOR_PURPLE)
ax.text(origin[0] - 1.2, origin[1], origin[2] - 2.5, ']', fontsize=80,
        ha='center', va='center', color=COLOR_PURPLE)

ax.text(origin[0], origin[1], origin[2] - 3.5, 'OUTPUT\n(numbers)', fontsize=12,
        ha='center', color=COLOR_PURPLE, fontweight='bold')

# Title
fig.suptitle('Step 1: Turn Words into Numbers',
             fontsize=16, fontweight='bold', y=0.95)

# Bottom explanation
ax.text(3, 2, 0, 'COMPUTATION: Look up "cat" in dictionary → Get vector [0.2, 0.5, -0.1, ...]',
        fontsize=12, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#F0F0F0',
                 edgecolor=COLOR_MAIN, linewidth=2))

# WHY box
ax.text(3, 2, -1.5, 'WHY: Computers only understand numbers, not words!',
        fontsize=11, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFE6CC',
                 edgecolor=COLOR_MAIN, linewidth=2))

# Set axis properties
ax.set_xlim(-2, 10)
ax.set_ylim(0, 4)
ax.set_zlim(-2, 5)
ax.set_xlabel('', fontsize=10)
ax.set_ylabel('', fontsize=10)
ax.set_zlabel('Vector Dimensions', fontsize=10, labelpad=10)

# Grid
ax.grid(True, alpha=0.3)
ax.set_box_aspect([3, 1, 1.5])

# Hide axes for cleaner look
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()

# Save figure
output_path = '../figures/sr_3d_simple_04_word_to_vector.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')

plt.close()
