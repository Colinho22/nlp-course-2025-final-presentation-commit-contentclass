"""
Generate 3D Parallel Processing Cube (NO RED LINE!)
Shows Transformer parallel processing - all words at once
NO sequential bottleneck - full GPU utilization
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# Template colors
COLOR_GREEN = '#44A044'
COLOR_MAIN = '#333366'
COLOR_LIGHT_GREEN = '#E6FFE6'

# Create figure
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Set view angle
ax.view_init(elev=25, azim=45)

# Create 3x3x3 grid of small cubes (all processed simultaneously)
words = ['The', 'cat', 'sat', 'on', 'the', 'mat', '...', '...', '...']
positions = []

for i in range(3):
    for j in range(3):
        for k in range(3):
            positions.append((i, j, k))

# Draw each small cube
for idx, (i, j, k) in enumerate(positions[:len(words)]):
    x, y, z = i * 2 + 1, j * 2 + 1, k * 2 + 1
    size = 0.8
    s = size / 2

    vertices = np.array([
        [x-s, y-s, z-s], [x+s, y-s, z-s], [x+s, y+s, z-s], [x-s, y+s, z-s],
        [x-s, y-s, z+s], [x+s, y-s, z+s], [x+s, y+s, z+s], [x-s, y+s, z+s]
    ])

    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]]
    ]

    collection = Poly3DCollection(faces, alpha=0.7, facecolor=COLOR_GREEN,
                                  edgecolor=COLOR_MAIN, linewidth=1.5)
    ax.add_collection3d(collection)

    # Label on cube
    if idx < len(words):
        ax.text(x, y, z, words[idx], fontsize=9, ha='center', va='center',
                color='white', fontweight='bold')

# Big arrow showing all at once
for idx, (i, j, k) in enumerate(positions[:len(words)]):
    x, y, z = i * 2 + 1, j * 2 + 1, k * 2 + 1
    # Draw vertical green line above each cube to show "all at once"
    ax.plot([x, x], [y, y], [z+0.5, z+1.5], color=COLOR_GREEN, linewidth=3, alpha=0.8)

# "ALL AT ONCE" label
ax.text(3, 3, 7, 'ALL AT ONCE!', fontsize=18, ha='center', fontweight='bold',
        color=COLOR_GREEN,
        bbox=dict(boxstyle='round,pad=0.8', facecolor=COLOR_LIGHT_GREEN,
                 edgecolor=COLOR_GREEN, linewidth=3))

# Time indicator - single timestamp
ax.text(3, 3, 8.5, 't=1 (10ms)', fontsize=14, ha='center',
        color=COLOR_GREEN, fontweight='bold', style='italic')

# Title
fig.suptitle('Transformer: Parallel Processing = NO RED LINE!',
             fontsize=16, fontweight='bold', y=0.95, color=COLOR_GREEN)

# Solution box at bottom
ax.text(3, 3, -1, 'SOLUTION: All words processed simultaneously',
        fontsize=12, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.8', facecolor=COLOR_LIGHT_GREEN,
                 edgecolor=COLOR_GREEN, linewidth=3))

# Stats box
stats_text = 'Time: 10ms\nGPU Usage: 92%\n1 day training'
ax.text(7, 3, 3, stats_text, fontsize=11, ha='left', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_LIGHT_GREEN,
                 edgecolor=COLOR_GREEN, linewidth=2))

# WHY box
ax.text(3, 3, -2.5, 'WHY THIS WORKS: Uses all GPU cores at once → 100% GPU power → 90x faster!',
        fontsize=11, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFE6CC',
                 edgecolor=COLOR_MAIN, linewidth=2))

# Comparison label
ax.text(-1, 3, 3, 'NO WAITING\nNO RED LINE\nFULL PARALLEL', fontsize=12, ha='center',
        color=COLOR_GREEN, fontweight='bold', rotation=0)

# Set axis properties
ax.set_xlim(-2, 8)
ax.set_ylim(-2, 8)
ax.set_zlim(-3, 9)
ax.set_xlabel('Word Dimension 1', fontsize=10, labelpad=10)
ax.set_ylabel('Word Dimension 2', fontsize=10, labelpad=10)
ax.set_zlabel('Processing Units', fontsize=10, labelpad=10)

# Grid
ax.grid(True, alpha=0.3)
ax.set_box_aspect([1, 1, 1.2])

plt.tight_layout()

# Save figure
output_path = '../figures/sr_3d_simple_03_parallel_cube.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')

plt.close()
