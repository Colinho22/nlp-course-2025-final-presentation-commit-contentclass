"""
Generate 3D Sequential Bottleneck (RED LINE)
Shows RNN sequential processing with prominent RED bottleneck line
Input → Word 1 → wait → Word 2 → wait → Word 3
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# Template colors
COLOR_RED = '#D62728'
COLOR_BLUE = '#4472C4'
COLOR_MAIN = '#333366'
COLOR_GRAY = '#999999'

# Create figure
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Set view angle
ax.view_init(elev=25, azim=45)

# Draw processing boxes for each word (staircase)
words = ['Word 1\n"The"', 'Word 2\n"cat"', 'Word 3\n"sat"']
z_levels = [1, 2, 3]  # Ascending stairs

for i, (word, z) in enumerate(zip(words, z_levels)):
    # Processing box
    x = i * 3 + 1
    y = 2

    # Create cube
    size = 1.2
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

    collection = Poly3DCollection(faces, alpha=0.6, facecolor=COLOR_BLUE,
                                  edgecolor=COLOR_MAIN, linewidth=2)
    ax.add_collection3d(collection)

    # Label
    ax.text(x, y-1.5, z, word, fontsize=11, ha='center', color=COLOR_BLUE, fontweight='bold')

    # Time label
    ax.text(x, y+1.5, z, f't={i+1}', fontsize=10, ha='center', color=COLOR_RED,
            fontweight='bold', style='italic')

# THE RED LINE - showing sequential bottleneck
# Zigzag path showing wait time
x_path = []
y_path = []
z_path = []

for i in range(len(words)):
    x = i * 3 + 1
    y = 2
    z = z_levels[i]

    x_path.extend([x, x])
    y_path.extend([y, y])
    z_path.extend([z-0.6, z+0.6])

    if i < len(words) - 1:
        # Connect to next box
        next_x = (i+1) * 3 + 1
        next_z = z_levels[i+1]
        x_path.extend([x, next_x])
        y_path.extend([y, y])
        z_path.extend([z+0.6, next_z-0.6])

# Draw THE RED LINE with extra thickness
ax.plot(x_path, y_path, z_path, color=COLOR_RED, linewidth=8,
        label='Sequential Bottleneck', zorder=10)

# Add "WAIT" labels between boxes
for i in range(len(words) - 1):
    x_mid = (i * 3 + 1 + (i+1) * 3 + 1) / 2
    y_mid = 2
    z_mid = (z_levels[i] + z_levels[i+1]) / 2

    ax.text(x_mid, y_mid + 1, z_mid, 'WAIT!', fontsize=14, ha='center',
            color=COLOR_RED, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE6E6',
                     edgecolor=COLOR_RED, linewidth=2))

# Title
fig.suptitle('RNN: Sequential Processing = RED LINE Bottleneck',
             fontsize=16, fontweight='bold', y=0.95, color=COLOR_RED)

# Problem box at bottom
ax.text(4, 2, -0.5, 'PROBLEM: Each word must wait for previous word to finish',
        fontsize=12, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFE6E6',
                 edgecolor=COLOR_RED, linewidth=3))

# Stats box
stats_text = 'Time: 300ms\nGPU Usage: 2%\n90 days training'
ax.text(8, 2, 1.5, stats_text, fontsize=10, ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_GRAY, alpha=0.3))

# WHY box
ax.text(4, 2, 4.5, 'WHY THIS IS BAD: Cannot use all GPU cores → 95% GPU sits idle → Takes forever!',
        fontsize=11, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFE6CC',
                 edgecolor=COLOR_MAIN, linewidth=2))

# Set axis properties
ax.set_xlim(-1, 9)
ax.set_ylim(0, 4)
ax.set_zlim(-1, 5)
ax.set_xlabel('Word Sequence', fontsize=10, labelpad=10)
ax.set_ylabel('', fontsize=10)
ax.set_zlabel('Time Steps', fontsize=10, labelpad=10)

# Grid
ax.grid(True, alpha=0.3)
ax.set_box_aspect([2, 1, 1.5])

plt.tight_layout()

# Save figure
output_path = '../figures/sr_3d_simple_02_sequential_redline.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')

plt.close()
