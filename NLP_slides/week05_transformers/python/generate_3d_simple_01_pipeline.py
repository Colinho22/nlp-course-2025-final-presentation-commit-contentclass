"""
Generate 3D Pipeline Visualization
Input → Model → Output flow in 3D space
Shows the basic transformer pipeline
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# Template colors
COLOR_BLUE = '#4472C4'
COLOR_PURPLE = '#8B5A9B'
COLOR_GREEN = '#44A044'
COLOR_MAIN = '#333366'

# Create figure
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Set view angle
ax.view_init(elev=20, azim=45)

# Stage 1: INPUT (left side - blue cube)
def draw_cube(ax, center, size, color, alpha, label):
    x, y, z = center
    s = size / 2

    vertices = np.array([
        [x-s, y-s, z-s], [x+s, y-s, z-s], [x+s, y+s, z-s], [x-s, y+s, z-s],  # bottom
        [x-s, y-s, z+s], [x+s, y-s, z+s], [x+s, y+s, z+s], [x-s, y+s, z+s]   # top
    ])

    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]]   # top
    ]

    collection = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=COLOR_MAIN, linewidth=2)
    ax.add_collection3d(collection)

    # Label
    ax.text(x, y, z+s+0.5, label, fontsize=14, fontweight='bold', ha='center', color=color)

# INPUT cube
draw_cube(ax, (0, 2, 2), 2, COLOR_BLUE, 0.6, 'INPUT\n"The cat sat"')

# Arrow 1: Input → Model (using simple line + marker)
ax.plot([1.2, 3.8], [2, 2], [2, 2], color=COLOR_MAIN, linewidth=4)
ax.scatter([3.8], [2], [2], c=COLOR_MAIN, s=200, marker='>')
ax.text(2.5, 2, 3, '1. Words enter', fontsize=10, ha='center', style='italic')

# MODEL cube (center - purple, larger)
draw_cube(ax, (5, 2, 2), 3, COLOR_PURPLE, 0.7, 'MODEL\n(Transformer)')

# Show internal "gears" - small spheres inside model
for i in range(6):
    angle = i * np.pi / 3
    x = 5 + 0.8 * np.cos(angle)
    y = 2 + 0.8 * np.sin(angle)
    ax.scatter([x], [y], [2], c=COLOR_PURPLE, s=100, alpha=0.8)

# Arrow 2: Model → Output (using simple line + marker)
ax.plot([6.8, 9.2], [2, 2], [2, 2], color=COLOR_MAIN, linewidth=4)
ax.scatter([9.2], [2], [2], c=COLOR_MAIN, s=200, marker='>')
ax.text(8, 2, 3, '2. Compute\n3. Predict', fontsize=10, ha='center', style='italic')

# OUTPUT cube
draw_cube(ax, (10.5, 2, 2), 2, COLOR_GREEN, 0.6, 'OUTPUT\n"Le chat..."')

# Title
fig.suptitle('The Transformer Pipeline: Input → Process → Output',
             fontsize=16, fontweight='bold', y=0.95)

# Bottom explanation boxes
ax.text(0, -1, 0.5, 'INPUT:\n7 words\n(text)', fontsize=9, ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_BLUE, alpha=0.3))

ax.text(5, -1, 0.5, 'PROCESS:\nAll words\nat once!', fontsize=9, ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_PURPLE, alpha=0.3))

ax.text(10.5, -1, 0.5, 'OUTPUT:\n7 words\n(translation)', fontsize=9, ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_GREEN, alpha=0.3))

# WHY box at bottom
ax.text(5, 2, -1.5, 'WHY: We need to understand what each word means in context to produce correct output',
        fontsize=11, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFE6CC', edgecolor=COLOR_MAIN, linewidth=2))

# Set axis properties
ax.set_xlim(-2, 13)
ax.set_ylim(-2, 6)
ax.set_zlim(-2, 5)
ax.set_xlabel('Pipeline Flow', fontsize=10, labelpad=10)
ax.set_ylabel('', fontsize=10)
ax.set_zlabel('', fontsize=10)

# Hide grid for cleaner look
ax.grid(True, alpha=0.3)
ax.set_box_aspect([3, 1, 1])

plt.tight_layout()

# Save figure
output_path = '../figures/sr_3d_simple_01_pipeline.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')

plt.close()
