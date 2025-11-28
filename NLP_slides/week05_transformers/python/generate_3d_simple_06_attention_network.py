"""
Generate 3D Attention Network
Shows attention weights as 3D connections between words
WHO looks at WHO with what strength
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Template colors
COLOR_BLUE = '#4472C4'
COLOR_PURPLE = '#8B5A9B'
COLOR_GREEN = '#44A044'
COLOR_MAIN = '#333366'

# Create figure
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=20, azim=45)

# Words as 3D points
words = ['The', 'cat', 'sat']
positions = [(0, 0, 2), (3, 1, 3), (6, 0, 2)]
colors = [COLOR_BLUE, COLOR_PURPLE, COLOR_GREEN]

# Draw word nodes
for word, pos, color in zip(words, positions, colors):
    ax.scatter([pos[0]], [pos[1]], [pos[2]], s=500, c=color, edgecolors=COLOR_MAIN, linewidth=3, alpha=0.8)
    ax.text(pos[0], pos[1], pos[2]+0.5, word, fontsize=14, ha='center', fontweight='bold', color=color)

# Attention connections (from "cat" to others)
# cat → The (50%)
ax.plot([3, 0], [1, 0], [3, 2], color=COLOR_PURPLE, linewidth=10, alpha=0.6)
ax.text(1.5, 0.5, 2.5, '50%', fontsize=12, ha='center', fontweight='bold', color=COLOR_PURPLE)

# cat → sat (30%)
ax.plot([3, 6], [1, 0], [3, 2], color=COLOR_PURPLE, linewidth=6, alpha=0.6)
ax.text(4.5, 0.5, 2.5, '30%', fontsize=12, ha='center', fontweight='bold', color=COLOR_PURPLE)

# cat → cat (20%)
angles = np.linspace(0, 2*np.pi, 50)
loop_x = 3 + 0.5*np.cos(angles)
loop_y = 1 + 0.5*np.sin(angles)
loop_z = 3 + 0.3*np.sin(2*angles)
ax.plot(loop_x, loop_y, loop_z, color=COLOR_PURPLE, linewidth=4, alpha=0.6)
ax.text(3.8, 1.5, 3.3, '20%', fontsize=12, ha='center', fontweight='bold', color=COLOR_PURPLE)

# Title
fig.suptitle('Step 3: Calculate Attention (Who Looks at Who)', fontsize=16, fontweight='bold', y=0.95)

# Computation box
ax.text(3, 3, 4.5, 'COMPUTATION: For "cat", calculate:\nHow much focus on each word?', fontsize=11, ha='center', fontweight='bold', bbox=dict(boxstyle='round,pad=0.8', facecolor='#F0F0F0', edgecolor=COLOR_MAIN, linewidth=2))

# Output box
ax.text(3, 3, 0.5, 'OUTPUT: Percentage weights\n"The"=50%, "sat"=30%, "cat"=20%', fontsize=11, ha='center', fontweight='bold', bbox=dict(boxstyle='round,pad=0.8', facecolor='#E6E6FF', edgecolor=COLOR_PURPLE, linewidth=2))

# WHY box
ax.text(3, 3, -1, 'WHY: Words need context! "sat" needs to know WHAT sat', fontsize=11, ha='center', fontweight='bold', bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFE6CC', edgecolor=COLOR_MAIN, linewidth=2))

# Set axis properties
ax.set_xlim(-2, 8)
ax.set_ylim(-1, 4)
ax.set_zlim(-2, 5)
ax.grid(True, alpha=0.3)
ax.set_box_aspect([2, 1, 1.5])
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_zlabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

plt.tight_layout()

# Save figure
output_path = '../figures/sr_3d_simple_06_attention_network.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')

plt.close()
