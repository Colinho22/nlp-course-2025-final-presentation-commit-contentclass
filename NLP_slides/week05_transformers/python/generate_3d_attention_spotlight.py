"""
Generate 3D attention spotlight visualization
Shows selective attention as spotlights in 3D space
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Custom arrow class for 3D
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

# Create figure
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Sentence words positioned in 3D
sentence = ['The', 'black', 'cat', 'sat', 'on', 'the', 'mat']
n_words = len(sentence)

# Arrange words in a curved line in 3D
positions = []
for i in range(n_words):
    x = i * 0.3 - (n_words-1) * 0.15
    y = 0.3 * np.sin(i * np.pi / n_words)
    z = 0
    positions.append([x, y, z])

# Attention weights for word "mat" attending to all others
attention_weights = [0.05, 0.05, 0.15, 0.20, 0.30, 0.25, 0.00]  # Sum to 1.0
# High attention on "on" and "the" (location pattern)

# Plot words
colors = plt.cm.coolwarm(attention_weights)
for i, (word, pos, weight) in enumerate(zip(sentence, positions, attention_weights)):
    # Size based on attention weight
    size = 200 + weight * 2000
    ax.scatter(pos[0], pos[1], pos[2], s=size, c=[colors[i]],
              edgecolors='black', linewidth=2, alpha=0.8)
    ax.text(pos[0], pos[1], pos[2] - 0.15, word,
           fontsize=10 + weight*20, fontweight='bold', ha='center')

    # Add percentage label
    if weight > 0.05:
        ax.text(pos[0], pos[1], pos[2] + 0.15, f'{weight:.0%}',
               fontsize=9, ha='center', color='darkred')

# Query word (mat) positioned above
query_pos = [positions[-1][0], positions[-1][1], 0.8]
ax.scatter(query_pos[0], query_pos[1], query_pos[2],
          s=800, c='gold', edgecolors='black', linewidth=3,
          marker='*', alpha=0.9)
ax.text(query_pos[0], query_pos[1], query_pos[2] + 0.15,
       'Predicting: "mat"', fontsize=12, fontweight='bold', ha='center')

# Draw attention beams (spotlights) from query to words
for i, (pos, weight) in enumerate(zip(positions[:-1], attention_weights[:-1])):
    if weight > 0.05:  # Only show significant attention
        # Create cone-like spotlight effect with multiple lines
        for offset in np.linspace(-0.05, 0.05, 3):
            arrow = Arrow3D([query_pos[0] + offset, pos[0]],
                          [query_pos[1] + offset, pos[1]],
                          [query_pos[2], pos[2]],
                          mutation_scale=20, lw=weight*10,
                          arrowstyle='-', color='yellow', alpha=weight*0.7)
            ax.add_artist(arrow)

# Add attention distribution pie chart (2D overlay)
ax2 = fig.add_axes([0.72, 0.65, 0.25, 0.25])
pie_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
wedges, texts, autotexts = ax2.pie(attention_weights,
                                    labels=sentence,
                                    colors=pie_colors,
                                    autopct=lambda pct: f'{pct:.0f}%' if pct > 5 else '',
                                    startangle=90)
ax2.set_title('Attention Distribution', fontsize=10, fontweight='bold')

# Main plot settings
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1, 1)
ax.set_zlim(-0.5, 1.5)
ax.view_init(elev=25, azim=-60)
ax.grid(True, alpha=0.3)

# Remove axis labels for cleaner look
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_title('Selective Attention: Focus on What Matters',
            fontsize=16, fontweight='bold', pad=20)

# Add explanation
ax.text2D(0.5, 0.05,
         'Attention acts like adjustable spotlights - brighter beams = more focus!',
         transform=ax.transAxes, fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3, pad=0.5))

plt.tight_layout()
plt.savefig('../figures/3d_attention_spotlight.pdf', dpi=300, bbox_inches='tight')
print("Generated: 3d_attention_spotlight.pdf")
plt.close()