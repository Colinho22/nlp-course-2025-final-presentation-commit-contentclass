"""
Generate 3D connection network showing all-to-all relationships
Demonstrates the complexity explosion in transformers
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure with subplots
fig = plt.figure(figsize=(16, 8))

# Left: Small network (5 words)
ax1 = fig.add_subplot(121, projection='3d')

# Position 5 words in 3D space (circular arrangement)
n_words_small = 5
words_small = ['The', 'cat', 'sat', 'on', 'the']
angles = np.linspace(0, 2*np.pi, n_words_small, endpoint=False)
positions_small = []
for i, angle in enumerate(angles):
    x = 0.5 * np.cos(angle)
    y = 0.5 * np.sin(angle)
    z = 0.3 * np.sin(i * np.pi / n_words_small)
    positions_small.append([x, y, z])

# Plot words
for i, (word, pos) in enumerate(zip(words_small, positions_small)):
    color = plt.cm.viridis(i / n_words_small)
    ax1.scatter(pos[0], pos[1], pos[2], s=500, c=[color],
               edgecolors='black', linewidth=2, alpha=0.8)
    ax1.text(pos[0], pos[1], pos[2] + 0.1, word,
            fontsize=11, fontweight='bold', ha='center')

# Draw all connections
connection_count = 0
for i in range(n_words_small):
    for j in range(i+1, n_words_small):
        pos1 = positions_small[i]
        pos2 = positions_small[j]
        # Vary line thickness based on hypothetical attention weight
        weight = np.random.uniform(0.1, 1.0)
        ax1.plot([pos1[0], pos2[0]],
                [pos1[1], pos2[1]],
                [pos1[2], pos2[2]],
                color='blue', alpha=weight*0.5, linewidth=weight*2)
        connection_count += 1

ax1.set_title(f'Small: 5 words = {connection_count} connections\n(Manageable!)',
             fontsize=12, fontweight='bold')
ax1.set_xlim(-0.8, 0.8)
ax1.set_ylim(-0.8, 0.8)
ax1.set_zlim(-0.5, 0.5)
ax1.view_init(elev=20, azim=45)
ax1.grid(True, alpha=0.3)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])

# Right: Large network (20 words)
ax2 = fig.add_subplot(122, projection='3d')

n_words_large = 20
# Create positions in 3D sphere
np.random.seed(42)
positions_large = []
for i in range(n_words_large):
    theta = np.random.uniform(0, 2*np.pi)
    phi = np.random.uniform(0, np.pi)
    r = 0.5
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    positions_large.append([x, y, z])

# Plot words as smaller dots
for i, pos in enumerate(positions_large):
    color = plt.cm.plasma(i / n_words_large)
    ax2.scatter(pos[0], pos[1], pos[2], s=100, c=[color],
               edgecolors='black', linewidth=1, alpha=0.6)

# Draw connections (but limit to show chaos)
connection_count_large = 0
for i in range(n_words_large):
    for j in range(i+1, n_words_large):
        connection_count_large += 1
        if connection_count_large <= 100:  # Only draw first 100 to avoid complete mess
            pos1 = positions_large[i]
            pos2 = positions_large[j]
            weight = np.random.uniform(0.05, 0.3)
            ax2.plot([pos1[0], pos2[0]],
                    [pos1[1], pos2[1]],
                    [pos1[2], pos2[2]],
                    color='red', alpha=weight, linewidth=weight)

total_connections = n_words_large * (n_words_large - 1) // 2
ax2.set_title(f'Large: 20 words = {total_connections} connections\n(Information overload!)',
             fontsize=12, fontweight='bold')
ax2.set_xlim(-0.8, 0.8)
ax2.set_ylim(-0.8, 0.8)
ax2.set_zlim(-0.8, 0.8)
ax2.view_init(elev=20, azim=45)
ax2.grid(True, alpha=0.3)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])

# Main title
fig.suptitle('All-to-All Connections: The Complexity Explosion',
            fontsize=16, fontweight='bold')

# Add annotation
fig.text(0.5, 0.02, 'Every word must consider every other word - connections grow quadratically!',
        fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3, pad=0.5))

plt.tight_layout()
plt.savefig('../figures/3d_connection_network.pdf', dpi=300, bbox_inches='tight')
print("Generated: 3d_connection_network.pdf")
plt.close()