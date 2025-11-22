"""
Generate 3D chaos network visualization
Shows what happens when you connect everything to everything
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Create figure with two subplots
fig = plt.figure(figsize=(14, 7))

# Left: Small manageable network
ax1 = fig.add_subplot(121, projection='3d')

# Create positions for 5 words
n_words_small = 5
np.random.seed(42)
positions_small = []
for i in range(n_words_small):
    angle = i * 2 * np.pi / n_words_small
    x = 0.5 * np.cos(angle)
    y = 0.5 * np.sin(angle)
    z = np.random.uniform(-0.2, 0.2)
    positions_small.append([x, y, z])

# Plot words
words_small = ['The', 'cat', 'sat', 'on', 'mat']
for i, (pos, word) in enumerate(zip(positions_small, words_small)):
    ax1.scatter(pos[0], pos[1], pos[2], s=500, c='blue',
               edgecolors='black', linewidth=2, alpha=0.8)
    ax1.text(pos[0], pos[1], pos[2], word,
            fontsize=10, ha='center', fontweight='bold', color='white')

# Draw ALL connections
for i in range(n_words_small):
    for j in range(i+1, n_words_small):
        ax1.plot([positions_small[i][0], positions_small[j][0]],
                [positions_small[i][1], positions_small[j][1]],
                [positions_small[i][2], positions_small[j][2]],
                'g-', alpha=0.5, linewidth=2)

ax1.set_title('5 Words = 10 Connections\nMANAGEABLE',
             fontsize=12, fontweight='bold', color='green')
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax1.set_zlim(-0.5, 0.5)
ax1.view_init(elev=20, azim=45)
ax1.grid(True, alpha=0.3)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])

# Right: Large chaotic network
ax2 = fig.add_subplot(122, projection='3d')

# Create positions for 20 words
n_words_large = 20
np.random.seed(42)
positions_large = []
for i in range(n_words_large):
    # More chaotic positioning
    theta = np.random.uniform(0, 2*np.pi)
    phi = np.random.uniform(0, np.pi)
    r = np.random.uniform(0.3, 0.8)

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi) - 0.5
    positions_large.append([x, y, z])

# Plot words (smaller due to crowding)
for i, pos in enumerate(positions_large):
    ax2.scatter(pos[0], pos[1], pos[2], s=200, c='red',
               edgecolors='black', linewidth=1, alpha=0.6)

# Draw ALL connections (this will be chaos!)
connection_count = 0
for i in range(n_words_large):
    for j in range(i+1, n_words_large):
        # Vary line properties to show chaos
        alpha = np.random.uniform(0.1, 0.3)
        width = np.random.uniform(0.5, 1.5)

        # Color code by distance
        dist = np.sqrt(sum((positions_large[i][k] - positions_large[j][k])**2 for k in range(3)))
        if dist < 0.5:
            color = 'red'
        elif dist < 1.0:
            color = 'orange'
        else:
            color = 'yellow'

        ax2.plot([positions_large[i][0], positions_large[j][0]],
                [positions_large[i][1], positions_large[j][1]],
                [positions_large[i][2], positions_large[j][2]],
                color=color, alpha=alpha, linewidth=width)
        connection_count += 1

# Add noise particles to emphasize chaos
for _ in range(50):
    x, y, z = np.random.uniform(-1, 1, 3)
    ax2.scatter(x, y, z, s=10, c='gray', alpha=0.3)

ax2.set_title(f'20 Words = {connection_count} Connections\nCOMPLETE CHAOS!',
             fontsize=12, fontweight='bold', color='red')
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.set_zlim(-1, 0.5)
ax2.view_init(elev=20, azim=-45)
ax2.grid(True, alpha=0.3)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])

# Add warning labels
ax2.text(0, 0, 0.6, 'INFORMATION\nOVERLOAD!',
        fontsize=11, fontweight='bold', color='red', ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Main title
fig.suptitle('The Naive Approach: Connect Everything to Everything',
            fontsize=16, fontweight='bold')

# Add statistics
fig.text(0.25, 0.15, '✓ Clear patterns\n✓ Easy to process\n✓ Works well',
        fontsize=10, ha='center', color='green',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

fig.text(0.75, 0.15, '✗ No clear patterns\n✗ Signal lost in noise\n✗ Computation explodes',
        fontsize=10, ha='center', color='red',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

# Bottom message
fig.text(0.5, 0.05,
        'Like everyone in a room shouting at once - you can\'t understand anything!',
        fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.subplots_adjust(top=0.88, bottom=0.18)
plt.savefig('../figures/chaos_network_3d.pdf', dpi=300, bbox_inches='tight')
print("Generated: chaos_network_3d.pdf")
plt.close()