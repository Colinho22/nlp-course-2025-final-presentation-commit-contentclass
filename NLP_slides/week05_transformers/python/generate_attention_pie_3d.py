"""
Generate 3D attention pie chart visualization
Shows attention distribution as percentages in 3D space
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Sentence words
sentence = ['The', 'cat', 'sat', 'on', 'the', 'mat']
query_word = 'mat'  # Word we're predicting

# Attention weights (sum to 1.0)
attention_weights = [0.05, 0.15, 0.20, 0.35, 0.25, 0.00]  # for words before 'mat'
words_attending = sentence[:-1]  # All words except 'mat'

# Colors for each word
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

# Position words in a semicircle at base
positions = []
for i in range(len(words_attending)):
    angle = i * np.pi / (len(words_attending) - 1) - np.pi/2
    x = 1.5 * np.cos(angle)
    y = 1.5 * np.sin(angle)
    z = 0
    positions.append([x, y, z])

# Plot words with size based on attention weight
for i, (word, pos, weight) in enumerate(zip(words_attending, positions, attention_weights)):
    size = 200 + weight * 2000  # Size proportional to attention

    ax.scatter(pos[0], pos[1], pos[2], s=size, c=[colors[i]],
              edgecolors='black', linewidth=2, alpha=0.8)
    ax.text(pos[0], pos[1], pos[2] - 0.2, word,
           fontsize=10 + weight*10, fontweight='bold', ha='center')

    # Add percentage labels
    ax.text(pos[0], pos[1], pos[2] + 0.2, f'{weight:.0%}',
           fontsize=11, ha='center', color='darkred', fontweight='bold')

# Add the query word 'mat' above
query_pos = [0, 0, 2]
ax.scatter(query_pos[0], query_pos[1], query_pos[2],
          s=1000, c='gold', edgecolors='black', linewidth=3,
          marker='*', alpha=0.9)
ax.text(query_pos[0], query_pos[1], query_pos[2] + 0.3,
       f'Predicting: "{query_word}"', fontsize=13, fontweight='bold', ha='center')

# Draw attention beams from query to words
for i, (pos, weight) in enumerate(zip(positions, attention_weights)):
    if weight > 0.05:  # Only show significant attention
        # Multiple lines for beam effect
        for offset in np.linspace(-0.1, 0.1, 3):
            ax.plot([query_pos[0] + offset, pos[0]],
                   [query_pos[1] + offset, pos[1]],
                   [query_pos[2], pos[2]],
                   color=colors[i], alpha=weight, linewidth=weight*15)

# Create 3D pie chart floating in space
pie_center = [-2.5, 0, 1]
pie_radius = 0.8
pie_height = 0.3

# Calculate angles for pie slices
angles = np.cumsum([0] + attention_weights[:-1]) * 2 * np.pi

for i in range(len(attention_weights)):
    if attention_weights[i] > 0:
        # Start and end angles
        theta1 = angles[i]
        theta2 = angles[i] + attention_weights[i] * 2 * np.pi

        # Create pie slice vertices
        n_points = max(int(attention_weights[i] * 30), 3)
        theta = np.linspace(theta1, theta2, n_points)

        # Bottom face
        x_bottom = pie_center[0] + pie_radius * np.cos(theta)
        y_bottom = pie_center[1] + pie_radius * np.sin(theta)
        z_bottom = np.full_like(theta, pie_center[2])

        # Top face
        z_top = np.full_like(theta, pie_center[2] + pie_height)

        # Draw sides
        for j in range(len(theta) - 1):
            verts = [
                [x_bottom[j], y_bottom[j], z_bottom[j]],
                [x_bottom[j+1], y_bottom[j+1], z_bottom[j+1]],
                [x_bottom[j+1], y_bottom[j+1], z_top[j+1]],
                [x_bottom[j], y_bottom[j], z_top[j]]
            ]

            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            poly = Poly3DCollection([verts], alpha=0.7, facecolor=colors[i],
                                   edgecolor='black', linewidth=0.5)
            ax.add_collection3d(poly)

        # Add label
        mid_angle = (theta1 + theta2) / 2
        label_x = pie_center[0] + 1.2 * pie_radius * np.cos(mid_angle)
        label_y = pie_center[1] + 1.2 * pie_radius * np.sin(mid_angle)
        label_z = pie_center[2] + pie_height/2

        ax.text(label_x, label_y, label_z,
               f'{words_attending[i]}\n{attention_weights[i]:.0%}',
               fontsize=9, ha='center')

# Add title for pie chart
ax.text(pie_center[0], pie_center[1], pie_center[2] + pie_height + 0.3,
       'Attention Distribution', fontsize=11, fontweight='bold', ha='center')

# Main title
ax.set_title('Attention as Percentages: Where to Look\nAll weights sum to 100%',
            fontsize=14, fontweight='bold')

# Set viewing angle
ax.view_init(elev=25, azim=-60)

# Set limits
ax.set_xlim(-4, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-0.5, 3)

# Remove axis labels for cleaner look
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Add explanation box
ax.text2D(0.7, 0.15,
         'Key Properties:\n'
         '• Percentages sum to 100%\n'
         '• Higher % = more important\n'
         '• Learned from data\n'
         '• Different for each word',
         transform=ax.transAxes, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Add bottom annotation
ax.text2D(0.5, 0.02,
         'Attention weights determine how much each word contributes to understanding',
         transform=ax.transAxes, fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('../figures/attention_pie_3d.pdf', dpi=300, bbox_inches='tight')
print("Generated: attention_pie_3d.pdf")
plt.close()