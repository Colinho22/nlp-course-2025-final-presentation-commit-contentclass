"""
Generate 3D word space visualization for BSc transformer lecture
Shows words floating in 3D semantic space
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Define colors
COLOR_ANIMALS = '#4472C4'  # Blue
COLOR_FURNITURE = '#44A044'  # Green
COLOR_VEHICLES = '#FF7F0E'  # Orange
COLOR_ACTIONS = '#D62728'  # Red
COLOR_MAIN = '#333366'

# Create figure
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Define words and their positions in 3D semantic space
words_data = {
    # Animals cluster (high on dimension 0)
    'cat': ([0.7, 0.2, 0.5], COLOR_ANIMALS),
    'dog': ([0.8, 0.3, 0.4], COLOR_ANIMALS),
    'bird': ([0.6, 0.1, 0.6], COLOR_ANIMALS),

    # Furniture cluster (high on dimension 1)
    'chair': ([0.2, 0.8, 0.3], COLOR_FURNITURE),
    'table': ([0.3, 0.9, 0.2], COLOR_FURNITURE),
    'sofa': ([0.1, 0.7, 0.4], COLOR_FURNITURE),

    # Vehicles cluster (high on dimension 2)
    'car': ([0.3, 0.2, 0.9], COLOR_VEHICLES),
    'bus': ([0.2, 0.3, 0.8], COLOR_VEHICLES),
    'bike': ([0.4, 0.1, 0.7], COLOR_VEHICLES),

    # Actions scattered
    'run': ([0.5, 0.5, 0.2], COLOR_ACTIONS),
    'sit': ([0.4, 0.6, 0.3], COLOR_ACTIONS),
    'jump': ([0.6, 0.4, 0.1], COLOR_ACTIONS),
}

# Plot words as spheres in 3D space
for word, (pos, color) in words_data.items():
    ax.scatter(pos[0], pos[1], pos[2],
              s=800, c=color, alpha=0.6,
              edgecolors='black', linewidth=2)
    ax.text(pos[0], pos[1], pos[2] + 0.08, word,
           fontsize=12, fontweight='bold', ha='center')

# Draw connections between similar words (same category)
# Animals connections
animal_words = ['cat', 'dog', 'bird']
for i in range(len(animal_words)):
    for j in range(i+1, len(animal_words)):
        pos1 = words_data[animal_words[i]][0]
        pos2 = words_data[animal_words[j]][0]
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
               'b--', alpha=0.3, linewidth=1)

# Furniture connections
furniture_words = ['chair', 'table', 'sofa']
for i in range(len(furniture_words)):
    for j in range(i+1, len(furniture_words)):
        pos1 = words_data[furniture_words[i]][0]
        pos2 = words_data[furniture_words[j]][0]
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
               'g--', alpha=0.3, linewidth=1)

# Vehicle connections
vehicle_words = ['car', 'bus', 'bike']
for i in range(len(vehicle_words)):
    for j in range(i+1, len(vehicle_words)):
        pos1 = words_data[vehicle_words[i]][0]
        pos2 = words_data[vehicle_words[j]][0]
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
               'orange', linestyle='--', alpha=0.3, linewidth=1)

# Set labels and title
ax.set_xlabel('Dimension 1: Animal-ness', fontsize=12, fontweight='bold')
ax.set_ylabel('Dimension 2: Furniture-ness', fontsize=12, fontweight='bold')
ax.set_zlabel('Dimension 3: Vehicle-ness', fontsize=12, fontweight='bold')
ax.set_title('Words as Points in 3D Meaning Space', fontsize=16, fontweight='bold', pad=20)

# Set viewing angle
ax.view_init(elev=20, azim=45)

# Add legend
legend_elements = [
    mpatches.Patch(color=COLOR_ANIMALS, label='Animals'),
    mpatches.Patch(color=COLOR_FURNITURE, label='Furniture'),
    mpatches.Patch(color=COLOR_VEHICLES, label='Vehicles'),
    mpatches.Patch(color=COLOR_ACTIONS, label='Actions')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

# Add annotation
ax.text2D(0.5, 0.02, 'Similar words cluster together in 3D space!',
         transform=ax.transAxes, fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3, pad=0.5))

# Grid and limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/3d_word_space.pdf', dpi=300, bbox_inches='tight')
print("Generated: 3d_word_space.pdf")
plt.close()