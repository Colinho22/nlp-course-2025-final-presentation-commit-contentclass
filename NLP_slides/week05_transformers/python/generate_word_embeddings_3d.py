import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Define words and their embeddings (semantic clusters)
animals = {
    'cat': [0.7, 0.2, 0.5],
    'dog': [0.8, 0.3, 0.4],
    'bird': [0.6, 0.15, 0.6],
    'fish': [0.65, 0.25, 0.3]
}

vehicles = {
    'car': [0.1, 0.9, 0.2],
    'truck': [0.15, 0.85, 0.25],
    'bike': [0.05, 0.95, 0.15],
    'plane': [0.2, 0.8, 0.3]
}

food = {
    'apple': [0.4, 0.5, 0.8],
    'bread': [0.45, 0.45, 0.85],
    'milk': [0.35, 0.55, 0.75],
    'cake': [0.42, 0.48, 0.9]
}

# Plot animal cluster
for word, coords in animals.items():
    ax.scatter(coords[0], coords[1], coords[2], c='red', s=200, alpha=0.7, edgecolors='darkred', linewidth=2)
    ax.text(coords[0]+0.02, coords[1]+0.02, coords[2]+0.02, word, fontsize=11, weight='bold', color='darkred')

# Draw connections within animal cluster
animal_coords = list(animals.values())
for i in range(len(animal_coords)):
    for j in range(i+1, len(animal_coords)):
        ax.plot([animal_coords[i][0], animal_coords[j][0]],
               [animal_coords[i][1], animal_coords[j][1]],
               [animal_coords[i][2], animal_coords[j][2]],
               'r-', alpha=0.2, linewidth=1)

# Plot vehicle cluster
for word, coords in vehicles.items():
    ax.scatter(coords[0], coords[1], coords[2], c='blue', s=200, alpha=0.7, edgecolors='darkblue', linewidth=2)
    ax.text(coords[0]+0.02, coords[1]+0.02, coords[2]+0.02, word, fontsize=11, weight='bold', color='darkblue')

# Draw connections within vehicle cluster
vehicle_coords = list(vehicles.values())
for i in range(len(vehicle_coords)):
    for j in range(i+1, len(vehicle_coords)):
        ax.plot([vehicle_coords[i][0], vehicle_coords[j][0]],
               [vehicle_coords[i][1], vehicle_coords[j][1]],
               [vehicle_coords[i][2], vehicle_coords[j][2]],
               'b-', alpha=0.2, linewidth=1)

# Plot food cluster
for word, coords in food.items():
    ax.scatter(coords[0], coords[1], coords[2], c='green', s=200, alpha=0.7, edgecolors='darkgreen', linewidth=2)
    ax.text(coords[0]+0.02, coords[1]+0.02, coords[2]+0.02, word, fontsize=11, weight='bold', color='darkgreen')

# Draw connections within food cluster
food_coords = list(food.values())
for i in range(len(food_coords)):
    for j in range(i+1, len(food_coords)):
        ax.plot([food_coords[i][0], food_coords[j][0]],
               [food_coords[i][1], food_coords[j][1]],
               [food_coords[i][2], food_coords[j][2]],
               'g-', alpha=0.2, linewidth=1)

# Add cluster labels with transparent spheres
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Calculate cluster centers
animal_center = np.mean(list(animals.values()), axis=0)
vehicle_center = np.mean(list(vehicles.values()), axis=0)
food_center = np.mean(list(food.values()), axis=0)

# Add cluster center markers
ax.scatter(*animal_center, c='red', s=500, alpha=0.1, marker='o')
ax.scatter(*vehicle_center, c='blue', s=500, alpha=0.1, marker='o')
ax.scatter(*food_center, c='green', s=500, alpha=0.1, marker='o')

# Add arrows showing similarity
# Cat to Dog (similar)
ax.quiver(animals['cat'][0], animals['cat'][1], animals['cat'][2],
         animals['dog'][0]-animals['cat'][0],
         animals['dog'][1]-animals['cat'][1],
         animals['dog'][2]-animals['cat'][2],
         color='orange', alpha=0.6, arrow_length_ratio=0.2, linewidth=2)

# Cat to Car (different)
ax.quiver(animals['cat'][0], animals['cat'][1], animals['cat'][2],
         vehicles['car'][0]-animals['cat'][0],
         vehicles['car'][1]-animals['cat'][1],
         vehicles['car'][2]-animals['cat'][2],
         color='gray', alpha=0.3, arrow_length_ratio=0.1, linewidth=1, linestyle='--')

# Styling
ax.set_xlabel('Dimension 1: Living vs Non-living', fontsize=12, weight='bold')
ax.set_ylabel('Dimension 2: Size/Speed', fontsize=12, weight='bold')
ax.set_zlabel('Dimension 3: Edibility/Function', fontsize=12, weight='bold')
ax.set_title('Word Embeddings: Similar Words Cluster in 3D Space', fontsize=14, weight='bold')

# Set limits
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

# Set viewing angle
ax.view_init(elev=20, azim=45)

# Add legend
legend_elements = [
    mpatches.Patch(color='red', alpha=0.7, label='Animals (similar meanings)'),
    mpatches.Patch(color='blue', alpha=0.7, label='Vehicles (similar meanings)'),
    mpatches.Patch(color='green', alpha=0.7, label='Food (similar meanings)')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

# Add annotation box
ax.text2D(0.02, 0.15,
         "Key Insight:\n" +
         "• Similar words are nearby\n" +
         "• Different categories are far apart\n" +
         "• Distance = Semantic difference",
         transform=ax.transAxes, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

plt.tight_layout()
plt.savefig('../figures/word_embeddings_3d.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Generated word_embeddings_3d.pdf")