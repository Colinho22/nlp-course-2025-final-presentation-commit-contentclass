import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create figure with subplots
fig = plt.figure(figsize=(12, 5))

# Left panel: 3D vector visualization
ax1 = fig.add_subplot(121, projection='3d')

# Define vectors
vec_cat = np.array([0.7, 0.2, 0.5])
vec_dog = np.array([0.8, 0.3, 0.4])
vec_car = np.array([0.1, 0.9, 0.2])

# Plot vectors from origin
origin = [0, 0, 0]
ax1.quiver(0, 0, 0, vec_cat[0], vec_cat[1], vec_cat[2], color='red', arrow_length_ratio=0.1, linewidth=3, label='cat')
ax1.quiver(0, 0, 0, vec_dog[0], vec_dog[1], vec_dog[2], color='green', arrow_length_ratio=0.1, linewidth=3, label='dog')
ax1.quiver(0, 0, 0, vec_car[0], vec_car[1], vec_car[2], color='blue', arrow_length_ratio=0.1, linewidth=3, label='car')

# Calculate angles
def angle_between(v1, v2):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi

angle_cat_dog = angle_between(vec_cat, vec_dog)
angle_cat_car = angle_between(vec_cat, vec_car)

# Draw angle arc between cat and dog
t = np.linspace(0, 1, 20)
arc_points = []
for ti in t:
    # Interpolate between normalized vectors
    v_norm_cat = vec_cat / np.linalg.norm(vec_cat)
    v_norm_dog = vec_dog / np.linalg.norm(vec_dog)
    interpolated = (1-ti) * v_norm_cat + ti * v_norm_dog
    interpolated = interpolated / np.linalg.norm(interpolated) * 0.3
    arc_points.append(interpolated)
arc_points = np.array(arc_points)
ax1.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], 'orange', linewidth=2, alpha=0.7)

# Labels
ax1.text(vec_cat[0]*1.1, vec_cat[1]*1.1, vec_cat[2]*1.1, 'cat', fontsize=12, weight='bold')
ax1.text(vec_dog[0]*1.1, vec_dog[1]*1.1, vec_dog[2]*1.1, 'dog', fontsize=12, weight='bold')
ax1.text(vec_car[0]*1.1, vec_car[1]*1.1, vec_car[2]*1.1, 'car', fontsize=12, weight='bold')

# Styling
ax1.set_xlabel('X', fontsize=10)
ax1.set_ylabel('Y', fontsize=10)
ax1.set_zlabel('Z', fontsize=10)
ax1.set_title(f'Word Vectors in 3D Space\nAngle(cat,dog)={angle_cat_dog:.1f}°\nAngle(cat,car)={angle_cat_car:.1f}°',
              fontsize=12, weight='bold')

# Set equal aspect ratio
max_range = 1.0
ax1.set_xlim([-0.1, max_range])
ax1.set_ylim([-0.1, max_range])
ax1.set_zlim([-0.1, max_range])

# Right panel: Similarity scores
ax2 = fig.add_subplot(122)

# Calculate dot products
dot_cat_dog = np.dot(vec_cat, vec_dog)
dot_cat_car = np.dot(vec_cat, vec_car)
dot_dog_car = np.dot(vec_dog, vec_car)

# Create similarity matrix
words = ['cat', 'dog', 'car']
similarity_matrix = np.array([
    [1.0, dot_cat_dog, dot_cat_car],
    [dot_cat_dog, 1.0, dot_dog_car],
    [dot_cat_car, dot_dog_car, 1.0]
])

# Plot heatmap
im = ax2.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

# Add colorbar
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('Similarity Score', fontsize=10)

# Add text annotations
for i in range(3):
    for j in range(3):
        text = ax2.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=12, weight='bold')

# Set ticks and labels
ax2.set_xticks(np.arange(3))
ax2.set_yticks(np.arange(3))
ax2.set_xticklabels(words)
ax2.set_yticklabels(words)
ax2.set_title('Dot Product Similarity Matrix', fontsize=12, weight='bold')

# Add formula annotation
ax2.text(1, -0.8, r'similarity = $\vec{A} \cdot \vec{B} = |\vec{A}| \times |\vec{B}| \times cos(\theta)$',
         ha='center', fontsize=11, style='italic')

plt.tight_layout()
plt.savefig('../figures/dot_product_visualization.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Generated dot_product_visualization.pdf")