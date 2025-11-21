"""
Generate 3D vector angles visualization
Shows how dot product measures similarity between Query and Key vectors
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

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

# Create figure with two subplots
fig = plt.figure(figsize=(16, 8))

# Left subplot: Similar vectors (small angle)
ax1 = fig.add_subplot(121, projection='3d')

# Query vector: "What follows 'on the'?"
query1 = np.array([0.8, 0.6, 0.4])
query1 = query1 / np.linalg.norm(query1) * 2  # Normalize and scale

# Key vector: "I am a location word" (similar)
key1 = np.array([0.7, 0.65, 0.3])
key1 = key1 / np.linalg.norm(key1) * 2

# Origin
origin = [0, 0, 0]

# Draw vectors
arrow_q1 = Arrow3D([origin[0], query1[0]], [origin[1], query1[1]], [origin[2], query1[2]],
                   mutation_scale=20, lw=4, arrowstyle='-|>', color='blue', label='Query')
ax1.add_artist(arrow_q1)

arrow_k1 = Arrow3D([origin[0], key1[0]], [origin[1], key1[1]], [origin[2], key1[2]],
                   mutation_scale=20, lw=4, arrowstyle='-|>', color='green', label='Key')
ax1.add_artist(arrow_k1)

# Calculate angle
dot_product1 = np.dot(query1, key1)
angle1 = np.arccos(dot_product1 / (np.linalg.norm(query1) * np.linalg.norm(key1)))
angle1_deg = np.degrees(angle1)

# Draw angle arc
theta = np.linspace(0, angle1, 20)
arc_radius = 0.5
arc_points = []
for t in theta:
    # Rotate query vector towards key vector by angle t
    v = query1 / np.linalg.norm(query1) * arc_radius
    # Simple rotation (approximation for visualization)
    arc_x = v[0] * np.cos(t) + v[1] * np.sin(t) * 0.3
    arc_y = v[1] * np.cos(t) - v[0] * np.sin(t) * 0.3
    arc_z = v[2] * (1 - t/angle1 * 0.2)
    arc_points.append([arc_x, arc_y, arc_z])

arc_points = np.array(arc_points)
ax1.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], 'r-', linewidth=2)

# Labels
ax1.text(query1[0]*1.2, query1[1]*1.2, query1[2]*1.2,
        'Query\n"What follows\n\'on the\'?"', fontsize=10, fontweight='bold', color='blue')
ax1.text(key1[0]*1.2, key1[1]*1.2, key1[2]*1.2,
        'Key\n"I am \'mat\'\n(furniture)"', fontsize=10, fontweight='bold', color='green')

# Add angle annotation
ax1.text(0.3, 0.3, 0.3, f'θ = {angle1_deg:.1f}°', fontsize=12, color='red', fontweight='bold')

# Add similarity score
similarity1 = dot_product1 / 4  # Normalized
ax1.set_title(f'HIGH Similarity\nAngle = {angle1_deg:.1f}°\nDot Product = {dot_product1:.2f}\nSimilarity = {similarity1:.1%}',
             fontsize=12, fontweight='bold', color='green')

# Setup
ax1.set_xlim(-0.5, 2.5)
ax1.set_ylim(-0.5, 2.5)
ax1.set_zlim(-0.5, 2.5)
ax1.set_xlabel('X', fontsize=10)
ax1.set_ylabel('Y', fontsize=10)
ax1.set_zlabel('Z', fontsize=10)
ax1.view_init(elev=20, azim=45)

# Right subplot: Dissimilar vectors (large angle)
ax2 = fig.add_subplot(122, projection='3d')

# Query vector: "What follows 'on the'?"
query2 = np.array([0.8, 0.6, 0.4])
query2 = query2 / np.linalg.norm(query2) * 2

# Key vector: "I am an action word" (dissimilar)
key2 = np.array([-0.2, 0.5, -0.7])
key2 = key2 / np.linalg.norm(key2) * 2

# Draw vectors
arrow_q2 = Arrow3D([origin[0], query2[0]], [origin[1], query2[1]], [origin[2], query2[2]],
                   mutation_scale=20, lw=4, arrowstyle='-|>', color='blue')
ax2.add_artist(arrow_q2)

arrow_k2 = Arrow3D([origin[0], key2[0]], [origin[1], key2[1]], [origin[2], key2[2]],
                   mutation_scale=20, lw=4, arrowstyle='-|>', color='red')
ax2.add_artist(arrow_k2)

# Calculate angle
dot_product2 = np.dot(query2, key2)
angle2 = np.arccos(np.clip(dot_product2 / (np.linalg.norm(query2) * np.linalg.norm(key2)), -1, 1))
angle2_deg = np.degrees(angle2)

# Draw angle arc
theta2 = np.linspace(0, angle2, 30)
arc_points2 = []
for t in theta2:
    # More complex rotation for larger angle
    v = query2 / np.linalg.norm(query2) * arc_radius
    arc_x = v[0] * (1 - t/angle2) + key2[0]/np.linalg.norm(key2) * arc_radius * (t/angle2)
    arc_y = v[1] * (1 - t/angle2) + key2[1]/np.linalg.norm(key2) * arc_radius * (t/angle2)
    arc_z = v[2] * (1 - t/angle2) + key2[2]/np.linalg.norm(key2) * arc_radius * (t/angle2)
    arc_points2.append([arc_x, arc_y, arc_z])

arc_points2 = np.array(arc_points2)
ax2.plot(arc_points2[:, 0], arc_points2[:, 1], arc_points2[:, 2], 'r-', linewidth=2)

# Labels
ax2.text(query2[0]*1.2, query2[1]*1.2, query2[2]*1.2,
        'Query\n"What follows\n\'on the\'?"', fontsize=10, fontweight='bold', color='blue')
ax2.text(key2[0]*1.2, key2[1]*1.2, key2[2]*1.2,
        'Key\n"I am \'jumped\'\n(action)"', fontsize=10, fontweight='bold', color='red')

# Add angle annotation
ax2.text(0.2, 0.2, 0, f'θ = {angle2_deg:.1f}°', fontsize=12, color='red', fontweight='bold')

# Add similarity score
similarity2 = max(0, dot_product2) / 4  # Normalized, no negative
ax2.set_title(f'LOW Similarity\nAngle = {angle2_deg:.1f}°\nDot Product = {dot_product2:.2f}\nSimilarity = {similarity2:.1%}',
             fontsize=12, fontweight='bold', color='red')

# Setup
ax2.set_xlim(-2, 2.5)
ax2.set_ylim(-2, 2.5)
ax2.set_zlim(-2, 2.5)
ax2.set_xlabel('X', fontsize=10)
ax2.set_ylabel('Y', fontsize=10)
ax2.set_zlabel('Z', fontsize=10)
ax2.view_init(elev=20, azim=45)

# Main title
fig.suptitle('Dot Product Measures Similarity: cos(angle) × magnitude',
            fontsize=16, fontweight='bold')

# Add formula box
fig.text(0.5, 0.15,
        'Attention Score = Q · K = |Q| × |K| × cos(θ)\n'
        'Small angle → High dot product → Strong attention\n'
        'Large angle → Low dot product → Weak attention',
        fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Bottom note
fig.text(0.5, 0.05,
        'In practice: 512-dimensional vectors, but same principle applies!',
        fontsize=10, ha='center', style='italic',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.subplots_adjust(top=0.88, bottom=0.22)
plt.savefig('../figures/vector_angles_3d.pdf', dpi=300, bbox_inches='tight')
print("Generated: vector_angles_3d.pdf")
plt.close()