"""
Generate 3D exponential explosion visualization
Shows how connections grow quadratically with word count
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Define word counts and their connections
word_counts = np.array([5, 10, 20, 50, 100, 200])
connections = word_counts * (word_counts - 1) / 2  # n(n-1)/2 formula

# Create growing cubes to represent computational complexity
for i, (words, conn) in enumerate(zip(word_counts, connections)):
    # Position cubes along diagonal
    x_pos = i * 2
    y_pos = i * 2
    z_pos = 0

    # Size proportional to cube root of connections (for visibility)
    size = np.cbrt(conn) / 10

    # Color gradient from blue to red
    color_intensity = i / len(word_counts)
    color = plt.cm.RdYlBu_r(color_intensity)

    # Draw cube
    # Define vertices
    vertices = [
        [x_pos, y_pos, z_pos],
        [x_pos + size, y_pos, z_pos],
        [x_pos + size, y_pos + size, z_pos],
        [x_pos, y_pos + size, z_pos],
        [x_pos, y_pos, z_pos + size],
        [x_pos + size, y_pos, z_pos + size],
        [x_pos + size, y_pos + size, z_pos + size],
        [x_pos, y_pos + size, z_pos + size]
    ]

    # Define the 6 faces
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]]
    ]

    # Plot faces
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    poly = Poly3DCollection(faces, alpha=0.7, facecolor=color, edgecolor='black', linewidth=1)
    ax.add_collection3d(poly)

    # Add labels
    ax.text(x_pos + size/2, y_pos + size/2, z_pos + size + 0.5,
           f'{int(words)} words\n{int(conn):,} connections',
           fontsize=9, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add explosion lines for larger cubes
    if i >= 3:  # Start showing "explosion" for 50+ words
        for _ in range(int(5 + i * 2)):
            # Random explosion rays
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)

            start_x = x_pos + size/2
            start_y = y_pos + size/2
            start_z = z_pos + size/2

            length = size * np.random.uniform(0.5, 1.5)
            end_x = start_x + length * np.sin(phi) * np.cos(theta)
            end_y = start_y + length * np.sin(phi) * np.sin(theta)
            end_z = start_z + length * np.cos(phi)

            ax.plot([start_x, end_x], [start_y, end_y], [start_z, end_z],
                   color='orange', alpha=0.3, linewidth=1)

# Add exponential curve in 3D
x_curve = np.arange(len(word_counts))
y_curve = x_curve * 2
z_curve = np.log10(connections + 1) * 2  # Log scale for visibility
ax.plot(x_curve * 2, y_curve, z_curve, 'r--', linewidth=2, alpha=0.5, label='Exponential Growth')

# Add warning zones
ax.text(8, 8, 10, 'MANAGEABLE', fontsize=11, color='green', fontweight='bold')
ax.text(10, 10, 12, 'CHALLENGING', fontsize=11, color='orange', fontweight='bold')
ax.text(12, 12, 14, 'EXPLOSION!', fontsize=12, color='red', fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Set labels and title
ax.set_xlabel('Sequence Length', fontsize=11, fontweight='bold')
ax.set_ylabel('Computational Load', fontsize=11, fontweight='bold')
ax.set_zlabel('Memory Required', fontsize=11, fontweight='bold')
ax.set_title('The Quadratic Explosion Problem\nConnections = n(n-1)/2',
            fontsize=14, fontweight='bold')

# Set viewing angle
ax.view_init(elev=20, azim=45)

# Set limits
ax.set_xlim(-1, 14)
ax.set_ylim(-1, 14)
ax.set_zlim(0, 15)

# Add legend with statistics
legend_text = [
    f'5 words = 10 connections',
    f'10 words = 45 connections',
    f'100 words = 4,950 connections',
    f'1000 words = 499,500 connections!'
]
for i, text in enumerate(legend_text):
    ax.text2D(0.02, 0.98 - i*0.05, text, transform=ax.transAxes,
             fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Add bottom annotation
ax.text2D(0.5, 0.02,
         'Each word must attend to every other word - complexity explodes!',
         transform=ax.transAxes, fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('../figures/exponential_explosion_3d.pdf', dpi=300, bbox_inches='tight')
print("Generated: exponential_explosion_3d.pdf")
plt.close()