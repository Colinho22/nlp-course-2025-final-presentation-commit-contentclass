import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Define sentence
words = ["The", "cat", "sat", "on", "mat"]
n_words = len(words)

# Create a circular arrangement for better visibility
angles = np.linspace(0, 2*np.pi, n_words, endpoint=False)
radius = 2
x = radius * np.cos(angles)
y = radius * np.sin(angles)
z = np.zeros(n_words)

# Plot words as spheres
colors = plt.cm.Set3(np.linspace(0, 1, n_words))
for i, word in enumerate(words):
    ax.scatter(x[i], y[i], z[i], s=500, c=[colors[i]], alpha=0.8,
              edgecolors='black', linewidth=2)
    ax.text(x[i]*1.2, y[i]*1.2, z[i], word, fontsize=14, ha='center',
           weight='bold')

# Draw ALL connections (n×n matrix)
connection_count = 0
for i in range(n_words):
    for j in range(n_words):
        if i != j:
            # Create curved connections for visual clarity
            t = np.linspace(0, 1, 20)

            # Create an arc that goes up in z-dimension
            arc_x = x[i] * (1-t) + x[j] * t
            arc_y = y[i] * (1-t) + y[j] * t
            arc_z = 2 * t * (1-t) * 1.5  # Parabolic arc

            # Color based on importance (simulate attention weights)
            if (i, j) in [(1, 2), (2, 1), (2, 4), (4, 2)]:  # Important connections
                color = 'red'
                alpha = 0.7
                linewidth = 2
            else:
                color = 'gray'
                alpha = 0.2
                linewidth = 0.5

            ax.plot(arc_x, arc_y, arc_z, color=color, alpha=alpha,
                   linewidth=linewidth)
            connection_count += 1

# Add center point showing parallel processing
ax.scatter(0, 0, 2, s=1000, c='gold', marker='*',
          edgecolors='black', linewidth=3)
ax.text(0, 0, 2.5, 'PARALLEL\nPROCESSING', ha='center',
       fontsize=12, weight='bold', color='darkred')

# Draw rays from center to each word (showing simultaneous processing)
for i in range(n_words):
    ax.plot([0, x[i]], [0, y[i]], [2, z[i]],
           'gold', linewidth=3, alpha=0.5)

# Add connection matrix visualization
matrix_x = -3
matrix_y = -3
matrix_size = 0.4
for i in range(n_words):
    for j in range(n_words):
        # Draw matrix cell
        if i == j:
            color = 'lightgray'
            val = 1.0
        elif (i, j) in [(1, 2), (2, 1), (2, 4), (4, 2)]:
            color = 'red'
            val = 0.8
        else:
            color = 'lightblue'
            val = 0.3

        cube_x = [matrix_x + i*matrix_size, matrix_x + (i+1)*matrix_size]
        cube_y = [matrix_y + j*matrix_size, matrix_y + (j+1)*matrix_size]

        # Draw a small square for each matrix element
        xx, yy = np.meshgrid(cube_x, cube_y)
        zz = np.ones_like(xx) * 0.1
        ax.plot_surface(xx, yy, zz, color=color, alpha=val)

# Labels and styling
ax.set_xlabel('X', fontsize=10)
ax.set_ylabel('Y', fontsize=10)
ax.set_zlabel('Connection Strength', fontsize=10)
ax.set_title(f'Every Word Connects to Every Other: {n_words}×{n_words} = {n_words*n_words} Connections',
            fontsize=14, weight='bold')

# Set viewing angle
ax.view_init(elev=25, azim=45)

# Set limits
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
ax.set_zlim([0, 3])

# Hide axes for cleaner look
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Add annotations
ax.text2D(0.02, 0.95,
         f"Total Connections: {n_words*n_words}\n" +
         f"Self-connections: {n_words}\n" +
         f"Cross-connections: {n_words*(n_words-1)}",
         transform=ax.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

ax.text2D(0.02, 0.02,
         "Key: All connections computed\nSIMULTANEOUSLY in transformers!",
         transform=ax.transAxes, fontsize=11,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9),
         weight='bold')

plt.tight_layout()
plt.savefig('../figures/parallel_connections_3d.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Generated parallel_connections_3d.pdf")