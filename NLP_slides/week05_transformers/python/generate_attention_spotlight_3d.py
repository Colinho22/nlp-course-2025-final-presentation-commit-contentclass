import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define sentence words
words = ["The", "cat", "sat", "on", "the", "mat"]
n_words = len(words)

# Create grid for words
x = np.arange(n_words)
y = np.zeros(n_words)
z = np.zeros(n_words)

# Plot words as spheres at base
for i, word in enumerate(words):
    ax.scatter(x[i], y[i], z[i], s=300, c='lightblue', alpha=0.6, edgecolors='navy', linewidth=2)
    ax.text(x[i], y[i]-0.3, z[i], word, fontsize=12, ha='center', weight='bold')

# Create spotlight effect from above
# Focus on "cat" and "sat" (indices 1 and 2)
spotlight_center = 1.5  # Between cat and sat
spotlight_radius = 1.5

# Create cone for spotlight
theta = np.linspace(0, 2*np.pi, 50)
height = np.linspace(0, 3, 20)
THETA, HEIGHT = np.meshgrid(theta, height)

# Varying radius from top to bottom
R = HEIGHT * spotlight_radius / 3

# Cylindrical to Cartesian
X_cone = R * np.cos(THETA) + spotlight_center
Y_cone = R * np.sin(THETA)
Z_cone = 3 - HEIGHT

# Plot spotlight cone with gradient
colors = plt.cm.YlOrRd(HEIGHT/3)
ax.plot_surface(X_cone, Y_cone, Z_cone, alpha=0.3, facecolors=colors, edgecolor='none')

# Add attention rays
for i in [1, 2]:  # Focus on cat and sat
    ax.plot([spotlight_center, x[i]], [0, y[i]], [3, z[i]],
            'r-', linewidth=3, alpha=0.8)
    # Add glow effect
    ax.scatter(x[i], y[i], z[i], s=500, c='yellow', alpha=0.5)

# Add dim rays for other words
for i in [0, 3, 4, 5]:
    ax.plot([spotlight_center, x[i]], [0, y[i]], [3, z[i]],
            'gray', linewidth=0.5, alpha=0.3, linestyle='--')

# Labels and styling
ax.set_xlabel('Word Position', fontsize=12)
ax.set_zlabel('Attention Focus', fontsize=12)
ax.set_title('Selective Attention: Spotlight on Important Words', fontsize=14, weight='bold')

# Remove y-axis for cleaner look
ax.set_yticks([])
ax.set_xticks(range(n_words))

# Set viewing angle
ax.view_init(elev=20, azim=45)

# Set limits
ax.set_xlim(-0.5, n_words-0.5)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 3.5)

# Add annotation
ax.text2D(0.05, 0.95, "Bright spotlight = High attention\nDim areas = Low attention",
          transform=ax.transAxes, fontsize=11, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('../figures/attention_spotlight_3d.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Generated attention_spotlight_3d.pdf")