"""
Generate 3D positional encoding waves visualization
Shows how sine/cosine waves create unique position signatures
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure
fig = plt.figure(figsize=(14, 10))

# Main 3D plot
ax = fig.add_subplot(111, projection='3d')

# Parameters
max_position = 50  # Maximum sequence position
n_dimensions = 8   # Number of encoding dimensions to show
positions = np.arange(0, max_position)

# Colors for different frequency waves
colors = plt.cm.coolwarm(np.linspace(0, 1, n_dimensions))

# Generate positional encoding waves
for dim in range(n_dimensions):
    if dim % 2 == 0:
        # Sine wave for even dimensions
        wavelength = 10000 ** (dim / n_dimensions)
        encoding = np.sin(positions / wavelength)
        wave_type = 'sin'
    else:
        # Cosine wave for odd dimensions
        wavelength = 10000 ** ((dim-1) / n_dimensions)
        encoding = np.cos(positions / wavelength)
        wave_type = 'cos'

    # Create 3D wave
    x = positions
    y = np.full_like(positions, dim * 0.5)  # Spread dimensions along y
    z = encoding * 2  # Scale encoding for visibility

    # Plot wave with varying thickness based on frequency
    linewidth = 3 - dim * 0.3
    ax.plot(x, y, z, color=colors[dim], linewidth=linewidth,
           label=f'Dim {dim}: {wave_type}(pos/{wavelength:.0f})',
           alpha=0.8)

    # Add markers at key positions
    key_positions = [0, 10, 25, 49]
    for kp in key_positions:
        if kp < len(positions):
            ax.scatter(x[kp], y[kp], z[kp], s=50, color=colors[dim],
                      edgecolors='black', linewidth=1, alpha=0.9)

# Highlight specific positions with vertical planes
highlight_positions = [0, 10, 25, 49]
for hp in highlight_positions:
    if hp < max_position:
        # Create vertical plane
        y_plane = np.linspace(-0.5, n_dimensions * 0.5, 10)
        z_plane = np.linspace(-2.5, 2.5, 10)
        Y_plane, Z_plane = np.meshgrid(y_plane, z_plane)
        X_plane = np.full_like(Y_plane, hp)

        ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.1,
                       color='gray', edgecolor='none')

        # Add position label
        ax.text(hp, -1, 3, f'Pos {hp}', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add word examples at specific positions
words = ['The', 'transformer', 'model', 'uses']
word_positions = [0, 10, 20, 30]
for word, pos in zip(words, word_positions):
    if pos < max_position:
        ax.text(pos, n_dimensions * 0.5 + 0.5, 0, word,
               fontsize=11, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Create unique encoding visualization for position 25
pos_25_encoding = []
for dim in range(n_dimensions):
    if dim % 2 == 0:
        wavelength = 10000 ** (dim / n_dimensions)
        value = np.sin(25 / wavelength)
    else:
        wavelength = 10000 ** ((dim-1) / n_dimensions)
        value = np.cos(25 / wavelength)
    pos_25_encoding.append(value)

# Draw encoding vector for position 25
start_x = 25
for i, val in enumerate(pos_25_encoding):
    ax.plot([start_x, start_x], [i * 0.5, i * 0.5], [0, val * 2],
           'k-', linewidth=2, alpha=0.5)
    ax.scatter(start_x, i * 0.5, val * 2, s=100, c='gold',
              edgecolors='black', linewidth=2)

# Labels and title
ax.set_xlabel('Position in Sequence', fontsize=12, fontweight='bold')
ax.set_ylabel('Encoding Dimension', fontsize=12, fontweight='bold')
ax.set_zlabel('Encoding Value', fontsize=12, fontweight='bold')
ax.set_title('Positional Encoding: Unique Wave Patterns for Each Position\n' +
            'Low frequencies capture global position, High frequencies capture local patterns',
            fontsize=14, fontweight='bold')

# Set viewing angle
ax.view_init(elev=20, azim=-45)

# Set limits
ax.set_xlim(0, max_position)
ax.set_ylim(-1, n_dimensions * 0.5 + 1)
ax.set_zlim(-3, 3)

# Add legend with frequency information
legend_elements = [
    mpatches.Patch(color=colors[0], label='Low frequency (global)'),
    mpatches.Patch(color=colors[3], label='Medium frequency'),
    mpatches.Patch(color=colors[7], label='High frequency (local)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Add explanation boxes
ax.text2D(0.02, 0.15,
         'Key Properties:\n'
         '• Each position gets unique pattern\n'
         '• Patterns are deterministic\n'
         '• No learned parameters\n'
         '• Works for any sequence length',
         transform=ax.transAxes, fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

ax.text2D(0.75, 0.15,
         'Position 25 encoding:\n' +
         '\n'.join([f'Dim {i}: {val:+.2f}' for i, val in enumerate(pos_25_encoding[:4])]) +
         '\n...',
         transform=ax.transAxes, fontsize=8, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Bottom annotation
ax.text2D(0.5, 0.02,
         'Sine and cosine waves at different frequencies create a unique "fingerprint" for each position',
         transform=ax.transAxes, fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('../figures/positional_waves_3d.pdf', dpi=300, bbox_inches='tight')
print("Generated: positional_waves_3d.pdf")
plt.close()