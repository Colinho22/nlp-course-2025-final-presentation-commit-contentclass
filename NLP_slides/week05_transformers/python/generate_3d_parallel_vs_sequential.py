"""
Generate 3D visualization comparing parallel vs sequential processing
Shows why transformers are faster than RNNs
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure with two 3D subplots
fig = plt.figure(figsize=(16, 8))

# Colors
COLOR_INACTIVE = '#CCCCCC'
COLOR_ACTIVE = '#FF6B6B'
COLOR_COMPLETE = '#4ECDC4'

# Left: Sequential Processing (RNN)
ax1 = fig.add_subplot(121, projection='3d')

# Create domino-like arrangement
n_words = 10
seq_positions = []
for i in range(n_words):
    x = i * 0.3 - 1.5
    y = 0
    z = 0
    seq_positions.append([x, y, z])

# Show time steps
time_steps = 5  # Show word 5 being processed
for i, pos in enumerate(seq_positions):
    if i < time_steps:
        color = COLOR_COMPLETE
        size = 300
        alpha = 0.9
    elif i == time_steps:
        color = COLOR_ACTIVE
        size = 500
        alpha = 1.0
    else:
        color = COLOR_INACTIVE
        size = 300
        alpha = 0.4

    ax1.scatter(pos[0], pos[1], pos[2], s=size, c=color,
               edgecolors='black', linewidth=2, alpha=alpha)
    ax1.text(pos[0], pos[1], pos[2] + 0.2, f'W{i+1}',
            fontsize=9, ha='center', fontweight='bold')

# Draw arrows showing sequential flow
for i in range(n_words - 1):
    if i < time_steps:
        ax1.plot([seq_positions[i][0] + 0.1, seq_positions[i+1][0] - 0.1],
                [0, 0], [0, 0], 'g-', linewidth=3, alpha=0.8)

# Add time axis
time_z = np.linspace(-0.5, 0.5, time_steps + 1)
for i in range(time_steps + 1):
    ax1.plot([-2, 2], [-0.8, -0.8], [time_z[i], time_z[i]],
            'k--', alpha=0.2)
    ax1.text(-2.2, -0.8, time_z[i], f't={i}',
            fontsize=8, ha='right')

ax1.set_title('Sequential (RNN): One Word at a Time\n' +
             f'Processing word {time_steps+1} of {n_words} (Time step {time_steps+1})',
             fontsize=12, fontweight='bold')
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-1, 1)
ax1.set_zlim(-1, 1)
ax1.view_init(elev=20, azim=-45)
ax1.grid(True, alpha=0.3)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])

# Right: Parallel Processing (Transformer)
ax2 = fig.add_subplot(122, projection='3d')

# Create sphere arrangement for parallel processing
np.random.seed(42)
par_positions = []
for i in range(n_words):
    theta = i * 2 * np.pi / n_words
    x = 0.8 * np.cos(theta)
    y = 0.8 * np.sin(theta)
    z = 0
    par_positions.append([x, y, z])

# All words processed simultaneously
for i, pos in enumerate(par_positions):
    ax2.scatter(pos[0], pos[1], pos[2], s=500, c=COLOR_ACTIVE,
               edgecolors='black', linewidth=2, alpha=1.0)
    ax2.text(pos[0], pos[1], pos[2], f'W{i+1}',
            fontsize=9, ha='center', fontweight='bold', color='white')

# Draw connections showing parallel attention
for i in range(n_words):
    for j in range(i+1, n_words):
        if np.random.random() > 0.7:  # Show some connections
            ax2.plot([par_positions[i][0], par_positions[j][0]],
                    [par_positions[i][1], par_positions[j][1]],
                    [par_positions[i][2], par_positions[j][2]],
                    'y-', alpha=0.3, linewidth=1)

# Add burst effect to show simultaneity
theta_burst = np.linspace(0, 2*np.pi, 20)
for r in [0.3, 0.5, 0.7]:
    x_burst = r * np.cos(theta_burst)
    y_burst = r * np.sin(theta_burst)
    z_burst = np.zeros_like(x_burst)
    ax2.plot(x_burst, y_burst, z_burst, 'gold', alpha=0.3, linewidth=2)

ax2.set_title('Parallel (Transformer): All Words at Once\n' +
             f'Processing all {n_words} words simultaneously (Time step 1)',
             fontsize=12, fontweight='bold')
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_zlim(-1, 1)
ax2.view_init(elev=20, azim=45)
ax2.grid(True, alpha=0.3)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])

# Main title
fig.suptitle('Processing Speed: Sequential vs Parallel',
            fontsize=16, fontweight='bold')

# Add comparison stats
stats_text = """
Sequential (RNN):
• 10 words = 10 time steps
• 100 words = 100 time steps
• GPU Utilization: ~5%
• Training: 90 days

Parallel (Transformer):
• 10 words = 1 time step
• 100 words = 1 time step
• GPU Utilization: ~95%
• Training: 1 day
"""

fig.text(0.5, 0.08, stats_text, fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1),
        family='monospace')

plt.tight_layout()
plt.subplots_adjust(top=0.88, bottom=0.18)
plt.savefig('../figures/3d_parallel_vs_sequential.pdf', dpi=300, bbox_inches='tight')
print("Generated: 3d_parallel_vs_sequential.pdf")
plt.close()