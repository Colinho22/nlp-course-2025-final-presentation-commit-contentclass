import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Define sentence with ambiguous words
words = ["The", "bank", "by", "the", "river", "bank"]
n_words = len(words)

# Position words in 3D space
x = np.arange(n_words)
y = np.zeros(n_words)
z = np.zeros(n_words)

# Define word meanings (for coloring)
word_colors = {
    0: 'gray',      # The (1)
    1: 'red',       # bank (financial)
    2: 'gray',      # by
    3: 'gray',      # the (2)
    4: 'blue',      # river
    5: 'green'      # bank (riverbank)
}

word_sizes = {
    0: 200,
    1: 500,
    2: 200,
    3: 200,
    4: 400,
    5: 500
}

# Plot words
for i, word in enumerate(words):
    ax.scatter(x[i], y[i], z[i], s=word_sizes[i], c=word_colors[i],
              alpha=0.7, edgecolors='black', linewidth=2)
    # Add word labels with meaning hints
    if i == 1:
        label = f'{word}\n(money?)'
    elif i == 5:
        label = f'{word}\n(shore?)'
    else:
        label = word
    ax.text(x[i], y[i], z[i]-0.5, label, fontsize=11, ha='center', weight='bold')

# Create explosion of connections (information overload)
np.random.seed(42)
n_connections = 50  # Many more than needed

for _ in range(n_connections):
    i, j = np.random.choice(n_words, 2, replace=False)

    # Random noise connections
    t = np.linspace(0, 1, 10)
    # Add random perturbation to make it look chaotic
    noise_x = np.random.normal(0, 0.1, 10)
    noise_y = np.random.normal(0, 0.1, 10)
    noise_z = np.random.normal(0, 0.1, 10)

    arc_x = x[i] * (1-t) + x[j] * t + noise_x
    arc_y = y[i] * (1-t) + y[j] * t + noise_y
    arc_z = 1.5 * t * (1-t) + noise_z

    # Most connections are noise (gray and thin)
    if np.random.random() < 0.8:
        ax.plot(arc_x, arc_y, arc_z, color='gray', alpha=0.2, linewidth=0.5)
    else:
        # Some important connections (but hard to find)
        ax.plot(arc_x, arc_y, arc_z, color='orange', alpha=0.4, linewidth=1)

# Add important connections (hidden in the noise)
# bank1 to river (context clue)
t = np.linspace(0, 1, 20)
arc_x = x[1] * (1-t) + x[4] * t
arc_y = y[1] * (1-t) + y[4] * t
arc_z = 2 * t * (1-t)
ax.plot(arc_x, arc_y, arc_z, color='red', alpha=0.8, linewidth=3)

# river to bank2 (context clue)
arc_x = x[4] * (1-t) + x[5] * t
arc_y = y[4] * (1-t) + y[5] * t
arc_z = 2 * t * (1-t)
ax.plot(arc_x, arc_y, arc_z, color='green', alpha=0.8, linewidth=3)

# Add noise cloud (representing confusion)
n_noise = 200
noise_x = np.random.normal(n_words/2, 2, n_noise)
noise_y = np.random.normal(0, 1, n_noise)
noise_z = np.random.normal(2, 0.5, n_noise)
ax.scatter(noise_x, noise_y, noise_z, s=5, c='gray', alpha=0.1)

# Add warning symbols
ax.text(n_words/2, 0, 4, '⚠ INFORMATION\nOVERLOAD!', ha='center',
       fontsize=16, weight='bold', color='darkred',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Add statistics
ax.text2D(0.02, 0.95,
         f"Words: {n_words}\n" +
         f"Possible connections: {n_words*(n_words-1)//2}\n" +
         f"Relevant connections: ~3\n" +
         f"Noise ratio: 90%+",
         transform=ax.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

# Add problem explanation
ax.text2D(0.7, 0.02,
         "Problem:\n" +
         "• Which 'bank'?\n" +
         "• Too many signals\n" +
         "• Important info lost",
         transform=ax.transAxes, fontsize=11,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

# Styling
ax.set_xlabel('Word Position', fontsize=12)
ax.set_ylabel('', fontsize=12)
ax.set_zlabel('Connection Activity', fontsize=12)
ax.set_title('Information Overload: Signal Lost in Noise', fontsize=14, weight='bold', color='darkred')

# Set viewing angle
ax.view_init(elev=20, azim=-60)

# Set limits
ax.set_xlim([-1, n_words])
ax.set_ylim([-3, 3])
ax.set_zlim([0, 5])

# Grid settings
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/information_overload_3d.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Generated information_overload_3d.pdf")