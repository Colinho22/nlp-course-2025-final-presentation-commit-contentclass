import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create figure
fig = plt.figure(figsize=(12, 10))

# Define sentence
sentence = "The cat sat on mat"
words = sentence.split()
n_words = len(words)
positions = np.arange(n_words)

# Define 4 attention heads with different focus patterns
heads_info = [
    {"name": "Grammar Expert", "color": "blue", "focus": [0, 1, 2, 0, 0]},  # Focus on subject-verb
    {"name": "Meaning Expert", "color": "green", "focus": [0, 1, 0, 0, 1]},  # Focus on cat-mat relationship
    {"name": "Position Expert", "color": "orange", "focus": [1, 0, 0, 1, 0]},  # Focus on position words
    {"name": "Context Expert", "color": "purple", "focus": [0.2, 0.2, 0.2, 0.2, 0.2]}  # Uniform attention
]

# Create 4 subplots
for idx, head_info in enumerate(heads_info):
    ax = fig.add_subplot(2, 2, idx + 1, projection='3d')

    # Normalize attention weights
    attention = np.array(head_info["focus"])
    attention = attention / attention.sum()

    # Create word positions
    x = positions
    y = np.zeros(n_words)
    z = np.zeros(n_words)

    # Plot words as spheres
    for i, word in enumerate(words):
        # Size based on attention
        size = 100 + attention[i] * 1000
        alpha = 0.3 + attention[i] * 0.7

        ax.scatter(x[i], y[i], z[i], s=size, c=head_info["color"],
                  alpha=alpha, edgecolors='black', linewidth=1)
        ax.text(x[i], y[i], z[i]-0.5, word, fontsize=10, ha='center')

    # Draw attention connections from center point
    center_x = n_words / 2 - 0.5
    center_y = 0
    center_z = 2

    # Draw rays to each word
    for i in range(n_words):
        linewidth = 0.5 + attention[i] * 5
        alpha = 0.2 + attention[i] * 0.8
        ax.plot([center_x, x[i]], [center_y, y[i]], [center_z, z[i]],
                color=head_info["color"], linewidth=linewidth, alpha=alpha)

    # Add spotlight source
    ax.scatter(center_x, center_y, center_z, s=200, c='gold',
              marker='*', edgecolors='black', linewidth=2)

    # Styling
    ax.set_title(f'Head {idx+1}: {head_info["name"]}',
                fontsize=12, weight='bold', color=head_info["color"])
    ax.set_xlabel('Word Position', fontsize=9)
    ax.set_zlabel('Attention', fontsize=9)

    # Remove y-axis
    ax.set_yticks([])
    ax.set_xticks(range(n_words))
    ax.set_xticklabels(range(n_words))

    # Set viewing angle
    ax.view_init(elev=15, azim=30)

    # Set limits
    ax.set_xlim(-0.5, n_words-0.5)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 3)

    # Add attention values as text
    attention_text = ', '.join([f'{a:.2f}' for a in attention])
    ax.text2D(0.02, 0.02, f'Weights: [{attention_text}]',
             transform=ax.transAxes, fontsize=8)

# Main title
fig.suptitle('Multi-Head Attention: 4 Different Perspectives',
            fontsize=14, weight='bold', y=0.98)

# Add description box
fig.text(0.5, 0.02,
        'Each head learns different attention patterns:\n' +
        '• Grammar: Subject-verb relationships\n' +
        '• Meaning: Semantic connections\n' +
        '• Position: Structural patterns\n' +
        '• Context: Global understanding',
        ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout(rect=[0, 0.08, 1, 0.96])
plt.savefig('../figures/multi_head_3d.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Generated multi_head_3d.pdf")