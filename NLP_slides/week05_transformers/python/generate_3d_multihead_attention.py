"""
Generate 3D multi-head attention visualization
Shows 4 different attention heads seeing different patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure with 4 3D subplots
fig = plt.figure(figsize=(16, 12))

# Colors for each head
HEAD_COLORS = ['#4472C4', '#44A044', '#D62728', '#8B5A9B']  # Blue, Green, Red, Purple
HEAD_NAMES = ['Grammar Head', 'Semantic Head', 'Position Head', 'Global Head']
HEAD_FOCUS = [
    'Focuses on articles and prepositions',
    'Focuses on meaning relationships',
    'Focuses on nearby words',
    'Focuses on sentence boundaries'
]

# Sentence for all heads
sentence = ['The', 'cat', 'sat', 'on', 'the', 'mat']
n_words = len(sentence)

# Define different attention patterns for each head
attention_patterns = [
    # Head 1 (Grammar): Focus on "The" and "on"
    [0.30, 0.10, 0.10, 0.30, 0.15, 0.05],
    # Head 2 (Semantic): Focus on "cat" and "mat"
    [0.05, 0.35, 0.15, 0.10, 0.05, 0.30],
    # Head 3 (Position): Focus on nearby words (for "on")
    [0.05, 0.10, 0.35, 0.00, 0.35, 0.15],
    # Head 4 (Global): Focus on boundaries
    [0.35, 0.10, 0.10, 0.10, 0.10, 0.25]
]

# Create 4 subplots
for head_idx in range(4):
    ax = fig.add_subplot(2, 2, head_idx + 1, projection='3d')

    # Position words in a semi-circle
    positions = []
    for i in range(n_words):
        angle = i * np.pi / (n_words - 1) - np.pi/2
        x = 0.5 * np.cos(angle)
        y = 0.5 * np.sin(angle)
        z = 0
        positions.append([x, y, z])

    # Get attention weights for this head
    weights = attention_patterns[head_idx]

    # Plot words with size based on attention
    for i, (word, pos, weight) in enumerate(zip(sentence, positions, weights)):
        size = 200 + weight * 1500
        color = HEAD_COLORS[head_idx]
        alpha = 0.3 + weight * 0.7

        ax.scatter(pos[0], pos[1], pos[2], s=size, c=color,
                  edgecolors='black', linewidth=2, alpha=alpha)
        ax.text(pos[0], pos[1], pos[2] - 0.15, word,
               fontsize=8 + weight*10, ha='center', fontweight='bold')

        # Add attention percentage if significant
        if weight > 0.15:
            ax.text(pos[0], pos[1], pos[2] + 0.15, f'{weight:.0%}',
                   fontsize=9, ha='center', color='darkred')

    # Draw attention connections (from center query)
    query_pos = [0, 0, 0.5]
    ax.scatter(query_pos[0], query_pos[1], query_pos[2],
              s=400, c='gold', marker='*', edgecolors='black', linewidth=2)

    for i, (pos, weight) in enumerate(zip(positions, weights)):
        if weight > 0.10:
            ax.plot([query_pos[0], pos[0]],
                   [query_pos[1], pos[1]],
                   [query_pos[2], pos[2]],
                   color=HEAD_COLORS[head_idx], alpha=weight, linewidth=weight*8)

    # Set title and labels
    ax.set_title(f'{HEAD_NAMES[head_idx]}\n{HEAD_FOCUS[head_idx]}',
                fontsize=11, fontweight='bold', color=HEAD_COLORS[head_idx])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-0.5, 1)
    ax.view_init(elev=25, azim=45 + head_idx*20)  # Different viewing angle for each
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Add attention distribution bar chart
    ax2 = fig.add_axes([0.08 + (head_idx % 2) * 0.48,
                        0.42 - (head_idx // 2) * 0.47,
                        0.12, 0.08])
    ax2.bar(range(n_words), weights, color=HEAD_COLORS[head_idx], alpha=0.7)
    ax2.set_ylim(0, 0.4)
    ax2.set_xticks(range(n_words))
    ax2.set_xticklabels(sentence, fontsize=7, rotation=45)
    ax2.set_ylabel('Attention', fontsize=8)
    ax2.set_title('Distribution', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

# Main title
fig.suptitle('Multi-Head Attention: Four Different Perspectives on Same Sentence',
            fontsize=16, fontweight='bold')

# Add explanation
fig.text(0.5, 0.02,
        'Each head learns to focus on different linguistic patterns - combined they capture all aspects!',
        fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3, pad=0.5))

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.08)
plt.savefig('../figures/3d_multihead_attention.pdf', dpi=300, bbox_inches='tight')
print("Generated: 3d_multihead_attention.pdf")
plt.close()