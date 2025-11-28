"""
Generate attention heatmap visualization for human reading experiment
Shows how humans selectively focus on certain words when reading
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Sentence and attention weights
sentence = "The black cat sat on the soft red mat"
words = sentence.split()
target_word = "mat"

# Attention weights when reading "mat"
attention_weights = [0.05, 0.05, 0.15, 0.20, 0.35, 0.25, 0.05, 0.05, 0.00]

# Top subplot: Visual highlighting of sentence
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 3)
ax1.axis('off')

# Title
ax1.text(5, 2.5, 'When your eyes reach "mat", your brain focuses on:',
         fontsize=16, fontweight='bold', ha='center')

# Display words with varying size and opacity based on attention
x_positions = np.linspace(1, 9, len(words))
for i, (word, weight, x_pos) in enumerate(zip(words, attention_weights, x_positions)):
    # Determine color based on attention level
    if weight >= 0.30:
        color = '#FF6B6B'  # Red for high attention
        alpha = 0.9
    elif weight >= 0.20:
        color = '#FFE66D'  # Yellow for medium attention
        alpha = 0.7
    elif weight >= 0.10:
        color = '#4ECDC4'  # Teal for low attention
        alpha = 0.5
    else:
        color = '#E0E0E0'  # Gray for very low attention
        alpha = 0.3

    # Size based on attention
    fontsize = 14 + weight * 40

    # Draw background box
    if i == len(words) - 1:  # Target word "mat"
        bbox = FancyBboxPatch((x_pos - 0.3, 0.8), 0.6, 0.8,
                              boxstyle="round,pad=0.05",
                              facecolor='gold', alpha=0.3,
                              edgecolor='black', linewidth=2)
        ax1.add_patch(bbox)

    # Draw word with attention-based styling
    ax1.text(x_pos, 1.2, word, fontsize=fontsize,
            color=color, alpha=alpha, fontweight='bold',
            ha='center', va='center')

    # Add percentage below important words
    if weight >= 0.15:
        ax1.text(x_pos, 0.6, f'{weight:.0%}',
                fontsize=10, color=color, alpha=0.8,
                ha='center', va='center')

# Add arrows showing attention flow
for i, (weight, x_pos) in enumerate(zip(attention_weights[:-1], x_positions[:-1])):
    if weight >= 0.20:
        # Draw arrow from word to "mat"
        ax1.annotate('', xy=(x_positions[-1], 1.0), xytext=(x_pos, 1.0),
                    arrowprops=dict(arrowstyle='->', lw=weight*5,
                                  color='orange', alpha=0.6))

# Bottom subplot: Attention matrix heatmap
ax2.set_title('Attention Pattern: Where Each Word Looks', fontsize=14, fontweight='bold')

# Create attention matrix (simplified for visualization)
n_words = len(words)
attention_matrix = np.zeros((n_words, n_words))

# Fill with example attention patterns
# Each row represents where that word attends to
patterns = [
    [0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # "The" looks mostly at itself
    [0.1, 0.6, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # "black" looks at itself and nearby
    [0.1, 0.2, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],  # "cat"
    [0.0, 0.1, 0.3, 0.4, 0.1, 0.1, 0.0, 0.0, 0.0],  # "sat"
    [0.0, 0.0, 0.1, 0.2, 0.5, 0.2, 0.0, 0.0, 0.0],  # "on"
    [0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.1, 0.0, 0.0],  # "the"
    [0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.6, 0.1, 0.0],  # "soft"
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.6, 0.1],  # "red"
    attention_weights  # "mat" looks back at relevant words
]

attention_matrix = np.array(patterns)

# Create heatmap
im = ax2.imshow(attention_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.7)

# Set ticks and labels
ax2.set_xticks(np.arange(n_words))
ax2.set_yticks(np.arange(n_words))
ax2.set_xticklabels(words, rotation=45, ha='right')
ax2.set_yticklabels(words)

# Add grid
ax2.set_xticks(np.arange(n_words + 1) - 0.5, minor=True)
ax2.set_yticks(np.arange(n_words + 1) - 0.5, minor=True)
ax2.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

# Highlight the last row (attention for "mat")
rect = patches.Rectangle((- 0.5, n_words - 1.5), n_words, 1,
                         linewidth=3, edgecolor='gold', facecolor='none')
ax2.add_patch(rect)

# Add colorbar
cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Attention Weight', rotation=270, labelpad=15)

# Add text annotations for high attention values
for i in range(n_words):
    for j in range(n_words):
        if attention_matrix[i, j] >= 0.3:
            text = ax2.text(j, i, f'{attention_matrix[i, j]:.2f}',
                          ha='center', va='center', color='white', fontweight='bold')
        elif attention_matrix[i, j] >= 0.15:
            text = ax2.text(j, i, f'{attention_matrix[i, j]:.2f}',
                          ha='center', va='center', color='black')

# Labels
ax2.set_xlabel('Attended To (Keys)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Attending From (Queries)', fontsize=12, fontweight='bold')

# Add annotation
ax2.text(n_words + 0.5, n_words - 1, '← Focus when\nreading "mat"',
        fontsize=10, color='gold', fontweight='bold')

# Add insights box
fig.text(0.5, 0.02,
         'Key Insight: Humans naturally use selective attention - focusing strongly on relevant context ("on the") while ignoring decorative adjectives',
         fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('../figures/attention_heatmap.pdf', dpi=300, bbox_inches='tight')
print("Generated: attention_heatmap.pdf")
plt.close()