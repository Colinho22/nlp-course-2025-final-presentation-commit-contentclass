"""
Generate Multi-Head Attention Heatmap
Week 5 Transformers - For Slide 19 (Multi-Head Attention)
Shows 4 attention heads with different patterns for "The bank by the river"
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Template colors
COLOR_MAIN = '#333366'
COLOR_BLUE = '#4472C4'
COLOR_PURPLE = '#8B5A9B'
COLOR_GREEN = '#44A044'
COLOR_ORANGE = '#FF7F0E'

# Sentence
words = ['The', 'bank', 'by', 'the', 'river']

# Create attention patterns for each head
# Head 1: Syntactic (articles to nouns)
attention_syntax = np.array([
    [0.1, 0.7, 0.1, 0.0, 0.1],  # The -> bank
    [0.1, 0.6, 0.1, 0.1, 0.1],  # bank -> bank
    [0.0, 0.6, 0.2, 0.0, 0.2],  # by -> bank
    [0.0, 0.1, 0.0, 0.1, 0.8],  # the -> river
    [0.0, 0.1, 0.1, 0.2, 0.6],  # river -> river
])

# Head 2: Semantic (related meanings)
attention_semantic = np.array([
    [0.2, 0.2, 0.2, 0.2, 0.2],  # The (uniform)
    [0.1, 0.3, 0.1, 0.1, 0.4],  # bank -> river
    [0.1, 0.3, 0.2, 0.1, 0.3],  # by (bridge words)
    [0.2, 0.2, 0.2, 0.2, 0.2],  # the (uniform)
    [0.1, 0.4, 0.1, 0.1, 0.3],  # river -> bank
])

# Head 3: Positional (adjacent words)
attention_position = np.array([
    [0.5, 0.4, 0.1, 0.0, 0.0],  # The -> bank (next)
    [0.3, 0.4, 0.3, 0.0, 0.0],  # bank (local)
    [0.0, 0.4, 0.3, 0.3, 0.0],  # by (neighbors)
    [0.0, 0.0, 0.3, 0.4, 0.3],  # the (neighbors)
    [0.0, 0.0, 0.0, 0.3, 0.7],  # river (self)
])

# Head 4: Global (sentence-level)
attention_global = np.array([
    [0.2, 0.2, 0.2, 0.2, 0.2],  # The (uniform)
    [0.15, 0.3, 0.15, 0.15, 0.25],  # bank (distributed)
    [0.2, 0.2, 0.2, 0.2, 0.2],  # by (uniform)
    [0.2, 0.2, 0.2, 0.2, 0.2],  # the (uniform)
    [0.15, 0.25, 0.15, 0.15, 0.3],  # river (distributed)
])

# Create figure
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('white')
fig.suptitle('Multi-Head Attention Patterns: "The bank by the river"', fontsize=14, fontweight='bold', y=0.96)

# 2x2 grid for 4 heads
heads = [
    (attention_syntax, 'Head 1: Syntactic\n(Articles → Nouns)', COLOR_BLUE),
    (attention_semantic, 'Head 2: Semantic\n(Related Meanings)', COLOR_GREEN),
    (attention_position, 'Head 3: Positional\n(Adjacent Words)', COLOR_ORANGE),
    (attention_global, 'Head 4: Global\n(Sentence-Level)', COLOR_PURPLE)
]

for idx, (attention, title, color) in enumerate(heads):
    ax = plt.subplot(2, 2, idx + 1)

    # Heatmap
    im = ax.imshow(attention, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.8)

    # Ticks and labels
    ax.set_xticks(np.arange(len(words)))
    ax.set_yticks(np.arange(len(words)))
    ax.set_xticklabels(words, fontsize=10)
    ax.set_yticklabels(words, fontsize=10)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Title with color
    ax.set_title(title, fontsize=11, fontweight='bold', color=color, pad=10)

    # Axis labels
    if idx >= 2:  # Bottom row
        ax.set_xlabel('Attending to →', fontsize=9, color=color)
    if idx % 2 == 0:  # Left column
        ax.set_ylabel('← Query word', fontsize=9, color=color)

    # Add values as text
    for i in range(len(words)):
        for j in range(len(words)):
            val = attention[i, j]
            text_color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.1f}', ha="center", va="center",
                   color=text_color, fontsize=8, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention\nWeight', fontsize=8)

# Overall explanation at bottom
fig.text(0.5, 0.02, 'Each head captures different relationships: Syntax (grammar), Semantics (meaning), Position (locality), Global (context)',
         ha='center', va='center', fontsize=10, style='italic', color=COLOR_MAIN,
         bbox=dict(boxstyle='round,pad=0.5', facecolor=('#F0F0F0'), edgecolor=COLOR_MAIN))

plt.tight_layout(rect=[0, 0.04, 1, 0.94])

# Save figure
output_path = '../figures/sr_21_multihead_heatmap.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')

plt.close()
