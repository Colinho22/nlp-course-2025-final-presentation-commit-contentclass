"""
Generate Architecture Comparison Diagram
Week 5 Transformers - RNN+Attention vs Pure Attention
Side-by-side visual comparison
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
import numpy as np

# Template colors
COLOR_MAIN = '#333366'
COLOR_SUCCESS = '#44A044'
COLOR_FAILURE = '#D62728'
COLOR_LIGHT_BG = '#F0F0F0'
COLOR_LAVENDER = '#ADADD8'
COLOR_BLUE = '#1F77B4'

# Create figure
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 8))
fig.patch.set_facecolor('white')

# LEFT PANEL: RNN + Attention (Sequential)
ax_left.set_xlim(0, 10)
ax_left.set_ylim(0, 10)
ax_left.axis('off')
ax_left.set_title('OLD WAY: RNN + Attention', fontsize=13, fontweight='bold',
                   color=COLOR_FAILURE, pad=15)

# Sequential processing boxes
y_positions = [8, 6.5, 5, 3.5]
words_left = ['The', 'cat', 'sat', '...']

for i, (y, word) in enumerate(zip(y_positions, words_left)):
    # Word input
    input_box = FancyBboxPatch((1, y-0.3), 1.5, 0.6,
                               boxstyle="round,pad=0.05",
                               facecolor=COLOR_LIGHT_BG, edgecolor=COLOR_BLUE, linewidth=2)
    ax_left.add_patch(input_box)
    ax_left.text(1.75, y, word, ha='center', va='center', fontsize=10, fontweight='bold')

    # RNN hidden state
    hidden_box = Circle((4.5, y), 0.5, facecolor=COLOR_LAVENDER, edgecolor=COLOR_MAIN, linewidth=2)
    ax_left.add_patch(hidden_box)
    ax_left.text(4.5, y, f'h{i+1}', ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrow from input to hidden
    arrow = FancyArrowPatch((2.5, y), (4, y),
                           arrowstyle='->', mutation_scale=15,
                           linewidth=2, color=COLOR_MAIN)
    ax_left.add_patch(arrow)

    # Sequential dependency arrow
    if i < len(y_positions) - 1:
        seq_arrow = FancyArrowPatch((4.5, y-0.6), (4.5, y_positions[i+1]+0.6),
                                   arrowstyle='->', mutation_scale=15,
                                   linewidth=3, color=COLOR_FAILURE)
        ax_left.add_patch(seq_arrow)

        # WAIT label
        ax_left.text(3.5, (y + y_positions[i+1])/2, 'WAIT', ha='center', va='center',
                     fontsize=9, fontweight='bold', color=COLOR_FAILURE,
                     bbox=dict(boxstyle='round', facecolor='#FFE6E6', edgecolor=COLOR_FAILURE))

# Attention mechanism (looking back)
for i, y in enumerate(y_positions[:3]):
    for j, y2 in enumerate(y_positions[:i+1]):
        attention_arrow = FancyArrowPatch((7, y), (5.5, y2),
                                         arrowstyle='->', mutation_scale=10,
                                         linewidth=1, color=COLOR_SUCCESS, alpha=0.4,
                                         linestyle='dashed')
        ax_left.add_patch(attention_arrow)

ax_left.text(7, 5, 'Attention\nlooks back', ha='center', va='center',
             fontsize=9, fontweight='bold', color=COLOR_SUCCESS)

# Sequential bottleneck annotation
bottleneck_box = FancyBboxPatch((0.5, 0.5), 9, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor='#FFE6E6', edgecolor=COLOR_FAILURE, linewidth=2)
ax_left.add_patch(bottleneck_box)
ax_left.text(5, 1.6, '⚠ Sequential Bottleneck', ha='center', va='center',
             fontsize=11, fontweight='bold', color=COLOR_FAILURE)
ax_left.text(5, 1.0, 'Process one word at a time\nGPU utilization: 1-5%',
             ha='center', va='center', fontsize=9, color=COLOR_FAILURE)

# RIGHT PANEL: Pure Attention (Parallel)
ax_right.set_xlim(0, 10)
ax_right.set_ylim(0, 10)
ax_right.axis('off')
ax_right.set_title('NEW WAY: Pure Attention', fontsize=13, fontweight='bold',
                    color=COLOR_SUCCESS, pad=15)

# All words at once
y_positions_right = [7.5, 6.5, 5.5, 4.5]
words_right = ['The', 'cat', 'sat', 'on']

# Word boxes arranged horizontally at top
for i, word in enumerate(words_right):
    x = 2 + i * 2
    word_box = FancyBboxPatch((x-0.5, 8.5-0.3), 1.0, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor=COLOR_LIGHT_BG, edgecolor=COLOR_SUCCESS, linewidth=2)
    ax_right.add_patch(word_box)
    ax_right.text(x, 8.5, word, ha='center', va='center', fontsize=10, fontweight='bold')

# Self-attention mesh (all-to-all connections)
positions = [(2, 6.5), (4, 6.5), (6, 6.5), (8, 6.5)]

for i, (x1, y1) in enumerate(positions):
    # Attention node
    circle = Circle((x1, y1), 0.4, facecolor=COLOR_LAVENDER, edgecolor=COLOR_MAIN, linewidth=2)
    ax_right.add_patch(circle)
    ax_right.text(x1, y1, f'A{i+1}', ha='center', va='center', fontsize=9, fontweight='bold')

    # Connect to all other positions (mesh)
    for j, (x2, y2) in enumerate(positions):
        if i != j:
            attention_arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                             arrowstyle='<->', mutation_scale=8,
                                             linewidth=1, color=COLOR_SUCCESS, alpha=0.3)
            ax_right.add_patch(attention_arrow)

ax_right.text(5, 5.2, 'All words\nconnect to all\nwords', ha='center', va='center',
              fontsize=9, fontweight='bold', color=COLOR_SUCCESS)

# Parallel annotation
parallel_box = FancyBboxPatch((0.5, 0.5), 9, 2.5,
                              boxstyle="round,pad=0.1",
                              facecolor='#E6FFE6', edgecolor=COLOR_SUCCESS, linewidth=2)
ax_right.add_patch(parallel_box)
ax_right.text(5, 2.5, '✓ Full Parallelization', ha='center', va='center',
              fontsize=11, fontweight='bold', color=COLOR_SUCCESS)
ax_right.text(5, 1.9, 'Process ALL words simultaneously', ha='center', va='center',
              fontsize=9, color=COLOR_SUCCESS)
ax_right.text(5, 1.4, 'GPU utilization: 85-92%', ha='center', va='center',
              fontsize=9, fontweight='bold', color=COLOR_SUCCESS)
ax_right.text(5, 0.8, 'No waiting! → 100x speedup', ha='center', va='center',
              fontsize=9, fontweight='bold', color=COLOR_SUCCESS)

plt.tight_layout()

# Save figure
output_path = '../figures/sr_15_architecture_comparison.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')

plt.close()
