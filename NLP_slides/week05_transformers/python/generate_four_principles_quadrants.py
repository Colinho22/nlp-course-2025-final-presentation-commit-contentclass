"""
Generate Four Principles Visual Quadrants
Week 5 Transformers - Replaces text-heavy "What We Learned" slide
Shows 4 key principles as visual quadrants
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Polygon
import numpy as np

# Template colors
COLOR_MAIN = '#333366'
COLOR_BLUE = '#4472C4'
COLOR_PURPLE = '#8B5A9B'
COLOR_GREEN = '#44A044'
COLOR_ORANGE = '#FF7F0E'
COLOR_LIGHT_BG = '#F0F0F0'

# Create figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
fig.patch.set_facecolor('white')
fig.suptitle('Four Key Principles from Transformers', fontsize=16, fontweight='bold', y=0.98)

# Common settings for all axes
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

# QUADRANT 1: Sequential → Explicit Encoding
ax1.set_title('1. Sequential Processing Not Always Necessary', fontsize=11, fontweight='bold',
              color=COLOR_BLUE, pad=10)

# Timeline showing position encoding
positions = [1, 3, 5, 7, 9]
for i, x in enumerate(positions):
    # Position marker
    circle = Circle((x, 7), 0.4, facecolor=COLOR_BLUE, edgecolor=COLOR_MAIN, linewidth=2, alpha=0.6)
    ax1.add_patch(circle)
    ax1.text(x, 7, str(i+1), ha='center', va='center',
             fontsize=10, fontweight='bold', color='white')

    # Word box
    word_box = FancyBboxPatch((x-0.5, 4.5-0.3), 1.0, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor=COLOR_LIGHT_BG, edgecolor=COLOR_BLUE, linewidth=2)
    ax1.add_patch(word_box)
    ax1.text(x, 4.5, f'W{i+1}', ha='center', va='center',
             fontsize=9, fontweight='bold')

    # Arrow connecting position to word
    arrow = FancyArrowPatch((x, 6.5), (x, 5.2),
                           arrowstyle='->', mutation_scale=12,
                           linewidth=2, color=COLOR_BLUE, alpha=0.6)
    ax1.add_patch(arrow)

# Key message box
key_box1 = FancyBboxPatch((0.5, 1), 9, 2,
                          boxstyle="round,pad=0.1",
                          facecolor=COLOR_BLUE, edgecolor=COLOR_BLUE, linewidth=2, alpha=0.1)
ax1.add_patch(key_box1)
ax1.text(5, 2.3, 'Order can be encoded,', ha='center', va='center',
         fontsize=11, fontweight='bold', color=COLOR_BLUE)
ax1.text(5, 1.7, 'not just processed sequentially', ha='center', va='center',
         fontsize=10, color=COLOR_BLUE)

# QUADRANT 2: Parallelization Through Independence
ax2.set_title('2. Parallelization Through Independence', fontsize=11, fontweight='bold',
              color=COLOR_PURPLE, pad=10)

# Multiple processors working simultaneously
processor_positions = [(2, 7), (5, 7), (8, 7),
                       (2, 4.5), (5, 4.5), (8, 4.5)]

for i, (x, y) in enumerate(processor_positions):
    # Processor
    proc_box = Rectangle((x-0.6, y-0.4), 1.2, 0.8,
                         facecolor=COLOR_PURPLE, edgecolor=COLOR_MAIN, linewidth=2, alpha=0.4)
    ax2.add_patch(proc_box)
    ax2.text(x, y, f'P{i+1}', ha='center', va='center',
             fontsize=9, fontweight='bold', color='white')

# Simultaneous arrows
for x, y in processor_positions:
    arrow = FancyArrowPatch((x, y-0.6), (x, 2.5),
                           arrowstyle='->', mutation_scale=12,
                           linewidth=2, color=COLOR_PURPLE, alpha=0.5)
    ax2.add_patch(arrow)

# Key message box
key_box2 = FancyBboxPatch((0.5, 1), 9, 1.2,
                          boxstyle="round,pad=0.1",
                          facecolor=COLOR_PURPLE, edgecolor=COLOR_PURPLE, linewidth=2, alpha=0.1)
ax2.add_patch(key_box2)
ax2.text(5, 1.8, 'Trade more compute operations', ha='center', va='center',
         fontsize=11, fontweight='bold', color=COLOR_PURPLE)
ax2.text(5, 1.3, 'for less wall-clock time', ha='center', va='center',
         fontsize=10, color=COLOR_PURPLE)

# QUADRANT 3: Selection > Compression
ax3.set_title('3. Selective Attention vs Compression', fontsize=11, fontweight='bold',
              color=COLOR_GREEN, pad=10)

# Left side: Full information set
info_items = [(2, 8), (3.5, 8), (2, 6.5), (3.5, 6.5), (2, 5), (3.5, 5)]
for i, (x, y) in enumerate(info_items):
    item_circle = Circle((x, y), 0.4, facecolor=COLOR_LIGHT_BG,
                         edgecolor=COLOR_GREEN, linewidth=2, alpha=0.7)
    ax3.add_patch(item_circle)
    ax3.text(x, y, f'{i+1}', ha='center', va='center',
             fontsize=9, fontweight='bold')

# Spotlight selecting relevant items
spotlight = Polygon([(5.5, 7.5), (6.5, 7.5), (7, 6), (5, 6)],
                   facecolor=COLOR_GREEN, alpha=0.2, edgecolor=COLOR_GREEN, linewidth=2)
ax3.add_patch(spotlight)

# Selected items (highlighted)
selected_positions = [(6, 7), (7, 6)]
for x, y in selected_positions:
    sel_circle = Circle((x, y), 0.5, facecolor=COLOR_GREEN,
                        edgecolor=COLOR_MAIN, linewidth=2, alpha=0.6)
    ax3.add_patch(sel_circle)
    ax3.text(x, y, '✓', ha='center', va='center',
             fontsize=14, fontweight='bold', color='white')

ax3.text(2.75, 3.5, 'Keep all\ninformation', ha='center', va='center',
         fontsize=10, fontweight='bold', color=COLOR_GREEN)
ax3.text(6.5, 4.5, 'Select\nrelevant', ha='center', va='center',
         fontsize=10, fontweight='bold', color=COLOR_GREEN)

# Key message box
key_box3 = FancyBboxPatch((0.5, 1), 9, 1.2,
                          boxstyle="round,pad=0.1",
                          facecolor=COLOR_GREEN, edgecolor=COLOR_GREEN, linewidth=2, alpha=0.1)
ax3.add_patch(key_box3)
ax3.text(5, 1.8, 'Keep information, let model', ha='center', va='center',
         fontsize=11, fontweight='bold', color=COLOR_GREEN)
ax3.text(5, 1.3, 'decide what\'s relevant', ha='center', va='center',
         fontsize=10, color=COLOR_GREEN)

# QUADRANT 4: Hardware-Algorithm Co-Design
ax4.set_title('4. Hardware-Algorithm Co-Design', fontsize=11, fontweight='bold',
              color=COLOR_ORANGE, pad=10)

# GPU (hardware) - left side
gpu_box = FancyBboxPatch((1, 5.5), 3, 2.5,
                         boxstyle="round,pad=0.1",
                         facecolor=COLOR_LIGHT_BG, edgecolor=COLOR_ORANGE, linewidth=2)
ax4.add_patch(gpu_box)
ax4.text(2.5, 7.5, 'GPU', ha='center', va='center',
         fontsize=11, fontweight='bold', color=COLOR_ORANGE)
ax4.text(2.5, 6.8, 'Parallel', ha='center', va='center',
         fontsize=9, color=COLOR_ORANGE)
ax4.text(2.5, 6.3, 'Processing', ha='center', va='center',
         fontsize=9, color=COLOR_ORANGE)

# Algorithm - right side
algo_box = FancyBboxPatch((6, 5.5), 3, 2.5,
                          boxstyle="round,pad=0.1",
                          facecolor=COLOR_LIGHT_BG, edgecolor=COLOR_ORANGE, linewidth=2)
ax4.add_patch(algo_box)
ax4.text(7.5, 7.5, 'Algorithm', ha='center', va='center',
         fontsize=11, fontweight='bold', color=COLOR_ORANGE)
ax4.text(7.5, 6.8, 'Transformer', ha='center', va='center',
         fontsize=9, color=COLOR_ORANGE)
ax4.text(7.5, 6.3, 'Architecture', ha='center', va='center',
         fontsize=9, color=COLOR_ORANGE)

# Puzzle pieces fitting together
arrow_left = FancyArrowPatch((4, 6.75), (5.5, 6.75),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=3, color=COLOR_ORANGE)
ax4.add_patch(arrow_left)
arrow_right = FancyArrowPatch((6.5, 6.75), (5, 6.75),
                             arrowstyle='->', mutation_scale=20,
                             linewidth=3, color=COLOR_ORANGE)
ax4.add_patch(arrow_right)

# Perfect fit indicator
ax4.text(5, 7.5, '✓', ha='center', va='center',
         fontsize=20, fontweight='bold', color=COLOR_ORANGE)
ax4.text(5, 6.1, 'Perfect\nMatch', ha='center', va='center',
         fontsize=9, fontweight='bold', color=COLOR_ORANGE)

# Key message box
key_box4 = FancyBboxPatch((0.5, 1), 9, 2.5,
                          boxstyle="round,pad=0.1",
                          facecolor=COLOR_ORANGE, edgecolor=COLOR_ORANGE, linewidth=2, alpha=0.1)
ax4.add_patch(key_box4)
ax4.text(5, 2.8, 'Match architecture to', ha='center', va='center',
         fontsize=11, fontweight='bold', color=COLOR_ORANGE)
ax4.text(5, 2.3, 'hardware capabilities', ha='center', va='center',
         fontsize=11, fontweight='bold', color=COLOR_ORANGE)
ax4.text(5, 1.6, 'Transformer perfectly utilizes', ha='center', va='center',
         fontsize=9, color=COLOR_ORANGE)
ax4.text(5, 1.2, 'GPU parallel processing power', ha='center', va='center',
         fontsize=9, color=COLOR_ORANGE)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
output_path = '../figures/sr_17_four_principles_visual.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')

plt.close()
