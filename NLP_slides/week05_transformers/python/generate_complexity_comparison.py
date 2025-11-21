"""
Generate Sequential vs Parallel Complexity Comparison
Week 5 Transformers - For Slide 5 (Quantifying the Speed Problem)
Shows O(n) sequential vs O(1) parallel with GPU utilization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# Template colors
COLOR_MAIN = '#333366'
COLOR_BLUE = '#4472C4'
COLOR_PURPLE = '#8B5A9B'
COLOR_GREEN = '#44A044'
COLOR_ORANGE = '#FF7F0E'
COLOR_RED = '#D62728'
COLOR_LIGHT_BG = '#F0F0F0'

# Create figure
fig, (ax_left, ax_right, ax_bottom) = plt.subplots(3, 1, figsize=(12, 10),
                                                     gridspec_kw={'height_ratios': [1, 1, 0.6]})
fig.patch.set_facecolor('white')
fig.suptitle('Sequential vs Parallel Processing Complexity', fontsize=14, fontweight='bold', y=0.98)

# LEFT PANEL: Sequential O(n)
ax_left.set_xlim(0, 10)
ax_left.set_ylim(0, 10)
ax_left.axis('off')
ax_left.set_title('Sequential Processing: O(n)', fontsize=12, fontweight='bold',
                   color=COLOR_RED, pad=10)

# Sequential blocks with waiting
y_start = 8
words = ['Word 1', 'Word 2', 'Word 3', '...', 'Word n']
for i, word in enumerate(words):
    x = 0.5 + i * 2
    y = y_start

    # Processing block
    block = FancyBboxPatch((x-0.4, y-0.4), 0.8, 0.8,
                           boxstyle="round,pad=0.05",
                           facecolor=COLOR_RED, edgecolor=COLOR_MAIN,
                           linewidth=2, alpha=0.4)
    ax_left.add_patch(block)
    ax_left.text(x, y, f'W{i+1}', ha='center', va='center',
                 fontsize=9, fontweight='bold')

    # Arrow to next (with WAIT label)
    if i < len(words) - 1:
        arrow = FancyArrowPatch((x+0.5, y), (x+1.5, y),
                               arrowstyle='->', mutation_scale=15,
                               linewidth=2, color=COLOR_RED)
        ax_left.add_patch(arrow)
        ax_left.text(x+1, y-0.8, 'WAIT', ha='center', va='center',
                     fontsize=8, fontweight='bold', color=COLOR_RED,
                     bbox=dict(boxstyle='round', facecolor='#FFE6E6', edgecolor=COLOR_RED))

# Complexity label
ax_left.text(5, 5, 'Time Complexity: O(n)', ha='center', va='center',
             fontsize=12, fontweight='bold', color=COLOR_RED,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=COLOR_RED, linewidth=2))

# Single lane highway metaphor
ax_left.text(5, 3, 'Like Single-Lane Highway:', ha='center', va='center',
             fontsize=10, fontweight='bold')
for i in range(5):
    car_x = 1 + i * 1.8
    car = Rectangle((car_x, 2-0.2), 0.6, 0.4, facecolor=COLOR_ORANGE,
                    edgecolor=COLOR_MAIN, linewidth=1)
    ax_left.add_patch(car)

# Road
road = Rectangle((0.5, 1.5), 9, 0.8, facecolor='#CCCCCC', alpha=0.3)
ax_left.add_patch(road)

# RIGHT PANEL: Parallel O(1)
ax_right.set_xlim(0, 10)
ax_right.set_ylim(0, 10)
ax_right.axis('off')
ax_right.set_title('Parallel Processing: O(1)', fontsize=12, fontweight='bold',
                    color=COLOR_GREEN, pad=10)

# Parallel blocks all at once
y_positions = [8, 6.5, 5, 3.5]
for i, y in enumerate(y_positions[:4]):
    x = 5

    # Processing blocks in parallel
    for j in range(3):
        block_x = 1.5 + j * 2.5
        block = FancyBboxPatch((block_x-0.4, y-0.4), 0.8, 0.8,
                               boxstyle="round,pad=0.05",
                               facecolor=COLOR_GREEN, edgecolor=COLOR_MAIN,
                               linewidth=2, alpha=0.4)
        ax_right.add_patch(block)
        ax_right.text(block_x, y, f'W{j+1}', ha='center', va='center',
                     fontsize=9, fontweight='bold')

# Parallel arrows (all pointing down simultaneously)
for j in range(3):
    block_x = 1.5 + j * 2.5
    arrow = FancyArrowPatch((block_x, 7.5), (block_x, 2.5),
                           arrowstyle='->', mutation_scale=15,
                           linewidth=2, color=COLOR_GREEN, alpha=0.6)
    ax_right.add_patch(arrow)

# ALL AT ONCE label
ax_right.text(5, 8.8, 'ALL AT ONCE', ha='center', va='center',
             fontsize=11, fontweight='bold', color=COLOR_GREEN,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E6FFE6', edgecolor=COLOR_GREEN))

# Complexity label
ax_right.text(5, 5.5, 'Time Complexity: O(1)', ha='center', va='center',
             fontsize=12, fontweight='bold', color=COLOR_GREEN,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=COLOR_GREEN, linewidth=2))

# Multi-lane highway metaphor
ax_right.text(5, 3, 'Like Multi-Lane Highway:', ha='center', va='center',
             fontsize=10, fontweight='bold')
for lane in range(3):
    car_x = 3
    car_y = 2 - 0.4 * lane
    car = Rectangle((car_x, car_y-0.15), 0.6, 0.3, facecolor=COLOR_GREEN,
                    edgecolor=COLOR_MAIN, linewidth=1)
    ax_right.add_patch(car)

# Roads (3 lanes)
for lane in range(3):
    road_y = 1.5 - 0.4 * lane
    road = Rectangle((0.5, road_y), 9, 0.25, facecolor='#CCCCCC', alpha=0.3)
    ax_right.add_patch(road)

# BOTTOM PANEL: GPU Utilization
ax_bottom.set_xlim(0, 10)
ax_bottom.set_ylim(0, 10)
ax_bottom.axis('off')
ax_bottom.set_title('GPU Core Utilization', fontsize=11, fontweight='bold', pad=10)

# Bar chart showing utilization
models = ['RNN\n(Sequential)', 'RNN+Attention\n(Hybrid)', 'Transformer\n(Parallel)']
utilizations = [1, 5, 90]
colors = [COLOR_RED, COLOR_ORANGE, COLOR_GREEN]
x_positions = [2, 5, 8]

for i, (model, util, color, x) in enumerate(zip(models, utilizations, colors, x_positions)):
    # Bar
    bar_height = util / 10  # Scale to fit
    bar = Rectangle((x-0.6, 5-bar_height), 1.2, bar_height,
                    facecolor=color, edgecolor=COLOR_MAIN, linewidth=2, alpha=0.5)
    ax_bottom.add_patch(bar)

    # Percentage label
    ax_bottom.text(x, 5.5, f'{util}%', ha='center', va='center',
                   fontsize=11, fontweight='bold', color=color)

    # Model label
    ax_bottom.text(x, 3.5, model, ha='center', va='center',
                   fontsize=9, fontweight='bold')

# Y-axis labels
ax_bottom.text(0.5, 5, '100%', ha='right', va='center', fontsize=9)
ax_bottom.text(0.5, 4, '50%', ha='right', va='center', fontsize=9)
ax_bottom.text(0.5, 3, '0%', ha='right', va='center', fontsize=9)

# Speedup annotation
ax_bottom.text(5, 8, '100x Speedup Achieved!', ha='center', va='center',
               fontsize=12, fontweight='bold', color=COLOR_GREEN,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#E6FFE6', edgecolor=COLOR_GREEN, linewidth=2))

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
output_path = '../figures/sr_18_complexity_comparison.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')

plt.close()
