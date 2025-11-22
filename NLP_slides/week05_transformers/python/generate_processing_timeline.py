"""
Generate Processing Timeline Comparison
Week 5 Transformers - For Slide 20 (Why This Solves Speed)
Gantt-style timeline showing RNN sequential vs Transformer parallel
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# Template colors
COLOR_MAIN = '#333366'
COLOR_RED = '#D62728'
COLOR_GREEN = '#44A044'
COLOR_LIGHT_BG = '#F0F0F0'

# Create figure
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 8),
                                         gridspec_kw={'height_ratios': [1, 1.3]})
fig.patch.set_facecolor('white')
fig.suptitle('Processing Timeline: RNN (Sequential) vs Transformer (Parallel)', fontsize=14, fontweight='bold', y=0.96)

# TOP PANEL: RNN Sequential Processing
ax_top.set_xlim(0, 310)
ax_top.set_ylim(0, 10)
ax_top.axis('off')
ax_top.set_title('RNN: Sequential Bottleneck', fontsize=12, fontweight='bold',
                  color=COLOR_RED, pad=10)

# Timeline axis
ax_top.plot([0, 310], [2, 2], 'k-', linewidth=2)
for t in range(0, 310, 50):
    ax_top.plot([t, t], [1.8, 2.2], 'k-', linewidth=1)
    ax_top.text(t, 1.3, f'{t}ms', ha='center', va='center', fontsize=8)

# Sequential processing blocks
words_rnn = ['Word 1', 'Word 2', 'Word 3']
times_start = [0, 100, 200]
for i, (word, start) in enumerate(zip(words_rnn, times_start)):
    # Processing block (100ms each)
    block = Rectangle((start, 3), 100, 1.5, facecolor=COLOR_RED,
                     edgecolor=COLOR_MAIN, linewidth=2, alpha=0.5)
    ax_top.add_patch(block)
    ax_top.text(start + 50, 3.75, word, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    # WAIT label between blocks
    if i < len(words_rnn) - 1:
        wait_box = FancyBboxPatch((start + 100, 3.2), 100, 1.1,
                                  boxstyle="round,pad=0.05",
                                  facecolor='#FFE6E6', edgecolor=COLOR_RED,
                                  linewidth=2, linestyle='dashed')
        ax_top.add_patch(wait_box)
        ax_top.text(start + 150, 3.75, 'WAIT', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=COLOR_RED)

# Total time annotation
ax_top.text(150, 6, 'Total Time: 300ms', ha='center', va='center',
            fontsize=13, fontweight='bold', color=COLOR_RED,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE6E6', edgecolor=COLOR_RED, linewidth=2))

# GPU utilization
ax_top.text(150, 7.5, 'GPU Utilization: 1-5% (95% idle!)', ha='center', va='center',
            fontsize=11, color=COLOR_RED, style='italic')

# BOTTOM PANEL: Transformer Parallel Processing
ax_bottom.set_xlim(0, 310)
ax_bottom.set_ylim(0, 10)
ax_bottom.axis('off')
ax_bottom.set_title('Transformer: Full Parallelization', fontsize=12, fontweight='bold',
                     color=COLOR_GREEN, pad=10)

# Timeline axis
ax_bottom.plot([0, 310], [2, 2], 'k-', linewidth=2)
for t in range(0, 310, 50):
    ax_bottom.plot([t, t], [1.8, 2.2], 'k-', linewidth=1)
    ax_bottom.text(t, 1.3, f'{t}ms', ha='center', va='center', fontsize=8)

# Parallel processing (all at once, 10ms each)
words_transformer = ['Word 1', 'Word 2', 'Word 3', '...', 'Word 100']
y_positions = [7, 6, 5, 4, 3]
for i, (word, y) in enumerate(zip(words_transformer, y_positions)):
    # All start at time 0, finish at time 10
    block = Rectangle((0, y), 10, 0.6, facecolor=COLOR_GREEN,
                     edgecolor=COLOR_MAIN, linewidth=2, alpha=0.6)
    ax_bottom.add_patch(block)
    ax_bottom.text(5, y + 0.3, word, ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')

    # PARALLEL label
    if i < len(words_transformer) - 1:
        ax_bottom.text(15, y + 0.3, '(parallel)', ha='left', va='center',
                       fontsize=8, style='italic', color=COLOR_GREEN)

# ALL AT ONCE annotation
ax_bottom.text(120, 5, 'ALL AT ONCE', ha='center', va='center',
               fontsize=13, fontweight='bold', color=COLOR_GREEN,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#E6FFE6', edgecolor=COLOR_GREEN, linewidth=2))

# Total time annotation
total_box = FancyBboxPatch((200, 5.5), 105, 2,
                           boxstyle="round,pad=0.1",
                           facecolor='#E6FFE6', edgecolor=COLOR_GREEN, linewidth=3)
ax_bottom.add_patch(total_box)
ax_bottom.text(252.5, 6.7, 'Total Time: 10ms', ha='center', va='center',
               fontsize=13, fontweight='bold', color=COLOR_GREEN)
ax_bottom.text(252.5, 6.0, '30x faster!', ha='center', va='center',
               fontsize=11, color=COLOR_GREEN)

# GPU utilization
ax_bottom.text(252.5, 4, 'GPU Utilization: 85-92% (full power!)', ha='center', va='center',
               fontsize=11, fontweight='bold', color=COLOR_GREEN, style='italic')

# GPU cores visualization
ax_bottom.text(252.5, 2.5, 'Using all 5,120 GPU cores simultaneously', ha='center', va='center',
               fontsize=10, color=COLOR_GREEN)

plt.tight_layout(rect=[0, 0, 1, 0.94])

# Save figure
output_path = '../figures/sr_22_processing_timeline.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')

plt.close()
