"""
Generate Position Problem + Solution Visual
Week 5 Transformers - Merges Slides 10+11
Shows problem diagnosis and solution requirements
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
import numpy as np

# Template colors
COLOR_MAIN = '#333366'
COLOR_SUCCESS = '#44A044'
COLOR_FAILURE = '#D62728'
COLOR_LIGHT_BG = '#F0F0F0'
COLOR_LAVENDER = '#ADADD8'
COLOR_ORANGE = '#FF7F0E'

# Create figure with 2 rows
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.25)

# TOP HALF: Problem Diagnosis
ax_top = fig.add_subplot(gs[0])
ax_top.set_xlim(0, 10)
ax_top.set_ylim(0, 5)
ax_top.axis('off')
ax_top.text(5, 4.7, 'THE PROBLEM: Attention is Permutation Invariant',
            ha='center', va='center', fontsize=14, fontweight='bold', color=COLOR_FAILURE)

# Show scrambled words
sentences = [
    (1, 3.5, ['The', 'cat', 'sat'], COLOR_SUCCESS, '✓ Original'),
    (1, 2.3, ['sat', 'The', 'cat'], COLOR_FAILURE, '✗ Scrambled'),
    (1, 1.1, ['cat', 'sat', 'The'], COLOR_FAILURE, '✗ Scrambled'),
]

for start_x, y, words, color, label in sentences:
    # Draw word boxes
    for i, word in enumerate(words):
        x = start_x + i * 1.2
        box = FancyBboxPatch((x, y-0.25), 1.0, 0.5,
                             boxstyle="round,pad=0.05",
                             facecolor=COLOR_LIGHT_BG, edgecolor=color, linewidth=2)
        ax_top.add_patch(box)
        ax_top.text(x+0.5, y, word, ha='center', va='center', fontsize=10, fontweight='bold')

    # Label
    ax_top.text(start_x + 3.8, y, label, ha='left', va='center',
                fontsize=10, fontweight='bold', color=color)

# Equal sign showing they're treated the same
ax_top.text(6.5, 2.3, '=', ha='center', va='center', fontsize=24,
            fontweight='bold', color=COLOR_MAIN)
ax_top.text(7.2, 2.3, 'SAME', ha='left', va='center', fontsize=11,
            fontweight='bold', color=COLOR_FAILURE)
ax_top.text(7.2, 1.9, 'attention', ha='left', va='center', fontsize=9, color=COLOR_FAILURE)
ax_top.text(7.2, 1.5, 'outputs!', ha='left', va='center', fontsize=9, color=COLOR_FAILURE)

# Root cause box
root_cause_box = FancyBboxPatch((0.5, 0.1), 9, 0.7,
                                boxstyle="round,pad=0.1",
                                facecolor='#FFE6E6', edgecolor=COLOR_FAILURE, linewidth=2)
ax_top.add_patch(root_cause_box)
ax_top.text(5, 0.45, 'Root Cause: No position information → Order doesn\'t matter!',
            ha='center', va='center', fontsize=11, fontweight='bold', color=COLOR_FAILURE)

# BOTTOM HALF: Solution Requirements
ax_bottom = fig.add_subplot(gs[1])
ax_bottom.set_xlim(0, 10)
ax_bottom.set_ylim(0, 6)
ax_bottom.axis('off')
ax_bottom.text(5, 5.7, 'THE SOLUTION: Four Requirements for Position Encoding',
               ha='center', va='center', fontsize=14, fontweight='bold', color=COLOR_SUCCESS)

# Central arrow showing transformation
arrow = FancyArrowPatch((5, 5), (5, 0.5),
                       arrowstyle='->', mutation_scale=30,
                       linewidth=4, color=COLOR_ORANGE, alpha=0.6)
ax_bottom.add_patch(arrow)
ax_bottom.text(5.3, 2.7, 'Position as\nadded signal', ha='left', va='center',
               fontsize=10, fontweight='bold', color=COLOR_ORANGE)

# Four requirement boxes in 2x2 grid
requirements = [
    (1, 4, '1️⃣ Inject Position Info', 'Add unique pattern\nto each position'),
    (6, 4, '2️⃣ Parallel Computation', 'No sequential\ndependency'),
    (1, 1.5, '3️⃣ Any Sequence Length', 'Works for\n10 or 1000 words'),
    (6, 1.5, '4️⃣ Relative Positions', 'Preserves distance\nbetween words'),
]

for x, y, title, desc in requirements:
    # Requirement box
    box = FancyBboxPatch((x-0.4, y-0.6), 3.8, 1.4,
                         boxstyle="round,pad=0.1",
                         facecolor=COLOR_LAVENDER, edgecolor=COLOR_SUCCESS, linewidth=2, alpha=0.3)
    ax_bottom.add_patch(box)

    # Title
    ax_bottom.text(x+1.5, y+0.5, title, ha='center', va='center',
                   fontsize=11, fontweight='bold', color=COLOR_MAIN)

    # Description
    ax_bottom.text(x+1.5, y-0.1, desc, ha='center', va='center',
                   fontsize=9, color=COLOR_MAIN)

# Key insight box
insight_box = FancyBboxPatch((0.5, 0.05), 9, 0.6,
                             boxstyle="round,pad=0.1",
                             facecolor='#E6FFE6', edgecolor=COLOR_SUCCESS, linewidth=2)
ax_bottom.add_patch(insight_box)
ax_bottom.text(5, 0.35, 'Key Insight: Position can be a feature (added signal), not a process (sequential)',
               ha='center', va='center', fontsize=11, fontweight='bold', color=COLOR_SUCCESS)

plt.tight_layout()

# Save figure
output_path = '../figures/sr_14_position_problem_solution.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')

plt.close()
