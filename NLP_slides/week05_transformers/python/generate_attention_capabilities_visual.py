"""
Generate Attention Capabilities Infographic
Week 5 Transformers - Visual Enhancement
Shows what pure attention CAN vs CANNOT see
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

# Template colors (matching Week 5 lavender theme)
COLOR_MAIN = '#333366'  # Dark purple
COLOR_SUCCESS = '#44A044'  # Green
COLOR_FAILURE = '#D62728'  # Red
COLOR_LIGHT_BG = '#F0F0F0'  # Light gray background
COLOR_LAVENDER = '#ADADD8'  # Lavender

# Create figure
fig, (ax_can, ax_barrier, ax_cannot) = plt.subplots(1, 3, figsize=(12, 6),
                                                      gridspec_kw={'width_ratios': [1, 0.15, 1]})
fig.patch.set_facecolor('white')

# LEFT PANEL: What Pure Attention CAN See
ax_can.set_xlim(0, 10)
ax_can.set_ylim(0, 10)
ax_can.axis('off')
ax_can.set_title('✓ What Pure Attention CAN See', fontsize=14, fontweight='bold',
                  color=COLOR_SUCCESS, pad=15)

# Add semantic network

# Words as nodes
words = {'cat': (5, 7), 'sat': (3, 4), 'mat': (7, 4), 'the': (2, 7), 'on': (8, 7)}

for word, (x, y) in words.items():
    circle = Circle((x, y), 0.6, facecolor=COLOR_LAVENDER, edgecolor=COLOR_MAIN, linewidth=2)
    ax_can.add_patch(circle)
    ax_can.text(x, y, word, ha='center', va='center', fontsize=11, fontweight='bold')

# Semantic relationships (connections)
relationships = [
    ('cat', 'sat', 'subject-verb'),
    ('sat', 'mat', 'verb-object'),
    ('cat', 'the', 'noun-article'),
    ('mat', 'the', 'noun-article'),
]

for word1, word2, rel_type in relationships:
    x1, y1 = words[word1]
    x2, y2 = words[word2]

    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='<->', mutation_scale=15,
                           linewidth=2, color=COLOR_SUCCESS, alpha=0.6,
                           connectionstyle="arc3,rad=.2")
    ax_can.add_patch(arrow)

# Add checkmarks
checkmarks = [
    (1.5, 9, '✓ Semantic relationships'),
    (1.5, 2, '✓ Word meanings'),
    (1.5, 1, '✓ Co-occurrence patterns'),
]

for x, y, text in checkmarks:
    ax_can.text(x, y, text, fontsize=10, color=COLOR_SUCCESS, fontweight='bold')

# MIDDLE PANEL: Barrier
ax_barrier.set_xlim(0, 1)
ax_barrier.set_ylim(0, 10)
ax_barrier.axis('off')

# Big X barrier
ax_barrier.plot([0.2, 0.8], [9, 1], 'r-', linewidth=6, alpha=0.7)
ax_barrier.plot([0.2, 0.8], [1, 9], 'r-', linewidth=6, alpha=0.7)

ax_barrier.text(0.5, 5.5, 'NO', ha='center', va='center', fontsize=18,
                fontweight='bold', color=COLOR_FAILURE, rotation=0)
ax_barrier.text(0.5, 4.5, 'POSITION', ha='center', va='center', fontsize=10,
                fontweight='bold', color=COLOR_FAILURE, rotation=0)
ax_barrier.text(0.5, 3.8, 'INFO', ha='center', va='center', fontsize=10,
                fontweight='bold', color=COLOR_FAILURE, rotation=0)

# RIGHT PANEL: What Pure Attention CANNOT See
ax_cannot.set_xlim(0, 10)
ax_cannot.set_ylim(0, 10)
ax_cannot.axis('off')
ax_cannot.set_title('✗ What Pure Attention CANNOT See', fontsize=14, fontweight='bold',
                    color=COLOR_FAILURE, pad=15)

# Scrambled words (showing permutation invariance)
scrambled_positions = [
    (2, 7, 'sat', True),   # Word 1 scrambled
    (5, 7, 'the', True),   # Word 2 scrambled
    (8, 7, 'cat', True),   # Word 3 scrambled
    (3, 4, 'on', True),    # Word 4 scrambled
    (7, 4, 'mat', True),   # Word 5 scrambled
]

for x, y, word, is_wrong in scrambled_positions:
    box = FancyBboxPatch((x-0.7, y-0.5), 1.4, 1.0,
                         boxstyle="round,pad=0.1",
                         facecolor='#FFE6E6', edgecolor=COLOR_FAILURE, linewidth=2)
    ax_cannot.add_patch(box)
    ax_cannot.text(x, y, word, ha='center', va='center', fontsize=11, fontweight='bold')
    # Add question mark
    ax_cannot.text(x+0.9, y+0.6, '?', ha='center', va='center', fontsize=16,
                   color=COLOR_FAILURE, fontweight='bold')

# X marks showing order problems
x_marks = [
    (2.5, 5.5),
    (5, 5.5),
    (7.5, 5.5),
]

for x, y in x_marks:
    ax_cannot.plot([x-0.3, x+0.3], [y-0.3, y+0.3], 'r-', linewidth=4, alpha=0.7)
    ax_cannot.plot([x-0.3, x+0.3], [y+0.3, y-0.3], 'r-', linewidth=4, alpha=0.7)

# Add X marks for missing capabilities
missing_caps = [
    (1.5, 2, '✗ Word order'),
    (1.5, 1.2, '✗ Temporal sequence'),
    (1.5, 0.4, '✗ Position information'),
]

for x, y, text in missing_caps:
    ax_cannot.text(x, y, text, fontsize=10, color=COLOR_FAILURE, fontweight='bold')

# Add bottom metric
fig.text(0.5, 0.02, 'Permutation Test: 52% accuracy (barely better than random 50%)',
         ha='center', va='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT_BG, edgecolor=COLOR_FAILURE, linewidth=2))

plt.tight_layout(rect=[0, 0.06, 1, 1])

# Save figure
output_path = '../figures/sr_13_attention_capabilities_infographic.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')

plt.close()
