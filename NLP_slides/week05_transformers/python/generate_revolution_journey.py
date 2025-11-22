"""
Generate Speed Revolution Journey Flowchart
Week 5 Transformers - For Slide 26 (Summary)
5-step flowchart: Problem → Attempt → Diagnosis → Insight → Breakthrough
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Polygon
import numpy as np

# Template colors
COLOR_RED = '#D62728'
COLOR_ORANGE = '#FF7F0E'
COLOR_YELLOW = '#FFD700'
COLOR_BLUE = '#4472C4'
COLOR_GREEN = '#44A044'
COLOR_MAIN = '#333366'

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7.5, 9.5, 'The Speed Revolution Journey', ha='center', va='center',
        fontsize=16, fontweight='bold', color=COLOR_MAIN)

# 5 Steps with visual flow
steps = [
    # (x, y, title, description, metric, color, icon_type)
    (2, 7, 'Problem', 'RNN Sequential\nBottleneck', '90 days training\n2% GPU usage', COLOR_RED, 'problem'),
    (5, 7, 'Attempt', 'Remove RNN\nPure Attention', '10x faster\nBUT lost order', COLOR_ORANGE, 'attempt'),
    (8, 7, 'Diagnosis', 'Attention is\nPermutation Invariant', 'Cannot tell\nword order', COLOR_YELLOW, 'diagnosis'),
    (5, 4, 'Insight', 'Add Position\nas Signal', 'Positional\nencoding idea', COLOR_BLUE, 'insight'),
    (8, 4, 'Breakthrough', 'Self-Attention +\nPosition Encoding', '1 day training\n92% GPU usage', COLOR_GREEN, 'breakthrough'),
]

for i, (x, y, title, desc, metric, color, icon_type) in enumerate(steps):
    # Step box
    box = FancyBboxPatch((x-1.2, y-0.8), 2.4, 1.6,
                         boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor=COLOR_MAIN,
                         linewidth=3, alpha=0.4)
    ax.add_patch(box)

    # Step number
    num_circle = Circle((x-1.0, y+0.6), 0.25, facecolor='white',
                        edgecolor=color, linewidth=2)
    ax.add_patch(num_circle)
    ax.text(x-1.0, y+0.6, str(i+1), ha='center', va='center',
            fontsize=12, fontweight='bold', color=color)

    # Title
    ax.text(x, y+0.5, title, ha='center', va='center',
            fontsize=12, fontweight='bold', color=color)

    # Description
    ax.text(x, y+0.0, desc, ha='center', va='center',
            fontsize=9, color=color)

    # Metric
    ax.text(x, y-0.5, metric, ha='center', va='center',
            fontsize=8, color=color, style='italic')

    # Icon
    if icon_type == 'problem':
        # X mark
        ax.plot([x+0.9, x+1.1], [y+0.5, y+0.7], 'r-', linewidth=3)
        ax.plot([x+0.9, x+1.1], [y+0.7, y+0.5], 'r-', linewidth=3)
    elif icon_type == 'attempt':
        # Question mark
        ax.text(x+1.0, y+0.6, '?', ha='center', va='center',
                fontsize=18, fontweight='bold', color=color)
    elif icon_type == 'diagnosis':
        # Magnifying glass (circle)
        diag_circle = Circle((x+1.0, y+0.65), 0.15, facecolor='none',
                            edgecolor=color, linewidth=2)
        ax.add_patch(diag_circle)
        ax.plot([x+1.1, x+1.25], [y+0.55, y+0.4], color=color, linewidth=2)
    elif icon_type == 'insight':
        # Light bulb (simplified)
        bulb = Circle((x+1.0, y+0.65), 0.15, facecolor=color, alpha=0.3)
        ax.add_patch(bulb)
        ax.plot([x+0.95, x+1.05], [y+0.45, y+0.45], color=color, linewidth=2)
    elif icon_type == 'breakthrough':
        # Check mark
        ax.plot([x+0.85, x+0.95], [y+0.6, y+0.5], color=color, linewidth=3)
        ax.plot([x+0.95, x+1.15], [y+0.5, y+0.75], color=color, linewidth=3)

# Arrows between steps
# 1 -> 2
arrow1 = FancyArrowPatch((3.2, 7), (3.8, 7),
                        arrowstyle='->', mutation_scale=25,
                        linewidth=3, color=COLOR_MAIN)
ax.add_patch(arrow1)

# 2 -> 3
arrow2 = FancyArrowPatch((6.2, 7), (6.8, 7),
                        arrowstyle='->', mutation_scale=25,
                        linewidth=3, color=COLOR_MAIN)
ax.add_patch(arrow2)

# 3 -> 4 (down)
arrow3 = FancyArrowPatch((8, 6.2), (5.8, 4.8),
                        arrowstyle='->', mutation_scale=25,
                        linewidth=3, color=COLOR_MAIN)
ax.add_patch(arrow3)

# 4 -> 5
arrow4 = FancyArrowPatch((6.2, 4), (6.8, 4),
                        arrowstyle='->', mutation_scale=25,
                        linewidth=3, color=COLOR_MAIN)
ax.add_patch(arrow4)

# Timeline at bottom
timeline_y = 2
ax.plot([1, 14], [timeline_y, timeline_y], 'k-', linewidth=2)

# Timeline markers
timeline_events = [
    (2, '2014-2016\nRNN era'),
    (5, '2016\nFirst attempt'),
    (8, '2016-2017\nDiagnosis'),
    (11, '2017\nBreakthrough'),
]

for x, label in timeline_events:
    ax.plot([x, x], [timeline_y-0.1, timeline_y+0.1], 'k-', linewidth=2)
    ax.text(x, timeline_y-0.5, label, ha='center', va='center',
            fontsize=8, style='italic')

# Key metrics summary at bottom
summary_box = FancyBboxPatch((1, 0.3), 13, 1,
                             boxstyle="round,pad=0.1",
                             facecolor='#E6FFE6', edgecolor=COLOR_GREEN, linewidth=2)
ax.add_patch(summary_box)
ax.text(7.5, 1.0, 'Result: 100x speedup unlocked modern AI', ha='center', va='center',
        fontsize=13, fontweight='bold', color=COLOR_GREEN)
ax.text(7.5, 0.6, '90 days → 1 day | 2% GPU → 92% GPU | $45K → $500 | Enabled ChatGPT, GPT-4, DALL-E', ha='center', va='center',
        fontsize=10, color=COLOR_GREEN)

plt.tight_layout()

# Save figure
output_path = '../figures/sr_25_revolution_journey.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')

plt.close()
