"""
Generate Human Reading Pattern Diagram
Week 5 Transformers - For Slide 12 (Human Introspection)
Shows how humans track position: spatial layout + mental counting
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Ellipse
import numpy as np

# Template colors
COLOR_MAIN = '#333366'
COLOR_BLUE = '#4472C4'
COLOR_PURPLE = '#8B5A9B'
COLOR_GREEN = '#44A044'
COLOR_ORANGE = '#FF7F0E'
COLOR_LIGHT_BG = '#F0F0F0'

# Create figure
fig, (ax_top, ax_mid, ax_bottom) = plt.subplots(3, 1, figsize=(12, 9),
                                                  gridspec_kw={'height_ratios': [1, 1, 1]})
fig.patch.set_facecolor('white')
fig.suptitle('How Humans Track Word Position While Reading', fontsize=14, fontweight='bold', y=0.98)

# TOP PANEL: Eye Movement / Spatial Layout
ax_top.set_xlim(0, 10)
ax_top.set_ylim(0, 10)
ax_top.axis('off')
ax_top.set_title('1. Spatial Layout (Visual Position)', fontsize=11, fontweight='bold',
                  color=COLOR_BLUE, pad=10)

# Sentence words with spatial positions
sentence = "The cat sat on the mat"
words = sentence.split()
x_positions = np.linspace(1, 9, len(words))

for i, (word, x) in enumerate(zip(words, x_positions)):
    # Word box
    word_box = FancyBboxPatch((x-0.5, 6-0.3), 1.0, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor=COLOR_LIGHT_BG, edgecolor=COLOR_BLUE, linewidth=2)
    ax_top.add_patch(word_box)
    ax_top.text(x, 6, word, ha='center', va='center',
                fontsize=10, fontweight='bold')

# Eye movement arrows (saccades)
for i in range(len(words) - 1):
    x1, x2 = x_positions[i], x_positions[i+1]
    arrow = FancyArrowPatch((x1+0.3, 6.8), (x2-0.3, 6.8),
                           arrowstyle='->', mutation_scale=15,
                           linewidth=2, color=COLOR_BLUE, alpha=0.6)
    ax_top.add_patch(arrow)

# Eye icon
eye = Ellipse((0.5, 6), 0.3, 0.15, facecolor='white', edgecolor=COLOR_BLUE, linewidth=2)
ax_top.add_patch(eye)
pupil = Circle((0.5, 6), 0.08, facecolor=COLOR_BLUE)
ax_top.add_patch(pupil)

# Explanation
ax_top.text(5, 4, 'Left-to-right visual scanning', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLOR_BLUE)
ax_top.text(5, 3, 'Position = Location on page', ha='center', va='center',
            fontsize=9, color=COLOR_BLUE)

# MIDDLE PANEL: Mental Position Tracking
ax_mid.set_xlim(0, 10)
ax_mid.set_ylim(0, 10)
ax_mid.axis('off')
ax_mid.set_title('2. Mental Counting (Numerical Position)', fontsize=11, fontweight='bold',
                  color=COLOR_PURPLE, pad=10)

# Same words with numerical labels
for i, (word, x) in enumerate(zip(words, x_positions)):
    # Word box
    word_box = FancyBboxPatch((x-0.5, 6-0.3), 1.0, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor=COLOR_LIGHT_BG, edgecolor=COLOR_PURPLE, linewidth=2)
    ax_mid.add_patch(word_box)
    ax_mid.text(x, 6, word, ha='center', va='center',
                fontsize=10, fontweight='bold')

    # Position number above
    num_circle = Circle((x, 7.5), 0.3, facecolor=COLOR_PURPLE,
                        edgecolor=COLOR_MAIN, linewidth=2, alpha=0.6)
    ax_mid.add_patch(num_circle)
    ax_mid.text(x, 7.5, str(i+1), ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    # Label for position type
    if i == 0:
        ax_mid.text(x, 8.5, '1st', ha='center', va='center',
                    fontsize=8, color=COLOR_PURPLE)
    elif i == 1:
        ax_mid.text(x, 8.5, '2nd', ha='center', va='center',
                    fontsize=8, color=COLOR_PURPLE)
    elif i == 2:
        ax_mid.text(x, 8.5, '3rd', ha='center', va='center',
                    fontsize=8, color=COLOR_PURPLE)
    else:
        ax_mid.text(x, 8.5, f'{i+1}th', ha='center', va='center',
                    fontsize=8, color=COLOR_PURPLE)

# Explanation
ax_mid.text(5, 4, 'Mental tracking: "This is the 1st word, that\'s the 2nd..."', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLOR_PURPLE)
ax_mid.text(5, 3, 'Position = Counting sequence', ha='center', va='center',
            fontsize=9, color=COLOR_PURPLE)

# BOTTOM PANEL: Fusion (Spatial + Numerical)
ax_bottom.set_xlim(0, 10)
ax_bottom.set_ylim(0, 10)
ax_bottom.axis('off')
ax_bottom.set_title('3. Combined Understanding (Meaning + Position)', fontsize=11, fontweight='bold',
                     color=COLOR_GREEN, pad=10)

# Fused representation
for i, (word, x) in enumerate(zip(words, x_positions)):
    # Word box with both spatial and numerical info
    word_box = FancyBboxPatch((x-0.5, 5.5-0.4), 1.0, 0.8,
                              boxstyle="round,pad=0.05",
                              facecolor=COLOR_GREEN, edgecolor=COLOR_MAIN,
                              linewidth=2, alpha=0.3)
    ax_bottom.add_patch(word_box)
    ax_bottom.text(x, 5.8, word, ha='center', va='center',
                   fontsize=10, fontweight='bold')
    ax_bottom.text(x, 5.3, f'#{i+1}', ha='center', va='center',
                   fontsize=8, color=COLOR_GREEN)

# Brain icon
brain_y = 7.5
brain = Ellipse((5, brain_y), 1.5, 0.8, facecolor=COLOR_LIGHT_BG,
                edgecolor=COLOR_GREEN, linewidth=2)
ax_bottom.add_patch(brain)
ax_bottom.text(5, brain_y, 'Brain', ha='center', va='center',
               fontsize=9, fontweight='bold', color=COLOR_GREEN)

# Arrows from brain to words
for i, x in enumerate(x_positions[::2]):  # Every other word
    arrow = FancyArrowPatch((5, brain_y-0.5), (x, 6.2),
                           arrowstyle='->', mutation_scale=12,
                           linewidth=1.5, color=COLOR_GREEN, alpha=0.5,
                           linestyle='dashed')
    ax_bottom.add_patch(arrow)

# Key insight box
insight_box = FancyBboxPatch((1, 2.5), 8, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor='#E6FFE6', edgecolor=COLOR_GREEN, linewidth=2)
ax_bottom.add_patch(insight_box)
ax_bottom.text(5, 3.5, 'You process BOTH simultaneously:', ha='center', va='center',
               fontsize=11, fontweight='bold', color=COLOR_GREEN)
ax_bottom.text(5, 2.9, 'Word meaning + Word position = Complete understanding', ha='center', va='center',
               fontsize=10, color=COLOR_GREEN)

# Analogy at bottom
ax_bottom.text(5, 1, 'Like timestamps on photos: Content + Location = Full information', ha='center', va='center',
               fontsize=9, style='italic', color=COLOR_GREEN)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
output_path = '../figures/sr_19_human_reading_pattern.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')

plt.close()
