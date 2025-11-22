"""
Generate attention computation flow diagram
Shows step-by-step how attention scores are calculated with actual numbers
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure
fig, ax = plt.subplots(figsize=(16, 12))

# Remove axes
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Attention Computation: Step-by-Step Flow',
        fontsize=18, fontweight='bold', ha='center')

# Define colors
COLOR_QUERY = '#FF6B6B'
COLOR_KEY = '#4ECDC4'
COLOR_VALUE = '#95E77E'
COLOR_ATTENTION = '#FFE66D'
COLOR_OUTPUT = '#B19CD9'

# Words in our example
words = ['The', 'cat', 'sat', 'on', 'the', 'mat']
target_word = 'mat'

# Step 1: Query and Keys
y_step1 = 8.0
ax.text(1, y_step1, 'STEP 1:', fontsize=14, fontweight='bold')
ax.text(2.2, y_step1, 'Query from "mat" meets all Keys', fontsize=12)

# Draw Query box
query_box = FancyBboxPatch((0.5, y_step1 - 0.8), 1.5, 0.6,
                           boxstyle="round,pad=0.05",
                           facecolor=COLOR_QUERY, alpha=0.3,
                           edgecolor=COLOR_QUERY, linewidth=2)
ax.add_patch(query_box)
ax.text(1.25, y_step1 - 0.5, 'Q("mat")\n[0.8, 0.6, 0.4]',
        fontsize=10, ha='center', fontweight='bold')

# Draw Keys
key_positions = np.linspace(2.5, 8, len(words) - 1)
for i, (word, x_pos) in enumerate(zip(words[:-1], key_positions)):
    key_box = FancyBboxPatch((x_pos - 0.3, y_step1 - 0.8), 0.6, 0.6,
                             boxstyle="round,pad=0.02",
                             facecolor=COLOR_KEY, alpha=0.3,
                             edgecolor=COLOR_KEY, linewidth=1)
    ax.add_patch(key_box)
    ax.text(x_pos, y_step1 - 0.5, f'K("{word}")',
            fontsize=9, ha='center')

# Step 2: Dot Products
y_step2 = 6.5
ax.text(1, y_step2, 'STEP 2:', fontsize=14, fontweight='bold')
ax.text(2.2, y_step2, 'Calculate Q · K (dot products)', fontsize=12)

# Show actual computation
scores_raw = [0.1, 0.3, 0.4, 0.8, 0.6]
words_short = ['The', 'cat', 'sat', 'on', 'the']

for i, (word, x_pos, score) in enumerate(zip(words_short, key_positions, scores_raw)):
    # Draw arrow from Query to Key
    arrow = FancyArrowPatch((1.25, y_step1 - 0.8), (x_pos, y_step2 + 0.3),
                           arrowstyle='->', lw=1, color='gray', alpha=0.3)
    ax.add_artist(arrow)

    # Draw score box
    score_color = 'red' if score >= 0.6 else 'orange' if score >= 0.3 else 'lightgray'
    score_box = FancyBboxPatch((x_pos - 0.3, y_step2 - 0.4), 0.6, 0.4,
                               boxstyle="round,pad=0.02",
                               facecolor=score_color, alpha=0.3,
                               edgecolor='black', linewidth=1)
    ax.add_patch(score_box)
    ax.text(x_pos, y_step2 - 0.2, f'{score:.1f}',
            fontsize=11, ha='center', fontweight='bold')

# Step 3: Softmax
y_step3 = 5.0
ax.text(1, y_step3, 'STEP 3:', fontsize=14, fontweight='bold')
ax.text(2.2, y_step3, 'Apply Softmax (convert to percentages)', fontsize=12)

# Show softmax formula
ax.text(5, y_step3 - 0.5, r'softmax(x)$_i$ = $\frac{e^{x_i}}{\sum_j e^{x_j}}$',
        fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

# Calculate actual softmax
scores_exp = np.exp(scores_raw)
attention_weights = scores_exp / scores_exp.sum()

# Draw attention weight boxes
for i, (word, x_pos, weight) in enumerate(zip(words_short, key_positions, attention_weights)):
    # Draw arrow from raw score to weight
    arrow = FancyArrowPatch((x_pos, y_step2 - 0.4), (x_pos, y_step3 + 0.3),
                           arrowstyle='->', lw=1.5, color='black', alpha=0.3)
    ax.add_artist(arrow)

    # Draw weight box
    weight_box = FancyBboxPatch((x_pos - 0.35, y_step3 - 0.4), 0.7, 0.4,
                                boxstyle="round,pad=0.02",
                                facecolor=COLOR_ATTENTION, alpha=weight,
                                edgecolor='black', linewidth=1)
    ax.add_patch(weight_box)
    ax.text(x_pos, y_step3 - 0.2, f'{weight:.0%}',
            fontsize=11, ha='center', fontweight='bold')

# Step 4: Weight Values
y_step4 = 3.5
ax.text(1, y_step4, 'STEP 4:', fontsize=14, fontweight='bold')
ax.text(2.2, y_step4, 'Multiply weights with Values', fontsize=12)

# Draw Values
for i, (word, x_pos, weight) in enumerate(zip(words_short, key_positions, attention_weights)):
    # Draw Value box
    value_box = FancyBboxPatch((x_pos - 0.3, y_step4 - 0.4), 0.6, 0.4,
                               boxstyle="round,pad=0.02",
                               facecolor=COLOR_VALUE, alpha=0.3,
                               edgecolor=COLOR_VALUE, linewidth=1)
    ax.add_patch(value_box)
    ax.text(x_pos, y_step4 - 0.2, f'V("{word}")',
            fontsize=9, ha='center')

    # Show multiplication
    ax.text(x_pos, y_step4 - 0.7, f'× {weight:.0%}',
            fontsize=9, ha='center', color='gray')

# Step 5: Sum
y_step5 = 2.0
ax.text(1, y_step5, 'STEP 5:', fontsize=14, fontweight='bold')
ax.text(2.2, y_step5, 'Sum all weighted values', fontsize=12)

# Draw summation visualization
sum_parts = []
for i, (word, weight) in enumerate(zip(words_short, attention_weights)):
    if weight > 0.05:  # Only show significant contributions
        sum_parts.append(f'{weight:.0%}×V("{word}")')

formula = ' + '.join(sum_parts[:3]) + ' + ...'
ax.text(5, y_step5 - 0.3, formula, fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# Final Output
y_output = 0.5
output_box = FancyBboxPatch((3.5, y_output - 0.5), 3, 0.8,
                            boxstyle="round,pad=0.05",
                            facecolor=COLOR_OUTPUT, alpha=0.3,
                            edgecolor=COLOR_OUTPUT, linewidth=3)
ax.add_patch(output_box)
ax.text(5, y_output - 0.1, 'OUTPUT: Context-aware representation of "mat"',
        fontsize=13, ha='center', fontweight='bold')
ax.text(5, y_output - 0.35, 'Knows it follows "on the" (location pattern)',
        fontsize=10, ha='center', style='italic')

# Add arrows showing flow
flow_y = [y_step1 - 0.5, y_step2 - 0.2, y_step3 - 0.2, y_step4 - 0.2, y_step5 - 0.2, y_output]
for i in range(len(flow_y) - 1):
    arrow = FancyArrowPatch((0.3, flow_y[i]), (0.3, flow_y[i+1]),
                           arrowstyle='->', lw=3, color='purple', alpha=0.3)
    ax.add_artist(arrow)

# Add side annotations
ax.text(9, y_step2, 'Higher score\n= more relevant',
        fontsize=9, color='red', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax.text(9, y_step3, 'Sum = 100%',
        fontsize=9, color='orange', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax.text(9, y_step4, 'Each Value\ncontributes\nproportionally',
        fontsize=9, color='green', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Add key insight box
ax.text(5, -0.5,
        'Key Insight: Attention learns to identify and combine relevant information automatically',
        fontsize=12, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('../figures/attention_flow_diagram.pdf', dpi=300, bbox_inches='tight')
print("Generated: attention_flow_diagram.pdf")
plt.close()