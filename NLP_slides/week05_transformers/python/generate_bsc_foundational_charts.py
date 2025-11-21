"""
Generate 10 BSc-Level Foundational Charts for Transformers
Zero pre-knowledge assumed - builds from absolute basics
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pickle

# Load simulation data
with open('simulation_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Colors
COLOR_BLUE = '#4472C4'
COLOR_PURPLE = '#8B5A9B'
COLOR_GREEN = '#44A044'
COLOR_ORANGE = '#FF7F0E'
COLOR_RED = '#D62728'
COLOR_MAIN = '#333366'

print("="*70)
print("Generating 10 BSc Foundational Charts...")
print("="*70)

# ===========================================
# CHART 1: Vector Explanation (2D word space)
# ===========================================
print("\n[1/10] Vector Explanation...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: 2D word space
words_2d = ['cat', 'dog', 'mat', 'rug', 'sat', 'ran']
animal_vals = [0.9, 0.9, 0.0, 0.0, 0.0, 0.0]
furniture_vals = [0.0, 0.0, 0.9, 0.8, 0.0, 0.0]

ax1.scatter(animal_vals, furniture_vals, s=500, c=[COLOR_BLUE, COLOR_BLUE, COLOR_GREEN, COLOR_GREEN, COLOR_PURPLE, COLOR_PURPLE],
           edgecolors=COLOR_MAIN, linewidth=2, alpha=0.7)

for i, word in enumerate(words_2d):
    ax1.annotate(word, (animal_vals[i], furniture_vals[i]), fontsize=12, fontweight='bold', ha='center', va='bottom')

ax1.set_xlabel('Dimension 1: Animal? (0=No, 1=Yes)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Dimension 2: Furniture? (0=No, 1=Yes)', fontsize=11, fontweight='bold')
ax1.set_title('Words as Points in 2D Space', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.2, 1.2)
ax1.set_ylim(-0.2, 1.2)

# Right: Vector as list
ax2.text(0.5, 0.9, '"cat" = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1]',
        fontsize=14, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=COLOR_BLUE, alpha=0.2, pad=1))

# Show dimensions
dims = ['Animal?', 'Object?', 'Action?', 'State?', 'Furniture?', 'Location?', 'Grammar1', 'Grammar2']
vals = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1]

y_pos = 0.7
for i, (dim, val) in enumerate(zip(dims, vals)):
    ax2.text(0.2, y_pos - i*0.08, f'{dim}:', fontsize=10, ha='right', fontweight='bold')
    ax2.text(0.25, y_pos - i*0.08, f'{val:.1f}', fontsize=10, ha='left', color=COLOR_PURPLE, fontweight='bold')

ax2.text(0.5, 0.05, 'A vector is just a LIST of numbers!\nEach number captures one aspect of meaning.',
        fontsize=11, ha='center', style='italic',
        bbox=dict(boxstyle='round', facecolor=COLOR_ORANGE, alpha=0.2, pad=0.8))

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.set_title('Vector = List of Numbers', fontsize=13, fontweight='bold')

plt.suptitle('What is a Vector?', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/sr_bsc_01_vector_explanation.pdf', dpi=300, bbox_inches='tight')
print("   Saved: sr_bsc_01_vector_explanation.pdf")
plt.close()

# ===========================================
# CHART 2: Dimensions Progression (1D→8D)
# ===========================================
print("[2/10] Dimensions Progression...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1D: Temperature
ax = axes[0, 0]
ax.plot([0, 30], [0, 0], 'k-', linewidth=2)
ax.scatter([20], [0], s=300, c=COLOR_RED, edgecolors='black', linewidth=2, zorder=5)
ax.text(20, 0.15, '20°C', fontsize=12, ha='center', fontweight='bold')
ax.set_xlim(-5, 35)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel('Temperature', fontsize=11, fontweight='bold')
ax.set_title('1D: One Number (Temperature)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_yticks([])

# 2D: Map Location
ax = axes[0, 1]
ax.scatter([2.3], [48.9], s=300, c=COLOR_BLUE, edgecolors='black', linewidth=2, marker='*')
ax.text(2.3, 49.2, 'Paris\n(2.3°E, 48.9°N)', fontsize=10, ha='center', fontweight='bold')
ax.set_xlim(-5, 10)
ax.set_ylim(40, 55)
ax.set_xlabel('Longitude (East)', fontsize=11, fontweight='bold')
ax.set_ylabel('Latitude (North)', fontsize=11, fontweight='bold')
ax.set_title('2D: Two Numbers (Location)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 3D: Room Position
ax = axes[1, 0]
ax.remove()
ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.scatter([2], [3], [1.5], s=300, c=COLOR_GREEN, edgecolors='black', linewidth=2)
ax.text(2, 3, 2, 'Object\n(2m, 3m, 1.5m)', fontsize=10, ha='center', fontweight='bold')
ax.set_xlabel('X (meters)', fontsize=10, fontweight='bold')
ax.set_ylabel('Y (meters)', fontsize=10, fontweight='bold')
ax.set_zlabel('Z (meters)', fontsize=10, fontweight='bold')
ax.set_title('3D: Three Numbers (Position)', fontsize=12, fontweight='bold')
ax.view_init(elev=20, azim=45)

# 8D: Word Meaning
ax = axes[1, 1]
dims_8 = ['Animal', 'Object', 'Action', 'State', 'Furniture', 'Location', 'Gram1', 'Gram2']
vals_8 = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1]
colors_8 = [COLOR_PURPLE if v > 0.5 else COLOR_BLUE if v > 0 else 'lightgray' for v in vals_8]
ax.barh(range(8), vals_8, color=colors_8, edgecolor='black', linewidth=1.5)
ax.set_yticks(range(8))
ax.set_yticklabels(dims_8, fontsize=9)
ax.set_xlabel('Value', fontsize=10, fontweight='bold')
ax.set_title('8D: Eight Numbers (Word="cat")', fontsize=12, fontweight='bold')
ax.set_xlim(0, 1)
ax.grid(axis='x', alpha=0.3)

plt.suptitle('Understanding Dimensions: From 1D to 8D', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/sr_bsc_02_dimensions_progression.pdf', dpi=300, bbox_inches='tight')
print("   Saved: sr_bsc_02_dimensions_progression.pdf")
plt.close()

# ===========================================
# CHART 3: Dot Product Visual
# ===========================================
print("[3/10] Dot Product Explanation...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Vector arrows
v1 = np.array([0.5, 0.3])
v2 = np.array([0.4, 0.6])

ax1.arrow(0, 0, v1[0], v1[1], head_width=0.05, head_length=0.05, fc=COLOR_BLUE, ec=COLOR_BLUE, linewidth=3, label='Vector 1: [0.5, 0.3]')
ax1.arrow(0, 0, v2[0], v2[1], head_width=0.05, head_length=0.05, fc=COLOR_RED, ec=COLOR_RED, linewidth=3, label='Vector 2: [0.4, 0.6]')

ax1.set_xlim(-0.1, 0.7)
ax1.set_ylim(-0.1, 0.7)
ax1.set_xlabel('Dimension 1', fontsize=11, fontweight='bold')
ax1.set_ylabel('Dimension 2', fontsize=11, fontweight='bold')
ax1.set_title('Two Vectors', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10, loc='upper left')
ax1.set_aspect('equal')

# Right: Calculation
ax2.text(0.5, 0.85, 'DOT PRODUCT CALCULATION', fontsize=14, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=COLOR_ORANGE, alpha=0.3, pad=0.8))

steps = [
    'Vector 1: [0.5, 0.3]',
    'Vector 2: [0.4, 0.6]',
    '',
    'Dot Product = (0.5 × 0.4) + (0.3 × 0.6)',
    '            = 0.20 + 0.18',
    '            = 0.38',
    '',
    'Higher dot product = More similar!',
    'Lower dot product  = Less similar!'
]

y = 0.7
for step in steps:
    if step.startswith('Dot Product') or step.startswith('            ='):
        ax2.text(0.5, y, step, fontsize=11, ha='center', fontweight='bold', color=COLOR_PURPLE)
    elif 'Higher' in step or 'Lower' in step:
        ax2.text(0.5, y, step, fontsize=10, ha='center', style='italic', color=COLOR_GREEN if 'Higher' in step else COLOR_RED)
    else:
        ax2.text(0.5, y, step, fontsize=10, ha='center')
    y -= 0.09

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.set_title('Step-by-Step Calculation', fontsize=13, fontweight='bold')

plt.suptitle('The Math: Dot Product Explained', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/sr_bsc_03_dot_product_visual.pdf', dpi=300, bbox_inches='tight')
print("   Saved: sr_bsc_03_dot_product_visual.pdf")
plt.close()

# ===========================================
# CHART 4: Softmax Transformation
# ===========================================
print("[4/10] Softmax Transformation...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Input scores
raw_scores = np.array([2.1, 1.5, 3.2, 0.8])
words_softmax = ['The', 'cat', 'sat', 'on']

# Step 1: Raw scores
ax = axes[0, 0]
ax.barh(range(4), raw_scores, color=COLOR_BLUE, edgecolor='black', linewidth=2)
ax.set_yticks(range(4))
ax.set_yticklabels(words_softmax, fontsize=11, fontweight='bold')
ax.set_xlabel('Raw Score', fontsize=11, fontweight='bold')
ax.set_title('Step 1: Raw Attention Scores', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, val in enumerate(raw_scores):
    ax.text(val + 0.1, i, f'{val:.1f}', va='center', fontsize=10, fontweight='bold')

# Step 2: Exponentiate
exp_scores = np.exp(raw_scores)
ax = axes[0, 1]
ax.barh(range(4), exp_scores, color=COLOR_PURPLE, edgecolor='black', linewidth=2)
ax.set_yticks(range(4))
ax.set_yticklabels(words_softmax, fontsize=11, fontweight='bold')
ax.set_xlabel('After exp()', fontsize=11, fontweight='bold')
ax.set_title('Step 2: Exponentiate', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, val in enumerate(exp_scores):
    ax.text(val + 1, i, f'{val:.1f}', va='center', fontsize=10, fontweight='bold')

# Step 3: Divide by sum
softmax_scores = exp_scores / np.sum(exp_scores)
ax = axes[1, 0]
ax.barh(range(4), softmax_scores * 100, color=COLOR_GREEN, edgecolor='black', linewidth=2)
ax.set_yticks(range(4))
ax.set_yticklabels(words_softmax, fontsize=11, fontweight='bold')
ax.set_xlabel('Probability (%)', fontsize=11, fontweight='bold')
ax.set_title('Step 3: Divide by Sum → Probabilities!', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, val in enumerate(softmax_scores):
    ax.text(val * 100 + 2, i, f'{val:.1%}', va='center', fontsize=10, fontweight='bold')

# Step 4: Pie chart
ax = axes[1, 1]
colors_pie = [COLOR_BLUE, COLOR_PURPLE, COLOR_GREEN, COLOR_ORANGE]
ax.pie(softmax_scores, labels=words_softmax, autopct='%1.0f%%', colors=colors_pie, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax.set_title('Sum = 100%!', fontsize=12, fontweight='bold')

plt.suptitle('What is Softmax? (Converts Any Numbers → Probabilities)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/sr_bsc_04_softmax_transformation.pdf', dpi=300, bbox_inches='tight')
print("   Saved: sr_bsc_04_softmax_transformation.pdf")
plt.close()

# ===========================================
# CHART 5: Attention Computation Walkthrough
# ===========================================
print("[5/10] Attention Computation Walkthrough...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Load simulation data
words = data['input_sentence']

# Step 1: Q and K matrices (simplified 2x2 for clarity)
ax = axes[0, 0]
Q_example = np.array([[0.5, 0.3], [0.2, 0.8]])
K_example = np.array([[0.4, 0.6], [0.7, 0.1]])
ax.text(0.5, 0.8, 'Query (Q)', fontsize=12, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=COLOR_BLUE, alpha=0.3, pad=0.8))
ax.text(0.5, 0.6, 'Word 1: [0.5, 0.3]\nWord 2: [0.2, 0.8]', fontsize=10, ha='center', fontfamily='monospace')
ax.text(0.5, 0.3, 'Key (K)', fontsize=12, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, alpha=0.3, pad=0.8))
ax.text(0.5, 0.1, 'Word 1: [0.4, 0.6]\nWord 2: [0.7, 0.1]', fontsize=10, ha='center', fontfamily='monospace')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Step 1: Q and K Matrices', fontsize=12, fontweight='bold')

# Step 2: Matrix multiplication
ax = axes[0, 1]
scores = Q_example @ K_example.T
im = ax.imshow(scores, cmap='Blues', aspect='auto', vmin=0, vmax=1)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Word 1', 'Word 2'], fontsize=10)
ax.set_yticklabels(['Word 1', 'Word 2'], fontsize=10)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{scores[i,j]:.2f}', ha='center', va='center', fontsize=12, fontweight='bold')
ax.set_title('Step 2: Q @ K^T (Attention Scores)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax)

# Step 3: Apply softmax
ax = axes[1, 0]
from scipy.special import softmax as scipy_softmax
weights = scipy_softmax(scores, axis=1)
im = ax.imshow(weights, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Word 1', 'Word 2'], fontsize=10)
ax.set_yticklabels(['Word 1', 'Word 2'], fontsize=10)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{weights[i,j]:.0%}', ha='center', va='center', fontsize=12, fontweight='bold')
ax.set_title('Step 3: Softmax (Probabilities)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax)

# Step 4: Weighted sum with V
ax = axes[1, 1]
V_example = np.array([[0.9, 0.2], [0.3, 0.7]])
output = weights @ V_example
ax.text(0.5, 0.7, 'Output = Attention @ V', fontsize=12, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=COLOR_PURPLE, alpha=0.3, pad=0.8))
ax.text(0.5, 0.5, f'Word 1 output:\n[{output[0,0]:.2f}, {output[0,1]:.2f}]', fontsize=11, ha='center', fontfamily='monospace')
ax.text(0.5, 0.3, f'Word 2 output:\n[{output[1,0]:.2f}, {output[1,1]:.2f}]', fontsize=11, ha='center', fontfamily='monospace')
ax.text(0.5, 0.05, 'This is the context-aware representation!', fontsize=9, ha='center', style='italic')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Step 4: Weighted Sum with V', fontsize=12, fontweight='bold')

plt.suptitle('Attention Mechanism: Complete Walkthrough', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/sr_bsc_05_attention_walkthrough.pdf', dpi=300, bbox_inches='tight')
print("   Saved: sr_bsc_05_attention_walkthrough.pdf")
plt.close()

# ===========================================
# CHART 6: Multi-Head Perspectives
# ===========================================
print("[6/10] Multi-Head Perspectives...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sentence = "The cat sat on the"
words_list = sentence.split()

# Head 1: Grammar focus
ax = axes[0, 0]
colors_h1 = [COLOR_BLUE if w in ['The', 'the'] else COLOR_ORANGE if w == 'on' else 'lightgray' for w in words_list]
for i, (word, color) in enumerate(zip(words_list, colors_h1)):
    ax.text(i*0.18 + 0.1, 0.5, word, fontsize=14, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.6, pad=0.5))
ax.text(0.5, 0.8, 'HEAD 1: Grammar Patterns', fontsize=12, ha='center', fontweight='bold')
ax.text(0.5, 0.2, 'Focuses on: Articles, prepositions', fontsize=10, ha='center', style='italic')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Head 2: Semantics focus
ax = axes[0, 1]
colors_h2 = [COLOR_GREEN if w == 'cat' else COLOR_PURPLE if w in ['sat', 'on'] else 'lightgray' for w in words_list]
for i, (word, color) in enumerate(zip(words_list, colors_h2)):
    ax.text(i*0.18 + 0.1, 0.5, word, fontsize=14, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.6, pad=0.5))
ax.text(0.5, 0.8, 'HEAD 2: Semantic Relationships', fontsize=12, ha='center', fontweight='bold')
ax.text(0.5, 0.2, 'Focuses on: Subject, action, location', fontsize=10, ha='center', style='italic')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Head 3: Position focus
ax = axes[1, 0]
colors_h3 = [COLOR_RED if i in [2, 3] else 'lightgray' for i in range(len(words_list))]
for i, (word, color) in enumerate(zip(words_list, colors_h3)):
    ax.text(i*0.18 + 0.1, 0.5, word, fontsize=14, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.6, pad=0.5))
ax.text(0.5, 0.8, 'HEAD 3: Positional Patterns', fontsize=12, ha='center', fontweight='bold')
ax.text(0.5, 0.2, 'Focuses on: Nearby words', fontsize=10, ha='center', style='italic')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Head 4: Global focus
ax = axes[1, 1]
colors_h4 = [COLOR_ORANGE if i in [0, 4] else 'lightgray' for i in range(len(words_list))]
for i, (word, color) in enumerate(zip(words_list, colors_h4)):
    ax.text(i*0.18 + 0.1, 0.5, word, fontsize=14, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.6, pad=0.5))
ax.text(0.5, 0.8, 'HEAD 4: Global Context', fontsize=12, ha='center', fontweight='bold')
ax.text(0.5, 0.2, 'Focuses on: Sentence boundaries', fontsize=10, ha='center', style='italic')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.suptitle('Why 4 Heads? Different Perspectives on Same Sentence', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/sr_bsc_06_multihead_perspectives.pdf', dpi=300, bbox_inches='tight')
print("   Saved: sr_bsc_06_multihead_perspectives.pdf")
plt.close()

# ===========================================
# CHART 7: Concatenation Visual (LEGO blocks)
# ===========================================
print("[7/10] Concatenation Visual...")

fig, ax = plt.subplots(figsize=(12, 8))

# Draw 4 separate head outputs as blocks
head_colors = [COLOR_BLUE, COLOR_PURPLE, COLOR_GREEN, COLOR_ORANGE]
head_labels = ['Head 1\n[2 dims]', 'Head 2\n[2 dims]', 'Head 3\n[2 dims]', 'Head 4\n[2 dims]']

y_start = 0.7
for i, (color, label) in enumerate(zip(head_colors, head_labels)):
    rect = mpatches.FancyBboxPatch((0.1 + i*0.2, y_start), 0.15, 0.15,
                                   boxstyle="round,pad=0.01",
                                   edgecolor='black', facecolor=color, linewidth=3, alpha=0.7)
    ax.add_patch(rect)
    ax.text(0.175 + i*0.2, y_start + 0.075, label, fontsize=10, ha='center', va='center', fontweight='bold')

# Arrow pointing down
ax.annotate('', xy=(0.5, 0.5), xytext=(0.5, 0.65),
            arrowprops=dict(arrowstyle='->', lw=3, color='black'))
ax.text(0.52, 0.575, 'CONCATENATE', fontsize=11, fontweight='bold', rotation=-90)

# Combined output as single long block
combined_rect = mpatches.FancyBboxPatch((0.1, 0.3), 0.8, 0.15,
                                        boxstyle="round,pad=0.01",
                                        edgecolor='black', facecolor=COLOR_MAIN, linewidth=3, alpha=0.3)
ax.add_patch(combined_rect)
ax.text(0.5, 0.375, 'Combined Output\n[2+2+2+2 = 8 dimensions]', fontsize=12, ha='center', va='center', fontweight='bold')

# Bottom explanation
ax.text(0.5, 0.1, 'Like stacking LEGO blocks side-by-side!\nEach head contributes 2 dimensions, total = 8 dimensions',
        fontsize=11, ha='center', style='italic',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.2, pad=0.8))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Concatenation: Combining All Head Outputs', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/sr_bsc_07_concatenation_visual.pdf', dpi=300, bbox_inches='tight')
print("   Saved: sr_bsc_07_concatenation_visual.pdf")
plt.close()

# ===========================================
# CHART 8: Complete Pipeline with Numbers
# ===========================================
print("[8/10] Complete Pipeline with Numbers...")

fig = plt.figure(figsize=(16, 10))

# Vertical flowchart with actual numbers
steps = [
    ('INPUT', 'The cat sat on the ___', COLOR_BLUE),
    ('STEP 1: Embeddings', 'cat = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1]', COLOR_PURPLE),
    ('STEP 2: + Position', 'Pos 1 = [0.84, 0.54, 0.01, 1.00, ...]', COLOR_GREEN),
    ('STEP 3: Attention', 'Q @ K^T / sqrt(2) -> Softmax', COLOR_ORANGE),
    ('STEP 4: Multi-Head (4 heads)', 'Each head: 2 dimensions\nTotal: 8 dimensions', COLOR_RED),
    ('STEP 5: Concatenate', '[Head1 | Head2 | Head3 | Head4]', COLOR_PURPLE),
    ('OUTPUT', 'Top predictions:\nmat: 6.9%\nsofa: 7.0%\nchair: 6.7%', COLOR_GREEN)
]

y_pos = 0.9
for i, (title, content, color) in enumerate(steps):
    # Box
    rect = mpatches.FancyBboxPatch((0.15, y_pos - 0.08), 0.7, 0.12,
                                   boxstyle="round,pad=0.01",
                                   edgecolor='black', facecolor=color, linewidth=2, alpha=0.3)
    fig.gca().add_patch(rect)

    # Text
    fig.gca().text(0.2, y_pos - 0.02, title, fontsize=11, fontweight='bold', va='top')
    fig.gca().text(0.5, y_pos - 0.05, content, fontsize=9, ha='center', va='top', fontfamily='monospace')

    # Arrow
    if i < len(steps) - 1:
        fig.gca().annotate('', xy=(0.5, y_pos - 0.09), xytext=(0.5, y_pos - 0.12),
                          arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    y_pos -= 0.14

fig.gca().set_xlim(0, 1)
fig.gca().set_ylim(0, 1)
fig.gca().axis('off')
fig.gca().set_title('Complete Transformer Pipeline: Input to Output with Example Numbers', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('../figures/sr_bsc_08_complete_pipeline_numbers.pdf', dpi=300, bbox_inches='tight')
print("   Saved: sr_bsc_08_complete_pipeline_numbers.pdf")
plt.close()

# ===========================================
# CHART 9: Common Misconceptions Table
# ===========================================
print("[9/10] Common Misconceptions Table...")

fig, ax = plt.subplots(figsize=(14, 10))

misconceptions = [
    ('Vectors are arrows', 'Vectors are LISTS of numbers\n(arrows are just one visualization)'),
    ('Attention copies words', 'Attention creates WEIGHTED AVERAGES\n(blends information, not copies)'),
    ('Softmax changes total value', 'Softmax creates PROBABILITIES\n(sum always = 100%)'),
    ('More heads = more accuracy', 'More heads = more PERSPECTIVES\n(not automatically better)'),
    ('Transformers understand meaning', 'Transformers find PATTERNS\n(statistical correlations, not understanding)'),
    ('Position encoding adds dimensions', 'Position encoding MODIFIES existing dims\n(same 8 dimensions before and after)')
]

y_pos = 0.95
row_height = 0.15

# Header
ax.text(0.25, y_pos, 'WRONG', fontsize=14, ha='center', fontweight='bold', color='red')
ax.text(0.75, y_pos, 'CORRECT', fontsize=14, ha='center', fontweight='bold', color='green')
ax.plot([0.5, 0.5], [0.05, 0.92], 'k-', linewidth=2)

y_pos -= 0.05

for wrong, right in misconceptions:
    # Wrong side (red background)
    rect_wrong = mpatches.FancyBboxPatch((0.05, y_pos - row_height + 0.01), 0.4, row_height - 0.02,
                                         boxstyle="round,pad=0.005",
                                         edgecolor='red', facecolor='red', linewidth=2, alpha=0.1)
    ax.add_patch(rect_wrong)
    ax.text(0.25, y_pos - row_height/2, wrong, fontsize=10, ha='center', va='center', wrap=True)

    # Right side (green background)
    rect_right = mpatches.FancyBboxPatch((0.55, y_pos - row_height + 0.01), 0.4, row_height - 0.02,
                                         boxstyle="round,pad=0.005",
                                         edgecolor='green', facecolor='green', linewidth=2, alpha=0.1)
    ax.add_patch(rect_right)
    ax.text(0.75, y_pos - row_height/2, right, fontsize=9, ha='center', va='center')

    y_pos -= row_height

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Common Mistakes Students Make', fontsize=15, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('../figures/sr_bsc_09_misconceptions_table.pdf', dpi=300, bbox_inches='tight')
print("   Saved: sr_bsc_09_misconceptions_table.pdf")
plt.close()

# ===========================================
# CHART 10: Quiz Layout
# ===========================================
print("[10/10] Quiz Layout...")

fig, ax = plt.subplots(figsize=(12, 10))

# Title
ax.text(0.5, 0.95, 'Quick Quiz: Test Your Understanding', fontsize=16, ha='center', fontweight='bold')

# Question 1
y_pos = 0.85
ax.text(0.5, y_pos, 'Question 1: What does softmax do?', fontsize=12, ha='center', fontweight='bold')
y_pos -= 0.08
options_q1 = [
    'A) Makes numbers bigger',
    'B) Converts any numbers into probabilities (sum = 100%)',
    'C) Removes negative numbers',
    'D) Sorts numbers in order'
]
for i, opt in enumerate(options_q1):
    color = COLOR_GREEN if i == 1 else COLOR_BLUE
    alpha = 0.3 if i == 1 else 0.1
    rect = mpatches.FancyBboxPatch((0.15, y_pos - 0.04), 0.7, 0.035,
                                   boxstyle="round,pad=0.003",
                                   edgecolor='black', facecolor=color, linewidth=1, alpha=alpha)
    ax.add_patch(rect)
    ax.text(0.17, y_pos - 0.0225, opt, fontsize=10, va='center')
    y_pos -= 0.05

# Question 2
y_pos -= 0.05
ax.text(0.5, y_pos, 'Question 2: Why do we need multiple attention heads?', fontsize=12, ha='center', fontweight='bold')
y_pos -= 0.08
options_q2 = [
    'A) To make computation faster',
    'B) To capture different perspectives (grammar, semantics, position)',
    'C) To use more GPU memory',
    'D) To make model bigger'
]
for i, opt in enumerate(options_q2):
    color = COLOR_GREEN if i == 1 else COLOR_BLUE
    alpha = 0.3 if i == 1 else 0.1
    rect = mpatches.FancyBboxPatch((0.15, y_pos - 0.04), 0.7, 0.035,
                                   boxstyle="round,pad=0.003",
                                   edgecolor='black', facecolor=color, linewidth=1, alpha=alpha)
    ax.add_patch(rect)
    ax.text(0.17, y_pos - 0.0225, opt, fontsize=10, va='center')
    y_pos -= 0.05

# Question 3
y_pos -= 0.05
ax.text(0.5, y_pos, 'Question 3: What is a vector?', fontsize=12, ha='center', fontweight='bold')
y_pos -= 0.08
options_q3 = [
    'A) An arrow in 2D space',
    'B) A list of numbers that represent meaning',
    'C) A matrix',
    'D) A word'
]
for i, opt in enumerate(options_q3):
    color = COLOR_GREEN if i == 1 else COLOR_BLUE
    alpha = 0.3 if i == 1 else 0.1
    rect = mpatches.FancyBboxPatch((0.15, y_pos - 0.04), 0.7, 0.035,
                                   boxstyle="round,pad=0.003",
                                   edgecolor='black', facecolor=color, linewidth=1, alpha=alpha)
    ax.add_patch(rect)
    ax.text(0.17, y_pos - 0.0225, opt, fontsize=10, va='center')
    y_pos -= 0.05

# Bottom note
ax.text(0.5, 0.08, 'Correct answers highlighted in green!\n(In actual quiz, answers would be hidden)',
        fontsize=9, ha='center', style='italic',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.2, pad=0.5))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
plt.savefig('../figures/sr_bsc_10_quiz_layout.pdf', dpi=300, bbox_inches='tight')
print("   Saved: sr_bsc_10_quiz_layout.pdf")
plt.close()

print("\n" + "="*70)
print("[OK] All 10 BSc foundational charts generated successfully!")
print("="*70)
