"""
Generate 8 additional mathematical/theoretical charts for Week 5
BSc-level mathematical concepts for understanding transformers
Created: 2025-09-28
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from scipy.special import softmax

os.makedirs('../figures', exist_ok=True)

# Academic color scheme
color_primary = '#2C3E50'  # Dark blue-gray
color_secondary = '#E74C3C'  # Red
color_accent = '#3498DB'  # Blue
color_success = '#27AE60'  # Green
color_warning = '#F39C12'  # Orange
color_neutral = '#95A5A6'  # Gray

plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def save_fig(filename):
    plt.savefig(f'../figures/{filename}', dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Generated: {filename}')
    plt.close()

print("Generating Additional Mathematical Charts for Week 5...")
print("="*60)

# Chart 13: Softmax Function Visualization
print("\nChart 13: Softmax Function")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Raw scores
words = ['cat', 'dog', 'mat', 'table', 'chair']
raw_scores = np.array([2.1, -0.5, 4.2, 1.3, 0.8])
probabilities = softmax(raw_scores)

# Before softmax
ax1.bar(words, raw_scores, color=color_neutral, edgecolor='black', linewidth=1.5)
ax1.set_title('Raw Scores (Logits)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=11)
ax1.axhline(y=0, color='black', linewidth=0.5)
ax1.set_ylim(-1, 5)

# Add values on bars
for i, (word, score) in enumerate(zip(words, raw_scores)):
    ax1.text(i, score + 0.2 if score > 0 else score - 0.3,
            f'{score:.1f}', ha='center', fontsize=10, fontweight='bold')

# After softmax
colors = [color_success if p == max(probabilities) else color_accent for p in probabilities]
ax2.bar(words, probabilities, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_title('After Softmax → Probabilities', fontsize=12, fontweight='bold')
ax2.set_ylabel('Probability', fontsize=11)
ax2.set_ylim(0, 0.8)

# Add probability values
for i, (word, prob) in enumerate(zip(words, probabilities)):
    ax2.text(i, prob + 0.02, f'{prob:.1%}', ha='center', fontsize=10, fontweight='bold')

# Add formula
fig.text(0.5, 0.02, r'Softmax: $P(word_i) = \frac{e^{score_i}}{\sum_j e^{score_j}}$',
         ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.suptitle('Softmax: Converting Scores to Probabilities', fontsize=14, fontweight='bold')
save_fig('math_13_softmax_function.pdf')

# Chart 14: Cross-Entropy Loss
print("Chart 14: Cross-Entropy Loss")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Loss vs probability of correct word
probs = np.linspace(0.01, 0.99, 100)
loss = -np.log(probs)

ax1.plot(probs, loss, linewidth=2.5, color=color_secondary)
ax1.set_xlabel('P(correct word)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Loss = -log P', fontsize=11, fontweight='bold')
ax1.set_title('Cross-Entropy Loss Function', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 5)
ax1.grid(True, alpha=0.3)

# Mark key points
ax1.plot(0.1, -np.log(0.1), 'ro', markersize=8)
ax1.text(0.1, -np.log(0.1) + 0.3, 'High loss\n(bad prediction)', ha='center', fontsize=9)
ax1.plot(0.9, -np.log(0.9), 'go', markersize=8)
ax1.text(0.9, -np.log(0.9) + 0.5, 'Low loss\n(good prediction)', ha='center', fontsize=9)

# Training curves comparison
epochs = np.arange(1, 21)
rnn_loss = 3.0 * np.exp(-0.1 * epochs) + 1.5
transformer_loss = 3.0 * np.exp(-0.2 * epochs) + 0.8

ax2.plot(epochs, rnn_loss, 'o-', color=color_secondary, linewidth=2,
         markersize=5, label='RNN', markeredgecolor='black')
ax2.plot(epochs, transformer_loss, 's-', color=color_success, linewidth=2,
         markersize=5, label='Transformer', markeredgecolor='black')
ax2.set_xlabel('Training Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax2.set_title('Training Loss Over Time', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.set_ylim(0, 4)
ax2.grid(True, alpha=0.3)

plt.suptitle('Cross-Entropy Loss for Next Word Prediction', fontsize=14, fontweight='bold')
save_fig('math_14_cross_entropy_loss.pdf')

# Chart 15: Attention Score Calculation
print("Chart 15: Attention Score Calculation")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Step 1: Q and K matrices
Q = np.array([[1, 0], [0, 1], [1, 1]])
K = np.array([[1, 0], [0, 1], [1, 1]])
words = ['The', 'cat', 'sat']

# Display Q
ax1.imshow(Q, cmap='Blues', aspect='auto', vmin=-1, vmax=2)
ax1.set_title('Query (Q)', fontsize=12, fontweight='bold')
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['d1', 'd2'])
ax1.set_yticks([0, 1, 2])
ax1.set_yticklabels(words)
for i in range(3):
    for j in range(2):
        ax1.text(j, i, f'{Q[i,j]:.0f}', ha='center', va='center', fontsize=10)

# Display K^T
KT = K.T
ax2.imshow(KT, cmap='Greens', aspect='auto', vmin=-1, vmax=2)
ax2.set_title('Key Transpose (K^T)', fontsize=12, fontweight='bold')
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(words)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['d1', 'd2'])
for i in range(2):
    for j in range(3):
        ax2.text(j, i, f'{KT[i,j]:.0f}', ha='center', va='center', fontsize=10)

# Step 2: Q @ K^T
scores = Q @ K.T
ax3.imshow(scores, cmap='Oranges', aspect='auto', vmin=0, vmax=3)
ax3.set_title('Attention Scores (Q·K^T)', fontsize=12, fontweight='bold')
ax3.set_xticks([0, 1, 2])
ax3.set_xticklabels(words)
ax3.set_yticks([0, 1, 2])
ax3.set_yticklabels(words)
for i in range(3):
    for j in range(3):
        ax3.text(j, i, f'{scores[i,j]:.0f}', ha='center', va='center',
                fontsize=11, fontweight='bold')

# Step 3: Softmax
attention_weights = softmax(scores / np.sqrt(2), axis=1)
ax4.imshow(attention_weights, cmap='Reds', aspect='auto', vmin=0, vmax=1)
ax4.set_title('Attention Weights (after Softmax)', fontsize=12, fontweight='bold')
ax4.set_xticks([0, 1, 2])
ax4.set_xticklabels(words)
ax4.set_yticks([0, 1, 2])
ax4.set_yticklabels(words)
for i in range(3):
    for j in range(3):
        ax4.text(j, i, f'{attention_weights[i,j]:.2f}', ha='center', va='center',
                fontsize=11, fontweight='bold')

plt.suptitle('Step-by-Step Attention Calculation', fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig('math_15_attention_calculation.pdf')

# Chart 16: Word Embedding Space (2D)
print("Chart 16: Word Embedding Space")
fig, ax = plt.subplots(figsize=(10, 8))

# Create word embeddings in 2D (simplified)
np.random.seed(42)
words_data = {
    # Animals cluster
    'cat': (2, 3),
    'dog': (2.5, 3.2),
    'mouse': (1.8, 2.8),
    'bird': (2.2, 2.5),

    # Vehicles cluster
    'car': (-2, 1),
    'truck': (-2.3, 0.8),
    'bike': (-1.7, 1.2),
    'bus': (-2.1, 0.5),

    # Actions cluster
    'run': (1, -2),
    'walk': (0.8, -1.8),
    'jump': (1.2, -2.2),
    'sit': (0.5, -1.5),

    # Objects cluster
    'table': (-1, -1),
    'chair': (-0.8, -0.8),
    'desk': (-1.2, -1.2),
    'lamp': (-0.5, -1.0)
}

# Plot clusters with different colors
cluster_colors = {
    'Animals': (color_success, ['cat', 'dog', 'mouse', 'bird']),
    'Vehicles': (color_accent, ['car', 'truck', 'bike', 'bus']),
    'Actions': (color_warning, ['run', 'walk', 'jump', 'sit']),
    'Objects': (color_secondary, ['table', 'chair', 'desk', 'lamp'])
}

for cluster_name, (color, words) in cluster_colors.items():
    xs = [words_data[w][0] for w in words]
    ys = [words_data[w][1] for w in words]
    ax.scatter(xs, ys, c=color, s=200, alpha=0.6, edgecolors='black',
              linewidth=2, label=cluster_name)

    # Add word labels
    for word in words:
        x, y = words_data[word]
        ax.annotate(word, (x, y), ha='center', va='center',
                   fontsize=10, fontweight='bold')

# Draw distance lines for similar words
ax.plot([2, 2.5], [3, 3.2], 'k--', alpha=0.3, linewidth=1)
ax.text(2.25, 3.1, 'd=0.5', fontsize=8, ha='center')

ax.set_xlabel('Embedding Dimension 1', fontsize=11, fontweight='bold')
ax.set_ylabel('Embedding Dimension 2', fontsize=11, fontweight='bold')
ax.set_title('Word Embeddings in 2D Space', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 4)

# Add explanation
ax.text(0, -2.8, 'Similar words cluster together\nDistance = Semantic similarity',
        ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

save_fig('math_16_embedding_space.pdf')

# Chart 17: Gradient Flow Comparison
print("Chart 17: Gradient Flow")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# RNN gradient flow (vanishing)
layers = np.arange(1, 11)
rnn_gradient = 0.9 ** layers
transformer_gradient = np.ones_like(layers) * 0.95

ax1.plot(layers, rnn_gradient, 'o-', color=color_secondary, linewidth=2.5,
         markersize=8, label='RNN: g = 0.9^n', markeredgecolor='black')
ax1.plot(layers, transformer_gradient, 's-', color=color_success, linewidth=2.5,
         markersize=6, label='Transformer: g ≈ 1', markeredgecolor='black')
ax1.set_xlabel('Layer Depth', fontsize=11, fontweight='bold')
ax1.set_ylabel('Gradient Magnitude', fontsize=11, fontweight='bold')
ax1.set_title('Gradient Flow Through Layers', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.1)
ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
ax1.text(7, 0.15, 'Vanishing threshold', color='red', fontsize=9)

# Visual representation
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 5)
ax2.axis('off')
ax2.set_title('Signal Strength Visualization', fontsize=12, fontweight='bold')

# RNN path (top)
for i in range(10):
    alpha = 0.9 ** i
    color = (1, 0, 0, alpha)
    ax2.add_patch(plt.Rectangle((i, 3), 0.8, 0.8, color=color, edgecolor='black'))
    if i < 9:
        ax2.arrow(i+0.8, 3.4, 0.15, 0, head_width=0.1, head_length=0.05,
                 fc='red', alpha=alpha)

ax2.text(-0.5, 3.4, 'RNN', fontsize=11, fontweight='bold', va='center')
ax2.text(5, 2.5, 'Signal fades', ha='center', fontsize=10, color='red')

# Transformer path (bottom)
for i in range(10):
    ax2.add_patch(plt.Rectangle((i, 0.5), 0.8, 0.8, color=color_success,
                                alpha=0.8, edgecolor='black'))
    if i < 9:
        ax2.arrow(i+0.8, 0.9, 0.15, 0, head_width=0.1, head_length=0.05,
                 fc=color_success)

ax2.text(-0.5, 0.9, 'Transformer', fontsize=11, fontweight='bold', va='center')
ax2.text(5, 0.1, 'Signal preserved', ha='center', fontsize=10, color=color_success)

plt.suptitle('Gradient Flow: Why Transformers Train Better', fontsize=14, fontweight='bold')
save_fig('math_17_gradient_flow.pdf')

# Chart 18: Layer Normalization
print("Chart 18: Layer Normalization")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Generate sample activations
np.random.seed(42)
neurons = np.arange(1, 9)
before = np.array([0.1, 3.5, 0.3, 8.2, 1.2, 0.5, 5.1, 2.3])
after = (before - before.mean()) / before.std()

ax1.bar(neurons, before, color=color_neutral, edgecolor='black', linewidth=1.5)
ax1.set_title('Before Layer Normalization', fontsize=12, fontweight='bold')
ax1.set_xlabel('Neuron', fontsize=11)
ax1.set_ylabel('Activation Value', fontsize=11)
ax1.set_ylim(0, 10)
ax1.axhline(y=before.mean(), color='red', linestyle='--', label=f'Mean={before.mean():.1f}')
ax1.legend()

# Add variance annotation
ax1.text(4.5, 9, f'Variance = {before.var():.1f}', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax2.bar(neurons, after, color=color_accent, edgecolor='black', linewidth=1.5)
ax2.set_title('After Layer Normalization', fontsize=12, fontweight='bold')
ax2.set_xlabel('Neuron', fontsize=11)
ax2.set_ylabel('Normalized Value', fontsize=11)
ax2.set_ylim(-2, 2)
ax2.axhline(y=0, color='red', linestyle='--', label='Mean=0')
ax2.legend()

# Add variance annotation
ax2.text(4.5, 1.7, 'Variance = 1.0', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.suptitle('Layer Normalization: Stabilizing Training', fontsize=14, fontweight='bold')
save_fig('math_18_layer_norm.pdf')

# Chart 19: Residual Connections
print("Chart 19: Residual Connections")
fig, ax = plt.subplots(figsize=(10, 6))

ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_title('Residual Connections: Preserving Information', fontsize=14, fontweight='bold')

# Input
input_box = plt.Rectangle((0.5, 2.5), 1.5, 1, color=color_accent,
                         edgecolor='black', linewidth=2)
ax.add_patch(input_box)
ax.text(1.25, 3, 'Input\n[1, 0, 1]', ha='center', va='center',
        fontsize=10, fontweight='bold')

# Transformation block
transform_box = plt.Rectangle((3.5, 2.5), 2, 1, color=color_warning,
                              edgecolor='black', linewidth=2)
ax.add_patch(transform_box)
ax.text(4.5, 3, 'Attention\n+FFN', ha='center', va='center',
        fontsize=10, fontweight='bold')

# Output
output_box = plt.Rectangle((7.5, 2.5), 1.5, 1, color=color_success,
                           edgecolor='black', linewidth=2)
ax.add_patch(output_box)
ax.text(8.25, 3, 'Output\n[1.2, 0.3, 1.1]', ha='center', va='center',
        fontsize=10, fontweight='bold')

# Main transformation path
ax.arrow(2, 3, 1.3, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
ax.arrow(5.5, 3, 1.8, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
ax.text(4.5, 3.5, 'Transform', ha='center', fontsize=9)

# Residual connection (bypass)
ax.add_patch(mpatches.FancyArrowPatch((1.25, 2.3), (8.25, 2.3),
                                 connectionstyle="arc3,rad=-.5",
                                 arrowstyle='->', mutation_scale=20,
                                 linewidth=2.5, color=color_secondary))
ax.text(4.5, 0.8, 'Residual (Identity) Connection', ha='center',
        fontsize=11, color=color_secondary, fontweight='bold')

# Addition symbol
ax.text(6.8, 3, '+', fontsize=20, fontweight='bold')

# Formula
ax.text(5, 5, 'Output = Input + Transformation(Input)',
        ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

save_fig('math_19_residual_connections.pdf')

# Chart 20: Temperature in Sampling
print("Chart 20: Temperature in Sampling")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))

# Base logits
words = ['cat', 'dog', 'mat', 'table', 'chair']
logits = np.array([2.0, 1.0, 3.0, 1.5, 0.5])

# Different temperatures
temps = [0.5, 1.0, 2.0]
axes = [ax1, ax2, ax3]
titles = ['Low Temperature (0.5)\nConfident',
          'Normal Temperature (1.0)\nBalanced',
          'High Temperature (2.0)\nCreative']
colors = [color_accent, color_neutral, color_warning]

for ax, temp, title, color in zip(axes, temps, titles, colors):
    probs = softmax(logits / temp)
    bars = ax.bar(words, probs, color=color, edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_ylim(0, 0.8)

    # Add probability values
    for i, prob in enumerate(probs):
        ax.text(i, prob + 0.02, f'{prob:.1%}', ha='center', fontsize=9)

    # Highlight max probability
    max_idx = np.argmax(probs)
    bars[max_idx].set_color(color_success)
    bars[max_idx].set_alpha(1.0)

    # Add entropy value (measure of randomness)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    ax.text(0.5, 0.95, f'Entropy: {entropy:.2f}', transform=ax.transAxes,
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white'))

plt.suptitle('Temperature Control in Next Word Sampling', fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig('math_20_temperature_sampling.pdf')

print("\n" + "="*60)
print("All 8 additional mathematical charts generated successfully!")
print("Charts provide BSc-level mathematical understanding of transformers")
print("="*60)