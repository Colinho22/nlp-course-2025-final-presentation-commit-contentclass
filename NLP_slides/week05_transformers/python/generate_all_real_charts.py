"""
Generate ALL educational charts with REAL simulation data
Uses data from transformer_educational_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

# Load simulation data
with open('simulation_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Template colors
COLOR_BLUE = '#4472C4'
COLOR_PURPLE = '#8B5A9B'
COLOR_GREEN = '#44A044'
COLOR_ORANGE = '#FF7F0E'
COLOR_RED = '#D62728'
COLOR_MAIN = '#333366'

print("Generating educational charts with REAL simulation data...")
print("="*70)

# ===========================================
# CHART 1: Real Attention Weights (3D Network)
# ===========================================
print("\n[1/6] Generating: Real attention network...")

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=20, azim=45)

words = data['input_sentence']
positions = [(0, 0, 2), (3, 1, 3), (6, 0, 2), (9, 1, 1.5), (12, 0, 2)]
colors = [COLOR_BLUE, COLOR_PURPLE, COLOR_GREEN, COLOR_ORANGE, COLOR_RED]

# Draw word nodes
for word, pos, color in zip(words, positions, colors):
    ax.scatter([pos[0]], [pos[1]], [pos[2]], s=500, c=color, edgecolors=COLOR_MAIN, linewidth=3, alpha=0.8)
    ax.text(pos[0], pos[1], pos[2]+0.5, word, fontsize=14, ha='center', fontweight='bold', color=color)

# REAL attention connections from Head 1 for word "the" (last word, index 4)
real_weights = data['all_heads_weights'][0][4]  # Head 1, last word
for i, weight in enumerate(real_weights):
    if i != 4 and weight > 0.15:  # Show significant connections
        ax.plot([positions[4][0], positions[i][0]],
               [positions[4][1], positions[i][1]],
               [positions[4][2], positions[i][2]],
               color=COLOR_RED, linewidth=int(weight*50), alpha=0.6)
        mid_x = (positions[4][0] + positions[i][0]) / 2
        mid_y = (positions[4][1] + positions[i][1]) / 2
        mid_z = (positions[4][2] + positions[i][2]) / 2
        ax.text(mid_x, mid_y, mid_z, f'{weight:.0%}', fontsize=10, ha='center',
               fontweight='bold', color=COLOR_RED)

fig.suptitle('Real Attention Weights (Head 1 for "the")', fontsize=16, fontweight='bold', y=0.95)
ax.text(6, 3, 4.5, f'REAL DATA: {words[4]} attends to:', fontsize=11, ha='center', fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='#F0F0F0', edgecolor=COLOR_MAIN, linewidth=2))

output_text = '\n'.join([f'{words[i]}: {real_weights[i]:.0%}' for i in range(len(words))])
ax.text(6, 3, -1, output_text, fontsize=10, ha='center', fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFE6CC', edgecolor=COLOR_MAIN, linewidth=2))

ax.set_xlim(-2, 14)
ax.set_ylim(-1, 4)
ax.set_zlim(-2, 5)
ax.grid(True, alpha=0.3)
ax.set_box_aspect([2.5, 1, 1.5])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

plt.tight_layout()
plt.savefig('../figures/sr_real_01_attention_network.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("   Saved: sr_real_01_attention_network.pdf")
plt.close()

# ===========================================
# CHART 2: Real Multi-Head Attention Heatmap
# ===========================================
print("[2/6] Generating: Multi-head attention heatmap...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Real Multi-Head Attention Patterns (4 Heads)', fontsize=16, fontweight='bold')

head_names = ['Grammar', 'Semantics', 'Position', 'Global']
for idx, (ax, name) in enumerate(zip(axes.flat, head_names)):
    weights = data['all_heads_weights'][idx]
    im = ax.imshow(weights, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.4)
    ax.set_xticks(range(len(words)))
    ax.set_yticks(range(len(words)))
    ax.set_xticklabels(words, fontsize=10)
    ax.set_yticklabels(words, fontsize=10)
    ax.set_title(f'Head {idx+1}: {name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Attends TO', fontsize=10)
    ax.set_ylabel('Attends FROM', fontsize=10)

    # Add text annotations
    for i in range(len(words)):
        for j in range(len(words)):
            text = ax.text(j, i, f'{weights[i, j]:.2f}',
                         ha="center", va="center", color="black" if weights[i, j] < 0.25 else "white",
                         fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('../figures/sr_real_02_multihead_heatmap.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("   Saved: sr_real_02_multihead_heatmap.pdf")
plt.close()

# ===========================================
# CHART 3: Real Output Probabilities
# ===========================================
print("[3/6] Generating: Real output probabilities...")

fig, ax = plt.subplots(figsize=(12, 8))

top_words = data['top_words']
top_probs = data['top_probs']
furniture_words = ['mat', 'floor', 'table', 'rug', 'chair', 'bed', 'sofa']

colors_bar = [COLOR_GREEN if w in furniture_words else COLOR_BLUE for w in top_words]
bars = ax.barh(range(len(top_words)), [p*100 for p in top_probs], color=colors_bar, alpha=0.8, edgecolor=COLOR_MAIN, linewidth=2)

ax.set_yticks(range(len(top_words)))
ax.set_yticklabels(top_words, fontsize=12, fontweight='bold')
ax.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
ax.set_title('Real Predictions: "The cat sat on the ___"', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add percentage labels
for i, (word, prob) in enumerate(zip(top_words, top_probs)):
    ax.text(prob*100 + 0.2, i, f'{prob:.1%}', va='center', fontsize=10, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLOR_GREEN, label='Furniture'),
                  Patch(facecolor=COLOR_BLUE, label='Other')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig('../figures/sr_real_03_output_probs.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("   Saved: sr_real_03_output_probs.pdf")
plt.close()

# ===========================================
# CHART 4: Real Embeddings 3D
# ===========================================
print("[4/6] Generating: Real word embeddings 3D...")

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

embeddings = data['embeddings']
# Use first 3 dimensions for visualization
for i, word in enumerate(words):
    x, y, z = embeddings[i, 0], embeddings[i, 1], embeddings[i, 2]
    ax.scatter([x], [y], [z], s=500, c=colors[i], edgecolors=COLOR_MAIN, linewidth=3, alpha=0.8)
    ax.text(x, y, z+0.1, word, fontsize=14, ha='center', fontweight='bold', color=colors[i])

ax.set_xlabel('Dim 0: Animals', fontsize=10, fontweight='bold')
ax.set_ylabel('Dim 1: Objects', fontsize=10, fontweight='bold')
ax.set_zlabel('Dim 2: Actions', fontsize=10, fontweight='bold')
ax.set_title('Real Word Embeddings (First 3 Dimensions)', fontsize=14, fontweight='bold')
ax.view_init(elev=20, azim=45)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/sr_real_04_embeddings_3d.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("   Saved: sr_real_04_embeddings_3d.pdf")
plt.close()

# ===========================================
# CHART 5: Real Positional Encoding
# ===========================================
print("[5/6] Generating: Real positional encoding...")

fig, ax = plt.subplots(figsize=(12, 8))

pos_enc = data['pos_encoding']
im = ax.imshow(pos_enc.T, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(len(words)))
ax.set_xticklabels([f'Pos {i}\n{w}' for i, w in enumerate(words)], fontsize=10)
ax.set_yticks(range(data['d_model']))
ax.set_yticklabels([f'Dim {i}' for i in range(data['d_model'])], fontsize=10)
ax.set_title('Real Positional Encoding (Sin/Cos)', fontsize=14, fontweight='bold')
ax.set_xlabel('Position in Sequence', fontsize=12, fontweight='bold')
ax.set_ylabel('Embedding Dimension', fontsize=12, fontweight='bold')

# Add text annotations
for i in range(data['d_model']):
    for j in range(len(words)):
        text = ax.text(j, i, f'{pos_enc[j, i]:.2f}',
                     ha="center", va="center", color="white" if abs(pos_enc[j, i]) > 0.5 else "black",
                     fontsize=7)

plt.colorbar(im, ax=ax, label='Value')
plt.tight_layout()
plt.savefig('../figures/sr_real_05_positional_encoding.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("   Saved: sr_real_05_positional_encoding.pdf")
plt.close()

# ===========================================
# CHART 6: Summary - Complete Pipeline
# ===========================================
print("[6/6] Generating: Complete pipeline summary...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Panel 1: Input
ax1 = fig.add_subplot(gs[0, 0])
ax1.text(0.5, 0.5, f'INPUT:\n{" ".join(words)} ___', fontsize=14, ha='center', va='center',
        fontweight='bold', bbox=dict(boxstyle='round', facecolor=COLOR_BLUE, alpha=0.3, pad=1))
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')
ax1.set_title('Step 1: Input Text', fontsize=12, fontweight='bold')

# Panel 2: Embeddings
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(embeddings[:, :4], cmap='viridis', aspect='auto')
ax2.set_xticks(range(4))
ax2.set_xticklabels(['D0', 'D1', 'D2', 'D3'], fontsize=8)
ax2.set_yticks(range(len(words)))
ax2.set_yticklabels(words, fontsize=8)
ax2.set_title('Step 2: Embeddings', fontsize=12, fontweight='bold')

# Panel 3: Position
ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(pos_enc[:, :4], cmap='RdBu', aspect='auto')
ax3.set_xticks(range(4))
ax3.set_xticklabels(['D0', 'D1', 'D2', 'D3'], fontsize=8)
ax3.set_yticks(range(len(words)))
ax3.set_yticklabels(['', '', '', '', ''], fontsize=8)
ax3.set_title('Step 3: + Position', fontsize=12, fontweight='bold')

# Panel 4-7: Multi-head attention
for head_idx in range(4):
    ax = fig.add_subplot(gs[1 + head_idx//2, head_idx%2])
    weights = data['all_heads_weights'][head_idx]
    im = ax.imshow(weights, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.4)
    ax.set_xticks(range(len(words)))
    ax.set_yticks(range(len(words)))
    ax.set_xticklabels(words if head_idx >= 2 else ['', '', '', '', ''], fontsize=7)
    ax.set_yticklabels(words if head_idx % 2 == 0 else ['', '', '', '', ''], fontsize=7)
    ax.set_title(f'Step 4.{head_idx+1}: Head {head_idx+1}', fontsize=10, fontweight='bold')

# Panel 8: Output
ax8 = fig.add_subplot(gs[1:, 2])
bars = ax8.barh(range(min(5, len(top_words))), [p*100 for p in top_probs[:5]],
               color=[COLOR_GREEN if w in furniture_words else COLOR_BLUE for w in top_words[:5]],
               alpha=0.8, edgecolor=COLOR_MAIN, linewidth=2)
ax8.set_yticks(range(min(5, len(top_words))))
ax8.set_yticklabels(top_words[:5], fontsize=10, fontweight='bold')
ax8.set_xlabel('Probability (%)', fontsize=10)
ax8.set_title('Step 5: Output Predictions', fontsize=12, fontweight='bold')
for i, prob in enumerate(top_probs[:5]):
    ax8.text(prob*100 + 0.3, i, f'{prob:.1%}', va='center', fontsize=9, fontweight='bold')

fig.suptitle('Complete Transformer Pipeline with REAL Data', fontsize=16, fontweight='bold')
plt.savefig('../figures/sr_real_06_complete_pipeline.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("   Saved: sr_real_06_complete_pipeline.pdf")
plt.close()

print("\n" + "=" * 70)
print("[OK] All 6 educational charts generated with REAL simulation data!")
print("=" * 70)
print("\nFiles created:")
print("  1. sr_real_01_attention_network.pdf")
print("  2. sr_real_02_multihead_heatmap.pdf")
print("  3. sr_real_03_output_probs.pdf")
print("  4. sr_real_04_embeddings_3d.pdf")
print("  5. sr_real_05_positional_encoding.pdf")
print("  6. sr_real_06_complete_pipeline.pdf")
