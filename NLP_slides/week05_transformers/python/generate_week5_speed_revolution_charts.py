"""
Generate 12 pedagogically-aligned charts for Week 5: Speed Revolution presentation
Lavender/Purple color scheme matching template_beamer_final.tex
Created: 2025-09-28
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('../figures', exist_ok=True)

mlpurple = (51/255, 51/255, 178/255)
mllavender = (173/255, 173/255, 224/255)
mllavender2 = (193/255, 193/255, 232/255)
mllavender3 = (204/255, 204/255, 235/255)
mlgreen = (44/255, 160/255, 44/255)
mlred = (214/255, 39/255, 40/255)
mlorange = (255/255, 127/255, 14/255)
mlblue = (0/255, 102/255, 204/255)

plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

def save_fig(filename):
    plt.savefig(f'../figures/{filename}', dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Generated: {filename}')
    plt.close()

print("Generating Week 5 Speed Revolution charts...")

# ============ ACT 1: THE WAITING GAME ============

# Chart 1: Training Time Comparison Timeline
print("\n=== Act 1: The Waiting Game ===")
fig, ax = plt.subplots(figsize=(10, 5))
models = ['RNN\n(Sequential)', 'RNN+Attention\n(Partial Parallel)', 'Transformer\n(Full Parallel)']
days = [90, 45, 1]
colors = [mlred, mlorange, mlgreen]

bars = ax.barh(models, days, color=colors, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Training Time (days)', fontweight='bold', fontsize=12)
ax.set_title('The Speed Revolution: From Months to Days', fontweight='bold', fontsize=14)
ax.set_xlim(0, 100)

for i, (bar, day) in enumerate(zip(bars, days)):
    width = bar.get_width()
    ax.text(width + 2, bar.get_y() + bar.get_height()/2,
            f'{day} days', va='center', fontweight='bold', fontsize=11)

ax.axvline(90, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax.text(90, 2.5, '90 days', ha='center', fontsize=9, color='gray')
ax.grid(axis='x', alpha=0.2)
save_fig('sr_01_training_time_comparison.pdf')

# Chart 2: GPU Utilization Waste
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['RNN\n(Sequential)', 'RNN+Attention\n(Partial)', 'Transformer\n(Full Parallel)']
utilized = [2, 5, 92]
wasted = [98, 95, 8]

x = np.arange(len(categories))
width = 0.6

p1 = ax.bar(x, utilized, width, label='Utilized (Active Cores)', color=mlgreen, edgecolor='black', linewidth=1.5)
p2 = ax.bar(x, wasted, width, bottom=utilized, label='Wasted (Idle Cores)', color=mllavender3, edgecolor='black', linewidth=1.5)

ax.set_ylabel('GPU Utilization (%)', fontweight='bold', fontsize=12)
ax.set_title('GPU Utilization: The $10,000 Hardware Sitting Idle', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(loc='upper left', fontsize=10)
ax.set_ylim(0, 100)

for i, (u, w) in enumerate(zip(utilized, wasted)):
    ax.text(i, u/2, f'{u}%', ha='center', va='center', fontweight='bold', fontsize=11, color='white')
    ax.text(i, u + w/2, f'{w}%', ha='center', va='center', fontweight='bold', fontsize=11, color=mlpurple)

ax.axhline(50, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax.text(2.5, 52, '50% utilization', fontsize=9, color='gray')
ax.grid(axis='y', alpha=0.2)
save_fig('sr_02_gpu_utilization_waste.pdf')

# Chart 3: Sequential Processing Bottleneck
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))

words = ['The', 'cat', 'sat', 'on', 'mat']
times_seq = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

ax1.set_xlim(0, 5.5)
ax1.set_ylim(0, 1)
for i, (start, end) in enumerate(times_seq):
    ax1.barh(0.5, end-start, left=start, height=0.3, color=mlred, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.text((start+end)/2, 0.5, words[i], ha='center', va='center', fontweight='bold', fontsize=11, color='white')
    if i < len(words) - 1:
        ax1.arrow(end, 0.5, 0.05, 0, head_width=0.1, head_length=0.05, fc='black', ec='black')

ax1.set_yticks([])
ax1.set_xlabel('Time (arbitrary units)', fontweight='bold', fontsize=11)
ax1.set_title('RNN: Sequential Processing (One Word at a Time)', fontweight='bold', fontsize=13, color=mlred)
ax1.text(5.2, 0.5, '5 time units', va='center', fontweight='bold', fontsize=10)
ax1.grid(axis='x', alpha=0.2)

ax2.set_xlim(0, 5.5)
ax2.set_ylim(0, len(words))
for i, word in enumerate(words):
    ax2.barh(i, 1, left=0, height=0.7, color=mlgreen, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.text(0.5, i, word, ha='center', va='center', fontweight='bold', fontsize=11, color='white')

ax2.set_yticks([])
ax2.set_xlabel('Time (arbitrary units)', fontweight='bold', fontsize=11)
ax2.set_title('Transformer: Parallel Processing (All Words Simultaneously)', fontweight='bold', fontsize=13, color=mlgreen)
ax2.text(1.2, len(words)//2, '1 time unit', va='center', fontweight='bold', fontsize=10)
ax2.axvline(1, color='black', linestyle='--', linewidth=2)
ax2.grid(axis='x', alpha=0.2)

plt.suptitle('Sequential vs Parallel: The Bottleneck', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
save_fig('sr_03_sequential_bottleneck.pdf')

# ============ ACT 2: THE FIRST ATTEMPT ============

print("\n=== Act 2: The First Attempt ===")

# Chart 4: Attention Success on Short Sequences
fig, ax = plt.subplots(figsize=(7, 7))
words_fr = ['Le', 'chat', "s'est", 'assis']
n = len(words_fr)

attention = np.array([
    [0.7, 0.2, 0.05, 0.05],
    [0.15, 0.6, 0.15, 0.1],
    [0.1, 0.3, 0.5, 0.1],
    [0.05, 0.25, 0.2, 0.5]
])

im = ax.imshow(attention, cmap='Purples', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(words_fr, fontweight='bold', fontsize=12)
ax.set_yticklabels(words_fr, fontweight='bold', fontsize=12)
ax.set_xlabel('Keys (Source)', fontweight='bold', fontsize=12)
ax.set_ylabel('Queries (Attending)', fontweight='bold', fontsize=12)
ax.set_title('Pure Attention SUCCESS (Short Sequence)', fontweight='bold', fontsize=14, color=mlgreen)

for i in range(n):
    for j in range(n):
        text = ax.text(j, i, f'{attention[i, j]:.2f}',
                      ha='center', va='center', color='white' if attention[i, j] > 0.4 else 'black',
                      fontweight='bold', fontsize=10)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Attention Weight', fontweight='bold', fontsize=11)

for i in range(n + 1):
    ax.axhline(i - 0.5, color='gray', linewidth=0.5)
    ax.axvline(i - 0.5, color='gray', linewidth=0.5)

save_fig('sr_04_attention_success_short.pdf')

# Chart 5: Failure Pattern - Quality Degradation
fig, ax = plt.subplots(figsize=(10, 6))
seq_lengths = [10, 20, 50, 100, 200]
bleu_scores = [32.1, 31.8, 18.4, 8.2, 3.1]

ax.plot(seq_lengths, bleu_scores, marker='o', linewidth=3, markersize=10, color=mlred, label='Pure Attention (No Positional Encoding)')
ax.axhline(32.1, color=mlgreen, linestyle='--', linewidth=2, alpha=0.7, label='Baseline (Short Sequences)')

ax.set_xlabel('Sequence Length (words)', fontweight='bold', fontsize=12)
ax.set_ylabel('BLEU Score (Translation Quality)', fontweight='bold', fontsize=12)
ax.set_title('The Failure Pattern: Pure Attention Collapses on Long Sequences', fontweight='bold', fontsize=14)
ax.legend(fontsize=11, loc='upper right')
ax.grid(alpha=0.3)

for x, y in zip(seq_lengths, bleu_scores):
    ax.annotate(f'{y}', (x, y), textcoords='offset points', xytext=(0,10), ha='center', fontweight='bold', fontsize=10)

ax.fill_between(seq_lengths, bleu_scores, alpha=0.2, color=mlred)
ax.text(150, 25, '90% quality drop', fontsize=12, fontweight='bold', color=mlred, bbox=dict(boxstyle='round', facecolor='white', edgecolor=mlred, linewidth=2))
save_fig('sr_05_quality_degradation.pdf')

# Chart 6: Permutation Invariance Problem
fig, ax = plt.subplots(figsize=(10, 6))
sentences = [
    'The cat sat on the mat',
    'The mat sat on the cat',
    'Cat the sat mat on the'
]
colors_sent = [mlgreen, mlred, mlred]
labels = ['Correct Sentence', 'Wrong Meaning!', 'Nonsense!']

y_pos = np.arange(len(sentences))
attention_identical = [0.85, 0.85, 0.85]

bars = ax.barh(y_pos, attention_identical, color=colors_sent, edgecolor='black', linewidth=2, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([f'{s}\n({l})' for s, l in zip(sentences, labels)], fontsize=11)
ax.set_xlabel('Pure Attention Representation Similarity', fontweight='bold', fontsize=12)
ax.set_title('Permutation Invariance Problem: All Orderings Look the Same!', fontweight='bold', fontsize=14)
ax.set_xlim(0, 1)

for i, (bar, val) in enumerate(zip(bars, attention_identical)):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center', fontweight='bold', fontsize=11)

ax.axvline(0.85, color='gray', linestyle='--', linewidth=2, alpha=0.5)
ax.text(0.85, 3, 'All identical to pure attention →', ha='right', va='bottom', fontsize=10, color='gray', fontweight='bold')
ax.grid(axis='x', alpha=0.2)
save_fig('sr_06_permutation_invariance.pdf')

# ============ ACT 3: THE POSITIONAL ENCODING REVOLUTION ============

print("\n=== Act 3: The Breakthrough ===")

# Chart 7: Positional Encoding Sine Waves
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

positions = np.arange(0, 50)
d_model = 128

def positional_encoding(position, i, d_model):
    if i % 2 == 0:
        return np.sin(position / (10000 ** (i / d_model)))
    else:
        return np.cos(position / (10000 ** ((i-1) / d_model)))

dimensions = [0, 2, 4, 8, 16, 32, 64]
colors_pe = [mlpurple, mlblue, mlgreen, mlorange, mlred, mllavender, 'gray']

for dim, color in zip(dimensions, colors_pe):
    encoding = [positional_encoding(pos, dim, d_model) for pos in positions]
    ax1.plot(positions, encoding, label=f'Dimension {dim}', color=color, linewidth=2)

ax1.set_xlabel('Position in Sequence', fontweight='bold', fontsize=12)
ax1.set_ylabel('Encoding Value', fontweight='bold', fontsize=12)
ax1.set_title('Positional Encoding: Sine Waves at Different Frequencies', fontweight='bold', fontsize=14)
ax1.legend(loc='upper right', fontsize=9, ncol=2)
ax1.grid(alpha=0.3)
ax1.axhline(0, color='black', linewidth=0.5)

pe_matrix = np.zeros((50, 64))
for pos in range(50):
    for i in range(64):
        pe_matrix[pos, i] = positional_encoding(pos, i, 128)

im = ax2.imshow(pe_matrix.T, cmap='RdBu', aspect='auto', interpolation='nearest')
ax2.set_xlabel('Position in Sequence', fontweight='bold', fontsize=12)
ax2.set_ylabel('Encoding Dimension', fontweight='bold', fontsize=12)
ax2.set_title('Complete Positional Encoding Matrix (First 64 dimensions)', fontweight='bold', fontsize=14)
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('Encoding Value', fontweight='bold', fontsize=11)

plt.tight_layout()
save_fig('sr_07_positional_encoding_waves.pdf')

# Chart 8: Vector Addition for Position
fig, ax = plt.subplots(figsize=(10, 6))

word_emb = np.array([0.3, 0.2, 0.5, 0.1])
pos_enc = np.array([0.1, 0.0, 0.0, 0.05])
combined = word_emb + pos_enc

x = np.arange(len(word_emb))
width = 0.25

bars1 = ax.bar(x - width, word_emb, width, label='Word Embedding (Meaning)', color=mlblue, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x, pos_enc, width, label='Positional Encoding (Position)', color=mlorange, edgecolor='black', linewidth=1.5)
bars3 = ax.bar(x + width, combined, width, label='Combined Representation', color=mlpurple, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Value', fontweight='bold', fontsize=12)
ax.set_xlabel('Dimension', fontweight='bold', fontsize=12)
ax.set_title('Vector Addition: Meaning + Position = Complete Representation', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(['Dim 0', 'Dim 1', 'Dim 2', 'Dim 3'], fontsize=11)
ax.legend(fontsize=11, loc='upper left')
ax.axhline(0, color='black', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)

for i, (b1, b2, b3) in enumerate(zip(bars1, bars2, bars3)):
    height1 = b1.get_height()
    height2 = b2.get_height()
    height3 = b3.get_height()
    if height1 != 0:
        ax.text(b1.get_x() + b1.get_width()/2, height1, f'{height1:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    if height2 != 0:
        ax.text(b2.get_x() + b2.get_width()/2, height2, f'{height2:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(b3.get_x() + b3.get_width()/2, height3 + 0.02, f'{height3:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.annotate('', xy=(x[0], 0.5), xytext=(x[0]-width, word_emb[0]),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(x[0]+width, combined[0]), xytext=(x[0], pos_enc[0]),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

save_fig('sr_08_vector_addition.pdf')

# Chart 9: Self-Attention 3-Step Visual
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

words = ['the', 'cat', 'sat']
n_words = len(words)

q_vectors = np.array([[0.1, 0.4], [0.6, 0.2], [0.3, 0.65]])
k_vectors = np.array([[0.1, 0.4], [0.6, 0.2], [0.3, 0.65]])

scores = np.dot(q_vectors, k_vectors.T)
im1 = ax1.imshow(scores, cmap='Purples', aspect='auto')
ax1.set_xticks(range(n_words))
ax1.set_yticks(range(n_words))
ax1.set_xticklabels(words, fontweight='bold', fontsize=12)
ax1.set_yticklabels(words, fontweight='bold', fontsize=12)
ax1.set_title('Step 1: Dot Products\n(Similarity Scores)', fontweight='bold', fontsize=13)

for i in range(n_words):
    for j in range(n_words):
        ax1.text(j, i, f'{scores[i,j]:.2f}', ha='center', va='center',
                color='white' if scores[i,j] > 0.3 else 'black', fontweight='bold', fontsize=10)

for i in range(n_words + 1):
    ax1.axhline(i - 0.5, color='gray', linewidth=0.5)
    ax1.axvline(i - 0.5, color='gray', linewidth=0.5)

softmax_scores = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
im2 = ax2.imshow(softmax_scores, cmap='Purples', vmin=0, vmax=1, aspect='auto')
ax2.set_xticks(range(n_words))
ax2.set_yticks(range(n_words))
ax2.set_xticklabels(words, fontweight='bold', fontsize=12)
ax2.set_yticklabels(words, fontweight='bold', fontsize=12)
ax2.set_title('Step 2: Softmax\n(Percentages)', fontweight='bold', fontsize=13)

for i in range(n_words):
    for j in range(n_words):
        ax2.text(j, i, f'{softmax_scores[i,j]*100:.0f}%', ha='center', va='center',
                color='white' if softmax_scores[i,j] > 0.4 else 'black', fontweight='bold', fontsize=10)

for i in range(n_words + 1):
    ax2.axhline(i - 0.5, color='gray', linewidth=0.5)
    ax2.axvline(i - 0.5, color='gray', linewidth=0.5)

v_vectors = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
output = np.dot(softmax_scores, v_vectors)

y_pos = np.arange(n_words)
ax3.barh(y_pos, output[:, 0], 0.3, label='Dimension 1', color=mlblue, edgecolor='black')
ax3.barh(y_pos + 0.3, output[:, 1], 0.3, label='Dimension 2', color=mlorange, edgecolor='black')
ax3.set_yticks(y_pos + 0.15)
ax3.set_yticklabels(words, fontweight='bold', fontsize=12)
ax3.set_xlabel('Output Value', fontweight='bold', fontsize=11)
ax3.set_title('Step 3: Weighted Sum\n(Final Representation)', fontweight='bold', fontsize=13)
ax3.legend(fontsize=10)
ax3.grid(axis='x', alpha=0.3)

plt.suptitle('Self-Attention: The Complete 3-Step Algorithm', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig('sr_09_self_attention_3steps.pdf')

# Chart 10: Full Numerical Walkthrough
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

ax_title = fig.add_subplot(gs[0, :])
ax_title.text(0.5, 0.5, 'Full Numerical Walkthrough: "The cat sat"',
             ha='center', va='center', fontsize=16, fontweight='bold')
ax_title.axis('off')

ax1 = fig.add_subplot(gs[1, 0])
ax1.text(0.1, 0.8, 'Given (with position):', fontweight='bold', fontsize=12)
ax1.text(0.1, 0.6, '"the": [0.1, 0.4]', fontsize=11, family='monospace')
ax1.text(0.1, 0.5, '"cat": [0.6, 0.2]', fontsize=11, family='monospace')
ax1.text(0.1, 0.4, '"sat": [0.3, 0.65]', fontsize=11, family='monospace')
ax1.axis('off')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

ax2 = fig.add_subplot(gs[1, 1])
ax2.text(0.1, 0.8, 'Step 1: Dot Products', fontweight='bold', fontsize=12)
ax2.text(0.1, 0.6, 'cat · the = 0.06 + 0.08 = 0.14', fontsize=10, family='monospace')
ax2.text(0.1, 0.5, 'cat · cat = 0.36 + 0.04 = 0.40', fontsize=10, family='monospace', color=mlgreen)
ax2.text(0.1, 0.4, 'cat · sat = 0.18 + 0.13 = 0.31', fontsize=10, family='monospace')
ax2.axis('off')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

ax3 = fig.add_subplot(gs[2, 0])
ax3.text(0.1, 0.8, 'Step 2: Softmax', fontweight='bold', fontsize=12)
ax3.text(0.1, 0.6, 'e^0.14 = 1.15', fontsize=10, family='monospace')
ax3.text(0.1, 0.5, 'e^0.40 = 1.49', fontsize=10, family='monospace')
ax3.text(0.1, 0.4, 'e^0.31 = 1.36', fontsize=10, family='monospace')
ax3.text(0.1, 0.3, 'Sum = 4.00', fontsize=10, family='monospace')
ax3.text(0.1, 0.1, 'Percentages: 29%, 37%, 34%', fontsize=10, fontweight='bold', color=mlpurple)
ax3.axis('off')
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

ax4 = fig.add_subplot(gs[2, 1])
ax4.text(0.1, 0.8, 'Step 3: Weighted Combination', fontweight='bold', fontsize=12)
ax4.text(0.1, 0.6, '0.29×[0.1,0.4]', fontsize=10, family='monospace')
ax4.text(0.1, 0.5, '+ 0.37×[0.6,0.2]', fontsize=10, family='monospace')
ax4.text(0.1, 0.4, '+ 0.34×[0.3,0.65]', fontsize=10, family='monospace')
ax4.text(0.1, 0.2, '= [0.35, 0.41]', fontsize=11, fontweight='bold', color=mlgreen, family='monospace')
ax4.text(0.1, 0.05, '← New "cat" representation', fontsize=9, color=mlgreen)
ax4.axis('off')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)

ax5 = fig.add_subplot(gs[3, :])
cats = ['the\n(29%)', 'cat\n(37%)', 'sat\n(34%)']
weights = [0.29, 0.37, 0.34]
colors_w = [mllavender2, mlgreen, mlorange]
bars = ax5.bar(cats, weights, color=colors_w, edgecolor='black', linewidth=2, alpha=0.8)
ax5.set_ylabel('Attention Weight', fontweight='bold', fontsize=12)
ax5.set_title('Focus Distribution: Where "cat" Attends', fontweight='bold', fontsize=13)
ax5.set_ylim(0, 0.5)
ax5.grid(axis='y', alpha=0.3)

for bar, w in zip(bars, weights):
    ax5.text(bar.get_x() + bar.get_width()/2, w + 0.01, f'{w:.2f}', ha='center', fontweight='bold', fontsize=11)

save_fig('sr_10_numerical_walkthrough.pdf')

# ============ ACT 4: THE REVOLUTION UNFOLDS ============

print("\n=== Act 4: The Revolution Unfolds ===")

# Chart 11: Speed vs Quality Scatter Plot
fig, ax = plt.subplots(figsize=(10, 7))

models_scatter = ['RNN', 'RNN+Attention', 'Transformer (base)', 'Transformer (big)']
training_time = [90, 45, 1, 3.5]
bleu_quality = [24.5, 28.4, 27.3, 28.4]
colors_scatter = [mlred, mlorange, mlgreen, mlblue]
sizes = [300, 300, 400, 400]

for model, time, bleu, color, size in zip(models_scatter, training_time, bleu_quality, colors_scatter, sizes):
    ax.scatter(time, bleu, s=size, c=[color], edgecolor='black', linewidth=2, alpha=0.7, label=model)
    ax.annotate(model, (time, bleu), textcoords='offset points', xytext=(10, 10),
               ha='left', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, linewidth=2))

ax.set_xlabel('Training Time (days)', fontweight='bold', fontsize=12)
ax.set_ylabel('BLEU Score (Translation Quality)', fontweight='bold', fontsize=12)
ax.set_title('Speed vs Quality: The Revolution Validated', fontweight='bold', fontsize=14)
ax.set_xscale('log')
ax.set_xlim(0.5, 100)
ax.set_ylim(23, 29)
ax.grid(alpha=0.3)

ax.arrow(45, 28, -40, -0.2, head_width=0.3, head_length=5, fc=mlgreen, ec=mlgreen, linewidth=2, alpha=0.5)
ax.text(25, 27.5, '45x speedup\nSame quality!', fontsize=11, fontweight='bold', color=mlgreen,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=mlgreen, linewidth=2))

ax.legend(fontsize=10, loc='lower right')
save_fig('sr_11_speed_vs_quality.pdf')

# Chart 12: Modern Applications Timeline
fig, ax = plt.subplots(figsize=(12, 8))

years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
milestones = [
    'Paper:\nAttention Is\nAll You Need',
    'BERT:\nGoogle\nSearch',
    'GPT-2:\nScale\nMatters',
    'GPT-3:\n175B params\nEmergent\nAbilities',
    'ViT:\nVision\nTransformers',
    'ChatGPT:\n100M users\nin 2 months',
    'GPT-4:\nMultimodal\nTransformers',
    '2024:\nTransformers\nEverywhere'
]

colors_timeline = [mlpurple, mlblue, mlgreen, mlorange, mlred, mlpurple, mlblue, mlgreen]
y_positions = [0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7]

ax.plot(years, [0.5]*len(years), color='gray', linewidth=3, zorder=1)

for year, milestone, color, y_pos in zip(years, milestones, colors_timeline, y_positions):
    ax.scatter(year, 0.5, s=500, c=[color], edgecolor='black', linewidth=2, zorder=3, alpha=0.8)
    ax.plot([year, year], [0.5, y_pos], color='gray', linestyle='--', linewidth=1, zorder=2)

    ax.text(year, y_pos, milestone, ha='center', va='center', fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='black', linewidth=2, alpha=0.8),
           color='white')

ax.set_xlim(2016.5, 2024.5)
ax.set_ylim(0, 1)
ax.set_xlabel('Year', fontweight='bold', fontsize=13)
ax.set_title('The Transformer Revolution: From Paper to Everywhere (2017-2024)', fontweight='bold', fontsize=15)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks(years)
ax.set_xticklabels(years, fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.2)

ax.text(2020.5, 0.05, 'All made possible by 100x speedup', fontsize=12, ha='center', fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='yellow', edgecolor='black', linewidth=2, alpha=0.7))

plt.tight_layout()
save_fig('sr_12_modern_applications_timeline.pdf')

print("\n=== All 12 charts generated successfully! ===")
print("\nCharts saved to ../figures/ with prefix 'sr_'")
print("\nNext steps:")
print("1. Update LaTeX presentation with \\includegraphics commands")
print("2. Recompile PDF")
print("3. Visual quality check")