"""
Generate 12 BSc-level charts focused on next word prediction
Simple mathematical visualizations for understanding transformers
Created: 2025-09-28
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('../figures', exist_ok=True)

# Clean, academic color scheme
color_rnn = '#E74C3C'  # Red for RNN
color_transformer = '#27AE60'  # Green for Transformer
color_neutral = '#34495E'  # Dark gray
color_accent = '#3498DB'  # Blue
color_light = '#ECF0F1'  # Light gray

plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def save_fig(filename):
    plt.savefig(f'../figures/{filename}', dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Generated: {filename}')
    plt.close()

print("Generating Week 5 Next Word Prediction Charts...")
print("="*60)

# ============ ACT 1: THE PREDICTION PROBLEM ============
print("\n=== Act 1: The Prediction Problem ===")

# Chart 1: Next Word Probability Distribution
fig, ax = plt.subplots(figsize=(10, 6))

words = ['mat', 'floor', 'chair', 'table', 'roof', 'ground', 'desk', 'other']
probs = [0.35, 0.25, 0.15, 0.10, 0.05, 0.04, 0.03, 0.03]
colors = [color_accent if p > 0.2 else color_neutral for p in probs]

bars = ax.bar(words, probs, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylim(0, 0.4)
ax.set_ylabel('Probability P(word | context)', fontsize=12, fontweight='bold')
ax.set_xlabel('Next Word Candidates', fontsize=12, fontweight='bold')
ax.set_title('Predicting: "The cat sat on the ___"', fontsize=14, fontweight='bold')

# Add probability values on bars
for bar, prob in zip(bars, probs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
            f'{prob:.0%}', ha='center', va='bottom', fontweight='bold')

# Add context box
ax.text(0.5, 0.95, 'Context: "The cat sat on the"',
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor=color_light, alpha=0.8),
        ha='center', va='top')

save_fig('pred_01_probability_distribution.pdf')

# Chart 2: Context Window for Prediction
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

# RNN context window (fading)
words_context = ['The', 'cat', 'who', 'ate', 'the', 'mouse', 'yesterday', 'was']
positions = np.arange(len(words_context))
importance_rnn = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9]
colors_rnn = plt.cm.Reds(importance_rnn)

ax1.set_title('RNN: Fading Memory Window', fontsize=12, fontweight='bold')
bars1 = ax1.bar(positions, importance_rnn, color=colors_rnn, edgecolor='black', linewidth=1.5)
ax1.set_xticks(positions)
ax1.set_xticklabels(words_context, fontsize=10)
ax1.set_ylabel('Word Importance', fontsize=11)
ax1.set_ylim(0, 1)
ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Importance threshold')

# Add arrow showing decay
ax1.annotate('', xy=(7.5, 0.9), xytext=(0.5, 0.05),
            arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.5))
ax1.text(4, 0.5, 'Memory fades with distance', ha='center', fontsize=10,
         style='italic', color='darkred')

# Transformer context window (equal)
importance_transformer = [0.7, 0.8, 0.6, 0.7, 0.6, 0.85, 0.7, 0.9]
ax2.set_title('Transformer: Equal Access to All Context', fontsize=12, fontweight='bold')
bars2 = ax2.bar(positions, importance_transformer, color=color_transformer,
                edgecolor='black', linewidth=1.5, alpha=0.8)
ax2.set_xticks(positions)
ax2.set_xticklabels(words_context, fontsize=10)
ax2.set_ylabel('Word Importance', fontsize=11)
ax2.set_ylim(0, 1)
ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Importance threshold')

ax2.text(4, 0.3, 'All words accessible equally', ha='center', fontsize=10,
         style='italic', color='darkgreen', fontweight='bold')

plt.tight_layout()
save_fig('pred_02_context_window.pdf')

# Chart 3: Prediction Accuracy vs Context Length
fig, ax = plt.subplots(figsize=(10, 6))

context_lengths = np.arange(1, 21)
accuracy_rnn = 100 * np.exp(-0.1 * context_lengths) + 20
accuracy_transformer = 95 - 2 * np.log(context_lengths)

ax.plot(context_lengths, accuracy_rnn, 'o-', color=color_rnn, linewidth=2.5,
        markersize=6, label='RNN', markeredgecolor='black')
ax.plot(context_lengths, accuracy_transformer, 's-', color=color_transformer,
        linewidth=2.5, markersize=6, label='Transformer', markeredgecolor='black')

ax.set_xlabel('Context Length (words)', fontsize=12, fontweight='bold')
ax.set_ylabel('Next Word Prediction Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('How Context Length Affects Prediction Quality', fontsize=14, fontweight='bold')
ax.set_ylim(0, 105)
ax.set_xlim(0, 21)
ax.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True)
ax.grid(True, alpha=0.3)

# Add annotations
ax.annotate('RNN degrades\nwith distance', xy=(15, accuracy_rnn[14]),
            xytext=(12, 40), fontsize=10,
            arrowprops=dict(arrowstyle='->', color=color_rnn, lw=1.5),
            color=color_rnn, fontweight='bold')
ax.annotate('Transformer\nmaintains quality', xy=(15, accuracy_transformer[14]),
            xytext=(17, 80), fontsize=10,
            arrowprops=dict(arrowstyle='->', color=color_transformer, lw=1.5),
            color=color_transformer, fontweight='bold')

save_fig('pred_03_accuracy_vs_context.pdf')

# ============ ACT 2: WHY RNN PREDICTIONS FAIL ============
print("\n=== Act 2: Why RNN Predictions Fail ===")

# Chart 4: Forgetting Important Words
fig, ax = plt.subplots(figsize=(11, 6))

sentence = ['The', 'cat', 'who', 'chased', 'the', 'mouse', 'yesterday', 'was', 'hungry']
positions = np.arange(len(sentence))
importance_scores = [0.1, 0.9, 0.2, 0.3, 0.2, 0.4, 0.3, 0.5, 0.7]
time_decay = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85, 1.0]

# Create bars with gradient colors
colors = [(1, 0, 0, t*i) for t, i in zip(time_decay, importance_scores)]
bars = ax.bar(positions, importance_scores, color=colors, edgecolor='black', linewidth=1.5)

ax.set_xticks(positions)
ax.set_xticklabels(sentence, fontsize=11, fontweight='bold')
ax.set_ylabel('Word Importance in Memory', fontsize=12, fontweight='bold')
ax.set_title('RNN Forgets "cat" (subject) by the End', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.1)

# Highlight the problem
ax.add_patch(plt.Rectangle((0.5, 0), 1, 1.1, color='red', alpha=0.2))
ax.text(1, 1.0, 'Critical word\n"cat" forgotten!', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Show decay arrow
for i in range(len(positions)-1):
    if i == 1:  # From 'cat' position
        ax.annotate('', xy=(8, importance_scores[1]*time_decay[8]),
                   xytext=(1, importance_scores[1]),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2,
                                 alpha=0.5, linestyle='dashed'))

save_fig('pred_04_forgetting_words.pdf')

# Chart 5: Order Confusion
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Sentence 1
words1 = ['Dog', 'bites', 'man']
next_options1 = ['severely', 'yesterday', 'surprisingly', 'again']
probs1 = [0.45, 0.25, 0.20, 0.10]

ax1.set_title('Context: "Dog bites man"', fontsize=12, fontweight='bold')
ax1.bar(next_options1, probs1, color=color_accent, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Probability', fontsize=11)
ax1.set_ylim(0, 0.5)
ax1.axhline(y=0.45, color='green', linestyle='--', alpha=0.5)
ax1.text(0, 0.47, 'Correct: "severely"', fontsize=10, color='green', fontweight='bold')

# Sentence 2
words2 = ['Man', 'bites', 'dog']
next_options2 = ['surprisingly', 'shockingly', 'severely', 'yesterday']
probs2_rnn = [0.25, 0.20, 0.35, 0.20]  # RNN confused, picks wrong

ax2.set_title('Context: "Man bites dog" (unusual)', fontsize=12, fontweight='bold')
bars = ax2.bar(next_options2, probs2_rnn, color=['red' if i==2 else color_accent for i in range(4)],
               edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Probability', fontsize=11)
ax2.set_ylim(0, 0.5)
ax2.text(2, 0.37, 'RNN picks\n"severely"\n(wrong!)', ha='center', fontsize=9, color='red',
         fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
ax2.text(0, 0.27, 'Should be\n"surprisingly"', ha='center', fontsize=9, color='green')

plt.suptitle('RNN Struggles with Word Order Impact', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig('pred_05_order_confusion.pdf')

# Chart 6: Long-Range Dependencies
fig, ax = plt.subplots(figsize=(12, 6))

# Sentence with long-range dependency
sentence_parts = ['The', 'key', 'that', 'I', 'found', 'under', 'the', 'old', 'oak', 'tree', 'yesterday']
verb_options = ['was', 'were']
positions = np.arange(len(sentence_parts))

# Show distance problem
ax.set_title('Long-Range Dependency: "The key ... was/were lost"', fontsize=14, fontweight='bold')
ax.axis('off')
ax.set_xlim(-1, 12)
ax.set_ylim(0, 3)

# Draw sentence
for i, word in enumerate(sentence_parts):
    color = color_accent if i == 1 else color_neutral  # Highlight 'key'
    ax.text(i, 2, word, ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3, edgecolor='black'))

# Draw dependency arc
ax.annotate('', xy=(1, 1.5), xytext=(11, 1.5),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2,
                          connectionstyle='arc3,rad=.3'))
ax.text(6, 0.8, '10 words distance', ha='center', fontsize=10,
        color='red', fontweight='bold')

# Show prediction
ax.text(11.5, 2, '?', ha='center', va='center', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', edgecolor='black'))

# RNN vs Transformer predictions
ax.text(3, 0.2, 'RNN: Forgets "key" is singular → predicts "were" (wrong)',
        fontsize=10, color=color_rnn)
ax.text(3, -0.2, 'Transformer: Remembers "key" → predicts "was" (correct)',
        fontsize=10, color=color_transformer)

save_fig('pred_06_long_range_dependency.pdf')

# ============ ACT 3: HOW TRANSFORMERS PREDICT BETTER ============
print("\n=== Act 3: How Transformers Predict Better ===")

# Chart 7: Attention to Relevant Words
fig, ax = plt.subplots(figsize=(10, 7))

# Context and attention weights
context = ['The', 'cat', 'sat', 'on', 'the']
attention_weights = [0.05, 0.25, 0.35, 0.30, 0.05]
colors = plt.cm.Greens(np.array(attention_weights) + 0.3)

bars = ax.bar(context, attention_weights, color=colors, edgecolor='black', linewidth=2)
ax.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
ax.set_title('Attention Weights for Predicting Next Word After "on the"',
            fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.45)

# Add values on bars
for bar, weight in zip(bars, attention_weights):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
            f'{weight:.0%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add explanations
ax.text(0.5, 0.95, 'High attention to "sat" and "on" → predicts location word',
        transform=ax.transAxes, ha='center', va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor=color_light, alpha=0.8))

# Highlight important words
ax.add_patch(plt.Rectangle((1.6, 0), 1.8, 0.45, color='green', alpha=0.1))
ax.text(2.5, 0.4, 'Most relevant\nfor prediction', ha='center', fontsize=10,
        color='darkgreen', fontweight='bold')

save_fig('pred_07_attention_weights.pdf')

# Chart 8: Multiple Prediction Hypotheses (Multi-Head)
fig, axs = plt.subplots(1, 3, figsize=(14, 5))

context = ['The', 'cat', 'sat', 'on', 'the']

# Head 1: Grammar focus
weights1 = [0.1, 0.4, 0.3, 0.15, 0.05]
axs[0].bar(context, weights1, color='#FF6B6B', edgecolor='black', linewidth=1.5)
axs[0].set_title('Head 1: Grammar\n(Subject-Verb)', fontsize=11, fontweight='bold')
axs[0].set_ylim(0, 0.5)
axs[0].set_ylabel('Attention', fontsize=10)
axs[0].text(0.5, 0.9, 'Predicts: "mat" (noun)', transform=axs[0].transAxes,
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow'))

# Head 2: Semantic focus
weights2 = [0.05, 0.35, 0.1, 0.1, 0.4]
axs[1].bar(context, weights2, color='#4ECDC4', edgecolor='black', linewidth=1.5)
axs[1].set_title('Head 2: Semantic\n(Object Association)', fontsize=11, fontweight='bold')
axs[1].set_ylim(0, 0.5)
axs[1].text(0.5, 0.9, 'Predicts: "floor" (surface)', transform=axs[1].transAxes,
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow'))

# Head 3: Position focus
weights3 = [0.05, 0.05, 0.2, 0.6, 0.1]
axs[2].bar(context, weights3, color='#95E77E', edgecolor='black', linewidth=1.5)
axs[2].set_title('Head 3: Position\n(Preposition Pattern)', fontsize=11, fontweight='bold')
axs[2].set_ylim(0, 0.7)
axs[2].text(0.5, 0.9, 'Predicts: "table" (furniture)', transform=axs[2].transAxes,
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow'))

plt.suptitle('Multi-Head Attention: Different Perspectives → Better Prediction',
            fontsize=13, fontweight='bold', y=1.05)
plt.tight_layout()
save_fig('pred_08_multihead_attention.pdf')

# Chart 9: Position-Aware Predictions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Example 1: "bank" early in sentence
sentence1 = ['I', 'went', 'to', 'the', 'bank', 'to', '___']
predictions1 = ['withdraw', 'deposit', 'transfer', 'river', 'fish']
probs1 = [0.35, 0.30, 0.25, 0.05, 0.05]

ax1.set_title('Position 5: "bank" → Financial Context', fontsize=12, fontweight='bold')
bars1 = ax1.bar(predictions1, probs1,
                color=['#2ECC71' if p > 0.2 else '#95A5A6' for p in probs1],
                edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Probability', fontsize=11)
ax1.set_ylim(0, 0.4)
ax1.set_xlabel('Predicted Next Words', fontsize=11)

# Add sentence context
ax1.text(0.5, 0.95, ' '.join(sentence1), transform=ax1.transAxes,
         ha='center', va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor=color_light))

# Example 2: "bank" later with river context
sentence2 = ['The', 'river', 'flowed', 'by', 'the', 'grassy', 'bank', 'where', '___']
predictions2 = ['fish', 'ducks', 'trees', 'withdraw', 'deposit']
probs2 = [0.30, 0.25, 0.25, 0.10, 0.10]

ax2.set_title('Position 7: "bank" → Nature Context', fontsize=12, fontweight='bold')
bars2 = ax2.bar(predictions2, probs2,
                color=['#3498DB' if p > 0.2 else '#95A5A6' for p in probs2],
                edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Probability', fontsize=11)
ax2.set_ylim(0, 0.4)
ax2.set_xlabel('Predicted Next Words', fontsize=11)

# Add sentence context
ax2.text(0.5, 0.95, ' '.join(sentence2[:8]) + '...', transform=ax2.transAxes,
         ha='center', va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor=color_light))

plt.suptitle('Position Encoding Helps Disambiguation', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig('pred_09_position_aware.pdf')

# ============ ACT 4: PREDICTION QUALITY ============
print("\n=== Act 4: Prediction Quality ===")

# Chart 10: Perplexity Comparison
fig, ax = plt.subplots(figsize=(10, 6))

models = ['RNN', 'LSTM', 'GRU', 'Transformer']
perplexity = [50, 35, 32, 10]
colors = [color_rnn, '#E67E22', '#F39C12', color_transformer]

bars = ax.bar(models, perplexity, color=colors, edgecolor='black', linewidth=2)
ax.set_ylabel('Perplexity (Lower is Better)', fontsize=12, fontweight='bold')
ax.set_title('Prediction Uncertainty: How Many Words to Choose From?', fontsize=14, fontweight='bold')
ax.set_ylim(0, 60)

# Add values on bars
for bar, perp in zip(bars, perplexity):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 1,
            f'{perp}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add interpretation
    if perp == 50:
        interpretation = 'Confused among\n50 words'
    elif perp == 10:
        interpretation = 'Confident!\nOnly 10 words'
    else:
        interpretation = f'~{perp} words'

    ax.text(bar.get_x() + bar.get_width()/2, height/2,
            interpretation, ha='center', va='center', fontsize=10,
            color='white', fontweight='bold')

# Add explanation
ax.text(0.5, 0.95, 'Perplexity = Average number of word choices model considers',
        transform=ax.transAxes, ha='center', va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor=color_light, alpha=0.9))

save_fig('pred_10_perplexity.pdf')

# Chart 11: Top-5 Prediction Accuracy
fig, ax = plt.subplots(figsize=(10, 6))

models = ['RNN', 'LSTM', 'Transformer\n(Small)', 'Transformer\n(Large)']
top1_acc = [45, 52, 68, 78]
top5_acc = [60, 70, 90, 95]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, top1_acc, width, label='Top-1 Accuracy',
               color='#E74C3C', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, top5_acc, width, label='Top-5 Accuracy',
               color='#27AE60', edgecolor='black', linewidth=1.5)

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('How Often is the Correct Word in Top Predictions?', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 100)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontweight='bold')

# Highlight transformer improvement
ax.add_patch(plt.Rectangle((2.3, 0), 1.4, 100, color='green', alpha=0.05))
ax.text(3, 85, 'Transformers\nexcel at\nprediction', ha='center', fontsize=10,
        color='darkgreen', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

save_fig('pred_11_top5_accuracy.pdf')

# Chart 12: Effective Context Length
fig, ax = plt.subplots(figsize=(11, 6))

models = ['RNN', 'LSTM', 'GRU', 'Transformer\n(Base)', 'Transformer\n(Large)']
context_lengths = [5, 10, 12, 50, 100]
colors_grad = plt.cm.Greens(np.linspace(0.3, 0.9, len(models)))

bars = ax.bar(models, context_lengths, color=colors_grad, edgecolor='black', linewidth=2)
ax.set_ylabel('Effective Context Length (words)', fontsize=12, fontweight='bold')
ax.set_title('How Much Context Can Each Model Actually Use?', fontsize=14, fontweight='bold')
ax.set_ylim(0, 120)

# Add values and interpretations
for bar, length in zip(bars, context_lengths):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 2,
            f'{length} words', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add example context size
    if length <= 10:
        example = 'One sentence'
    elif length <= 50:
        example = 'Full paragraph'
    else:
        example = 'Multiple paragraphs!'

    ax.text(bar.get_x() + bar.get_width()/2, height/2,
            example, ha='center', va='center', fontsize=10,
            color='white' if length > 20 else 'black', fontweight='bold')

# Add comparison lines
ax.axhline(y=7, color='red', linestyle='--', alpha=0.5, label='Human short-term memory')
ax.axhline(y=50, color='blue', linestyle='--', alpha=0.5, label='Typical paragraph')

ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

save_fig('pred_12_context_length.pdf')

print("\n" + "="*60)
print("All 12 prediction-focused charts generated successfully!")
print("Charts emphasize next-word prediction theory for BSc level")
print("="*60)