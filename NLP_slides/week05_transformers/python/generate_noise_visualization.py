import matplotlib.pyplot as plt
import numpy as np

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Sentence for analysis
sentence = "The bank agreed to provide a loan for the house by the river bank after reviewing the application"
words = sentence.split()
n_words = len(words)

# Subplot 1: Connection density heatmap
ax1 = axes[0, 0]
# Create a random connection matrix (simulating all connections)
np.random.seed(42)
connection_matrix = np.random.rand(n_words, n_words)
np.fill_diagonal(connection_matrix, 1.0)

# Make it symmetric
connection_matrix = (connection_matrix + connection_matrix.T) / 2

# Highlight important connections (make them stand out)
important_pairs = [(1, 13), (13, 1), (4, 5), (5, 4), (12, 13), (13, 12)]
for i, j in important_pairs:
    connection_matrix[i, j] = 0.95

im1 = ax1.imshow(connection_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
ax1.set_title('Connection Matrix: 17×17 = 289 connections!', fontsize=12, weight='bold')
ax1.set_xlabel('Word Index', fontsize=10)
ax1.set_ylabel('Word Index', fontsize=10)
plt.colorbar(im1, ax=ax1, label='Connection Strength')

# Add grid
ax1.set_xticks(np.arange(0, n_words, 5))
ax1.set_yticks(np.arange(0, n_words, 5))
ax1.grid(True, alpha=0.3, linewidth=0.5)

# Subplot 2: Signal vs Noise ratio
ax2 = axes[0, 1]
categories = ['Important\nConnections', 'Noise\nConnections']
values = [len(important_pairs), n_words*n_words - len(important_pairs)]
colors = ['green', 'red']

bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_title('Signal vs Noise', fontsize=12, weight='bold')
ax2.set_ylabel('Number of Connections', fontsize=10)

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{val}\n({val/sum(values)*100:.1f}%)',
            ha='center', va='bottom', fontsize=11, weight='bold')

# Add noise ratio annotation
ax2.text(0.5, max(values)*0.8, f'Noise Ratio:\n{values[1]/sum(values)*100:.0f}%',
        ha='center', fontsize=14, weight='bold', color='darkred',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Subplot 3: Attention distribution (what should happen)
ax3 = axes[1, 0]
word_importance = np.zeros(n_words)
# Set importance for key words
key_words = {
    1: 0.8,   # bank (1)
    4: 0.6,   # provide
    5: 0.7,   # loan
    8: 0.5,   # house
    11: 0.6,  # river
    13: 0.8,  # bank (2)
    15: 0.5   # reviewing
}
for idx, importance in key_words.items():
    word_importance[idx] = importance

# Others get minimal attention
for i in range(n_words):
    if i not in key_words:
        word_importance[i] = 0.1

bars3 = ax3.bar(range(n_words), word_importance, color='blue', alpha=0.6, edgecolor='darkblue')
ax3.set_title('Ideal Attention Distribution (What We Need)', fontsize=12, weight='bold')
ax3.set_xlabel('Word Position', fontsize=10)
ax3.set_ylabel('Attention Weight', fontsize=10)
ax3.set_ylim([0, 1])

# Highlight the two "bank" words
bars3[1].set_color('red')
bars3[1].set_alpha(0.8)
bars3[13].set_color('green')
bars3[13].set_alpha(0.8)

# Add word labels for important words
for idx in [1, 13]:
    ax3.text(idx, word_importance[idx] + 0.05, words[idx],
            ha='center', fontsize=10, weight='bold')

# Subplot 4: Performance degradation curve
ax4 = axes[1, 1]
sentence_lengths = np.arange(3, 51, 2)
# Performance drops as sentence length increases
performance = 100 * np.exp(-0.05 * (sentence_lengths - 3))
performance = np.maximum(performance, 40)  # Floor at 40%

ax4.plot(sentence_lengths, performance, 'r-', linewidth=3, label='Without Attention')
# With attention mechanism (stays high)
performance_attention = 95 - 0.2 * (sentence_lengths - 3)
performance_attention = np.maximum(performance_attention, 85)
ax4.plot(sentence_lengths, performance_attention, 'g-', linewidth=3, label='With Attention')

ax4.fill_between(sentence_lengths, 0, 60, alpha=0.2, color='red', label='Failure Zone')
ax4.fill_between(sentence_lengths, 85, 100, alpha=0.2, color='green', label='Success Zone')

ax4.set_title('Performance vs Sentence Length', fontsize=12, weight='bold')
ax4.set_xlabel('Number of Words', fontsize=10)
ax4.set_ylabel('Accuracy (%)', fontsize=10)
ax4.set_ylim([0, 105])
ax4.grid(True, alpha=0.3)
ax4.legend(loc='lower left')

# Add critical point annotation
critical_idx = np.where(performance < 60)[0][0]
critical_length = sentence_lengths[critical_idx]
ax4.annotate('Critical Failure Point',
            xy=(critical_length, performance[critical_idx]),
            xytext=(critical_length+5, performance[critical_idx]+10),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, weight='bold', color='darkred')

# Main title
fig.suptitle('FAILURE: Signal Lost in Noise (Long Sentences)', fontsize=16, weight='bold', color='darkred')

# Add main insight box
fig.text(0.5, 0.02,
        'Key Problem: Without selective attention, important connections are drowned in noise.\n' +
        f'Example: "{sentence[:50]}..." has {n_words*n_words} possible connections but only ~6 matter!',
        ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.savefig('../figures/noise_visualization.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Generated noise_visualization.pdf")