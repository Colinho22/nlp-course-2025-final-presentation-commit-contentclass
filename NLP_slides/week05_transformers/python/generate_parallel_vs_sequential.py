import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Create figure with two panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Common setup
words = ["The", "cat", "sat", "on", "the", "mat"]
n_words = len(words)
colors = plt.cm.Set3(np.linspace(0, 1, n_words))

# Left panel: Sequential (RNN)
ax1.set_title('Sequential Processing (RNN)\n100 words = 100 time steps',
              fontsize=12, weight='bold')
ax1.set_xlim(-0.5, 7)
ax1.set_ylim(-0.5, n_words + 0.5)
ax1.set_xlabel('Time Steps →', fontsize=11)
ax1.set_ylabel('Words', fontsize=11)

# Draw sequential processing
for i, word in enumerate(words):
    # Draw processing boxes in sequence
    for t in range(i + 1):
        if t == i:
            # Current processing
            rect = patches.Rectangle((t, n_words-i-1), 0.8, 0.8,
                                    linewidth=2, edgecolor='red',
                                    facecolor=colors[i], alpha=0.9)
            ax1.add_patch(rect)
            ax1.text(t+0.4, n_words-i-0.5, word, ha='center', va='center',
                    fontsize=10, weight='bold')
        else:
            # Already processed
            rect = patches.Rectangle((t, n_words-i-1), 0.8, 0.8,
                                    linewidth=1, edgecolor='gray',
                                    facecolor='lightgray', alpha=0.3)
            ax1.add_patch(rect)

    # Draw arrow to next
    if i < n_words - 1:
        ax1.arrow(i+0.9, n_words-i-0.5, 0.15, 0, head_width=0.15,
                 head_length=0.05, fc='black', ec='black')

# Add timing annotation
ax1.text(3, -1.2, 'Total Time = N × processing_time\nExample: 100 words × 1s = 100 seconds',
         ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Right panel: Parallel (Transformer)
ax2.set_title('Parallel Processing (Transformer)\n100 words = 1 time step!',
              fontsize=12, weight='bold', color='green')
ax2.set_xlim(-0.5, 3)
ax2.set_ylim(-0.5, n_words + 0.5)
ax2.set_xlabel('Time Step →', fontsize=11)
ax2.set_ylabel('Words', fontsize=11)

# Draw all words processing simultaneously
for i, word in enumerate(words):
    # All processing at once
    rect = patches.Rectangle((1, n_words-i-1), 0.8, 0.8,
                            linewidth=2, edgecolor='green',
                            facecolor=colors[i], alpha=0.9)
    ax2.add_patch(rect)
    ax2.text(1.4, n_words-i-0.5, word, ha='center', va='center',
            fontsize=10, weight='bold')

# Draw input and output arrows
for i in range(n_words):
    # Input arrows
    ax2.arrow(0.2, n_words-i-0.5, 0.6, 0, head_width=0.1,
             head_length=0.05, fc='blue', ec='blue', alpha=0.5)
    # Output arrows
    ax2.arrow(1.9, n_words-i-0.5, 0.6, 0, head_width=0.1,
             head_length=0.05, fc='red', ec='red', alpha=0.5)

# Add parallel processing indicator
ax2.add_patch(patches.FancyBboxPatch((0.7, -0.3), 1.4, n_words+0.1,
                                     boxstyle="round,pad=0.1",
                                     facecolor='lightgreen', alpha=0.2,
                                     edgecolor='green', linewidth=3))
ax2.text(1.4, n_words+0.8, 'All at once!', ha='center', fontsize=12,
         weight='bold', color='green')

# Add timing annotation
ax2.text(1.4, -1.2, 'Total Time = 1 × processing_time\nExample: 100 words × 1s = 1 second!',
         ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# Remove ticks
ax1.set_xticks(range(7))
ax1.set_yticks([])
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(['Input', 'Process', 'Output'])
ax2.set_yticks([])

# Add speedup annotation
fig.text(0.5, 0.02, '🚀 90× Speedup through Parallelization!',
         ha='center', fontsize=14, weight='bold', color='red',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('../figures/parallel_vs_sequential.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Generated parallel_vs_sequential.pdf")