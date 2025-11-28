"""
Generate visualization charts for the word embeddings discovery handout
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Educational color scheme
COLOR_CURRENT = '#FF6B6B'  # Red - current position/focus
COLOR_CONTEXT = '#4ECDC4'  # Teal - context/surrounding
COLOR_PREDICT = '#95E77E'  # Green - predictions/output
COLOR_NEUTRAL = '#E0E0E0'  # Gray - neutral elements

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

def generate_2d_word_space():
    """Generate a 2D word space visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Word positions
    words = {
        'cat': [2, 3],
        'dog': [3, 3],
        'kitten': [1.5, 2.5],
        'car': [8, 1],
        'truck': [9, 1.5],
        'vehicle': [8.5, 0.5],
        'pet': [2.5, 3.5],
        'animal': [2, 4]
    }

    # Color code by category
    animal_words = ['cat', 'dog', 'kitten', 'pet', 'animal']
    vehicle_words = ['car', 'truck', 'vehicle']

    # Plot words
    for word, pos in words.items():
        if word in animal_words:
            color = COLOR_CONTEXT
            marker = 'o'
        else:
            color = COLOR_CURRENT
            marker = 's'

        ax.scatter(pos[0], pos[1], s=200, c=color, alpha=0.6,
                  edgecolors='black', linewidths=2, marker=marker)
        ax.annotate(word, (pos[0], pos[1]), fontsize=11, fontweight='bold',
                   ha='center', va='center')

    # Draw similarity circles
    ax.add_patch(plt.Circle((2.5, 3), 1.5, fill=False,
                           edgecolor=COLOR_CONTEXT, linestyle='--', alpha=0.5))
    ax.add_patch(plt.Circle((8.5, 1), 1.2, fill=False,
                           edgecolor=COLOR_CURRENT, linestyle='--', alpha=0.5))

    # Labels
    ax.text(2.5, 4.8, 'Animal Cluster', fontsize=12, fontweight='bold',
           ha='center', color=COLOR_CONTEXT)
    ax.text(8.5, 2.5, 'Vehicle Cluster', fontsize=12, fontweight='bold',
           ha='center', color=COLOR_CURRENT)

    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title('Word Embeddings in 2D Space - Similar Words Cluster Together',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig('../figures/discovery_2d_word_space.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_one_hot_visualization():
    """Visualize one-hot encoding limitations"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # One-hot vectors
    vocab = ['cat', 'dog', 'mat', 'sat', 'hat']
    vectors = np.eye(5)

    # Visualize vectors
    im = ax1.imshow(vectors, cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)
    ax1.set_yticks(range(5))
    ax1.set_yticklabels(vocab)
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(['d1', 'd2', 'd3', 'd4', 'd5'])
    ax1.set_title('One-Hot Encoding Vectors', fontsize=14, fontweight='bold')

    # Add values
    for i in range(5):
        for j in range(5):
            text = ax1.text(j, i, int(vectors[i, j]),
                          ha="center", va="center", color="black", fontweight='bold')

    # Similarity matrix
    similarity = np.dot(vectors, vectors.T)
    im2 = ax2.imshow(similarity, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(vocab, rotation=45)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(vocab)
    ax2.set_title('Similarity Matrix (All Orthogonal!)', fontsize=14, fontweight='bold')

    # Add values
    for i in range(5):
        for j in range(5):
            text = ax2.text(j, i, int(similarity[i, j]),
                          ha="center", va="center",
                          color="white" if i==j else "black",
                          fontweight='bold')

    # Add colorbars
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.suptitle('The Problem with One-Hot Encoding: No Semantic Similarity',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/discovery_one_hot_problem.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_word_arithmetic():
    """Visualize word arithmetic in vector space"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define vectors
    king = np.array([5, 3])
    man = np.array([4, 2])
    woman = np.array([4, 1])
    queen = np.array([5, 2])  # king - man + woman

    # Plot vectors as points
    ax.scatter(*king, s=300, c=COLOR_CURRENT, marker='*',
              edgecolors='black', linewidths=2, label='king', zorder=5)
    ax.scatter(*queen, s=300, c=COLOR_PREDICT, marker='*',
              edgecolors='black', linewidths=2, label='queen', zorder=5)
    ax.scatter(*man, s=200, c=COLOR_CONTEXT, marker='o',
              edgecolors='black', linewidths=2, label='man', zorder=5)
    ax.scatter(*woman, s=200, c=COLOR_CONTEXT, marker='o',
              edgecolors='black', linewidths=2, label='woman', zorder=5)

    # Draw arrows showing the arithmetic
    # king - man
    ax.arrow(king[0], king[1], -1, -1, head_width=0.1, head_length=0.1,
            fc='red', ec='red', linestyle='--', alpha=0.5)
    ax.text(4.5, 2.5, '- man', fontsize=10, color='red')

    # + woman
    ax.arrow(man[0], man[1], 0, -1, head_width=0.1, head_length=0.1,
            fc='blue', ec='blue', linestyle='--', alpha=0.5)
    ax.text(3.5, 1.5, '+ woman', fontsize=10, color='blue')

    # Result arrow to queen
    ax.arrow(4, 1, 0.8, 0.8, head_width=0.15, head_length=0.1,
            fc=COLOR_PREDICT, ec=COLOR_PREDICT, linewidth=2)
    ax.text(4.5, 1.5, '= queen!', fontsize=12, fontweight='bold',
           color=COLOR_PREDICT)

    # Annotations
    ax.annotate('king', king, xytext=(5.2, 3.2), fontsize=12, fontweight='bold')
    ax.annotate('queen', queen, xytext=(5.2, 2.2), fontsize=12, fontweight='bold')
    ax.annotate('man', man, xytext=(4.2, 2.2), fontsize=12, fontweight='bold')
    ax.annotate('woman', woman, xytext=(4.2, 0.8), fontsize=12, fontweight='bold')

    # Add relationship lines
    ax.plot([king[0], queen[0]], [king[1], queen[1]], 'g--', alpha=0.3, linewidth=2)
    ax.plot([man[0], woman[0]], [man[1], woman[1]], 'g--', alpha=0.3, linewidth=2)

    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title('Word Arithmetic: king - man + woman = queen',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2, 7)
    ax.set_ylim(0, 4)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('../figures/discovery_word_arithmetic.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_context_matters():
    """Visualize why context matters for word embeddings"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Static embedding problem
    ax1.set_title('Problem: Static Embeddings', fontsize=14, fontweight='bold')

    # Plot words
    bank_static = [5, 5]
    money = [8, 8]
    river = [2, 2]
    water = [1, 3]
    finance = [9, 7]

    ax1.scatter(*bank_static, s=400, c=COLOR_CURRENT, marker='*',
               edgecolors='black', linewidths=2, zorder=5)
    ax1.scatter(*money, s=200, c='gold', marker='o',
               edgecolors='black', linewidths=2, zorder=5)
    ax1.scatter(*river, s=200, c='blue', marker='o',
               edgecolors='black', linewidths=2, zorder=5)
    ax1.scatter(*water, s=200, c='blue', marker='o',
               edgecolors='black', linewidths=2, zorder=5, alpha=0.5)
    ax1.scatter(*finance, s=200, c='gold', marker='o',
               edgecolors='black', linewidths=2, zorder=5, alpha=0.5)

    # Labels
    ax1.annotate('bank', bank_static, xytext=(5.2, 5.5),
                fontsize=12, fontweight='bold', color=COLOR_CURRENT)
    ax1.annotate('money', money, xytext=(8.2, 8.2), fontsize=10)
    ax1.annotate('river', river, xytext=(2.2, 1.8), fontsize=10)
    ax1.annotate('water', water, xytext=(1.2, 3.2), fontsize=10, alpha=0.7)
    ax1.annotate('finance', finance, xytext=(9.2, 7.2), fontsize=10, alpha=0.7)

    # Show conflict with arrows
    ax1.arrow(bank_static[0], bank_static[1], 2.5, 2.5,
             head_width=0.2, head_length=0.2, fc='red', ec='red',
             linestyle='--', alpha=0.5)
    ax1.arrow(bank_static[0], bank_static[1], -2.5, -2.5,
             head_width=0.2, head_length=0.2, fc='red', ec='red',
             linestyle='--', alpha=0.5)
    ax1.text(6.5, 6.5, '?', fontsize=20, fontweight='bold', color='red')
    ax1.text(3.5, 3.5, '?', fontsize=20, fontweight='bold', color='red')

    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Dimension 1', fontsize=12)
    ax1.set_ylabel('Dimension 2', fontsize=12)

    # Contextual embedding solution
    ax2.set_title('Solution: Contextual Embeddings', fontsize=14, fontweight='bold')

    # Two different bank embeddings
    bank_financial = [8, 7.5]
    bank_river = [2, 2.5]

    ax2.scatter(*bank_financial, s=400, c=COLOR_PREDICT, marker='*',
               edgecolors='black', linewidths=2, zorder=5)
    ax2.scatter(*bank_river, s=400, c=COLOR_PREDICT, marker='*',
               edgecolors='black', linewidths=2, zorder=5)
    ax2.scatter(*money, s=200, c='gold', marker='o',
               edgecolors='black', linewidths=2, zorder=5)
    ax2.scatter(*river, s=200, c='blue', marker='o',
               edgecolors='black', linewidths=2, zorder=5)
    ax2.scatter(*water, s=200, c='blue', marker='o',
               edgecolors='black', linewidths=2, zorder=5, alpha=0.5)
    ax2.scatter(*finance, s=200, c='gold', marker='o',
               edgecolors='black', linewidths=2, zorder=5, alpha=0.5)

    # Labels
    ax2.annotate('bank\n(financial)', bank_financial, xytext=(8.2, 8.5),
                fontsize=10, fontweight='bold', color=COLOR_PREDICT, ha='center')
    ax2.annotate('bank\n(river)', bank_river, xytext=(2, 3.5),
                fontsize=10, fontweight='bold', color=COLOR_PREDICT, ha='center')
    ax2.annotate('money', money, xytext=(8.2, 8.2), fontsize=10)
    ax2.annotate('river', river, xytext=(2.2, 1.8), fontsize=10)
    ax2.annotate('water', water, xytext=(1.2, 3.2), fontsize=10, alpha=0.7)
    ax2.annotate('finance', finance, xytext=(9.2, 7.2), fontsize=10, alpha=0.7)

    # Show connections
    ax2.plot([bank_financial[0], money[0]], [bank_financial[1], money[1]],
            'g--', alpha=0.5, linewidth=2)
    ax2.plot([bank_river[0], river[0]], [bank_river[1], river[1]],
            'g--', alpha=0.5, linewidth=2)

    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Dimension 1', fontsize=12)
    ax2.set_ylabel('Dimension 2', fontsize=12)

    plt.suptitle('Context Matters: Same Word, Different Meanings',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/discovery_context_matters.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_similarity_heatmap():
    """Generate a heatmap showing word similarities"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Words for comparison
    words = ['cat', 'kitten', 'dog', 'puppy', 'car', 'truck', 'happy', 'joyful']

    # Character-based similarity (problematic)
    char_sim = np.zeros((8, 8))
    for i, w1 in enumerate(words):
        for j, w2 in enumerate(words):
            if i == j:
                char_sim[i, j] = 1.0
            else:
                # Simple character overlap
                common = len(set(w1) & set(w2))
                char_sim[i, j] = common / max(len(set(w1)), len(set(w2)))

    # Semantic similarity (what we want)
    # Manually defined for demonstration
    semantic_sim = np.array([
        [1.0, 0.9, 0.7, 0.6, 0.1, 0.1, 0.2, 0.2],  # cat
        [0.9, 1.0, 0.6, 0.7, 0.1, 0.1, 0.2, 0.2],  # kitten
        [0.7, 0.6, 1.0, 0.9, 0.1, 0.1, 0.2, 0.2],  # dog
        [0.6, 0.7, 0.9, 1.0, 0.1, 0.1, 0.3, 0.3],  # puppy
        [0.1, 0.1, 0.1, 0.1, 1.0, 0.8, 0.0, 0.0],  # car
        [0.1, 0.1, 0.1, 0.1, 0.8, 1.0, 0.0, 0.0],  # truck
        [0.2, 0.2, 0.2, 0.3, 0.0, 0.0, 1.0, 0.9],  # happy
        [0.2, 0.2, 0.2, 0.3, 0.0, 0.0, 0.9, 1.0],  # joyful
    ])

    # Plot character similarity
    im1 = ax1.imshow(char_sim, cmap='YlOrRd', vmin=0, vmax=1)
    ax1.set_xticks(range(8))
    ax1.set_xticklabels(words, rotation=45, ha='right')
    ax1.set_yticks(range(8))
    ax1.set_yticklabels(words)
    ax1.set_title('Character-based Similarity\n(Not Meaningful!)',
                 fontsize=12, fontweight='bold')

    # Add values
    for i in range(8):
        for j in range(8):
            text = ax1.text(j, i, f'{char_sim[i, j]:.2f}',
                          ha="center", va="center",
                          color="white" if char_sim[i, j] > 0.5 else "black",
                          fontsize=8)

    # Plot semantic similarity
    im2 = ax2.imshow(semantic_sim, cmap='YlGnBu', vmin=0, vmax=1)
    ax2.set_xticks(range(8))
    ax2.set_xticklabels(words, rotation=45, ha='right')
    ax2.set_yticks(range(8))
    ax2.set_yticklabels(words)
    ax2.set_title('Semantic Similarity\n(What Embeddings Capture!)',
                 fontsize=12, fontweight='bold')

    # Add values
    for i in range(8):
        for j in range(8):
            text = ax2.text(j, i, f'{semantic_sim[i, j]:.1f}',
                          ha="center", va="center",
                          color="white" if semantic_sim[i, j] > 0.5 else "black",
                          fontsize=8)

    # Add colorbars
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.suptitle('Character Similarity vs. Semantic Similarity',
                fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig('../figures/discovery_similarity_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all discovery handout figures"""
    print("Generating discovery handout figures...")

    print("  - 2D word space...")
    generate_2d_word_space()

    print("  - One-hot encoding visualization...")
    generate_one_hot_visualization()

    print("  - Word arithmetic...")
    generate_word_arithmetic()

    print("  - Context matters...")
    generate_context_matters()

    print("  - Similarity heatmap...")
    generate_similarity_heatmap()

    print("All figures generated successfully in ../figures/")

if __name__ == "__main__":
    main()