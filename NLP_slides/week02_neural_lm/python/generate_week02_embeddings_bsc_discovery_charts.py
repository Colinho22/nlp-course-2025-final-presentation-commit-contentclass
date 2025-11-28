"""
Generate BSc-level discovery-based visualizations for Week 2: Word Embeddings
Two-Tier Presentation: 20 concise main + 15 deep appendix

Date: 2025-10-27
Charts: 18 visualizations (15 main + 3 appendix)
Focus: ALL four visualization types - 2D/3D spaces, architectures, training, similarity
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D
import seaborn as sns
import os

# Create figures directory
os.makedirs('../figures', exist_ok=True)

# Educational Framework colors
COLOR_MLPURPLE = '#3333B2'
COLOR_DARKGRAY = '#404040'
COLOR_MIDGRAY = '#B4B4B4'
COLOR_LIGHTGRAY = '#F0F0F0'

COLOR_CURRENT = '#FF6B6B'
COLOR_CONTEXT = '#4ECDC4'
COLOR_PREDICT = '#95E77E'
COLOR_NEUTRAL = '#E0E0E0'

def set_minimalist_style(ax):
    """Apply minimalist style"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_MIDGRAY)
    ax.spines['bottom'].set_color(COLOR_MIDGRAY)
    ax.tick_params(colors=COLOR_DARKGRAY, which='both')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_facecolor('white')

# ===== MAIN PRESENTATION CHARTS (1-15) =====

# Chart 1: Word Arithmetic 3D
def plot_word_arithmetic_3d():
    """3D visualization of king - man + woman = queen"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Simplified 3D positions (for visualization)
    positions = {
        'king': np.array([2, 3, 2]),
        'queen': np.array([2, 3, -2]),
        'man': np.array([0, 0, 2]),
        'woman': np.array([0, 0, -2])
    }

    colors = {'king': '#3498DB', 'queen': '#E74C3C', 'man': '#9B59B6', 'woman': '#F39C12'}

    # Plot words
    for word, pos in positions.items():
        ax.scatter(*pos, s=300, c=colors[word], alpha=0.8, edgecolor='black', linewidth=2)
        ax.text(pos[0], pos[1], pos[2], f'  {word}', fontsize=12, fontweight='bold')

    # Vector operations
    # king - man
    ax.quiver(*positions['king'], *(positions['man'] - positions['king']),
              arrow_length_ratio=0.15, color='red', linewidth=2.5, alpha=0.7)
    ax.text(1, 1.5, 1, 'king - man', fontsize=10, color='red', fontweight='bold')

    # + woman
    result_intermediate = positions['king'] - positions['man']
    ax.quiver(*result_intermediate, *(positions['woman'] - np.array([0,0,0])),
              arrow_length_ratio=0.15, color='green', linewidth=2.5, alpha=0.7)

    # = queen (approximately)
    result = positions['king'] - positions['man'] + positions['woman']
    ax.scatter(*result, s=400, marker='*', c='gold', edgecolor='black', linewidth=2)
    ax.text(result[0], result[1], result[2]+0.5, '  Result\n  (≈ queen)', fontsize=10,
            fontweight='bold', color='gold')

    ax.set_xlabel('Dimension 1', fontsize=11, fontweight='bold')
    ax.set_ylabel('Dimension 2', fontsize=11, fontweight='bold')
    ax.set_zlabel('Dimension 3', fontsize=11, fontweight='bold')
    ax.set_title('Word Arithmetic in 3D Embedding Space', fontsize=14, fontweight='bold')

    ax.view_init(elev=20, azim=-45)

    plt.tight_layout()
    plt.savefig('../figures/word_arithmetic_3d_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 2: Similarity Clustering 2D
def plot_similarity_clustering_2d():
    """2D visualization showing semantic clusters"""
    fig, ax = plt.subplots(figsize=(11, 9))

    # Semantic categories
    animals = {'cat': [2, 3], 'dog': [2.3, 2.8], 'lion': [1.8, 3.2], 'tiger': [1.7, 2.9]}
    vehicles = {'car': [8, 2], 'truck': [8.2, 2.3], 'bus': [7.8, 1.9], 'bike': [8.1, 2.1]}
    colors_dict = {'red': [5, 8], 'blue': [5.3, 7.9], 'green': [4.9, 8.2], 'yellow': [5.2, 7.7]}

    # Plot clusters
    for word, pos in animals.items():
        ax.scatter(*pos, s=250, c='#E74C3C', alpha=0.7, edgecolor='black', linewidth=2)
        ax.text(pos[0], pos[1]+0.3, word, ha='center', fontsize=10, fontweight='bold')

    for word, pos in vehicles.items():
        ax.scatter(*pos, s=250, c='#3498DB', alpha=0.7, edgecolor='black', linewidth=2)
        ax.text(pos[0], pos[1]+0.3, word, ha='center', fontsize=10, fontweight='bold')

    for word, pos in colors_dict.items():
        ax.scatter(*pos, s=250, c='#F39C12', alpha=0.7, edgecolor='black', linewidth=2)
        ax.text(pos[0], pos[1]+0.3, word, ha='center', fontsize=10, fontweight='bold')

    # Cluster boundaries
    from matplotlib.patches import Ellipse
    animals_ellipse = Ellipse((2, 3), 1.2, 0.9, fill=False, edgecolor='#E74C3C',
                              linewidth=2.5, linestyle='--', alpha=0.6)
    ax.add_patch(animals_ellipse)
    ax.text(2, 4.2, 'Animals', ha='center', fontsize=11, fontweight='bold', color='#E74C3C')

    vehicles_ellipse = Ellipse((8, 2), 1, 0.8, fill=False, edgecolor='#3498DB',
                               linewidth=2.5, linestyle='--', alpha=0.6)
    ax.add_patch(vehicles_ellipse)
    ax.text(8, 3.2, 'Vehicles', ha='center', fontsize=11, fontweight='bold', color='#3498DB')

    colors_ellipse = Ellipse((5, 8), 1, 0.8, fill=False, edgecolor='#F39C12',
                             linewidth=2.5, linestyle='--', alpha=0.6)
    ax.add_patch(colors_ellipse)
    ax.text(5, 9.2, 'Colors', ha='center', fontsize=11, fontweight='bold', color='#F39C12')

    ax.set_xlabel('Embedding Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Embedding Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title('Semantic Clustering in 2D Embedding Space', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/similarity_clustering_2d_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 3: Dimensionality Comparison
def plot_dimensionality_comparison():
    """Compare sparse one-hot vs dense embeddings"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # One-hot (sparse)
    ax1.set_title('One-Hot: 50,000 Dimensions (Sparse)', fontsize=12, fontweight='bold')

    # Visualize sparsity
    sparse_vector = np.zeros(100)
    sparse_vector[42] = 1  # One position = 1

    ax1.bar(range(100), sparse_vector, color=COLOR_NEUTRAL, width=1)
    ax1.bar([42], [1], color=COLOR_CURRENT, width=1, label='Single 1')

    ax1.set_xlabel('Dimension Index', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1.2)
    ax1.legend(fontsize=10)
    ax1.text(50, 0.9, '99.998% zeros!', ha='center', fontsize=11, color='red',
            fontweight='bold')

    # Dense embedding
    ax2.set_title('Embedding: 300 Dimensions (Dense)', fontsize=12, fontweight='bold')

    # Visualize dense values
    np.random.seed(42)
    dense_vector = np.random.randn(50) * 0.3  # 50 shown out of 300

    colors_dense = [COLOR_PREDICT if v > 0 else COLOR_CONTEXT for v in dense_vector]
    ax2.bar(range(50), dense_vector, color=colors_dense, width=1, alpha=0.7)

    ax2.set_xlabel('Dimension Index (showing 50 of 300)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 50)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.text(25, 0.7, 'All dimensions active!', ha='center', fontsize=11,
            color='green', fontweight='bold')

    set_minimalist_style(ax1)
    set_minimalist_style(ax2)

    plt.tight_layout()
    plt.savefig('../figures/dimensionality_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 4: Context-Dependent Bank
def plot_context_dependent_bank():
    """Show 'bank' in two different contexts"""
    fig, ax = plt.subplots(figsize=(11, 9))

    # Two contexts for "bank"
    contexts = [
        ("The river bank was flooded", [2, 7], '#3498DB', 'River context'),
        ("Money in the bank is safe", [8, 3], '#27AE60', 'Finance context')
    ]

    # Plot "bank" at averaged position
    bank_pos = np.array([5, 5])
    ax.scatter(*bank_pos, s=500, c=COLOR_CURRENT, marker='*', edgecolor='black',
              linewidth=2.5, label='"bank" (static embedding)')
    ax.text(bank_pos[0], bank_pos[1]-0.6, 'bank\n(averages both meanings)',
            ha='center', fontsize=11, fontweight='bold', color=COLOR_CURRENT)

    # Plot contexts
    for context, pos, color, label in contexts:
        ax.scatter(*pos, s=300, c=color, alpha=0.6, edgecolor='black', linewidth=2)
        ax.text(pos[0], pos[1]+0.6, label, ha='center', fontsize=10,
                fontweight='bold', color=color)

        # Arrow from bank to context
        ax.annotate('', xy=pos, xytext=bank_pos,
                   arrowprops=dict(arrowstyle='<->', color=color, lw=2.5, linestyle='--'))

        # Context sentence
        ax.text(pos[0], pos[1]-0.8, f'"{context}"', ha='center', fontsize=8,
               style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Embedding Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Embedding Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title('Limitation: Static Embeddings Average Multiple Meanings', fontsize=14,
                 fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.legend(fontsize=10, loc='upper right')

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/context_dependent_bank_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 5: Word as Vector Concept
def plot_word_as_vector_concept():
    """Conceptual diagram of word to vector mapping"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Words
    words = ['cat', 'dog', 'mat', 'happiness']
    word_y = [4, 3, 2, 1]

    for word, y in zip(words, word_y):
        # Word box
        word_box = FancyBboxPatch((0.5, y), 2, 0.6, boxstyle="round,pad=0.08",
                                  facecolor=COLOR_LIGHTGRAY,
                                  edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(word_box)
        ax.text(1.5, y+0.3, word, ha='center', fontsize=11, fontweight='bold')

        # Arrow
        ax.annotate('', xy=(3.5, y+0.3), xytext=(2.5, y+0.3),
                   arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2.5))

        # Vector representation (showing first 8 dimensions)
        np.random.seed(hash(word) % 100)
        vector_vals = np.random.randn(8) * 0.5
        vector_text = '[' + ', '.join([f'{v:.2f}' for v in vector_vals[:4]]) + ', ...]'

        vector_box = FancyBboxPatch((4, y), 7, 0.6, boxstyle="round,pad=0.08",
                                    facecolor=COLOR_PREDICT, alpha=0.5,
                                    edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(vector_box)
        ax.text(7.5, y+0.3, vector_text, ha='center', fontsize=9, family='monospace')

    ax.text(1.5, 5, 'Words\n(Discrete)', ha='center', fontsize=12, fontweight='bold',
           color=COLOR_DARKGRAY)
    ax.text(7.5, 5, 'Vectors in $\\mathbb{R}^{300}$\n(Continuous)', ha='center',
            fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.5)
    ax.axis('off')
    ax.set_title('Words Become Dense Continuous Vectors', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/word_as_vector_concept_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 6: Semantic Space 2D (detailed)
def plot_semantic_space_2d():
    """Detailed 2D semantic space with multiple clusters"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # More comprehensive semantic groups
    groups = {
        'Animals': {
            'words': ['cat', 'dog', 'lion', 'tiger', 'elephant'],
            'positions': [[2,3], [2.5,2.8], [1.8,3.3], [1.6,2.7], [2.2,3.5]],
            'color': '#E74C3C'
        },
        'Vehicles': {
            'words': ['car', 'truck', 'bus', 'train', 'plane'],
            'positions': [[8,2], [8.3,2.4], [7.8,1.8], [8.5,2.2], [8.1,2.6]],
            'color': '#3498DB'
        },
        'Food': {
            'words': ['apple', 'banana', 'pizza', 'bread'],
            'positions': [[5,9], [5.3,8.8], [4.8,9.2], [5.1,9.4]],
            'color': '#F39C12'
        },
        'Numbers': {
            'words': ['one', 'two', 'three', 'four'],
            'positions': [[2,8], [2.2,7.8], [1.9,8.3], [2.3,8.1]],
            'color': '#9B59B6'
        }
    }

    for group_name, group_data in groups.items():
        words = group_data['words']
        positions = group_data['positions']
        color = group_data['color']

        # Plot points
        x_vals = [p[0] for p in positions]
        y_vals = [p[1] for p in positions]

        ax.scatter(x_vals, y_vals, s=200, c=color, alpha=0.7, edgecolor='black', linewidth=1.5)

        for word, pos in zip(words, positions):
            ax.text(pos[0], pos[1]-0.3, word, ha='center', fontsize=9, fontweight='bold')

        # Group label
        center_x = np.mean(x_vals)
        center_y = np.mean(y_vals)
        ax.text(center_x, center_y+1.2, group_name, ha='center', fontsize=12,
               fontweight='bold', color=color,
               bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=2))

    ax.set_xlabel('Embedding Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Embedding Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title('Semantic Space: Similar Words Cluster Together', fontsize=14,
                 fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/semantic_space_2d_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 7: Distributed Features
def plot_distributed_features():
    """Show dimensions as semantic features"""
    fig, ax = plt.subplots(figsize=(12, 8))

    words = ['king', 'queen', 'man', 'woman', 'cat', 'kitten']
    features = ['Gender:\nMale', 'Royalty', 'Adult', 'Animal', 'Small']

    # Feature values (hypothetical)
    feature_matrix = np.array([
        [0.9, 0.9, 0.9, 0.1, 0.1],  # king
        [-0.9, 0.9, 0.9, 0.1, 0.1],  # queen
        [0.9, 0.1, 0.9, 0.1, 0.1],  # man
        [-0.9, 0.1, 0.9, 0.1, 0.1],  # woman
        [0.1, 0.1, 0.8, 0.9, 0.4],  # cat
        [0.1, 0.1, 0.2, 0.9, 0.9]   # kitten
    ])

    # Heatmap
    im = ax.imshow(feature_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(words)))
    ax.set_xticklabels(features, fontsize=10, fontweight='bold')
    ax.set_yticklabels(words, fontsize=11, fontweight='bold')

    ax.set_xlabel('Semantic Features (Dimensions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Words', fontsize=12, fontweight='bold')
    ax.set_title('Distributed Representation: Each Dimension = Semantic Feature',
                 fontsize=13, fontweight='bold')

    # Add values
    for i in range(len(words)):
        for j in range(len(features)):
            ax.text(j, i, f'{feature_matrix[i,j]:.1f}', ha='center', va='center',
                   fontsize=9, fontweight='bold',
                   color='white' if abs(feature_matrix[i,j]) > 0.5 else 'black')

    plt.colorbar(im, ax=ax, label='Feature Activation')
    plt.tight_layout()
    plt.savefig('../figures/distributed_features_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 8: Skip-gram Architecture
def plot_skipgram_architecture():
    """Neural network diagram for Skip-gram"""
    fig, ax = plt.subplots(figsize=(11, 10))

    # Input layer
    input_box = FancyBboxPatch((1, 7.5), 2.5, 1.2, boxstyle="round,pad=0.12",
                               facecolor=COLOR_LIGHTGRAY,
                               edgecolor=COLOR_DARKGRAY, linewidth=2.5)
    ax.add_patch(input_box)
    ax.text(2.25, 8.4, 'Input', ha='center', fontsize=11, fontweight='bold')
    ax.text(2.25, 8.0, 'One-hot', ha='center', fontsize=9)
    ax.text(2.25, 7.7, f'|V| = 50K', ha='center', fontsize=9, style='italic')

    # Hidden layer (embedding)
    hidden_box = FancyBboxPatch((5, 5.5), 3, 2, boxstyle="round,pad=0.15",
                                facecolor=COLOR_MLPURPLE, alpha=0.3,
                                edgecolor=COLOR_DARKGRAY, linewidth=3)
    ax.add_patch(hidden_box)
    ax.text(6.5, 6.9, 'Hidden = Embedding', ha='center', fontsize=12, fontweight='bold')
    ax.text(6.5, 6.4, '$h = W^T x$', ha='center', fontsize=11)
    ax.text(6.5, 5.9, '$d = 300$', ha='center', fontsize=10, style='italic')

    # Output layer
    output_box = FancyBboxPatch((10, 7.5), 2.5, 1.2, boxstyle="round,pad=0.12",
                                facecolor=COLOR_PREDICT, alpha=0.5,
                                edgecolor=COLOR_DARKGRAY, linewidth=2.5)
    ax.add_patch(output_box)
    ax.text(11.25, 8.4, 'Output', ha='center', fontsize=11, fontweight='bold')
    ax.text(11.25, 8.0, 'Context predictions', ha='center', fontsize=9)
    ax.text(11.25, 7.7, f'|V| = 50K', ha='center', fontsize=9, style='italic')

    # Arrows
    ax.annotate('', xy=(5, 6.5), xytext=(3.5, 8.1),
               arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=3))
    ax.text(4, 7.5, '$W$\n$50K \\times 300$', ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLOR_DARKGRAY))

    ax.annotate('', xy=(10, 8.1), xytext=(8, 6.5),
               arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=3))
    ax.text(9, 7.5, "$W'$\n$300 \\times 50K$", ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLOR_DARKGRAY))

    # Example
    example_box = FancyBboxPatch((3.5, 3), 6, 1.8, boxstyle="round,pad=0.12",
                                 facecolor='#FFF9E6', alpha=0.7,
                                 edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(example_box)
    ax.text(6.5, 4.5, 'Example:', ha='center', fontsize=10, fontweight='bold')
    ax.text(6.5, 4.0, 'Input: "cat" (center word)', ha='center', fontsize=9)
    ax.text(6.5, 3.6, 'Hidden: cat embedding (300-dim)', ha='center', fontsize=9)
    ax.text(6.5, 3.2, 'Output: P(the), P(sat), ... (context)', ha='center', fontsize=9)

    ax.set_xlim(0, 14)
    ax.set_ylim(2, 10)
    ax.axis('off')
    ax.set_title('Skip-gram Architecture: 3-Layer Neural Network', fontsize=14,
                 fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/skipgram_architecture_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 9: CBOW Architecture
def plot_cbow_architecture():
    """CBOW neural network diagram"""
    fig, ax = plt.subplots(figsize=(11, 10))

    # Multiple inputs (context words)
    context_words = ['the', 'sat']
    for i, word in enumerate(context_words):
        y = 8 - i * 1.2
        input_box = FancyBboxPatch((0.5, y), 2, 0.8, boxstyle="round,pad=0.08",
                                   facecolor=COLOR_CONTEXT, alpha=0.5,
                                   edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(input_box)
        ax.text(1.5, y+0.4, f'"{word}"', ha='center', fontsize=10, fontweight='bold')

        # Arrow to hidden
        ax.annotate('', xy=(5, 6), xytext=(2.5, y+0.4),
                   arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))

    # Hidden layer (average)
    hidden_box = FancyBboxPatch((4.5, 5), 3, 1.5, boxstyle="round,pad=0.15",
                                facecolor=COLOR_MLPURPLE, alpha=0.3,
                                edgecolor=COLOR_DARKGRAY, linewidth=3)
    ax.add_patch(hidden_box)
    ax.text(6, 6.1, 'Hidden', ha='center', fontsize=11, fontweight='bold')
    ax.text(6, 5.7, 'Average context', ha='center', fontsize=9)
    ax.text(6, 5.3, '$d = 300$', ha='center', fontsize=9, style='italic')

    # Output (center word)
    output_box = FancyBboxPatch((10, 5.5), 2.5, 1, boxstyle="round,pad=0.12",
                                facecolor=COLOR_PREDICT, alpha=0.6,
                                edgecolor=COLOR_DARKGRAY, linewidth=2.5)
    ax.add_patch(output_box)
    ax.text(11.25, 6.3, 'Output', ha='center', fontsize=11, fontweight='bold')
    ax.text(11.25, 5.8, 'Predict "cat"', ha='center', fontsize=9)

    # Arrow
    ax.annotate('', xy=(10, 6), xytext=(7.5, 6),
               arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=3))

    # Comparison box
    comp_box = FancyBboxPatch((2, 2.5), 9, 1.8, boxstyle="round,pad=0.12",
                              facecolor='#E8F5E9', alpha=0.7,
                              edgecolor='green', linewidth=2)
    ax.add_patch(comp_box)
    ax.text(6.5, 3.9, 'CBOW vs Skip-gram', ha='center', fontsize=11, fontweight='bold')
    ax.text(6.5, 3.4, 'CBOW: Context → Word (faster, frequent words)', ha='center', fontsize=9)
    ax.text(6.5, 3.0, 'Skip-gram: Word → Context (better rare words)', ha='center', fontsize=9)
    ax.text(6.5, 2.7, 'Both produce similar quality embeddings', ha='center', fontsize=9,
           style='italic')

    ax.set_xlim(0, 13)
    ax.set_ylim(2, 9.5)
    ax.axis('off')
    ax.set_title('CBOW Architecture: Predict Word from Context', fontsize=14,
                 fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/cbow_architecture_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 10: Skip-gram Training Steps
def plot_skipgram_training_steps():
    """Step-by-step training visualization"""
    fig, ax = plt.subplots(figsize=(12, 9))

    steps_data = [
        (8, 'Input: "cat" (one-hot vector)', COLOR_LIGHTGRAY),
        (6.8, 'Lookup: embedding = W[cat_id]', COLOR_CONTEXT),
        (5.6, 'Compute: scores = W\' × embedding', COLOR_MLPURPLE),
        (4.4, 'Softmax: probabilities over vocab', COLOR_CONTEXT),
        (3.2, 'Loss: -log P(context|cat)', COLOR_CURRENT),
        (2, 'Backprop: Update W to increase P', COLOR_PREDICT)
    ]

    for y, text, color in steps_data:
        box = FancyBboxPatch((2, y-0.35), 9, 0.7, boxstyle="round,pad=0.1",
                             facecolor=color, alpha=0.4,
                             edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(box)
        ax.text(6.5, y, text, ha='center', fontsize=10,
               fontweight='bold' if y > 7 else 'normal')

        if y > 2:
            ax.annotate('', xy=(6.5, y-0.5), xytext=(6.5, y-0.35),
                       arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2.5))

    ax.set_xlim(0, 13)
    ax.set_ylim(1, 9)
    ax.axis('off')
    ax.set_title('Skip-gram Training: Forward and Backward Pass', fontsize=14,
                 fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/skipgram_training_steps_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 11: Softmax Problem Visualization
def plot_softmax_problem():
    """Show computational cost of softmax"""
    fig, ax = plt.subplots(figsize=(11, 7))

    # Vocabulary size on x-axis, computations on y-axis
    vocab_sizes = np.array([1000, 5000, 10000, 30000, 50000, 100000])
    computations = vocab_sizes  # One exp per word

    ax.plot(vocab_sizes, computations, marker='o', markersize=10, linewidth=3,
            color=COLOR_CURRENT, label='Softmax denominator')

    # Negative sampling (constant)
    k = 5
    ax.axhline(y=k, color=COLOR_PREDICT, linestyle='--', linewidth=3, alpha=0.8,
              label=f'Negative sampling (k={k})')

    ax.set_xlabel('Vocabulary Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Computations Per Training Example', fontsize=12, fontweight='bold')
    ax.set_title('The Softmax Bottleneck: Why We Need Tricks', fontsize=14,
                 fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=11, loc='upper left')

    # Annotation
    ax.annotate(f'{50000/5:.0f}x speedup!', xy=(50000, k), xytext=(20000, 100),
               arrowprops=dict(arrowstyle='->', color='green', lw=2.5),
               fontsize=12, fontweight='bold', color='green')

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/softmax_problem_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 12: Negative Sampling Process
def plot_negative_sampling_process():
    """Visual of positive + negative pairs"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Center word
    center_box = FancyBboxPatch((5, 6.5), 2, 1, boxstyle="round,pad=0.12",
                                facecolor=COLOR_MLPURPLE, alpha=0.5,
                                edgecolor=COLOR_DARKGRAY, linewidth=3)
    ax.add_patch(center_box)
    ax.text(6, 7, 'Center: "cat"', ha='center', fontsize=11, fontweight='bold')

    # Positive pair
    pos_box = FancyBboxPatch((0.5, 4), 2, 0.8, boxstyle="round,pad=0.08",
                             facecolor=COLOR_PREDICT, alpha=0.6,
                             edgecolor='green', linewidth=2.5)
    ax.add_patch(pos_box)
    ax.text(1.5, 4.4, 'Positive:\n"sat"', ha='center', fontsize=10, fontweight='bold')

    ax.annotate('', xy=(5, 7), xytext=(2.5, 4.8),
               arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax.text(3.5, 6, 'Maximize\n$\\sigma(v_{sat} \\cdot v_{cat})$', ha='center',
           fontsize=8, color='green', fontweight='bold')

    # Negative samples
    negatives = ['xylophone', 'democracy', 'telescope', 'ancient', 'purple']
    for i, neg_word in enumerate(negatives):
        y = 3.5 - i * 0.7
        neg_box = FancyBboxPatch((9, y), 2.5, 0.5, boxstyle="round,pad=0.06",
                                 facecolor=COLOR_CURRENT, alpha=0.4,
                                 edgecolor='red', linewidth=1.5)
        ax.add_patch(neg_box)
        ax.text(10.25, y+0.25, f'Negative: "{neg_word}"', ha='center', fontsize=8)

        ax.annotate('', xy=(8.5, y+0.25), xytext=(7, 7),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.6))

    ax.text(8, 5.5, 'Minimize\n$\\sigma(-v_{neg} \\cdot v_{cat})$', ha='center',
           fontsize=8, color='red', fontweight='bold')

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Negative Sampling: 1 Positive + k Negatives', fontsize=14,
                 fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/negative_sampling_process_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 13: Training Evolution Animation (static snapshot)
def plot_training_evolution():
    """Show how embeddings evolve during training"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    titles = ['Epoch 1\n(Random)', 'Epoch 5\n(Learning)', 'Epoch 20\n(Converged)']

    for ax_idx, title in enumerate(titles):
        ax = axes[ax_idx]

        # Simulate embeddings at different epochs
        np.random.seed(42 + ax_idx)

        if ax_idx == 0:  # Random
            animals = np.random.randn(4, 2) * 3
            vehicles = np.random.randn(4, 2) * 3
        elif ax_idx == 1:  # Learning
            animals = np.random.randn(4, 2) * 0.8 + np.array([2, 2])
            vehicles = np.random.randn(4, 2) * 0.8 + np.array([6, 6])
        else:  # Converged
            animals = np.random.randn(4, 2) * 0.3 + np.array([2, 2])
            vehicles = np.random.randn(4, 2) * 0.3 + np.array([7, 7])

        ax.scatter(animals[:, 0], animals[:, 1], s=150, c='#E74C3C',
                  alpha=0.6, label='Animals', edgecolor='black', linewidth=1.5)
        ax.scatter(vehicles[:, 0], vehicles[:, 1], s=150, c='#3498DB',
                  alpha=0.6, label='Vehicles', edgecolor='black', linewidth=1.5)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(-4, 10)
        ax.set_ylim(-4, 10)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)

        if ax_idx == 0:
            ax.set_ylabel('Dimension 2', fontsize=10, fontweight='bold')
        ax.set_xlabel('Dimension 1', fontsize=10, fontweight='bold')

    plt.suptitle('Training Evolution: Embeddings Learn to Cluster', fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/training_evolution_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 14: Analogy Results
def plot_analogy_results():
    """Bar chart of analogy task accuracy"""
    fig, ax = plt.subplots(figsize=(11, 6))

    categories = ['Capital:\nCountry', 'Gender', 'Comparative', 'Superlative', 'Overall']
    word2vec_acc = [94, 88, 86, 74, 72]
    glove_acc = [95, 90, 88, 76, 75]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, word2vec_acc, width, label='Word2Vec',
                   color=COLOR_MLPURPLE, alpha=0.7, edgecolor=COLOR_DARKGRAY, linewidth=1.5)
    bars2 = ax.bar(x + width/2, glove_acc, width, label='GloVe',
                   color=COLOR_PREDICT, alpha=0.7, edgecolor=COLOR_DARKGRAY, linewidth=1.5)

    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Word Analogy Task Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)

    # Add values
    for i, (w2v, glv) in enumerate(zip(word2vec_acc, glove_acc)):
        ax.text(i - width/2, w2v + 2, f'{w2v}%', ha='center', fontsize=9, fontweight='bold')
        ax.text(i + width/2, glv + 2, f'{glv}%', ha='center', fontsize=9, fontweight='bold')

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/analogy_results_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 15: Pre-trained Embeddings Comparison
def plot_pretrained_comparison():
    """Compare Word2Vec, GloVe, FastText"""
    fig, ax = plt.subplots(figsize=(11, 7))

    models = ['Word2Vec', 'GloVe', 'FastText']
    vocab_sizes = [3000000, 2200000, 2000000]  # Millions of words in pre-trained
    dimensions = [300, 300, 300]
    training_corpus = ['Google News\n100B words', 'Wikipedia\n6B words', 'Common Crawl\n600B words']

    colors = [COLOR_MLPURPLE, COLOR_PREDICT, COLOR_CONTEXT]

    for i, (model, vocab, color) in enumerate(zip(models, vocab_sizes, colors)):
        x = i * 3.5 + 1

        model_box = FancyBboxPatch((x-0.8, 5), 2.5, 1.5, boxstyle="round,pad=0.12",
                                   facecolor=color, alpha=0.5,
                                   edgecolor=COLOR_DARKGRAY, linewidth=2.5)
        ax.add_patch(model_box)
        ax.text(x+0.45, 6.1, model, ha='center', fontsize=12, fontweight='bold')
        ax.text(x+0.45, 5.6, f'{vocab/1e6:.1f}M words', ha='center', fontsize=9)
        ax.text(x+0.45, 5.3, training_corpus[i], ha='center', fontsize=8, style='italic')

        # Download arrow
        ax.annotate('', xy=(x+0.45, 4.5), xytext=(x+0.45, 5),
                   arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2.5))
        ax.text(x+0.45, 4.7, 'Download', ha='center', fontsize=8, fontweight='bold')

        # Your task box
        task_box = FancyBboxPatch((x-0.8, 3), 2.5, 1.2, boxstyle="round,pad=0.08",
                                  facecolor='#FFF3CD', alpha=0.7,
                                  edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(task_box)
        ax.text(x+0.45, 3.7, 'Your Task', ha='center', fontsize=10, fontweight='bold')
        ax.text(x+0.45, 3.3, 'Use directly!', ha='center', fontsize=9)

    ax.set_xlim(0, 12)
    ax.set_ylim(2, 7)
    ax.axis('off')
    ax.set_title('Pre-trained Embeddings: Ready to Use', fontsize=14, fontweight='bold')

    ax.text(6, 1.8, 'All three available for free - choose based on your corpus/task',
           ha='center', fontsize=10, style='italic', color=COLOR_DARKGRAY)

    plt.tight_layout()
    plt.savefig('../figures/pretrained_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# ===== APPENDIX CHARTS (16-18) =====

# Chart 16: Word2Vec Objectives Comparison
def plot_word2vec_objectives():
    """Compare Skip-gram vs CBOW objectives"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Skip-gram
    ax1.text(5, 5.5, 'Skip-gram Objective', ha='center', fontsize=12, fontweight='bold')
    ax1.text(5, 5, 'Given: Center word', ha='center', fontsize=10)
    ax1.text(5, 4.6, 'Predict: Context words', ha='center', fontsize=10)

    # Equation
    eq_box = FancyBboxPatch((2, 3), 6, 1.2, boxstyle="round,pad=0.1",
                            facecolor=COLOR_MLPURPLE, alpha=0.2,
                            edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax1.add_patch(eq_box)
    ax1.text(5, 3.6, '$\\max \\sum_t \\sum_{j \\neq 0} \\log P(w_{t+j} | w_t)$',
            ha='center', fontsize=11, fontweight='bold')

    ax1.text(5, 2, 'Better for: Rare words', ha='center', fontsize=10, fontweight='bold',
            color='green')
    ax1.text(5, 1.5, 'More common in practice', ha='center', fontsize=9, style='italic')

    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.axis('off')

    # CBOW
    ax2.text(5, 5.5, 'CBOW Objective', ha='center', fontsize=12, fontweight='bold')
    ax2.text(5, 5, 'Given: Context words', ha='center', fontsize=10)
    ax2.text(5, 4.6, 'Predict: Center word', ha='center', fontsize=10)

    # Equation
    eq_box2 = FancyBboxPatch((2, 3), 6, 1.2, boxstyle="round,pad=0.1",
                             facecolor=COLOR_PREDICT, alpha=0.2,
                             edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax2.add_patch(eq_box2)
    ax2.text(5, 3.6, '$\\max \\sum_t \\log P(w_t | context_t)$', ha='center',
            fontsize=11, fontweight='bold')

    ax2.text(5, 2, 'Better for: Frequent words', ha='center', fontsize=10,
            fontweight='bold', color='green')
    ax2.text(5, 1.5, 'Faster training', ha='center', fontsize=9, style='italic')

    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis('off')

    plt.suptitle('Word2Vec: Two Training Objectives', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../figures/word2vec_objectives_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 17: GloVe Co-occurrence Heatmap
def plot_glove_cooccurrence_heatmap():
    """Example co-occurrence matrix"""
    fig, ax = plt.subplots(figsize=(10, 9))

    words = ['cat', 'dog', 'sat', 'ran', 'the', 'on', 'quickly', 'slowly']

    # Simulated co-occurrence matrix
    np.random.seed(42)
    cooc = np.array([
        [0, 45, 23, 8, 150, 12, 3, 2],    # cat
        [45, 0, 18, 32, 140, 15, 4, 3],   # dog
        [23, 18, 0, 1, 89, 67, 5, 8],     # sat
        [8, 32, 1, 0, 92, 43, 12, 6],     # ran
        [150, 140, 89, 92, 0, 210, 78, 82], # the
        [12, 15, 67, 43, 210, 0, 8, 9],   # on
        [3, 4, 5, 12, 78, 8, 0, 15],      # quickly
        [2, 3, 8, 6, 82, 9, 15, 0]        # slowly
    ])

    im = ax.imshow(cooc, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(words)))
    ax.set_yticks(range(len(words)))
    ax.set_xticklabels(words, fontsize=11)
    ax.set_yticklabels(words, fontsize=11)

    ax.set_xlabel('Context Word', fontsize=12, fontweight='bold')
    ax.set_ylabel('Center Word', fontsize=12, fontweight='bold')
    ax.set_title('GloVe Co-occurrence Matrix Example', fontsize=13, fontweight='bold')

    # Add values
    for i in range(len(words)):
        for j in range(len(words)):
            ax.text(j, i, str(cooc[i,j]), ha='center', va='center',
                   fontsize=8, fontweight='bold',
                   color='white' if cooc[i,j] > 100 else 'black')

    plt.colorbar(im, ax=ax, label='Co-occurrence Count')
    plt.tight_layout()
    plt.savefig('../figures/glove_cooccurrence_heatmap_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 18: FastText Subword Visualization
def plot_fasttext_subword():
    """Show character n-gram decomposition"""
    fig, ax = plt.subplots(figsize=(12, 8))

    words_with_ngrams = [
        ('playing', ['<pl', 'pla', 'lay', 'ayi', 'yin', 'ing', 'ng>'], 3.5),
        ('unhappiness', ['<un', 'unh', 'nha', 'hap', 'app', 'ppi', '...', 'ss>'], 2.3),
        ('COVID', ['<CO', 'COV', 'OVI', 'VID', 'ID>'], 1.1)
    ]

    for word, ngrams, y in words_with_ngrams:
        # Word
        word_box = FancyBboxPatch((0.5, y), 2.3, 0.6, boxstyle="round,pad=0.08",
                                  facecolor=COLOR_LIGHTGRAY,
                                  edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(word_box)
        ax.text(1.65, y+0.3, word, ha='center', fontsize=11, fontweight='bold')

        # Arrow
        ax.annotate('', xy=(3.5, y+0.3), xytext=(2.8, y+0.3),
                   arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))

        # N-grams
        for i, ngram in enumerate(ngrams[:7]):  # Show first 7
            ngram_box = FancyBboxPatch((4 + i*1.05, y+0.05), 1, 0.5,
                                       boxstyle="round,pad=0.04",
                                       facecolor=COLOR_PREDICT, alpha=0.5,
                                       edgecolor=COLOR_DARKGRAY, linewidth=1)
            ax.add_patch(ngram_box)
            ax.text(4.5 + i*1.05, y+0.3, ngram, ha='center', fontsize=7,
                   family='monospace', fontweight='bold')

    ax.text(1.65, 4.5, 'Words', ha='center', fontsize=12, fontweight='bold')
    ax.text(7.5, 4.5, 'Character N-grams (3-6 chars)', ha='center', fontsize=12,
           fontweight='bold')

    ax.text(6, 0.3, 'FastText: $v_{word} = \\sum_{g \\in ngrams} v_g$', ha='center',
           fontsize=11, style='italic')

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('FastText: Subword Embeddings via Character N-grams', fontsize=14,
                 fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/fasttext_subword_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_charts():
    """Generate all 18 Week 2 BSc Discovery charts"""
    print("Generating Week 2 BSc Discovery Charts - Visual Emphasis Edition...")
    print("Target: 18 charts (15 main + 3 appendix)\n")

    print("=== MAIN PRESENTATION (Charts 1-15) ===")
    print("1/18: Word arithmetic 3D...")
    plot_word_arithmetic_3d()

    print("2/18: Similarity clustering 2D...")
    plot_similarity_clustering_2d()

    print("3/18: Dimensionality comparison...")
    plot_dimensionality_comparison()

    print("4/18: Context-dependent bank...")
    plot_context_dependent_bank()

    print("5/18: Word as vector concept...")
    plot_word_as_vector_concept()

    print("6/18: Semantic space 2D (detailed)...")
    plot_semantic_space_2d()

    print("7/18: Distributed features...")
    plot_distributed_features()

    print("8/18: Skip-gram architecture...")
    plot_skipgram_architecture()

    print("9/18: CBOW architecture...")
    plot_cbow_architecture()

    print("10/18: Skip-gram training steps...")
    plot_skipgram_training_steps()

    print("11/18: Softmax problem...")
    plot_softmax_problem()

    print("12/18: Negative sampling process...")
    plot_negative_sampling_process()

    print("13/18: Training evolution...")
    plot_training_evolution()

    print("14/18: Analogy results...")
    plot_analogy_results()

    print("15/18: Pre-trained comparison...")
    plot_pretrained_comparison()

    print("\n=== APPENDIX (Charts 16-18) ===")
    print("16/18: Word2Vec objectives comparison...")
    plot_word2vec_objectives()

    print("17/18: GloVe co-occurrence heatmap...")
    plot_glove_cooccurrence_heatmap()

    print("18/18: FastText subword visualization...")
    plot_fasttext_subword()

    print("\n" + "="*60)
    print("CHART GENERATION COMPLETE!")
    print("="*60)
    print(f"  Total charts: 18")
    print(f"  Main presentation: 15 (0.75 ratio for 20 slides)")
    print(f"  Appendix: 3")
    print(f"  Visualization types: 4 (2D/3D, architecture, training, similarity)")
    print(f"\nAll files saved to: ../figures/*_bsc.pdf")
    print("="*60)

if __name__ == "__main__":
    generate_all_charts()
