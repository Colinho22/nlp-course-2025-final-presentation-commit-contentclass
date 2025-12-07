#!/usr/bin/env python3
"""
Generate charts for extended embeddings page
Charts that add genuine value over text explanations
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# BSc Discovery colors
COLOR_MAIN = '#404040'
COLOR_ACCENT = '#3333B2'
COLOR_GRAY = '#B4B4B4'
COLOR_LIGHT = '#F0F0F0'
COLOR_GREEN = '#2CA02C'
COLOR_RED = '#D62728'
COLOR_ORANGE = '#FF7F0E'
COLOR_BLUE = '#0066CC'

# Font sizes
FONTSIZE_TITLE = 20
FONTSIZE_LABEL = 16
FONTSIZE_TICK = 14
FONTSIZE_TEXT = 14

OUTPUT_DIR = Path(__file__).parent.parent / 'assets' / 'charts' / 'embeddings'

def create_evolution_timeline():
    """Timeline of embedding methods evolution"""
    fig, ax = plt.subplots(figsize=(14, 6))

    methods = [
        (2003, 'NNLM\nBengio et al.', 'First neural LM'),
        (2013, 'Word2Vec\nMikolov et al.', 'Skip-gram, CBOW'),
        (2014, 'GloVe\nPennington et al.', 'Global vectors'),
        (2016, 'FastText\nBojanowski et al.', 'Subword'),
        (2018, 'ELMo\nPeters et al.', 'Contextual'),
        (2018.5, 'BERT\nDevlin et al.', 'Bidirectional'),
        (2020, 'GPT-3\nOpenAI', 'In-context'),
        (2022, 'Sentence-T5\nNi et al.', 'Sentence'),
    ]

    # Draw timeline
    years = [m[0] for m in methods]
    ax.axhline(y=0, color=COLOR_ACCENT, linewidth=3, zorder=1)

    for i, (year, name, desc) in enumerate(methods):
        # Alternate above/below
        y_pos = 0.6 if i % 2 == 0 else -0.6

        # Dot on timeline
        ax.scatter(year, 0, s=150, color=COLOR_ACCENT, zorder=3)

        # Connecting line
        ax.plot([year, year], [0, y_pos * 0.7], color=COLOR_GRAY, linewidth=1.5, zorder=2)

        # Text box
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=COLOR_ACCENT, linewidth=1.5)
        ax.text(year, y_pos, f'{name}\n({desc})', ha='center', va='center',
                fontsize=FONTSIZE_TEXT - 2, bbox=bbox_props)

    ax.set_xlim(2002, 2024)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('Year', fontsize=FONTSIZE_LABEL)
    ax.set_title('Evolution of Word Embeddings (2003-2022)', fontsize=FONTSIZE_TITLE, fontweight='bold')

    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.tick_params(labelsize=FONTSIZE_TICK)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'evolution_timeline.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: evolution_timeline.png")

def create_onehot_vs_dense():
    """Visual comparison of one-hot vs dense embeddings"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # One-hot encoding
    ax1 = axes[0]
    vocab = ['cat', 'dog', 'bird', 'fish', 'tree']
    onehot = np.eye(5)

    im1 = ax1.imshow(onehot, cmap='Blues', aspect='auto')
    ax1.set_xticks(range(5))
    ax1.set_xticklabels([f'd{i+1}' for i in range(5)], fontsize=FONTSIZE_TICK)
    ax1.set_yticks(range(5))
    ax1.set_yticklabels(vocab, fontsize=FONTSIZE_TICK)
    ax1.set_xlabel('Dimensions (V = 5)', fontsize=FONTSIZE_LABEL)
    ax1.set_ylabel('Words', fontsize=FONTSIZE_LABEL)
    ax1.set_title('One-Hot Encoding\n(Sparse, V dimensions)', fontsize=FONTSIZE_TITLE - 2, fontweight='bold')

    # Add values
    for i in range(5):
        for j in range(5):
            val = int(onehot[i, j])
            color = 'white' if val == 1 else 'black'
            ax1.text(j, i, str(val), ha='center', va='center', fontsize=FONTSIZE_TEXT, color=color)

    # Dense embeddings
    ax2 = axes[1]
    np.random.seed(42)
    dense = np.random.randn(5, 3) * 0.5
    dense = np.round(dense, 2)

    im2 = ax2.imshow(dense, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['d1', 'd2', 'd3'], fontsize=FONTSIZE_TICK)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(vocab, fontsize=FONTSIZE_TICK)
    ax2.set_xlabel('Dimensions (d = 3)', fontsize=FONTSIZE_LABEL)
    ax2.set_title('Dense Embeddings\n(Learned, d << V)', fontsize=FONTSIZE_TITLE - 2, fontweight='bold')

    # Add values
    for i in range(5):
        for j in range(3):
            ax2.text(j, i, f'{dense[i,j]:.2f}', ha='center', va='center', fontsize=FONTSIZE_TEXT)

    plt.colorbar(im2, ax=ax2, label='Value')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'onehot_vs_dense.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: onehot_vs_dense.png")

def create_word2vec_training():
    """Word2Vec training process visualization"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Training steps data
    steps = np.arange(0, 100001, 10000)
    loss = 5 * np.exp(-steps / 20000) + 0.5 + np.random.randn(len(steps)) * 0.1

    ax.plot(steps / 1000, loss, color=COLOR_ACCENT, linewidth=2.5, marker='o', markersize=6)
    ax.fill_between(steps / 1000, loss, alpha=0.2, color=COLOR_ACCENT)

    # Annotations
    ax.annotate('High loss:\nRandom embeddings', xy=(5, 4.5), xytext=(20, 4.5),
                fontsize=FONTSIZE_TEXT, arrowprops=dict(arrowstyle='->', color=COLOR_GRAY))
    ax.annotate('Converged:\nMeaningful vectors', xy=(90, 0.7), xytext=(60, 1.5),
                fontsize=FONTSIZE_TEXT, arrowprops=dict(arrowstyle='->', color=COLOR_GRAY))

    ax.set_xlabel('Training Steps (thousands)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Loss', fontsize=FONTSIZE_LABEL)
    ax.set_title('Word2Vec Training: Loss Over Time', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'word2vec_training.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: word2vec_training.png")

def create_analogy_demo():
    """King - Man + Woman = Queen visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # 2D positions for demo
    words = {
        'man': (1, 1),
        'woman': (1, 3),
        'king': (4, 1),
        'queen': (4, 3),
    }

    # Plot words
    for word, (x, y) in words.items():
        ax.scatter(x, y, s=300, c=COLOR_ACCENT, zorder=3)
        ax.annotate(word.upper(), (x, y), xytext=(10, 10), textcoords='offset points',
                    fontsize=FONTSIZE_LABEL, fontweight='bold')

    # Draw arrows
    # man -> woman (gender axis)
    ax.annotate('', xy=(1, 2.8), xytext=(1, 1.2),
                arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=2))
    ax.text(0.5, 2, 'gender\nvector', fontsize=FONTSIZE_TEXT, color=COLOR_GREEN, ha='center')

    # king -> queen (same gender axis)
    ax.annotate('', xy=(4, 2.8), xytext=(4, 1.2),
                arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=2))

    # man -> king (royalty axis)
    ax.annotate('', xy=(3.8, 1), xytext=(1.2, 1),
                arrowprops=dict(arrowstyle='->', color=COLOR_BLUE, lw=2))
    ax.text(2.5, 0.5, 'royalty vector', fontsize=FONTSIZE_TEXT, color=COLOR_BLUE, ha='center')

    # woman -> queen (same royalty axis)
    ax.annotate('', xy=(3.8, 3), xytext=(1.2, 3),
                arrowprops=dict(arrowstyle='->', color=COLOR_BLUE, lw=2))

    # Equation
    ax.text(2.5, 4.2, 'king - man + woman = queen', fontsize=FONTSIZE_TITLE,
            ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLOR_ACCENT, linewidth=2))

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Word Analogies: Semantic Arithmetic', fontsize=FONTSIZE_TITLE, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'analogy_demo.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: analogy_demo.png")

def create_contextual_comparison():
    """Static vs Contextual embeddings comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Static embeddings
    ax1 = axes[0]
    ax1.text(0.5, 0.8, '"bank"', fontsize=FONTSIZE_TITLE, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT))

    ax1.text(0.2, 0.4, 'river bank', fontsize=FONTSIZE_LABEL, ha='center', color=COLOR_BLUE)
    ax1.text(0.8, 0.4, 'money bank', fontsize=FONTSIZE_LABEL, ha='center', color=COLOR_GREEN)

    ax1.annotate('', xy=(0.5, 0.65), xytext=(0.2, 0.45),
                arrowprops=dict(arrowstyle='->', color=COLOR_GRAY, lw=2))
    ax1.annotate('', xy=(0.5, 0.65), xytext=(0.8, 0.45),
                arrowprops=dict(arrowstyle='->', color=COLOR_GRAY, lw=2))

    ax1.text(0.5, 0.15, 'Same vector for both!', fontsize=FONTSIZE_LABEL, ha='center',
             color=COLOR_RED, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Static Embeddings\n(Word2Vec, GloVe)', fontsize=FONTSIZE_TITLE - 2, fontweight='bold')

    # Contextual embeddings
    ax2 = axes[1]
    ax2.text(0.2, 0.8, '"bank"', fontsize=FONTSIZE_TITLE, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor=COLOR_BLUE))
    ax2.text(0.8, 0.8, '"bank"', fontsize=FONTSIZE_TITLE, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor=COLOR_GREEN))

    ax2.text(0.2, 0.5, 'Context:\n"The river bank\nwas flooded"', fontsize=FONTSIZE_TEXT, ha='center', color=COLOR_BLUE)
    ax2.text(0.8, 0.5, 'Context:\n"The bank approved\nmy loan"', fontsize=FONTSIZE_TEXT, ha='center', color=COLOR_GREEN)

    ax2.text(0.2, 0.15, 'Vector A', fontsize=FONTSIZE_LABEL, ha='center',
             color=COLOR_BLUE, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor=COLOR_BLUE))
    ax2.text(0.8, 0.15, 'Vector B', fontsize=FONTSIZE_LABEL, ha='center',
             color=COLOR_GREEN, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor=COLOR_GREEN))

    ax2.annotate('', xy=(0.2, 0.25), xytext=(0.2, 0.4),
                arrowprops=dict(arrowstyle='->', color=COLOR_BLUE, lw=2))
    ax2.annotate('', xy=(0.8, 0.25), xytext=(0.8, 0.4),
                arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=2))

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Contextual Embeddings\n(ELMo, BERT)', fontsize=FONTSIZE_TITLE - 2, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'contextual_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: contextual_comparison.png")

def create_similarity_heatmap():
    """Cosine similarity heatmap between words"""
    fig, ax = plt.subplots(figsize=(10, 8))

    words = ['cat', 'dog', 'kitten', 'puppy', 'car', 'truck', 'king', 'queen']
    n = len(words)

    # Simulated similarity matrix
    sim = np.array([
        [1.0, 0.76, 0.85, 0.52, 0.12, 0.15, 0.08, 0.11],
        [0.76, 1.0, 0.58, 0.82, 0.18, 0.21, 0.10, 0.09],
        [0.85, 0.58, 1.0, 0.61, 0.08, 0.11, 0.06, 0.09],
        [0.52, 0.82, 0.61, 1.0, 0.14, 0.17, 0.07, 0.08],
        [0.12, 0.18, 0.08, 0.14, 1.0, 0.89, 0.11, 0.13],
        [0.15, 0.21, 0.11, 0.17, 0.89, 1.0, 0.09, 0.11],
        [0.08, 0.10, 0.06, 0.07, 0.11, 0.09, 1.0, 0.78],
        [0.11, 0.09, 0.09, 0.08, 0.13, 0.11, 0.78, 1.0],
    ])

    im = ax.imshow(sim, cmap='RdYlGn', vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(words, fontsize=FONTSIZE_TICK, rotation=45, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(words, fontsize=FONTSIZE_TICK)

    # Add values
    for i in range(n):
        for j in range(n):
            color = 'white' if sim[i, j] > 0.6 else 'black'
            ax.text(j, i, f'{sim[i,j]:.2f}', ha='center', va='center',
                    fontsize=FONTSIZE_TEXT - 2, color=color)

    plt.colorbar(im, ax=ax, label='Cosine Similarity', shrink=0.8)
    ax.set_title('Word Similarity Matrix\n(Cosine Similarity)', fontsize=FONTSIZE_TITLE, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'similarity_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: similarity_heatmap.png")

def create_applications_overview():
    """Applications of word embeddings"""
    fig, ax = plt.subplots(figsize=(14, 8))

    applications = [
        ('Semantic Search', 0.2, 0.8, COLOR_BLUE, 'Find similar documents'),
        ('Sentiment Analysis', 0.5, 0.8, COLOR_GREEN, 'Classify text emotions'),
        ('Machine Translation', 0.8, 0.8, COLOR_ORANGE, 'Cross-lingual mapping'),
        ('Named Entity\nRecognition', 0.2, 0.4, COLOR_RED, 'Identify names, places'),
        ('Text Classification', 0.5, 0.4, COLOR_ACCENT, 'Categorize documents'),
        ('Question\nAnswering', 0.8, 0.4, '#9C27B0', 'Find relevant answers'),
    ]

    for app, x, y, color, desc in applications:
        circle = plt.Circle((x, y), 0.12, color=color, alpha=0.3)
        ax.add_patch(circle)
        ax.scatter(x, y, s=500, c=color, zorder=3)
        ax.text(x, y, app, ha='center', va='center', fontsize=FONTSIZE_TEXT,
                fontweight='bold', color='white')
        ax.text(x, y - 0.18, desc, ha='center', va='top', fontsize=FONTSIZE_TEXT - 2,
                color=COLOR_MAIN)

    # Central node
    ax.scatter(0.5, 0.6, s=1500, c=COLOR_ACCENT, zorder=4, marker='s')
    ax.text(0.5, 0.6, 'Word\nEmbeddings', ha='center', va='center',
            fontsize=FONTSIZE_LABEL, fontweight='bold', color='white')

    # Connect to center
    for _, x, y, _, _ in applications:
        ax.plot([0.5, x], [0.6, y], color=COLOR_GRAY, linewidth=1.5, alpha=0.5, zorder=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Applications of Word Embeddings', fontsize=FONTSIZE_TITLE, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'applications_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: applications_overview.png")

if __name__ == '__main__':
    print("=" * 60)
    print("Generating Embeddings Charts")
    print("=" * 60)

    create_evolution_timeline()
    create_onehot_vs_dense()
    create_word2vec_training()
    create_analogy_demo()
    create_contextual_comparison()
    create_similarity_heatmap()
    create_applications_overview()

    print("=" * 60)
    print("Done! Generated 7 charts")
    print("=" * 60)
