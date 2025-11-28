"""
Generate 3D visualization charts for the compact word embeddings discovery handout
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# Educational color scheme
COLORS = {
    'animals': '#4ECDC4',  # Teal
    'vehicles': '#FF6B6B',  # Red
    'emotions': '#95E77E',  # Green
    'royalty': '#9B59B6',   # Purple
    'finance': '#F39C12',   # Orange
    'nature': '#3498DB',    # Blue
}

def generate_3d_embedding_space():
    """Generate static 3D embedding space visualization"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define word positions in 3D space
    words = {
        # Animals cluster
        'cat': [2, 3, 2.5],
        'dog': [2.5, 3.2, 2.7],
        'kitten': [1.8, 2.8, 2.3],
        'puppy': [2.7, 3.3, 2.6],
        'pet': [2.3, 3.1, 2.8],

        # Vehicles cluster
        'car': [8, 1, 1.5],
        'truck': [8.5, 1.2, 1.7],
        'bus': [8.3, 0.9, 1.4],
        'vehicle': [8.2, 1.1, 1.6],

        # Emotions cluster
        'happy': [5, 7, 5],
        'joyful': [5.3, 7.2, 5.2],
        'sad': [5, 6.5, 4.5],
        'angry': [4.7, 6.8, 4.8],
    }

    # Plot clusters with different colors
    for word, pos in words.items():
        if word in ['cat', 'dog', 'kitten', 'puppy', 'pet']:
            color = COLORS['animals']
            marker = 'o'
        elif word in ['car', 'truck', 'bus', 'vehicle']:
            color = COLORS['vehicles']
            marker = 's'
        else:
            color = COLORS['emotions']
            marker = '^'

        ax.scatter(pos[0], pos[1], pos[2], c=color, marker=marker, s=100, alpha=0.8)
        ax.text(pos[0], pos[1], pos[2], word, fontsize=8)

    # Draw cluster boundaries (simplified as lines connecting cluster points)
    animals = ['cat', 'dog', 'kitten', 'puppy', 'pet']
    for i in range(len(animals)-1):
        p1 = words[animals[i]]
        p2 = words[animals[i+1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                'b--', alpha=0.3, linewidth=1)

    ax.set_xlabel('Dimension 1', fontsize=10)
    ax.set_ylabel('Dimension 2', fontsize=10)
    ax.set_zlabel('Dimension 3', fontsize=10)
    ax.set_title('3D Word Embedding Space - Semantic Clusters', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/3d_embedding_space.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_3d_word_arithmetic():
    """Generate 3D visualization of word arithmetic"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define vectors
    king = np.array([5, 3, 4])
    man = np.array([4, 2, 2])
    woman = np.array([4, 1, 2])
    queen = king - man + woman

    # Plot points
    ax.scatter(*king, s=300, c=COLORS['royalty'], marker='*',
              edgecolors='black', linewidths=2, label='king')
    ax.scatter(*queen, s=300, c=COLORS['royalty'], marker='*',
              edgecolors='black', linewidths=2, label='queen')
    ax.scatter(*man, s=200, c='lightblue', marker='o',
              edgecolors='black', linewidths=2, label='man')
    ax.scatter(*woman, s=200, c='lightpink', marker='o',
              edgecolors='black', linewidths=2, label='woman')

    # Draw vectors
    # king - man
    ax.quiver(king[0], king[1], king[2],
             man[0]-king[0], man[1]-king[1], man[2]-king[2],
             color='red', alpha=0.6, arrow_length_ratio=0.1)

    # + woman
    ax.quiver(man[0], man[1], man[2],
             woman[0]-man[0], woman[1]-man[1], woman[2]-man[2],
             color='blue', alpha=0.6, arrow_length_ratio=0.1)

    # = queen
    ax.quiver(woman[0], woman[1], woman[2],
             queen[0]-woman[0], queen[1]-woman[1], queen[2]-woman[2],
             color='green', alpha=0.8, arrow_length_ratio=0.1, linewidth=2)

    # Labels
    ax.text(king[0], king[1], king[2]+0.3, 'king', fontsize=10, fontweight='bold')
    ax.text(queen[0], queen[1], queen[2]+0.3, 'queen', fontsize=10, fontweight='bold')
    ax.text(man[0], man[1], man[2]+0.3, 'man', fontsize=10, fontweight='bold')
    ax.text(woman[0], woman[1], woman[2]+0.3, 'woman', fontsize=10, fontweight='bold')

    # Draw relationship planes
    # Gender plane
    gender_points = np.array([king, queen, man, woman])
    ax.plot([king[0], queen[0]], [king[1], queen[1]], [king[2], queen[2]],
            'g--', alpha=0.5, linewidth=2)
    ax.plot([man[0], woman[0]], [man[1], woman[1]], [man[2], woman[2]],
            'g--', alpha=0.5, linewidth=2)

    ax.set_xlabel('Dimension 1 (Power)', fontsize=10)
    ax.set_ylabel('Dimension 2 (Gender)', fontsize=10)
    ax.set_zlabel('Dimension 3 (Status)', fontsize=10)
    ax.set_title('3D Word Arithmetic: king - man + woman = queen', fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig('../figures/3d_word_arithmetic.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_3d_context_embeddings():
    """Generate 3D visualization of contextual embeddings"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Financial context cluster
    money = [8, 8, 3]
    account = [8.5, 7.5, 3.2]
    deposit = [7.8, 8.2, 2.9]
    bank_financial = [8.2, 7.9, 3.1]

    # Nature context cluster
    river = [2, 2, 1]
    water = [1.8, 2.3, 0.9]
    flow = [2.2, 1.9, 1.1]
    bank_river = [2.0, 2.1, 1.0]

    # Plot financial cluster
    for word, pos, label in [(money, 'money'), (account, 'account'),
                             (deposit, 'deposit'), (bank_financial, 'bank (fin)')]:
        ax.scatter(*word, s=150, c=COLORS['finance'], alpha=0.8)
        ax.text(word[0], word[1], word[2], label, fontsize=8)

    # Plot nature cluster
    for word, pos, label in [(river, 'river'), (water, 'water'),
                             (flow, 'flow'), (bank_river, 'bank (river)')]:
        ax.scatter(*word, s=150, c=COLORS['nature'], alpha=0.8)
        ax.text(word[0], word[1], word[2], label, fontsize=8)

    # Highlight the two "bank" positions
    ax.scatter(*bank_financial, s=400, c=COLORS['finance'], marker='*',
              edgecolors='black', linewidths=2)
    ax.scatter(*bank_river, s=400, c=COLORS['nature'], marker='*',
              edgecolors='black', linewidths=2)

    # Draw movement arrow
    ax.plot([bank_financial[0], bank_river[0]],
           [bank_financial[1], bank_river[1]],
           [bank_financial[2], bank_river[2]],
           'r--', linewidth=2, alpha=0.7)

    # Add midpoint label
    mid_x = (bank_financial[0] + bank_river[0]) / 2
    mid_y = (bank_financial[1] + bank_river[1]) / 2
    mid_z = (bank_financial[2] + bank_river[2]) / 2
    ax.text(mid_x, mid_y, mid_z+0.5, 'Context\nShift', fontsize=10,
           fontweight='bold', color='red', ha='center')

    ax.set_xlabel('Dimension 1', fontsize=10)
    ax.set_ylabel('Dimension 2', fontsize=10)
    ax.set_zlabel('Dimension 3', fontsize=10)
    ax.set_title('Contextual Embeddings: Same Word, Different Meanings',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/3d_contextual_embeddings.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_interactive_3d_embeddings():
    """Create interactive 3D plot using plotly"""
    # Prepare data
    words_data = []

    # Animals cluster
    animals = {
        'cat': [2, 3, 2.5],
        'dog': [2.5, 3.2, 2.7],
        'kitten': [1.8, 2.8, 2.3],
        'puppy': [2.7, 3.3, 2.6],
        'pet': [2.3, 3.1, 2.8],
    }

    # Vehicles cluster
    vehicles = {
        'car': [8, 1, 1.5],
        'truck': [8.5, 1.2, 1.7],
        'bus': [8.3, 0.9, 1.4],
        'vehicle': [8.2, 1.1, 1.6],
    }

    # Emotions cluster
    emotions = {
        'happy': [5, 7, 5],
        'joyful': [5.3, 7.2, 5.2],
        'sad': [5, 6.5, 4.5],
        'angry': [4.7, 6.8, 4.8],
    }

    # Combine all data
    for word, pos in animals.items():
        words_data.append({'word': word, 'x': pos[0], 'y': pos[1], 'z': pos[2],
                          'category': 'Animals', 'color': COLORS['animals']})

    for word, pos in vehicles.items():
        words_data.append({'word': word, 'x': pos[0], 'y': pos[1], 'z': pos[2],
                          'category': 'Vehicles', 'color': COLORS['vehicles']})

    for word, pos in emotions.items():
        words_data.append({'word': word, 'x': pos[0], 'y': pos[1], 'z': pos[2],
                          'category': 'Emotions', 'color': COLORS['emotions']})

    df = pd.DataFrame(words_data)

    # Create 3D scatter plot
    fig = go.Figure()

    # Add traces for each category
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        fig.add_trace(go.Scatter3d(
            x=cat_data['x'],
            y=cat_data['y'],
            z=cat_data['z'],
            mode='markers+text',
            name=category,
            text=cat_data['word'],
            textposition="top center",
            marker=dict(
                size=12,
                color=cat_data['color'].iloc[0],
                opacity=0.8,
                line=dict(color='black', width=1)
            )
        ))

    fig.update_layout(
        title='Interactive 3D Word Embeddings - Rotate to Explore!',
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=700,
        showlegend=True
    )

    # Save as HTML
    fig.write_html('../figures/interactive_3d_embeddings.html')
    return fig

def generate_dimensionality_reduction_viz():
    """Show how 100D embeddings are reduced to 3D for visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # High-dimensional representation (show as heatmap)
    np.random.seed(42)
    high_dim = np.random.randn(5, 100)  # 5 words, 100 dimensions
    words = ['cat', 'dog', 'car', 'happy', 'bank']

    # Normalize for visualization
    high_dim_norm = (high_dim - high_dim.min()) / (high_dim.max() - high_dim.min())

    # Plot heatmap
    im = ax1.imshow(high_dim_norm, cmap='coolwarm', aspect='auto')
    ax1.set_yticks(range(5))
    ax1.set_yticklabels(words)
    ax1.set_xlabel('100 Dimensions')
    ax1.set_title('Original: 100-Dimensional Embeddings', fontsize=12, fontweight='bold')
    ax1.set_xticks([0, 25, 50, 75, 99])
    ax1.set_xticklabels(['1', '25', '50', '75', '100'])

    # 3D reduction (PCA-like)
    ax2 = fig.add_subplot(122, projection='3d')

    # Simulate reduced positions
    reduced = {
        'cat': [2, 3, 2.5],
        'dog': [2.5, 3.2, 2.7],
        'car': [8, 1, 1.5],
        'happy': [5, 7, 5],
        'bank': [5, 5, 3],  # Ambiguous position
    }

    colors = ['teal', 'teal', 'red', 'green', 'gray']
    for (word, pos), color in zip(reduced.items(), colors):
        ax2.scatter(*pos, s=200, c=color, alpha=0.7, edgecolors='black', linewidths=2)
        ax2.text(pos[0], pos[1], pos[2]+0.2, word, fontsize=10, fontweight='bold')

    ax2.set_xlabel('PC 1')
    ax2.set_ylabel('PC 2')
    ax2.set_zlabel('PC 3')
    ax2.set_title('After Reduction: 3D Visualization', fontsize=12, fontweight='bold')

    plt.suptitle('Dimensionality Reduction: 100D → 3D', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/dimensionality_reduction.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all 3D visualizations"""
    print("Generating 3D visualizations...")

    print("  - 3D embedding space...")
    generate_3d_embedding_space()

    print("  - 3D word arithmetic...")
    generate_3d_word_arithmetic()

    print("  - 3D contextual embeddings...")
    generate_3d_context_embeddings()

    print("  - Interactive 3D plot...")
    create_interactive_3d_embeddings()

    print("  - Dimensionality reduction visualization...")
    generate_dimensionality_reduction_viz()

    print("All 3D figures generated successfully in ../figures/")
    print("Interactive plot saved as: ../figures/interactive_3d_embeddings.html")

if __name__ == "__main__":
    main()