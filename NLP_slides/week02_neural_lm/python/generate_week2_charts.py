import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# Generate word embeddings visualization
def plot_word_embeddings_space():
    # Simulate word embeddings (in real case, these would be trained)
    np.random.seed(42)
    
    # Categories of words
    animals = ['cat', 'dog', 'horse', 'cow', 'pig']
    countries = ['france', 'germany', 'spain', 'italy', 'poland']
    verbs = ['run', 'walk', 'jump', 'swim', 'fly']
    foods = ['apple', 'bread', 'cheese', 'meat', 'rice']
    
    # Generate embeddings with category clustering
    embeddings = []
    labels = []
    colors = []
    
    # Animals - cluster around (0, 0)
    for word in animals:
        embed = np.random.randn(50) * 0.3 + np.array([1.0] + [0.0]*49)
        embeddings.append(embed)
        labels.append(word)
        colors.append('red')
    
    # Countries - cluster around (5, 5)
    for word in countries:
        embed = np.random.randn(50) * 0.3 + np.array([0.0, 1.0] + [0.0]*48)
        embeddings.append(embed)
        labels.append(word)
        colors.append('blue')
    
    # Verbs - cluster around (-5, 5)
    for word in verbs:
        embed = np.random.randn(50) * 0.3 + np.array([-1.0, 0.0] + [0.0]*48)
        embeddings.append(embed)
        labels.append(word)
        colors.append('green')
    
    # Foods - cluster around (5, -5)
    for word in foods:
        embed = np.random.randn(50) * 0.3 + np.array([0.0, -1.0] + [0.0]*48)
        embeddings.append(embed)
        labels.append(word)
        colors.append('orange')
    
    # Add some relationship examples
    # King - Man + Woman = Queen (simplified)
    king_embed = np.random.randn(50) * 0.3 + np.array([0.5, 0.5] + [0.0]*48)
    man_embed = np.random.randn(50) * 0.3 + np.array([0.4, 0.3] + [0.0]*48)
    woman_embed = np.random.randn(50) * 0.3 + np.array([0.6, 0.3] + [0.0]*48)
    queen_embed = king_embed - man_embed + woman_embed + np.random.randn(50) * 0.1
    
    embeddings.extend([king_embed, man_embed, woman_embed, queen_embed])
    labels.extend(['king', 'man', 'woman', 'queen'])
    colors.extend(['purple', 'purple', 'purple', 'purple'])
    
    # Convert to numpy array
    embeddings = np.array(embeddings)
    
    # Use t-SNE for 2D visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot points
    unique_colors = list(set(colors))
    color_labels = ['Animals', 'Countries', 'Verbs', 'Foods', 'Royalty']
    
    for i, (color, label) in enumerate(zip(unique_colors, color_labels)):
        indices = [j for j, c in enumerate(colors) if c == color]
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], 
                   c=color, label=label, s=200, alpha=0.7, edgecolors='black')
    
    # Add labels
    for i, label in enumerate(labels):
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Add arrow for king-man+woman=queen
    king_idx = labels.index('king')
    man_idx = labels.index('man')
    woman_idx = labels.index('woman')
    queen_idx = labels.index('queen')
    
    # Draw vectors
    plt.arrow(embeddings_2d[man_idx, 0], embeddings_2d[man_idx, 1],
             embeddings_2d[king_idx, 0] - embeddings_2d[man_idx, 0],
             embeddings_2d[king_idx, 1] - embeddings_2d[man_idx, 1],
             head_width=0.5, head_length=0.3, fc='gray', ec='gray', alpha=0.5)
    
    plt.arrow(embeddings_2d[woman_idx, 0], embeddings_2d[woman_idx, 1],
             embeddings_2d[queen_idx, 0] - embeddings_2d[woman_idx, 0],
             embeddings_2d[queen_idx, 1] - embeddings_2d[woman_idx, 1],
             head_width=0.5, head_length=0.3, fc='gray', ec='gray', alpha=0.5)
    
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.title('Word Embeddings Visualization (t-SNE projection)', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../figures/word_embeddings_space.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate skip-gram training visualization
def plot_skipgram_training():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Skip-gram architecture
    ax1.text(0.5, 0.9, 'Skip-gram Model', ha='center', fontsize=14, fontweight='bold')
    
    # Center word
    ax1.add_patch(plt.Rectangle((0.4, 0.6), 0.2, 0.1, fill=True, color='lightblue', ec='black'))
    ax1.text(0.5, 0.65, 'cat', ha='center', va='center', fontsize=12)
    
    # Arrows to context words
    context_words = ['the', 'black', 'sat', 'on']
    positions = [(0.1, 0.3), (0.3, 0.3), (0.7, 0.3), (0.9, 0.3)]
    
    for word, pos in zip(context_words, positions):
        ax1.add_patch(plt.Rectangle((pos[0]-0.05, pos[1]-0.05), 0.1, 0.1, 
                                   fill=True, color='lightgreen', ec='black'))
        ax1.text(pos[0], pos[1], word, ha='center', va='center', fontsize=10)
        ax1.arrow(0.5, 0.6, pos[0]-0.5, pos[1]-0.6, 
                 head_width=0.02, head_length=0.02, fc='gray', ec='gray')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # CBOW architecture
    ax2.text(0.5, 0.9, 'CBOW Model', ha='center', fontsize=14, fontweight='bold')
    
    # Context words
    for i, (word, pos) in enumerate(zip(context_words, positions)):
        y_pos = 0.7 - i*0.1
        ax2.add_patch(plt.Rectangle((0.1, y_pos-0.05), 0.1, 0.08, 
                                   fill=True, color='lightgreen', ec='black'))
        ax2.text(0.15, y_pos, word, ha='center', va='center', fontsize=10)
        ax2.arrow(0.2, y_pos, 0.2, 0, head_width=0.02, head_length=0.02, fc='gray', ec='gray')
    
    # Hidden layer
    ax2.add_patch(plt.Rectangle((0.4, 0.35), 0.2, 0.3, fill=True, color='lightyellow', ec='black'))
    ax2.text(0.5, 0.5, 'Average', ha='center', va='center', fontsize=11)
    
    # Target word
    ax2.add_patch(plt.Rectangle((0.7, 0.45), 0.15, 0.1, fill=True, color='lightblue', ec='black'))
    ax2.text(0.775, 0.5, 'cat', ha='center', va='center', fontsize=12)
    ax2.arrow(0.6, 0.5, 0.1, 0, head_width=0.02, head_length=0.02, fc='gray', ec='gray')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.suptitle('Word2Vec Training Approaches', fontsize=16)
    plt.tight_layout()
    plt.savefig('../figures/skipgram_cbow.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate embedding similarity heatmap
def plot_embedding_similarity():
    # Sample words
    words = ['king', 'queen', 'man', 'woman', 'prince', 'princess',
             'cat', 'dog', 'kitten', 'puppy',
             'good', 'bad', 'happy', 'sad']
    
    # Create similarity matrix (simulated)
    n_words = len(words)
    similarity = np.random.rand(n_words, n_words) * 0.3
    
    # Make it symmetric
    similarity = (similarity + similarity.T) / 2
    
    # Add stronger similarities within categories
    # Royalty
    for i in range(6):
        for j in range(6):
            similarity[i, j] += 0.5
    
    # Animals
    for i in range(6, 10):
        for j in range(6, 10):
            similarity[i, j] += 0.5
    
    # Emotions
    for i in range(10, 14):
        for j in range(10, 14):
            similarity[i, j] += 0.4
    
    # Diagonal = 1
    np.fill_diagonal(similarity, 1.0)
    
    # Clip values
    similarity = np.clip(similarity, 0, 1)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity, xticklabels=words, yticklabels=words,
                cmap='Blues', annot=False, cbar_kws={'label': 'Cosine Similarity'})
    plt.title('Word Embedding Similarity Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig('../figures/embedding_similarity.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating Week 2 visualizations...")
    plot_word_embeddings_space()
    print("- Word embeddings space visualization created")
    plot_skipgram_training()
    print("- Skip-gram/CBOW architecture created")
    plot_embedding_similarity()
    print("- Embedding similarity heatmap created")
    print("\nWeek 2 visualizations generated successfully!")