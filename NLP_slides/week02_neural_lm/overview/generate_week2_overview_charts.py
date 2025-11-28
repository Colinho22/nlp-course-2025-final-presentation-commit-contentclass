"""
Generate Week 2 overview visualizations for Neural LM and Word Embeddings
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Educational colors
COLOR_CURRENT = '#FF6B6B'  # Red
COLOR_CONTEXT = '#4ECDC4'  # Teal  
COLOR_PREDICT = '#95E77E'  # Green
COLOR_NEUTRAL = '#E0E0E0'  # Gray

def create_neural_evolution():
    """Create neural language model evolution visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: From counts to neural
    ax1.text(0.5, 0.95, 'From Counts to Continuous', fontsize=12,
            fontweight='bold', ha='center')
    
    # N-gram side
    y_ngram = 0.7
    rect_ngram = FancyBboxPatch((0.05, y_ngram - 0.1), 0.35, 0.2,
                                boxstyle="round,pad=0.01",
                                facecolor=COLOR_NEUTRAL, alpha=0.7,
                                edgecolor='white', linewidth=2)
    ax1.add_patch(rect_ngram)
    ax1.text(0.225, y_ngram, 'N-gram Model', fontsize=10,
            ha='center', va='center', fontweight='bold')
    
    # Issues
    issues = ['Sparse', 'Discrete', 'No generalization']
    for i, issue in enumerate(issues):
        ax1.text(0.225, y_ngram - 0.15 - i*0.05, f'• {issue}', 
                fontsize=8, ha='center', color='darkred')
    
    # Neural side
    y_neural = 0.7
    rect_neural = FancyBboxPatch((0.55, y_neural - 0.1), 0.35, 0.2,
                                 boxstyle="round,pad=0.01",
                                 facecolor=COLOR_PREDICT, alpha=0.7,
                                 edgecolor='white', linewidth=2)
    ax1.add_patch(rect_neural)
    ax1.text(0.725, y_neural, 'Neural LM', fontsize=10,
            ha='center', va='center', fontweight='bold', color='white')
    
    # Benefits
    benefits = ['Dense', 'Continuous', 'Generalizes']
    for i, benefit in enumerate(benefits):
        ax1.text(0.725, y_neural - 0.15 - i*0.05, f'• {benefit}',
                fontsize=8, ha='center', color='darkgreen')
    
    # Arrow transformation
    ax1.annotate('', xy=(0.52, y_ngram), xytext=(0.43, y_ngram),
                arrowprops=dict(arrowstyle='->', lw=3, 
                              color=COLOR_CURRENT))
    ax1.text(0.475, y_ngram + 0.05, 'Transform', fontsize=9,
            ha='center', fontweight='bold', color=COLOR_CURRENT)
    
    # Word representations
    ax1.text(0.225, 0.25, 'Words as indices\n[1, 0, 0, ...]', 
            fontsize=9, ha='center', 
            bbox=dict(boxstyle='round', facecolor='lightgray'))
    ax1.text(0.725, 0.25, 'Words as vectors\n[0.2, -0.5, 0.8, ...]',
            fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Right: Neural network architecture
    ax2.text(0.5, 0.95, 'Neural LM Architecture', fontsize=12,
            fontweight='bold', ha='center')
    
    # Input layer
    x_input = 0.15
    words_in = ['the', 'cat', 'sat']
    for i, word in enumerate(words_in):
        y = 0.7 - i * 0.2
        circle = Circle((x_input, y), 0.04, facecolor=COLOR_CONTEXT,
                       edgecolor='white', linewidth=2)
        ax2.add_patch(circle)
        ax2.text(x_input - 0.08, y, word, fontsize=9, ha='right')
    
    # Embedding layer
    x_embed = 0.35
    ax2.text(x_embed, 0.85, 'Embed', fontsize=9, ha='center',
            fontweight='bold', color=COLOR_CONTEXT)
    for i in range(4):
        y = 0.75 - i * 0.15
        circle = Circle((x_embed, y), 0.03, facecolor=COLOR_CONTEXT,
                       alpha=0.5, edgecolor='white', linewidth=1)
        ax2.add_patch(circle)
        # Connect from input
        for j in range(3):
            y_in = 0.7 - j * 0.2
            ax2.plot([x_input + 0.04, x_embed - 0.03], [y_in, y],
                    'k-', alpha=0.1, linewidth=0.5)
    
    # Hidden layer
    x_hidden = 0.55
    ax2.text(x_hidden, 0.85, 'Hidden', fontsize=9, ha='center',
            fontweight='bold', color=COLOR_CURRENT)
    for i in range(5):
        y = 0.75 - i * 0.12
        circle = Circle((x_hidden, y), 0.03, facecolor=COLOR_CURRENT,
                       alpha=0.7, edgecolor='white', linewidth=1)
        ax2.add_patch(circle)
        # Connect from embedding
        for j in range(4):
            y_emb = 0.75 - j * 0.15
            ax2.plot([x_embed + 0.03, x_hidden - 0.03], [y_emb, y],
                    'k-', alpha=0.1, linewidth=0.5)
    
    # Output layer
    x_output = 0.75
    ax2.text(x_output, 0.85, 'Softmax', fontsize=9, ha='center',
            fontweight='bold', color=COLOR_PREDICT)
    vocab = ['on', 'under', 'near', '...']
    for i, word in enumerate(vocab):
        y = 0.7 - i * 0.15
        circle = Circle((x_output, y), 0.03, facecolor=COLOR_PREDICT,
                       alpha=0.7, edgecolor='white', linewidth=1)
        ax2.add_patch(circle)
        ax2.text(x_output + 0.08, y, word, fontsize=8, ha='left')
        # Connect from hidden
        for j in range(5):
            y_hid = 0.75 - j * 0.12
            ax2.plot([x_hidden + 0.03, x_output - 0.03], [y_hid, y],
                    'k-', alpha=0.1, linewidth=0.5)
    
    # Prediction highlight
    ax2.add_patch(Rectangle((x_output + 0.05, 0.68), 0.1, 0.05,
                           facecolor='yellow', alpha=0.5))
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/week2_neural_evolution.pdf', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()

def create_embedding_space():
    """Create word embedding visualization"""
    fig = plt.figure(figsize=(12, 6))
    
    # 2D embedding space
    ax1 = plt.subplot(121)
    
    # Word vectors
    words = {
        'king': (0.5, 0.7),
        'queen': (0.45, 0.65),
        'man': (0.55, 0.5),
        'woman': (0.5, 0.45),
        'prince': (0.6, 0.6),
        'princess': (0.55, 0.55),
        'cat': (-0.3, 0.2),
        'dog': (-0.25, 0.25),
        'pet': (-0.27, 0.3),
        'animal': (-0.2, 0.35),
        'paris': (0.2, -0.3),
        'france': (0.25, -0.25),
        'berlin': (0.15, -0.35),
        'germany': (0.2, -0.3)
    }
    
    # Plot words
    for word, (x, y) in words.items():
        # Determine color based on category
        if word in ['king', 'queen', 'man', 'woman', 'prince', 'princess']:
            color = COLOR_CURRENT
            category = 'royalty/gender'
        elif word in ['cat', 'dog', 'pet', 'animal']:
            color = COLOR_CONTEXT
            category = 'animals'
        else:
            color = COLOR_PREDICT
            category = 'locations'
        
        ax1.scatter(x, y, s=100, color=color, alpha=0.7,
                   edgecolors='white', linewidth=2)
        ax1.text(x, y - 0.03, word, fontsize=9, ha='center')
    
    # Show relationships
    # King - Man + Woman = Queen
    ax1.annotate('', xy=(0.45, 0.65), xytext=(0.5, 0.7),
                arrowprops=dict(arrowstyle='->', lw=1.5,
                              color='purple', alpha=0.5))
    ax1.text(0.52, 0.75, 'king - man\n+ woman', fontsize=8,
            ha='center', color='purple',
            bbox=dict(boxstyle='round', facecolor='lavender'))
    
    ax1.set_xlim(-0.5, 0.8)
    ax1.set_ylim(-0.5, 0.9)
    ax1.set_xlabel('Dimension 1', fontsize=10)
    ax1.set_ylabel('Dimension 2', fontsize=10)
    ax1.set_title('Word Embedding Space (2D projection)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 3D semantic dimensions
    ax2 = plt.subplot(122, projection='3d')
    
    # Generate sample embeddings
    np.random.seed(42)
    n_words = 30
    
    # Different semantic clusters
    # Cluster 1: Animals
    cluster1_x = np.random.normal(-0.3, 0.1, 10)
    cluster1_y = np.random.normal(0.2, 0.1, 10)
    cluster1_z = np.random.normal(0.0, 0.1, 10)
    
    # Cluster 2: People
    cluster2_x = np.random.normal(0.5, 0.1, 10)
    cluster2_y = np.random.normal(0.5, 0.1, 10)
    cluster2_z = np.random.normal(0.3, 0.1, 10)
    
    # Cluster 3: Places
    cluster3_x = np.random.normal(0.2, 0.1, 10)
    cluster3_y = np.random.normal(-0.3, 0.1, 10)
    cluster3_z = np.random.normal(-0.2, 0.1, 10)
    
    ax2.scatter(cluster1_x, cluster1_y, cluster1_z, c=COLOR_CONTEXT,
               s=50, alpha=0.6, label='Animals')
    ax2.scatter(cluster2_x, cluster2_y, cluster2_z, c=COLOR_CURRENT,
               s=50, alpha=0.6, label='People')
    ax2.scatter(cluster3_x, cluster3_y, cluster3_z, c=COLOR_PREDICT,
               s=50, alpha=0.6, label='Places')
    
    ax2.set_xlabel('Semantic Dim 1', fontsize=9)
    ax2.set_ylabel('Semantic Dim 2', fontsize=9)
    ax2.set_zlabel('Semantic Dim 3', fontsize=9)
    ax2.set_title('High-Dimensional Semantic Space', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('../figures/week2_embedding_space.pdf', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()

def create_applications_impact():
    """Create applications and impact visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: Word2Vec algorithms
    ax1.text(0.5, 0.95, 'Word2Vec Algorithms', fontsize=12,
            fontweight='bold', ha='center')
    
    # CBOW
    cbow_y = 0.7
    ax1.text(0.25, cbow_y + 0.08, 'CBOW', fontsize=11,
            fontweight='bold', ha='center', color=COLOR_CONTEXT)
    
    # Context words
    context_words = ['The', 'cat', 'on', 'mat']
    for i, word in enumerate(context_words):
        if i == 1:  # Skip middle word
            ax1.text(0.25, cbow_y - 0.1 - i*0.08, '?', fontsize=9,
                    ha='center', color='red', fontweight='bold')
        else:
            ax1.text(0.25, cbow_y - 0.1 - i*0.08, word, fontsize=9,
                    ha='center')
    
    ax1.annotate('Predict center\nfrom context', 
                xy=(0.25, cbow_y - 0.4),
                xytext=(0.25, cbow_y - 0.5),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor=COLOR_CONTEXT, alpha=0.3))
    
    # Skip-gram
    sg_y = 0.7
    ax1.text(0.75, sg_y + 0.08, 'Skip-gram', fontsize=11,
            fontweight='bold', ha='center', color=COLOR_CURRENT)
    
    # Center word
    ax1.text(0.75, sg_y - 0.1, 'sat', fontsize=9,
            ha='center', fontweight='bold')
    
    # Predicted context
    predicted = ['?', '?', '?', '?']
    for i, word in enumerate(predicted):
        y_pos = sg_y - 0.2 - i*0.06
        ax1.text(0.75, y_pos, word, fontsize=9,
                ha='center', color='red', fontweight='bold')
    
    ax1.annotate('Predict context\nfrom center',
                xy=(0.75, sg_y - 0.4),
                xytext=(0.75, sg_y - 0.5),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor=COLOR_CURRENT, alpha=0.3))
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Right: Performance comparison
    methods = ['One-hot', 'TF-IDF', 'Word2Vec', 'GloVe', 'FastText']
    similarity_scores = [20, 35, 85, 82, 88]
    analogy_scores = [10, 15, 75, 73, 80]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, similarity_scores, width,
                   label='Similarity Task', color=COLOR_CONTEXT, alpha=0.7)
    bars2 = ax2.bar(x + width/2, analogy_scores, width,
                   label='Analogy Task', color=COLOR_PREDICT, alpha=0.7)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}%', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('Method', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
    ax2.set_title('Representation Quality', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 100)
    
    # Add breakthrough annotation
    ax2.annotate('Neural\nBreakthrough!',
                xy=(2, 85), xytext=(3.5, 60),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('../figures/week2_applications_impact.pdf', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    create_neural_evolution()
    create_embedding_space()
    create_applications_impact()
    print("Week 2 overview visualizations generated successfully!")