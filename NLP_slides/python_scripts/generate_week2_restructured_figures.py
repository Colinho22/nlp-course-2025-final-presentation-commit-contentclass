"""
Generate figures for restructured Week 2: Neural Language Models presentation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define colors for consistency
COLOR_CURRENT = '#FF6B6B'  # Red for current position
COLOR_CONTEXT = '#4ECDC4'  # Teal for context
COLOR_PREDICT = '#95E77E'  # Green for prediction
COLOR_NEUTRAL = '#E0E0E0'  # Gray for neutral elements

def create_embedding_dimensions_performance():
    """Create chart showing performance vs embedding dimensions"""
    plt.figure(figsize=(10, 6))
    
    # Data
    dimensions = [10, 25, 50, 100, 200, 300, 500, 1000, 2000]
    word_similarity = [0.42, 0.58, 0.65, 0.71, 0.73, 0.74, 0.74, 0.73, 0.71]
    analogy_accuracy = [0.31, 0.48, 0.59, 0.68, 0.72, 0.74, 0.74, 0.73, 0.72]
    training_time = [0.5, 0.8, 1.2, 2.1, 4.2, 6.5, 10.8, 21.5, 45.2]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Performance metrics
    ax1.plot(dimensions, word_similarity, 'o-', label='Word Similarity', linewidth=2, markersize=8)
    ax1.plot(dimensions, analogy_accuracy, 's-', label='Analogy Accuracy', linewidth=2, markersize=8)
    ax1.set_xlabel('Embedding Dimensions', fontsize=12)
    ax1.set_ylabel('Performance Score', fontsize=12)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Highlight sweet spot
    ax1.axvspan(100, 300, alpha=0.2, color='green', label='Sweet Spot')
    
    # Training time on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(dimensions, training_time, '^-', color='red', label='Training Time (hours)', 
             linewidth=2, markersize=8, alpha=0.7)
    ax2.set_ylabel('Training Time (hours)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title('Embedding Dimensions vs Performance Trade-off', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/embedding_dimensions_performance.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_cbow_architecture():
    """Create CBOW architecture diagram"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Hide axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Input words
    input_words = ['The', 'cat', 'on', 'the', 'mat']
    input_y = [8, 6.5, 5, 3.5, 2]
    
    for i, word in enumerate(input_words):
        if word != 'on':  # Skip center word
            ax.add_patch(plt.Rectangle((0.5, input_y[i]-0.3), 1.5, 0.6, 
                                      fill=True, facecolor=COLOR_CONTEXT, edgecolor='black'))
            ax.text(1.25, input_y[i], word, ha='center', va='center', fontsize=11)
            # Arrows to projection
            ax.arrow(2, input_y[i], 1.5, 0, head_width=0.15, head_length=0.1, fc='gray', ec='gray')
    
    # Projection layer
    ax.add_patch(plt.Rectangle((3.5, 1), 2, 8, fill=True, facecolor=COLOR_NEUTRAL, edgecolor='black'))
    ax.text(4.5, 9.3, 'Average', ha='center', fontsize=11, fontweight='bold')
    ax.text(4.5, 5, 'Hidden\nLayer\n(100-300d)', ha='center', va='center', fontsize=10)
    
    # Arrow to output
    ax.arrow(5.5, 5, 1.5, 0, head_width=0.2, head_length=0.15, fc='black', ec='black', linewidth=2)
    
    # Output layer
    ax.add_patch(plt.Rectangle((7, 3.5), 2, 3, fill=True, facecolor=COLOR_PREDICT, edgecolor='black'))
    ax.text(8, 5, 'Predicted\nWord:\n"on"', ha='center', va='center', fontsize=11, fontweight='bold')
    
    plt.title('CBOW Architecture: Predict Center from Context', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/cbow_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_skipgram_architecture():
    """Create Skip-gram architecture diagram"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Hide axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Input word (center)
    ax.add_patch(plt.Rectangle((0.5, 4.5), 1.5, 1, fill=True, facecolor=COLOR_CURRENT, edgecolor='black'))
    ax.text(1.25, 5, 'on', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrow to projection
    ax.arrow(2, 5, 1.5, 0, head_width=0.2, head_length=0.15, fc='black', ec='black', linewidth=2)
    
    # Projection layer
    ax.add_patch(plt.Rectangle((3.5, 3), 2, 4, fill=True, facecolor=COLOR_NEUTRAL, edgecolor='black'))
    ax.text(4.5, 7.3, 'Embedding', ha='center', fontsize=11, fontweight='bold')
    ax.text(4.5, 5, 'Hidden\nLayer\n(100-300d)', ha='center', va='center', fontsize=10)
    
    # Arrows to outputs
    output_y = [7.5, 6, 4, 2.5]
    for y in output_y:
        ax.arrow(5.5, 5, 1.3, y-5, head_width=0.15, head_length=0.1, fc='gray', ec='gray')
    
    # Output words
    output_words = ['The', 'cat', 'the', 'mat']
    
    for i, word in enumerate(output_words):
        ax.add_patch(plt.Rectangle((7, output_y[i]-0.3), 1.5, 0.6, 
                                  fill=True, facecolor=COLOR_PREDICT, edgecolor='black'))
        ax.text(7.75, output_y[i], word, ha='center', va='center', fontsize=11)
    
    plt.title('Skip-gram Architecture: Predict Context from Center', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/skipgram_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_word2vec_clusters():
    """Create visualization of word clusters in 2D space"""
    np.random.seed(42)
    
    plt.figure(figsize=(12, 8))
    
    # Define word clusters
    clusters = {
        'Animals': ['cat', 'dog', 'mouse', 'rabbit', 'horse', 'cow', 'sheep'],
        'Countries': ['France', 'Germany', 'Italy', 'Spain', 'Japan', 'China', 'Brazil'],
        'Colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black'],
        'Actions': ['run', 'walk', 'jump', 'sit', 'stand', 'eat', 'sleep'],
        'Fruits': ['apple', 'banana', 'orange', 'grape', 'mango', 'peach', 'berry']
    }
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for idx, (category, words) in enumerate(clusters.items()):
        # Generate clustered points
        center_x = np.random.uniform(-3, 3)
        center_y = np.random.uniform(-3, 3)
        
        x = np.random.normal(center_x, 0.5, len(words))
        y = np.random.normal(center_y, 0.5, len(words))
        
        plt.scatter(x, y, s=100, c=colors[idx], alpha=0.6, label=category)
        
        # Add word labels
        for i, word in enumerate(words):
            plt.annotate(word, (x[i], y[i]), fontsize=9, 
                        xytext=(5, 5), textcoords='offset points')
    
    # Add arrows showing relationships
    plt.annotate('', xy=(1.5, 0.5), xytext=(0.5, -0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.5))
    plt.text(1, 0, 'semantic\nrelation', fontsize=9, ha='center', color='gray')
    
    plt.xlabel('Dimension 1 (after t-SNE)', fontsize=12)
    plt.ylabel('Dimension 2 (after t-SNE)', fontsize=12)
    plt.title('Word Embeddings Automatically Cluster by Meaning', fontsize=14, fontweight='bold')
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../figures/word2vec_clusters.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_polysemy_problem():
    """Visualize the polysemy problem with 'bank' example"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot
    np.random.seed(42)
    
    # Financial words cluster
    financial_words = ['money', 'loan', 'deposit', 'account', 'credit']
    financial_x = np.random.normal(2, 0.3, len(financial_words))
    financial_y = np.random.normal(2, 0.3, len(financial_words))
    
    # Nature words cluster
    nature_words = ['river', 'stream', 'water', 'shore', 'flow']
    nature_x = np.random.normal(-2, 0.3, len(nature_words))
    nature_y = np.random.normal(-2, 0.3, len(nature_words))
    
    # Plot clusters
    ax.scatter(financial_x, financial_y, s=100, c=COLOR_CONTEXT, alpha=0.6, label='Financial Context')
    ax.scatter(nature_x, nature_y, s=100, c=COLOR_PREDICT, alpha=0.6, label='Nature Context')
    
    # Add word labels
    for i, word in enumerate(financial_words):
        ax.annotate(word, (financial_x[i], financial_y[i]), fontsize=9)
    for i, word in enumerate(nature_words):
        ax.annotate(word, (nature_x[i], nature_y[i]), fontsize=9)
    
    # The problematic "bank" word - positioned between clusters
    bank_x, bank_y = 0, 0
    ax.scatter([bank_x], [bank_y], s=200, c=COLOR_CURRENT, marker='*', 
              edgecolors='black', linewidth=2, label='bank (averaged)')
    ax.annotate('bank', (bank_x, bank_y), fontsize=12, fontweight='bold',
               xytext=(10, 10), textcoords='offset points')
    
    # Add arrows showing the averaging problem
    ax.annotate('', xy=(bank_x, bank_y), xytext=(2, 2),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5))
    ax.annotate('', xy=(bank_x, bank_y), xytext=(-2, -2),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5))
    
    ax.set_xlabel('Semantic Dimension 1', fontsize=12)
    ax.set_ylabel('Semantic Dimension 2', fontsize=12)
    ax.set_title('The Polysemy Problem: One Vector for Multiple Meanings', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/polysemy_problem.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_gender_bias_embeddings():
    """Visualize gender bias in embeddings"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define professions and their typical bias
    professions = {
        'Male-biased': ['programmer', 'engineer', 'CEO', 'scientist', 'doctor'],
        'Female-biased': ['nurse', 'teacher', 'secretary', 'receptionist', 'homemaker'],
        'Neutral': ['writer', 'student', 'manager', 'analyst', 'designer']
    }
    
    # Gender words
    male_words = ['man', 'he', 'male', 'boy']
    female_words = ['woman', 'she', 'female', 'girl']
    
    # Plot on gender axis
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot male words
    for i, word in enumerate(male_words):
        ax.scatter(-2 + np.random.normal(0, 0.1), np.random.normal(3, 0.2), 
                  s=100, c='blue', alpha=0.6)
        ax.annotate(word, (-2, 3 + i*0.3), fontsize=9)
    
    # Plot female words
    for i, word in enumerate(female_words):
        ax.scatter(2 + np.random.normal(0, 0.1), np.random.normal(3, 0.2), 
                  s=100, c='pink', alpha=0.6)
        ax.annotate(word, (2, 3 + i*0.3), fontsize=9)
    
    # Plot professions
    for category, words in professions.items():
        if category == 'Male-biased':
            x_base = -1.5
            color = 'lightblue'
        elif category == 'Female-biased':
            x_base = 1.5
            color = 'lightpink'
        else:
            x_base = 0
            color = 'lightgray'
        
        for i, word in enumerate(words):
            x = x_base + np.random.normal(0, 0.2)
            y = -1 - i*0.4
            ax.scatter(x, y, s=100, c=color, edgecolors='black', linewidth=1)
            ax.annotate(word, (x, y), fontsize=9, ha='center')
    
    # Add gender axis arrow
    ax.annotate('', xy=(3, 0), xytext=(-3, 0),
               arrowprops=dict(arrowstyle='<->', lw=2, color='red'))
    ax.text(0, -3.5, 'Gender Direction', fontsize=12, ha='center', color='red', fontweight='bold')
    ax.text(-3, 0.3, 'Male', fontsize=11, ha='center')
    ax.text(3, 0.3, 'Female', fontsize=11, ha='center')
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel('Gender Bias Dimension', fontsize=12)
    ax.set_ylabel('Semantic Dimension', fontsize=12)
    ax.set_title('Gender Bias in Word Embeddings', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/gender_bias_embeddings.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_word_embeddings_space():
    """Create visualization of words in vector space with relationships"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define word positions
    words = {
        # Royal relationships
        'king': (2, 3),
        'queen': (3, 3),
        'man': (2, 1),
        'woman': (3, 1),
        
        # Countries and capitals
        'France': (-2, 2),
        'Paris': (-2, 3),
        'Italy': (-1, 2),
        'Rome': (-1, 3),
        'Spain': (-3, 2),
        'Madrid': (-3, 3),
        
        # Animals
        'cat': (1, -2),
        'kitten': (1.5, -2.2),
        'dog': (2, -2),
        'puppy': (2.5, -2.2),
        
        # Actions
        'walk': (-2, -1),
        'walking': (-1.5, -1),
        'run': (-2, -2),
        'running': (-1.5, -2)
    }
    
    # Plot words
    for word, (x, y) in words.items():
        ax.scatter(x, y, s=100, c=COLOR_NEUTRAL, edgecolors='black', linewidth=1)
        ax.annotate(word, (x, y), fontsize=10, ha='center', xytext=(0, -12), 
                   textcoords='offset points')
    
    # Draw relationship arrows
    # Gender relationship
    ax.annotate('', xy=(3, 3), xytext=(2, 3),
               arrowprops=dict(arrowstyle='->', lw=2, color='purple', alpha=0.7))
    ax.annotate('', xy=(3, 1), xytext=(2, 1),
               arrowprops=dict(arrowstyle='->', lw=2, color='purple', alpha=0.7))
    ax.text(2.5, 3.3, 'gender', fontsize=9, ha='center', color='purple')
    
    # Royal relationship
    ax.annotate('', xy=(2, 3), xytext=(2, 1),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=0.7))
    ax.annotate('', xy=(3, 3), xytext=(3, 1),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=0.7))
    ax.text(1.5, 2, 'royalty', fontsize=9, ha='center', color='blue')
    
    # Capital relationship
    ax.annotate('', xy=(-2, 3), xytext=(-2, 2),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='green', alpha=0.7))
    ax.annotate('', xy=(-1, 3), xytext=(-1, 2),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='green', alpha=0.7))
    ax.annotate('', xy=(-3, 3), xytext=(-3, 2),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='green', alpha=0.7))
    ax.text(-2, 3.5, 'capital-of', fontsize=9, ha='center', color='green')
    
    # Add clusters
    from matplotlib.patches import Circle
    animal_cluster = Circle((1.75, -2.1), 0.8, fill=False, edgecolor='orange', 
                           linewidth=2, linestyle='--', alpha=0.5)
    ax.add_patch(animal_cluster)
    ax.text(1.75, -3, 'Animals', fontsize=10, ha='center', color='orange')
    
    country_cluster = Circle((-2, 2.5), 1.2, fill=False, edgecolor='red', 
                            linewidth=2, linestyle='--', alpha=0.5)
    ax.add_patch(country_cluster)
    ax.text(-2, 4, 'Geography', fontsize=10, ha='center', color='red')
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 4)
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title('Word Embeddings Capture Semantic Relationships', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/word_embeddings_space.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_generation_probability():
    """Create visualization of text generation probability tree"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Hide axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Root word
    ax.add_patch(plt.Circle((2, 5), 0.4, fill=True, facecolor=COLOR_CURRENT, edgecolor='black'))
    ax.text(2, 5, 'cat', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Context words with probabilities
    context_words = [
        ('sat', 0.35, 7),
        ('sleeps', 0.25, 5.5),
        ('ate', 0.20, 4),
        ('ran', 0.15, 2.5),
        ('...', 0.05, 1)
    ]
    
    for word, prob, y in context_words:
        # Draw arrow
        ax.arrow(2.4, 5, 2.5, y-5, head_width=0.15, head_length=0.1, 
                fc='gray', ec='gray', alpha=0.5)
        
        # Draw word box
        box_color = COLOR_PREDICT if prob > 0.2 else COLOR_NEUTRAL
        ax.add_patch(plt.Rectangle((5, y-0.3), 1.5, 0.6, fill=True, 
                                  facecolor=box_color, edgecolor='black'))
        ax.text(5.75, y, word, ha='center', va='center', fontsize=10)
        
        # Add probability
        ax.text(4, (5+y)/2, f'{prob:.0%}', ha='center', fontsize=9, color='blue')
    
    # Add softmax notation
    ax.text(8, 8, 'Softmax', fontsize=11, fontweight='bold')
    ax.text(8, 7.5, r'$P(w|context)$', fontsize=10)
    ax.text(8, 7, r'$= \frac{e^{v_w \cdot v_c}}{\sum e^{v_i \cdot v_c}}$', fontsize=9)
    
    plt.title('Word Prediction Probabilities from Context', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/generation_probability.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_vocabulary_coverage():
    """Create Zipf's law and vocabulary coverage visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Zipf's law distribution
    ranks = np.arange(1, 10001)
    frequencies = 1 / (ranks ** 0.8) * 100000
    
    ax1.loglog(ranks[:1000], frequencies[:1000], 'b-', linewidth=2)
    ax1.fill_between(ranks[:100], frequencies[:100], alpha=0.3, color='red', 
                     label='Top 100 words\n(50% of text)')
    ax1.fill_between(ranks[100:1000], frequencies[100:1000], alpha=0.3, color='orange',
                     label='Next 900 words\n(40% of text)')
    ax1.set_xlabel('Word Rank', fontsize=11)
    ax1.set_ylabel('Word Frequency', fontsize=11)
    ax1.set_title("Zipf's Law in Natural Language", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Vocabulary coverage
    vocab_sizes = [100, 500, 1000, 5000, 10000, 50000]
    coverage = [45, 72, 83, 95, 98, 99.5]
    
    ax2.plot(vocab_sizes, coverage, 'o-', linewidth=2, markersize=8, color='green')
    ax2.fill_between(vocab_sizes, coverage, alpha=0.3, color='green')
    ax2.set_xscale('log')
    ax2.set_xlabel('Vocabulary Size', fontsize=11)
    ax2.set_ylabel('Text Coverage (%)', fontsize=11)
    ax2.set_title('Vocabulary Size vs Coverage', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add annotations
    ax2.annotate('Word2Vec typical\nvocab size', xy=(50000, 99.5), xytext=(20000, 90),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, ha='center')
    
    plt.suptitle('Word Frequency Distribution and Coverage', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/vocabulary_coverage.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_sonnet_evaluation():
    """Create evaluation metrics dashboard for text generation"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Perplexity over training
    epochs = np.arange(1, 11)
    perplexity = [850, 520, 320, 210, 150, 110, 85, 70, 62, 58]
    
    ax1.plot(epochs, perplexity, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Training Epoch', fontsize=11)
    ax1.set_ylabel('Perplexity', fontsize=11)
    ax1.set_title('Model Perplexity During Training', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Good threshold')
    ax1.legend()
    
    # Word similarity correlation
    human_scores = np.random.uniform(1, 10, 50)
    model_scores = human_scores + np.random.normal(0, 1.5, 50)
    model_scores = np.clip(model_scores, 1, 10)
    
    ax2.scatter(human_scores, model_scores, alpha=0.6)
    ax2.plot([1, 10], [1, 10], 'r--', alpha=0.5, label='Perfect correlation')
    ax2.set_xlabel('Human Similarity Score', fontsize=11)
    ax2.set_ylabel('Model Similarity Score', fontsize=11)
    ax2.set_title('Word Similarity: Human vs Model\n(r=0.72)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Analogy accuracy by category
    categories = ['Semantic', 'Syntactic', 'Capital-Country', 'Gender', 'Plural']
    accuracies = [72, 68, 85, 61, 78]
    
    bars = ax3.bar(categories, accuracies, color=['blue', 'green', 'orange', 'red', 'purple'])
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('Analogy Task Performance by Category', fontsize=12, fontweight='bold')
    ax3.axhline(y=70, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylim(0, 100)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc}%', ha='center', va='bottom', fontsize=9)
    
    # Downstream task performance
    tasks = ['Sentiment\nAnalysis', 'NER', 'POS\nTagging', 'Text\nClassification']
    baseline = [82, 87, 91, 78]
    with_embeddings = [89, 93, 95, 86]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    ax4.bar(x - width/2, baseline, width, label='Without embeddings', color='gray')
    ax4.bar(x + width/2, with_embeddings, width, label='With Word2Vec', color='green')
    
    ax4.set_ylabel('F1 Score (%)', fontsize=11)
    ax4.set_title('Downstream Task Performance', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(tasks)
    ax4.legend()
    ax4.set_ylim(70, 100)
    
    plt.suptitle('Word Embedding Evaluation Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/sonnet_evaluation.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all figures"""
    print("Generating Week 2 figures...")
    
    print("1. Embedding dimensions performance...")
    create_embedding_dimensions_performance()
    
    print("2. CBOW architecture...")
    create_cbow_architecture()
    
    print("3. Skip-gram architecture...")
    create_skipgram_architecture()
    
    print("4. Word2Vec clusters...")
    create_word2vec_clusters()
    
    print("5. Polysemy problem...")
    create_polysemy_problem()
    
    print("6. Gender bias embeddings...")
    create_gender_bias_embeddings()
    
    print("7. Word embeddings space...")
    create_word_embeddings_space()
    
    print("8. Generation probability...")
    create_generation_probability()
    
    print("9. Vocabulary coverage...")
    create_vocabulary_coverage()
    
    print("10. Evaluation metrics...")
    create_sonnet_evaluation()
    
    print("\nAll figures generated successfully!")
    print("Saved to ../figures/ directory")

if __name__ == "__main__":
    main()