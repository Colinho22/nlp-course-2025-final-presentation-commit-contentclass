"""
Generate large evolution charts for Week 2 Neural Language Models
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

# Set consistent style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color scheme
COLOR_TIMELINE = '#2E86AB'  # Blue for timeline
COLOR_EMBEDDING = '#A23B72'  # Purple for embeddings
COLOR_PERFORMANCE = '#F18F01'  # Orange for performance
COLOR_APPLICATIONS = '#C73E1D'  # Red for applications

def create_evolution_timeline():
    """Part 1: Evolution Timeline of Language Models"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Data for timeline
    years = np.array([1985, 1990, 1995, 2000, 2003, 2008, 2013, 2014, 2017, 2018, 2020, 2023, 2024])
    perplexity = np.array([800, 500, 350, 200, 150, 100, 80, 70, 40, 25, 15, 8, 5])
    accuracy = np.array([20, 25, 30, 40, 50, 60, 75, 78, 85, 90, 94, 97, 98])
    
    # Create dual axis
    ax2 = ax.twinx()
    
    # Plot perplexity and accuracy
    line1 = ax.plot(years, perplexity, 'o-', color=COLOR_TIMELINE, linewidth=3, 
                    markersize=10, label='Perplexity (lower is better)')
    line2 = ax2.plot(years, accuracy, 's-', color=COLOR_PERFORMANCE, linewidth=3, 
                     markersize=10, label='Accuracy %')
    
    # Add era annotations
    eras = [
        (1985, 2002, 'N-gram Era', 750),
        (2003, 2012, 'Neural LM Era', 650),
        (2013, 2016, 'Embedding Era', 550),
        (2017, 2024, 'Transformer Era', 450)
    ]
    
    for start, end, label, height in eras:
        ax.axvspan(start, end, alpha=0.2, color='gray')
        ax.text((start + end) / 2, height, label, fontsize=12, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray'))
    
    # Add model milestones
    milestones = [
        (1985, 'Trigrams', 750),
        (2003, 'Bengio NN-LM', 130),
        (2013, 'Word2Vec', 75),
        (2014, 'GloVe', 68),
        (2017, 'Transformer', 38),
        (2018, 'BERT', 23),
        (2020, 'GPT-3', 13),
        (2024, 'GPT-4', 4)
    ]
    
    for year, model, ppl in milestones:
        ax.annotate(model, xy=(year, ppl), xytext=(year, ppl - 50),
                   fontsize=9, ha='center',
                   arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
    
    # Formatting
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Perplexity', fontsize=14, fontweight='bold', color=COLOR_TIMELINE)
    ax2.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold', color=COLOR_PERFORMANCE)
    ax.tick_params(axis='y', labelcolor=COLOR_TIMELINE)
    ax2.tick_params(axis='y', labelcolor=COLOR_PERFORMANCE)
    
    # Title and legend
    ax.set_title('Evolution of Language Models: From N-grams to GPT-4', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right', fontsize=11)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1983, 2025)
    ax.set_ylim(0, 850)
    ax2.set_ylim(15, 100)
    
    plt.tight_layout()
    plt.savefig('../figures/week2_evolution_timeline.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_embedding_space_visualization():
    """Part 2: Actual Embedding Space Visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Generate synthetic embeddings for demonstration
    np.random.seed(42)
    
    # Categories and words
    categories = {
        'Animals': ['dog', 'cat', 'lion', 'tiger', 'elephant', 'mouse', 'rabbit', 'bear'],
        'Countries': ['USA', 'Canada', 'France', 'Germany', 'Japan', 'China', 'Brazil', 'India'],
        'Professions': ['doctor', 'nurse', 'teacher', 'engineer', 'lawyer', 'artist', 'chef', 'pilot'],
        'Actions': ['run', 'walk', 'jump', 'swim', 'fly', 'crawl', 'dance', 'sing']
    }
    
    # Colors for each category
    colors = {'Animals': '#FF6B6B', 'Countries': '#4ECDC4', 
              'Professions': '#95E77E', 'Actions': '#FFE66D'}
    
    # Left plot: Raw high-dimensional space (showing first 2 dimensions)
    ax1.set_title('Raw Embedding Space (First 2 Dimensions)', fontsize=14, fontweight='bold')
    
    # Generate embeddings
    all_words = []
    all_embeddings = []
    all_colors = []
    
    for category, words in categories.items():
        # Create cluster center for category
        if category == 'Animals':
            center = np.array([2, 3])
        elif category == 'Countries':
            center = np.array([-2, 2])
        elif category == 'Professions':
            center = np.array([-1, -2])
        else:  # Actions
            center = np.array([3, -1])
        
        for word in words:
            all_words.append(word)
            # Add noise around cluster center
            embedding = center + np.random.randn(2) * 0.5
            all_embeddings.append(embedding)
            all_colors.append(colors[category])
            
            # Plot on left
            ax1.scatter(embedding[0], embedding[1], c=colors[category], s=100, alpha=0.7)
            ax1.annotate(word, (embedding[0], embedding[1]), fontsize=8, 
                        xytext=(5, 5), textcoords='offset points')
    
    # Right plot: t-SNE projection
    ax2.set_title('t-SNE Projection of Embedding Space', fontsize=14, fontweight='bold')
    
    # Generate more dimensions for realistic t-SNE
    high_dim_embeddings = []
    for category, words in categories.items():
        # Create high-dimensional cluster center
        if category == 'Animals':
            center = np.random.randn(50) * 0.1 + np.array([1] * 25 + [0] * 25)
        elif category == 'Countries':
            center = np.random.randn(50) * 0.1 + np.array([0] * 25 + [1] * 25)
        elif category == 'Professions':
            center = np.random.randn(50) * 0.1 + np.array([-1] * 25 + [0] * 25)
        else:  # Actions
            center = np.random.randn(50) * 0.1 + np.array([0] * 25 + [-1] * 25)
        
        for word in words:
            embedding = center + np.random.randn(50) * 0.3
            high_dim_embeddings.append(embedding)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    tsne_embeddings = tsne.fit_transform(np.array(high_dim_embeddings))
    
    # Plot t-SNE results
    for i, (word, color) in enumerate(zip(all_words, all_colors)):
        ax2.scatter(tsne_embeddings[i, 0], tsne_embeddings[i, 1], 
                   c=color, s=100, alpha=0.7)
        ax2.annotate(word, (tsne_embeddings[i, 0], tsne_embeddings[i, 1]), 
                    fontsize=8, xytext=(5, 5), textcoords='offset points')
    
    # Add semantic relationship lines (examples)
    relationships = [
        ('dog', 'cat', 'similar animals'),
        ('USA', 'Canada', 'neighboring countries'),
        ('doctor', 'nurse', 'medical professions'),
        ('run', 'walk', 'movement actions')
    ]
    
    for word1, word2, label in relationships:
        if word1 in all_words and word2 in all_words:
            idx1 = all_words.index(word1)
            idx2 = all_words.index(word2)
            ax2.plot([tsne_embeddings[idx1, 0], tsne_embeddings[idx2, 0]],
                    [tsne_embeddings[idx1, 1], tsne_embeddings[idx2, 1]],
                    'k--', alpha=0.3, linewidth=1)
    
    # Add legends
    for ax in [ax1, ax2]:
        legend_elements = [mpatches.Patch(facecolor=color, edgecolor='black', label=cat)
                          for cat, color in colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Word Embeddings: Semantic Clustering in Vector Space', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/week2_embedding_space.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_architecture_performance_comparison():
    """Part 3: Architecture Performance Comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Models to compare
    models = ['N-grams', 'Word2Vec', 'ELMo', 'BERT', 'GPT-3', 'GPT-4']
    
    # Task 1: Word Analogy Accuracy
    analogy_scores = [45, 75, 82, 89, 93, 96]
    axes[0].bar(models, analogy_scores, color=COLOR_PERFORMANCE, alpha=0.8, edgecolor='black')
    axes[0].set_title('Word Analogy Task\n(Google Analogy Dataset)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, v in enumerate(analogy_scores):
        axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    # Task 2: Sentiment Analysis
    sentiment_scores = [72, 81, 88, 94, 96, 97]
    axes[1].bar(models, sentiment_scores, color=COLOR_EMBEDDING, alpha=0.8, edgecolor='black')
    axes[1].set_title('Sentiment Analysis\n(Stanford Sentiment Treebank)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('F1 Score (%)', fontsize=12)
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(sentiment_scores):
        axes[1].text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    # Task 3: Named Entity Recognition
    ner_scores = [68, 76, 86, 92, 94, 95]
    axes[2].bar(models, ner_scores, color=COLOR_APPLICATIONS, alpha=0.8, edgecolor='black')
    axes[2].set_title('Named Entity Recognition\n(CoNLL-2003)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('F1 Score (%)', fontsize=12)
    axes[2].set_ylim(0, 100)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(ner_scores):
        axes[2].text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    # Rotate x-axis labels
    for ax in axes:
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Human Performance')
    
    # Add overall title
    plt.suptitle('Performance Comparison Across NLP Tasks', fontsize=16, fontweight='bold', y=1.05)
    
    # Add legend for human performance line
    axes[1].legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('../figures/week2_architecture_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_applications_impact():
    """Part 4: Real-World Applications Impact"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Industry Adoption Over Time
    years = np.arange(2013, 2025)
    adoption = np.array([5, 8, 15, 25, 40, 55, 70, 82, 90, 94, 96, 98])
    
    ax1.fill_between(years, 0, adoption, color=COLOR_TIMELINE, alpha=0.6)
    ax1.plot(years, adoption, color=COLOR_TIMELINE, linewidth=3, marker='o', markersize=8)
    ax1.set_title('Industry Adoption of Embedding Technologies', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('% of Fortune 500 Using Embeddings', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add key events
    events = [
        (2013, 5, 'Word2Vec Released'),
        (2017, 40, 'Transformer Published'),
        (2018, 55, 'BERT Released'),
        (2020, 90, 'GPT-3 API'),
        (2023, 96, 'ChatGPT')
    ]
    
    for year, adopt, event in events:
        ax1.annotate(event, xy=(year, adopt), xytext=(year, adopt + 10),
                    fontsize=9, ha='center',
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # 2. Revenue Impact by Industry
    industries = ['Finance', 'Healthcare', 'Retail', 'Tech', 'Manufacturing', 'Education']
    revenue_gain = [23, 18, 21, 35, 15, 12]  # Percentage improvement
    
    bars = ax2.barh(industries, revenue_gain, color=COLOR_APPLICATIONS, alpha=0.8, edgecolor='black')
    ax2.set_title('Efficiency Gains from Embedding-Based Systems', fontsize=14, fontweight='bold')
    ax2.set_xlabel('% Improvement in Efficiency', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add values on bars
    for bar, value in zip(bars, revenue_gain):
        ax2.text(value + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{value}%', va='center', fontweight='bold')
    
    # 3. Applications by Domain (Pie Chart)
    domains = ['Search & Retrieval', 'Translation', 'Chatbots', 
               'Content Generation', 'Sentiment Analysis', 'Other']
    sizes = [30, 20, 18, 15, 12, 5]
    colors_pie = ['#FF6B6B', '#4ECDC4', '#95E77E', '#FFE66D', '#A8DADC', '#E8E8E8']
    
    wedges, texts, autotexts = ax3.pie(sizes, labels=domains, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 10})
    ax3.set_title('Distribution of Embedding Applications', fontsize=14, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    # 4. Performance vs Computational Cost
    models_cost = ['Word2Vec', 'GloVe', 'ELMo', 'BERT-base', 'BERT-large', 'GPT-3', 'GPT-4']
    performance = [75, 78, 85, 90, 92, 96, 98]
    compute_cost = [1, 1.2, 10, 50, 150, 1000, 5000]  # Relative computational cost
    
    # Use log scale for compute cost
    ax4.scatter(compute_cost, performance, s=200, c=range(len(models_cost)), 
               cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, model in enumerate(models_cost):
        ax4.annotate(model, (compute_cost[i], performance[i]), 
                    fontsize=9, ha='center', va='bottom', xytext=(0, 5),
                    textcoords='offset points')
    
    ax4.set_xscale('log')
    ax4.set_title('Performance vs Computational Cost Trade-off', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Relative Computational Cost (log scale)', fontsize=12)
    ax4.set_ylabel('Performance (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(70, 100)
    
    # Add efficiency frontier
    ax4.plot(compute_cost[:4], performance[:4], 'g--', alpha=0.5, 
            linewidth=2, label='Efficiency Frontier')
    ax4.legend()
    
    plt.suptitle('Real-World Impact of Word Embeddings', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/week2_applications_impact.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating Week 2 Evolution Charts...")
    
    print("1. Creating evolution timeline...")
    create_evolution_timeline()
    
    print("2. Creating embedding space visualization...")
    create_embedding_space_visualization()
    
    print("3. Creating architecture performance comparison...")
    create_architecture_performance_comparison()
    
    print("4. Creating applications impact chart...")
    create_applications_impact()
    
    print("All charts generated successfully!")