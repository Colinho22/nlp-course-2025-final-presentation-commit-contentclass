import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# 1. N-gram probability distribution
def plot_ngram_probabilities():
    words = ['the', 'cat', 'sat', 'on', 'mat', 'dog', 'ran', 'jumped', 'slept', 'ate']
    probs = [0.25, 0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(words, probs, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xlabel('Next Word', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('Bigram Probability Distribution: P(word | "the")', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{prob:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../figures/ngram_probabilities.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Perplexity comparison across models
def plot_perplexity_comparison():
    models = ['Unigram', 'Bigram', 'Trigram', 'LSTM', 'Transformer']
    perplexities = [987, 234, 156, 45, 12]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, perplexities, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
    plt.xlabel('Model Type', fontsize=12)
    plt.ylabel('Perplexity (lower is better)', fontsize=12)
    plt.title('Language Model Perplexity Comparison', fontsize=14)
    plt.yscale('log')
    
    # Add value labels
    for bar, perp in zip(bars, perplexities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                str(perp), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../figures/perplexity_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Attention heatmap
def plot_attention_heatmap():
    # Sample attention weights
    tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    attention = np.random.rand(6, 6)
    # Make it more realistic (tokens attend more to nearby tokens)
    for i in range(6):
        for j in range(6):
            attention[i, j] *= np.exp(-abs(i-j)/2)
    
    # Normalize rows
    attention = attention / attention.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=tokens, yticklabels=tokens, cbar_kws={'label': 'Attention Weight'})
    plt.xlabel('Keys', fontsize=12)
    plt.ylabel('Queries', fontsize=12)
    plt.title('Self-Attention Weights Visualization', fontsize=14)
    plt.tight_layout()
    plt.savefig('../figures/attention_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Training loss curves
def plot_training_curves():
    epochs = np.arange(1, 51)
    
    # Simulate training curves
    rnn_loss = 4.5 * np.exp(-0.05 * epochs) + 0.5 + 0.1 * np.random.randn(50)
    lstm_loss = 4.0 * np.exp(-0.07 * epochs) + 0.3 + 0.08 * np.random.randn(50)
    transformer_loss = 3.5 * np.exp(-0.1 * epochs) + 0.15 + 0.05 * np.random.randn(50)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, rnn_loss, label='RNN', linewidth=2)
    plt.plot(epochs, lstm_loss, label='LSTM', linewidth=2)
    plt.plot(epochs, transformer_loss, label='Transformer', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.title('Training Loss Comparison Across Architectures', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../figures/training_curves.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Model size vs performance
def plot_size_vs_performance():
    # Data for various models
    model_sizes = np.array([0.1, 0.3, 1.3, 6.7, 13, 175, 530, 1760])  # in billions
    perplexities = np.array([45, 32, 20, 12, 8, 3.5, 2.8, 2.2])
    model_names = ['GPT-Small', 'GPT-Medium', 'GPT-Large', 'GPT-2', 
                   'GPT-3-13B', 'GPT-3', 'PaLM', 'GPT-4']
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(model_sizes, perplexities, s=100, alpha=0.7, c=range(len(model_sizes)), cmap='viridis')
    
    # Add labels
    for i, name in enumerate(model_names):
        plt.annotate(name, (model_sizes[i], perplexities[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Model Size (Billions of Parameters)', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title('Scaling Laws: Model Size vs Performance', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../figures/model_scaling.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 6. Tokenization example
def plot_tokenization_example():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    
    # Character-level
    text1 = "Natural Language Processing"
    chars = list(text1)
    x_pos = np.arange(len(chars))
    
    for i, char in enumerate(chars):
        rect = Rectangle((i, 0), 1, 1, linewidth=1, edgecolor='black', 
                        facecolor='lightblue' if char != ' ' else 'white')
        ax1.add_patch(rect)
        ax1.text(i+0.5, 0.5, char, ha='center', va='center', fontsize=10)
    
    ax1.set_xlim(0, len(chars))
    ax1.set_ylim(0, 1)
    ax1.set_title('Character-level Tokenization', fontsize=12)
    ax1.axis('off')
    
    # Word-level
    words = text1.split()
    x_pos = 0
    for word in words:
        width = len(word) * 0.8
        rect = Rectangle((x_pos, 0), width, 1, linewidth=1, edgecolor='black', facecolor='lightgreen')
        ax2.add_patch(rect)
        ax2.text(x_pos + width/2, 0.5, word, ha='center', va='center', fontsize=10)
        x_pos += width + 0.5
    
    ax2.set_xlim(0, x_pos)
    ax2.set_ylim(0, 1)
    ax2.set_title('Word-level Tokenization', fontsize=12)
    ax2.axis('off')
    
    # Subword (BPE-style)
    subwords = ["Nat", "ural", "Language", "Process", "ing"]
    x_pos = 0
    colors = ['#FFB6C1', '#98FB98', '#87CEEB', '#DDA0DD', '#F0E68C']
    for i, subword in enumerate(subwords):
        width = len(subword) * 0.8
        rect = Rectangle((x_pos, 0), width, 1, linewidth=1, edgecolor='black', facecolor=colors[i])
        ax3.add_patch(rect)
        ax3.text(x_pos + width/2, 0.5, subword, ha='center', va='center', fontsize=10)
        x_pos += width + 0.2
    
    ax3.set_xlim(0, x_pos)
    ax3.set_ylim(0, 1)
    ax3.set_title('Subword (BPE) Tokenization', fontsize=12)
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/tokenization_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 7. Beam search visualization
def plot_beam_search():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define the tree structure
    levels = 4
    beam_width = 3
    
    # Positions
    y_positions = np.linspace(0, 10, levels)
    
    # Sample words and probabilities
    words = [
        ["<START>"],
        ["The", "A", "An"],
        ["cat", "dog", "quick"],
        ["sat", "ran", "jumped"]
    ]
    
    probs = [
        [1.0],
        [0.6, 0.3, 0.1],
        [0.4, 0.35, 0.25],
        [0.5, 0.3, 0.2]
    ]
    
    # Draw nodes and edges
    for level in range(levels):
        x_positions = np.linspace(2, 10, len(words[level]))
        
        for i, (word, prob) in enumerate(zip(words[level], probs[level])):
            # Draw node
            circle = plt.Circle((x_positions[i], y_positions[level]), 0.8, 
                               color='lightblue', ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x_positions[i], y_positions[level], f'{word}\n{prob:.2f}', 
                   ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Draw edges to next level
            if level < levels - 1:
                next_x_positions = np.linspace(2, 10, len(words[level + 1]))
                for j in range(min(beam_width, len(words[level + 1]))):
                    ax.plot([x_positions[i], next_x_positions[j]], 
                           [y_positions[level], y_positions[level + 1]], 
                           'k-', alpha=0.3, linewidth=1)
    
    ax.set_xlim(0, 12)
    ax.set_ylim(-1, 11)
    ax.set_title('Beam Search Visualization (beam_width=3)', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/beam_search.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all plots
if __name__ == "__main__":
    print("Generating NLP course visualizations...")
    plot_ngram_probabilities()
    print("- N-gram probabilities plot created")
    plot_perplexity_comparison()
    print("- Perplexity comparison plot created")
    plot_attention_heatmap()
    print("- Attention heatmap created")
    plot_training_curves()
    print("- Training curves plot created")
    plot_size_vs_performance()
    print("- Model scaling plot created")
    plot_tokenization_example()
    print("- Tokenization comparison plot created")
    plot_beam_search()
    print("- Beam search visualization created")
    print("\nAll visualizations generated successfully!")