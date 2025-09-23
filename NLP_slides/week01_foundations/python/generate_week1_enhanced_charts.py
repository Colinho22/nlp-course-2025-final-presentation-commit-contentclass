import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# 1. Smartphone typing visualization
def plot_smartphone_typing():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create phone outline
    phone = FancyBboxPatch((1, 0.5), 3, 5, 
                           boxstyle="round,pad=0.1",
                           facecolor='lightgray',
                           edgecolor='black',
                           linewidth=2)
    ax.add_patch(phone)
    
    # Add screen
    screen = Rectangle((1.2, 1), 2.6, 3.5, 
                      facecolor='white',
                      edgecolor='black')
    ax.add_patch(screen)
    
    # Add text being typed
    ax.text(2.5, 3.5, "The weather is really", 
            ha='center', va='center', fontsize=12)
    
    # Add prediction bar
    pred_bar = Rectangle((1.2, 2.3), 2.6, 0.5,
                        facecolor='lightblue',
                        edgecolor='black')
    ax.add_patch(pred_bar)
    
    # Add predictions
    predictions = ['nice', 'good', 'bad']
    for i, pred in enumerate(predictions):
        ax.text(1.5 + i*0.8, 2.55, pred,
               ha='center', va='center', fontsize=10)
    
    # Add keyboard hint
    keyboard = Rectangle((1.2, 1), 2.6, 1,
                        facecolor='lightgray',
                        edgecolor='black',
                        alpha=0.5)
    ax.add_patch(keyboard)
    ax.text(2.5, 1.5, 'QWERTY...', ha='center', va='center',
            fontsize=9, style='italic', alpha=0.7)
    
    # Add statistics around the phone
    stats = [
        (0.2, 4, '36.2 WPM\naverage speed'),
        (4.8, 4, '25-30%\nkeystroke savings'),
        (2.5, 6, '5 hours\ndaily usage')
    ]
    
    for x, y, text in stats:
        ax.text(x, y, text, ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor='yellow', alpha=0.7),
               fontsize=10)
    
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 6.5)
    ax.axis('off')
    ax.set_title('Predictive Text: A Daily Essential', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../figures/smartphone_typing.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Shannon portrait placeholder
def plot_shannon_portrait():
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Create a simple placeholder for Shannon
    # In practice, you would use an actual photo
    portrait = Rectangle((1, 2), 4, 5,
                        facecolor='lightgray',
                        edgecolor='black',
                        linewidth=2)
    ax.add_patch(portrait)
    
    # Add name and dates
    ax.text(3, 4.5, 'Claude Shannon', ha='center', va='center',
            fontsize=16, fontweight='bold')
    ax.text(3, 3.5, '1916-2001', ha='center', va='center',
            fontsize=12)
    ax.text(3, 2.5, 'Father of\nInformation Theory', ha='center', va='center',
            fontsize=11, style='italic')
    
    # Add quote
    ax.text(3, 1, '"Information is the resolution\nof uncertainty"',
            ha='center', va='center', fontsize=10,
            style='italic', bbox=dict(boxstyle="round,pad=0.3",
                                     facecolor='lightyellow'))
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/shannon_portrait.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Updated n-gram probability distribution
def plot_ngram_probabilities_enhanced():
    # Example: "The weather is really ___"
    words = ['nice', 'good', 'bad', 'cold', 'hot', 'beautiful', 'terrible', 'warm', 'wet', 'humid']
    # Real-world inspired probabilities
    probs = [0.25, 0.18, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04]
    
    plt.figure(figsize=(12, 7))
    
    # Create bars with gradient colors
    bars = plt.bar(words, probs, color=plt.cm.Blues(np.array(probs)/max(probs)),
                   edgecolor='navy', linewidth=1.5, alpha=0.8)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{prob:.0%}', ha='center', va='bottom', fontweight='bold')
    
    # Add cumulative probability line
    cumsum = np.cumsum(probs)
    ax2 = plt.gca().twinx()
    ax2.plot(range(len(words)), cumsum, 'r--', marker='o', linewidth=2, 
             markersize=6, label='Cumulative')
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='right')
    
    plt.xlabel('Next Word Prediction', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.title('Bigram Predictions for "The weather is really ___"', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    # Add context box
    plt.text(0.02, 0.95, 'Context: "The weather is really"', 
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'),
             fontsize=11)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('../figures/ngram_probabilities.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Enhanced perplexity comparison with real numbers
def plot_perplexity_comparison_enhanced():
    models = ['Unigram', 'Bigram', 'Trigram', '5-gram\n(Kneser-Ney)', 'LSTM\n(2018)', 'Transformer\n(GPT-2)']
    # Based on Penn Treebank results
    perplexities = [962, 170, 148, 141, 78, 35]
    
    plt.figure(figsize=(12, 8))
    
    # Create bars with different colors for different model types
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#DDA0DD', '#FECA57']
    bars = plt.bar(models, perplexities, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, perp in zip(bars, perplexities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                str(perp), ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add improvement annotations
    improvements = [
        (0.5, 1.5, 400, '-82%', 'Adding context\nhelps a lot!'),
        (3.5, 4.5, 120, '-45%', 'Neural models\nbreak through!'),
        (4.5, 5.5, 60, '-55%', 'Attention is\nall you need!')
    ]
    
    for x1, x2, y, pct, text in improvements:
        plt.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        plt.text((x1+x2)/2, y+5, pct, ha='center', fontweight='bold', color='red')
        plt.text((x1+x2)/2, y-15, text, ha='center', fontsize=9, style='italic')
    
    plt.xlabel('Model Type', fontsize=14)
    plt.ylabel('Perplexity (lower is better)', fontsize=14)
    plt.title('Language Model Performance on Penn Treebank\n(40K vocabulary, 1M tokens)', 
              fontsize=16, fontweight='bold')
    
    # Add logarithmic scale option
    plt.yscale('log')
    plt.ylim(10, 2000)
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add note about perplexity
    plt.text(0.98, 0.02, 'Perplexity ≈ Average branching factor\n(choices per word)',
             transform=plt.gca().transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../figures/perplexity_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Smoothing techniques visualization
def plot_smoothing_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Raw counts problem
    words = ['cat', 'dog', 'bird', 'elephant', 'dragon']
    counts = [100, 80, 30, 5, 0]
    
    bars1 = ax1.bar(words, counts, color='lightcoral', edgecolor='black')
    ax1.set_title('Raw Counts: The Zero Problem', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_xlabel('Word after "the"', fontsize=12)
    
    # Highlight zero count
    ax1.add_patch(Rectangle((3.6, -5), 0.8, 10, 
                           facecolor='red', alpha=0.3))
    ax1.text(4, 20, 'ZERO!\nModel fails', ha='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
             fontsize=10, color='white', fontweight='bold')
    
    # Right plot: After smoothing
    words = ['cat', 'dog', 'bird', 'elephant', 'dragon', 'unicorn*']
    smoothed = [99, 79, 29, 5, 1, 1]  # After add-one smoothing
    
    bars2 = ax2.bar(words, smoothed, color='lightgreen', edgecolor='black')
    ax2.set_title('After Smoothing: Everyone Gets Something', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Adjusted Count', fontsize=12)
    ax2.set_xlabel('Word after "the"', fontsize=12)
    ax2.set_xticklabels(words, rotation=45, ha='right')
    
    # Add note
    ax2.text(0.5, 0.95, '* Unseen word', transform=ax2.transAxes,
             ha='center', va='top', fontsize=10, style='italic')
    
    plt.suptitle('Why Smoothing Matters', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/smoothing_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all enhanced plots
if __name__ == "__main__":
    print("Generating enhanced Week 1 visualizations...")
    plot_smartphone_typing()
    print("- Smartphone typing visualization created")
    plot_shannon_portrait()
    print("- Shannon portrait placeholder created")
    plot_ngram_probabilities_enhanced()
    print("- Enhanced n-gram probabilities plot created")
    plot_perplexity_comparison_enhanced()
    print("- Enhanced perplexity comparison plot created")
    plot_smoothing_comparison()
    print("- Smoothing comparison visualization created")
    print("\nEnhanced Week 1 visualizations completed!")