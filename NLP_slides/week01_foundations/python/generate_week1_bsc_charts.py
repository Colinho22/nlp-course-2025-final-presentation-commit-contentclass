"""
Generate enhanced visualizations for Week 1 BSc-level NLP course.
Focuses on clear, educational diagrams for understanding n-gram models.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import os

# Set style for educational clarity
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# Define consistent colors for educational purposes
COLOR_CURRENT = '#FF6B6B'  # Red for current position
COLOR_CONTEXT = '#4ECDC4'  # Teal for context
COLOR_PREDICT = '#95E77E'  # Green for prediction
COLOR_NEUTRAL = '#E0E0E0'  # Gray for neutral elements

def plot_ngram_sliding_window():
    """Create visualization of n-gram sliding window through text."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    
    # Unigram (n=1)
    ax = axes[0]
    ax.set_title("Unigram Model (n=1): Each word independently", fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')
    
    for i, word in enumerate(sentence):
        color = COLOR_CURRENT if i == 2 else COLOR_NEUTRAL
        rect = FancyBboxPatch((i-0.4, 0), 0.8, 0.8, 
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(i, 0.4, word, ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax.text(3, -0.3, "Current: 'sat'", ha='center', fontsize=10, color=COLOR_CURRENT)
    ax.text(3, -0.5, "No context used", ha='center', fontsize=10, style='italic')
    
    # Bigram (n=2)
    ax = axes[1]
    ax.set_title("Bigram Model (n=2): Previous word as context", fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')
    
    for i, word in enumerate(sentence):
        if i == 1:
            color = COLOR_CONTEXT
        elif i == 2:
            color = COLOR_PREDICT
        else:
            color = COLOR_NEUTRAL
        rect = FancyBboxPatch((i-0.4, 0), 0.8, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(i, 0.4, word, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Draw arrow from context to prediction
    arrow = FancyArrowPatch((1.5, 0.4), (2.5, 0.4),
                           connectionstyle="arc3,rad=0", 
                           arrowstyle='->', mutation_scale=20, linewidth=2,
                           color='black')
    ax.add_patch(arrow)
    
    ax.text(1.5, -0.3, "Context: 'cat'", ha='center', fontsize=10, color=COLOR_CONTEXT)
    ax.text(2.5, -0.3, "Predict: 'sat'", ha='center', fontsize=10, color=COLOR_PREDICT)
    ax.text(2, -0.5, "P(sat|cat)", ha='center', fontsize=10, weight='bold')
    
    # Trigram (n=3)
    ax = axes[2]
    ax.set_title("Trigram Model (n=3): Two previous words as context", fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')
    
    for i, word in enumerate(sentence):
        if i in [0, 1]:
            color = COLOR_CONTEXT
        elif i == 2:
            color = COLOR_PREDICT
        else:
            color = COLOR_NEUTRAL
        rect = FancyBboxPatch((i-0.4, 0), 0.8, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(i, 0.4, word, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Draw bracket for context
    ax.plot([0, 1.8], [-0.1, -0.1], 'k-', linewidth=2)
    ax.plot([0, 0], [-0.1, -0.15], 'k-', linewidth=2)
    ax.plot([1.8, 1.8], [-0.1, -0.15], 'k-', linewidth=2)
    
    ax.text(0.9, -0.35, "Context: 'The cat'", ha='center', fontsize=10, color=COLOR_CONTEXT)
    ax.text(2.5, -0.3, "Predict: 'sat'", ha='center', fontsize=10, color=COLOR_PREDICT)
    ax.text(1.5, -0.55, "P(sat|The, cat)", ha='center', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig('../figures/ngram_sliding_window.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_probability_tree():
    """Create a probability tree for word prediction."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, "Probability Tree: Predicting Next Word", 
            fontsize=16, fontweight='bold', ha='center')
    
    # Root node
    ax.add_patch(Circle((5, 8), 0.5, facecolor=COLOR_NEUTRAL, edgecolor='black', linewidth=2))
    ax.text(5, 8, "the", ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Branch probabilities and nodes
    branches = [
        ("cat", 2, 6, 0.4, COLOR_CONTEXT),
        ("dog", 4, 6, 0.3, COLOR_CONTEXT),
        ("book", 6, 6, 0.2, COLOR_NEUTRAL),
        ("car", 8, 6, 0.1, COLOR_NEUTRAL)
    ]
    
    for word, x, y, prob, color in branches:
        # Draw connection
        ax.plot([5, x], [7.5, y+0.5], 'k-', linewidth=1.5)
        # Draw node
        ax.add_patch(Circle((x, y), 0.4, facecolor=color, edgecolor='black', linewidth=2))
        ax.text(x, y, word, ha='center', va='center', fontsize=10)
        # Add probability
        ax.text((5+x)/2, (7.5+y+0.5)/2 + 0.2, f"{prob:.0%}", 
               ha='center', fontsize=9, style='italic')
    
    # Second level for "cat"
    sub_branches = [
        ("sat", 0.5, 4, 0.5),
        ("ran", 1.5, 4, 0.3),
        ("slept", 2.5, 4, 0.2)
    ]
    
    for word, x, y, prob in sub_branches:
        ax.plot([2, x], [5.5, y+0.5], 'k-', linewidth=1)
        ax.add_patch(Circle((x, y), 0.35, facecolor=COLOR_PREDICT, 
                           edgecolor='black', linewidth=2))
        ax.text(x, y, word, ha='center', va='center', fontsize=9)
        ax.text((2+x)/2, (5.5+y+0.5)/2 + 0.15, f"{prob:.0%}", 
               ha='center', fontsize=8, style='italic')
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_CONTEXT,
               markersize=10, label='High probability'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_NEUTRAL,
               markersize=10, label='Low probability'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_PREDICT,
               markersize=10, label='Final prediction')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Add explanation
    ax.text(5, 1.5, "Reading: Start from 'the', follow branches based on probability", 
           ha='center', fontsize=10, style='italic')
    ax.text(5, 0.8, "Most likely path: the → cat (40%) → sat (50%) = 20% overall", 
           ha='center', fontsize=10, fontweight='bold', color=COLOR_CURRENT)
    
    plt.tight_layout()
    plt.savefig('../figures/probability_tree.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_smoothing_comparison():
    """Show the effect of smoothing on probability distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    words = ['cat', 'dog', 'sat', 'ran', 'jumped', 'flew', 'swam', 'crawled']
    
    # Without smoothing
    counts = [20, 15, 10, 8, 5, 0, 0, 0]
    total = sum(counts)
    probs_unsmoothed = [c/total if total > 0 else 0 for c in counts]
    
    ax1.bar(range(len(words)), probs_unsmoothed, color=COLOR_CURRENT, edgecolor='black')
    ax1.set_xticks(range(len(words)))
    ax1.set_xticklabels(words, rotation=45, ha='right')
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title('Without Smoothing: Zero Probabilities!', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 0.4)
    
    # Add annotations for zero probabilities
    for i, (word, prob) in enumerate(zip(words, probs_unsmoothed)):
        if prob == 0:
            ax1.text(i, 0.01, 'ZERO!', ha='center', color='red', fontweight='bold', fontsize=9)
        else:
            ax1.text(i, prob + 0.01, f'{prob:.2f}', ha='center', fontsize=9)
    
    # With add-one smoothing
    vocab_size = len(words)
    counts_smooth = [c + 1 for c in counts]
    total_smooth = sum(counts_smooth)
    probs_smoothed = [c/total_smooth for c in counts_smooth]
    
    colors = [COLOR_CONTEXT if counts[i] > 0 else COLOR_PREDICT for i in range(len(words))]
    ax2.bar(range(len(words)), probs_smoothed, color=colors, edgecolor='black')
    ax2.set_xticks(range(len(words)))
    ax2.set_xticklabels(words, rotation=45, ha='right')
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title('With Add-One Smoothing: All Words Possible!', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 0.4)
    
    # Add annotations
    for i, (word, prob) in enumerate(zip(words, probs_smoothed)):
        ax2.text(i, prob + 0.01, f'{prob:.2f}', ha='center', fontsize=9)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=COLOR_CONTEXT, label='Seen in training'),
        patches.Patch(color=COLOR_PREDICT, label='Unseen (smoothed)')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add formula
    fig.text(0.5, 0.02, 
            'Add-One Formula: P(word) = (count + 1) / (total + vocabulary_size)',
            ha='center', fontsize=11, style='italic')
    
    plt.tight_layout()
    plt.savefig('../figures/smoothing_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_ngram_timeline():
    """Create a timeline of n-gram model applications."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Timeline data
    events = [
        (1948, "Shannon\ninvents\nconcept", "Theory", COLOR_CONTEXT),
        (1970, "IBM\nspeech\nrecognition", "Application", COLOR_PREDICT),
        (1980, "Statistical\nNLP\nbegins", "Research", COLOR_NEUTRAL),
        (1990, "Machine\ntranslation\nrevolution", "Application", COLOR_PREDICT),
        (1998, "Google\nsearch\nsuggestions", "Application", COLOR_PREDICT),
        (2007, "iPhone\npredictive\ntext", "Product", COLOR_CURRENT),
        (2010, "N-grams in\nall smartphones", "Product", COLOR_CURRENT),
        (2015, "Neural models\nstart replacing\nn-grams", "Transition", COLOR_NEUTRAL),
        (2020, "N-grams still\nused for\nefficiency", "Hybrid", COLOR_CONTEXT)
    ]
    
    # Draw timeline
    ax.set_xlim(1945, 2025)
    ax.set_ylim(-2, 10)
    
    # Main timeline
    ax.axhline(y=0, color='black', linewidth=3)
    
    # Add events
    for i, (year, label, category, color) in enumerate(events):
        y_pos = 4 if i % 2 == 0 else -1.5
        
        # Draw connection line
        ax.plot([year, year], [0, y_pos], 'k--', linewidth=1, alpha=0.5)
        
        # Draw event box
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8, edgecolor='black')
        ax.text(year, y_pos, label, ha='center', va='center', fontsize=9,
               bbox=bbox_props, fontweight='bold')
        
        # Add year
        ax.plot(year, 0, 'ko', markersize=8)
        ax.text(year, -0.5 if y_pos > 0 else 0.5, str(year), 
               ha='center', va='top' if y_pos > 0 else 'bottom', fontsize=8)
        
        # Add category
        ax.text(year, y_pos + (1.5 if y_pos > 0 else -1.2), category,
               ha='center', fontsize=7, style='italic', alpha=0.7)
    
    # Add era labels
    ax.text(1960, 8, "Foundation Era", fontsize=12, fontweight='bold', style='italic')
    ax.text(1990, 8, "Statistical Revolution", fontsize=12, fontweight='bold', style='italic')
    ax.text(2010, 8, "Modern Applications", fontsize=12, fontweight='bold', style='italic')
    
    # Title
    ax.text(1985, 9.5, "N-gram Models: 75 Years of Language Prediction", 
           fontsize=16, fontweight='bold', ha='center')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('../figures/ngram_timeline.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_zipf_law_visualization():
    """Visualize Zipf's law in language."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Generate Zipfian distribution
    ranks = np.arange(1, 101)
    frequencies = 1000 / ranks  # Zipf's law: frequency proportional to 1/rank
    
    # Most common words
    common_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I']
    
    # Linear scale
    ax1.bar(ranks[:20], frequencies[:20], color=COLOR_CONTEXT, edgecolor='black')
    ax1.set_xlabel('Word Rank', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title("Zipf's Law: Word Frequency Distribution", fontsize=13, fontweight='bold')
    
    # Annotate top words
    for i, word in enumerate(common_words[:5]):
        ax1.text(i+1, frequencies[i] + 20, word, ha='center', fontsize=8, rotation=0)
    
    # Log-log scale
    ax2.loglog(ranks, frequencies, 'o-', color=COLOR_CURRENT, linewidth=2, markersize=4)
    ax2.set_xlabel('Word Rank (log scale)', fontsize=12)
    ax2.set_ylabel('Frequency (log scale)', fontsize=12)
    ax2.set_title("Log-Log Plot: Reveals Power Law", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add explanation
    ax2.text(10, 5, "Straight line =\nPower law", fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))
    
    # Add key insight
    fig.text(0.5, 0.02, 
            "Key Insight: A few words appear very often, most words are rare. 50% of word types appear only once!",
            ha='center', fontsize=11, style='italic', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../figures/zipf_law_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_perplexity_intuition():
    """Visual explanation of perplexity."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Subplot 1: What is perplexity?
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title("What is Perplexity?", fontsize=13, fontweight='bold')
    
    # Draw branching choices
    ax.add_patch(Circle((2, 5), 0.3, facecolor=COLOR_NEUTRAL, edgecolor='black'))
    ax.text(2, 5, "?", ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw options
    for i, angle in enumerate(np.linspace(0, 2*np.pi, 10, endpoint=False)):
        x = 2 + 1.5 * np.cos(angle)
        y = 5 + 1.5 * np.sin(angle)
        ax.plot([2, x], [5, y], 'k-', linewidth=1, alpha=0.5)
        ax.add_patch(Circle((x, y), 0.2, facecolor=COLOR_PREDICT, edgecolor='black'))
    
    ax.text(5, 8, "Perplexity = 10", fontsize=12, fontweight='bold')
    ax.text(5, 7.2, "Model is choosing among", fontsize=10)
    ax.text(5, 6.6, "10 equally likely options", fontsize=10)
    
    # Subplot 2: Good vs Bad perplexity
    ax = axes[0, 1]
    models = ['Random', 'Unigram', 'Bigram', 'Trigram', 'Neural']
    perplexities = [10000, 1000, 200, 100, 20]
    colors = [COLOR_CURRENT if p > 500 else COLOR_PREDICT if p < 50 else COLOR_CONTEXT 
              for p in perplexities]
    
    bars = ax.bar(models, perplexities, color=colors, edgecolor='black')
    ax.set_ylabel('Perplexity (log scale)', fontsize=11)
    ax.set_yscale('log')
    ax.set_title("Model Comparison", fontsize=13, fontweight='bold')
    
    # Add values
    for bar, val in zip(bars, perplexities):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.1, str(val),
               ha='center', fontsize=9)
    
    # Subplot 3: Formula
    ax = axes[1, 0]
    ax.axis('off')
    ax.set_title("The Math", fontsize=13, fontweight='bold')
    
    ax.text(0.5, 0.8, "Perplexity = 2^H", fontsize=14, ha='center', 
           bbox=dict(boxstyle="round,pad=0.5", facecolor=COLOR_NEUTRAL))
    ax.text(0.5, 0.5, "where H = -1/N Σ log₂ P(wᵢ|context)", fontsize=11, ha='center')
    ax.text(0.5, 0.2, "Lower perplexity = Better model", fontsize=12, ha='center',
           fontweight='bold', color=COLOR_PREDICT)
    
    # Subplot 4: Intuitive examples
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_title("Intuitive Examples", fontsize=13, fontweight='bold')
    
    examples = [
        ("PP = 2", "Like flipping a coin", COLOR_PREDICT),
        ("PP = 6", "Like rolling a die", COLOR_CONTEXT),
        ("PP = 26", "Like choosing a letter", COLOR_CONTEXT),
        ("PP = 1000", "Like choosing from dictionary", COLOR_CURRENT)
    ]
    
    for i, (pp, desc, color) in enumerate(examples):
        y_pos = 0.8 - i * 0.2
        ax.text(0.2, y_pos, pp, fontsize=11, fontweight='bold')
        ax.text(0.5, y_pos, "→", fontsize=11)
        ax.text(0.6, y_pos, desc, fontsize=10, color=color)
    
    plt.tight_layout()
    plt.savefig('../figures/perplexity_intuition.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_spell_correction_flow():
    """Flowchart for spell correction algorithm."""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(5, 13, "Spell Correction with N-grams", fontsize=16, fontweight='bold', ha='center')
    
    # Define box properties
    box_props = dict(boxstyle="round,pad=0.3", facecolor=COLOR_NEUTRAL, edgecolor='black', linewidth=2)
    decision_props = dict(boxstyle="round,pad=0.3", facecolor=COLOR_CONTEXT, edgecolor='black', linewidth=2)
    action_props = dict(boxstyle="round,pad=0.3", facecolor=COLOR_PREDICT, edgecolor='black', linewidth=2)
    
    # Flow elements
    elements = [
        (5, 11.5, "Input: 'speling'", box_props),
        (5, 10, "Check if word exists\nin dictionary", decision_props),
        (2, 8.5, "YES\nReturn word", action_props),
        (8, 8.5, "NO\nGenerate candidates", box_props),
        (8, 7, "1. One edit away\n(insert, delete, replace)", box_props),
        (8, 5.5, "2. Two edits away\n(if needed)", box_props),
        (5, 4, "Score each candidate\nwith n-gram model", decision_props),
        (5, 2.5, "Return top candidate\n'spelling'", action_props),
    ]
    
    for x, y, text, props in elements:
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
               bbox=props, fontweight='bold' if 'Input' in text or 'Return' in text else 'normal')
    
    # Draw arrows
    arrows = [
        ((5, 11.2), (5, 10.4)),  # Input to check
        ((4.5, 9.6), (2.5, 8.9)),  # Check to YES
        ((5.5, 9.6), (7.5, 8.9)),  # Check to NO
        ((8, 8.1), (8, 7.4)),  # Generate to one edit
        ((8, 6.6), (8, 5.9)),  # One edit to two edits
        ((7.5, 5.1), (5.5, 4.4)),  # Two edits to score
        ((5, 3.6), (5, 2.9)),  # Score to return
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add labels on arrows
    ax.text(3.2, 9.2, "Found", fontsize=8, style='italic')
    ax.text(6.8, 9.2, "Not found", fontsize=8, style='italic')
    
    # Add example candidates
    ax.text(1, 6, "Candidates:\n• spelling\n• spoiling\n• speeding\n• spieling", 
           fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))
    
    # Add scoring example
    ax.text(1, 3.5, "Scores (bigram):\nP('spelling correction') = 0.8\nP('spoiling correction') = 0.1\nP('speeding correction') = 0.05",
           fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.savefig('../figures/spell_correction_flow.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all figures for Week 1 BSc presentation."""
    print("Generating Week 1 BSc-level visualizations...")
    
    print("1. N-gram sliding window...")
    plot_ngram_sliding_window()
    
    print("2. Probability tree...")
    plot_probability_tree()
    
    print("3. Smoothing comparison...")
    plot_smoothing_comparison()
    
    print("4. N-gram timeline...")
    plot_ngram_timeline()
    
    print("5. Zipf's law visualization...")
    plot_zipf_law_visualization()
    
    print("6. Perplexity intuition...")
    plot_perplexity_intuition()
    
    print("7. Spell correction flowchart...")
    plot_spell_correction_flow()
    
    print("\nAll visualizations generated successfully!")
    print("Files saved in ../figures/")

if __name__ == "__main__":
    main()