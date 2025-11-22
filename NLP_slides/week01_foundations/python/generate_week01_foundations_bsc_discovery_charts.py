"""
Generate BSc-level discovery-based visualizations for Week 1: Foundations
Following Educational Presentation Framework - minimalist style with dual-slide pattern support

Date: 2025-10-26
Charts: 15 comprehensive visualizations for 42-slide presentation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch, Wedge
from matplotlib.lines import Line2D
import seaborn as sns
import os

# Create figures directory
os.makedirs('../figures', exist_ok=True)

# Educational Presentation Framework color scheme
COLOR_MLPURPLE = '#3333B2'    # Primary accent
COLOR_DARKGRAY = '#404040'    # Main text
COLOR_MIDGRAY = '#B4B4B4'     # Secondary elements
COLOR_LIGHTGRAY = '#F0F0F0'   # Backgrounds

# Pedagogical colors
COLOR_CURRENT = '#FF6B6B'  # Red for current/focus
COLOR_CONTEXT = '#4ECDC4'  # Teal for context
COLOR_PREDICT = '#95E77E'  # Green for prediction
COLOR_NEUTRAL = '#E0E0E0'  # Gray for neutral

def set_minimalist_style(ax):
    """Apply minimalist style to axis"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_MIDGRAY)
    ax.spines['bottom'].set_color(COLOR_MIDGRAY)
    ax.tick_params(colors=COLOR_DARKGRAY, which='both')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_facecolor('white')

# Chart 1: Text Quality Progression (1-gram → 2-gram → 3-gram)
def plot_text_quality_progression():
    """Show improving text quality with higher n"""
    fig, ax = plt.subplots(figsize=(12, 6))

    examples = [
        ("1-gram\n(no context)", "the the cat dog sat on mat the the dog", COLOR_NEUTRAL),
        ("2-gram\n(1 word)", "the cat sat on the mat and dog", COLOR_CONTEXT),
        ("3-gram\n(2 words)", "the cat sat on the mat", COLOR_PREDICT)
    ]

    y_positions = [2.5, 1.5, 0.5]

    for i, (model, text, color) in enumerate(examples):
        y = y_positions[i]

        # Model name box
        rect = FancyBboxPatch((0, y-0.15), 1.5, 0.3, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(rect)
        ax.text(0.75, y, model, ha='center', va='center', fontsize=11, fontweight='bold')

        # Generated text box
        rect2 = FancyBboxPatch((2, y-0.15), 8, 0.3, boxstyle="round,pad=0.05",
                               facecolor='white', edgecolor=COLOR_DARKGRAY, linewidth=1)
        ax.add_patch(rect2)
        ax.text(6, y, f'"{text}"', ha='center', va='center', fontsize=10, style='italic')

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0, 3)
    ax.axis('off')
    ax.set_title("Text Quality Improves with N-gram Order", fontsize=14, fontweight='bold',
                 color=COLOR_DARKGRAY, pad=20)

    plt.tight_layout()
    plt.savefig('../figures/text_quality_progression_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 2: Dice Probability
def plot_dice_probability():
    """Visualize uniform probability distribution for die"""
    fig, ax = plt.subplots(figsize=(10, 6))

    outcomes = [1, 2, 3, 4, 5, 6]
    probabilities = [1/6] * 6

    bars = ax.bar(outcomes, probabilities, color=COLOR_MLPURPLE, alpha=0.7, edgecolor=COLOR_DARKGRAY, linewidth=2)

    # Add probability labels
    for i, (outcome, prob) in enumerate(zip(outcomes, probabilities)):
        ax.text(outcome, prob + 0.01, f'P={prob:.3f}\n(1/6)', ha='center', va='bottom',
                fontsize=10, color=COLOR_DARKGRAY)

    ax.axhline(y=1/6, color=COLOR_MIDGRAY, linestyle='--', linewidth=1.5, alpha=0.7, label='Equal probability')

    ax.set_xlabel('Die Outcome', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_ylabel('Probability', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_title('Fair Die: All Outcomes Equally Likely', fontsize=14, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_ylim(0, 0.25)
    ax.legend(fontsize=10)

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/dice_probability_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 3: Conditional Probability Tree
def plot_conditional_probability_tree():
    """Illustrate conditional probability with simple tree"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Root node
    ax.add_patch(Circle((5, 6), 0.3, facecolor=COLOR_MLPURPLE, edgecolor=COLOR_DARKGRAY, linewidth=2))
    ax.text(5, 6, 'Start', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # First level (unconditional)
    positions_1 = [(2.5, 4), (5, 4), (7.5, 4)]
    labels_1 = ['A', 'B', 'C']
    probs_1 = ['P=0.5', 'P=0.3', 'P=0.2']

    for pos, label, prob in zip(positions_1, labels_1, probs_1):
        ax.add_patch(Circle(pos, 0.25, facecolor=COLOR_CONTEXT, edgecolor=COLOR_DARKGRAY, linewidth=1.5))
        ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=10, fontweight='bold')

        # Arrow from root
        ax.annotate('', xy=pos, xytext=(5, 6),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_DARKGRAY))
        ax.text((pos[0] + 5)/2, 5.2, prob, ha='center', fontsize=8, color=COLOR_DARKGRAY)

    # Second level (conditional on B)
    positions_2 = [(4, 2), (5, 2), (6, 2)]
    labels_2 = ['X', 'Y', 'Z']
    probs_2 = ['P=0.6|B', 'P=0.3|B', 'P=0.1|B']

    for pos, label, prob in zip(positions_2, labels_2, probs_2):
        ax.add_patch(Circle(pos, 0.2, facecolor=COLOR_PREDICT, edgecolor=COLOR_DARKGRAY, linewidth=1.5))
        ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=9, fontweight='bold')

        # Arrow from B
        ax.annotate('', xy=pos, xytext=(5, 4),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_DARKGRAY))
        ax.text(pos[0], 2.8, prob, ha='center', fontsize=7, color=COLOR_DARKGRAY)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Conditional Probability: P(next | previous)', fontsize=14, fontweight='bold',
                 color=COLOR_DARKGRAY, pad=10)

    plt.tight_layout()
    plt.savefig('../figures/conditional_probability_tree_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 4: Text as Sequence
def plot_text_sequence():
    """Visualize text as sequence of predictions"""
    fig, ax = plt.subplots(figsize=(12, 4))

    words = ["The", "cat", "sat", "on", "the", "mat"]

    for i, word in enumerate(words):
        x = i * 1.5

        # Word box
        rect = FancyBboxPatch((x, 0.5), 1.2, 0.6, boxstyle="round,pad=0.08",
                              facecolor=COLOR_PREDICT if i > 0 else COLOR_MLPURPLE,
                              edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.6, 0.8, word, ha='center', va='center', fontsize=11, fontweight='bold',
                color='white' if i == 0 else COLOR_DARKGRAY)

        # Prediction arrow (except for first word)
        if i > 0:
            arrow = FancyArrowPatch((x - 0.4, 0.8), (x + 0.1, 0.8),
                                   arrowstyle='->', mutation_scale=15, linewidth=2, color=COLOR_DARKGRAY)
            ax.add_patch(arrow)

            # Probability label
            ax.text(x - 0.15, 1.3, f'P(·|context)', ha='center', fontsize=8,
                   style='italic', color=COLOR_DARKGRAY)

    ax.set_xlim(-0.5, 8)
    ax.set_ylim(0, 2)
    ax.axis('off')
    ax.set_title('Text as Sequence of Conditional Predictions', fontsize=14, fontweight='bold',
                 color=COLOR_DARKGRAY, pad=10)

    plt.tight_layout()
    plt.savefig('../figures/text_sequence_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 5: Corpus Statistics
def plot_corpus_statistics():
    """Word frequency distribution (Zipf's law)"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate Zipfian distribution
    ranks = np.arange(1, 101)
    frequencies = 1000 / ranks  # Simplified Zipf

    ax.bar(ranks[:20], frequencies[:20], color=COLOR_MLPURPLE, alpha=0.7,
           edgecolor=COLOR_DARKGRAY, linewidth=1)

    # Highlight most common
    ax.bar(ranks[0], frequencies[0], color=COLOR_PREDICT, alpha=0.9,
           edgecolor=COLOR_DARKGRAY, linewidth=2, label='Most frequent (e.g., "the")')

    ax.set_xlabel('Word Rank', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_ylabel('Frequency Count', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_title('Word Frequency Distribution (Zipf\'s Law)', fontsize=14, fontweight='bold',
                 color=COLOR_DARKGRAY)
    ax.legend(fontsize=10)

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/corpus_statistics_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 6: N-gram Context Windows
def plot_ngram_context_windows():
    """Show unigram, bigram, trigram contexts"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    sentence = ["The", "cat", "sat", "on", "the", "mat"]

    # Unigram (n=1)
    ax = axes[0]
    ax.set_title("Unigram (n=1): No Context", fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')

    for i, word in enumerate(sentence):
        color = COLOR_CURRENT if i == 2 else COLOR_NEUTRAL
        rect = FancyBboxPatch((i-0.4, 0), 0.8, 0.8, boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(rect)
        ax.text(i, 0.4, word, ha='center', va='center', fontsize=10, fontweight='bold')

    ax.text(3, -0.3, "Focus: 'sat' (P(sat) only)", ha='center', fontsize=9, color=COLOR_DARKGRAY)

    # Bigram (n=2)
    ax = axes[1]
    ax.set_title("Bigram (n=2): Previous Word", fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
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
        rect = FancyBboxPatch((i-0.4, 0), 0.8, 0.8, boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(rect)
        ax.text(i, 0.4, word, ha='center', va='center', fontsize=10, fontweight='bold')

    arrow = FancyArrowPatch((1.5, 0.4), (2.5, 0.4), arrowstyle='->', mutation_scale=15, linewidth=2,
                           color=COLOR_DARKGRAY)
    ax.add_patch(arrow)
    ax.text(2, -0.3, "P(sat | cat)", ha='center', fontsize=9, fontweight='bold', color=COLOR_DARKGRAY)

    # Trigram (n=3)
    ax = axes[2]
    ax.set_title("Trigram (n=3): Two Previous Words", fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
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
        rect = FancyBboxPatch((i-0.4, 0), 0.8, 0.8, boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(rect)
        ax.text(i, 0.4, word, ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows from both context words
    for start in [0.5, 1.5]:
        arrow = FancyArrowPatch((start, 0.4), (2.5, 0.4), arrowstyle='->', mutation_scale=15,
                               linewidth=2, color=COLOR_DARKGRAY, alpha=0.7)
        ax.add_patch(arrow)

    ax.text(2, -0.3, "P(sat | The, cat)", ha='center', fontsize=9, fontweight='bold', color=COLOR_DARKGRAY)

    plt.tight_layout()
    plt.savefig('../figures/ngram_context_windows_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 7: Unigram Frequencies
def plot_unigram_frequencies():
    """Simple bar chart of word frequencies"""
    fig, ax = plt.subplots(figsize=(10, 6))

    words = ['the', 'cat', 'sat', 'on', 'mat', 'dog', 'and', 'or']
    counts = [7000, 850, 420, 520, 340, 650, 3200, 2100]

    bars = ax.bar(words, counts, color=COLOR_MLPURPLE, alpha=0.7, edgecolor=COLOR_DARKGRAY, linewidth=2)
    bars[0].set_color(COLOR_PREDICT)  # Highlight "the"

    ax.set_xlabel('Word', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_ylabel('Count', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_title('Unigram Model: Word Frequencies', fontsize=14, fontweight='bold', color=COLOR_DARKGRAY)

    # Add percentage labels
    total = sum(counts)
    for i, (word, count) in enumerate(zip(words, counts)):
        pct = (count / total) * 100
        ax.text(i, count + 200, f'{pct:.1f}%', ha='center', fontsize=9, color=COLOR_DARKGRAY)

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/unigram_frequencies_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 8: Bigram Context
def plot_bigram_context():
    """Visualize bigram context narrowing possibilities"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # "the" can be followed by many words
    words_after_the = ['cat', 'dog', 'mat', 'house', 'man', 'woman', 'tree', 'car']
    probs_after_the = [0.012, 0.009, 0.005, 0.007, 0.011, 0.008, 0.004, 0.006]

    bars = ax.barh(words_after_the, probs_after_the, color=COLOR_CONTEXT, alpha=0.7,
                   edgecolor=COLOR_DARKGRAY, linewidth=1.5)
    bars[0].set_color(COLOR_PREDICT)  # Highlight "cat"

    ax.set_xlabel('P(word | "the")', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_ylabel('Following Word', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_title('Bigram Context: What Follows "the"?', fontsize=14, fontweight='bold', color=COLOR_DARKGRAY)

    # Add probability values
    for i, prob in enumerate(probs_after_the):
        ax.text(prob + 0.0005, i, f'{prob:.3f}', va='center', fontsize=9, color=COLOR_DARKGRAY)

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/bigram_context_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 9: Trigram Context
def plot_trigram_context():
    """Show further narrowing with trigram"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # "on the" strongly suggests location nouns
    words_after_on_the = ['mat', 'floor', 'table', 'ground', 'roof', 'wall']
    probs_after_on_the = [0.024, 0.031, 0.019, 0.028, 0.012, 0.015]

    bars = ax.barh(words_after_on_the, probs_after_on_the, color=COLOR_CONTEXT, alpha=0.7,
                   edgecolor=COLOR_DARKGRAY, linewidth=1.5)
    bars[1].set_color(COLOR_PREDICT)  # Highlight "floor"

    ax.set_xlabel('P(word | "on", "the")', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_ylabel('Following Word', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_title('Trigram Context: What Follows "on the"?', fontsize=14, fontweight='bold',
                 color=COLOR_DARKGRAY)

    # Add probability values
    for i, prob in enumerate(probs_after_on_the):
        ax.text(prob + 0.001, i, f'{prob:.3f}', va='center', fontsize=9, color=COLOR_DARKGRAY)

    # Note about narrowing
    ax.text(0.032, -0.7, 'Note: Probabilities higher and more focused than bigram',
           fontsize=9, style='italic', color=COLOR_DARKGRAY)

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/trigram_context_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 10: Parameter Growth
def plot_parameter_growth():
    """Exponential parameter growth with n"""
    fig, ax = plt.subplots(figsize=(10, 6))

    V = 50000  # Vocabulary size
    n_values = [1, 2, 3, 4]
    params = [V**n for n in n_values]
    params_readable = ['50K', '2.5B', '125T', '6.25Q']  # K=thousand, B=billion, T=trillion, Q=quadrillion

    bars = ax.bar(n_values, params, color=COLOR_MLPURPLE, alpha=0.7, edgecolor=COLOR_DARKGRAY, linewidth=2)

    # Highlight the problem
    bars[2].set_color(COLOR_CURRENT)  # Trigram already huge
    bars[3].set_color('#CC0000')  # 4-gram impossible

    ax.set_xlabel('N-gram Order (n)', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_ylabel('Number of Parameters', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_title('Parameter Explosion: Why We Stop at Trigrams', fontsize=14, fontweight='bold',
                 color=COLOR_DARKGRAY)
    ax.set_yscale('log')
    ax.set_xticks(n_values)

    # Add readable labels
    for i, (n, param, readable) in enumerate(zip(n_values, params, params_readable)):
        ax.text(n, param * 1.5, readable, ha='center', fontsize=10, fontweight='bold', color=COLOR_DARKGRAY)

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/parameter_growth_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 11: Smoothing Comparison
def plot_smoothing_comparison():
    """Compare different smoothing methods"""
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = ['No Smoothing\n(MLE)', 'Add-1\n(Laplace)', 'Add-0.1', 'Add-0.01', 'Kneser-Ney']
    perplexities = [9999, 245, 156, 132, 98]  # 9999 represents infinity

    colors = ['#CC0000', '#FF9999', COLOR_CONTEXT, COLOR_CONTEXT, COLOR_PREDICT]
    bars = ax.bar(methods, perplexities, color=colors, alpha=0.7, edgecolor=COLOR_DARKGRAY, linewidth=2)

    ax.set_ylabel('Perplexity (lower is better)', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_title('Smoothing Methods: Performance Comparison', fontsize=14, fontweight='bold',
                 color=COLOR_DARKGRAY)
    ax.set_ylim(0, 300)

    # Add value labels
    for i, (method, perp) in enumerate(zip(methods, perplexities)):
        label = '∞' if perp > 1000 else f'{perp}'
        ax.text(i, perp + 10, label, ha='center', fontsize=10, fontweight='bold', color=COLOR_DARKGRAY)

    # Annotation
    ax.annotate('Fails on\nunseen n-grams', xy=(0, 9999), xytext=(1, 8000),
               arrowprops=dict(arrowstyle='->', color=COLOR_CURRENT, lw=2),
               fontsize=9, color=COLOR_CURRENT, fontweight='bold')

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/smoothing_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 12: Perplexity Comparison
def plot_perplexity_comparison():
    """Perplexity across n-gram orders"""
    fig, ax = plt.subplots(figsize=(10, 6))

    n_grams = ['Unigram\n(n=1)', 'Bigram\n(n=2)', 'Trigram\n(n=3)', '4-gram\n(n=4)']
    perplexities = [250, 125, 78, 72]

    bars = ax.bar(n_grams, perplexities, color=COLOR_MLPURPLE, alpha=0.7,
                  edgecolor=COLOR_DARKGRAY, linewidth=2)
    bars[2].set_color(COLOR_PREDICT)  # Highlight trigram sweet spot

    ax.set_ylabel('Perplexity', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_title('Perplexity vs N-gram Order (with smoothing)', fontsize=14, fontweight='bold',
                 color=COLOR_DARKGRAY)

    # Add value labels and improvement percentages
    for i in range(len(perplexities)):
        ax.text(i, perplexities[i] + 5, f'{perplexities[i]}', ha='center', fontsize=11,
               fontweight='bold', color=COLOR_DARKGRAY)

        if i > 0:
            improvement = ((perplexities[i-1] - perplexities[i]) / perplexities[i-1]) * 100
            ax.text(i, perplexities[i] - 15, f'−{improvement:.0f}%', ha='center', fontsize=9,
                   color=COLOR_PREDICT, fontweight='bold')

    # Annotation for diminishing returns
    ax.annotate('Diminishing\nreturns', xy=(3, 72), xytext=(3.3, 120),
               arrowprops=dict(arrowstyle='->', color=COLOR_CURRENT, lw=2),
               fontsize=9, color=COLOR_CURRENT, fontweight='bold')

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/perplexity_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 13: Add-k Smoothing Visualization
def plot_addk_smoothing():
    """Visualize effect of k parameter"""
    fig, ax = plt.subplots(figsize=(10, 6))

    k_values = [0, 0.01, 0.1, 0.5, 1.0]

    # For a seen bigram (count=850 out of 70000)
    seen_probs = []
    for k in k_values:
        prob = (850 + k) / (70000 + k * 50000)
        seen_probs.append(prob)

    # For unseen bigram (count=0)
    unseen_probs = []
    for k in k_values:
        prob = (0 + k) / (70000 + k * 50000)
        unseen_probs.append(prob)

    x = np.arange(len(k_values))
    width = 0.35

    bars1 = ax.bar(x - width/2, [p * 1000 for p in seen_probs], width, label='Seen bigram',
                   color=COLOR_PREDICT, alpha=0.7, edgecolor=COLOR_DARKGRAY, linewidth=1.5)
    bars2 = ax.bar(x + width/2, [p * 1000 for p in unseen_probs], width, label='Unseen bigram',
                   color=COLOR_CONTEXT, alpha=0.7, edgecolor=COLOR_DARKGRAY, linewidth=1.5)

    ax.set_xlabel('Smoothing Parameter k', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_ylabel('Probability (×1000)', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_title('Effect of Add-k Smoothing on Probabilities', fontsize=14, fontweight='bold',
                 color=COLOR_DARKGRAY)
    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.legend(fontsize=10)

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/addk_smoothing_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 14: Information Theory Basics
def plot_information_theory():
    """Visualize entropy concept"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Different probability distributions
    scenarios = ['Fair coin\n(max entropy)', 'Biased coin\n(90% heads)', 'Certain\n(100% heads)']
    entropies = [1.0, 0.47, 0.0]  # in bits

    bars = ax.bar(scenarios, entropies, color=[COLOR_MLPURPLE, COLOR_CONTEXT, COLOR_NEUTRAL],
                  alpha=0.7, edgecolor=COLOR_DARKGRAY, linewidth=2)

    ax.set_ylabel('Entropy (bits)', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_title('Information Theory: Entropy Measures Uncertainty', fontsize=14, fontweight='bold',
                 color=COLOR_DARKGRAY)
    ax.set_ylim(0, 1.2)

    # Add value labels
    for i, (scenario, entropy) in enumerate(zip(scenarios, entropies)):
        ax.text(i, entropy + 0.05, f'{entropy:.2f} bits', ha='center', fontsize=10,
               fontweight='bold', color=COLOR_DARKGRAY)

    # Annotations
    ax.text(0, 0.3, 'Most\nuncertain', ha='center', fontsize=9, style='italic', color=COLOR_DARKGRAY)
    ax.text(2, 0.3, 'No\nuncertainty', ha='center', fontsize=9, style='italic', color=COLOR_DARKGRAY)

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/information_theory_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 15: Unified Framework
def plot_unified_framework():
    """Final unified view of n-gram modeling"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Central concept
    central = FancyBboxPatch((4, 6), 4, 1.2, boxstyle="round,pad=0.15",
                             facecolor=COLOR_MLPURPLE, edgecolor=COLOR_DARKGRAY, linewidth=3)
    ax.add_patch(central)
    ax.text(6, 6.6, 'Language Modeling', ha='center', va='center', fontsize=14,
           fontweight='bold', color='white')
    ax.text(6, 6.3, 'P(next | previous)', ha='center', va='center', fontsize=11, color='white')

    # Components
    components = [
        ((1, 4.5), 'Text Corpus\n(Data)', COLOR_CONTEXT),
        ((4, 4.5), 'N-gram\n(Model)', COLOR_PREDICT),
        ((7, 4.5), 'MLE\n(Estimation)', COLOR_CONTEXT),
        ((10, 4.5), 'Smoothing\n(Fix zeros)', COLOR_PREDICT),
        ((2.5, 2.5), 'Training', COLOR_NEUTRAL),
        ((7, 2.5), 'Evaluation\n(Perplexity)', COLOR_NEUTRAL),
        ((10, 2.5), 'Generation', COLOR_NEUTRAL)
    ]

    for (x, y), label, color in components:
        rect = FancyBboxPatch((x, y), 2, 1, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor=COLOR_DARKGRAY, linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + 1, y + 0.5, label, ha='center', va='center', fontsize=10, fontweight='bold')

        # Arrows to/from central
        if y > 3:  # Top row connects to central
            ax.annotate('', xy=(6, 6), xytext=(x + 1, y),
                       arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_DARKGRAY))

    ax.set_xlim(0, 12)
    ax.set_ylim(1, 8)
    ax.axis('off')
    ax.set_title('Unified Framework: Statistical Language Modeling', fontsize=16, fontweight='bold',
                 color=COLOR_DARKGRAY, pad=20)

    plt.tight_layout()
    plt.savefig('../figures/unified_framework_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 16: OOV Rate vs N-gram Order
def plot_oov_rate():
    """Show increasing sparsity with higher n"""
    fig, ax = plt.subplots(figsize=(10, 6))

    n_values = [1, 2, 3, 4, 5]
    oov_rates = [5, 14, 19, 27, 35]  # Percentage of unseen n-grams in test

    ax.plot(n_values, oov_rates, marker='o', linewidth=3, markersize=10,
            color=COLOR_CURRENT, label='Out-of-Vocabulary Rate')

    for n, rate in zip(n_values, oov_rates):
        ax.text(n, rate + 1.5, f'{rate}%', ha='center', fontsize=10, fontweight='bold',
                color=COLOR_DARKGRAY)

    ax.axhline(y=20, color=COLOR_MIDGRAY, linestyle='--', alpha=0.5, linewidth=1.5,
               label='Critical threshold')

    ax.set_xlabel('N-gram Order (n)', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_ylabel('OOV Rate (%)', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_title('Sparsity Problem: Unseen N-grams in Test Data', fontsize=14,
                 fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_xticks(n_values)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 40)

    set_minimalist_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/oov_rate_ngram_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 17: Probability Distribution Before/After Smoothing
def plot_probability_distribution_smoothing():
    """Visualize probability mass redistribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Before smoothing
    words = ['seen\nhigh', 'seen\nmed', 'seen\nlow', 'unseen\n1', 'unseen\n2', 'unseen\n3']
    probs_before = [0.45, 0.30, 0.25, 0, 0, 0]

    bars1 = ax1.bar(words, probs_before, color=[COLOR_PREDICT, COLOR_PREDICT, COLOR_PREDICT,
                                                 COLOR_CURRENT, COLOR_CURRENT, COLOR_CURRENT],
                    alpha=0.7, edgecolor=COLOR_DARKGRAY, linewidth=2)

    ax1.set_ylabel('Probability', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax1.set_title('Before Smoothing (MLE)', fontsize=13, fontweight='bold', color=COLOR_DARKGRAY)
    ax1.set_ylim(0, 0.5)

    # Highlight zeros
    for i in [3, 4, 5]:
        ax1.text(i, 0.02, 'ZERO!', ha='center', fontsize=9, color='red', fontweight='bold')

    # After smoothing
    probs_after = [0.40, 0.27, 0.23, 0.03, 0.03, 0.04]

    bars2 = ax2.bar(words, probs_after, color=[COLOR_CONTEXT, COLOR_CONTEXT, COLOR_CONTEXT,
                                                COLOR_PREDICT, COLOR_PREDICT, COLOR_PREDICT],
                    alpha=0.7, edgecolor=COLOR_DARKGRAY, linewidth=2)

    ax2.set_ylabel('Probability', fontsize=12, fontweight='bold', color=COLOR_DARKGRAY)
    ax2.set_title('After Smoothing (Add-k)', fontsize=13, fontweight='bold', color=COLOR_DARKGRAY)
    ax2.set_ylim(0, 0.5)

    # Show small values
    for i in [3, 4, 5]:
        ax2.text(i, probs_after[i] + 0.02, f'{probs_after[i]:.2f}', ha='center',
                fontsize=9, color=COLOR_PREDICT, fontweight='bold')

    set_minimalist_style(ax1)
    set_minimalist_style(ax2)

    plt.tight_layout()
    plt.savefig('../figures/probability_redistribution_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 18: Bigram Calculation Step-by-Step
def plot_bigram_calculation_visual():
    """Visual walkthrough of bigram probability calculation"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Step 1: Corpus
    ax.text(6, 6.5, 'Step 1: Count in Corpus', ha='center', fontsize=13,
            fontweight='bold', color=COLOR_DARKGRAY)

    corpus_box = FancyBboxPatch((2, 5.5), 8, 0.8, boxstyle="round,pad=0.1",
                                facecolor=COLOR_LIGHTGRAY, edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(corpus_box)
    ax.text(6, 5.9, '"...the cat sat...the dog ran...the cat jumped..."', ha='center',
            fontsize=10, style='italic')

    # Step 2: Count occurrences
    ax.text(6, 4.8, 'Step 2: Count Bigrams', ha='center', fontsize=13,
            fontweight='bold', color=COLOR_DARKGRAY)

    counts = [
        ('count("the")', '= 3', COLOR_CONTEXT),
        ('count("the cat")', '= 2', COLOR_PREDICT),
    ]

    y_pos = 4.2
    for label, value, color in counts:
        rect = FancyBboxPatch((2.5, y_pos-0.15), 3, 0.3, boxstyle="round,pad=0.05",
                              facecolor=color, alpha=0.3, edgecolor=COLOR_DARKGRAY, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(4, y_pos, label, ha='center', fontsize=10, fontweight='bold')
        ax.text(7, y_pos, value, ha='center', fontsize=11, fontweight='bold', color=color)
        y_pos -= 0.5

    # Step 3: Formula
    ax.text(6, 2.8, 'Step 3: Apply MLE Formula', ha='center', fontsize=13,
            fontweight='bold', color=COLOR_DARKGRAY)

    formula_box = FancyBboxPatch((3, 1.8), 6, 0.8, boxstyle="round,pad=0.1",
                                 facecolor=COLOR_MLPURPLE, alpha=0.2,
                                 edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(formula_box)
    ax.text(6, 2.2, r'P(cat | the) = count(the, cat) / count(the)', ha='center',
            fontsize=11, fontweight='bold')
    ax.text(6, 1.9, r'= 2 / 3 = 0.667', ha='center', fontsize=12, fontweight='bold',
            color=COLOR_PREDICT)

    # Step 4: Result
    ax.text(6, 1.0, 'Result: 66.7% probability', ha='center', fontsize=13,
            fontweight='bold', color=COLOR_PREDICT)
    ax.text(6, 0.6, '"the" followed by "cat" in 2 out of 3 cases', ha='center',
            fontsize=10, style='italic', color=COLOR_DARKGRAY)

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Worked Example: Computing P(cat | the)', fontsize=14, fontweight='bold',
                 color=COLOR_DARKGRAY, pad=20)

    plt.tight_layout()
    plt.savefig('../figures/bigram_calculation_steps_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 19: Markov Assumption Illustration
def plot_markov_assumption():
    """Show context window limitation"""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))

    sentence = ["The", "author", "who", "wrote", "several", "books", "was", "awarded"]

    # Without Markov assumption (all history)
    ax = axes[0]
    ax.set_title("Without Markov Assumption: Full History", fontsize=13, fontweight='bold',
                 color=COLOR_DARKGRAY)
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')

    for i, word in enumerate(sentence):
        color = COLOR_CONTEXT if i < 7 else COLOR_PREDICT
        rect = FancyBboxPatch((i-0.4, 0), 0.8, 0.7, boxstyle="round,pad=0.06",
                              facecolor=color, edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(rect)
        ax.text(i, 0.35, word, ha='center', va='center', fontsize=9, fontweight='bold')

        # Arrows from all previous to predict
        if i < 7:
            arrow = FancyArrowPatch((i+0.4, 0.35), (6.6, 0.35), arrowstyle='->',
                                   mutation_scale=10, linewidth=1, color=COLOR_DARKGRAY, alpha=0.3)
            ax.add_patch(arrow)

    ax.text(4, -0.3, 'P(awarded | The, author, who, wrote, several, books, was)',
            ha='center', fontsize=9, fontweight='bold', color=COLOR_DARKGRAY)

    # With Markov assumption (limited context)
    ax = axes[1]
    ax.set_title("With Markov Assumption (Trigram): Last 2 Words Only", fontsize=13,
                 fontweight='bold', color=COLOR_DARKGRAY)
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')

    for i, word in enumerate(sentence):
        if i < 5:
            color = COLOR_NEUTRAL
        elif i in [5, 6]:
            color = COLOR_CONTEXT
        else:
            color = COLOR_PREDICT

        rect = FancyBboxPatch((i-0.4, 0), 0.8, 0.7, boxstyle="round,pad=0.06",
                              facecolor=color, edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(rect)
        ax.text(i, 0.35, word, ha='center', va='center', fontsize=9, fontweight='bold')

        # Arrows from last 2 only
        if i in [5, 6]:
            arrow = FancyArrowPatch((i+0.4, 0.35), (6.6, 0.35), arrowstyle='->',
                                   mutation_scale=15, linewidth=2, color=COLOR_DARKGRAY)
            ax.add_patch(arrow)

    ax.text(4, -0.3, 'P(awarded | books, was)', ha='center', fontsize=10,
            fontweight='bold', color=COLOR_PREDICT)

    plt.tight_layout()
    plt.savefig('../figures/markov_assumption_visual_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 20: Perplexity Interpretation Scale
def plot_perplexity_scale():
    """Visual scale showing what different perplexity values mean"""
    fig, ax = plt.subplots(figsize=(12, 7))

    perplexities = [10, 50, 100, 250, 500]
    y_positions = [5, 4, 3, 2, 1]
    colors = [COLOR_PREDICT, '#A0D890', COLOR_CONTEXT, '#FFB366', COLOR_CURRENT]
    interpretations = [
        'Excellent: Like choosing from 10 words\n(Neural models on clean data)',
        'Very Good: Choosing from 50 words\n(Good n-gram model)',
        'Good: Choosing from 100 words\n(Standard bigram/trigram)',
        'Poor: Choosing from 250 words\n(Weak model or noisy data)',
        'Very Poor: Choosing from 500+ words\n(Baseline or bad smoothing)'
    ]

    for pp, y, color, interp in zip(perplexities, y_positions, colors, interpretations):
        # Bar
        rect = Rectangle((0, y-0.35), pp/50, 0.7, facecolor=color, alpha=0.7,
                        edgecolor=COLOR_DARKGRAY, linewidth=2)
        ax.add_patch(rect)

        # PP value
        ax.text(pp/50 + 0.3, y, f'PP = {pp}', ha='left', va='center', fontsize=12,
                fontweight='bold', color=COLOR_DARKGRAY)

        # Interpretation
        ax.text(10.5, y, interp, ha='left', va='center', fontsize=10, color=COLOR_DARKGRAY)

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Perplexity Interpretation: What Do These Numbers Mean?', fontsize=14,
                 fontweight='bold', color=COLOR_DARKGRAY, pad=20)

    # Add arrow
    ax.annotate('Lower is Better', xy=(0.5, 5.5), xytext=(0.5, 0.5),
               arrowprops=dict(arrowstyle='->', color=COLOR_PREDICT, lw=3),
               fontsize=12, fontweight='bold', color=COLOR_PREDICT, ha='center')

    plt.tight_layout()
    plt.savefig('../figures/perplexity_scale_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Chart 21: Training vs Test Split
def plot_train_test_split():
    """Visualize corpus division for evaluation"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Full corpus
    full_width = 10
    train_width = 8
    test_width = 2

    # Full corpus box
    full_rect = Rectangle((0, 3), full_width, 1.5, facecolor=COLOR_LIGHTGRAY, alpha=0.3,
                          edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(full_rect)
    ax.text(full_width/2, 3.75, 'Full Corpus (100%)\n1,000,000 words', ha='center',
            fontsize=11, fontweight='bold', color=COLOR_DARKGRAY)

    # Arrow down
    ax.annotate('', xy=(5, 2.7), xytext=(5, 3),
               arrowprops=dict(arrowstyle='->', color=COLOR_DARKGRAY, lw=2))

    # Training set
    train_rect = Rectangle((0, 0.8), train_width, 1.5, facecolor=COLOR_PREDICT, alpha=0.5,
                           edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(train_rect)
    ax.text(train_width/2, 1.55, 'Training Set (80%)\n800,000 words\n\nBuild n-gram counts',
            ha='center', fontsize=10, fontweight='bold', color=COLOR_DARKGRAY)

    # Test set
    test_rect = Rectangle((train_width + 0.2, 0.8), test_width - 0.2, 1.5,
                          facecolor=COLOR_CONTEXT, alpha=0.5,
                          edgecolor=COLOR_DARKGRAY, linewidth=2)
    ax.add_patch(test_rect)
    ax.text(train_width + test_width/2, 1.55, 'Test Set (20%)\n200,000 words\n\nEvaluate perplexity',
            ha='center', fontsize=10, fontweight='bold', color=COLOR_DARKGRAY)

    # Important note
    note_box = FancyBboxPatch((1, -0.3), 8, 0.7, boxstyle="round,pad=0.1",
                              facecolor='#FFF3CD', edgecolor=COLOR_CURRENT, linewidth=2)
    ax.add_patch(note_box)
    ax.text(5, 0.05, 'CRITICAL: Never train on test data!\nTest set must be held out to measure true performance.',
            ha='center', fontsize=10, fontweight='bold', color=COLOR_CURRENT)

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 5)
    ax.axis('off')
    ax.set_title('Train/Test Split: Proper Evaluation Setup', fontsize=14, fontweight='bold',
                 color=COLOR_DARKGRAY, pad=20)

    plt.tight_layout()
    plt.savefig('../figures/train_test_split_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all charts
def generate_all_charts():
    """Generate all 21 BSc discovery charts (Enhanced Edition)"""
    print("Generating Week 1 BSc Discovery Charts - Enhanced Edition...")
    print("Target: 21 charts for 0.5 chart-to-slide ratio\n")

    print("1/21: Text quality progression...")
    plot_text_quality_progression()

    print("2/21: Dice probability...")
    plot_dice_probability()

    print("3/21: Conditional probability tree...")
    plot_conditional_probability_tree()

    print("4/21: Text as sequence...")
    plot_text_sequence()

    print("5/21: Corpus statistics...")
    plot_corpus_statistics()

    print("6/21: N-gram context windows...")
    plot_ngram_context_windows()

    print("7/21: Unigram frequencies...")
    plot_unigram_frequencies()

    print("8/21: Bigram context...")
    plot_bigram_context()

    print("9/21: Trigram context...")
    plot_trigram_context()

    print("10/21: Parameter growth...")
    plot_parameter_growth()

    print("11/21: Smoothing comparison...")
    plot_smoothing_comparison()

    print("12/21: Perplexity comparison...")
    plot_perplexity_comparison()

    print("13/21: Add-k smoothing...")
    plot_addk_smoothing()

    print("14/21: Information theory...")
    plot_information_theory()

    print("15/21: Unified framework...")
    plot_unified_framework()

    print("\n=== NEW CHARTS (Phase 2 Enhancement) ===")

    print("16/21: OOV rate vs n-gram order...")
    plot_oov_rate()

    print("17/21: Probability redistribution (smoothing)...")
    plot_probability_distribution_smoothing()

    print("18/21: Bigram calculation step-by-step...")
    plot_bigram_calculation_visual()

    print("19/21: Markov assumption illustration...")
    plot_markov_assumption()

    print("20/21: Perplexity interpretation scale...")
    plot_perplexity_scale()

    print("21/21: Train/test split visualization...")
    plot_train_test_split()

    print("\nAll 21 charts generated successfully in ../figures/")
    print("  Original: 15 charts")
    print("  Added: 6 new charts")
    print("  Total: 21 charts")
    print("  Chart-to-slide ratio: 21/42 = 0.50")
    print("\nFiles: *_bsc.pdf (BSc discovery versions)")

if __name__ == "__main__":
    generate_all_charts()
