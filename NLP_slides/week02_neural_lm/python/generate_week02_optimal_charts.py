"""
Generate charts for Week 2: Neural Language Models and Word Embeddings
Using Optimal Readability color scheme
"""

import sys
import os
sys.path.append(r'D:\Joerg\Research\slides\2025_NLP_16\NLP_slides\common')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from chart_utils import *
from mpl_toolkits.mplot3d import Axes3D

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# ==================================
# 1. Evolution Timeline
# ==================================
def generate_evolution_timeline():
    fig, ax = plt.subplots(figsize=(10, 5))
    setup_plotting_style()

    # Timeline data
    years = [1985, 2003, 2013, 2018, 2023]
    labels = ['N-grams', 'Neural LM', 'Word2Vec', 'BERT/GPT', 'ChatGPT']
    heights = [20, 40, 65, 85, 95]  # Performance metric

    # Draw timeline
    ax.plot(years, [0]*len(years), 'k-', linewidth=2)

    # Add milestones
    for year, label, height in zip(years, labels, heights):
        # Vertical line
        ax.plot([year, year], [0, height], color=COLORS['chart1'], linewidth=2)

        # Milestone circle
        ax.scatter(year, height, s=200, color=COLORS['chart1'],
                  edgecolor='black', linewidth=2, zorder=5)

        # Label
        ax.text(year, height + 5, label, ha='center', fontweight='bold', fontsize=10)
        ax.text(year, -8, str(year), ha='center', fontsize=9, color=COLORS['gray'])

    # Add performance curve
    from scipy.interpolate import interp1d
    f = interp1d(years, heights, kind='cubic')
    years_smooth = np.linspace(1985, 2023, 100)
    heights_smooth = f(years_smooth)
    ax.fill_between(years_smooth, 0, heights_smooth, alpha=0.2, color=COLORS['chart1'])

    # Annotations for key breakthroughs
    ax.annotate('Statistical Era', xy=(1995, 10), xytext=(1995, 35),
               ha='center', fontsize=9, color=COLORS['gray'],
               arrowprops=dict(arrowstyle='-', color=COLORS['gray'], alpha=0.5))

    ax.annotate('Deep Learning Era', xy=(2015, 50), xytext=(2015, 75),
               ha='center', fontsize=9, color=COLORS['gray'],
               arrowprops=dict(arrowstyle='-', color=COLORS['gray'], alpha=0.5))

    ax.set_xlim(1980, 2025)
    ax.set_ylim(-15, 105)
    ax.set_xlabel('Year', fontweight='bold', fontsize=11)
    ax.set_ylabel('Capability Level', fontweight='bold', fontsize=11)
    ax.set_title('Evolution of Language Modeling', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['bottom'].set_position(('data', 0))

    save_figure(fig, '../figures/evolution_timeline.pdf')
    plt.close()

# ==================================
# 2. Word2Vec Architectures
# ==================================
def generate_word2vec_architectures():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    setup_plotting_style()

    # CBOW Architecture
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('CBOW: Context → Center', fontweight='bold', fontsize=11)

    # Input words (context)
    context_words = ['The', 'cat', 'on', 'the']
    for i, word in enumerate(context_words):
        y_pos = 8 - i*2
        ax1.add_patch(plt.Rectangle((0.5, y_pos-0.3), 1.5, 0.6,
                                   facecolor=COLORS['chart2'],
                                   edgecolor='black', linewidth=1))
        ax1.text(1.25, y_pos, word, ha='center', va='center', fontweight='bold')
        ax1.arrow(2.1, y_pos, 2, 0, head_width=0.2, head_length=0.2,
                 fc=COLORS['gray'], ec=COLORS['gray'])

    # Hidden layer
    ax1.add_patch(plt.Circle((5, 5), 0.8, facecolor=COLORS['chart3'],
                            edgecolor='black', linewidth=2))
    ax1.text(5, 5, 'Sum/Avg', ha='center', va='center', fontweight='bold')

    # Output
    ax1.arrow(5.9, 5, 2, 0, head_width=0.2, head_length=0.2,
             fc=COLORS['gray'], ec=COLORS['gray'])
    ax1.add_patch(plt.Rectangle((8.5, 4.7), 1.5, 0.6,
                               facecolor=COLORS['success'],
                               edgecolor='black', linewidth=1))
    ax1.text(9.25, 5, 'sat', ha='center', va='center', fontweight='bold')

    # Skip-gram Architecture
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Skip-gram: Center → Context', fontweight='bold', fontsize=11)

    # Input (center word)
    ax2.add_patch(plt.Rectangle((0.5, 4.7), 1.5, 0.6,
                               facecolor=COLORS['success'],
                               edgecolor='black', linewidth=1))
    ax2.text(1.25, 5, 'sat', ha='center', va='center', fontweight='bold')

    # Hidden layer
    ax2.arrow(2.1, 5, 2, 0, head_width=0.2, head_length=0.2,
             fc=COLORS['gray'], ec=COLORS['gray'])
    ax2.add_patch(plt.Circle((5, 5), 0.8, facecolor=COLORS['chart3'],
                            edgecolor='black', linewidth=2))
    ax2.text(5, 5, 'Embed', ha='center', va='center', fontweight='bold')

    # Output words (context)
    for i, word in enumerate(context_words):
        y_pos = 8 - i*2
        ax2.arrow(5.9, 5, 2, y_pos-5, head_width=0.2, head_length=0.2,
                 fc=COLORS['gray'], ec=COLORS['gray'])
        ax2.add_patch(plt.Rectangle((8.5, y_pos-0.3), 1.5, 0.6,
                                   facecolor=COLORS['chart2'],
                                   edgecolor='black', linewidth=1))
        ax2.text(9.25, y_pos, word, ha='center', va='center', fontweight='bold')

    plt.suptitle('Word2Vec Architecture Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, '../figures/word2vec_architectures.pdf')
    plt.close()

# ==================================
# 3. Embedding Space 3D Visualization
# ==================================
def generate_embedding_space_3d():
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    setup_plotting_style()

    # Generate sample embeddings
    np.random.seed(42)

    # Animal cluster
    animals = ['cat', 'dog', 'horse', 'cow', 'sheep']
    animal_center = np.array([2, 2, 2])
    animal_points = animal_center + np.random.randn(len(animals), 3) * 0.5

    # Country cluster
    countries = ['France', 'Germany', 'Italy', 'Spain', 'Greece']
    country_center = np.array([-2, 2, -1])
    country_points = country_center + np.random.randn(len(countries), 3) * 0.5

    # Verb cluster
    verbs = ['run', 'walk', 'jump', 'swim', 'fly']
    verb_center = np.array([0, -2, 1])
    verb_points = verb_center + np.random.randn(len(verbs), 3) * 0.5

    # Plot clusters
    ax.scatter(animal_points[:, 0], animal_points[:, 1], animal_points[:, 2],
              c=COLORS['chart1'], s=100, label='Animals', edgecolor='black', linewidth=1)
    ax.scatter(country_points[:, 0], country_points[:, 1], country_points[:, 2],
              c=COLORS['chart2'], s=100, label='Countries', edgecolor='black', linewidth=1)
    ax.scatter(verb_points[:, 0], verb_points[:, 1], verb_points[:, 2],
              c=COLORS['chart3'], s=100, label='Verbs', edgecolor='black', linewidth=1)

    # Add labels
    for i, txt in enumerate(animals):
        ax.text(animal_points[i, 0], animal_points[i, 1], animal_points[i, 2],
               txt, fontsize=8, fontweight='bold')
    for i, txt in enumerate(countries):
        ax.text(country_points[i, 0], country_points[i, 1], country_points[i, 2],
               txt, fontsize=8, fontweight='bold')
    for i, txt in enumerate(verbs):
        ax.text(verb_points[i, 0], verb_points[i, 1], verb_points[i, 2],
               txt, fontsize=8, fontweight='bold')

    # Draw semantic relationship arrows
    # King - Man + Woman = Queen analogy
    king = np.array([1, 0, 0])
    queen = np.array([1, 0, 2])
    man = np.array([0, 0, 0])
    woman = np.array([0, 0, 2])

    ax.scatter([king[0]], [king[1]], [king[2]], c=COLORS['warning'],
              s=150, marker='^', edgecolor='black', linewidth=2)
    ax.text(king[0], king[1], king[2]+0.3, 'King', fontweight='bold', ha='center')

    ax.scatter([queen[0]], [queen[1]], [queen[2]], c=COLORS['warning'],
              s=150, marker='^', edgecolor='black', linewidth=2)
    ax.text(queen[0], queen[1], queen[2]+0.3, 'Queen', fontweight='bold', ha='center')

    # Draw relationship vector
    ax.quiver(man[0], man[1], man[2], woman[0]-man[0], woman[1]-man[1], woman[2]-man[2],
             color=COLORS['success'], arrow_length_ratio=0.1, linewidth=2, alpha=0.7)

    ax.set_xlabel('Dimension 1', fontweight='bold')
    ax.set_ylabel('Dimension 2', fontweight='bold')
    ax.set_zlabel('Dimension 3', fontweight='bold')
    ax.set_title('Word Embeddings in 3D Space', fontweight='bold', fontsize=13)
    ax.legend(loc='upper right', frameon=True, edgecolor='black')

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-3, 4)

    save_figure(fig, '../figures/embedding_space_3d.pdf')
    plt.close()

# ==================================
# 4. Softmax Visualization
# ==================================
def generate_softmax_visualization():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    setup_plotting_style()

    # Raw scores
    words = ['cat', 'dog', 'car', 'the', 'ran']
    scores = [3.2, 2.8, -1.5, -2.0, 0.5]

    ax1.barh(words, scores, color=[COLORS['success'] if s > 2 else COLORS['gray']
                                   for s in scores], edgecolor='black', linewidth=1)
    ax1.set_xlabel('Raw Score (Dot Product)', fontweight='bold')
    ax1.set_title('Before Softmax: Raw Similarity Scores', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # After softmax
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)

    ax2.barh(words, probs, color=[COLORS['success'] if p > 0.2 else COLORS['gray']
                                  for p in probs], edgecolor='black', linewidth=1)
    ax2.set_xlabel('Probability', fontweight='bold')
    ax2.set_title('After Softmax: Normalized Probabilities', fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (word, score) in enumerate(zip(words, scores)):
        ax1.text(score + 0.1 if score > 0 else score - 0.1, i, f'{score:.1f}',
                va='center', ha='left' if score > 0 else 'right', fontweight='bold')

    for i, (word, prob) in enumerate(zip(words, probs)):
        ax2.text(prob + 0.02, i, f'{prob:.2%}', va='center', fontweight='bold')

    # Add formula
    ax2.text(0.5, -1.5, r'$\mathrm{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$',
            ha='center', fontsize=11, transform=ax2.transData)

    plt.suptitle('Softmax: Converting Scores to Probabilities', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, '../figures/softmax_visualization.pdf')
    plt.close()

# ==================================
# 5. Negative Sampling Comparison
# ==================================
def generate_negative_sampling():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    setup_plotting_style()

    # Full softmax
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Full Softmax: Compute All 50,000 Words', fontweight='bold')

    # Center word
    ax1.add_patch(plt.Circle((2, 5), 0.6, facecolor=COLORS['chart1'],
                            edgecolor='black', linewidth=2))
    ax1.text(2, 5, 'cat', ha='center', fontweight='bold')

    # Draw many output connections
    vocab_sample = 20  # Represent 50k words
    for i in range(vocab_sample):
        y_pos = 0.5 + i * 0.45
        ax1.plot([2.7, 7], [5, y_pos], 'gray', alpha=0.3, linewidth=0.5)
        if i < 5:  # Label first few
            ax1.text(7.5, y_pos, f'word_{i+1}', fontsize=7, color=COLORS['gray'])

    ax1.text(7.5, 8.5, '...', fontsize=12, color=COLORS['gray'], fontweight='bold')
    ax1.text(7.5, 9, '50,000 total', fontsize=9, color=COLORS['warning'], fontweight='bold')

    # Computation cost
    ax1.text(5, 1, 'Cost: O(50,000 × d)', fontsize=10,
            color=COLORS['warning'], fontweight='bold', ha='center')

    # Negative sampling
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Negative Sampling: Only 5-20 Words', fontweight='bold')

    # Center word
    ax2.add_patch(plt.Circle((2, 5), 0.6, facecolor=COLORS['chart1'],
                            edgecolor='black', linewidth=2))
    ax2.text(2, 5, 'cat', ha='center', fontweight='bold')

    # Positive sample
    ax2.plot([2.7, 7], [5, 7], color=COLORS['success'], linewidth=2)
    ax2.add_patch(plt.Rectangle((7, 6.7), 1.5, 0.6, facecolor=COLORS['success'],
                               edgecolor='black', linewidth=1))
    ax2.text(7.75, 7, 'sat', ha='center', fontweight='bold')
    ax2.text(8.7, 7, '✓', fontsize=12, color=COLORS['success'], fontweight='bold')

    # Negative samples
    neg_words = ['car', 'tree', 'blue', 'ran', 'big']
    for i, word in enumerate(neg_words):
        y_pos = 5.5 - i * 0.8
        ax2.plot([2.7, 7], [5, y_pos], color=COLORS['warning'], linewidth=1, alpha=0.7)
        ax2.add_patch(plt.Rectangle((7, y_pos-0.3), 1.5, 0.6,
                                   facecolor='white',
                                   edgecolor=COLORS['warning'], linewidth=1))
        ax2.text(7.75, y_pos, word, ha='center', fontsize=9)
        ax2.text(8.7, y_pos, '✗', fontsize=10, color=COLORS['warning'])

    # Computation cost
    ax2.text(5, 1, 'Cost: O(6 × d)', fontsize=10,
            color=COLORS['success'], fontweight='bold', ha='center')
    ax2.text(5, 0.3, '99.98% faster!', fontsize=9,
            color=COLORS['success'], fontweight='bold', ha='center')

    plt.suptitle('Negative Sampling: The Optimization That Made Word2Vec Practical',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, '../figures/negative_sampling.pdf')
    plt.close()

# ==================================
# 6. Training Dynamics
# ==================================
def generate_training_dynamics():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    setup_plotting_style()

    epochs = np.arange(0, 50)

    # Loss curve
    ax = axes[0, 0]
    loss = 5 * np.exp(-epochs/10) + 0.5 + np.random.randn(50) * 0.05
    ax.plot(epochs, loss, color=COLORS['chart1'], linewidth=2)
    ax.fill_between(epochs, loss - 0.1, loss + 0.1, alpha=0.2, color=COLORS['chart1'])
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('Training Loss Convergence', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Similarity quality
    ax = axes[0, 1]
    similarity = 1 - np.exp(-epochs/8) + np.random.randn(50) * 0.02
    ax.plot(epochs, similarity, color=COLORS['chart2'], linewidth=2)
    ax.axhline(y=0.9, color=COLORS['success'], linestyle='--',
              linewidth=1, label='Human Performance')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Correlation with Humans', fontweight='bold')
    ax.set_title('Semantic Quality Improvement', fontweight='bold')
    ax.legend(frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Analogy accuracy
    ax = axes[1, 0]
    accuracy = 100 * (1 - np.exp(-epochs/12)) + np.random.randn(50) * 2
    accuracy = np.clip(accuracy, 0, 95)
    ax.plot(epochs, accuracy, color=COLORS['chart3'], linewidth=2)
    ax.fill_between(epochs, accuracy - 5, accuracy + 5, alpha=0.2, color=COLORS['chart3'])
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Analogy Accuracy (%)', fontweight='bold')
    ax.set_title('King-Man+Woman=Queen Performance', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # Vocabulary coverage
    ax = axes[1, 1]
    stages = ['Start', 'Epoch 10', 'Epoch 25', 'Epoch 50']
    coverage = [15, 60, 85, 95]
    colors = [COLORS['gray'], COLORS['chart4'], COLORS['chart1'], COLORS['success']]

    bars = ax.bar(stages, coverage, color=colors, edgecolor='black', linewidth=1)

    for bar, val in zip(bars, coverage):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{val}%', ha='center', fontweight='bold')

    ax.set_ylabel('Words with Good Embeddings (%)', fontweight='bold')
    ax.set_title('Vocabulary Coverage Over Time', fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Word2Vec Training Dynamics', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, '../figures/training_dynamics.pdf')
    plt.close()

# ==================================
# 7. Semantic Arithmetic
# ==================================
def generate_semantic_arithmetic():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    setup_plotting_style()

    # Helper function to draw equation
    def draw_equation(ax, eq_parts, result):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)
        ax.axis('off')

        x_pos = 1
        for i, (text, color) in enumerate(eq_parts):
            if text in ['+', '-', '=']:
                ax.text(x_pos, 1.5, text, fontsize=16, ha='center', fontweight='bold')
                x_pos += 0.8
            else:
                ax.add_patch(plt.Rectangle((x_pos-0.4, 1.2), 0.8, 0.6,
                                          facecolor=color, edgecolor='black', linewidth=1))
                ax.text(x_pos, 1.5, text, ha='center', fontweight='bold', fontsize=11)
                x_pos += 1.5

        # Result
        ax.add_patch(plt.Rectangle((x_pos-0.4, 1.2), 0.8, 0.6,
                                  facecolor=COLORS['success'], edgecolor='black', linewidth=2))
        ax.text(x_pos, 1.5, result, ha='center', fontweight='bold', fontsize=11)

    # King - Man + Woman = Queen
    draw_equation(axes[0,0],
                 [('King', COLORS['chart1']), ('-', None), ('Man', COLORS['chart2']),
                  ('+', None), ('Woman', COLORS['chart3']), ('=', None)],
                 'Queen')
    axes[0,0].set_title('Gender Relationship', fontweight='bold', y=0.8)

    # Paris - France + Italy = Rome
    draw_equation(axes[0,1],
                 [('Paris', COLORS['chart1']), ('-', None), ('France', COLORS['chart2']),
                  ('+', None), ('Italy', COLORS['chart3']), ('=', None)],
                 'Rome')
    axes[0,1].set_title('Capital Cities', fontweight='bold', y=0.8)

    # Walking - Walk + Swim = Swimming
    draw_equation(axes[1,0],
                 [('Walking', COLORS['chart1']), ('-', None), ('Walk', COLORS['chart2']),
                  ('+', None), ('Swim', COLORS['chart3']), ('=', None)],
                 'Swimming')
    axes[1,0].set_title('Verb Conjugation', fontweight='bold', y=0.8)

    # Bigger - Big + Small = Smaller
    draw_equation(axes[1,1],
                 [('Bigger', COLORS['chart1']), ('-', None), ('Big', COLORS['chart2']),
                  ('+', None), ('Small', COLORS['chart3']), ('=', None)],
                 'Smaller')
    axes[1,1].set_title('Comparative Forms', fontweight='bold', y=0.8)

    plt.suptitle('Semantic Arithmetic: Mathematical Operations on Meaning',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, '../figures/semantic_arithmetic.pdf')
    plt.close()

# ==================================
# 8. Applications Dashboard
# ==================================
def generate_applications_dashboard():
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    setup_plotting_style()

    # Search relevance improvement
    ax = axes[0, 0]
    systems = ['Keyword\nOnly', 'With\nEmbeddings']
    relevance = [65, 82]
    bars = ax.bar(systems, relevance, color=[COLORS['gray'], COLORS['success']],
                  edgecolor='black', linewidth=1)
    for bar, val in zip(bars, relevance):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
               f'{val}%', ha='center', fontweight='bold')
    ax.set_ylabel('Search Relevance', fontweight='bold')
    ax.set_title('Search Engines', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Translation quality
    ax = axes[0, 1]
    years = [2010, 2013, 2016, 2020, 2024]
    bleu_scores = [18, 24, 29, 35, 42]
    ax.plot(years, bleu_scores, 'o-', color=COLORS['chart1'],
           linewidth=2, markersize=8)
    ax.axvline(x=2013, color=COLORS['warning'], linestyle='--',
              alpha=0.5, label='Word2Vec')
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('BLEU Score', fontweight='bold')
    ax.set_title('Machine Translation', fontweight='bold')
    ax.legend(frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3)

    # Recommendation accuracy
    ax = axes[0, 2]
    categories = ['Movies', 'Music', 'Products']
    without = [70, 65, 68]
    with_emb = [85, 88, 82]

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, without, width, label='Without',
          color=COLORS['gray'], edgecolor='black', linewidth=1)
    ax.bar(x + width/2, with_emb, width, label='With Embeddings',
          color=COLORS['chart2'], edgecolor='black', linewidth=1)
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Recommendation Systems', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.legend(frameon=True, edgecolor='black', fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Sentiment analysis
    ax = axes[1, 0]
    models = ['Bag of\nWords', 'TF-IDF', 'Word2Vec', 'BERT']
    f1_scores = [72, 76, 84, 92]
    colors = [COLORS['gray'], COLORS['gray'], COLORS['chart3'], COLORS['success']]
    bars = ax.bar(models, f1_scores, color=colors, edgecolor='black', linewidth=1)
    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
               f'{val}', ha='center', fontweight='bold', fontsize=9)
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('Sentiment Analysis', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Market growth
    ax = axes[1, 1]
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    market_size = [0.3, 0.5, 0.8, 1.2, 1.6, 2.0, 2.4, 2.7]
    ax.fill_between(years, 0, market_size, color=COLORS['chart4'], alpha=0.3)
    ax.plot(years, market_size, color=COLORS['chart4'], linewidth=2, marker='o')
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Market Size ($B)', fontweight='bold')
    ax.set_title('Embedding API Market', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Developer adoption
    ax = axes[1, 2]
    platforms = ['OpenAI', 'Google', 'Hugging\nFace', 'Cohere']
    users = [450, 320, 280, 150]  # In thousands
    bars = ax.bar(platforms, users, color=COLORS['chart1'],
                  edgecolor='black', linewidth=1)
    for bar, val in zip(bars, users):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
               f'{val}K', ha='center', fontweight='bold', fontsize=9)
    ax.set_ylabel('Active Developers (1000s)', fontweight='bold')
    ax.set_title('Platform Usage (2024)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Word Embeddings: Real-World Impact Across Industries',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, '../figures/applications_dashboard.pdf')
    plt.close()

# ==================================
# Main execution
# ==================================
if __name__ == '__main__':
    print("Generating Week 2 Word Embeddings charts with Optimal Readability...")

    generate_evolution_timeline()
    print("Generated evolution timeline")

    generate_word2vec_architectures()
    print("Generated Word2Vec architectures comparison")

    generate_embedding_space_3d()
    print("Generated 3D embedding space visualization")

    generate_softmax_visualization()
    print("Generated softmax visualization")

    generate_negative_sampling()
    print("Generated negative sampling comparison")

    generate_training_dynamics()
    print("Generated training dynamics charts")

    generate_semantic_arithmetic()
    print("Generated semantic arithmetic examples")

    generate_applications_dashboard()
    print("Generated applications dashboard")

    print("All Week 2 charts generated successfully!")