import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrow, ConnectionPatch
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# 1. Seq2Seq Architecture Visualization
def plot_seq2seq_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Input sequence
    input_words = ['How', 'are', 'you', '?']
    input_x = [0.1, 0.2, 0.3, 0.4]
    
    # Output sequence  
    output_words = ['Comment', 'allez', '-', 'vous', '?']
    output_x = [0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Draw encoder
    for i, (word, x) in enumerate(zip(input_words, input_x)):
        # Input word
        ax.text(x, 0.1, word, ha='center', fontsize=12, fontweight='bold')
        
        # Encoder LSTM cell
        encoder_cell = FancyBboxPatch((x-0.04, 0.3), 0.08, 0.15,
                                     boxstyle="round,pad=0.02",
                                     facecolor='lightblue',
                                     edgecolor='black',
                                     linewidth=2)
        ax.add_patch(encoder_cell)
        ax.text(x, 0.375, 'Enc', ha='center', va='center', fontsize=10)
        
        # Input arrow
        ax.arrow(x, 0.15, 0, 0.12, head_width=0.015, head_length=0.02, fc='black', ec='black')
        
        # Hidden state connections
        if i < len(input_words) - 1:
            ax.arrow(x + 0.04, 0.375, input_x[i+1] - x - 0.08, 0,
                    head_width=0.02, head_length=0.015, fc='red', ec='red')
    
    # Context vector
    context_box = FancyBboxPatch((0.45, 0.35), 0.1, 0.1,
                                boxstyle="round,pad=0.02",
                                facecolor='gold',
                                edgecolor='black',
                                linewidth=3)
    ax.add_patch(context_box)
    ax.text(0.5, 0.4, 'Context\nVector', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow from encoder to context
    ax.arrow(0.44, 0.375, 0.01, 0, head_width=0.02, head_length=0.01, fc='red', ec='red')
    
    # Arrow from context to decoder
    ax.arrow(0.55, 0.4, 0.01, 0, head_width=0.02, head_length=0.01, fc='green', ec='green')
    
    # Draw decoder
    ax.text(0.5, 0.1, '<START>', ha='center', fontsize=10, style='italic', color='gray')
    
    for i, (word, x) in enumerate(zip(output_words, output_x)):
        # Decoder LSTM cell
        decoder_cell = FancyBboxPatch((x-0.04, 0.3), 0.08, 0.15,
                                     boxstyle="round,pad=0.02",
                                     facecolor='lightgreen',
                                     edgecolor='black',
                                     linewidth=2)
        ax.add_patch(decoder_cell)
        ax.text(x, 0.375, 'Dec', ha='center', va='center', fontsize=10)
        
        # Output arrow
        ax.arrow(x, 0.47, 0, 0.08, head_width=0.015, head_length=0.02, fc='black', ec='black')
        
        # Output word
        ax.text(x, 0.6, word, ha='center', fontsize=12, fontweight='bold')
        
        # Hidden state connections
        if i == 0:
            # Connection from context
            ax.arrow(0.55, 0.375, output_x[i] - 0.55 - 0.04, 0,
                    head_width=0.02, head_length=0.015, fc='green', ec='green')
        elif i < len(output_words) - 1:
            ax.arrow(output_x[i-1] + 0.04, 0.375, x - output_x[i-1] - 0.08, 0,
                    head_width=0.02, head_length=0.015, fc='green', ec='green')
    
    # Labels
    ax.text(0.25, 0.7, 'ENCODER', ha='center', fontsize=16, fontweight='bold', color='blue')
    ax.text(0.75, 0.7, 'DECODER', ha='center', fontsize=16, fontweight='bold', color='green')
    
    # Annotations
    ax.text(0.25, 0.05, 'Process entire input', ha='center', fontsize=11, style='italic')
    ax.text(0.75, 0.05, 'Generate output step-by-step', ha='center', fontsize=11, style='italic')
    
    # Title
    ax.text(0.5, 0.85, 'Sequence-to-Sequence Architecture', 
            ha='center', fontsize=18, fontweight='bold')
    ax.text(0.5, 0.8, 'Variable input length → Fixed context → Variable output length', 
            ha='center', fontsize=12, style='italic')
    
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 0.9)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/seq2seq_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Bottleneck Problem Visualization
def plot_bottleneck_problem():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top: Short sentence (works well)
    ax1.set_title('Short Sentence: Context Vector Captures Everything', fontsize=14, fontweight='bold')
    
    short_words = ['I', 'love', 'you']
    positions = [0.2, 0.3, 0.4]
    
    for i, (word, pos) in enumerate(zip(short_words, positions)):
        # Word
        ax1.text(pos, 0.2, word, ha='center', fontsize=14, fontweight='bold')
        
        # Information flow
        ax1.arrow(pos, 0.25, 0.5-pos, 0.15, 
                 head_width=0.02, head_length=0.02, 
                 fc='green', ec='green', alpha=0.7, linewidth=2)
    
    # Context vector
    context1 = FancyBboxPatch((0.45, 0.45), 0.1, 0.1,
                             boxstyle="round,pad=0.02",
                             facecolor='lightgreen',
                             edgecolor='black',
                             linewidth=3)
    ax1.add_patch(context1)
    ax1.text(0.5, 0.5, 'Context', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Reconstruction
    ax1.text(0.5, 0.7, '-> Full information preserved', 
            ha='center', fontsize=12, color='green', fontweight='bold')
    
    ax1.set_xlim(0.1, 0.9)
    ax1.set_ylim(0.1, 0.8)
    ax1.axis('off')
    
    # Bottom: Long sentence (bottleneck)
    ax2.set_title('Long Sentence: Information Bottleneck!', fontsize=14, fontweight='bold', color='red')
    
    long_words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
    positions = np.linspace(0.1, 0.4, len(long_words))
    
    for i, (word, pos) in enumerate(zip(long_words, positions)):
        # Word
        ax2.text(pos, 0.2, word, ha='center', fontsize=10, rotation=45)
        
        # Information flow with decay
        alpha = 0.9 - (i * 0.08)  # Earlier words fade more
        ax2.arrow(pos, 0.25, 0.5-pos, 0.15, 
                 head_width=0.015, head_length=0.015, 
                 fc='red', ec='red', alpha=alpha, linewidth=1)
    
    # Context vector (overwhelmed)
    context2 = FancyBboxPatch((0.45, 0.45), 0.1, 0.1,
                             boxstyle="round,pad=0.02",
                             facecolor='lightcoral',
                             edgecolor='black',
                             linewidth=3)
    ax2.add_patch(context2)
    ax2.text(0.5, 0.5, 'Context\n(?!)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Lost information
    ax2.text(0.5, 0.7, 'X Early words forgotten!', 
            ha='center', fontsize=12, color='red', fontweight='bold')
    
    # Fading effect for early words
    for i in range(3):
        ax2.text(positions[i], 0.15, '↓', ha='center', fontsize=20, 
                color='red', alpha=0.3 + i*0.1)
    
    ax2.set_xlim(0.05, 0.95)
    ax2.set_ylim(0.1, 0.8)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/bottleneck_problem.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Attention Visualization Heatmap
def plot_attention_visualization():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Example translation: "The black cat sat" -> "Le chat noir s'assit"
    source = ['The', 'black', 'cat', 'sat']
    target = ['Le', 'chat', 'noir', "s'assit"]
    
    # Attention weights (realistic pattern)
    attention_weights = np.array([
        [0.7, 0.1, 0.15, 0.05],  # Le -> The
        [0.1, 0.1, 0.7, 0.1],    # chat -> cat
        [0.05, 0.8, 0.1, 0.05],  # noir -> black
        [0.05, 0.05, 0.1, 0.8]   # s'assit -> sat
    ])
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(source)))
    ax.set_yticks(np.arange(len(target)))
    ax.set_xticklabels(source, fontsize=14)
    ax.set_yticklabels(target, fontsize=14)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=12)
    
    # Add text annotations
    for i in range(len(target)):
        for j in range(len(source)):
            text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                         ha="center", va="center", color="black" if attention_weights[i, j] < 0.5 else "white",
                         fontsize=12, fontweight='bold')
    
    # Labels
    ax.set_xlabel('Source (English)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Target (French)', fontsize=14, fontweight='bold')
    ax.set_title('Attention Mechanism: Where the Model Looks\nWhen Generating Each Word', 
                fontsize=16, fontweight='bold')
    
    # Add grid
    ax.set_xticks(np.arange(len(source)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(target)+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    
    # Highlight diagonal pattern
    ax.text(1.05, 0.5, 'Diagonal pattern\nshows alignment', 
            transform=ax.transAxes, fontsize=11, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'),
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('../figures/attention_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Seq2Seq Performance Timeline
def plot_seq2seq_performance():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Data points (year, BLEU score, method)
    years = [2003, 2006, 2010, 2014, 2015, 2016, 2017, 2020]
    bleu_scores = [23.0, 28.4, 31.2, 34.8, 41.8, 41.0, 43.2, 45.0]
    methods = ['Phrase-based', 'Hierarchical', 'Syntax-based', 
               'Seq2Seq', 'Seq2Seq+Attention', 'Google NMT', 
               'Transformer', 'Large Transformer']
    colors = ['gray', 'gray', 'gray', 'blue', 'green', 'green', 'red', 'red']
    
    # Plot timeline
    for i, (year, bleu, method, color) in enumerate(zip(years, bleu_scores, methods, colors)):
        ax.scatter(year, bleu, s=200, c=color, edgecolors='black', linewidth=2, zorder=5)
        
        # Add method labels
        if i < 3:
            ax.annotate(method, (year, bleu), xytext=(5, -15), 
                       textcoords='offset points', fontsize=10, ha='left')
        else:
            ax.annotate(method, (year, bleu), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10, ha='left',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
    
    # Connect points
    ax.plot(years, bleu_scores, 'k--', alpha=0.5, linewidth=1)
    
    # Highlight major breakthroughs
    ax.axvline(x=2014, color='blue', linestyle=':', alpha=0.5)
    ax.text(2014, 25, 'Neural Revolution', rotation=90, 
            fontsize=10, color='blue', ha='right', va='bottom')
    
    ax.axvline(x=2015, color='green', linestyle=':', alpha=0.5)
    ax.text(2015, 25, 'Attention Is All You Need', rotation=90, 
            fontsize=10, color='green', ha='right', va='bottom')
    
    # Labels and title
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('BLEU Score (EN→FR)', fontsize=14, fontweight='bold')
    ax.set_title('Machine Translation Quality Over Time\nThe Seq2Seq Revolution', 
                fontsize=16, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2000, 2022)
    ax.set_ylim(20, 48)
    
    # Add annotations
    ax.text(0.5, 0.95, 'Higher is better. Human performance ≈ 45-50 BLEU', 
            transform=ax.transAxes, ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    # Add performance jumps
    ax.annotate('', xy=(2015, 41.8), xytext=(2014, 34.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(2014.5, 38, '+20%!', fontsize=12, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../figures/seq2seq_performance.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Encoder-Decoder Information Flow
def plot_information_flow():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Example: "I love machine learning" -> "J'aime l'apprentissage automatique"
    
    # Encoder side
    encoder_words = ['I', 'love', 'machine', 'learning']
    enc_x = np.linspace(0.1, 0.35, len(encoder_words))
    
    for i, (word, x) in enumerate(zip(encoder_words, enc_x)):
        # Word
        ax.text(x, 0.2, word, ha='center', fontsize=12)
        
        # Embedding
        ax.arrow(x, 0.25, 0, 0.05, head_width=0.01, head_length=0.01, fc='blue', ec='blue')
        
        # LSTM cells
        cell = FancyBboxPatch((x-0.03, 0.35), 0.06, 0.1,
                             boxstyle="round,pad=0.01",
                             facecolor='lightblue', alpha=0.8,
                             edgecolor='black')
        ax.add_patch(cell)
        
        # Hidden states
        if i < len(encoder_words) - 1:
            ax.arrow(x + 0.03, 0.4, enc_x[i+1] - x - 0.06, 0,
                    head_width=0.02, head_length=0.01, fc='blue', ec='blue', alpha=0.7)
    
    # Context vector with information
    context = FancyBboxPatch((0.42, 0.3), 0.16, 0.2,
                            boxstyle="round,pad=0.02",
                            facecolor='gold', alpha=0.9,
                            edgecolor='black', linewidth=3)
    ax.add_patch(context)
    
    # Information inside context
    ax.text(0.5, 0.42, 'Compressed\nMeaning', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(0.5, 0.35, '"I love ML"', ha='center', va='center', 
            fontsize=8, style='italic')
    
    # Decoder side
    decoder_words = ["J'aime", "l'apprentissage", 'automatique']
    dec_x = np.linspace(0.65, 0.9, len(decoder_words))
    
    for i, (word, x) in enumerate(zip(decoder_words, dec_x)):
        # LSTM cells
        cell = FancyBboxPatch((x-0.03, 0.35), 0.06, 0.1,
                             boxstyle="round,pad=0.01",
                             facecolor='lightgreen', alpha=0.8,
                             edgecolor='black')
        ax.add_patch(cell)
        
        # Output
        ax.arrow(x, 0.45, 0, 0.05, head_width=0.01, head_length=0.01, fc='green', ec='green')
        
        # Word
        ax.text(x, 0.55, word, ha='center', fontsize=10)
        
        # Hidden states
        if i == 0:
            ax.arrow(0.58, 0.4, dec_x[i] - 0.58 - 0.03, 0,
                    head_width=0.02, head_length=0.01, fc='green', ec='green', alpha=0.7)
        elif i < len(decoder_words) - 1:
            ax.arrow(dec_x[i-1] + 0.03, 0.4, x - dec_x[i-1] - 0.06, 0,
                    head_width=0.02, head_length=0.01, fc='green', ec='green', alpha=0.7)
    
    # Labels
    ax.text(0.225, 0.65, 'ENCODER', fontsize=14, fontweight='bold', color='blue')
    ax.text(0.225, 0.1, 'Understand meaning', fontsize=10, style='italic', color='blue')
    
    ax.text(0.775, 0.65, 'DECODER', fontsize=14, fontweight='bold', color='green')
    ax.text(0.775, 0.1, 'Express in target language', fontsize=10, style='italic', color='green')
    
    # Title
    ax.text(0.5, 0.8, 'Information Flow in Seq2Seq', 
            ha='center', fontsize=16, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.85)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/information_flow.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 6. Attention Pattern Comparison - Shows with/without attention
def plot_attention_pattern_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Source and target sentences
    source_words = ['The', 'black', 'cat', 'sat', 'on', 'mat']
    target_words = ['Le', 'chat', 'noir', 's\'est', 'assis', 'sur', 'tapis']
    
    # Without Attention (left)
    ax1.set_title('Without Attention\n(Fixed Context)', fontsize=14, fontweight='bold')
    
    # Draw uniform connections (all source to context)
    for i, src in enumerate(source_words):
        ax1.text(i * 0.15 + 0.1, 0.8, src, ha='center', fontsize=11)
        # Draw arrow to context
        ax1.arrow(i * 0.15 + 0.1, 0.75, 0, -0.15, 
                 head_width=0.02, head_length=0.02, fc='gray', ec='gray', alpha=0.3)
    
    # Context vector (bottleneck)
    ax1.add_patch(plt.Rectangle((0.05, 0.45), 0.85, 0.12, 
                                fill=True, facecolor='red', alpha=0.3))
    ax1.text(0.475, 0.51, 'Fixed Context Vector', ha='center', fontsize=12, 
            fontweight='bold', color='darkred')
    
    # Draw connections to outputs
    for i, tgt in enumerate(target_words):
        ax1.text(i * 0.13 + 0.08, 0.2, tgt, ha='center', fontsize=11)
        ax1.arrow(0.475, 0.45, (i * 0.13 + 0.08) - 0.475, -0.2, 
                 head_width=0.02, head_length=0.02, fc='gray', ec='gray', alpha=0.3)
    
    ax1.text(0.5, 0.05, '⚠ Same context for all outputs', ha='center', 
            fontsize=10, color='red', style='italic')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 0.9)
    ax1.axis('off')
    
    # With Attention (right)
    ax2.set_title('With Attention\n(Dynamic Focus)', fontsize=14, fontweight='bold')
    
    # Attention weights for each target word
    attention_patterns = [
        [0.8, 0.1, 0.05, 0.03, 0.01, 0.01],  # Le -> The
        [0.1, 0.05, 0.7, 0.1, 0.03, 0.02],   # chat -> cat
        [0.05, 0.8, 0.1, 0.03, 0.01, 0.01],  # noir -> black
        [0.02, 0.03, 0.1, 0.8, 0.03, 0.02],  # s'est -> sat
        [0.01, 0.02, 0.03, 0.04, 0.8, 0.1],  # assis -> on
        [0.01, 0.01, 0.02, 0.03, 0.03, 0.9], # sur -> mat
        [0.02, 0.02, 0.03, 0.03, 0.1, 0.8],  # tapis -> mat
    ]
    
    # Draw source words
    for i, src in enumerate(source_words):
        ax2.text(i * 0.15 + 0.1, 0.8, src, ha='center', fontsize=11)
    
    # Draw target words with attention connections
    for j, (tgt, weights) in enumerate(zip(target_words, attention_patterns)):
        y_pos = 0.7 - j * 0.08
        ax2.text(0.85, y_pos, tgt, ha='left', fontsize=11, fontweight='bold')
        
        # Draw attention arrows
        for i, weight in enumerate(weights):
            if weight > 0.05:  # Only show significant connections
                ax2.arrow(i * 0.15 + 0.1, 0.75, 0.7 - i * 0.15, y_pos - 0.75,
                         head_width=0.015, head_length=0.01, 
                         fc='green', ec='green', alpha=weight, linewidth=weight*3)
    
    ax2.text(0.5, 0.05, '✓ Dynamic focus on relevant source words', 
            ha='center', fontsize=10, color='green', style='italic')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.9)
    ax2.axis('off')
    
    plt.suptitle('How Attention Solves the Bottleneck Problem', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../figures/attention_pattern_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 7. Translation Quality Heatmap
def plot_translation_quality_heatmap():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Data: BLEU improvement with attention (%)
    sentence_lengths = ['5-10', '10-15', '15-20', '20-25', '25-30', '30+']
    language_pairs = ['EN→FR', 'EN→DE', 'EN→ZH', 'EN→ES', 'EN→JA', 'EN→RU']
    
    # Improvement matrix (% improvement with attention)
    improvements = np.array([
        [3, 8, 15, 28, 45, 92],    # EN→FR
        [4, 10, 18, 32, 52, 105],  # EN→DE
        [5, 12, 22, 38, 61, 125],  # EN→ZH
        [3, 7, 14, 25, 42, 88],    # EN→ES
        [6, 14, 25, 42, 68, 135],  # EN→JA
        [4, 9, 17, 30, 48, 98],    # EN→RU
    ])
    
    # Create heatmap
    im = ax.imshow(improvements, cmap='YlOrRd', aspect='auto', vmin=0, vmax=140)
    
    # Set ticks
    ax.set_xticks(np.arange(len(sentence_lengths)))
    ax.set_yticks(np.arange(len(language_pairs)))
    ax.set_xticklabels(sentence_lengths)
    ax.set_yticklabels(language_pairs)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(language_pairs)):
        for j in range(len(sentence_lengths)):
            text = ax.text(j, i, f'+{improvements[i, j]}%',
                          ha="center", va="center", color="black" if improvements[i, j] < 70 else "white",
                          fontsize=10, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('BLEU Score Improvement (%)', rotation=270, labelpad=20, fontsize=12)
    
    # Labels and title
    ax.set_xlabel('Sentence Length (words)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Language Pair', fontsize=12, fontweight='bold')
    ax.set_title('Attention Impact Across Languages and Sentence Lengths\n'
                'Percentage Improvement in BLEU Score', fontsize=14, fontweight='bold')
    
    # Add dividing lines
    for i in range(1, len(language_pairs)):
        ax.axhline(i - 0.5, color='white', linewidth=2)
    for j in range(1, len(sentence_lengths)):
        ax.axvline(j - 0.5, color='white', linewidth=2)
    
    # Add annotation
    ax.text(0.5, -0.15, 'Longer sentences benefit dramatically more from attention',
           transform=ax.transAxes, ha='center', fontsize=11, style='italic',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    plt.tight_layout()
    plt.savefig('../figures/translation_quality_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 8. Attention Flow Steps - Shows step-by-step attention
def plot_attention_flow_steps():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Example translation: "The cat sat" -> "Le chat s'est assis"
    source = ['The', 'cat', 'sat']
    targets = ['Le', 'chat', "s'est", 'assis']
    
    # Attention patterns for each decoding step
    attention_steps = [
        ([0.8, 0.15, 0.05], 'Le'),      # Step 1
        ([0.1, 0.85, 0.05], 'chat'),     # Step 2
        ([0.05, 0.15, 0.8], "s'est"),    # Step 3
        ([0.05, 0.1, 0.85], 'assis'),    # Step 4
    ]
    
    for step, (ax, (weights, word)) in enumerate(zip(axes[:4], attention_steps)):
        ax.set_title(f'Step {step+1}: Generating "{word}"', fontsize=12, fontweight='bold')
        
        # Draw source words
        for i, src in enumerate(source):
            y = 0.7
            ax.text(0.2 + i * 0.3, y, src, ha='center', fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor='lightblue' if weights[i] > 0.3 else 'white',
                            alpha=weights[i]))
        
        # Draw attention weights as bars
        bar_width = 0.06
        bars = ax.bar([0.2 + i * 0.3 for i in range(3)], weights, 
                      bar_width, color=['red' if w > 0.5 else 'orange' if w > 0.2 else 'gray' 
                                       for w in weights], alpha=0.7)
        
        # Add percentage labels
        for i, (bar, w) in enumerate(zip(bars, weights)):
            ax.text(bar.get_x() + bar.get_width()/2, w + 0.02, f'{w*100:.0f}%',
                   ha='center', fontsize=10, fontweight='bold')
        
        # Draw output word
        ax.text(0.5, 0.15, word, ha='center', fontsize=16, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen'))
        
        # Arrow from attention to output
        ax.arrow(0.5, 0.35, 0, -0.12, head_width=0.03, head_length=0.02,
                fc='green', ec='green', linewidth=2)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Summary in the 5th subplot
    axes[4].text(0.5, 0.7, 'Key Insights:', ha='center', fontsize=14, fontweight='bold')
    insights = [
        '• Attention changes for each output word',
        '• Focuses on relevant source words',
        '• Handles word reordering naturally',
        '• No information bottleneck'
    ]
    for i, insight in enumerate(insights):
        axes[4].text(0.1, 0.5 - i*0.1, insight, fontsize=12)
    
    axes[4].set_xlim(0, 1)
    axes[4].set_ylim(0, 1)
    axes[4].axis('off')
    
    # Hide the 6th subplot
    axes[5].axis('off')
    
    plt.suptitle('Step-by-Step Attention Mechanism', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/attention_flow_steps.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all Week 4 visualizations
if __name__ == "__main__":
    print("Generating Week 4 Seq2Seq visualizations...")
    plot_seq2seq_architecture()
    print("- Seq2seq architecture visualization created")
    plot_bottleneck_problem()
    print("- Bottleneck problem visualization created")
    plot_attention_visualization()
    print("- Attention heatmap visualization created")
    plot_seq2seq_performance()
    print("- Performance timeline visualization created")
    plot_information_flow()
    print("- Information flow visualization created")
    plot_attention_pattern_comparison()
    print("- Attention pattern comparison created")
    plot_translation_quality_heatmap()
    print("- Translation quality heatmap created")
    plot_attention_flow_steps()
    print("- Attention flow steps visualization created")
    print("\nWeek 4 visualizations completed!")