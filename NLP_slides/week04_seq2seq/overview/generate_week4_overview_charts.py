"""
Generate Week 4 overview visualizations for Seq2Seq models
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrow
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Educational colors
COLOR_CURRENT = '#FF6B6B'  # Red
COLOR_CONTEXT = '#4ECDC4'  # Teal  
COLOR_PREDICT = '#95E77E'  # Green
COLOR_NEUTRAL = '#E0E0E0'  # Gray

def create_attention_heatmap():
    """Create attention weight heatmap visualization"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sample translation task
    source_words = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    target_words = ['Le', 'chat', 'était', 'assis', 'sur', 'le', 'tapis']
    
    # Generate attention weights (simulated)
    np.random.seed(42)
    attention_weights = np.random.rand(len(target_words), len(source_words))
    
    # Create stronger alignments for realistic pattern
    for i in range(min(len(source_words), len(target_words))):
        attention_weights[i, i] = np.random.uniform(0.7, 1.0)
    
    # Normalize rows to sum to 1
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.5)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(source_words)))
    ax.set_yticks(np.arange(len(target_words)))
    ax.set_xticklabels(source_words, fontsize=11)
    ax.set_yticklabels(target_words, fontsize=11)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20, fontsize=10)
    
    # Add values in cells
    for i in range(len(target_words)):
        for j in range(len(source_words)):
            text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                         ha="center", va="center", color="black" if attention_weights[i, j] < 0.3 else "white",
                         fontsize=8)
    
    ax.set_title('Attention Weights: English → French Translation', fontsize=13, fontweight='bold', pad=20)
    ax.set_xlabel('Source (English)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Target (French)', fontsize=11, fontweight='bold')
    
    # Add annotation
    ax.text(1.15, 0.5, 'Brighter = Higher Attention\nModel learns alignment', 
           transform=ax.transAxes, fontsize=10, va='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('../figures/week4_attention_heatmap.pdf', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()

def create_encoder_decoder_flow():
    """Create encoder-decoder architecture flow diagram"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Encoder side
    encoder_x = 0.25
    encoder_words = ['Hello', 'world', '!']
    
    for i, word in enumerate(encoder_words):
        y = 0.7 - i * 0.15
        # Input boxes
        input_box = FancyBboxPatch((encoder_x - 0.08, y - 0.04), 0.12, 0.08,
                                  boxstyle="round,pad=0.01",
                                  facecolor=COLOR_CONTEXT, alpha=0.7,
                                  edgecolor='white', linewidth=2)
        ax.add_patch(input_box)
        ax.text(encoder_x - 0.02, y, word, fontsize=10, ha='center', va='center',
               color='white', fontweight='bold')
        
        # Encoder RNN cells
        encoder_cell = Circle((encoder_x + 0.15, y), 0.03,
                             facecolor=COLOR_CONTEXT, alpha=0.5,
                             edgecolor='white', linewidth=2)
        ax.add_patch(encoder_cell)
        
        # Arrows
        ax.arrow(encoder_x + 0.05, y, 0.07, 0,
                head_width=0.015, head_length=0.01, fc='gray', alpha=0.5)
    
    # Context vector
    context_box = FancyBboxPatch((0.45, 0.45), 0.15, 0.1,
                                boxstyle="round,pad=0.02",
                                facecolor=COLOR_CURRENT, alpha=0.8,
                                edgecolor='white', linewidth=3)
    ax.add_patch(context_box)
    ax.text(0.525, 0.5, 'Context\nVector', fontsize=11, ha='center', va='center',
           color='white', fontweight='bold')
    
    # Arrow from encoder to context
    ax.arrow(encoder_x + 0.18, 0.5, 0.12, 0,
            head_width=0.02, head_length=0.02, fc=COLOR_CONTEXT, alpha=0.5)
    
    # Decoder side
    decoder_x = 0.75
    decoder_words = ['Bonjour', 'monde', '!']
    
    for i, word in enumerate(decoder_words):
        y = 0.7 - i * 0.15
        # Decoder RNN cells
        decoder_cell = Circle((decoder_x - 0.05, y), 0.03,
                             facecolor=COLOR_PREDICT, alpha=0.5,
                             edgecolor='white', linewidth=2)
        ax.add_patch(decoder_cell)
        
        # Output boxes
        output_box = FancyBboxPatch((decoder_x + 0.02, y - 0.04), 0.12, 0.08,
                                   boxstyle="round,pad=0.01",
                                   facecolor=COLOR_PREDICT, alpha=0.7,
                                   edgecolor='white', linewidth=2)
        ax.add_patch(output_box)
        ax.text(decoder_x + 0.08, y, word, fontsize=10, ha='center', va='center',
               color='white', fontweight='bold')
        
        # Arrows
        ax.arrow(decoder_x - 0.02, y, 0.03, 0,
                head_width=0.015, head_length=0.01, fc='gray', alpha=0.5)
        
        # Context connection
        ax.plot([0.6, decoder_x - 0.08], [0.5, y], 'k--', alpha=0.3, linewidth=1)
    
    # Arrow from context to decoder
    ax.arrow(0.61, 0.5, 0.08, 0,
            head_width=0.02, head_length=0.02, fc=COLOR_CURRENT, alpha=0.5)
    
    # Labels
    ax.text(encoder_x, 0.85, 'ENCODER', fontsize=13, ha='center',
           fontweight='bold', color=COLOR_CONTEXT)
    ax.text(decoder_x + 0.05, 0.85, 'DECODER', fontsize=13, ha='center',
           fontweight='bold', color=COLOR_PREDICT)
    
    # Title
    ax.text(0.5, 0.95, 'Sequence-to-Sequence Architecture', fontsize=14,
           ha='center', fontweight='bold')
    
    # Information bottleneck annotation
    ax.annotate('Information\nBottleneck', xy=(0.525, 0.45), xytext=(0.525, 0.3),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=10, ha='center', color='red', fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/week4_encoder_decoder_flow.pdf', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()

def create_bleu_scores():
    """Create BLEU scores comparison visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: Model comparison
    models = ['RNN\nSeq2Seq', 'LSTM\nSeq2Seq', 'GRU\nSeq2Seq', 
              'Seq2Seq\n+Attention', 'Transformer']
    bleu_scores = [18.5, 24.3, 23.8, 31.2, 35.7]
    
    bars = ax1.bar(models, bleu_scores, color=[COLOR_NEUTRAL, COLOR_CONTEXT, 
                                                COLOR_CONTEXT, COLOR_CURRENT, COLOR_PREDICT],
                   alpha=0.7)
    
    # Add value labels
    for bar, score in zip(bars, bleu_scores):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{score}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('BLEU Score', fontsize=11, fontweight='bold')
    ax1.set_title('Translation Quality Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 40)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add breakthrough annotation
    ax1.annotate('Attention\nBreakthrough!', xy=(3, 31.2), xytext=(3.5, 25),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', color='red')
    
    # Right: Performance over sequence length
    sequence_lengths = [10, 20, 30, 40, 50, 60, 70]
    vanilla_performance = [28, 25, 20, 15, 10, 7, 5]
    attention_performance = [30, 29, 28, 26, 24, 22, 20]
    
    ax2.plot(sequence_lengths, vanilla_performance, 'o-', label='Without Attention',
            color=COLOR_NEUTRAL, linewidth=2, markersize=8)
    ax2.plot(sequence_lengths, attention_performance, 's-', label='With Attention',
            color=COLOR_CURRENT, linewidth=2, markersize=8)
    
    ax2.set_xlabel('Sequence Length', fontsize=11, fontweight='bold')
    ax2.set_ylabel('BLEU Score', fontsize=11, fontweight='bold')
    ax2.set_title('Performance vs Sequence Length', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add insight
    ax2.text(0.5, 0.15, 'Attention maintains quality\non long sequences',
            transform=ax2.transAxes, fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('../figures/week4_bleu_scores.pdf', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    create_attention_heatmap()
    create_encoder_decoder_flow()
    create_bleu_scores()
    print("Week 4 overview visualizations generated successfully!")