"""
Generate figures for Week 5 restructured BSc presentation on Transformers
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Educational color scheme
COLOR_CURRENT = '#FF6B6B'  # Red - current position/focus
COLOR_CONTEXT = '#4ECDC4'  # Teal - context/surrounding
COLOR_PREDICT = '#95E77E'  # Green - predictions/output
COLOR_NEUTRAL = '#E0E0E0'  # Gray - neutral elements
COLOR_HIGHLIGHT = '#FFD93D'  # Yellow - highlights

def generate_gpu_utilization_comparison():
    """GPU utilization comparison between RNN and Transformer"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # RNN GPU utilization
    time_steps = 10
    gpu_cores = 8
    rnn_utilization = np.zeros((gpu_cores, time_steps))
    for t in range(time_steps):
        rnn_utilization[0, t] = 1  # Only one GPU core active at each time step
    
    im1 = ax1.imshow(rnn_utilization, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax1.set_title('RNN: Sequential Processing\n(Low GPU Utilization)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Steps', fontsize=12)
    ax1.set_ylabel('GPU Cores', fontsize=12)
    ax1.set_xticks(range(time_steps))
    ax1.set_yticks(range(gpu_cores))
    ax1.set_xticklabels(range(1, time_steps+1))
    ax1.set_yticklabels([f'Core {i+1}' for i in range(gpu_cores)])
    
    # Add text annotation
    ax1.text(time_steps/2, gpu_cores + 0.5, 'Only 1 core active at a time!', 
             ha='center', fontsize=11, color=COLOR_CURRENT, fontweight='bold')
    
    # Transformer GPU utilization
    transformer_utilization = np.ones((gpu_cores, time_steps))  # All cores active
    
    im2 = ax2.imshow(transformer_utilization, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax2.set_title('Transformer: Parallel Processing\n(Full GPU Utilization)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Steps', fontsize=12)
    ax2.set_ylabel('GPU Cores', fontsize=12)
    ax2.set_xticks(range(time_steps))
    ax2.set_yticks(range(gpu_cores))
    ax2.set_xticklabels(range(1, time_steps+1))
    ax2.set_yticklabels([f'Core {i+1}' for i in range(gpu_cores)])
    
    # Add text annotation
    ax2.text(time_steps/2, gpu_cores + 0.5, 'All cores active simultaneously!', 
             ha='center', fontsize=11, color=COLOR_PREDICT, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im2, ax=[ax1, ax2], orientation='horizontal', pad=0.1, fraction=0.05)
    cbar.set_label('GPU Core Utilization', fontsize=12)
    
    plt.suptitle('GPU Efficiency: Why Transformers Train Faster', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig('../figures/gpu_utilization_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_rnn_sequential_bottleneck():
    """Visualize RNN sequential processing bottleneck"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    words = ['The', 'student', 'who', 'studied', 'hard', 'passed']
    positions = np.arange(len(words))
    
    # Draw boxes for each word
    for i, (pos, word) in enumerate(zip(positions, words)):
        # Draw word box
        rect = FancyBboxPatch((pos - 0.4, 1), 0.8, 0.5,
                               boxstyle="round,pad=0.1",
                               facecolor=COLOR_NEUTRAL if i > 0 else COLOR_CURRENT,
                               edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(pos, 1.25, word, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Draw hidden state box
        rect2 = FancyBboxPatch((pos - 0.3, 0), 0.6, 0.4,
                                boxstyle="round,pad=0.05",
                                facecolor=COLOR_NEUTRAL if i > 0 else COLOR_CONTEXT,
                                edgecolor='gray', linewidth=1)
        ax.add_patch(rect2)
        ax.text(pos, 0.2, f'h{i+1}', ha='center', va='center', fontsize=10)
        
        # Draw sequential arrows
        if i < len(words) - 1:
            arrow = FancyArrowPatch((pos + 0.3, 0.2), (pos + 0.7, 0.2),
                                    connectionstyle="arc3", 
                                    arrowstyle='->', mutation_scale=20,
                                    color='red', linewidth=2)
            ax.add_patch(arrow)
            
            # Add waiting indicator
            ax.text(pos + 0.5, -0.3, 'WAIT', ha='center', fontsize=9, 
                   color='red', fontweight='bold', rotation=0)
    
    # Add bottleneck annotation
    ax.annotate('Sequential Bottleneck:\nEach step must wait for the previous one!',
                xy=(2.5, 0.2), xytext=(2.5, -1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, ha='center', color='red', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.7))
    
    ax.set_xlim(-0.5, len(words) - 0.5)
    ax.set_ylim(-1.5, 2)
    ax.axis('off')
    ax.set_title('RNN: The Sequential Processing Bottleneck', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('../figures/rnn_sequential_bottleneck.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_qkv_visualization():
    """Visualize Query, Key, Value mechanism"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Words in sentence
    words = ['The', 'cat', 'sat', 'on', 'mat']
    
    # Draw the main word (Query)
    query_word = 'sat'
    query_pos = 2
    
    # Draw Query box
    query_box = FancyBboxPatch((4, 6), 2, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor=COLOR_CURRENT,
                                edgecolor='black', linewidth=3)
    ax.add_patch(query_box)
    ax.text(5, 6.6, 'Query: "sat"', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='white')
    ax.text(5, 6.2, '(Looking for...)', ha='center', va='center', 
           fontsize=10, style='italic', color='white')
    
    # Draw Key-Value pairs
    y_positions = [4, 4, 4, 4, 4]
    x_positions = [0, 2, 4, 6, 8]
    attention_scores = [0.1, 0.7, 0.1, 0.05, 0.05]  # Attention weights
    
    for i, (word, x, y, score) in enumerate(zip(words, x_positions, y_positions, attention_scores)):
        # Key box
        key_box = FancyBboxPatch((x - 0.8, y), 1.6, 0.8,
                                  boxstyle="round,pad=0.05",
                                  facecolor=COLOR_CONTEXT,
                                  edgecolor='black', linewidth=2,
                                  alpha=0.7)
        ax.add_patch(key_box)
        ax.text(x, y + 0.6, f'Key: "{word}"', ha='center', va='center', 
               fontsize=11, fontweight='bold')
        ax.text(x, y + 0.2, f'Match: {score:.0%}', ha='center', va='center', 
               fontsize=9, style='italic')
        
        # Value box
        value_box = FancyBboxPatch((x - 0.8, y - 1.5), 1.6, 0.8,
                                    boxstyle="round,pad=0.05",
                                    facecolor=COLOR_PREDICT,
                                    edgecolor='black', linewidth=2,
                                    alpha=score)  # Alpha based on attention score
        ax.add_patch(value_box)
        ax.text(x, y - 0.9, f'Value: {word}', ha='center', va='center', 
               fontsize=11, fontweight='bold')
        
        # Draw attention arrows
        arrow_width = max(1, score * 10)
        arrow = FancyArrowPatch((5, 6), (x, y + 0.8),
                                connectionstyle="arc3,rad=0.3",
                                arrowstyle='->', mutation_scale=20,
                                color='red', linewidth=arrow_width,
                                alpha=max(0.3, score))
        ax.add_patch(arrow)
    
    # Add explanation boxes
    ax.text(5, 8, 'Self-Attention: How Words Look at Each Other', 
           ha='center', fontsize=16, fontweight='bold')
    
    # Legend
    ax.text(10, 6.5, 'Q: What am I looking for?', fontsize=10, color=COLOR_CURRENT, fontweight='bold')
    ax.text(10, 6, 'K: What do I offer?', fontsize=10, color=COLOR_CONTEXT, fontweight='bold')
    ax.text(10, 5.5, 'V: My actual information', fontsize=10, color=COLOR_PREDICT, fontweight='bold')
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(1, 8.5)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/qkv_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_position_problem_visual():
    """Visualize the position problem in self-attention"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Without position encoding
    words1 = ['Dog', 'bites', 'man']
    words2 = ['Man', 'bites', 'dog']
    
    for ax, words, title in [(ax1, words1, 'Without Position: Same Attention Pattern!'),
                              (ax2, words2, 'With Position: Different Patterns!')]:
        # Draw word boxes
        for i, word in enumerate(words):
            color = COLOR_CURRENT if word == 'bites' else COLOR_NEUTRAL
            rect = FancyBboxPatch((i * 2.5, 0), 1.8, 0.8,
                                   boxstyle="round,pad=0.1",
                                   facecolor=color,
                                   edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(i * 2.5 + 0.9, 0.4, word, ha='center', va='center',
                   fontsize=12, fontweight='bold',
                   color='white' if word == 'bites' else 'black')
            
            if ax == ax2:
                # Add position encoding visualization
                pos_rect = FancyBboxPatch((i * 2.5, -0.5), 1.8, 0.3,
                                          boxstyle="round,pad=0.05",
                                          facecolor=COLOR_HIGHLIGHT,
                                          edgecolor='gray', linewidth=1)
                ax.add_patch(pos_rect)
                ax.text(i * 2.5 + 0.9, -0.35, f'Pos {i+1}', ha='center', va='center',
                       fontsize=9, style='italic')
        
        ax.set_xlim(-0.5, 6)
        ax.set_ylim(-1 if ax == ax2 else -0.5, 1.5)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    # Add main title
    fig.suptitle('The Position Problem: Order Matters!', fontsize=16, fontweight='bold')
    
    # Add explanation
    ax1.text(2.5, 1.2, 'Self-attention treats these as identical!', 
            ha='center', fontsize=11, color='red', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    ax2.text(2.5, 1.2, 'Position encoding preserves word order!', 
            ha='center', fontsize=11, color='green', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('../figures/position_problem_visual.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_positional_encoding_viz():
    """Visualize positional encoding patterns"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Generate positional encodings
    def get_positional_encoding(seq_len, d_model):
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe
    
    # Small example for understanding
    pe_small = get_positional_encoding(10, 8)
    im1 = ax1.imshow(pe_small.T, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax1.set_title('Positional Encoding Pattern\n(10 positions, 8 dimensions)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Position in Sequence', fontsize=11)
    ax1.set_ylabel('Encoding Dimension', fontsize=11)
    ax1.set_xticks(range(10))
    ax1.set_xticklabels(range(1, 11))
    
    # Larger example
    pe_large = get_positional_encoding(50, 64)
    im2 = ax2.imshow(pe_large[:, :32].T, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax2.set_title('Larger Positional Encoding\n(50 positions, first 32 dimensions)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Position in Sequence', fontsize=11)
    ax2.set_ylabel('Encoding Dimension', fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(im2, ax=[ax1, ax2], orientation='vertical', pad=0.02, fraction=0.05)
    cbar.set_label('Encoding Value', fontsize=11)
    
    # Add explanation
    fig.text(0.5, 0.02, 'Each position gets a unique "fingerprint" using sine and cosine waves', 
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLOR_HIGHLIGHT, alpha=0.7))
    
    plt.suptitle('Positional Encoding: Teaching Order to Transformers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/positional_encoding_viz.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_transformer_architecture_detailed():
    """Generate detailed transformer architecture diagram"""
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Define layer positions
    layers = [
        ('Input Embeddings', 0, COLOR_NEUTRAL),
        ('+ Positional Encoding', 1, COLOR_HIGHLIGHT),
        ('Multi-Head Attention', 2.5, COLOR_CURRENT),
        ('Add & Norm', 3.5, COLOR_NEUTRAL),
        ('Feed Forward', 4.5, COLOR_CONTEXT),
        ('Add & Norm', 5.5, COLOR_NEUTRAL),
        ('Multi-Head Attention', 7, COLOR_CURRENT),
        ('Add & Norm', 8, COLOR_NEUTRAL),
        ('Feed Forward', 9, COLOR_CONTEXT),
        ('Add & Norm', 10, COLOR_NEUTRAL),
        ('Output Layer', 11.5, COLOR_PREDICT)
    ]
    
    # Draw layers
    for name, y, color in layers:
        rect = FancyBboxPatch((1, y), 8, 0.8,
                               boxstyle="round,pad=0.1",
                               facecolor=color,
                               edgecolor='black', linewidth=2,
                               alpha=0.7)
        ax.add_patch(rect)
        ax.text(5, y + 0.4, name, ha='center', va='center',
               fontsize=12, fontweight='bold',
               color='white' if color == COLOR_CURRENT else 'black')
    
    # Draw connections
    for i in range(len(layers) - 1):
        ax.arrow(5, layers[i][1] + 0.8, 0, layers[i+1][1] - layers[i][1] - 0.8,
                head_width=0.2, head_length=0.1, fc='gray', ec='gray')
    
    # Draw residual connections
    # First residual
    ax.annotate('', xy=(9.5, 3.5), xytext=(9.5, 2.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text(10, 3, 'Residual', fontsize=9, color='red', rotation=90, va='center')
    
    # Second residual
    ax.annotate('', xy=(9.5, 5.5), xytext=(9.5, 4.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text(10, 5, 'Residual', fontsize=9, color='red', rotation=90, va='center')
    
    # Add "Nx" indicator
    ax.add_patch(Rectangle((0.5, 2), 9, 4.5, fill=False, edgecolor='blue', 
                           linewidth=3, linestyle='--'))
    ax.text(0.2, 4.25, 'Nx', fontsize=16, fontweight='bold', color='blue', rotation=90, va='center')
    
    # Add title and labels
    ax.text(5, 13, 'Transformer Architecture', fontsize=16, fontweight='bold', ha='center')
    ax.text(5, 12.5, '(Simplified for BSc Understanding)', fontsize=11, ha='center', style='italic')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 13.5)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/transformer_architecture_detailed.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_layer_by_layer_transformation():
    """Show how information transforms through layers"""
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    
    stages = [
        ('Input\n"The cat"', ['The', 'cat'], [0.2, 0.3]),
        ('After Attention\n(Context Added)', ['The+context', 'cat+context'], [0.5, 0.7]),
        ('After FFN\n(Processed)', ['The_processed', 'cat_processed'], [0.7, 0.9]),
        ('Final Output\n(Ready for next)', ['The_final', 'cat_final'], [0.9, 1.0])
    ]
    
    for ax, (title, labels, values) in zip(axes, stages):
        # Create bars
        colors = [COLOR_CURRENT if i == 1 else COLOR_NEUTRAL for i in range(len(labels))]
        bars = ax.bar(range(len(labels)), values, color=colors, edgecolor='black', linewidth=2)
        
        # Customize
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Information Richness', fontsize=10)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.1f}', ha='center', fontsize=9)
    
    # Add arrows between stages
    for i in range(3):
        fig.add_artist(ConnectionPatch((1, 0.5), (0, 0.5), "axes fraction", "axes fraction",
                                      axesA=axes[i], axesB=axes[i+1],
                                      arrowstyle="->", lw=2, color='green'))
    
    plt.suptitle('Information Transformation Through Transformer Layers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/layer_by_layer_transformation.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_transformer_vs_rnn_performance():
    """Performance comparison chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training time comparison
    models = ['RNN', 'LSTM', 'Transformer']
    times = [100, 120, 25]  # Relative training time
    colors = [COLOR_CURRENT, COLOR_CONTEXT, COLOR_PREDICT]
    
    bars1 = ax1.bar(models, times, color=colors, edgecolor='black', linewidth=2)
    ax1.set_title('Training Time Comparison\n(Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Relative Training Time', fontsize=11)
    ax1.set_ylim(0, 140)
    
    # Add value labels
    for bar, time in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{time}h', ha='center', fontsize=11, fontweight='bold')
    
    # Performance comparison
    sequence_lengths = [10, 50, 100, 200, 500]
    rnn_performance = [95, 85, 70, 50, 30]
    transformer_performance = [92, 90, 88, 85, 80]
    
    ax2.plot(sequence_lengths, rnn_performance, 'o-', color=COLOR_CURRENT, 
            linewidth=2, markersize=8, label='RNN/LSTM')
    ax2.plot(sequence_lengths, transformer_performance, 's-', color=COLOR_PREDICT,
            linewidth=2, markersize=8, label='Transformer')
    
    ax2.set_title('Long Sequence Performance\n(Higher is Better)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Sequence Length', fontsize=11)
    ax2.set_ylabel('Performance (%)', fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Add annotation
    ax2.annotate('Transformers maintain\nperformance better!',
                xy=(300, 82), xytext=(350, 60),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, ha='center', color='green', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    plt.suptitle('Why Transformers Beat RNNs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/transformer_vs_rnn_performance.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all figures
if __name__ == "__main__":
    print("Generating Week 5 restructured figures...")
    
    generate_gpu_utilization_comparison()
    print("[DONE] GPU utilization comparison")
    
    generate_rnn_sequential_bottleneck()
    print("[DONE] RNN sequential bottleneck")
    
    generate_qkv_visualization()
    print("[DONE] Query-Key-Value visualization")
    
    generate_position_problem_visual()
    print("[DONE] Position problem visualization")
    
    generate_positional_encoding_viz()
    print("[DONE] Positional encoding visualization")
    
    generate_transformer_architecture_detailed()
    print("[DONE] Transformer architecture detailed")
    
    generate_layer_by_layer_transformation()
    print("[DONE] Layer-by-layer transformation")
    
    generate_transformer_vs_rnn_performance()
    print("[DONE] Transformer vs RNN performance")
    
    print("\nAll figures generated successfully!")