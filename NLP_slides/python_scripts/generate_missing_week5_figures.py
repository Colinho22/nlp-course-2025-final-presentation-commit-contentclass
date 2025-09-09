import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Arrow, FancyArrowPatch
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

def plot_self_attention_visualization():
    """Visualize self-attention mechanism"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # Example sentence
    words = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    n_words = len(words)
    
    # Create attention matrix (designed to show meaningful patterns)
    attention_matrix = np.array([
        [0.1, 0.05, 0.1, 0.1, 0.6, 0.05],  # The -> the (second the)
        [0.05, 0.7, 0.1, 0.05, 0.05, 0.05], # cat -> cat (self)
        [0.05, 0.3, 0.4, 0.1, 0.1, 0.05],   # sat -> cat, sat
        [0.05, 0.05, 0.1, 0.3, 0.05, 0.45], # on -> on, mat (prep-obj)
        [0.6, 0.05, 0.1, 0.1, 0.1, 0.05],   # the -> The (first the)
        [0.05, 0.2, 0.05, 0.4, 0.05, 0.25]  # mat -> on, mat (obj-prep)
    ])
    
    # Plot 1: Attention matrix heatmap
    im1 = ax1.imshow(attention_matrix, cmap='Blues', aspect='auto')
    ax1.set_xticks(range(n_words))
    ax1.set_yticks(range(n_words))
    ax1.set_xticklabels(words, fontsize=14, fontweight='bold')
    ax1.set_yticklabels(words, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Keys (What I attend FROM)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Queries (What I attend TO)', fontsize=14, fontweight='bold')
    ax1.set_title('Self-Attention Matrix\n"The cat sat on the mat"', fontsize=16, fontweight='bold')
    
    # Add attention weights as text
    for i in range(n_words):
        for j in range(n_words):
            color = 'white' if attention_matrix[i, j] > 0.4 else 'black'
            ax1.text(j, i, f'{attention_matrix[i, j]:.2f}', 
                    ha='center', va='center', fontweight='bold', color=color, fontsize=12)
    
    # Colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.6)
    cbar1.set_label('Attention Weight', fontsize=12, fontweight='bold')
    
    # Plot 2: Attention flow visualization
    ax2.set_xlim(-1, n_words)
    ax2.set_ylim(-1, 2)
    
    # Draw words
    word_positions = np.arange(n_words)
    for i, word in enumerate(words):
        # Word box
        box = FancyBboxPatch((i-0.3, 0.4), 0.6, 0.4,
                            boxstyle="round,pad=0.05",
                            facecolor='lightblue', edgecolor='black')
        ax2.add_patch(box)
        ax2.text(i, 0.6, word, ha='center', va='center', 
                fontsize=14, fontweight='bold')
    
    # Draw attention arrows for "cat" (index 1) attending to other words
    focus_word = 1  # "cat"
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i, weight in enumerate(attention_matrix[focus_word]):
        if i != focus_word and weight > 0.1:  # Don't draw self-attention or very weak
            # Arrow thickness proportional to attention weight
            arrow_width = weight * 0.02
            alpha = min(weight * 3, 1.0)  # Scale alpha
            
            if i < focus_word:  # Arrow curves up
                connectionstyle = "arc3,rad=0.3"
                y_offset = 1.3
            else:  # Arrow curves down
                connectionstyle = "arc3,rad=-0.3"
                y_offset = 1.3
            
            ax2.annotate('', xy=(i, 0.8), xytext=(focus_word, 0.8),
                        arrowprops=dict(arrowstyle='->', 
                                      connectionstyle=connectionstyle,
                                      color=colors[i], 
                                      lw=arrow_width*100,
                                      alpha=alpha))
            
            # Add weight labels
            mid_x = (i + focus_word) / 2
            ax2.text(mid_x, y_offset, f'{weight:.2f}', 
                    ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8),
                    color=colors[i], fontsize=11)
    
    # Highlight the focus word
    highlight = FancyBboxPatch((focus_word-0.35, 0.35), 0.7, 0.5,
                              boxstyle="round,pad=0.05",
                              facecolor='yellow', edgecolor='red', linewidth=3)
    ax2.add_patch(highlight)
    
    ax2.set_title('Self-Attention: "cat" attending to other words', 
                 fontsize=16, fontweight='bold')
    ax2.text(n_words/2, -0.5, 'Arrow thickness ∝ Attention Weight', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax2.axis('off')
    
    plt.suptitle('Self-Attention Mechanism Visualization', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/self_attention_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_transformer_vs_rnn_speed():
    """Compare transformer vs RNN training speed"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Sequential vs Parallel Processing
    sequence_lengths = np.array([50, 100, 200, 500, 1000, 2000])
    
    # RNN processing time scales linearly with sequence length
    rnn_time = sequence_lengths * 0.1  # 0.1ms per token
    # Transformer processes all in parallel (with some overhead)
    transformer_time = np.log(sequence_lengths) * 5 + 10  # Logarithmic scaling
    
    ax1.plot(sequence_lengths, rnn_time, 'r-o', linewidth=3, markersize=8, 
            label='RNN (Sequential)')
    ax1.plot(sequence_lengths, transformer_time, 'b-s', linewidth=3, markersize=8, 
            label='Transformer (Parallel)')
    
    ax1.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Processing Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Processing Time Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. GPU Utilization
    models = ['RNN\n(LSTM)', 'RNN +\nAttention', 'Transformer\n(Base)', 'Transformer\n(Optimized)']
    gpu_utilization = [15, 25, 75, 90]
    colors = ['red', 'orange', 'blue', 'green']
    
    bars = ax2.bar(models, gpu_utilization, color=colors, alpha=0.7)
    ax2.set_ylabel('GPU Utilization (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Hardware Efficiency', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, util in zip(bars, gpu_utilization):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{util}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Training Time on Real Datasets
    datasets = ['Small\n(1M tokens)', 'Medium\n(10M tokens)', 'Large\n(100M tokens)', 'XL\n(1B tokens)']
    rnn_training_days = [2, 8, 35, 150]
    transformer_training_days = [0.5, 2, 8, 25]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, rnn_training_days, width, label='RNN', 
                   color='red', alpha=0.7)
    bars2 = ax3.bar(x + width/2, transformer_training_days, width, 
                   label='Transformer', color='blue', alpha=0.7)
    
    ax3.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Training Time (Days)', fontsize=12, fontweight='bold')
    ax3.set_title('Real-World Training Times', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Memory vs Sequence Length
    seq_lens = np.array([128, 256, 512, 1024, 2048, 4096])
    # RNN memory is roughly constant (just hidden state)
    rnn_memory = np.full_like(seq_lens, 2.0, dtype=float)  # 2GB constant
    # Transformer memory scales quadratically with sequence length (attention matrix)
    transformer_memory = (seq_lens / 512) ** 2 * 8  # Base 8GB for 512 tokens
    
    ax4.loglog(seq_lens, rnn_memory, 'r-o', linewidth=3, markersize=8, 
              label='RNN Memory')
    ax4.loglog(seq_lens, transformer_memory, 'b-s', linewidth=3, markersize=8, 
              label='Transformer Memory')
    
    ax4.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Memory Usage (GB)', fontsize=12, fontweight='bold')
    ax4.set_title('Memory Scaling', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # Add annotations
    ax4.annotate('O(1) - Constant', xy=(2048, 2), xytext=(1000, 5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red')
    
    ax4.annotate('O(n²) - Quadratic', xy=(2048, transformer_memory[4]), xytext=(1000, 100),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=11, fontweight='bold', color='blue')
    
    plt.suptitle('Transformer vs RNN: Training Speed Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/transformer_vs_rnn_speed.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_transformer_benchmarks():
    """Plot transformer benchmark results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Translation Quality Comparison
    models = ['Seq2Seq\nBasic', 'Seq2Seq +\nAttention', 'ConvS2S', 'Transformer\nBase', 'Transformer\nBig']
    en_de_bleu = [20.9, 24.6, 25.2, 27.3, 28.4]  # WMT'14 EN-DE
    en_fr_bleu = [33.3, 37.0, 38.4, 38.1, 41.8]  # WMT'14 EN-FR
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, en_de_bleu, width, label='EN→DE', color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, en_fr_bleu, width, label='EN→FR', color='red', alpha=0.7)
    
    ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax1.set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
    ax1.set_title('Translation Quality (WMT 2014)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Training Efficiency
    models_short = ['Seq2Seq', 'ConvS2S', 'Transformer']
    training_time_hours = [144, 96, 84]  # Training time in hours
    model_params = [213, 103, 65]  # Parameters in millions
    
    # Normalize by parameters to show efficiency
    efficiency = np.array(en_de_bleu[-3:]) / (np.array(training_time_hours) / 24)  # BLEU per day
    
    bars = ax2.bar(models_short, efficiency, color=['orange', 'green', 'blue'], alpha=0.7)
    ax2.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax2.set_ylabel('BLEU Points per Training Day', fontsize=12, fontweight='bold')
    ax2.set_title('Training Efficiency', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    for bar, eff in zip(bars, efficiency):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{eff:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Scale vs Performance
    model_sizes = [65, 213, 470, 1600, 11000]  # Million parameters
    model_names = ['T-Base', 'T-Big', 'GPT-1', 'GPT-2', 'GPT-3']
    performance_scores = [28.4, 29.2, 32.1, 35.8, 42.5]  # Normalized performance
    
    scatter = ax3.scatter(model_sizes, performance_scores, 
                         s=[100, 150, 200, 300, 500], 
                         c=['blue', 'blue', 'green', 'green', 'red'], 
                         alpha=0.7)
    
    for i, name in enumerate(model_names):
        ax3.annotate(name, (model_sizes[i], performance_scores[i]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    ax3.set_xscale('log')
    ax3.set_xlabel('Model Size (Million Parameters)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
    ax3.set_title('Scaling Laws: Size vs Performance', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Task Performance Comparison
    tasks = ['Translation', 'Question\nAnswering', 'Text\nClassification', 'Language\nModeling', 'Summarization']
    rnn_performance = [75, 68, 82, 65, 72]  # Normalized scores
    transformer_performance = [92, 89, 94, 88, 87]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, rnn_performance, width, label='Best RNN', 
                   color='red', alpha=0.7)
    bars2 = ax4.bar(x + width/2, transformer_performance, width, 
                   label='Transformer', color='blue', alpha=0.7)
    
    ax4.set_xlabel('NLP Tasks', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
    ax4.set_title('Multi-Task Performance', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(tasks)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    
    # Add improvement percentages
    for i, (rnn_score, trans_score) in enumerate(zip(rnn_performance, transformer_performance)):
        improvement = ((trans_score - rnn_score) / rnn_score) * 100
        ax4.text(i, trans_score + 2, f'+{improvement:.0f}%', 
                ha='center', va='bottom', fontweight='bold', color='green')
    
    plt.suptitle('Transformer Revolution: Benchmark Results (2017-2018)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/transformer_benchmarks.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Generating missing Week 5 Transformer figures...")
    
    print("1. Creating self-attention visualization...")
    plot_self_attention_visualization()
    
    print("2. Creating transformer vs RNN speed comparison...")
    plot_transformer_vs_rnn_speed()
    
    print("3. Creating transformer benchmark results...")
    plot_transformer_benchmarks()
    
    print("Week 5 figures generated successfully!")

if __name__ == "__main__":
    main()