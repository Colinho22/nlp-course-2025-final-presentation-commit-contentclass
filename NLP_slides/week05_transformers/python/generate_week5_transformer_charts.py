import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrow, ConnectionPatch
import seaborn as sns
import os
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# 1. Self-Attention Visualization
def plot_self_attention_visualization():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sentence
    words = ['The', 'student', 'who', 'studied', 'hard', 'passed']
    positions = np.linspace(0.1, 0.9, len(words))
    
    # Draw words
    for i, (word, pos) in enumerate(zip(words, positions)):
        # Word box
        word_box = FancyBboxPatch((pos-0.06, 0.4), 0.12, 0.1,
                                  boxstyle="round,pad=0.02",
                                  facecolor='lightblue',
                                  edgecolor='black',
                                  linewidth=2)
        ax.add_patch(word_box)
        ax.text(pos, 0.45, word, ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Self-attention connections (showing key relationships)
    connections = [
        (5, 1, 0.9, 'passed → student'),  # passed attends to student
        (3, 1, 0.7, 'studied → student'),  # studied attends to student
        (4, 3, 0.8, 'hard → studied'),     # hard attends to studied
        (2, 1, 0.6, 'who → student'),      # who attends to student
        (1, 0, 0.5, 'student → The'),      # student attends to The
    ]
    
    # Draw attention connections
    for src_idx, tgt_idx, weight, label in connections:
        src_pos = positions[src_idx]
        tgt_pos = positions[tgt_idx]
        
        # Curved arrow
        style = "arc3,rad=.3" if src_idx > tgt_idx else "arc3,rad=-.3"
        arrow = mpatches.FancyArrowPatch((src_pos, 0.5), (tgt_pos, 0.5),
                                        connectionstyle=style,
                                        arrowstyle='->,head_width=0.4,head_length=0.2',
                                        color='red',
                                        alpha=weight,
                                        linewidth=2*weight)
        ax.add_patch(arrow)
    
    # Add attention weight legend
    ax.text(0.5, 0.8, 'Self-Attention: Every Word Attends to Relevant Words',
            ha='center', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.75, 'All connections computed in parallel!',
            ha='center', fontsize=12, style='italic', color='green')
    
    # Key insight boxes
    ax.text(0.2, 0.2, '"passed" knows it refers\nto "student", not "hard"',
            ha='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    ax.text(0.8, 0.2, 'No sequential processing -\nall at once!',
            ha='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 0.85)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/self_attention_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Transformer vs RNN Speed Comparison
def plot_transformer_vs_rnn_speed():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Processing time vs sequence length
    seq_lengths = np.array([10, 50, 100, 200, 500, 1000])
    rnn_time = seq_lengths * 0.1  # Linear with sequence length
    transformer_time = np.ones_like(seq_lengths) * 0.5  # Constant (parallel)
    
    ax1.plot(seq_lengths, rnn_time, 'o-', label='RNN (Sequential)', 
             color='red', linewidth=3, markersize=8)
    ax1.plot(seq_lengths, transformer_time, 's-', label='Transformer (Parallel)', 
             color='blue', linewidth=3, markersize=8)
    
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Processing Time (relative)', fontsize=12)
    ax1.set_title('Processing Time Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1100)
    
    # Add annotations
    ax1.annotate('Linear growth\n(sequential bottleneck)', 
                xy=(500, 50), xytext=(600, 70),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    ax1.annotate('Constant time\n(full parallelization)', 
                xy=(500, 0.5), xytext=(600, 10),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue')
    
    # Right: GPU utilization
    models = ['RNN/LSTM', 'Transformer']
    gpu_util = [15, 90]
    training_days = [21, 3.5]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, gpu_util, width, label='GPU Utilization (%)', color='green')
    bars2 = ax2.bar(x + width/2, training_days, width, label='Training Days', color='orange')
    
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Training Efficiency Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height}%', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height} days', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Why Transformers Train Faster: Parallelization Advantage', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/transformer_vs_rnn_speed.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Transformer Benchmark Results
def plot_transformer_benchmarks():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Benchmark data
    tasks = ['WMT14\nEN-DE', 'WMT14\nEN-FR', 'WMT17\nEN-DE', 'GLUE\nScore', 'SQuAD\nF1']
    
    # Previous SOTA and Transformer results
    previous_sota = [25.2, 37.0, 26.3, 72.8, 84.6]
    transformer = [28.4, 41.8, 28.3, 82.1, 90.9]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, previous_sota, width, label='Previous SOTA', color='gray')
    bars2 = ax.bar(x + width/2, transformer, width, label='Transformer (2017)', color='#2ecc71')
    
    # Add improvement percentages
    for i, (prev, trans) in enumerate(zip(previous_sota, transformer)):
        improvement = ((trans - prev) / prev) * 100
        ax.text(i, max(prev, trans) + 1, f'+{improvement:.1f}%', 
                ha='center', fontsize=10, fontweight='bold', color='#2ecc71')
    
    # Labels and title
    ax.set_xlabel('Benchmark Task', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Transformer: Immediate State-of-the-Art on All Tasks\n"Attention Is All You Need" (2017)', 
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 100)
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add note
    ax.text(0.5, 0.02, 'Higher is better for all metrics. Transformer achieved SOTA on every benchmark at launch.',
            transform=ax.transAxes, ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    plt.tight_layout()
    plt.savefig('../figures/transformer_benchmarks.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Transformer Architecture Diagram
def plot_transformer_architecture():
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Component dimensions
    comp_width = 0.6
    comp_height = 0.08
    
    # Bottom to top components
    components = [
        ('Input Embedding', 0.1, 'lightblue'),
        ('+ Positional Encoding', 0.2, 'lightcyan'),
        ('Multi-Head Attention', 0.35, 'lightcoral'),
        ('Add & Norm', 0.45, 'lightgray'),
        ('Feed Forward', 0.55, 'lightgreen'),
        ('Add & Norm', 0.65, 'lightgray'),
        ('Output Probabilities', 0.85, 'gold')
    ]
    
    # Draw components
    for i, (name, y, color) in enumerate(components):
        if i < 6:  # Stack 6 layers
            # Draw component
            rect = FancyBboxPatch((0.2, y), comp_width, comp_height,
                                 boxstyle="round,pad=0.02",
                                 facecolor=color,
                                 edgecolor='black',
                                 linewidth=2)
            ax.add_patch(rect)
            ax.text(0.5, y + comp_height/2, name, ha='center', va='center', 
                   fontsize=11, fontweight='bold')
            
            # Residual connections
            if 'Add & Norm' in name:
                # Draw residual arrow
                ax.annotate('', xy=(0.15, y - 0.15), xytext=(0.15, y + comp_height/2),
                           arrowprops=dict(arrowstyle='->', lw=2, color='red'))
                ax.text(0.1, y - 0.075, 'Residual', rotation=90, 
                       ha='center', va='center', fontsize=8, color='red')
    
    # Draw "6x" to indicate stacking
    ax.text(0.9, 0.5, '6×', fontsize=24, fontweight='bold', 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
    
    # Arrows between components
    arrow_positions = [0.18, 0.28, 0.43, 0.53, 0.63, 0.73]
    for y in arrow_positions:
        ax.arrow(0.5, y, 0, 0.05, head_width=0.03, head_length=0.02, 
                fc='black', ec='black')
    
    # Input and output
    ax.text(0.5, 0.05, 'Input: "The cat sat"', ha='center', fontsize=12, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    ax.text(0.5, 0.95, 'Output: P(next word)', ha='center', fontsize=12,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    # Title
    ax.text(0.5, 1.05, 'Transformer Architecture', ha='center', 
           fontsize=18, fontweight='bold', transform=ax.transAxes)
    
    # Key features annotations
    ax.text(0.05, 0.35, 'Parallel\nProcessing', fontsize=10, 
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen'),
           ha='center')
    ax.text(0.95, 0.55, 'No\nRecurrence', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral'),
           ha='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/transformer_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Positional Encoding Visualization
def plot_positional_encoding():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Generate positional encodings
    max_len = 50
    d_model = 128
    
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    # Plot heatmap of positional encodings
    im = ax1.imshow(pe.T, aspect='auto', cmap='RdBu')
    ax1.set_xlabel('Position in Sequence', fontsize=12)
    ax1.set_ylabel('Encoding Dimension', fontsize=12)
    ax1.set_title('Positional Encoding Pattern (Sinusoidal)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Encoding Value', fontsize=10)
    
    # Plot specific positions
    positions_to_plot = [0, 10, 20, 30, 40]
    colors = plt.cm.viridis(np.linspace(0, 1, len(positions_to_plot)))
    
    for pos, color in zip(positions_to_plot, colors):
        ax2.plot(pe[pos, :50], label=f'Position {pos}', color=color, linewidth=2)
    
    ax2.set_xlabel('Encoding Dimension', fontsize=12)
    ax2.set_ylabel('Encoding Value', fontsize=12)
    ax2.set_title('Positional Encoding for Different Positions', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add explanation
    ax2.text(0.5, -0.15, 'Each position gets a unique "fingerprint" that the model can use to understand word order',
             transform=ax2.transAxes, ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    plt.suptitle('Positional Encoding: Giving Transformers a Sense of Order', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/positional_encoding.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all Week 5 visualizations
if __name__ == "__main__":
    print("Generating Week 5 Transformer visualizations...")
    plot_self_attention_visualization()
    print("- Self-attention visualization created")
    plot_transformer_vs_rnn_speed()
    print("- Speed comparison visualization created")
    plot_transformer_benchmarks()
    print("- Benchmark results visualization created")
    plot_transformer_architecture()
    print("- Architecture diagram created")
    plot_positional_encoding()
    print("- Positional encoding visualization created")
    print("\nWeek 5 visualizations completed!")