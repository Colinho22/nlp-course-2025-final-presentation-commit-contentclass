"""
Generate Week 5 overview visualizations for Transformers
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Educational colors
COLOR_CURRENT = '#FF6B6B'  # Red
COLOR_CONTEXT = '#4ECDC4'  # Teal  
COLOR_PREDICT = '#95E77E'  # Green
COLOR_NEUTRAL = '#E0E0E0'  # Gray

def create_attention_matrix():
    """Create multi-head attention visualization matrix"""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle('Multi-Head Attention: 8 Parallel Attention Heads', fontsize=14, fontweight='bold')
    
    sentence = ['The', 'cat', 'sat', 'on', 'mat']
    
    for idx, ax in enumerate(axes.flat):
        # Generate different attention patterns for each head
        np.random.seed(42 + idx)
        
        if idx == 0:  # Head focuses on adjacent words
            attention = np.eye(5) * 0.8 + np.eye(5, k=1) * 0.4 + np.eye(5, k=-1) * 0.4
        elif idx == 1:  # Head focuses on first word
            attention = np.random.rand(5, 5) * 0.3
            attention[:, 0] = 0.8
        elif idx == 2:  # Head focuses on last word
            attention = np.random.rand(5, 5) * 0.3
            attention[:, -1] = 0.8
        else:  # Random patterns
            attention = np.random.rand(5, 5)
        
        # Normalize
        attention = attention / attention.sum(axis=1, keepdims=True)
        
        im = ax.imshow(attention, cmap='Blues', vmin=0, vmax=0.5)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(sentence, fontsize=8)
        ax.set_yticklabels(sentence, fontsize=8)
        ax.set_title(f'Head {idx+1}', fontsize=10)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Add colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', 
                 fraction=0.05, pad=0.1, label='Attention Weight')
    
    plt.tight_layout()
    plt.savefig('../figures/week5_attention_matrix.pdf', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()

def create_efficiency_comparison():
    """Create computational efficiency comparison between RNN and Transformer"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: Parallelization comparison
    sequence_lengths = [10, 50, 100, 200, 500, 1000]
    rnn_time = [1, 5, 10, 20, 50, 100]  # Linear with sequence length
    transformer_time = [1, 1.5, 2, 2.5, 3, 3.5]  # Constant due to parallelization
    
    ax1.plot(sequence_lengths, rnn_time, 'o-', label='RNN (Sequential)',
            color=COLOR_CURRENT, linewidth=2, markersize=8)
    ax1.plot(sequence_lengths, transformer_time, 's-', label='Transformer (Parallel)',
            color=COLOR_PREDICT, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Sequence Length', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Training Time (relative)', fontsize=11, fontweight='bold')
    ax1.set_title('Training Speed Comparison', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add speedup annotation
    ax1.annotate('28x faster\nat length 1000', xy=(1000, 3.5), xytext=(500, 10),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, fontweight='bold', color='green')
    
    # Right: Memory usage comparison
    model_sizes = ['Small\n(1M)', 'Medium\n(10M)', 'Large\n(100M)', 'XL\n(1B)']
    rnn_memory = [1, 10, 100, 1000]
    transformer_memory = [2, 15, 120, 1100]
    
    x = np.arange(len(model_sizes))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, rnn_memory, width, label='RNN',
                   color=COLOR_CURRENT, alpha=0.7)
    bars2 = ax2.bar(x + width/2, transformer_memory, width, label='Transformer',
                   color=COLOR_PREDICT, alpha=0.7)
    
    ax2.set_xlabel('Model Size', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Memory Usage (GB)', fontsize=11, fontweight='bold')
    ax2.set_title('Memory Requirements', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_sizes)
    ax2.set_yscale('log')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('../figures/week5_efficiency_comparison.pdf', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()

def create_benchmark_timeline():
    """Create benchmark performance timeline"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Timeline data
    years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
    models = ['Transformer', 'BERT', 'GPT-2', 'GPT-3', 'T5', 'PaLM', 'GPT-4']
    glue_scores = [72.8, 82.1, 85.4, 88.5, 90.3, 92.5, 94.7]
    model_sizes = [65, 340, 1500, 175000, 11000, 540000, 1700000]  # In millions
    
    # Normalize model sizes for bubble size
    bubble_sizes = np.log10(np.array(model_sizes) + 1) * 100
    
    # Create scatter plot
    scatter = ax.scatter(years, glue_scores, s=bubble_sizes, 
                        c=range(len(years)), cmap='viridis',
                        alpha=0.6, edgecolors='white', linewidth=2)
    
    # Add model labels
    for i, (year, score, model) in enumerate(zip(years, glue_scores, models)):
        ax.annotate(model, (year, score), xytext=(0, 10),
                   textcoords='offset points', ha='center',
                   fontsize=9, fontweight='bold')
    
    # Add trend line
    z = np.polyfit(years, glue_scores, 2)
    p = np.poly1d(z)
    years_smooth = np.linspace(2017, 2023, 100)
    ax.plot(years_smooth, p(years_smooth), 'r--', alpha=0.5, linewidth=2)
    
    # Add human performance line
    ax.axhline(y=87.1, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(2017.5, 87.5, 'Human Performance', fontsize=10, color='green', fontweight='bold')
    
    ax.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax.set_ylabel('GLUE Benchmark Score', fontsize=11, fontweight='bold')
    ax.set_title('Transformer Models: Rapid Progress on NLP Benchmarks', 
                fontsize=13, fontweight='bold')
    ax.set_xlim(2016.5, 2023.5)
    ax.set_ylim(70, 96)
    ax.grid(True, alpha=0.3)
    
    # Add legend for bubble size
    legend_sizes = [10, 1000, 100000, 1000000]
    legend_labels = ['10M', '1B', '100B', '1T']
    for size, label in zip(legend_sizes, legend_labels):
        ax.scatter([], [], s=np.log10(size + 1) * 100, c='gray', alpha=0.5,
                  label=label)
    ax.legend(title='Model Size', loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('../figures/week5_benchmark_timeline.pdf', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    create_attention_matrix()
    create_efficiency_comparison()
    create_benchmark_timeline()
    print("Week 5 overview visualizations generated successfully!")