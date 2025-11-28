"""
Generate charts for Week 5: Transformers
Using Optimal Readability color scheme
"""

import sys
import os
sys.path.append(r'D:\Joerg\Research\slides\2025_NLP_16\NLP_slides\common')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from chart_utils import *

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# ==================================
# 1. Attention Visualization
# ==================================
def generate_attention_visualization():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    setup_plotting_style()

    # Sample sentence
    words = ['The', 'student', 'who', 'studied', 'hard', 'passed', 'the', 'exam']
    n_words = len(words)

    # Create attention weights (focused pattern)
    np.random.seed(42)
    attention_weights = np.random.rand(n_words, n_words) * 0.3

    # Add stronger connections
    attention_weights[5, 1] = 0.9  # "passed" attends to "student"
    attention_weights[4, 3] = 0.8  # "hard" attends to "studied"
    attention_weights[7, 5] = 0.7  # "exam" attends to "passed"
    attention_weights[1, 0] = 0.6  # "student" attends to "The"

    # Normalize rows
    for i in range(n_words):
        if attention_weights[i].sum() > 0:
            attention_weights[i] = attention_weights[i] / attention_weights[i].sum()

    # Plot heatmap
    im = ax1.imshow(attention_weights, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax1.set_xticks(range(n_words))
    ax1.set_yticks(range(n_words))
    ax1.set_xticklabels(words, rotation=45, ha='right', fontweight='bold')
    ax1.set_yticklabels(words, fontweight='bold')
    ax1.set_xlabel('Keys (Input)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Queries (Output)', fontweight='bold', fontsize=11)
    ax1.set_title('Self-Attention Weights', fontweight='bold', fontsize=12)

    # Add grid
    for i in range(n_words + 1):
        ax1.axhline(i - 0.5, color='lightgray', linewidth=0.5)
        ax1.axvline(i - 0.5, color='lightgray', linewidth=0.5)

    # Plot attention connections as graph
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.5, n_words - 0.5)

    # Draw words on left and right
    for i, word in enumerate(words):
        ax2.text(0, n_words - 1 - i, word, ha='right', va='center',
                fontweight='bold', fontsize=10)
        ax2.text(1, n_words - 1 - i, word, ha='left', va='center',
                fontweight='bold', fontsize=10)

    # Draw attention connections
    for i in range(n_words):
        for j in range(n_words):
            if attention_weights[i, j] > 0.2:  # Only show strong connections
                ax2.plot([0.05, 0.95], [n_words - 1 - i, n_words - 1 - j],
                        color=COLORS['chart1'],
                        alpha=attention_weights[i, j],
                        linewidth=attention_weights[i, j] * 3)

    ax2.axis('off')
    ax2.set_title('Attention Flow', fontweight='bold', fontsize=12)

    plt.suptitle('Self-Attention Mechanism', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, '../figures/attention_visualization.pdf')
    plt.close()

# ==================================
# 2. Multi-Head Attention Patterns
# ==================================
def generate_multihead_attention():
    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    setup_plotting_style()

    words = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    n_words = len(words)

    # Different attention patterns for different heads
    patterns = [
        'Position',  # Head 1: Position-based
        'Syntax',    # Head 2: Syntactic relations
        'Semantic',  # Head 3: Semantic similarity
        'Previous',  # Head 4: Previous word
        'Next',      # Head 5: Next word
        'Self',      # Head 6: Self-attention
        'Global',    # Head 7: Global context
        'Random'     # Head 8: Mixed patterns
    ]

    np.random.seed(42)
    for idx, (ax, pattern) in enumerate(zip(axes.flat, patterns)):
        attention = np.zeros((n_words, n_words))

        if pattern == 'Position':
            # Attend to similar positions
            for i in range(n_words):
                for j in range(n_words):
                    attention[i, j] = np.exp(-abs(i - j) / 2)
        elif pattern == 'Syntax':
            # Subject-verb, verb-object patterns
            attention[1, 2] = 1  # cat -> sat
            attention[2, 1] = 1  # sat -> cat
            attention[2, 5] = 1  # sat -> mat
        elif pattern == 'Semantic':
            # Related words
            attention[1, 5] = 1  # cat -> mat
            attention[5, 1] = 1  # mat -> cat
        elif pattern == 'Previous':
            # Attend to previous word
            for i in range(1, n_words):
                attention[i, i-1] = 1
        elif pattern == 'Next':
            # Attend to next word
            for i in range(n_words-1):
                attention[i, i+1] = 1
        elif pattern == 'Self':
            # Self-attention diagonal
            np.fill_diagonal(attention, 1)
        elif pattern == 'Global':
            # Attend to all words equally
            attention.fill(1/n_words)
        else:  # Random
            attention = np.random.rand(n_words, n_words)

        # Normalize
        for i in range(n_words):
            if attention[i].sum() > 0:
                attention[i] = attention[i] / attention[i].sum()

        im = ax.imshow(attention, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        ax.set_title(f'Head {idx+1}: {pattern}', fontweight='bold', fontsize=9)
        ax.set_xticks(range(n_words))
        ax.set_yticks(range(n_words))

        if idx >= 4:  # Bottom row
            ax.set_xticklabels(words, rotation=45, ha='right', fontsize=8)
        else:
            ax.set_xticklabels([])

        if idx % 4 == 0:  # Left column
            ax.set_yticklabels(words, fontsize=8)
        else:
            ax.set_yticklabels([])

        # Grid
        for i in range(n_words + 1):
            ax.axhline(i - 0.5, color='lightgray', linewidth=0.3)
            ax.axvline(i - 0.5, color='lightgray', linewidth=0.3)

    plt.suptitle('Multi-Head Attention: Different Heads Learn Different Patterns',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, '../figures/multihead_attention.pdf')
    plt.close()

# ==================================
# 3. Positional Encoding Visualization
# ==================================
def generate_positional_encoding():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    setup_plotting_style()

    # Parameters
    d_model = 128  # Embedding dimension
    max_len = 100  # Maximum sequence length

    # Generate positional encodings
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    # Plot 1: Full positional encoding matrix
    im1 = ax1.imshow(pe.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_xlabel('Position in Sequence', fontweight='bold')
    ax1.set_ylabel('Encoding Dimension', fontweight='bold')
    ax1.set_title('Positional Encoding Pattern (Sinusoidal)', fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Encoding Value')

    # Plot 2: Sample encodings for specific positions
    positions_to_show = [0, 10, 20, 50, 99]
    colors = [COLORS['chart1'], COLORS['chart2'], COLORS['chart3'],
              COLORS['chart4'], COLORS['success']]

    for pos, color in zip(positions_to_show, colors):
        ax2.plot(pe[pos, :64], label=f'Position {pos}', color=color, linewidth=2)

    ax2.set_xlabel('Encoding Dimension', fontweight='bold')
    ax2.set_ylabel('Encoding Value', fontweight='bold')
    ax2.set_title('Positional Encoding for Different Positions', fontweight='bold')
    ax2.legend(frameon=True, edgecolor='black')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 63)

    plt.suptitle('Positional Encoding: Giving Transformers Position Awareness',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, '../figures/positional_encoding.pdf')
    plt.close()

# ==================================
# 4. Speed Comparison (RNN vs Transformer)
# ==================================
def generate_speed_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    setup_plotting_style()

    # Data for sequence length scaling
    seq_lengths = np.array([10, 50, 100, 200, 500, 1000])

    # RNN: O(n) sequential steps
    rnn_time = seq_lengths * 1.5  # Linear in sequence length

    # Transformer: O(1) parallel steps but O(n²) computation
    transformer_time = np.log(seq_lengths) * 20  # Much better scaling

    ax1.plot(seq_lengths, rnn_time, 'o-', color=COLORS['chart2'],
            linewidth=2, markersize=8, label='RNN (Sequential)')
    ax1.plot(seq_lengths, transformer_time, 's-', color=COLORS['chart1'],
            linewidth=2, markersize=8, label='Transformer (Parallel)')

    ax1.set_xlabel('Sequence Length', fontweight='bold')
    ax1.set_ylabel('Training Time (arbitrary units)', fontweight='bold')
    ax1.set_title('Training Time vs Sequence Length', fontweight='bold')
    ax1.legend(frameon=True, edgecolor='black')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # GPU utilization comparison
    models = ['RNN', 'LSTM', 'GRU', 'Transformer']
    gpu_util = [15, 20, 18, 85]  # Percentage
    colors = [COLORS['gray'], COLORS['gray'], COLORS['gray'], COLORS['success']]

    bars = ax2.bar(models, gpu_util, color=colors, edgecolor='black', linewidth=2)

    # Add value labels
    for bar, val in zip(bars, gpu_util):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val}%', ha='center', fontweight='bold')

    ax2.set_ylabel('GPU Utilization (%)', fontweight='bold')
    ax2.set_title('GPU Efficiency: Parallel vs Sequential', fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)

    # Add threshold line
    ax2.axhline(y=50, color=COLORS['warning'], linestyle='--',
               linewidth=1, label='Efficient Threshold')
    ax2.legend(frameon=True, edgecolor='black')

    plt.suptitle('Why Transformers Train Faster', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, '../figures/speed_comparison.pdf')
    plt.close()

# ==================================
# 5. Architecture Comparison
# ==================================
def generate_architecture_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))
    setup_plotting_style()

    # Performance data for different architectures
    architectures = ['RNN\n(2015)', 'LSTM\n(2016)', 'Seq2Seq\n+Attention\n(2017)',
                    'Transformer\n(2017)', 'BERT\n(2018)', 'GPT-3\n(2020)']

    bleu_scores = [18.5, 23.2, 25.4, 28.4, 31.2, 35.8]
    params_millions = [10, 20, 50, 65, 340, 175000]  # Parameters in millions

    # Normalize parameters for bubble size (log scale)
    bubble_sizes = np.log10(np.array(params_millions) + 1) * 100

    # Create bubble chart
    for i, (arch, bleu, size) in enumerate(zip(architectures, bleu_scores, bubble_sizes)):
        color = COLORS['success'] if 'Transformer' in arch or 'BERT' in arch or 'GPT' in arch else COLORS['gray']
        ax.scatter(i, bleu, s=size, color=color, alpha=0.6,
                  edgecolors='black', linewidth=2)

        # Add parameter count labels
        param_str = f'{params_millions[i]}M' if params_millions[i] < 1000 else f'{params_millions[i]/1000:.0f}B'
        ax.text(i, bleu - 1.5, param_str, ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(range(len(architectures)))
    ax.set_xticklabels(architectures, fontweight='bold')
    ax.set_ylabel('BLEU Score (Translation Quality)', fontweight='bold', fontsize=11)
    ax.set_title('Evolution of NLP Architectures: Quality vs Scale',
                fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(15, 40)

    # Add trend line
    ax.plot(range(len(architectures)), bleu_scores, 'k--', alpha=0.3, linewidth=1)

    # Add annotations
    ax.annotate('Attention Revolution', xy=(3, 28.4), xytext=(3, 35),
               arrowprops=dict(arrowstyle='->', color=COLORS['chart1'], lw=2),
               fontweight='bold', ha='center')

    save_figure(fig, '../figures/architecture_evolution.pdf')
    plt.close()

# ==================================
# 6. Training Dynamics
# ==================================
def generate_training_dynamics():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    setup_plotting_style()

    epochs = np.arange(1, 101)

    # Loss curves
    ax = axes[0, 0]
    rnn_loss = 4.0 * np.exp(-epochs/30) + 0.8 + np.random.randn(100) * 0.05
    transformer_loss = 4.0 * np.exp(-epochs/15) + 0.3 + np.random.randn(100) * 0.03

    ax.plot(epochs, rnn_loss, color=COLORS['chart2'], linewidth=2,
           label='RNN', alpha=0.7)
    ax.plot(epochs, transformer_loss, color=COLORS['chart1'], linewidth=2,
           label='Transformer', alpha=0.7)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('Convergence Speed', fontweight='bold')
    ax.legend(frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3)

    # Gradient norm
    ax = axes[0, 1]
    rnn_grad = 5.0 * np.exp(-epochs/25) + np.random.randn(100) * 0.5 + 2
    transformer_grad = 3.0 * np.exp(-epochs/20) + np.random.randn(100) * 0.2 + 0.5

    ax.plot(epochs, rnn_grad, color=COLORS['chart2'], linewidth=2,
           label='RNN', alpha=0.7)
    ax.plot(epochs, transformer_grad, color=COLORS['chart1'], linewidth=2,
           label='Transformer', alpha=0.7)
    ax.axhline(y=1.0, color=COLORS['warning'], linestyle='--',
              label='Ideal Range')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Gradient Norm', fontweight='bold')
    ax.set_title('Gradient Stability', fontweight='bold')
    ax.legend(frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 8)

    # Attention entropy (diversity of attention)
    ax = axes[1, 0]
    attention_entropy = 3 - 2 * np.exp(-epochs/40) + np.random.randn(100) * 0.1

    ax.plot(epochs, attention_entropy, color=COLORS['chart3'], linewidth=2)
    ax.fill_between(epochs, attention_entropy - 0.2, attention_entropy + 0.2,
                    color=COLORS['chart3'], alpha=0.2)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Attention Entropy', fontweight='bold')
    ax.set_title('Attention Pattern Diversity', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 4)

    # Performance metrics
    ax = axes[1, 1]
    metrics = ['BLEU', 'Perplexity\n(inverted)', 'Accuracy', 'F1 Score']
    rnn_scores = [65, 70, 78, 72]
    transformer_scores = [85, 88, 92, 89]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, rnn_scores, width, label='RNN',
          color=COLORS['chart2'], edgecolor='black', linewidth=1)
    ax.bar(x + width/2, transformer_scores, width, label='Transformer',
          color=COLORS['chart1'], edgecolor='black', linewidth=1)

    ax.set_xlabel('Metric', fontweight='bold')
    ax.set_ylabel('Score (%)', fontweight='bold')
    ax.set_title('Final Performance', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.legend(frameon=True, edgecolor='black')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    plt.suptitle('Training Dynamics: RNN vs Transformer',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, '../figures/training_dynamics.pdf')
    plt.close()

# ==================================
# Main execution
# ==================================
if __name__ == '__main__':
    print("Generating Week 5 Transformer charts with Optimal Readability...")

    generate_attention_visualization()
    print("Generated attention visualization")

    generate_multihead_attention()
    print("Generated multi-head attention patterns")

    generate_positional_encoding()
    print("Generated positional encoding visualization")

    generate_speed_comparison()
    print("Generated speed comparison charts")

    generate_architecture_comparison()
    print("Generated architecture evolution chart")

    generate_training_dynamics()
    print("Generated training dynamics comparison")

    print("All Week 5 charts generated successfully!")