"""
Generate charts for template showcase presentation
Using Optimal Readability color scheme
"""

import sys
import os
sys.path.append(r'D:\Joerg\Research\slides\2025_NLP_16\NLP_slides\common')

import numpy as np
import matplotlib.pyplot as plt
from chart_utils import *

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# ==================================
# 1. Model Performance Comparison
# ==================================
def generate_model_comparison():
    fig, ax = create_figure(figsize=(9, 5))

    models = ['RNN', 'LSTM', 'GRU', 'Transformer']
    accuracy = [72.3, 85.4, 84.1, 92.7]
    speed = [45.2, 38.9, 40.1, 65.3]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, accuracy, width, label='Accuracy (%)',
                   color=COLORS['chart1'])
    bars2 = ax.bar(x + width/2, speed, width, label='Speed (samples/sec)',
                   color=COLORS['chart2'])

    add_value_labels(ax, bars1, format_str='{:.1f}', offset=2)
    add_value_labels(ax, bars2, format_str='{:.1f}', offset=2)

    ax.set_xlabel('Model Architecture', fontweight='bold', fontsize=11)
    ax.set_ylabel('Performance Metrics', fontweight='bold', fontsize=11)
    ax.set_title('Sequence Model Performance Comparison', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='bold')
    ax.legend(frameon=True, edgecolor='black', loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    save_figure(fig, '../figures/model_comparison.pdf')
    plt.close()

# ==================================
# 2. Attention Weights Heatmap
# ==================================
def generate_attention_heatmap():
    fig, ax = create_figure(figsize=(8, 5.5))

    # Generate sample attention weights
    source_words = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    target_words = ['Le', 'chat', 'est', 'assis', 'sur', 'le', 'tapis']

    np.random.seed(42)
    attention_weights = np.random.rand(len(target_words), len(source_words))
    # Make diagonal stronger to simulate alignment
    for i in range(min(len(source_words), len(target_words))):
        attention_weights[i, i] += 0.5
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)

    create_heatmap(ax, attention_weights, source_words, target_words,
                  title='Attention Mechanism Visualization',
                  cbar_label='Attention Weight', annotate=True)

    ax.set_xlabel('Source Sentence (English)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Target Sentence (French)', fontweight='bold', fontsize=11)

    save_figure(fig, '../figures/attention_heatmap.pdf')
    plt.close()

# ==================================
# 3. Training Progress Dashboard
# ==================================
def generate_training_dashboard():
    fig, axes = create_multi_panel_figure(2, 2, figsize=(11, 7))

    epochs = np.arange(1, 51)

    # Loss curves
    ax = axes[0, 0]
    train_loss = 4.0 * np.exp(-epochs/10) + 0.5 + np.random.randn(50) * 0.1
    val_loss = 4.0 * np.exp(-epochs/10) + 0.7 + np.random.randn(50) * 0.15

    create_line_plot(ax, epochs,
                    {'Training': train_loss, 'Validation': val_loss},
                    xlabel='Epoch', ylabel='Loss', title='Training Progress')

    # Accuracy curves
    ax = axes[0, 1]
    train_acc = 100 * (1 - np.exp(-epochs/15)) + np.random.randn(50) * 2
    val_acc = 100 * (1 - np.exp(-epochs/15)) - 5 + np.random.randn(50) * 3

    create_line_plot(ax, epochs,
                    {'Training': train_acc, 'Validation': val_acc},
                    xlabel='Epoch', ylabel='Accuracy (%)', title='Accuracy Evolution')

    # Learning rate schedule
    ax = axes[1, 0]
    lr = 0.001 * np.power(0.95, epochs/5)
    ax.plot(epochs, lr, color=COLORS['chart3'], linewidth=2)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Learning Rate', fontweight='bold')
    ax.set_title('Learning Rate Schedule', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Gradient norm
    ax = axes[1, 1]
    grad_norm = 5.0 * np.exp(-epochs/20) + 0.5 + np.random.randn(50) * 0.3
    ax.plot(epochs, grad_norm, color=COLORS['chart4'], linewidth=2)
    add_threshold_line(ax, 1.0, 'Gradient Clipping Threshold')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Gradient Norm', fontweight='bold')
    ax.set_title('Gradient Norm Monitoring', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, edgecolor='black')

    plt.suptitle('Training Dashboard - Seq2Seq Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, '../figures/training_dashboard.pdf')
    plt.close()

# ==================================
# 4. Sequence Length Analysis
# ==================================
def generate_sequence_analysis():
    fig, ax = create_figure(figsize=(8, 5))

    lengths = np.arange(10, 101, 10)
    rnn_time = lengths ** 1.2
    transformer_time = lengths * np.log(lengths)

    create_line_plot(ax, lengths,
                    {'RNN (Sequential)': rnn_time, 'Transformer (Parallel)': transformer_time},
                    xlabel='Sequence Length', ylabel='Computation Time (ms)',
                    title='Computational Complexity Analysis')

    # Add annotation
    add_annotation_box(ax, 'Transformer wins\nfor long sequences',
                      xy=(80, transformer_time[7]),
                      xytext=(60, 400))

    ax.set_xlim(5, 105)

    save_figure(fig, '../figures/sequence_analysis.pdf')
    plt.close()

# ==================================
# 5. Encoder-Decoder Architecture
# ==================================
def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(10, 5))
    setup_plotting_style()

    # Remove axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Encoder blocks
    encoder_x = [1, 2, 3]
    for i, x in enumerate(encoder_x):
        rect = plt.Rectangle((x-0.3, 1), 0.6, 0.8,
                            facecolor=COLORS['chart1'],
                            edgecolor=COLORS['black'], linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 1.4, f'Enc\n{i+1}', ha='center', va='center',
               fontweight='bold', color='white')

    # Context vector
    rect = plt.Rectangle((4.7, 1), 0.8, 0.8,
                        facecolor=COLORS['chart3'],
                        edgecolor=COLORS['black'], linewidth=2)
    ax.add_patch(rect)
    ax.text(5.1, 1.4, 'Context', ha='center', va='center',
           fontweight='bold', color='white')

    # Decoder blocks
    decoder_x = [6.5, 7.5, 8.5]
    for i, x in enumerate(decoder_x):
        rect = plt.Rectangle((x-0.3, 1), 0.6, 0.8,
                            facecolor=COLORS['chart2'],
                            edgecolor=COLORS['black'], linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 1.4, f'Dec\n{i+1}', ha='center', va='center',
               fontweight='bold', color='white')

    # Arrows
    for i in range(len(encoder_x)-1):
        ax.arrow(encoder_x[i]+0.3, 1.4, 0.4, 0, head_width=0.1,
                head_length=0.1, fc=COLORS['black'], ec=COLORS['black'])

    ax.arrow(encoder_x[-1]+0.3, 1.4, 0.9, 0, head_width=0.1,
            head_length=0.1, fc=COLORS['black'], ec=COLORS['black'])

    ax.arrow(5.5, 1.4, 0.7, 0, head_width=0.1,
            head_length=0.1, fc=COLORS['black'], ec=COLORS['black'])

    for i in range(len(decoder_x)-1):
        ax.arrow(decoder_x[i]+0.3, 1.4, 0.4, 0, head_width=0.1,
                head_length=0.1, fc=COLORS['black'], ec=COLORS['black'])

    # Labels
    ax.text(2, 0.3, 'Input Sequence', ha='center', fontsize=12, fontweight='bold')
    ax.text(7.5, 0.3, 'Output Sequence', ha='center', fontsize=12, fontweight='bold')
    ax.text(5, 3.5, 'Encoder-Decoder Architecture', ha='center',
           fontsize=14, fontweight='bold')

    # Input/Output examples
    ax.text(2, 2.5, '"Hello world"', ha='center', fontsize=10,
           style='italic', color=COLORS['gray'])
    ax.text(7.5, 2.5, '"Bonjour monde"', ha='center', fontsize=10,
           style='italic', color=COLORS['gray'])

    save_figure(fig, '../figures/encoder_decoder.pdf')
    plt.close()

# ==================================
# 6. BLEU Score Comparison
# ==================================
def generate_bleu_comparison():
    fig, ax = create_figure(figsize=(8, 5))

    systems = ['Baseline\nRNN', 'LSTM\n+Attention', 'Transformer\nBase', 'Transformer\nLarge']
    bleu_scores = [23.5, 31.2, 37.8, 41.3]
    colors = [COLORS['gray'], COLORS['chart1'], COLORS['chart2'], COLORS['success']]

    bars = ax.bar(systems, bleu_scores, color=colors, edgecolor='black', linewidth=2)
    add_value_labels(ax, bars, format_str='{:.1f}', offset=1)

    ax.set_ylabel('BLEU Score', fontweight='bold', fontsize=11)
    ax.set_title('Machine Translation Quality Evolution', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 45)
    ax.grid(axis='y', alpha=0.3)

    # Add reference line
    add_threshold_line(ax, 35, 'Human Quality Threshold', color=COLORS['warning'])
    ax.legend(frameon=True, edgecolor='black')

    save_figure(fig, '../figures/bleu_comparison.pdf')
    plt.close()

# ==================================
# Main execution
# ==================================
if __name__ == '__main__':
    print("Generating showcase charts with Optimal Readability color scheme...")

    generate_model_comparison()
    generate_attention_heatmap()
    generate_training_dashboard()
    generate_sequence_analysis()
    generate_architecture_diagram()
    generate_bleu_comparison()

    print("All charts generated successfully!")