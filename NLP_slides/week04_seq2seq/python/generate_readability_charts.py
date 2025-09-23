import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Optimal Readability Color Palette
COLORS = {
    'black': '#000000',          # Pure Black - main text
    'blue_accent': '#003D7A',    # Deep Blue - primary accent
    'gray': '#4A4A4A',           # Dark Gray - secondary text
    'light_gray': '#E5E5E5',     # Light gray for grid
    'white': '#FFFFFF',          # Pure white backgrounds
    # Chart colors (colorblind-safe)
    'chart1': '#0066CC',         # Strong Blue
    'chart2': '#FF8800',         # Orange
    'chart3': '#00A0A0',         # Teal
    'chart4': '#8B4789',         # Purple
    'success': '#228B22',        # Dark Green
    'warning': '#CC0000',        # Dark Red
}

# Set high-readability defaults
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = COLORS['black']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = COLORS['light_gray']
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['text.color'] = COLORS['black']
plt.rcParams['axes.labelcolor'] = COLORS['black']
plt.rcParams['xtick.color'] = COLORS['black']
plt.rcParams['ytick.color'] = COLORS['black']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

def create_attention_heatmap():
    """Create detailed attention visualization with annotations"""
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[3, 1])

    # Main heatmap
    ax1 = fig.add_subplot(gs[0, :2])

    # Create attention matrix with realistic patterns
    seq_len = 8
    attention = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            # Stronger attention near diagonal and with some distance dependency
            distance = abs(i - j)
            attention[i, j] = np.exp(-distance * 0.3) * (0.5 + 0.5 * np.random.random())

    # Normalize rows
    attention = attention / attention.sum(axis=1, keepdims=True)

    # Create custom colormap for readability
    colors_map = plt.cm.colors.LinearSegmentedColormap.from_list('',
        ['white', COLORS['chart3'], COLORS['chart1'], COLORS['blue_accent']])

    im = ax1.imshow(attention, cmap=colors_map, aspect='auto', vmin=0, vmax=0.5)

    # Add text annotations with high contrast
    for i in range(seq_len):
        for j in range(seq_len):
            val = attention[i, j]
            text_color = 'white' if val > 0.25 else COLORS['black']
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=text_color, fontsize=9, fontweight='bold')

    # Source and target words
    source_words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'dog']
    target_words = ['Le', 'rapide', 'renard', 'brun', 'saute', 'sur', 'le', 'chien']

    ax1.set_xticks(range(seq_len))
    ax1.set_yticks(range(seq_len))
    ax1.set_xticklabels(source_words, fontsize=10, fontweight='bold')
    ax1.set_yticklabels(target_words, fontsize=10, fontweight='bold')
    ax1.set_xlabel('Source Sequence', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Target Sequence', fontsize=11, fontweight='bold')
    ax1.set_title('Attention Weight Distribution', fontsize=12, fontweight='bold', pad=10)

    # Add grid for clarity
    ax1.set_xticks(np.arange(-0.5, seq_len, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, seq_len, 1), minor=True)
    ax1.grid(which='minor', color=COLORS['gray'], linestyle='-', linewidth=0.5)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', fontsize=10, fontweight='bold')

    # Attention distribution plot
    ax2 = fig.add_subplot(gs[0, 2])

    # Show attention distribution for selected position
    selected_pos = 3
    ax2.bar(range(seq_len), attention[selected_pos], color=COLORS['chart1'])
    ax2.set_xticks(range(seq_len))
    ax2.set_xticklabels(source_words, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Weight', fontsize=10, fontweight='bold')
    ax2.set_title(f'Attention from "{target_words[selected_pos]}"',
                  fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 0.5])

    # Summary statistics
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')

    # Calculate entropy for each position
    entropy = -np.sum(attention * np.log(attention + 1e-10), axis=1)
    avg_entropy = np.mean(entropy)

    stats_text = f"Average Entropy: {avg_entropy:.3f}  |  Max Weight: {np.max(attention):.3f}  |  "
    stats_text += f"Sparsity: {np.sum(attention < 0.1) / attention.size:.1%}"

    ax3.text(0.5, 0.5, stats_text, transform=ax3.transAxes,
             fontsize=11, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['light_gray'],
                      edgecolor=COLORS['black'], linewidth=1))

    plt.suptitle('Comprehensive Attention Analysis', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../figures/readability_attention_comprehensive.pdf',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("[DONE] Generated readability_attention_comprehensive.pdf")

def create_model_comparison_charts():
    """Create multiple model comparison visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Data
    models = ['RNN', 'LSTM', 'GRU', 'Seq2Seq', 'Attention', 'Transformer']

    # 1. BLEU Score Comparison
    ax = axes[0, 0]
    bleu_scores = [65, 72, 74, 78, 89, 95]
    bars = ax.bar(models, bleu_scores, color=[COLORS['gray'], COLORS['gray'],
                   COLORS['chart3'], COLORS['chart2'], COLORS['chart1'], COLORS['success']])

    # Add value labels
    for bar, val in zip(bars, bleu_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', fontweight='bold', fontsize=10)

    ax.set_ylabel('BLEU Score (%)', fontweight='bold')
    ax.set_title('Translation Quality', fontweight='bold', fontsize=12)
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(models, rotation=30, ha='right')

    # 2. Training Time Comparison
    ax = axes[0, 1]
    training_hours = [2, 8, 6, 12, 18, 24]
    ax.barh(models, training_hours, color=COLORS['chart2'])
    ax.set_xlabel('Training Time (hours)', fontweight='bold')
    ax.set_title('Computational Cost', fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)

    for i, val in enumerate(training_hours):
        ax.text(val + 0.5, i, f'{val}h', va='center', fontweight='bold')

    # 3. Parameter Count
    ax = axes[0, 2]
    params = [0.5, 2.1, 1.8, 3.5, 4.2, 8.5]  # in millions
    ax.scatter(params, bleu_scores, s=200, c=[COLORS['chart1'], COLORS['chart1'],
               COLORS['chart3'], COLORS['chart3'], COLORS['chart2'], COLORS['success']],
               alpha=0.7, edgecolors='black', linewidth=2)

    for i, model in enumerate(models):
        ax.annotate(model, (params[i], bleu_scores[i]),
                   xytext=(5, 5), textcoords='offset points', fontweight='bold')

    ax.set_xlabel('Parameters (Millions)', fontweight='bold')
    ax.set_ylabel('BLEU Score (%)', fontweight='bold')
    ax.set_title('Efficiency vs Performance', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 4. Learning Curves
    ax = axes[1, 0]
    epochs = np.arange(1, 51)

    for i, (model, color) in enumerate(zip(['RNN', 'LSTM', 'Transformer'],
                                          [COLORS['gray'], COLORS['chart2'], COLORS['chart1']])):
        loss = 4 * np.exp(-epochs/(10+i*5)) + 0.3 + np.random.normal(0, 0.02, 50)
        ax.plot(epochs, loss, label=model, color=color, linewidth=2)

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('Training Dynamics', fontweight='bold', fontsize=12)
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3)

    # 5. Inference Speed
    ax = axes[1, 1]
    inference_ms = [15, 25, 20, 35, 45, 10]
    colors = [COLORS['success'] if x < 20 else COLORS['chart2'] if x < 40 else COLORS['warning']
              for x in inference_ms]
    bars = ax.bar(models, inference_ms, color=colors)

    ax.set_ylabel('Inference Time (ms)', fontweight='bold')
    ax.set_title('Speed Comparison', fontweight='bold', fontsize=12)
    ax.axhline(y=30, color=COLORS['warning'], linestyle='--', alpha=0.5, label='Real-time threshold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(models, rotation=30, ha='right')

    # 6. Feature Comparison Radar Chart
    ax = axes[1, 2]

    # Categories
    categories = ['Speed', 'Accuracy', 'Memory\nEfficiency', 'Training\nStability', 'Interpretability']
    N = len(categories)

    # Create radar chart for top 3 models
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax = plt.subplot(2, 3, 6, projection='polar')

    # Data for each model
    transformer_stats = [0.9, 0.95, 0.6, 0.85, 0.7]
    attention_stats = [0.7, 0.89, 0.75, 0.8, 0.85]
    lstm_stats = [0.6, 0.72, 0.8, 0.7, 0.5]

    for stats, label, color in [(transformer_stats, 'Transformer', COLORS['chart1']),
                                 (attention_stats, 'Attention', COLORS['chart2']),
                                 (lstm_stats, 'LSTM', COLORS['chart3'])]:
        stats = stats + stats[:1]
        ax.plot(angles, stats, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, stats, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Metric Comparison', fontweight='bold', fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.suptitle('Comprehensive Model Analysis Dashboard', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/readability_model_comparison.pdf',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("[DONE] Generated readability_model_comparison.pdf")

def create_training_monitoring_dashboard():
    """Create training monitoring dashboard with multiple metrics"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    epochs = np.arange(1, 101)

    # 1. Loss curves with confidence bands
    ax1 = fig.add_subplot(gs[0, :2])

    train_loss = 4 * np.exp(-epochs/20) + 0.3 + np.random.normal(0, 0.05, 100)
    val_loss = 4 * np.exp(-epochs/25) + 0.4 + np.random.normal(0, 0.08, 100)

    # Smooth the curves
    from scipy.ndimage import gaussian_filter1d
    train_smooth = gaussian_filter1d(train_loss, sigma=2)
    val_smooth = gaussian_filter1d(val_loss, sigma=2)

    ax1.plot(epochs, train_smooth, color=COLORS['chart1'], linewidth=2, label='Training')
    ax1.plot(epochs, val_smooth, color=COLORS['chart2'], linewidth=2, label='Validation')

    # Add confidence bands
    ax1.fill_between(epochs, train_smooth - 0.1, train_smooth + 0.1,
                     color=COLORS['chart1'], alpha=0.2)
    ax1.fill_between(epochs, val_smooth - 0.15, val_smooth + 0.15,
                     color=COLORS['chart2'], alpha=0.2)

    ax1.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Loss', fontweight='bold', fontsize=11)
    ax1.set_title('Training Progress with Confidence Bands', fontweight='bold', fontsize=12)
    ax1.legend(frameon=True, edgecolor='black', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Learning rate schedule
    ax2 = fig.add_subplot(gs[0, 2])

    lr = np.ones(30) * 0.001
    lr = np.concatenate([lr, np.ones(30) * 0.0005])
    lr = np.concatenate([lr, np.ones(20) * 0.0001])
    lr = np.concatenate([lr, np.ones(20) * 0.00005])

    ax2.step(epochs, lr, color=COLORS['success'], linewidth=2, where='mid')
    ax2.fill_between(epochs, 0, lr, color=COLORS['success'], alpha=0.2, step='mid')
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Learning Rate', fontweight='bold')
    ax2.set_title('LR Schedule', fontweight='bold', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # 3. Gradient norm monitoring
    ax3 = fig.add_subplot(gs[1, 0])

    grad_norm = 10 * np.exp(-epochs/30) + np.random.normal(0, 0.5, 100)
    grad_norm = np.abs(grad_norm)

    ax3.plot(epochs, grad_norm, color=COLORS['chart4'], linewidth=1.5)
    ax3.axhline(y=5, color=COLORS['warning'], linestyle='--', label='Clipping threshold')
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('Gradient Norm', fontweight='bold')
    ax3.set_title('Gradient Monitoring', fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Accuracy metrics
    ax4 = fig.add_subplot(gs[1, 1])

    train_acc = 100 * (1 - np.exp(-epochs/25)) + np.random.normal(0, 1, 100)
    val_acc = 100 * (1 - np.exp(-epochs/30)) - 5 + np.random.normal(0, 1.5, 100)

    ax4.plot(epochs, train_acc, color=COLORS['chart1'], linewidth=2, label='Train Acc')
    ax4.plot(epochs, val_acc, color=COLORS['chart2'], linewidth=2, label='Val Acc')
    ax4.set_xlabel('Epoch', fontweight='bold')
    ax4.set_ylabel('Accuracy (%)', fontweight='bold')
    ax4.set_title('Accuracy Trends', fontweight='bold', fontsize=12)
    ax4.legend(frameon=True, edgecolor='black')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 105])

    # 5. Memory usage
    ax5 = fig.add_subplot(gs[1, 2])

    memory_usage = 60 + 20 * np.sin(epochs/10) + np.random.normal(0, 2, 100)

    ax5.fill_between(epochs, 0, memory_usage, color=COLORS['chart3'], alpha=0.3)
    ax5.plot(epochs, memory_usage, color=COLORS['chart3'], linewidth=2)
    ax5.axhline(y=80, color=COLORS['warning'], linestyle='--', label='Memory limit')
    ax5.set_xlabel('Epoch', fontweight='bold')
    ax5.set_ylabel('Memory (GB)', fontweight='bold')
    ax5.set_title('GPU Memory Usage', fontweight='bold', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 100])

    # 6. Batch processing time
    ax6 = fig.add_subplot(gs[2, :])

    batch_times = 200 + np.random.normal(0, 20, 100)

    ax6.bar(epochs[::5], batch_times[::5], width=3, color=COLORS['chart4'], alpha=0.7)
    ax6.axhline(y=np.mean(batch_times), color=COLORS['black'],
                linestyle='-', linewidth=2, label=f'Mean: {np.mean(batch_times):.0f}ms')
    ax6.set_xlabel('Epoch', fontweight='bold')
    ax6.set_ylabel('Batch Time (ms)', fontweight='bold')
    ax6.set_title('Processing Speed per Batch', fontweight='bold', fontsize=12)
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)

    plt.suptitle('Training Monitoring Dashboard', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../figures/readability_training_dashboard.pdf',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("[DONE] Generated readability_training_dashboard.pdf")

def create_architecture_flow_diagram():
    """Create detailed architecture flow diagram"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Encoder-Decoder with Attention
    ax1.set_xlim(0, 14)
    ax1.set_ylim(0, 6)
    ax1.axis('off')

    # Encoder blocks
    encoder_x = [1, 2.5, 4]
    for i, x in enumerate(encoder_x):
        rect = FancyBboxPatch((x, 2), 1.2, 1.5,
                              boxstyle="round,pad=0.05",
                              facecolor='white',
                              edgecolor=COLORS['chart1'],
                              linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x + 0.6, 2.75, f'Enc\n{i+1}', ha='center', va='center',
                fontweight='bold', fontsize=11)

    # Hidden states
    for i, x in enumerate(encoder_x):
        circle = Circle((x + 0.6, 4.5), 0.3,
                       facecolor='white',
                       edgecolor=COLORS['chart3'],
                       linewidth=2)
        ax1.add_patch(circle)
        ax1.text(x + 0.6, 4.5, f'h{i+1}', ha='center', va='center',
                fontweight='bold', fontsize=10)

    # Attention mechanism
    att_rect = FancyBboxPatch((6, 3.5), 2, 1.5,
                             boxstyle="round,pad=0.05",
                             facecolor=COLORS['light_gray'],
                             edgecolor=COLORS['blue_accent'],
                             linewidth=2.5)
    ax1.add_patch(att_rect)
    ax1.text(7, 4.25, 'Attention\nMechanism', ha='center', va='center',
            fontweight='bold', fontsize=11)

    # Decoder blocks
    decoder_x = [9, 10.5, 12]
    for i, x in enumerate(decoder_x):
        rect = FancyBboxPatch((x, 2), 1.2, 1.5,
                              boxstyle="round,pad=0.05",
                              facecolor='white',
                              edgecolor=COLORS['chart2'],
                              linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x + 0.6, 2.75, f'Dec\n{i+1}', ha='center', va='center',
                fontweight='bold', fontsize=11)

    # Arrows
    for i in range(len(encoder_x) - 1):
        ax1.arrow(encoder_x[i] + 1.2, 2.75, 1.1, 0,
                 head_width=0.15, head_length=0.1,
                 fc=COLORS['gray'], ec=COLORS['gray'])

    # Attention connections
    for x in encoder_x:
        ax1.plot([x + 0.6, 6.5], [4.2, 4.25],
                color=COLORS['chart3'], linewidth=1, linestyle='--', alpha=0.7)

    for x in decoder_x:
        ax1.plot([7.5, x + 0.6], [4.25, 3.5],
                color=COLORS['chart2'], linewidth=1, linestyle='--', alpha=0.7)

    # Labels
    ax1.text(2.5, 0.8, 'Input Sequence', ha='center', fontweight='bold', fontsize=12)
    ax1.text(10.5, 0.8, 'Output Sequence', ha='center', fontweight='bold', fontsize=12)
    ax1.text(7, 5.5, 'Encoder-Decoder with Attention', ha='center',
            fontweight='bold', fontsize=14)

    # Data flow visualization
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 5)
    ax2.axis('off')

    # Processing stages
    stages = ['Input\nEmbedding', 'Positional\nEncoding', 'Multi-Head\nAttention',
              'Feed\nForward', 'Output\nProjection']

    stage_x = np.linspace(1, 9, 5)

    for i, (x, stage) in enumerate(zip(stage_x, stages)):
        # Stage box
        rect = FancyBboxPatch((x - 0.5, 2), 1, 1.5,
                              boxstyle="round,pad=0.05",
                              facecolor='white',
                              edgecolor=COLORS['chart1'] if i < 2 else COLORS['chart2'] if i < 4 else COLORS['success'],
                              linewidth=2)
        ax2.add_patch(rect)
        ax2.text(x, 2.75, stage, ha='center', va='center',
                fontweight='bold', fontsize=10)

        # Data dimensions
        if i < len(stage_x) - 1:
            ax2.arrow(x + 0.5, 2.75, stage_x[i+1] - x - 1, 0,
                     head_width=0.15, head_length=0.1,
                     fc=COLORS['black'], ec=COLORS['black'], linewidth=1.5)

        # Dimension labels
        dims = ['[B, L, D]', '[B, L, D]', '[B, L, D]', '[B, L, D]', '[B, L, V]']
        ax2.text(x, 1.3, dims[i], ha='center', fontsize=9,
                style='italic', color=COLORS['gray'])

    ax2.text(5, 4.5, 'Transformer Processing Pipeline', ha='center',
            fontweight='bold', fontsize=14)

    ax2.text(5, 0.5, 'B: Batch Size  |  L: Sequence Length  |  D: Model Dimension  |  V: Vocabulary Size',
            ha='center', fontsize=10, color=COLORS['gray'], style='italic')

    plt.tight_layout()
    plt.savefig('../figures/readability_architecture_flow.pdf',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("[DONE] Generated readability_architecture_flow.pdf")

def create_performance_breakdown():
    """Create detailed performance breakdown charts"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Task-specific performance
    ax = axes[0, 0]

    tasks = ['Translation', 'Summarization', 'QA', 'Generation']
    models = ['LSTM', 'Attention', 'Transformer']

    x = np.arange(len(tasks))
    width = 0.25

    scores_lstm = [72, 65, 60, 68]
    scores_att = [85, 78, 75, 80]
    scores_trans = [95, 92, 88, 93]

    ax.bar(x - width, scores_lstm, width, label='LSTM', color=COLORS['chart3'])
    ax.bar(x, scores_att, width, label='Attention', color=COLORS['chart2'])
    ax.bar(x + width, scores_trans, width, label='Transformer', color=COLORS['chart1'])

    ax.set_xlabel('Task', fontweight='bold')
    ax.set_ylabel('Score (%)', fontweight='bold')
    ax.set_title('Task-Specific Performance', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend(frameon=True, edgecolor='black')
    ax.grid(axis='y', alpha=0.3)

    # 2. Length generalization
    ax = axes[0, 1]

    lengths = [10, 20, 30, 40, 50, 60, 70]

    for model, color in [('Transformer', COLORS['chart1']),
                         ('Attention', COLORS['chart2']),
                         ('LSTM', COLORS['chart3'])]:
        if model == 'Transformer':
            perf = 95 - np.array(lengths) * 0.2
        elif model == 'Attention':
            perf = 85 - np.array(lengths) * 0.5
        else:
            perf = 75 - np.array(lengths) * 0.8

        perf = np.maximum(perf, 20)  # Floor at 20%
        ax.plot(lengths, perf, 'o-', label=model, color=color,
               linewidth=2, markersize=6)

    ax.set_xlabel('Sequence Length', fontweight='bold')
    ax.set_ylabel('Performance (%)', fontweight='bold')
    ax.set_title('Length Generalization', fontweight='bold', fontsize=12)
    ax.legend(frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3)

    # 3. Resource utilization
    ax = axes[1, 0]

    metrics = ['GPU\nMemory', 'Training\nTime', 'Inference\nSpeed', 'Energy\nUsage']

    # Normalized scores (lower is better for all)
    lstm_util = [0.3, 0.4, 0.3, 0.3]
    att_util = [0.5, 0.6, 0.5, 0.5]
    trans_util = [0.8, 0.7, 0.2, 0.6]

    x = np.arange(len(metrics))

    ax.bar(x - width, lstm_util, width, label='LSTM', color=COLORS['success'])
    ax.bar(x, att_util, width, label='Attention', color=COLORS['chart2'])
    ax.bar(x + width, trans_util, width, label='Transformer', color=COLORS['warning'])

    ax.set_ylabel('Normalized Usage', fontweight='bold')
    ax.set_title('Resource Utilization (Lower is Better)', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(frameon=True, edgecolor='black')
    ax.grid(axis='y', alpha=0.3)

    # 4. Error analysis
    ax = axes[1, 1]

    error_types = ['Grammatical', 'Semantic', 'Fluency', 'Coherence']
    transformer_errors = [5, 8, 6, 7]
    attention_errors = [8, 12, 10, 11]
    lstm_errors = [15, 18, 16, 20]

    x = np.arange(len(error_types))

    ax.barh(x - width, lstm_errors, width, label='LSTM', color=COLORS['warning'])
    ax.barh(x, attention_errors, width, label='Attention', color=COLORS['chart2'])
    ax.barh(x + width, transformer_errors, width, label='Transformer', color=COLORS['success'])

    ax.set_xlabel('Error Rate (%)', fontweight='bold')
    ax.set_title('Error Analysis by Type', fontweight='bold', fontsize=12)
    ax.set_yticks(x)
    ax.set_yticklabels(error_types)
    ax.legend(frameon=True, edgecolor='black')
    ax.grid(axis='x', alpha=0.3)

    plt.suptitle('Comprehensive Performance Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/readability_performance_breakdown.pdf',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("[DONE] Generated readability_performance_breakdown.pdf")

def create_beam_search_visualization():
    """Create detailed beam search visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Beam search tree
    ax1.set_xlim(-0.5, 4.5)
    ax1.set_ylim(-0.5, 4.5)
    ax1.axis('off')

    # Tree structure
    levels = [
        [(2, 4, 'START')],
        [(1, 3, 'The'), (2, 3, 'A'), (3, 3, 'This')],
        [(0.5, 2, 'cat'), (1.5, 2, 'dog'), (2.5, 2, 'quick'), (3.5, 2, 'small')],
        [(0.5, 1, 'sits'), (1.5, 1, 'runs'), (2.5, 1, 'brown'), (3.5, 1, 'jumps')],
        [(1, 0, 'on'), (2, 0, 'the'), (3, 0, 'over')]
    ]

    # Draw nodes and connections
    for level_idx, level in enumerate(levels):
        for x, y, word in level:
            # Node
            circle = Circle((x, y), 0.3,
                          facecolor='white' if level_idx == 0 else COLORS['light_gray'],
                          edgecolor=COLORS['blue_accent'] if level_idx == 0 else COLORS['chart1'],
                          linewidth=2)
            ax1.add_patch(circle)
            ax1.text(x, y, word, ha='center', va='center',
                    fontsize=9, fontweight='bold')

            # Connections to next level
            if level_idx < len(levels) - 1:
                next_level = levels[level_idx + 1]
                for next_x, next_y, _ in next_level[:2]:  # Connect to top 2 in next level
                    ax1.plot([x, next_x], [y - 0.3, next_y + 0.3],
                            color=COLORS['gray'], linewidth=1, alpha=0.5)

    # Highlight best path
    best_path = [(2, 4), (2, 3), (2.5, 2), (2.5, 1), (2, 0)]
    for i in range(len(best_path) - 1):
        ax1.plot([best_path[i][0], best_path[i+1][0]],
                [best_path[i][1] - 0.3, best_path[i+1][1] + 0.3],
                color=COLORS['success'], linewidth=3, alpha=0.8)

    ax1.text(2, 4.7, 'Beam Search Tree (k=3)', ha='center',
            fontweight='bold', fontsize=12)
    ax1.text(2, -0.7, 'Best Path: "A quick brown the"', ha='center',
            fontweight='bold', fontsize=10, color=COLORS['success'])

    # Probability distribution over time
    ax2.set_xlabel('Decoding Step', fontweight='bold')
    ax2.set_ylabel('Probability', fontweight='bold')
    ax2.set_title('Beam Probabilities Over Time', fontweight='bold', fontsize=12)

    steps = range(1, 6)
    beam1 = [0.4, 0.35, 0.32, 0.30, 0.28]
    beam2 = [0.35, 0.33, 0.30, 0.28, 0.26]
    beam3 = [0.25, 0.22, 0.20, 0.18, 0.16]
    dropped = [0, 0.1, 0.18, 0.24, 0.30]

    ax2.fill_between(steps, 0, beam1, label='Beam 1', color=COLORS['chart1'], alpha=0.7)
    ax2.fill_between(steps, beam1, np.array(beam1) + np.array(beam2),
                     label='Beam 2', color=COLORS['chart2'], alpha=0.7)
    ax2.fill_between(steps, np.array(beam1) + np.array(beam2),
                     np.array(beam1) + np.array(beam2) + np.array(beam3),
                     label='Beam 3', color=COLORS['chart3'], alpha=0.7)
    ax2.fill_between(steps, np.array(beam1) + np.array(beam2) + np.array(beam3),
                     1, label='Pruned', color=COLORS['gray'], alpha=0.3)

    ax2.set_ylim([0, 1])
    ax2.legend(frameon=True, edgecolor='black', loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/readability_beam_search.pdf',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("[DONE] Generated readability_beam_search.pdf")

if __name__ == '__main__':
    print("Generating Optimal Readability charts...")
    print("=" * 50)

    create_attention_heatmap()
    create_model_comparison_charts()
    create_training_monitoring_dashboard()
    create_architecture_flow_diagram()
    create_performance_breakdown()
    create_beam_search_visualization()

    print("=" * 50)
    print("All readability charts generated successfully!")
    print("Total: 6 comprehensive multi-panel visualizations")