"""
Generate enhanced colorful charts for Week 4 Seq2Seq presentation
Includes multiple color schemes and additional visualization types
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

# Define color palettes
PALETTES = {
    'modern_tech': {
        'primary': '#2E4057',
        'secondary': '#048A81',
        'accent': '#F18F01',
        'neutral': '#7D8491',
        'background': '#F5F9FC',
        'gradient': ['#2E4057', '#048A81', '#F18F01']
    },
    'educational_warm': {
        'primary': '#1E3A8A',
        'secondary': '#60A5FA',
        'accent': '#FB7185',
        'support': '#FCD34D',
        'background': '#FEF3C7',
        'gradient': ['#1E3A8A', '#60A5FA', '#FB7185', '#FCD34D']
    },
    'nature_professional': {
        'primary': '#14532D',
        'secondary': '#0D9488',
        'accent': '#F59E0B',
        'support': '#475569',
        'background': '#F0FDF4',
        'gradient': ['#14532D', '#0D9488', '#F59E0B']
    }
}

# Use the modern tech palette as default
COLORS = PALETTES['modern_tech']

def set_style():
    """Set the overall plot style"""
    plt.rcParams['figure.facecolor'] = COLORS['background']
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = COLORS['neutral']
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.color'] = COLORS['neutral']
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11

def create_attention_heatmap():
    """Create an enhanced attention visualization heatmap"""
    set_style()

    # Generate sample attention weights
    source_words = ['The', 'cat', 'sat', 'on', 'mat', '<EOS>']
    target_words = ['Le', 'chat', 'assis', 'sur', 'tapis', '<EOS>']

    # Create realistic attention pattern
    attention = np.random.rand(len(target_words), len(source_words))
    # Add diagonal tendency
    for i in range(min(len(source_words), len(target_words))):
        attention[i, i] = attention[i, i] * 2 + 0.5
    # Normalize
    attention = attention / attention.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap with custom colormap
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        'custom', ['white', COLORS['secondary'], COLORS['primary']]
    )
    im = ax.imshow(attention, cmap=cmap, aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(source_words)))
    ax.set_yticks(np.arange(len(target_words)))
    ax.set_xticklabels(source_words, fontsize=12)
    ax.set_yticklabels(target_words, fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20, fontsize=11)

    # Add value annotations
    for i in range(len(target_words)):
        for j in range(len(source_words)):
            text = ax.text(j, i, f'{attention[i, j]:.2f}',
                          ha="center", va="center", color="black" if attention[i, j] < 0.5 else "white",
                          fontsize=9)

    ax.set_xlabel('Source Sentence', fontsize=13, color=COLORS['primary'])
    ax.set_ylabel('Target Sentence', fontsize=13, color=COLORS['primary'])
    ax.set_title('Attention Mechanism Heatmap\nTranslation: "The cat sat on mat" → "Le chat assis sur tapis"',
                fontsize=14, color=COLORS['primary'], pad=20)

    plt.tight_layout()
    plt.savefig('../figures/week4_attention_heatmap_colorful.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_model_complexity_comparison():
    """Create a comparison chart of model complexities"""
    set_style()

    models = ['Simple RNN', 'LSTM', 'GRU', 'Basic Seq2Seq', 'Attention Seq2Seq', 'Transformer']
    params = [0.5, 2.1, 1.8, 3.5, 5.2, 8.7]  # Millions of parameters
    training_time = [1, 2.5, 2.2, 3, 4, 2]  # Relative training time
    performance = [65, 78, 76, 82, 89, 94]  # BLEU scores

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Parameters comparison
    ax1 = axes[0]
    bars1 = ax1.bar(models, params, color=COLORS['gradient'][:len(models)])
    ax1.set_ylabel('Parameters (Millions)', fontsize=12, color=COLORS['primary'])
    ax1.set_title('Model Complexity', fontsize=13, color=COLORS['primary'])
    ax1.tick_params(axis='x', rotation=45)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add value labels on bars
    for bar, val in zip(bars1, params):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val}M', ha='center', va='bottom', fontsize=10)

    # Training time comparison
    ax2 = axes[1]
    bars2 = ax2.bar(models, training_time, color=COLORS['gradient'][:len(models)])
    ax2.set_ylabel('Relative Training Time', fontsize=12, color=COLORS['primary'])
    ax2.set_title('Training Efficiency', fontsize=13, color=COLORS['primary'])
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Performance comparison
    ax3 = axes[2]
    bars3 = ax3.bar(models, performance, color=COLORS['gradient'][:len(models)])
    ax3.set_ylabel('BLEU Score', fontsize=12, color=COLORS['primary'])
    ax3.set_title('Translation Quality', fontsize=13, color=COLORS['primary'])
    ax3.set_ylim([60, 100])
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add value labels
    for bar, val in zip(bars3, performance):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('Seq2Seq Model Evolution: Complexity vs Performance Trade-offs',
                fontsize=15, color=COLORS['primary'], y=1.05)

    plt.tight_layout()
    plt.savefig('../figures/week4_model_complexity_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_beam_search_visualization():
    """Create an interactive-style beam search tree visualization"""
    set_style()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Define the beam search tree structure
    levels = [
        ['<START>'],
        ['Le', 'Un', 'La'],
        ['Le chat', 'Le chien', 'Un chat'],
        ['Le chat est', 'Le chat assis', 'Le chien est']
    ]

    scores = [
        [1.0],
        [0.8, 0.6, 0.5],
        [0.75, 0.65, 0.55],
        [0.7, 0.85, 0.6]
    ]

    # Node positions
    y_positions = [3, 2, 1, 0]
    x_spacing = 12

    # Draw nodes and connections
    for level_idx, (level, level_scores) in enumerate(zip(levels, scores)):
        x_positions = np.linspace(2, x_spacing-2, len(level))

        for node_idx, (node, score) in enumerate(zip(level, level_scores)):
            x = x_positions[node_idx]
            y = y_positions[level_idx]

            # Determine node color based on score
            if score >= 0.8:
                color = COLORS['accent']
            elif score >= 0.65:
                color = COLORS['secondary']
            else:
                color = COLORS['neutral']

            # Draw node
            circle = plt.Circle((x, y), 0.5, color=color, alpha=0.8, zorder=3)
            ax.add_patch(circle)

            # Add text
            ax.text(x, y, node, ha='center', va='center', fontsize=10,
                   weight='bold' if score >= 0.8 else 'normal', zorder=4)

            # Add score below node
            ax.text(x, y-0.7, f'{score:.2f}', ha='center', va='center',
                   fontsize=9, color=COLORS['primary'], zorder=4)

            # Draw connections to previous level
            if level_idx > 0:
                prev_x_positions = np.linspace(2, x_spacing-2, len(levels[level_idx-1]))
                for prev_x in prev_x_positions:
                    arrow = FancyArrowPatch((prev_x, y_positions[level_idx-1]-0.5),
                                          (x, y+0.5),
                                          arrowstyle='->', mutation_scale=15,
                                          color=COLORS['neutral'], alpha=0.3,
                                          linewidth=1 if score < 0.65 else 2,
                                          zorder=1)
                    ax.add_patch(arrow)

    # Add legend
    legend_elements = [
        mpatches.Patch(color=COLORS['accent'], label='High Score (≥0.8)'),
        mpatches.Patch(color=COLORS['secondary'], label='Medium Score (0.65-0.8)'),
        mpatches.Patch(color=COLORS['neutral'], label='Low Score (<0.65)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Styling
    ax.set_xlim(-1, x_spacing+1)
    ax.set_ylim(-1.5, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('Beam Search Visualization (Beam Width = 3)\nFrench Translation: "The cat is..."',
                fontsize=14, color=COLORS['primary'], pad=20)

    # Add step labels
    for i, y in enumerate(y_positions):
        ax.text(-0.5, y, f'Step {i}', ha='right', va='center',
               fontsize=11, color=COLORS['primary'], weight='bold')

    plt.tight_layout()
    plt.savefig('../figures/week4_beam_search_tree.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_training_dynamics_chart():
    """Create a multi-panel training dynamics visualization"""
    set_style()

    epochs = np.arange(1, 51)

    # Generate synthetic training data
    train_loss = 4.5 * np.exp(-epochs/10) + 0.5 + np.random.normal(0, 0.1, len(epochs))
    val_loss = 4.5 * np.exp(-epochs/12) + 0.7 + np.random.normal(0, 0.15, len(epochs))
    bleu_score = 100 * (1 - np.exp(-epochs/15)) + np.random.normal(0, 2, len(epochs))
    attention_entropy = 3 * np.exp(-epochs/20) + 0.5 + np.random.normal(0, 0.1, len(epochs))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_loss, label='Training Loss', color=COLORS['primary'], linewidth=2)
    ax1.plot(epochs, val_loss, label='Validation Loss', color=COLORS['accent'], linewidth=2, linestyle='--')
    ax1.fill_between(epochs, train_loss, val_loss, where=(val_loss > train_loss),
                     color=COLORS['accent'], alpha=0.2, label='Overfitting Region')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training vs Validation Loss', fontsize=12, color=COLORS['primary'])
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # BLEU score evolution
    ax2 = axes[0, 1]
    ax2.plot(epochs, bleu_score, color=COLORS['secondary'], linewidth=2)
    ax2.fill_between(epochs, bleu_score, alpha=0.3, color=COLORS['secondary'])
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('BLEU Score', fontsize=11)
    ax2.set_title('Translation Quality Evolution', fontsize=12, color=COLORS['primary'])
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3)

    # Attention entropy
    ax3 = axes[1, 0]
    ax3.plot(epochs, attention_entropy, color=COLORS['accent'], linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Attention Entropy', fontsize=11)
    ax3.set_title('Attention Focus Development', fontsize=12, color=COLORS['primary'])
    ax3.grid(True, alpha=0.3)

    # Learning rate schedule
    ax4 = axes[1, 1]
    lr_warmup = np.concatenate([np.linspace(0, 1, 10), np.ones(5)])
    lr_decay = 0.5 ** (np.arange(35) / 10)
    lr_schedule = np.concatenate([lr_warmup, lr_decay])

    ax4.plot(epochs, lr_schedule, color=COLORS['primary'], linewidth=2)
    ax4.fill_between(epochs[:15], lr_schedule[:15], alpha=0.3,
                     color=COLORS['secondary'], label='Warmup Phase')
    ax4.fill_between(epochs[15:], lr_schedule[15:], alpha=0.3,
                     color=COLORS['accent'], label='Decay Phase')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Learning Rate (Relative)', fontsize=11)
    ax4.set_title('Learning Rate Schedule', fontsize=12, color=COLORS['primary'])
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Seq2Seq Training Dynamics Dashboard', fontsize=15,
                color=COLORS['primary'], y=1.02)

    plt.tight_layout()
    plt.savefig('../figures/week4_training_dynamics.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_context_window_analysis():
    """Create a visualization of context window effects"""
    set_style()

    window_sizes = [5, 10, 20, 50, 100, 200]
    short_sentences = [82, 85, 86, 87, 87, 87]
    medium_sentences = [70, 78, 83, 85, 86, 86]
    long_sentences = [55, 65, 75, 82, 85, 85]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Performance by sentence length
    x = np.arange(len(window_sizes))
    width = 0.25

    bars1 = ax1.bar(x - width, short_sentences, width, label='Short (<10 words)',
                   color=COLORS['secondary'])
    bars2 = ax1.bar(x, medium_sentences, width, label='Medium (10-25 words)',
                   color=COLORS['primary'])
    bars3 = ax1.bar(x + width, long_sentences, width, label='Long (>25 words)',
                   color=COLORS['accent'])

    ax1.set_xlabel('Context Window Size', fontsize=12)
    ax1.set_ylabel('BLEU Score', fontsize=12)
    ax1.set_title('Context Window Impact on Translation Quality', fontsize=13, color=COLORS['primary'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(window_sizes)
    ax1.legend()
    ax1.set_ylim([50, 90])
    ax1.grid(True, alpha=0.3, axis='y')

    # Memory usage visualization
    memory_usage = [0.5, 1.2, 3.5, 12, 35, 120]
    ax2.semilogy(window_sizes, memory_usage, 'o-', color=COLORS['accent'],
                linewidth=2, markersize=8)
    ax2.fill_between(window_sizes, memory_usage, alpha=0.3, color=COLORS['accent'])
    ax2.set_xlabel('Context Window Size', fontsize=12)
    ax2.set_ylabel('Memory Usage (GB)', fontsize=12)
    ax2.set_title('Memory Requirements Scaling', fontsize=13, color=COLORS['primary'])
    ax2.grid(True, alpha=0.3)

    # Add annotations for key points
    ax2.annotate('Practical Limit', xy=(100, 35), xytext=(120, 10),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2),
                fontsize=11, color=COLORS['primary'])

    plt.suptitle('Context Window Analysis: Performance vs Resources',
                fontsize=15, color=COLORS['primary'], y=1.02)

    plt.tight_layout()
    plt.savefig('../figures/week4_context_window_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_architecture_evolution_timeline():
    """Create a timeline showing architecture evolution"""
    set_style()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Timeline data
    years = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
    milestones = [
        ('Basic Seq2Seq\n(Sutskever et al.)', 2014, 65),
        ('Attention\n(Bahdanau et al.)', 2015, 78),
        ('Google NMT\n(Wu et al.)', 2016, 85),
        ('Transformer\n(Vaswani et al.)', 2017, 92),
        ('BERT\n(Devlin et al.)', 2018, 94),
        ('GPT-2\n(Radford et al.)', 2019, 95),
        ('GPT-3\n(Brown et al.)', 2020, 97)
    ]

    # Create timeline
    ax.plot(years, [m[2] for m in milestones], 'o-', color=COLORS['primary'],
           linewidth=3, markersize=10)

    # Add milestone markers and labels
    for i, (name, year, score) in enumerate(milestones):
        # Determine color based on architecture type
        if 'Seq2Seq' in name or 'Attention' in name:
            color = COLORS['secondary']
        elif 'Transformer' in name or 'GPT' in name or 'BERT' in name:
            color = COLORS['accent']
        else:
            color = COLORS['primary']

        # Draw vertical line
        ax.vlines(year, 60, score, colors=color, linestyles='--', alpha=0.5)

        # Add text box
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2)
        ax.text(year, score + 2 + (i % 2) * 2, name, ha='center', va='bottom',
               fontsize=10, bbox=bbox_props, weight='bold')

    # Add shaded regions for different eras
    ax.axvspan(2014, 2016.5, alpha=0.1, color=COLORS['secondary'], label='Seq2Seq Era')
    ax.axvspan(2016.5, 2020, alpha=0.1, color=COLORS['accent'], label='Transformer Era')

    ax.set_xlabel('Year', fontsize=13, color=COLORS['primary'])
    ax.set_ylabel('Translation Quality (BLEU Score)', fontsize=13, color=COLORS['primary'])
    ax.set_title('Evolution of Neural Machine Translation Architectures',
                fontsize=15, color=COLORS['primary'], pad=20)
    ax.set_ylim([60, 105])
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/week4_architecture_evolution.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_charts():
    """Generate all enhanced charts"""
    print("Generating enhanced colorful charts for Week 4...")

    create_attention_heatmap()
    print("[DONE] Attention heatmap created")

    create_model_complexity_comparison()
    print("[DONE] Model complexity comparison created")

    create_beam_search_visualization()
    print("[DONE] Beam search visualization created")

    create_training_dynamics_chart()
    print("[DONE] Training dynamics dashboard created")

    create_context_window_analysis()
    print("[DONE] Context window analysis created")

    create_architecture_evolution_timeline()
    print("[DONE] Architecture evolution timeline created")

    print("\nAll enhanced charts generated successfully!")
    print("Color scheme used: Modern Tech Palette")
    print("Files saved in ../figures/")

if __name__ == "__main__":
    generate_all_charts()