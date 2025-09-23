"""
Generate Nature Professional themed charts for Week 4 Seq2Seq presentation
Using forest green, teal, and amber color palette
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import matplotlib.patches as mpatches

# Nature Professional Color Palette
NATURE_COLORS = {
    'primary': '#14532D',      # Forest Green
    'secondary': '#0D9488',    # Teal
    'accent': '#F59E0B',       # Amber
    'support': '#475569',      # Slate
    'background': '#F0FDF4',   # Mint Cream
    'light_green': '#86EFAC',  # Light Green
    'dark_teal': '#0F766E',    # Dark Teal
    'light_amber': '#FCD34D',  # Light Amber
    'gradient': ['#14532D', '#0D9488', '#F59E0B', '#86EFAC']
}

def set_nature_style():
    """Set the Nature Professional plot style"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.facecolor'] = NATURE_COLORS['background']
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = NATURE_COLORS['primary']
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.color'] = NATURE_COLORS['light_green']
    plt.rcParams['grid.alpha'] = 0.2
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['text.color'] = NATURE_COLORS['primary']
    plt.rcParams['axes.labelcolor'] = NATURE_COLORS['primary']
    plt.rcParams['xtick.color'] = NATURE_COLORS['primary']
    plt.rcParams['ytick.color'] = NATURE_COLORS['primary']

def create_attention_heatmap_nature():
    """Create Nature-themed attention visualization heatmap"""
    set_nature_style()

    # Generate sample attention weights
    source_words = ['The', 'cat', 'sat', 'on', 'mat', '<EOS>']
    target_words = ['Le', 'chat', 'assis', 'sur', 'tapis', '<EOS>']

    # Create realistic attention pattern
    attention = np.random.rand(len(target_words), len(source_words))
    for i in range(min(len(source_words), len(target_words))):
        attention[i, i] = attention[i, i] * 2 + 0.5
    attention = attention / attention.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create nature-themed colormap
    colors = ['#F0FDF4', NATURE_COLORS['light_green'], NATURE_COLORS['secondary'], NATURE_COLORS['primary']]
    n_bins = 100
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('nature', colors, N=n_bins)

    im = ax.imshow(attention, cmap=cmap, aspect='auto', alpha=0.9)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(source_words)))
    ax.set_yticks(np.arange(len(target_words)))
    ax.set_xticklabels(source_words, fontsize=12, color=NATURE_COLORS['primary'])
    ax.set_yticklabels(target_words, fontsize=12, color=NATURE_COLORS['primary'])

    # Add colorbar with nature styling
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20, fontsize=11, color=NATURE_COLORS['primary'])
    cbar.ax.yaxis.set_tick_params(color=NATURE_COLORS['primary'])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=NATURE_COLORS['primary'])

    # Add value annotations
    for i in range(len(target_words)):
        for j in range(len(source_words)):
            color = NATURE_COLORS['primary'] if attention[i, j] < 0.5 else 'white'
            text = ax.text(j, i, f'{attention[i, j]:.2f}',
                          ha="center", va="center", color=color,
                          fontsize=9, weight='bold')

    ax.set_xlabel('Source Sentence', fontsize=13, color=NATURE_COLORS['primary'], weight='bold')
    ax.set_ylabel('Target Sentence', fontsize=13, color=NATURE_COLORS['primary'], weight='bold')
    ax.set_title('Attention Mechanism Heatmap\nTranslation: English → French',
                fontsize=14, color=NATURE_COLORS['primary'], pad=20, weight='bold')

    # Add nature-themed border
    for spine in ax.spines.values():
        spine.set_edgecolor(NATURE_COLORS['primary'])
        spine.set_linewidth(2)

    plt.tight_layout()
    plt.savefig('../figures/week4_attention_heatmap_nature.pdf', dpi=300, bbox_inches='tight', facecolor=NATURE_COLORS['background'])
    plt.close()

def create_model_complexity_nature():
    """Create nature-themed model complexity comparison"""
    set_nature_style()

    models = ['Simple\nRNN', 'LSTM', 'GRU', 'Basic\nSeq2Seq', 'Attention\nSeq2Seq', 'Transformer']
    params = [0.5, 2.1, 1.8, 3.5, 5.2, 8.7]
    training_time = [1, 2.5, 2.2, 3, 4, 2]
    performance = [65, 78, 76, 82, 89, 94]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=NATURE_COLORS['background'])

    # Color gradient for bars
    bar_colors = [NATURE_COLORS['light_green'], NATURE_COLORS['secondary'],
                  NATURE_COLORS['dark_teal'], NATURE_COLORS['primary'],
                  NATURE_COLORS['accent'], NATURE_COLORS['light_amber']]

    # Parameters comparison
    ax1 = axes[0]
    bars1 = ax1.bar(models, params, color=bar_colors, edgecolor=NATURE_COLORS['primary'], linewidth=2)
    ax1.set_ylabel('Parameters (Millions)', fontsize=12, color=NATURE_COLORS['primary'], weight='bold')
    ax1.set_title('Model Complexity', fontsize=13, color=NATURE_COLORS['primary'], weight='bold')
    ax1.tick_params(axis='x', rotation=0, colors=NATURE_COLORS['primary'])
    ax1.tick_params(axis='y', colors=NATURE_COLORS['primary'])

    for bar, val in zip(bars1, params):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val}M', ha='center', va='bottom', fontsize=10,
                color=NATURE_COLORS['primary'], weight='bold')

    # Training time comparison
    ax2 = axes[1]
    bars2 = ax2.bar(models, training_time, color=bar_colors, edgecolor=NATURE_COLORS['primary'], linewidth=2)
    ax2.set_ylabel('Relative Training Time', fontsize=12, color=NATURE_COLORS['primary'], weight='bold')
    ax2.set_title('Training Efficiency', fontsize=13, color=NATURE_COLORS['primary'], weight='bold')
    ax2.tick_params(colors=NATURE_COLORS['primary'])

    # Performance comparison
    ax3 = axes[2]
    bars3 = ax3.bar(models, performance, color=bar_colors, edgecolor=NATURE_COLORS['primary'], linewidth=2)
    ax3.set_ylabel('BLEU Score', fontsize=12, color=NATURE_COLORS['primary'], weight='bold')
    ax3.set_title('Translation Quality', fontsize=13, color=NATURE_COLORS['primary'], weight='bold')
    ax3.set_ylim([60, 100])
    ax3.tick_params(colors=NATURE_COLORS['primary'])

    for bar, val in zip(bars3, performance):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val}', ha='center', va='bottom', fontsize=10,
                color=NATURE_COLORS['primary'], weight='bold')

    # Style all axes
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor(NATURE_COLORS['primary'])
            spine.set_linewidth(1.5)
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.2, color=NATURE_COLORS['light_green'])

    plt.suptitle('Seq2Seq Model Evolution: Nature Professional Theme',
                fontsize=15, color=NATURE_COLORS['primary'], y=1.05, weight='bold')

    plt.tight_layout()
    plt.savefig('../figures/week4_model_complexity_nature.pdf', dpi=300, bbox_inches='tight', facecolor=NATURE_COLORS['background'])
    plt.close()

def create_beam_search_nature():
    """Create nature-themed beam search visualization"""
    set_nature_style()

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=NATURE_COLORS['background'])

    # Tree structure
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

    y_positions = [3, 2, 1, 0]
    x_spacing = 12

    # Draw nodes and connections
    for level_idx, (level, level_scores) in enumerate(zip(levels, scores)):
        x_positions = np.linspace(2, x_spacing-2, len(level))

        for node_idx, (node, score) in enumerate(zip(level, level_scores)):
            x = x_positions[node_idx]
            y = y_positions[level_idx]

            # Nature-themed color based on score
            if score >= 0.8:
                color = NATURE_COLORS['primary']
                edge_color = NATURE_COLORS['dark_teal']
            elif score >= 0.65:
                color = NATURE_COLORS['secondary']
                edge_color = NATURE_COLORS['primary']
            else:
                color = NATURE_COLORS['light_green']
                edge_color = NATURE_COLORS['support']

            # Draw node with nature styling
            circle = plt.Circle((x, y), 0.5, color=color, alpha=0.8,
                               edgecolor=edge_color, linewidth=2, zorder=3)
            ax.add_patch(circle)

            # Add text
            text_color = 'white' if score >= 0.65 else NATURE_COLORS['primary']
            ax.text(x, y, node, ha='center', va='center', fontsize=10,
                   weight='bold', color=text_color, zorder=4)

            # Add score with amber accent
            ax.text(x, y-0.7, f'{score:.2f}', ha='center', va='center',
                   fontsize=9, color=NATURE_COLORS['accent'], weight='bold', zorder=4)

            # Draw connections
            if level_idx > 0:
                prev_x_positions = np.linspace(2, x_spacing-2, len(levels[level_idx-1]))
                for prev_x in prev_x_positions:
                    arrow = FancyArrowPatch((prev_x, y_positions[level_idx-1]-0.5),
                                          (x, y+0.5),
                                          arrowstyle='->', mutation_scale=15,
                                          color=NATURE_COLORS['secondary'],
                                          alpha=0.4 if score < 0.65 else 0.7,
                                          linewidth=1 if score < 0.65 else 2,
                                          zorder=1)
                    ax.add_patch(arrow)

    # Nature-themed legend
    legend_elements = [
        mpatches.Patch(color=NATURE_COLORS['primary'], label='High Score (≥0.8)'),
        mpatches.Patch(color=NATURE_COLORS['secondary'], label='Medium Score (0.65-0.8)'),
        mpatches.Patch(color=NATURE_COLORS['light_green'], label='Low Score (<0.65)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
             frameon=True, fancybox=True, shadow=True,
             facecolor=NATURE_COLORS['background'], edgecolor=NATURE_COLORS['primary'])

    ax.set_xlim(-1, x_spacing+1)
    ax.set_ylim(-1.5, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('Beam Search Visualization - Nature Professional Theme\nDecoding: "The cat is..."',
                fontsize=14, color=NATURE_COLORS['primary'], pad=20, weight='bold')

    # Add step labels with nature accent
    for i, y in enumerate(y_positions):
        ax.text(-0.5, y, f'Step {i}', ha='right', va='center',
               fontsize=11, color=NATURE_COLORS['accent'], weight='bold')

    plt.tight_layout()
    plt.savefig('../figures/week4_beam_search_nature.pdf', dpi=300, bbox_inches='tight', facecolor=NATURE_COLORS['background'])
    plt.close()

def create_training_dynamics_nature():
    """Create nature-themed training dynamics dashboard"""
    set_nature_style()

    epochs = np.arange(1, 51)
    train_loss = 4.5 * np.exp(-epochs/10) + 0.5 + np.random.normal(0, 0.1, len(epochs))
    val_loss = 4.5 * np.exp(-epochs/12) + 0.7 + np.random.normal(0, 0.15, len(epochs))
    bleu_score = 100 * (1 - np.exp(-epochs/15)) + np.random.normal(0, 2, len(epochs))
    attention_entropy = 3 * np.exp(-epochs/20) + 0.5 + np.random.normal(0, 0.1, len(epochs))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor=NATURE_COLORS['background'])

    # Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_loss, label='Training Loss', color=NATURE_COLORS['primary'], linewidth=2.5)
    ax1.plot(epochs, val_loss, label='Validation Loss', color=NATURE_COLORS['accent'], linewidth=2.5, linestyle='--')
    ax1.fill_between(epochs, train_loss, val_loss, where=(val_loss > train_loss),
                     color=NATURE_COLORS['light_amber'], alpha=0.3, label='Overfitting Region')
    ax1.set_xlabel('Epoch', fontsize=11, color=NATURE_COLORS['primary'])
    ax1.set_ylabel('Loss', fontsize=11, color=NATURE_COLORS['primary'])
    ax1.set_title('Training vs Validation Loss', fontsize=12, color=NATURE_COLORS['primary'], weight='bold')
    ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor=NATURE_COLORS['primary'])
    ax1.grid(True, alpha=0.2, color=NATURE_COLORS['light_green'])

    # BLEU score evolution
    ax2 = axes[0, 1]
    ax2.plot(epochs, bleu_score, color=NATURE_COLORS['secondary'], linewidth=2.5)
    ax2.fill_between(epochs, bleu_score, alpha=0.3, color=NATURE_COLORS['light_green'])
    ax2.set_xlabel('Epoch', fontsize=11, color=NATURE_COLORS['primary'])
    ax2.set_ylabel('BLEU Score', fontsize=11, color=NATURE_COLORS['primary'])
    ax2.set_title('Translation Quality Evolution', fontsize=12, color=NATURE_COLORS['primary'], weight='bold')
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.2, color=NATURE_COLORS['light_green'])

    # Attention entropy
    ax3 = axes[1, 0]
    gradient = ax3.scatter(epochs, attention_entropy, c=epochs, cmap='YlGn', s=20, alpha=0.6)
    ax3.plot(epochs, attention_entropy, color=NATURE_COLORS['dark_teal'], linewidth=2, alpha=0.8)
    ax3.set_xlabel('Epoch', fontsize=11, color=NATURE_COLORS['primary'])
    ax3.set_ylabel('Attention Entropy', fontsize=11, color=NATURE_COLORS['primary'])
    ax3.set_title('Attention Focus Development', fontsize=12, color=NATURE_COLORS['primary'], weight='bold')
    ax3.grid(True, alpha=0.2, color=NATURE_COLORS['light_green'])

    # Learning rate schedule
    ax4 = axes[1, 1]
    lr_warmup = np.concatenate([np.linspace(0, 1, 10), np.ones(5)])
    lr_decay = 0.5 ** (np.arange(35) / 10)
    lr_schedule = np.concatenate([lr_warmup, lr_decay])

    ax4.plot(epochs, lr_schedule, color=NATURE_COLORS['primary'], linewidth=2.5)
    ax4.fill_between(epochs[:15], lr_schedule[:15], alpha=0.3,
                     color=NATURE_COLORS['light_green'], label='Warmup Phase')
    ax4.fill_between(epochs[15:], lr_schedule[15:], alpha=0.3,
                     color=NATURE_COLORS['light_amber'], label='Decay Phase')
    ax4.set_xlabel('Epoch', fontsize=11, color=NATURE_COLORS['primary'])
    ax4.set_ylabel('Learning Rate (Relative)', fontsize=11, color=NATURE_COLORS['primary'])
    ax4.set_title('Learning Rate Schedule', fontsize=12, color=NATURE_COLORS['primary'], weight='bold')
    ax4.legend(loc='upper right', frameon=True, facecolor='white', edgecolor=NATURE_COLORS['primary'])
    ax4.grid(True, alpha=0.2, color=NATURE_COLORS['light_green'])

    # Style all axes
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_edgecolor(NATURE_COLORS['primary'])
            spine.set_linewidth(1.5)
        ax.set_facecolor('white')
        ax.tick_params(colors=NATURE_COLORS['primary'])

    plt.suptitle('Training Dynamics Dashboard - Nature Professional',
                fontsize=15, color=NATURE_COLORS['primary'], y=1.02, weight='bold')

    plt.tight_layout()
    plt.savefig('../figures/week4_training_dynamics_nature.pdf', dpi=300, bbox_inches='tight', facecolor=NATURE_COLORS['background'])
    plt.close()

def create_evolution_timeline_nature():
    """Create nature-themed architecture evolution timeline"""
    set_nature_style()

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=NATURE_COLORS['background'])

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

    # Main timeline with nature gradient
    scores = [m[2] for m in milestones]
    ax.plot(years, scores, 'o-', color=NATURE_COLORS['primary'],
           linewidth=3, markersize=12, markeredgecolor=NATURE_COLORS['dark_teal'],
           markeredgewidth=2, markerfacecolor=NATURE_COLORS['secondary'])

    # Add milestone markers
    for i, (name, year, score) in enumerate(milestones):
        if 'Seq2Seq' in name or 'Attention' in name or 'NMT' in name:
            color = NATURE_COLORS['secondary']
            box_color = NATURE_COLORS['light_green']
        else:
            color = NATURE_COLORS['accent']
            box_color = NATURE_COLORS['light_amber']

        # Vertical lines with nature style
        ax.vlines(year, 60, score, colors=color, linestyles='--', alpha=0.5, linewidth=2)

        # Text boxes with nature theme
        bbox_props = dict(boxstyle="round,pad=0.4", facecolor=box_color,
                         alpha=0.7, edgecolor=NATURE_COLORS['primary'], linewidth=1.5)
        y_offset = 2 + (i % 2) * 3
        ax.text(year, score + y_offset, name, ha='center', va='bottom',
               fontsize=10, bbox=bbox_props, weight='bold', color=NATURE_COLORS['primary'])

    # Shaded regions with nature colors
    ax.axvspan(2014, 2016.5, alpha=0.15, color=NATURE_COLORS['light_green'], label='Seq2Seq Era')
    ax.axvspan(2016.5, 2020, alpha=0.15, color=NATURE_COLORS['light_amber'], label='Transformer Era')

    # Styling
    ax.set_xlabel('Year', fontsize=13, color=NATURE_COLORS['primary'], weight='bold')
    ax.set_ylabel('Translation Quality (BLEU Score)', fontsize=13, color=NATURE_COLORS['primary'], weight='bold')
    ax.set_title('Evolution of Neural Machine Translation - Nature Professional',
                fontsize=15, color=NATURE_COLORS['primary'], pad=20, weight='bold')
    ax.set_ylim([60, 105])

    # Nature-themed legend
    ax.legend(loc='lower right', fontsize=11, frameon=True,
             facecolor=NATURE_COLORS['background'], edgecolor=NATURE_COLORS['primary'],
             framealpha=0.9)

    # Grid and spines
    ax.grid(True, alpha=0.2, color=NATURE_COLORS['light_green'])
    for spine in ax.spines.values():
        spine.set_edgecolor(NATURE_COLORS['primary'])
        spine.set_linewidth(2)

    ax.set_facecolor('white')
    ax.tick_params(colors=NATURE_COLORS['primary'])

    plt.tight_layout()
    plt.savefig('../figures/week4_evolution_timeline_nature.pdf', dpi=300, bbox_inches='tight', facecolor=NATURE_COLORS['background'])
    plt.close()

def generate_all_nature_charts():
    """Generate all Nature Professional themed charts"""
    print("Generating Nature Professional themed charts for Week 4...")

    create_attention_heatmap_nature()
    print("[DONE] Nature attention heatmap created")

    create_model_complexity_nature()
    print("[DONE] Nature model complexity comparison created")

    create_beam_search_nature()
    print("[DONE] Nature beam search visualization created")

    create_training_dynamics_nature()
    print("[DONE] Nature training dynamics dashboard created")

    create_evolution_timeline_nature()
    print("[DONE] Nature evolution timeline created")

    print("\nAll Nature Professional charts generated successfully!")
    print("Color scheme: Forest Green, Teal, Amber")
    print("Files saved in ../figures/")

if __name__ == "__main__":
    generate_all_nature_charts()