import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Ink & Accent Color Palette
COLORS = {
    'black': '#000000',          # Pure Black - main text
    'coral': '#EF4444',          # Warm Coral - primary accent
    'slate': '#64748B',          # Slate Gray - secondary text
    'green': '#10B981',          # Success Green
    'amber': '#F59E0B',          # Warning Amber
    'blue': '#3B82F6',           # Info Blue
    'gray': '#9CA3AF',           # Neutral Gray for borders
    'light_gray': '#F3F4F6',     # Very light gray for grid only
    'white': '#FFFFFF'           # Pure white backgrounds
}

# Set default style - minimal with white backgrounds
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = COLORS['gray']
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['grid.color'] = COLORS['light_gray']
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['text.color'] = COLORS['black']
plt.rcParams['axes.labelcolor'] = COLORS['black']
plt.rcParams['xtick.color'] = COLORS['slate']
plt.rcParams['ytick.color'] = COLORS['slate']

def create_attention_visualization():
    """Create attention heatmap with Ink & Accent colors"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')

    # Sample attention weights
    attention = np.array([
        [0.1, 0.7, 0.15, 0.05],
        [0.05, 0.1, 0.8, 0.05],
        [0.02, 0.08, 0.15, 0.75],
        [0.6, 0.2, 0.15, 0.05]
    ])

    # Left plot - Heatmap with custom colormap
    colors_map = plt.cm.colors.LinearSegmentedColormap.from_list('',
        ['white', COLORS['coral']])
    im = ax1.imshow(attention, cmap=colors_map, aspect='auto', vmin=0, vmax=1)

    # Add text annotations
    for i in range(4):
        for j in range(4):
            text = ax1.text(j, i, f'{attention[i,j]:.2f}',
                          ha='center', va='center',
                          color=COLORS['black'] if attention[i,j] < 0.5 else 'white',
                          fontsize=9)

    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels(['The', 'cat', 'sat', 'down'], color=COLORS['slate'])
    ax1.set_yticklabels(['Le', 'chat', 'est', 'assis'], color=COLORS['slate'])
    ax1.set_xlabel('Source', color=COLORS['black'], fontweight='bold')
    ax1.set_ylabel('Target', color=COLORS['black'], fontweight='bold')
    ax1.set_title('Attention Weights', color=COLORS['black'], fontweight='bold', pad=15)

    # Minimal grid
    ax1.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax1.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax1.grid(which='minor', color=COLORS['gray'], linestyle='-', linewidth=0.3)
    ax1.tick_params(which='minor', size=0)

    # Right plot - Line visualization
    positions = range(4)
    values = [0.7, 0.8, 0.75, 0.6]
    baseline = [0.25, 0.25, 0.25, 0.25]

    ax2.plot(positions, values, 'o-', color=COLORS['coral'],
             linewidth=2, markersize=8, label='With Attention')
    ax2.plot(positions, baseline, 's--', color=COLORS['blue'],
             linewidth=1.5, markersize=6, label='Baseline', alpha=0.7)

    ax2.set_xticks(positions)
    ax2.set_xticklabels(['Step 1', 'Step 2', 'Step 3', 'Step 4'], color=COLORS['slate'])
    ax2.set_ylabel('Alignment Score', color=COLORS['black'], fontweight='bold')
    ax2.set_xlabel('Decoding Step', color=COLORS['black'], fontweight='bold')
    ax2.set_title('Attention Focus', color=COLORS['black'], fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, color=COLORS['light_gray'])
    ax2.set_ylim([0, 1])

    # Clean legend
    ax2.legend(frameon=False, loc='upper right')

    # Remove top and right spines
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

    plt.suptitle('Ink & Accent Color Scheme - Clean Visualization',
                 color=COLORS['black'], fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/ink_accent_attention.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("[DONE] Generated ink_accent_attention.pdf")

def create_performance_comparison():
    """Create performance comparison chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')

    models = ['RNN', 'LSTM', 'GRU', 'Seq2Seq', 'Attention', 'Transformer']
    performance = [65, 72, 74, 78, 89, 95]

    # Create bars with different colors for emphasis
    colors = [COLORS['slate'], COLORS['slate'], COLORS['slate'],
              COLORS['blue'], COLORS['coral'], COLORS['green']]

    bars = ax.bar(models, performance, color=colors, edgecolor='none', width=0.7)

    # Add value labels on bars
    for bar, val in zip(bars, performance):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val}%', ha='center', va='bottom',
                color=COLORS['black'], fontweight='bold')

    ax.set_ylabel('BLEU Score', color=COLORS['black'], fontweight='bold', fontsize=11)
    ax.set_title('Model Performance Comparison', color=COLORS['black'],
                 fontweight='bold', fontsize=14, pad=20)
    ax.set_ylim([0, 105])

    # Minimal grid
    ax.grid(True, axis='y', alpha=0.3, color=COLORS['light_gray'])
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Rotate x labels slightly for better readability
    ax.set_xticklabels(models, rotation=15, ha='right', color=COLORS['slate'])

    # Add subtle annotation
    ax.text(0.98, 0.02, 'Higher is better', transform=ax.transAxes,
            ha='right', va='bottom', color=COLORS['slate'], fontsize=8, style='italic')

    plt.tight_layout()
    plt.savefig('../figures/ink_accent_performance.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("[DONE] Generated ink_accent_performance.pdf")

def create_training_dynamics():
    """Create training dynamics visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')

    epochs = np.arange(1, 51)

    # Training curves
    train_loss = 4.5 * np.exp(-epochs/15) + 0.3 + np.random.normal(0, 0.05, 50)
    val_loss = 4.5 * np.exp(-epochs/18) + 0.4 + np.random.normal(0, 0.08, 50)

    # Left plot - Loss curves
    ax1.plot(epochs, train_loss, color=COLORS['coral'], linewidth=2, label='Training Loss')
    ax1.plot(epochs, val_loss, color=COLORS['blue'], linewidth=2, label='Validation Loss', alpha=0.8)

    ax1.set_xlabel('Epoch', color=COLORS['black'], fontweight='bold')
    ax1.set_ylabel('Loss', color=COLORS['black'], fontweight='bold')
    ax1.set_title('Training Progress', color=COLORS['black'], fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, color=COLORS['light_gray'])
    ax1.legend(frameon=False, loc='upper right')

    # Right plot - Learning rate schedule
    lr = 0.001 * np.ones(20)
    lr = np.concatenate([lr, 0.001 * 0.5 * np.ones(15)])
    lr = np.concatenate([lr, 0.001 * 0.25 * np.ones(15)])

    ax2.plot(epochs, lr, color=COLORS['green'], linewidth=2)
    ax2.fill_between(epochs, 0, lr, color=COLORS['green'], alpha=0.1)

    ax2.set_xlabel('Epoch', color=COLORS['black'], fontweight='bold')
    ax2.set_ylabel('Learning Rate', color=COLORS['black'], fontweight='bold')
    ax2.set_title('Learning Rate Schedule', color=COLORS['black'], fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, color=COLORS['light_gray'])

    # Add step indicators
    ax2.axvline(x=20, color=COLORS['amber'], linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(x=35, color=COLORS['amber'], linestyle='--', linewidth=1, alpha=0.5)
    ax2.text(20, ax2.get_ylim()[1]*0.9, 'Step 1', color=COLORS['amber'], fontsize=8, ha='center')
    ax2.text(35, ax2.get_ylim()[1]*0.9, 'Step 2', color=COLORS['amber'], fontsize=8, ha='center')

    # Clean up spines
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

    plt.suptitle('Training Dynamics - Clean & Professional',
                 color=COLORS['black'], fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/ink_accent_training.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("[DONE] Generated ink_accent_training.pdf")

def create_architecture_diagram():
    """Create clean architecture diagram"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Encoder blocks
    for i in range(3):
        rect = FancyBboxPatch((1 + i*1.2, 2), 1, 1.5,
                              boxstyle="round,pad=0.05",
                              facecolor='white',
                              edgecolor=COLORS['blue'],
                              linewidth=1.5)
        ax.add_patch(rect)
        ax.text(1.5 + i*1.2, 2.75, f'E{i+1}', ha='center', va='center',
                color=COLORS['blue'], fontweight='bold')

    # Context vector
    rect = FancyBboxPatch((5.5, 2.25), 1.5, 1,
                         boxstyle="round,pad=0.05",
                         facecolor='white',
                         edgecolor=COLORS['coral'],
                         linewidth=2)
    ax.add_patch(rect)
    ax.text(6.25, 2.75, 'Context', ha='center', va='center',
            color=COLORS['coral'], fontweight='bold')

    # Decoder blocks
    for i in range(3):
        rect = FancyBboxPatch((8 + i*1.2, 2), 1, 1.5,
                              boxstyle="round,pad=0.05",
                              facecolor='white',
                              edgecolor=COLORS['green'],
                              linewidth=1.5)
        ax.add_patch(rect)
        ax.text(8.5 + i*1.2, 2.75, f'D{i+1}', ha='center', va='center',
                color=COLORS['green'], fontweight='bold')

    # Arrows
    for i in range(2):
        ax.arrow(2 + i*1.2, 2.75, 0.9, 0, head_width=0.15, head_length=0.1,
                fc=COLORS['gray'], ec=COLORS['gray'], alpha=0.5)

    ax.arrow(4.6, 2.75, 0.8, 0, head_width=0.15, head_length=0.1,
            fc=COLORS['coral'], ec=COLORS['coral'])
    ax.arrow(7, 2.75, 0.9, 0, head_width=0.15, head_length=0.1,
            fc=COLORS['coral'], ec=COLORS['coral'])

    for i in range(2):
        ax.arrow(9 + i*1.2, 2.75, 0.9, 0, head_width=0.15, head_length=0.1,
                fc=COLORS['gray'], ec=COLORS['gray'], alpha=0.5)

    # Labels
    ax.text(2.5, 1.3, 'Encoder', ha='center', color=COLORS['black'],
            fontsize=12, fontweight='bold')
    ax.text(9.5, 1.3, 'Decoder', ha='center', color=COLORS['black'],
            fontsize=12, fontweight='bold')
    ax.text(6.25, 4, 'Sequence-to-Sequence Architecture', ha='center',
            color=COLORS['black'], fontsize=14, fontweight='bold')

    # Input/Output labels
    ax.text(0.5, 2.75, 'Input', ha='center', va='center',
            color=COLORS['slate'], fontsize=10)
    ax.text(11.5, 2.75, 'Output', ha='center', va='center',
            color=COLORS['slate'], fontsize=10)

    # Description
    ax.text(6, 0.5, 'Clean design with no background colors - text and borders only',
            ha='center', color=COLORS['slate'], fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig('../figures/ink_accent_architecture.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("[DONE] Generated ink_accent_architecture.pdf")

if __name__ == '__main__':
    print("Generating Ink & Accent color scheme figures...")
    print("=" * 50)

    create_attention_visualization()
    create_performance_comparison()
    create_training_dynamics()
    create_architecture_diagram()

    print("=" * 50)
    print("All Ink & Accent figures generated successfully!")
    print("Color scheme: No backgrounds, clean and professional")