"""
Generate all figures for LSTM Primer presentation
Comprehensive script covering autocomplete, gates, gradients, and memory cells
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Minimalist color scheme
COLOR_MAIN = '#404040'
COLOR_ACCENT = '#B4B4B4'
COLOR_LIGHT = '#F0F0F0'
COLOR_FORGET = '#FF6B6B'
COLOR_INPUT = '#4ECDC4'
COLOR_OUTPUT = '#FFD93D'
COLOR_GRADIENT_GOOD = '#2E7D32'
COLOR_GRADIENT_BAD = '#C62828'

OUTPUT_DIR = '../figures/'

plt.style.use('seaborn-v0_8-whitegrid')


def create_autocomplete_screenshot():
    """Figure 1: Phone keyboard autocomplete simulation"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Message bubble
    bubble = FancyBboxPatch((0.5, 6), 8, 2.5,
                            boxstyle="round,pad=0.15",
                            facecolor='lightblue', edgecolor=COLOR_MAIN, linewidth=2)
    ax.add_patch(bubble)
    ax.text(4.5, 7.5, '"I love chocolate ice cream but I prefer..."',
            fontsize=14, ha='center', va='center', fontweight='bold')

    # Suggestions bar
    suggestions = ['vanilla', 'strawberry', 'mint']
    colors = ['#90EE90', '#FFE4B5', '#F0E68C']

    for i, (word, color) in enumerate(zip(suggestions, colors)):
        x_pos = 1.5 + i * 2.5
        box = FancyBboxPatch((x_pos, 4), 2, 1,
                            boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor=COLOR_MAIN, linewidth=2)
        ax.add_patch(box)
        ax.text(x_pos + 1, 4.5, word, fontsize=13, ha='center', va='center',
                fontweight='bold')

    # Annotation
    ax.annotate('How does it know?\nIt remembered "chocolate" context!',
                xy=(4.5, 3.5), xytext=(4.5, 1.5),
                fontsize=12, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_MAIN))

    ax.set_title('Your Phone Predicts the Next Word', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}autocomplete_screenshot.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: autocomplete_screenshot.pdf")
    plt.close()


def create_context_window_comparison():
    """Figure 2: N-gram vs LSTM context windows"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    text = "The cat who was fluffy loved napping in sunny spots finally sat".split()

    # N-gram (2-word window)
    ax1.set_xlim(0, len(text))
    ax1.set_ylim(0, 2)
    ax1.set_title('N-Gram: Fixed 2-Word Window (Forgets "cat"!)',
                  fontsize=14, fontweight='bold', color=COLOR_GRADIENT_BAD)

    for i, word in enumerate(text):
        color = COLOR_GRADIENT_BAD if i == len(text)-2 or i == len(text)-1 else COLOR_LIGHT
        alpha = 1.0 if i >= len(text)-2 else 0.3
        ax1.text(i+0.5, 1, word, fontsize=11, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor=color, alpha=alpha, edgecolor=COLOR_MAIN))

    # Highlight window
    window = Rectangle((len(text)-2, 0.5), 2, 1, fill=False,
                       edgecolor=COLOR_GRADIENT_BAD, linewidth=3, linestyle='--')
    ax1.add_patch(window)
    ax1.text(len(text)-1, 0.2, 'Only sees these 2 words!', ha='center', fontsize=10,
            color=COLOR_GRADIENT_BAD, fontweight='bold')

    ax1.axis('off')

    # LSTM (selective memory)
    ax2.set_xlim(0, len(text))
    ax2.set_ylim(0, 2)
    ax2.set_title('LSTM: Selective Memory (Remembers "cat"!)',
                  fontsize=14, fontweight='bold', color=COLOR_GRADIENT_GOOD)

    important_indices = [1, 11, 13, 14]  # cat, spots, finally, sat

    for i, word in enumerate(text):
        if i in important_indices:
            color = COLOR_GRADIENT_GOOD
            alpha = 1.0
            fontweight = 'bold'
        else:
            color = COLOR_LIGHT
            alpha = 0.4
            fontweight = 'normal'

        ax2.text(i+0.5, 1, word, fontsize=11, ha='center', va='center', fontweight=fontweight,
                bbox=dict(boxstyle='round', facecolor=color, alpha=alpha, edgecolor=COLOR_MAIN))

    # Arrows showing memory
    ax2.annotate('', xy=(len(text)-0.5, 1), xytext=(1.5, 1),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_GRADIENT_GOOD, linestyle='--'))
    ax2.text(len(text)/2, 0.2, 'Remembers important words across distance!', ha='center',
            fontsize=10, color=COLOR_GRADIENT_GOOD, fontweight='bold')

    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}context_window_comparison.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: context_window_comparison.pdf")
    plt.close()


def create_lstm_architecture():
    """Figure 3: LSTM cell with three gates"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Cell state (top highway)
    cell_y = 9
    cell_path = FancyArrowPatch((1, cell_y), (11, cell_y),
                               arrowstyle='->', mutation_scale=30,
                               linewidth=4, color=COLOR_MAIN, zorder=1)
    ax.add_patch(cell_path)
    ax.text(6, cell_y+0.5, 'Cell State (Memory Highway)', ha='center',
           fontsize=12, fontweight='bold')

    # Forget Gate
    forget_box = Circle((3, 6), 0.8, facecolor=COLOR_FORGET, edgecolor=COLOR_MAIN, linewidth=2)
    ax.add_patch(forget_box)
    ax.text(3, 6, 'f', fontsize=18, ha='center', va='center', fontweight='bold', color='white')
    ax.text(3, 4.5, 'FORGET', ha='center', fontsize=11, fontweight='bold')
    ax.text(3, 4, 'Discard old info', ha='center', fontsize=9)

    # Arrow from forget to cell
    ax.annotate('', xy=(3, cell_y-0.3), xytext=(3, 6.8),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_FORGET))

    # Input Gate
    input_box = Circle((6, 6), 0.8, facecolor=COLOR_INPUT, edgecolor=COLOR_MAIN, linewidth=2)
    ax.add_patch(input_box)
    ax.text(6, 6, 'i', fontsize=18, ha='center', va='center', fontweight='bold', color='white')
    ax.text(6, 4.5, 'INPUT', ha='center', fontsize=11, fontweight='bold')
    ax.text(6, 4, 'Save new info', ha='center', fontsize=9)

    # Arrow from input to cell
    ax.annotate('', xy=(6, cell_y-0.3), xytext=(6, 6.8),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_INPUT))

    # Output Gate
    output_box = Circle((9, 6), 0.8, facecolor=COLOR_OUTPUT, edgecolor=COLOR_MAIN, linewidth=2)
    ax.add_patch(output_box)
    ax.text(9, 6, 'o', fontsize=18, ha='center', va='center', fontweight='bold', color='white')
    ax.text(9, 4.5, 'OUTPUT', ha='center', fontsize=11, fontweight='bold')
    ax.text(9, 4, 'Use memory', ha='center', fontsize=9)

    # Arrow from cell to output
    ax.annotate('', xy=(9, 6.8), xytext=(9, cell_y-0.3),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_OUTPUT))

    # Input/Output flows
    ax.annotate('', xy=(1, 6), xytext=(0, 6),
               arrowprops=dict(arrowstyle='->', lw=3, color=COLOR_MAIN))
    ax.text(0.5, 6.5, 'Input\n(word)', ha='center', fontsize=10)

    ax.annotate('', xy=(12, 6), xytext=(9.8, 6),
               arrowprops=dict(arrowstyle='->', lw=3, color=COLOR_MAIN))
    ax.text(11, 6.5, 'Prediction', ha='center', fontsize=10)

    # Title
    ax.text(6, 11, 'LSTM Cell: Three Gates Control Memory', ha='center',
           fontsize=16, fontweight='bold')

    # Traffic light analogy
    ax.text(6, 2, 'Like Traffic Lights: Red (forget) • Green (input) • Yellow (output)',
           ha='center', fontsize=11, style='italic', color=COLOR_ACCENT)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}lstm_architecture.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: lstm_architecture.pdf")
    plt.close()


def create_gradient_flow_comparison():
    """Figure 4: RNN vs LSTM gradient flow"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    timesteps = 10

    # RNN vanishing gradient
    ax1.set_xlim(0, timesteps+1)
    ax1.set_ylim(0, 10)
    ax1.set_title('RNN: Vanishing Gradient', fontsize=14, fontweight='bold',
                 color=COLOR_GRADIENT_BAD)

    # Draw cells
    for t in range(timesteps):
        circle = Circle((t+1, 5), 0.4, facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN)
        ax1.add_patch(circle)
        ax1.text(t+1, 5, f't{t}', ha='center', va='center', fontsize=8)

    # Gradient arrows (shrinking)
    for t in range(timesteps-1, 0, -1):
        alpha = (t / timesteps) ** 2  # Quadratic decay for visibility
        width = 2 * alpha
        ax1.annotate('', xy=(t, 5), xytext=(t+1, 5),
                    arrowprops=dict(arrowstyle='->', lw=width, color=COLOR_GRADIENT_BAD, alpha=alpha))

    # Gradient values
    gradients_rnn = [0.9 ** t for t in range(timesteps)]
    for t, grad in enumerate(gradients_rnn):
        ax1.text(t+1, 3, f'{grad:.3f}', ha='center', fontsize=8, color=COLOR_GRADIENT_BAD)

    ax1.text(timesteps/2, 1, 'Gradient shrinks exponentially: 0.9^10 ≈ 0.35',
            ha='center', fontsize=11, fontweight='bold', color=COLOR_GRADIENT_BAD)
    ax1.axis('off')

    # LSTM gradient highway
    ax2.set_xlim(0, timesteps+1)
    ax2.set_ylim(0, 10)
    ax2.set_title('LSTM: Gradient Highway', fontsize=14, fontweight='bold',
                 color=COLOR_GRADIENT_GOOD)

    # Draw cells
    for t in range(timesteps):
        circle = Circle((t+1, 5), 0.4, facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN)
        ax2.add_patch(circle)
        ax2.text(t+1, 5, f't{t}', ha='center', va='center', fontsize=8)

    # Highway (thick constant arrow)
    ax2.annotate('', xy=(1, 7), xytext=(timesteps, 7),
                arrowprops=dict(arrowstyle='->', lw=6, color=COLOR_GRADIENT_GOOD))
    ax2.text(timesteps/2, 7.7, 'Cell State Highway', ha='center', fontsize=11, fontweight='bold')

    # Gradient arrows (constant)
    for t in range(timesteps-1, 0, -1):
        ax2.annotate('', xy=(t, 5), xytext=(t+1, 5),
                    arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_GRADIENT_GOOD))

    # Gradient values
    gradients_lstm = [1.0 for _ in range(timesteps)]
    for t, grad in enumerate(gradients_lstm):
        ax2.text(t+1, 3, f'{grad:.2f}', ha='center', fontsize=8, color=COLOR_GRADIENT_GOOD)

    ax2.text(timesteps/2, 1, 'Gradient preserved: 1.0^10 = 1.0',
            ha='center', fontsize=11, fontweight='bold', color=COLOR_GRADIENT_GOOD)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}gradient_flow_comparison.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: gradient_flow_comparison.pdf")
    plt.close()


def create_gate_activation_heatmap():
    """Figure 5: Gate activations over a sentence"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

    # Sentence
    words = ["I", "love", "chocolate", ".", "But", "I", "prefer", "vanilla", "."]
    timesteps = len(words)

    # Simulated gate values
    forget_vals = np.array([0.1, 0.2, 0.3, 0.9, 0.1, 0.2, 0.3, 0.4, 0.9])
    input_vals = np.array([0.8, 0.9, 0.95, 0.1, 0.8, 0.7, 0.9, 0.95, 0.1])
    output_vals = np.array([0.3, 0.4, 0.6, 0.2, 0.3, 0.4, 0.7, 0.8, 0.3])

    # Forget gate
    ax1.imshow([forget_vals], cmap='Reds', aspect='auto', vmin=0, vmax=1)
    ax1.set_yticks([])
    ax1.set_xticks(range(timesteps))
    ax1.set_xticklabels(words, fontsize=12)
    ax1.set_title('Forget Gate (High = Forget Old Info)', fontsize=13, fontweight='bold')

    for i, val in enumerate(forget_vals):
        ax1.text(i, 0, f'{val:.1f}', ha='center', va='center',
                color='white' if val > 0.5 else 'black', fontweight='bold')

    # Annotation
    ax1.annotate('Forgets after period!', xy=(3, 0), xytext=(4, 1),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=10, fontweight='bold')

    # Input gate
    ax2.imshow([input_vals], cmap='Greens', aspect='auto', vmin=0, vmax=1)
    ax2.set_yticks([])
    ax2.set_xticks(range(timesteps))
    ax2.set_xticklabels(words, fontsize=12)
    ax2.set_title('Input Gate (High = Save New Info)', fontsize=13, fontweight='bold')

    for i, val in enumerate(input_vals):
        ax2.text(i, 0, f'{val:.2f}', ha='center', va='center',
                color='white' if val > 0.5 else 'black', fontweight='bold', fontsize=9)

    # Annotation
    ax2.annotate('Saves keywords!', xy=(2, 0), xytext=(1, 1),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=10, fontweight='bold')

    # Output gate
    ax3.imshow([output_vals], cmap='YlOrBr', aspect='auto', vmin=0, vmax=1)
    ax3.set_yticks([])
    ax3.set_xticks(range(timesteps))
    ax3.set_xticklabels(words, fontsize=12)
    ax3.set_title('Output Gate (High = Use Memory for Prediction)', fontsize=13, fontweight='bold')

    for i, val in enumerate(output_vals):
        ax3.text(i, 0, f'{val:.1f}', ha='center', va='center',
                color='white' if val > 0.5 else 'black', fontweight='bold')

    # Annotation
    ax3.annotate('Uses "prefer" strongly!', xy=(6, 0), xytext=(5, 1),
                arrowprops=dict(arrowstyle='->', lw=2, color='orange'),
                fontsize=10, fontweight='bold')

    plt.suptitle('LSTM Gate Activations: "I love chocolate. But I prefer vanilla."',
                fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}gate_activation_heatmap.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: gate_activation_heatmap.pdf")
    plt.close()


def create_training_progression():
    """Figure 6: Training progression showing improvement"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    prompt = "I love chocolate"

    # Epoch 1 - Random
    ax1.text(0.5, 0.8, 'Epoch 1: Random Initialization', ha='center', va='center',
            fontsize=14, fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.5, 0.6, f'Input: "{prompt}"', ha='center', va='center',
            fontsize=12, transform=ax1.transAxes, style='italic')
    ax1.text(0.5, 0.4, 'Prediction: "xjwkq"', ha='center', va='center',
            fontsize=13, color=COLOR_GRADIENT_BAD, transform=ax1.transAxes, fontweight='bold')
    ax1.text(0.5, 0.2, 'Loss: 8.5 (Gibberish!)', ha='center', va='center',
            fontsize=11, color=COLOR_GRADIENT_BAD, transform=ax1.transAxes)
    ax1.axis('off')
    ax1.set_facecolor('#FFE4E4')

    # Epoch 10 - Learning letters
    ax2.text(0.5, 0.8, 'Epoch 10: Learning Letters', ha='center', va='center',
            fontsize=14, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.5, 0.6, f'Input: "{prompt}"', ha='center', va='center',
            fontsize=12, transform=ax2.transAxes, style='italic')
    ax2.text(0.5, 0.4, 'Prediction: "cream"', ha='center', va='center',
            fontsize=13, color='orange', transform=ax2.transAxes, fontweight='bold')
    ax2.text(0.5, 0.2, 'Loss: 2.1 (Better!)', ha='center', va='center',
            fontsize=11, color='orange', transform=ax2.transAxes)
    ax2.axis('off')
    ax2.set_facecolor('#FFF4E4')

    # Epoch 50 - Learning context
    ax3.text(0.5, 0.8, 'Epoch 50: Understanding Context', ha='center', va='center',
            fontsize=14, fontweight='bold', transform=ax3.transAxes)
    ax3.text(0.5, 0.6, f'Input: "{prompt}"', ha='center', va='center',
            fontsize=12, transform=ax3.transAxes, style='italic')
    ax3.text(0.5, 0.4, 'Prediction: "ice cream"', ha='center', va='center',
            fontsize=13, color='#4CAF50', transform=ax3.transAxes, fontweight='bold')
    ax3.text(0.5, 0.2, 'Loss: 0.4 (Good!)', ha='center', va='center',
            fontsize=11, color='#4CAF50', transform=ax3.transAxes)
    ax3.axis('off')
    ax3.set_facecolor('#E8F5E9')

    # Epoch 200 - Fluent
    ax4.text(0.5, 0.8, 'Epoch 200: Fluent Generation', ha='center', va='center',
            fontsize=14, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.5, 0.6, f'Input: "{prompt}"', ha='center', va='center',
            fontsize=12, transform=ax4.transAxes, style='italic')
    ax4.text(0.5, 0.4, 'Prediction: "ice cream\nand strawberry cake"', ha='center', va='center',
            fontsize=13, color=COLOR_GRADIENT_GOOD, transform=ax4.transAxes, fontweight='bold')
    ax4.text(0.5, 0.1, 'Loss: 0.08 (Excellent!)', ha='center', va='center',
            fontsize=11, color=COLOR_GRADIENT_GOOD, transform=ax4.transAxes)
    ax4.axis('off')
    ax4.set_facecolor('#C8E6C9')

    plt.suptitle('LSTM Training: Watching It Learn', fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}training_progression.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: training_progression.pdf")
    plt.close()


def create_rnn_vs_lstm_comparison():
    """Figure 7: Side-by-side comparison table"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    table_data = [
        ['Feature', 'N-gram', 'RNN', 'LSTM'],
        ['Memory Type', 'Fixed window', 'Fading', 'Selective'],
        ['Long Context', '❌', '❌', '✓'],
        ['Parameters', 'Few', 'Moderate', 'Many'],
        ['Training Speed', 'Fast', 'Medium', 'Slow'],
        ['Vanishing Gradient', 'N/A', 'Yes ❌', 'Solved ✓'],
        ['Best For', 'Short (2-3 words)', 'Medium (10 words)', 'Long (50+ words)'],
        ['Example', '"I love..."', '"The cat sat..."', '"The cat, who was...finally..."']
    ]

    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)

    # Style header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#404040')
        cell.set_text_props(weight='bold', color='white', fontsize=12)

    # Style cells
    colors = ['#F0F0F0', '#FFE4E4', '#FFF4E4', '#E8F5E9']
    for row in range(1, 8):
        for col in range(4):
            cell = table[(row, col)]
            if col > 0:
                cell.set_facecolor(colors[col])
            cell.set_text_props(fontsize=10)

    ax.set_title('Comparison: N-gram vs RNN vs LSTM', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}model_comparison_table.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: model_comparison_table.pdf")
    plt.close()


def main():
    """Generate all LSTM primer figures"""
    print("="*70)
    print("Generating LSTM Primer Figures")
    print("="*70)
    print()

    create_autocomplete_screenshot()
    create_context_window_comparison()
    create_lstm_architecture()
    create_gradient_flow_comparison()
    create_gate_activation_heatmap()
    create_training_progression()
    create_rnn_vs_lstm_comparison()

    print()
    print("="*70)
    print("[SUCCESS] All 7 figures generated successfully!")
    print(f"Location: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()