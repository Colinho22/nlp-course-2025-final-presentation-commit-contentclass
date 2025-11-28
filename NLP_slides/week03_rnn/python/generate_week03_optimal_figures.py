"""
Generate optimal readability figures for Week 3: RNNs
Using monochromatic style with high contrast
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import numpy as np
import seaborn as sns
from matplotlib.patches import Arc
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define color palette (optimal readability)
COLOR_BLACK = '#000000'      # Pure black for text
COLOR_BLUE = '#003D7A'        # Deep blue for primary
COLOR_GRAY = '#4A4A4A'        # Dark gray for secondary
COLOR_LIGHT = '#E5E5E5'       # Light gray for backgrounds
COLOR_ACCENT = '#0066CC'      # Chart blue
COLOR_ORANGE = '#FF8800'      # Chart orange
COLOR_TEAL = '#00A0A0'        # Chart teal
COLOR_PURPLE = '#8B4789'      # Chart purple
COLOR_GREEN = '#228B22'       # Success green
COLOR_RED = '#CC0000'         # Warning red

def setup_figure(figsize=(10, 6)):
    """Setup figure with consistent style"""
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax

def save_figure(fig, filename):
    """Save figure with consistent settings"""
    fig.savefig(f'../figures/{filename}',
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()
    print(f"Generated: {filename}")

# 1. Course Timeline
def create_course_timeline():
    fig, ax = setup_figure((12, 4))

    weeks = np.arange(1, 13)
    topics = ['N-grams', 'Word2Vec', 'RNNs', 'Seq2Seq', 'Transformers', 'BERT/GPT',
              'Advanced', 'Tokenization', 'Decoding', 'Fine-tuning', 'Efficiency', 'Ethics']

    # Main timeline
    ax.plot([0.5, 12.5], [0, 0], 'k-', linewidth=3)

    for i, (week, topic) in enumerate(zip(weeks, topics)):
        # Week markers
        if week == 3:  # Current week
            color = COLOR_RED
            size = 150
            ax.scatter(week, 0, s=size, c=color, zorder=5)
            ax.text(week, -0.5, f'Week {week}', ha='center', fontsize=10, fontweight='bold', color=color)
            ax.text(week, 0.5, topic, ha='center', fontsize=10, fontweight='bold', color=color)
            # Highlight arrow
            ax.annotate('You are here', xy=(week, 0), xytext=(week, -1.2),
                       arrowprops=dict(arrowstyle='->', color=COLOR_RED, lw=2),
                       ha='center', fontsize=11, color=COLOR_RED, fontweight='bold')
        else:
            color = COLOR_BLUE if week < 3 else COLOR_GRAY
            ax.scatter(week, 0, s=100, c=color, zorder=4)
            ax.text(week, -0.5, f'Week {week}', ha='center', fontsize=9, color=color)
            ax.text(week, 0.5, topic, ha='center', fontsize=9, color=color)

    ax.set_xlim(0, 13)
    ax.set_ylim(-1.5, 1)
    ax.set_title('NLP Course Journey', fontsize=14, fontweight='bold', pad=20)

    save_figure(fig, 'course_timeline.pdf')

# 2. RNN Unrolled View
def create_rnn_unrolled():
    fig, ax = setup_figure((12, 5))

    # Folded view (left)
    ax.text(1.5, 4, 'Folded View', ha='center', fontsize=12, fontweight='bold')

    # RNN cell
    rect = FancyBboxPatch((0.5, 1.5), 2, 2,
                          boxstyle="round,pad=0.1",
                          facecolor=COLOR_LIGHT,
                          edgecolor=COLOR_BLUE, linewidth=2)
    ax.add_patch(rect)
    ax.text(1.5, 2.5, 'RNN', ha='center', fontsize=11, fontweight='bold')

    # Self-loop
    arc = Arc((1.5, 3.5), 1.5, 1, angle=0, theta1=30, theta2=150,
              color=COLOR_BLUE, linewidth=2)
    ax.add_patch(arc)
    ax.annotate('', xy=(2.2, 3.2), xytext=(2.3, 3.4),
               arrowprops=dict(arrowstyle='->', color=COLOR_BLUE, lw=2))

    # Arrow to unrolled
    ax.annotate('', xy=(4, 2.5), xytext=(3, 2.5),
               arrowprops=dict(arrowstyle='->', color=COLOR_RED, lw=3))
    ax.text(3.5, 3, 'Unroll', ha='center', fontsize=10, color=COLOR_RED)

    # Unrolled view (right)
    ax.text(7.5, 4, 'Unrolled View', ha='center', fontsize=12, fontweight='bold')

    timesteps = ['t-1', 't', 't+1']
    for i, t in enumerate(timesteps):
        x = 5 + i * 2.5

        # RNN cell
        rect = FancyBboxPatch((x-0.5, 1.5), 1, 2,
                             boxstyle="round,pad=0.05",
                             facecolor=COLOR_LIGHT,
                             edgecolor=COLOR_BLUE, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 2.5, 'RNN', ha='center', fontsize=10)

        # Input
        ax.annotate('', xy=(x, 1.5), xytext=(x, 0.5),
                   arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=2))
        ax.text(x, 0.2, f'$x_{{{t}}}$', ha='center', fontsize=10)

        # Output
        ax.annotate('', xy=(x, 4.5), xytext=(x, 3.5),
                   arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=2))
        ax.text(x, 4.8, f'$y_{{{t}}}$', ha='center', fontsize=10)

        # Hidden state connection
        if i < len(timesteps) - 1:
            ax.annotate('', xy=(x+1.5, 2.5), xytext=(x+0.5, 2.5),
                       arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=2))
            ax.text(x+1, 2.8, f'$h_{{{t}}}$', ha='center', fontsize=9, color=COLOR_GREEN)

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)

    save_figure(fig, 'rnn_unrolled.pdf')

# 3. RNN Forward Pass
def create_rnn_forward_pass():
    fig, ax = setup_figure((12, 6))

    sequence = ['The', 'cat', 'sat', 'on']

    for i, word in enumerate(sequence):
        x = 1.5 + i * 2.5

        # Input word
        ax.text(x, 0.5, word, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor=COLOR_ACCENT, alpha=0.3))

        # RNN cell
        rect = FancyBboxPatch((x-0.6, 2), 1.2, 1.5,
                             boxstyle="round,pad=0.05",
                             facecolor='white',
                             edgecolor=COLOR_BLUE, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 2.75, f'RNN', ha='center', fontsize=10, fontweight='bold')

        # Hidden state value (example)
        h_values = ['[0.2, -0.1, 0.5]', '[0.4, 0.3, -0.2]', '[-0.1, 0.6, 0.3]', '[0.5, 0.2, 0.1]']
        ax.text(x, 2.3, f'$h_{i}$', ha='center', fontsize=8, style='italic')

        # Input arrow
        ax.annotate('', xy=(x, 2), xytext=(x, 1),
                   arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=2))

        # Output
        outputs = ['cat', 'sat', 'on', 'the']
        ax.text(x, 4.5, outputs[i], ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor=COLOR_ORANGE, alpha=0.3))
        ax.annotate('', xy=(x, 4.2), xytext=(x, 3.5),
                   arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=2))

        # Hidden state flow
        if i < len(sequence) - 1:
            ax.annotate('', xy=(x+1.9, 2.75), xytext=(x+0.6, 2.75),
                       arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=2))

        # Initial hidden state
        if i == 0:
            ax.text(x-1.2, 2.75, '$h_0 = 0$', ha='center', fontsize=9, color=COLOR_GREEN)
            ax.annotate('', xy=(x-0.6, 2.75), xytext=(x-1.2, 2.75),
                       arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=2))

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.set_title('RNN Forward Pass: Processing "The cat sat on"', fontsize=12, fontweight='bold')

    save_figure(fig, 'rnn_forward_pass.pdf')

# 4. Vanishing Gradient Visualization
def create_vanishing_gradient():
    fig, ax = setup_figure((10, 6))

    timesteps = np.arange(1, 21)

    # Different gradient decay rates
    normal_gradient = 0.9 ** timesteps
    severe_gradient = 0.7 ** timesteps
    lstm_gradient = 0.98 ** timesteps

    ax.semilogy(timesteps, normal_gradient, 'o-', color=COLOR_ORANGE,
                linewidth=2, label='Vanilla RNN (λ=0.9)', markersize=6)
    ax.semilogy(timesteps, severe_gradient, 's-', color=COLOR_RED,
                linewidth=2, label='Vanilla RNN (λ=0.7)', markersize=6)
    ax.semilogy(timesteps, lstm_gradient, '^-', color=COLOR_GREEN,
                linewidth=2, label='LSTM (λ=0.98)', markersize=6)

    # Add threshold line
    ax.axhline(y=0.001, color=COLOR_GRAY, linestyle='--', alpha=0.5)
    ax.text(15, 0.0015, 'Gradient ≈ 0', fontsize=9, color=COLOR_GRAY)

    # Annotations
    ax.annotate('Gradients vanish\nexponentially', xy=(10, 0.01), xytext=(6, 0.1),
               arrowprops=dict(arrowstyle='->', color=COLOR_RED),
               fontsize=10, color=COLOR_RED)

    ax.annotate('LSTM maintains\ngradient flow', xy=(15, 0.3), xytext=(11, 0.5),
               arrowprops=dict(arrowstyle='->', color=COLOR_GREEN),
               fontsize=10, color=COLOR_GREEN)

    ax.set_xlabel('Timesteps', fontsize=12)
    ax.set_ylabel('Gradient Magnitude', fontsize=12)
    ax.set_title('Vanishing Gradient Problem in RNNs', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 21])
    ax.set_ylim([0.0001, 1.5])

    # Reset to show grid and axes
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_xticks(np.arange(0, 21, 5))

    save_figure(fig, 'vanishing_gradient.pdf')

# 5. Gradient Flow Comparison
def create_gradient_flow_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')

    # Vanilla RNN (left)
    ax = axes[0]
    ax.set_facecolor('white')
    ax.set_title('Vanilla RNN Gradient Flow', fontsize=12, fontweight='bold')

    for i in range(4):
        x = i * 2
        # Cell
        rect = Rectangle((x, 1), 1.5, 2, facecolor=COLOR_LIGHT,
                        edgecolor=COLOR_BLUE, linewidth=2)
        ax.add_patch(rect)
        ax.text(x+0.75, 2, f't={i}', ha='center', fontsize=10)

        # Gradient flow (backward)
        if i > 0:
            width = 2 - 0.4 * i  # Decreasing width
            alpha = max(0.2, 1 - 0.25 * i)  # Fading
            arrow = FancyArrowPatch((x-0.25, 2), (x-1.75, 2),
                                  arrowstyle='->', lw=width,
                                  color=COLOR_RED, alpha=alpha)
            ax.add_patch(arrow)
            ax.text(x-1, 2.5, f'×0.9', ha='center', fontsize=8, color=COLOR_RED)

    ax.text(3, 0.5, 'Gradient vanishes through multiplications',
            ha='center', fontsize=10, style='italic', color=COLOR_RED)

    ax.set_xlim(-1, 8)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # LSTM (right)
    ax = axes[1]
    ax.set_facecolor('white')
    ax.set_title('LSTM Gradient Flow', fontsize=12, fontweight='bold')

    for i in range(4):
        x = i * 2
        # Cell
        rect = Rectangle((x, 1), 1.5, 2, facecolor=COLOR_LIGHT,
                        edgecolor=COLOR_GREEN, linewidth=2)
        ax.add_patch(rect)
        ax.text(x+0.75, 2, f't={i}', ha='center', fontsize=10)

        # Cell state highway (top)
        if i < 3:
            arrow = FancyArrowPatch((x+1.5, 3.5), (x+2, 3.5),
                                  arrowstyle='->', lw=3,
                                  color=COLOR_GREEN)
            ax.add_patch(arrow)

        # Gradient flow (backward) - constant
        if i > 0:
            arrow = FancyArrowPatch((x-0.25, 3.5), (x-1.75, 3.5),
                                  arrowstyle='->', lw=3,
                                  color=COLOR_GREEN, alpha=0.8)
            ax.add_patch(arrow)
            ax.text(x-1, 3.8, f'+', ha='center', fontsize=10,
                   color=COLOR_GREEN, fontweight='bold')

    ax.text(3, 0.5, 'Gradient flows through cell state highway',
            ha='center', fontsize=10, style='italic', color=COLOR_GREEN)

    ax.set_xlim(-1, 8)
    ax.set_ylim(0, 4.5)
    ax.axis('off')

    plt.tight_layout()
    save_figure(fig, 'gradient_flow_comparison.pdf')

# 6. LSTM Architecture
def create_lstm_architecture():
    fig, ax = setup_figure((12, 8))

    # Cell state line
    ax.plot([1, 9], [6, 6], 'k-', linewidth=3)
    ax.text(5, 6.5, 'Cell State $C_t$', ha='center', fontsize=11, fontweight='bold')

    # Gates
    gates = [
        (2.5, 'Forget\nGate', COLOR_RED),
        (5, 'Input\nGate', COLOR_BLUE),
        (7.5, 'Output\nGate', COLOR_ORANGE)
    ]

    for x, name, color in gates:
        # Gate box
        rect = FancyBboxPatch((x-0.6, 3.5), 1.2, 1.5,
                             boxstyle="round,pad=0.05",
                             facecolor=color, alpha=0.3,
                             edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 4.25, name, ha='center', fontsize=10, fontweight='bold')

        # Sigmoid symbol
        ax.text(x, 3.8, 'σ', ha='center', fontsize=14, fontweight='bold')

        # Connection to cell state
        ax.annotate('', xy=(x, 5.5), xytext=(x, 5),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))

    # Input modulation
    rect = FancyBboxPatch((4.4, 1.5), 1.2, 1.5,
                         boxstyle="round,pad=0.05",
                         facecolor=COLOR_TEAL, alpha=0.3,
                         edgecolor=COLOR_TEAL, linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 2.25, 'Candidate\nValues', ha='center', fontsize=9)
    ax.text(5, 1.8, 'tanh', ha='center', fontsize=12, fontweight='bold')

    # Inputs and outputs
    ax.text(1, 1, '$x_t$', ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor=COLOR_ACCENT, alpha=0.3))
    ax.text(1, 2.5, '$h_{t-1}$', ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, alpha=0.3))

    ax.text(9, 2.5, '$h_t$', ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, alpha=0.3))

    # Operations
    operations = [(2.5, 6, '×'), (5, 6, '+'), (7.5, 4, '×')]
    for x, y, op in operations:
        circle = Circle((x, y), 0.3, facecolor='white',
                       edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, op, ha='center', va='center', fontsize=14, fontweight='bold')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_title('LSTM Architecture: Information Flow Through Gates',
                fontsize=14, fontweight='bold', pad=20)

    save_figure(fig, 'lstm_architecture.pdf')

# 7. GRU Architecture
def create_gru_architecture():
    fig, ax = setup_figure((10, 6))

    # Main flow line
    ax.plot([1, 8], [4, 4], 'k-', linewidth=3)
    ax.text(4.5, 4.5, 'Hidden State $h_t$', ha='center', fontsize=11, fontweight='bold')

    # Gates
    gates = [
        (2.5, 'Reset\nGate', COLOR_RED, 'r'),
        (5.5, 'Update\nGate', COLOR_BLUE, 'z')
    ]

    for x, name, color, symbol in gates:
        rect = FancyBboxPatch((x-0.6, 2), 1.2, 1.5,
                             boxstyle="round,pad=0.05",
                             facecolor=color, alpha=0.3,
                             edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 2.75, name, ha='center', fontsize=10, fontweight='bold')
        ax.text(x, 2.3, f'${symbol}_t$', ha='center', fontsize=11)

    # Candidate hidden state
    rect = FancyBboxPatch((3.4, 0.5), 1.2, 1.2,
                         boxstyle="round,pad=0.05",
                         facecolor=COLOR_TEAL, alpha=0.3,
                         edgecolor=COLOR_TEAL, linewidth=2)
    ax.add_patch(rect)
    ax.text(4, 1.1, '$\\tilde{h}_t$', ha='center', fontsize=11, fontweight='bold')

    # Mathematical operations
    ax.text(6.5, 5, '$(1-z_t) \\cdot h_{t-1} + z_t \\cdot \\tilde{h}_t$',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

    # Inputs
    ax.text(0.5, 1, '$x_t$', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor=COLOR_ACCENT, alpha=0.3))
    ax.text(0.5, 2.5, '$h_{t-1}$', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, alpha=0.3))

    ax.set_xlim(0, 9)
    ax.set_ylim(0, 6)
    ax.set_title('GRU Architecture: Simplified Gating',
                fontsize=14, fontweight='bold', pad=20)

    save_figure(fig, 'gru_architecture.pdf')

# 8. Bidirectional RNN
def create_bidirectional_rnn():
    fig, ax = setup_figure((12, 6))

    words = ['The', 'bank', 'by', 'the', 'river']

    # Forward RNN
    for i, word in enumerate(words):
        x = 1 + i * 2

        # Word
        ax.text(x, 3, word, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT))

        # Forward cell
        rect = Rectangle((x-0.4, 4), 0.8, 1, facecolor=COLOR_BLUE, alpha=0.3,
                        edgecolor=COLOR_BLUE, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 4.5, '→', ha='center', fontsize=14, fontweight='bold')

        # Backward cell
        rect = Rectangle((x-0.4, 1.5), 0.8, 1, facecolor=COLOR_ORANGE, alpha=0.3,
                        edgecolor=COLOR_ORANGE, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 2, '←', ha='center', fontsize=14, fontweight='bold')

        # Forward connections
        if i < len(words) - 1:
            ax.annotate('', xy=(x+1.6, 4.5), xytext=(x+0.4, 4.5),
                       arrowprops=dict(arrowstyle='->', color=COLOR_BLUE, lw=2))

        # Backward connections
        if i > 0:
            ax.annotate('', xy=(x-1.6, 2), xytext=(x-0.4, 2),
                       arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=2))

        # Concatenation
        ax.plot([x, x], [2.5, 4], 'k--', alpha=0.5)
        circle = Circle((x, 3.25), 0.2, facecolor='white',
                       edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, 3.25, '⊕', ha='center', va='center', fontsize=10)

        # Output
        ax.annotate('', xy=(x, 5.8), xytext=(x, 5.2),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Labels
    ax.text(0.2, 4.5, 'Forward', fontsize=10, color=COLOR_BLUE, fontweight='bold')
    ax.text(0.2, 2, 'Backward', fontsize=10, color=COLOR_ORANGE, fontweight='bold')

    # Highlight "bank" ambiguity resolution
    ax.add_patch(Rectangle((2.5, 1.3), 1, 4, fill=False,
                          edgecolor=COLOR_RED, linewidth=2, linestyle='--'))
    ax.text(3, 0.8, '"bank" disambiguated\nusing future context',
            ha='center', fontsize=9, color=COLOR_RED, style='italic')

    ax.set_xlim(0, 10)
    ax.set_ylim(0.5, 6)
    ax.set_title('Bidirectional RNN: Using Past and Future Context',
                fontsize=12, fontweight='bold', pad=20)

    save_figure(fig, 'bidirectional_rnn.pdf')

# 9. RNN vs Transformer Timeline
def create_rnn_vs_transformer_timeline():
    fig, ax = setup_figure((12, 6))

    years = np.array([2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    rnn_usage = np.array([90, 92, 95, 90, 85, 60, 35, 20, 15, 12, 10, 10])
    transformer_usage = np.array([0, 0, 0, 0, 5, 30, 55, 75, 82, 85, 87, 88])

    ax.fill_between(years, 0, rnn_usage, color=COLOR_BLUE, alpha=0.3, label='RNN/LSTM')
    ax.fill_between(years, 0, transformer_usage, color=COLOR_ORANGE, alpha=0.3, label='Transformers')

    ax.plot(years, rnn_usage, 'o-', color=COLOR_BLUE, linewidth=2, markersize=6)
    ax.plot(years, transformer_usage, 's-', color=COLOR_ORANGE, linewidth=2, markersize=6)

    # Key events
    events = [
        (2014, 85, 'GRU\nintroduced'),
        (2015, 90, 'Attention\nmechanism'),
        (2017, 45, 'Transformer\npaper'),
        (2018, 45, 'BERT\nreleased'),
        (2020, 50, 'GPT-3\nscale')
    ]

    for year, y, text in events:
        ax.annotate(text, xy=(year, y), xytext=(year, y+15),
                   arrowprops=dict(arrowstyle='->', color=COLOR_GRAY, lw=1),
                   ha='center', fontsize=8, color=COLOR_GRAY)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Relative Usage in Research (%)', fontsize=12)
    ax.set_title('The Evolution: RNNs to Transformers', fontsize=14, fontweight='bold')
    ax.legend(loc='right', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([2013, 2024])
    ax.set_ylim([0, 100])

    # Show axes
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_xticks(years[::2])

    save_figure(fig, 'rnn_vs_transformer_timeline.pdf')

# 10. Evolution to Transformers
def create_evolution_to_transformers():
    fig, ax = setup_figure((12, 6))

    stages = [
        (2, 'RNN\n(2013)', COLOR_BLUE, 'Sequential\nNo parallelization'),
        (5, 'RNN+Attention\n(2015)', COLOR_TEAL, 'Better gradients\nStill sequential'),
        (8, 'Transformer\n(2017)', COLOR_ORANGE, 'Fully parallel\nGlobal context')
    ]

    for x, title, color, desc in stages:
        # Main box
        rect = FancyBboxPatch((x-1, 2), 2, 2.5,
                             boxstyle="round,pad=0.1",
                             facecolor=color, alpha=0.3,
                             edgecolor=color, linewidth=2)
        ax.add_patch(rect)

        ax.text(x, 4, title, ha='center', fontsize=11, fontweight='bold')
        ax.text(x, 2.5, desc, ha='center', fontsize=9, style='italic')

        # Evolution arrow
        if x < 8:
            ax.annotate('', xy=(x+2, 3.25), xytext=(x+1, 3.25),
                       arrowprops=dict(arrowstyle='->', color=COLOR_RED, lw=3))

    # Key improvements
    ax.text(3.5, 5, 'Add attention', ha='center', fontsize=9, color=COLOR_RED)
    ax.text(6.5, 5, 'Remove recurrence', ha='center', fontsize=9, color=COLOR_RED)

    ax.set_xlim(0, 10)
    ax.set_ylim(1, 6)
    ax.set_title('Evolution of Sequence Models', fontsize=14, fontweight='bold', pad=20)

    save_figure(fig, 'evolution_to_transformers.pdf')

# Generate all figures
if __name__ == "__main__":
    print("Generating Week 3 RNN figures...")

    create_course_timeline()
    create_rnn_unrolled()
    create_rnn_forward_pass()
    create_vanishing_gradient()
    create_gradient_flow_comparison()
    create_lstm_architecture()
    create_gru_architecture()
    create_bidirectional_rnn()
    create_rnn_vs_transformer_timeline()
    create_evolution_to_transformers()

    print("\nAll figures generated successfully!")
    print("Saved to: NLP_slides/week03_rnn/figures/")