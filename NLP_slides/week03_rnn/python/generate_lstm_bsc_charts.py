"""
Generate BSc-level figures for LSTM (Long Short-Term Memory)
Simple, clear visualizations for introductory students
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Arc
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define BSc-friendly color palette
COLOR_BLUE = '#2E86C1'        # Primary blue
COLOR_ORANGE = '#FF8800'      # Accent orange
COLOR_GREEN = '#27AE60'       # Success green
COLOR_RED = '#E74C3C'         # Warning red
COLOR_GRAY = '#95A5A6'        # Neutral gray
COLOR_LIGHT = '#ECF0F1'       # Light background
COLOR_PURPLE = '#9B59B6'      # Insight purple

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


# 1. Vanishing Gradient Problem (RNN vs LSTM)
def create_vanishing_gradient_problem():
    """Show gradient flow comparison between RNN and LSTM"""
    fig, ax = setup_figure((12, 7))

    # Title
    ax.text(6, 6.5, 'The Vanishing Gradient Problem', ha='center',
           fontsize=14, fontweight='bold')

    # RNN gradient flow (top)
    ax.text(1, 5.5, 'Standard RNN:', ha='left', fontsize=12, fontweight='bold', color=COLOR_RED)

    # RNN cells
    for i in range(5):
        x_pos = 1.5 + i * 2
        circle = Circle((x_pos, 4.5), 0.3, facecolor=COLOR_LIGHT,
                       edgecolor=COLOR_RED, linewidth=2)
        ax.add_patch(circle)
        ax.text(x_pos, 4.5, 'RNN', ha='center', va='center', fontsize=8, fontweight='bold')

        if i < 4:
            # Gradient arrow getting weaker
            alpha = max(0.2, 1.0 - i * 0.2)
            ax.annotate('', xy=(x_pos + 1.4, 4.5), xytext=(x_pos + 0.4, 4.5),
                       arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_RED, alpha=alpha))
            ax.text(x_pos + 0.9, 4.8, f'{int(alpha*100)}%', ha='center',
                   fontsize=7, color=COLOR_RED, alpha=alpha)

    ax.text(10.5, 4.5, 'X', ha='center', fontsize=20, color=COLOR_RED, fontweight='bold')
    ax.text(10.5, 3.8, 'Gradient\nvanishes!', ha='center', fontsize=8, color=COLOR_RED)

    # LSTM gradient flow (bottom)
    ax.text(1, 2.8, 'LSTM:', ha='left', fontsize=12, fontweight='bold', color=COLOR_GREEN)

    # Cell state highway
    ax.plot([1.2, 10.8], [2.2, 2.2], linewidth=8, color=COLOR_ORANGE, alpha=0.5)
    ax.text(6, 2.5, 'Cell State Highway', ha='center', fontsize=9,
           color=COLOR_ORANGE, fontweight='bold', style='italic')

    # LSTM cells
    for i in range(5):
        x_pos = 1.5 + i * 2
        rect = Rectangle((x_pos - 0.3, 1.5), 0.6, 0.6,
                        facecolor=COLOR_LIGHT, edgecolor=COLOR_GREEN, linewidth=2)
        ax.add_patch(rect)
        ax.text(x_pos, 1.8, 'LSTM', ha='center', va='center', fontsize=7, fontweight='bold')

        if i < 4:
            # Gradient arrow staying strong
            ax.annotate('', xy=(x_pos + 1.4, 2.2), xytext=(x_pos + 0.4, 2.2),
                       arrowprops=dict(arrowstyle='->', lw=3, color=COLOR_GREEN))
            ax.text(x_pos + 0.9, 2.5, '100%', ha='center', fontsize=7, color=COLOR_GREEN)

    ax.text(10.5, 1.8, '✓', ha='center', fontsize=20, color=COLOR_GREEN, fontweight='bold')
    ax.text(10.5, 1.1, 'Gradient\nflows!', ha='center', fontsize=8, color=COLOR_GREEN)

    # Key insight
    insight = 'Key: LSTM uses addition (cell state) instead of multiplication (RNN hidden state)'
    ax.text(6, 0.3, insight, ha='center', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_PURPLE, alpha=0.2))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)

    save_figure(fig, 'vanishing_gradient_problem_bsc.pdf')


# 2. LSTM Architecture Overview
def create_lstm_architecture_overview():
    """High-level LSTM architecture with 3 gates"""
    fig, ax = setup_figure((12, 8))

    # Title
    ax.text(6, 7.5, 'LSTM Architecture: Three Smart Gates', ha='center',
           fontsize=14, fontweight='bold')

    # Cell state line
    ax.plot([1, 11], [4, 4], linewidth=10, color=COLOR_ORANGE, alpha=0.5)
    ax.text(0.5, 4, '$C_{t-1}$', ha='right', fontsize=12, color=COLOR_ORANGE, fontweight='bold')
    ax.text(11.5, 4, '$C_t$', ha='left', fontsize=12, color=COLOR_ORANGE, fontweight='bold')
    ax.text(6, 4.6, 'Cell State (Long-term Memory)', ha='center', fontsize=10,
           color=COLOR_ORANGE, fontweight='bold', style='italic')

    # Three gates positioned along cell state
    gates = [
        {'name': 'Forget Gate', 'x': 3, 'color': COLOR_RED, 'symbol': '×'},
        {'name': 'Input Gate', 'x': 6, 'color': COLOR_BLUE, 'symbol': '+'},
        {'name': 'Output Gate', 'x': 9, 'color': COLOR_GREEN, 'symbol': '→'}
    ]

    for gate in gates:
        x = gate['x']

        # Gate box
        rect = FancyBboxPatch((x - 0.8, 2.5), 1.6, 1.2,
                              boxstyle="round,pad=0.08",
                              facecolor=gate['color'], alpha=0.3,
                              edgecolor=gate['color'], linewidth=3)
        ax.add_patch(rect)

        # Gate name
        ax.text(x, 3.4, gate['name'], ha='center', fontsize=10,
               fontweight='bold', color=gate['color'])

        # Symbol
        ax.text(x, 2.9, gate['symbol'], ha='center', fontsize=18, fontweight='bold')

        # Arrow to cell state
        ax.annotate('', xy=(x, 3.8), xytext=(x, 3.7),
                   arrowprops=dict(arrowstyle='->', lw=2, color=gate['color']))

    # Input
    ax.annotate('', xy=(3, 2.4), xytext=(3, 1.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(3, 1.2, '$h_{t-1}$, $x_t$', ha='center', fontsize=10, fontweight='bold')
    ax.text(3, 0.8, 'Previous output\n+ Current input', ha='center', fontsize=8, style='italic')

    # Output
    ax.annotate('', xy=(9, 5.5), xytext=(9, 4.2),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_GREEN))
    ax.text(9, 5.8, '$h_t$', ha='center', fontsize=10, fontweight='bold', color=COLOR_GREEN)
    ax.text(9, 6.2, 'Output', ha='center', fontsize=8, style='italic')

    # Function labels
    ax.text(2, 6.5, 'What to', ha='center', fontsize=9, color=COLOR_RED)
    ax.text(2, 6.1, 'FORGET?', ha='center', fontsize=9, fontweight='bold', color=COLOR_RED)

    ax.text(6, 6.5, 'What to', ha='center', fontsize=9, color=COLOR_BLUE)
    ax.text(6, 6.1, 'REMEMBER?', ha='center', fontsize=9, fontweight='bold', color=COLOR_BLUE)

    ax.text(10, 6.5, 'What to', ha='center', fontsize=9, color=COLOR_GREEN)
    ax.text(10, 6.1, 'OUTPUT?', ha='center', fontsize=9, fontweight='bold', color=COLOR_GREEN)

    ax.set_xlim(0, 12)
    ax.set_ylim(0.5, 8)

    save_figure(fig, 'lstm_architecture_overview_bsc.pdf')


# 3. Forget Gate Detail
def create_forget_gate_detail():
    """Detailed forget gate mechanism"""
    fig, ax = setup_figure((12, 6))

    # Title
    ax.text(6, 5.5, 'Forget Gate: What to Erase?', ha='center',
           fontsize=14, fontweight='bold', color=COLOR_RED)

    # Example context
    ax.text(6, 4.8, 'Example: "The cat was hungry. The dog ..."', ha='center',
           fontsize=11, style='italic')

    # Input
    ax.text(2, 3.8, 'Inputs:', ha='left', fontsize=10, fontweight='bold')
    ax.text(2, 3.4, '$h_{t-1}$: Previous output', ha='left', fontsize=9)
    ax.text(2, 3.0, '$x_t$: Current word ("dog")', ha='left', fontsize=9)

    # Forget gate box
    rect = FancyBboxPatch((5, 2.5), 2, 1.5,
                          boxstyle="round,pad=0.1",
                          facecolor=COLOR_RED, alpha=0.2,
                          edgecolor=COLOR_RED, linewidth=3)
    ax.add_patch(rect)

    ax.text(6, 3.6, 'Forget Gate', ha='center', fontsize=11, fontweight='bold', color=COLOR_RED)
    ax.text(6, 3.2, '$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$', ha='center', fontsize=9)
    ax.text(6, 2.8, 'Output: 0 to 1', ha='center', fontsize=8, style='italic')

    # Decision visualization
    ax.text(9.5, 3.8, 'Decision:', ha='left', fontsize=10, fontweight='bold')

    # Memory cells
    memory_items = [
        {'label': '"cat" info', 'value': 0.1, 'desc': 'Forget! (new subject)'},
        {'label': '"hungry" info', 'value': 0.2, 'desc': 'Forget! (not relevant)'}
    ]

    for i, item in enumerate(memory_items):
        y = 3.3 - i * 0.6

        # Bar showing forget value
        bar_width = item['value'] * 1.5
        rect = Rectangle((9.5, y - 0.15), bar_width, 0.3,
                        facecolor=COLOR_RED, alpha=0.5, edgecolor=COLOR_RED, linewidth=1)
        ax.add_patch(rect)

        ax.text(9.3, y, item['label'], ha='right', fontsize=8)
        ax.text(9.5 + bar_width + 0.1, y, f'{int(item["value"]*100)}%',
               ha='left', fontsize=8, color=COLOR_RED, fontweight='bold')
        ax.text(11.3, y, item['desc'], ha='right', fontsize=7, style='italic')

    # Key insight
    insight = 'Lower values (close to 0) = FORGET\nHigher values (close to 1) = KEEP'
    ax.text(6, 1.5, insight, ha='center', fontsize=9,
           bbox=dict(boxstyle='round,pad=0.4', facecolor=COLOR_RED, alpha=0.1))

    # Intuition
    ax.text(6, 0.5, 'Intuition: When you see "dog", forget information about "cat"',
           ha='center', fontsize=9, style='italic', color=COLOR_PURPLE)

    ax.set_xlim(1, 12)
    ax.set_ylim(0, 6)

    save_figure(fig, 'forget_gate_detail_bsc.pdf')


# 4. Input Gate Detail
def create_input_gate_detail():
    """Detailed input gate mechanism"""
    fig, ax = setup_figure((12, 6))

    # Title
    ax.text(6, 5.5, 'Input Gate: What to Remember?', ha='center',
           fontsize=14, fontweight='bold', color=COLOR_BLUE)

    # Example context
    ax.text(6, 4.8, 'Example: "The dog was sleeping ..."', ha='center',
           fontsize=11, style='italic')

    # Input
    ax.text(2, 3.8, 'Inputs:', ha='left', fontsize=10, fontweight='bold')
    ax.text(2, 3.4, '$h_{t-1}$: Previous output', ha='left', fontsize=9)
    ax.text(2, 3.0, '$x_t$: Current word ("sleeping")', ha='left', fontsize=9)

    # Two-part process
    # Part 1: Input gate
    rect1 = FancyBboxPatch((4.5, 2.2), 1.8, 1.2,
                           boxstyle="round,pad=0.08",
                           facecolor=COLOR_BLUE, alpha=0.2,
                           edgecolor=COLOR_BLUE, linewidth=2)
    ax.add_patch(rect1)

    ax.text(5.4, 3.0, 'Input Gate', ha='center', fontsize=9, fontweight='bold', color=COLOR_BLUE)
    ax.text(5.4, 2.7, '$i_t = \sigma(...)$', ha='center', fontsize=8)
    ax.text(5.4, 2.4, 'How much?', ha='center', fontsize=7, style='italic')

    # Part 2: Candidate values
    rect2 = FancyBboxPatch((6.7, 2.2), 1.8, 1.2,
                           boxstyle="round,pad=0.08",
                           facecolor=COLOR_BLUE, alpha=0.2,
                           edgecolor=COLOR_BLUE, linewidth=2)
    ax.add_patch(rect2)

    ax.text(7.6, 3.0, 'Candidate', ha='center', fontsize=9, fontweight='bold', color=COLOR_BLUE)
    ax.text(7.6, 2.7, '$\\tilde{C}_t = \\tanh(...)$', ha='center', fontsize=8)
    ax.text(7.6, 2.4, 'What info?', ha='center', fontsize=7, style='italic')

    # Multiplication
    ax.text(6.5, 1.8, '×', ha='center', fontsize=16, fontweight='bold', color=COLOR_BLUE)

    # Result
    ax.annotate('', xy=(6.5, 1.5), xytext=(6.5, 1.6),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_BLUE))
    ax.text(6.5, 1.2, 'New info to add:', ha='center', fontsize=9, fontweight='bold')
    ax.text(6.5, 0.8, '$i_t * \\tilde{C}_t$', ha='center', fontsize=9, color=COLOR_BLUE)

    # Decision visualization
    ax.text(9.5, 3.8, 'Decision:', ha='left', fontsize=10, fontweight='bold')

    new_info = [
        {'label': '"dog" subject', 'value': 0.9},
        {'label': '"sleeping" state', 'value': 0.8}
    ]

    for i, item in enumerate(new_info):
        y = 3.3 - i * 0.6

        # Bar showing input value
        bar_width = item['value'] * 1.5
        rect = Rectangle((9.5, y - 0.15), bar_width, 0.3,
                        facecolor=COLOR_BLUE, alpha=0.5, edgecolor=COLOR_BLUE, linewidth=1)
        ax.add_patch(rect)

        ax.text(9.3, y, item['label'], ha='right', fontsize=8)
        ax.text(9.5 + bar_width + 0.1, y, f'{int(item["value"]*100)}%',
               ha='left', fontsize=8, color=COLOR_BLUE, fontweight='bold')

    # Intuition
    ax.text(6, 0.3, 'Intuition: Remember "dog is sleeping" for future predictions',
           ha='center', fontsize=9, style='italic', color=COLOR_PURPLE)

    ax.set_xlim(1, 12)
    ax.set_ylim(0, 6)

    save_figure(fig, 'input_gate_detail_bsc.pdf')


# 5. Output Gate Detail
def create_output_gate_detail():
    """Detailed output gate mechanism"""
    fig, ax = setup_figure((12, 6))

    # Title
    ax.text(6, 5.5, 'Output Gate: What to Output?', ha='center',
           fontsize=14, fontweight='bold', color=COLOR_GREEN)

    # Example context
    ax.text(6, 4.8, 'Example: "The dog was sleeping and ..." → predict next word', ha='center',
           fontsize=11, style='italic')

    # Cell state input
    ax.text(2, 3.8, 'Cell State:', ha='left', fontsize=10, fontweight='bold')
    ax.text(2, 3.4, 'Contains: dog, sleeping, etc.', ha='left', fontsize=9)
    ax.text(2, 3.0, 'Question: What\'s relevant NOW?', ha='left', fontsize=9, style='italic', color=COLOR_PURPLE)

    # Output gate box
    rect = FancyBboxPatch((4.5, 2.2), 2.5, 1.2,
                          boxstyle="round,pad=0.1",
                          facecolor=COLOR_GREEN, alpha=0.2,
                          edgecolor=COLOR_GREEN, linewidth=3)
    ax.add_patch(rect)

    ax.text(5.75, 3.0, 'Output Gate', ha='center', fontsize=11, fontweight='bold', color=COLOR_GREEN)
    ax.text(5.75, 2.7, '$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$', ha='center', fontsize=9)
    ax.text(5.75, 2.4, 'How much to output?', ha='center', fontsize=8, style='italic')

    # Final output computation
    ax.text(5.75, 1.6, 'Final Output:', ha='center', fontsize=10, fontweight='bold')
    ax.text(5.75, 1.2, '$h_t = o_t * \\tanh(C_t)$', ha='center', fontsize=9, color=COLOR_GREEN)

    # Arrow to output
    ax.annotate('', xy=(5.75, 0.8), xytext=(5.75, 1.0),
               arrowprops=dict(arrowstyle='->', lw=3, color=COLOR_GREEN))
    ax.text(5.75, 0.5, 'To next layer / prediction', ha='center', fontsize=8, style='italic')

    # Decision visualization
    ax.text(9, 3.8, 'Decision:', ha='left', fontsize=10, fontweight='bold')

    output_info = [
        {'label': '"dog" info', 'value': 0.9, 'desc': 'Output! (subject)'},
        {'label': '"sleeping" info', 'value': 0.7, 'desc': 'Output! (state)'},
        {'label': 'old context', 'value': 0.1, 'desc': 'Hide (not needed)'}
    ]

    for i, item in enumerate(output_info):
        y = 3.3 - i * 0.6

        # Bar showing output value
        bar_width = item['value'] * 1.2
        color = COLOR_GREEN if item['value'] > 0.5 else COLOR_GRAY
        rect = Rectangle((9, y - 0.15), bar_width, 0.3,
                        facecolor=color, alpha=0.5, edgecolor=color, linewidth=1)
        ax.add_patch(rect)

        ax.text(8.8, y, item['label'], ha='right', fontsize=8)
        ax.text(9 + bar_width + 0.1, y, f'{int(item["value"]*100)}%',
               ha='left', fontsize=8, fontweight='bold')
        ax.text(11.3, y, item['desc'], ha='right', fontsize=7, style='italic')

    # Intuition
    ax.text(6, 0.05, 'Intuition: Only share relevant parts of memory for current prediction',
           ha='center', fontsize=9, style='italic', color=COLOR_PURPLE)

    ax.set_xlim(1, 12)
    ax.set_ylim(0, 6)

    save_figure(fig, 'output_gate_detail_bsc.pdf')


# Generate all figures
if __name__ == '__main__':
    print("Generating BSc-level LSTM figures...")
    print("-" * 50)

    create_vanishing_gradient_problem()
    create_lstm_architecture_overview()
    create_forget_gate_detail()
    create_input_gate_detail()
    create_output_gate_detail()

    print("-" * 50)
    print("All BSc-level LSTM figures generated successfully!")
