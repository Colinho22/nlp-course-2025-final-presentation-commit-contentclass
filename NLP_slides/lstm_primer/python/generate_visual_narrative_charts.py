import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

COLOR_FORGET = '#FF6B6B'
COLOR_INPUT = '#4ECDC4'
COLOR_OUTPUT = '#FFD93D'
COLOR_CELL = '#FFA07A'
COLOR_MAIN = '#404040'
COLOR_GRADIENT_GOOD = '#2E7D32'
COLOR_GRADIENT_BAD = '#C62828'
COLOR_LIGHT = '#F0F0F0'

def create_memory_mechanisms_diagram():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Three Gates Control Memory Flow', fontsize=24, weight='bold', ha='center')

    gate_data = [
        (1.5, 5, COLOR_FORGET, 'FORGET GATE', 'What to remove\nfrom memory', '0.0 = Erase all\n1.0 = Keep all'),
        (5, 5, COLOR_INPUT, 'INPUT GATE', 'What new info\nto store', '0.0 = Ignore new\n1.0 = Store all'),
        (8.5, 5, COLOR_OUTPUT, 'OUTPUT GATE', 'What to reveal\nfrom memory', '0.0 = Hide all\n1.0 = Show all')
    ]

    for x, y, color, title, desc, values in gate_data:
        rect = FancyBboxPatch((x-0.8, y-0.8), 1.6, 1.6,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y+0.9, title, fontsize=14, weight='bold', ha='center')
        ax.text(x, y-1.5, desc, fontsize=11, ha='center', va='top')
        ax.text(x, y-2.5, values, fontsize=9, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, alpha=0.8))

    ax.text(5, 1.5, 'All gates use Sigmoid function: output between 0 and 1',
            fontsize=12, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig('../figures/memory_mechanisms_diagram.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_rnn_gradient_vanishing():
    fig, ax = plt.subplots(figsize=(12, 8))

    steps = np.arange(0, 51)
    rnn_gradient = 0.5 ** steps
    lstm_gradient = np.ones_like(steps) * 0.8

    ax.semilogy(steps, rnn_gradient, linewidth=3, color=COLOR_GRADIENT_BAD,
                label='RNN: Gradient decays exponentially', marker='o', markersize=4)
    ax.semilogy(steps, lstm_gradient, linewidth=3, color=COLOR_GRADIENT_GOOD,
                label='LSTM: Gradient stays strong', linestyle='--', marker='s', markersize=4)

    ax.axhline(y=1e-10, color='red', linestyle=':', linewidth=2, alpha=0.5)
    ax.text(25, 1e-11, 'Vanishing threshold (unusable)', fontsize=12, ha='center', color='red')

    ax.annotate('Step 50: $0.5^{50} = 9 \\times 10^{-16}$\n(VANISHED)',
                xy=(50, rnn_gradient[-1]), xytext=(40, 1e-8),
                fontsize=12, color=COLOR_GRADIENT_BAD, weight='bold',
                arrowprops=dict(arrowstyle='->', color=COLOR_GRADIENT_BAD, lw=2))

    ax.annotate('LSTM maintains gradient\nacross 50 steps',
                xy=(50, lstm_gradient[-1]), xytext=(35, 0.5),
                fontsize=12, color=COLOR_GRADIENT_GOOD, weight='bold',
                arrowprops=dict(arrowstyle='->', color=COLOR_GRADIENT_GOOD, lw=2))

    ax.set_xlabel('Time Steps Back', fontsize=14, weight='bold')
    ax.set_ylabel('Gradient Magnitude (log scale)', fontsize=14, weight='bold')
    ax.set_title('Why RNNs Fail: The Vanishing Gradient Problem', fontsize=18, weight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('../figures/rnn_gradient_vanishing.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_forget_gate_flow():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Forget Gate: Step-by-Step', fontsize=22, weight='bold', ha='center')

    ax.add_patch(Rectangle((0.5, 5.5), 1.5, 1, facecolor=COLOR_LIGHT, edgecolor='black', linewidth=2))
    ax.text(1.25, 6, '$h_{t-1}$\n[0.8, 0.6]', ha='center', va='center', fontsize=11, weight='bold')

    ax.add_patch(Rectangle((0.5, 4), 1.5, 1, facecolor=COLOR_LIGHT, edgecolor='black', linewidth=2))
    ax.text(1.25, 4.5, '$x_t$\n[1.0, 0.2]', ha='center', va='center', fontsize=11, weight='bold')

    arrow1 = FancyArrowPatch((2, 6), (3.5, 5.5), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    arrow2 = FancyArrowPatch((2, 4.5), (3.5, 5), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)

    ax.add_patch(FancyBboxPatch((3.5, 4.8), 1.5, 1, boxstyle="round,pad=0.1",
                                facecolor=COLOR_FORGET, edgecolor='black', linewidth=2))
    ax.text(4.25, 5.3, '$\\sigma$', ha='center', va='center', fontsize=16, weight='bold')

    ax.text(4.25, 4.3, '$W_f \\cdot [h_{t-1}, x_t] + b_f$', ha='center', fontsize=9)
    ax.text(4.25, 3.9, '= [1.2, -0.5]', ha='center', fontsize=9, weight='bold')

    arrow3 = FancyArrowPatch((5, 5.3), (6.5, 5.3), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow3)

    ax.add_patch(Rectangle((6.5, 4.8), 1.5, 1, facecolor=COLOR_FORGET, edgecolor='black', linewidth=3))
    ax.text(7.25, 5.3, '$f_t$\n[0.77, 0.38]', ha='center', va='center', fontsize=11, weight='bold')

    ax.add_patch(Rectangle((0.5, 2), 2, 0.8, facecolor=COLOR_CELL, edgecolor='black', linewidth=2))
    ax.text(1.5, 2.4, '$C_{t-1}$\n[0.9, 0.7]', ha='center', va='center', fontsize=11, weight='bold')

    arrow4 = FancyArrowPatch((7.25, 4.8), (7.25, 3.5), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow4)
    arrow5 = FancyArrowPatch((2.5, 2.4), (6, 2.4), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow5)

    ax.text(5, 3, '$\\odot$', fontsize=18, weight='bold', ha='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=2))

    ax.add_patch(Rectangle((7.5, 1.5), 2, 0.8, facecolor='lightgreen', edgecolor='black', linewidth=3))
    ax.text(8.5, 1.9, 'Filtered Memory\n[0.69, 0.27]', ha='center', va='center', fontsize=11, weight='bold')

    ax.text(5, 0.5, 'Result: Keep 77% of first value, only 38% of second',
            fontsize=12, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig('../figures/forget_gate_flow.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_input_gate_flow():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Input Gate: Step-by-Step', fontsize=22, weight='bold', ha='center')

    ax.add_patch(Rectangle((0.5, 5.5), 1.5, 1, facecolor=COLOR_LIGHT, edgecolor='black', linewidth=2))
    ax.text(1.25, 6, '$h_{t-1}$\n[0.8, 0.6]', ha='center', va='center', fontsize=11, weight='bold')

    ax.add_patch(Rectangle((0.5, 4), 1.5, 1, facecolor=COLOR_LIGHT, edgecolor='black', linewidth=2))
    ax.text(1.25, 4.5, '$x_t$\n[1.0, 0.2]', ha='center', va='center', fontsize=11, weight='bold')

    ax.add_patch(FancyBboxPatch((3, 5.3), 1.5, 0.8, boxstyle="round,pad=0.1",
                                facecolor=COLOR_INPUT, edgecolor='black', linewidth=2))
    ax.text(3.75, 5.7, '$\\sigma$ (Gate)', ha='center', fontsize=11, weight='bold')
    ax.text(3.75, 5.0, '$i_t$ = [0.82, 0.45]', ha='center', fontsize=9, weight='bold')

    ax.add_patch(FancyBboxPatch((3, 3.8), 1.5, 0.8, boxstyle="round,pad=0.1",
                                facecolor=COLOR_CELL, edgecolor='black', linewidth=2))
    ax.text(3.75, 4.2, '$\\tanh$ (Content)', ha='center', fontsize=11, weight='bold')
    ax.text(3.75, 3.5, '$\\tilde{C}_t$ = [0.95, -0.30]', ha='center', fontsize=9, weight='bold')

    arrow1 = FancyArrowPatch((2, 6), (3, 5.7), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    arrow2 = FancyArrowPatch((2, 4.5), (3, 4.2), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)

    arrow3 = FancyArrowPatch((4.5, 5.7), (6, 5), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow3)
    arrow4 = FancyArrowPatch((4.5, 4.2), (6, 4.7), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow4)

    ax.text(5.5, 4.85, '$\\odot$', fontsize=18, weight='bold', ha='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=2))

    ax.add_patch(Rectangle((6.5, 4.3), 2, 0.8, facecolor='lightgreen', edgecolor='black', linewidth=3))
    ax.text(7.5, 4.7, 'New Candidate\n[0.78, -0.14]', ha='center', va='center', fontsize=11, weight='bold')

    ax.text(5, 2.5, 'Gate decides: Store 82% of first value, only 45% of second',
            fontsize=12, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    ax.text(5, 1.5, 'Tanh creates candidate: New memory content (-1 to 1 range)',
            fontsize=11, ha='center', style='italic')

    ax.text(5, 0.5, 'Result: Controlled new memory = [0.78, -0.14]',
            fontsize=12, ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor=COLOR_INPUT, alpha=0.3))

    plt.tight_layout()
    plt.savefig('../figures/input_gate_flow.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_output_gate_flow():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Output Gate: Step-by-Step', fontsize=22, weight='bold', ha='center')

    ax.add_patch(Rectangle((0.5, 5.5), 1.5, 1, facecolor=COLOR_LIGHT, edgecolor='black', linewidth=2))
    ax.text(1.25, 6, '$h_{t-1}$\n[0.8, 0.6]', ha='center', va='center', fontsize=11, weight='bold')

    ax.add_patch(Rectangle((0.5, 4), 1.5, 1, facecolor=COLOR_LIGHT, edgecolor='black', linewidth=2))
    ax.text(1.25, 4.5, '$x_t$\n[1.0, 0.2]', ha='center', va='center', fontsize=11, weight='bold')

    ax.add_patch(FancyBboxPatch((3.5, 4.8), 1.5, 1, boxstyle="round,pad=0.1",
                                facecolor=COLOR_OUTPUT, edgecolor='black', linewidth=2))
    ax.text(4.25, 5.3, '$\\sigma$', ha='center', va='center', fontsize=16, weight='bold')
    ax.text(4.25, 4.3, '$o_t$ = [0.91, 0.62]', ha='center', fontsize=9, weight='bold')

    arrow1 = FancyArrowPatch((2, 6), (3.5, 5.5), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    arrow2 = FancyArrowPatch((2, 4.5), (3.5, 5), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)

    ax.add_patch(Rectangle((0.5, 2.5), 2, 0.8, facecolor=COLOR_CELL, edgecolor='black', linewidth=3))
    ax.text(1.5, 2.9, '$C_t$ (updated)\n[0.85, 0.40]', ha='center', va='center', fontsize=11, weight='bold')

    ax.add_patch(FancyBboxPatch((4, 2.3), 1, 0.6, boxstyle="round,pad=0.05",
                                facecolor=COLOR_CELL, edgecolor='black', linewidth=2))
    ax.text(4.5, 2.6, '$\\tanh$', ha='center', fontsize=12, weight='bold')

    arrow3 = FancyArrowPatch((2.5, 2.9), (4, 2.6), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow3)

    ax.add_patch(Rectangle((5.5, 2.5), 1.5, 0.8, facecolor=COLOR_LIGHT, edgecolor='black', linewidth=2))
    ax.text(6.25, 2.9, 'tanh($C_t$)\n[0.69, 0.38]', ha='center', va='center', fontsize=10, weight='bold')

    arrow4 = FancyArrowPatch((5, 2.6), (5.5, 2.9), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow4)

    arrow5 = FancyArrowPatch((4.25, 4.8), (7.5, 3.8), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow5)
    arrow6 = FancyArrowPatch((7, 2.9), (7.5, 3.3), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow6)

    ax.text(7.5, 3.5, '$\\odot$', fontsize=18, weight='bold', ha='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=2))

    ax.add_patch(Rectangle((7.5, 1), 2, 0.8, facecolor='lightgreen', edgecolor='black', linewidth=3))
    ax.text(8.5, 1.4, '$h_t$ (output)\n[0.63, 0.24]', ha='center', va='center', fontsize=11, weight='bold')

    arrow7 = FancyArrowPatch((8, 3.3), (8.5, 1.8), arrowstyle='->', mutation_scale=20, linewidth=3, color='black')
    ax.add_patch(arrow7)

    ax.text(5, 0.3, 'Gate controls: Reveal 91% of first value, 62% of second to output',
            fontsize=12, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig('../figures/output_gate_flow.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_cell_state_sequence():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

    ax1.text(5, 9, 'BEFORE', fontsize=18, weight='bold', ha='center')
    ax1.add_patch(Rectangle((2, 5), 6, 2, facecolor=COLOR_CELL, alpha=0.6, edgecolor='black', linewidth=3))
    ax1.text(5, 6, 'Old Cell State $C_{t-1}$', ha='center', va='center', fontsize=14, weight='bold')
    ax1.text(5, 4.5, '[0.9, 0.7, 0.5]', ha='center', fontsize=12, weight='bold')
    ax1.text(5, 2, 'Contains old memory\nfrom previous words', ha='center', fontsize=11, style='italic')

    ax2.text(5, 9, 'DURING UPDATE', fontsize=18, weight='bold', ha='center')
    ax2.add_patch(Rectangle((1, 6.5), 3, 1.5, facecolor=COLOR_FORGET, alpha=0.6, edgecolor='black', linewidth=2))
    ax2.text(2.5, 7.25, 'Forget Gate\n[0.77, 0.38, 0.20]', ha='center', va='center', fontsize=10, weight='bold')
    ax2.add_patch(Rectangle((6, 6.5), 3, 1.5, facecolor=COLOR_INPUT, alpha=0.6, edgecolor='black', linewidth=2))
    ax2.text(7.5, 7.25, 'Input Gate\n[0.78, -0.14, 0.60]', ha='center', va='center', fontsize=10, weight='bold')

    ax2.text(5, 5, '$\\times$', fontsize=24, weight='bold', ha='center')
    ax2.text(2.5, 4.5, 'Filter old', ha='center', fontsize=10, style='italic')
    ax2.text(5, 4.5, '+', fontsize=20, weight='bold', ha='center')
    ax2.text(7.5, 4.5, 'Add new', ha='center', fontsize=10, style='italic')

    ax2.text(5, 2, '$C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    ax3.text(5, 9, 'AFTER', fontsize=18, weight='bold', ha='center')
    ax3.add_patch(Rectangle((2, 5), 6, 2, facecolor='lightgreen', alpha=0.8, edgecolor='black', linewidth=3))
    ax3.text(5, 6, 'New Cell State $C_t$', ha='center', va='center', fontsize=14, weight='bold')
    ax3.text(5, 4.5, '[0.69, 0.27, 0.10]\n+\n[0.78, -0.14, 0.60]', ha='center', fontsize=11, weight='bold')
    ax3.text(5, 3, '=', fontsize=16, weight='bold', ha='center')
    ax3.text(5, 2.3, '[1.47, 0.13, 0.70]', ha='center', fontsize=13, weight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax3.text(5, 1, 'Updated memory\nready for next word', ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig('../figures/cell_state_sequence.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_forward_pass_flowchart():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    ax.text(5, 11.5, 'Complete LSTM Forward Pass: All 6 Equations', fontsize=20, weight='bold', ha='center')

    y = 10
    ax.add_patch(Rectangle((1, y-0.3), 3, 0.6, facecolor=COLOR_LIGHT, edgecolor='black', linewidth=2))
    ax.text(2.5, y, 'Inputs: $h_{t-1}$, $x_t$', ha='center', va='center', fontsize=11, weight='bold')
    ax.add_patch(Rectangle((6, y-0.3), 3, 0.6, facecolor=COLOR_CELL, edgecolor='black', linewidth=2))
    ax.text(7.5, y, 'Cell State: $C_{t-1}$', ha='center', va='center', fontsize=11, weight='bold')

    y = 8.5
    ax.add_patch(FancyBboxPatch((0.5, y-0.4), 4, 0.8, boxstyle="round,pad=0.1",
                                facecolor=COLOR_FORGET, alpha=0.7, edgecolor='black', linewidth=2))
    ax.text(2.5, y, '1. $f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)$', ha='center', fontsize=10, weight='bold')

    ax.add_patch(FancyBboxPatch((5.5, y-0.4), 4, 0.8, boxstyle="round,pad=0.1",
                                facecolor=COLOR_INPUT, alpha=0.7, edgecolor='black', linewidth=2))
    ax.text(7.5, y, '2. $i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i)$', ha='center', fontsize=10, weight='bold')

    arrow1 = FancyArrowPatch((2.5, y-0.5), (2.5, y-1.2), arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
    ax.add_patch(arrow1)
    arrow2 = FancyArrowPatch((7.5, y-0.5), (7.5, y-1.2), arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
    ax.add_patch(arrow2)

    y = 6.5
    ax.add_patch(FancyBboxPatch((5.5, y-0.4), 4, 0.8, boxstyle="round,pad=0.1",
                                facecolor=COLOR_CELL, alpha=0.7, edgecolor='black', linewidth=2))
    ax.text(7.5, y, '3. $\\tilde{C}_t = \\tanh(W_C \\cdot [h_{t-1}, x_t] + b_C)$', ha='center', fontsize=10, weight='bold')

    arrow3 = FancyArrowPatch((7.5, y-0.5), (5, y-1.5), arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
    ax.add_patch(arrow3)
    arrow4 = FancyArrowPatch((2.5, 7.5), (3.5, y-1.5), arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
    ax.add_patch(arrow4)

    y = 4.5
    ax.add_patch(FancyBboxPatch((1, y-0.5), 8, 1, boxstyle="round,pad=0.1",
                                facecolor='lightblue', alpha=0.7, edgecolor='black', linewidth=3))
    ax.text(5, y, '4. $C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$', ha='center', fontsize=11, weight='bold')
    ax.text(5, y-0.7, 'CELL STATE UPDATE (Core Memory)', ha='center', fontsize=9, style='italic')

    arrow5 = FancyArrowPatch((5, y-1.2), (5, y-2), arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
    ax.add_patch(arrow5)

    y = 2.5
    ax.add_patch(FancyBboxPatch((1, y-0.4), 4, 0.8, boxstyle="round,pad=0.1",
                                facecolor=COLOR_OUTPUT, alpha=0.7, edgecolor='black', linewidth=2))
    ax.text(3, y, '5. $o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o)$', ha='center', fontsize=10, weight='bold')

    ax.add_patch(FancyBboxPatch((5.5, y-0.4), 4, 0.8, boxstyle="round,pad=0.1",
                                facecolor='lightyellow', alpha=0.7, edgecolor='black', linewidth=2))
    ax.text(7.5, y, 'tanh($C_t$)', ha='center', fontsize=11, weight='bold')

    arrow6 = FancyArrowPatch((3, y-0.5), (4.5, y-1.5), arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
    ax.add_patch(arrow6)
    arrow7 = FancyArrowPatch((7.5, y-0.5), (5.5, y-1.5), arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
    ax.add_patch(arrow7)

    y = 0.5
    ax.add_patch(Rectangle((3, y-0.4), 4, 0.8, facecolor='lightgreen', edgecolor='black', linewidth=3))
    ax.text(5, y, '6. $h_t = o_t \\odot \\tanh(C_t)$', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(5, y-0.7, 'FINAL OUTPUT', ha='center', fontsize=10, weight='bold')

    plt.tight_layout()
    plt.savefig('../figures/forward_pass_flowchart.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_sigmoid_curve():
    fig, ax = plt.subplots(figsize=(12, 8))

    z = np.linspace(-6, 6, 200)
    sigmoid = 1 / (1 + np.exp(-z))

    ax.plot(z, sigmoid, linewidth=4, color=COLOR_INPUT, label='$\\sigma(z) = \\frac{1}{1 + e^{-z}}$')
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    examples = [(-5, 0.007), (0, 0.5), (5, 0.993)]
    for z_val, sig_val in examples:
        ax.plot(z_val, sig_val, 'ro', markersize=12)
        ax.annotate(f'$\\sigma({z_val})$ = {sig_val:.3f}',
                    xy=(z_val, sig_val), xytext=(z_val, sig_val+0.15),
                    fontsize=12, weight='bold', ha='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.text(-4, 0.95, 'Gate OPEN\n(value ≈ 1)', fontsize=12, weight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor=COLOR_INPUT, alpha=0.3))
    ax.text(4, 0.05, 'Gate CLOSED\n(value ≈ 0)', fontsize=12, weight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor=COLOR_FORGET, alpha=0.3))
    ax.text(0, 0.3, 'Decision point\n(value = 0.5)', fontsize=11, ha='center', style='italic')

    ax.set_xlabel('Input $z$', fontsize=14, weight='bold')
    ax.set_ylabel('Sigmoid Output $\\sigma(z)$', fontsize=14, weight='bold')
    ax.set_title('Sigmoid Function: Gate Control (0 to 1)', fontsize=18, weight='bold', pad=20)
    ax.legend(fontsize=14, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig('../figures/sigmoid_curve.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_tanh_curve():
    fig, ax = plt.subplots(figsize=(12, 8))

    z = np.linspace(-4, 4, 200)
    tanh = np.tanh(z)

    ax.plot(z, tanh, linewidth=4, color=COLOR_CELL, label='$\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=-1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    examples = [(-3, -0.995), (0, 0), (3, 0.995)]
    for z_val, tanh_val in examples:
        ax.plot(z_val, tanh_val, 'ro', markersize=12)
        ax.annotate(f'$\\tanh({z_val})$ = {tanh_val:.3f}',
                    xy=(z_val, tanh_val), xytext=(z_val+0.3, tanh_val+0.3 if z_val >= 0 else tanh_val-0.3),
                    fontsize=12, weight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.text(-2, 0.7, 'Strong NEGATIVE\nmemory', fontsize=12, weight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor=COLOR_FORGET, alpha=0.3))
    ax.text(2, -0.7, 'Strong POSITIVE\nmemory', fontsize=12, weight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor=COLOR_INPUT, alpha=0.3))
    ax.text(0.5, -0.15, 'Neutral', fontsize=11, ha='left', style='italic')

    ax.set_xlabel('Input $z$', fontsize=14, weight='bold')
    ax.set_ylabel('Tanh Output $\\tanh(z)$', fontsize=14, weight='bold')
    ax.set_title('Tanh Function: Memory Content (-1 to 1)', fontsize=18, weight='bold', pad=20)
    ax.legend(fontsize=14, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.2, 1.2)

    plt.tight_layout()
    plt.savefig('../figures/tanh_curve.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_elementwise_operations():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Element-wise Multiplication: Position by Position', fontsize=20, weight='bold', ha='center')

    ax.text(2, 6, 'Gate values $f_t$:', fontsize=12, weight='bold')
    for i, val in enumerate([0.9, 0.5, 0.1]):
        ax.add_patch(Rectangle((1.5+i*0.8, 5), 0.7, 0.7, facecolor=COLOR_FORGET, edgecolor='black', linewidth=2))
        ax.text(1.85+i*0.8, 5.35, f'{val}', ha='center', va='center', fontsize=14, weight='bold')

    ax.text(5, 5.35, '$\\odot$', fontsize=24, weight='bold')

    ax.text(6.5, 6, 'Memory $C_{t-1}$:', fontsize=12, weight='bold')
    for i, val in enumerate([0.8, 0.6, 0.4]):
        ax.add_patch(Rectangle((6.5+i*0.8, 5), 0.7, 0.7, facecolor=COLOR_CELL, edgecolor='black', linewidth=2))
        ax.text(6.85+i*0.8, 5.35, f'{val}', ha='center', va='center', fontsize=14, weight='bold')

    for i in range(3):
        arrow = FancyArrowPatch((1.85+i*0.8, 4.9), (1.85+i*0.8, 3.8),
                                arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
        ax.add_patch(arrow)
        arrow2 = FancyArrowPatch((6.85+i*0.8, 4.9), (6.85+i*0.8, 3.8),
                                 arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
        ax.add_patch(arrow2)

    ax.text(2, 3.5, 'Position 0:', fontsize=11, weight='bold')
    ax.text(2, 3, '$0.9 \\times 0.8 = 0.72$', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    ax.text(4.2, 3.5, 'Position 1:', fontsize=11, weight='bold')
    ax.text(4.2, 3, '$0.5 \\times 0.6 = 0.30$', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    ax.text(6.5, 3.5, 'Position 2:', fontsize=11, weight='bold')
    ax.text(6.5, 3, '$0.1 \\times 0.4 = 0.04$', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    ax.text(5, 2, '$=$', fontsize=20, weight='bold', ha='center')

    ax.text(5, 1.2, 'Result:', fontsize=12, weight='bold', ha='center')
    for i, val in enumerate([0.72, 0.30, 0.04]):
        ax.add_patch(Rectangle((3.5+i*0.8, 0.5), 0.7, 0.7, facecolor='lightgreen', edgecolor='black', linewidth=3))
        ax.text(3.85+i*0.8, 0.85, f'{val}', ha='center', va='center', fontsize=14, weight='bold')

    plt.tight_layout()
    plt.savefig('../figures/elementwise_operations.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_equation_anatomy():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'Equation Anatomy: Reading LSTM Formulas', fontsize=22, weight='bold', ha='center')

    ax.text(5, 8.5, '$f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)$', fontsize=20, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_LIGHT, edgecolor='black', linewidth=2))

    annotations = [
        (2, 8.5, '$f_t$', 'Forget gate\noutput\n(what to keep)', COLOR_FORGET, (1, 7.2)),
        (3.5, 8.5, '$\\sigma$', 'Sigmoid\nfunction\n(0 to 1)', COLOR_INPUT, (2, 5.8)),
        (4.5, 8.5, '$W_f$', 'Weight\nmatrix\n(learned)', 'lightblue', (3.2, 4.5)),
        (5.8, 8.5, '$h_{t-1}$', 'Previous\nhidden state', 'lightyellow', (5, 3.5)),
        (6.8, 8.5, '$x_t$', 'Current\ninput', 'lightyellow', (6.5, 3.5)),
        (8, 8.5, '$b_f$', 'Bias\nterm\n(learned)', 'lightgray', (8.5, 5))
    ]

    for x_eq, y_eq, text, desc, color, xy_text in annotations:
        ax.annotate(desc, xy=(x_eq, y_eq), xytext=xy_text,
                    fontsize=11, weight='bold', ha='center',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7, edgecolor='black', linewidth=2),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2, connectionstyle='arc3,rad=0.3'))

    ax.text(5, 2, 'Reading left to right:', fontsize=14, weight='bold', ha='center')
    steps = [
        '1. Concatenate previous state and current input: $[h_{t-1}, x_t]$',
        '2. Multiply by weight matrix: $W_f \\cdot [h_{t-1}, x_t]$',
        '3. Add bias: $+ b_f$',
        '4. Apply sigmoid: $\\sigma(...)$ → output between 0 and 1',
        '5. Store result in forget gate: $f_t$'
    ]
    y = 1.2
    for step in steps:
        ax.text(5, y, step, fontsize=11, ha='center')
        y -= 0.25

    plt.tight_layout()
    plt.savefig('../figures/equation_anatomy.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_bptt_visualization():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Backpropagation Through Time (BPTT)', fontsize=22, weight='bold', ha='center')

    positions = [(1.5, 5), (3.5, 5), (5.5, 5), (7.5, 5)]
    labels = ['$t-2$', '$t-1$', '$t$', '$t+1$']

    for (x, y), label in zip(positions, labels):
        ax.add_patch(Circle((x, y), 0.4, facecolor=COLOR_CELL, edgecolor='black', linewidth=2))
        ax.text(x, y, 'LSTM', ha='center', va='center', fontsize=10, weight='bold')
        ax.text(x, y-0.8, label, ha='center', fontsize=12, weight='bold')

    for i in range(len(positions)-1):
        arrow = FancyArrowPatch(positions[i], positions[i+1],
                                arrowstyle='->', mutation_scale=20, linewidth=2, color='blue')
        ax.add_patch(arrow)
        ax.text((positions[i][0]+positions[i+1][0])/2, positions[i][1]+0.5,
                'forward', fontsize=9, ha='center', color='blue', style='italic')

    for i in range(len(positions)-1, 0, -1):
        arrow = FancyArrowPatch(positions[i], positions[i-1],
                                arrowstyle='->', mutation_scale=20, linewidth=3, color='red')
        ax.add_patch(arrow)
        ax.text((positions[i][0]+positions[i-1][0])/2, positions[i][1]-1.2,
                'gradient', fontsize=10, ha='center', color='red', weight='bold')

    ax.text(7.5, 6.5, 'Loss', fontsize=14, weight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, edgecolor='black', linewidth=2))
    arrow_loss = FancyArrowPatch((7.5, 6.3), (7.5, 5.5),
                                 arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
    ax.add_patch(arrow_loss)

    ax.text(5, 2.5, 'Forward Pass (blue): Compute outputs from left to right',
            fontsize=12, ha='center', color='blue', weight='bold')
    ax.text(5, 2, 'Backward Pass (red): Propagate gradients from right to left',
            fontsize=12, ha='center', color='red', weight='bold')
    ax.text(5, 1.3, 'LSTM cell state acts as gradient highway - prevents vanishing!',
            fontsize=11, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor=COLOR_GRADIENT_GOOD, alpha=0.3))

    plt.tight_layout()
    plt.savefig('../figures/bptt_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_lstm_summary_flow():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    ax.text(5, 11.5, 'Complete LSTM Flow: End-to-End', fontsize=22, weight='bold', ha='center')

    ax.add_patch(Rectangle((1, 10), 2, 0.8, facecolor=COLOR_LIGHT, edgecolor='black', linewidth=2))
    ax.text(2, 10.4, 'Inputs\n$h_{t-1}$, $x_t$', ha='center', va='center', fontsize=11, weight='bold')

    ax.add_patch(Rectangle((7, 10), 2, 0.8, facecolor=COLOR_CELL, edgecolor='black', linewidth=2))
    ax.text(8, 10.4, 'Old Memory\n$C_{t-1}$', ha='center', va='center', fontsize=11, weight='bold')

    y = 8.5
    gates = [
        (1.5, COLOR_FORGET, 'FORGET\n$f_t$'),
        (4, COLOR_INPUT, 'INPUT\n$i_t$'),
        (6.5, COLOR_OUTPUT, 'OUTPUT\n$o_t$')
    ]
    for x, color, label in gates:
        ax.add_patch(FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8, boxstyle="round,pad=0.1",
                                    facecolor=color, alpha=0.7, edgecolor='black', linewidth=2))
        ax.text(x, y, label, ha='center', va='center', fontsize=10, weight='bold')

    ax.add_patch(Rectangle((8.5, y-0.4), 1, 0.8, facecolor=COLOR_CELL, edgecolor='black', linewidth=2))
    ax.text(9, y, 'NEW\n$\\tilde{C}_t$', ha='center', va='center', fontsize=10, weight='bold')

    arrow1 = FancyArrowPatch((2, 10), (1.5, y+0.5), arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
    ax.add_patch(arrow1)
    arrow2 = FancyArrowPatch((2, 10), (4, y+0.5), arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
    ax.add_patch(arrow2)
    arrow3 = FancyArrowPatch((2, 10), (6.5, y+0.5), arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
    ax.add_patch(arrow3)
    arrow4 = FancyArrowPatch((2, 10), (9, y+0.5), arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
    ax.add_patch(arrow4)

    y = 6.5
    ax.add_patch(FancyBboxPatch((2, y-0.6), 6, 1.2, boxstyle="round,pad=0.1",
                                facecolor='lightblue', alpha=0.7, edgecolor='black', linewidth=3))
    ax.text(5, y, 'UPDATE CELL STATE', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(5, y-0.4, '$C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$', ha='center', fontsize=10)

    arrow5 = FancyArrowPatch((1.5, 8), (3, y+0.7), arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
    ax.add_patch(arrow5)
    arrow6 = FancyArrowPatch((4, 8), (4.5, y+0.7), arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
    ax.add_patch(arrow6)
    arrow7 = FancyArrowPatch((9, 8), (6.5, y+0.7), arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
    ax.add_patch(arrow7)
    arrow8 = FancyArrowPatch((8, 10), (5, y+0.7), arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
    ax.add_patch(arrow8)

    y = 4.5
    ax.add_patch(Rectangle((3.5, y-0.4), 3, 0.8, facecolor=COLOR_CELL, edgecolor='black', linewidth=3))
    ax.text(5, y, 'New Cell State $C_t$', ha='center', va='center', fontsize=12, weight='bold')

    arrow9 = FancyArrowPatch((5, 5.8), (5, y+0.5), arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
    ax.add_patch(arrow9)

    y = 2.5
    ax.add_patch(FancyBboxPatch((3, y-0.5), 4, 1, boxstyle="round,pad=0.1",
                                facecolor='lightyellow', alpha=0.7, edgecolor='black', linewidth=2))
    ax.text(5, y, 'COMPUTE OUTPUT', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(5, y-0.35, '$h_t = o_t \\odot \\tanh(C_t)$', ha='center', fontsize=10)

    arrow10 = FancyArrowPatch((6.5, 8), (6, y+0.6), arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
    ax.add_patch(arrow10)
    arrow11 = FancyArrowPatch((5, 4), (5, y+0.6), arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
    ax.add_patch(arrow11)

    y = 0.5
    ax.add_patch(Rectangle((3.5, y-0.4), 3, 0.8, facecolor='lightgreen', edgecolor='black', linewidth=3))
    ax.text(5, y, 'Final Output $h_t$', ha='center', va='center', fontsize=12, weight='bold')

    arrow12 = FancyArrowPatch((5, 1.4), (5, y+0.5), arrowstyle='->', mutation_scale=15, linewidth=3, color='black')
    ax.add_patch(arrow12)

    plt.tight_layout()
    plt.savefig('../figures/lstm_summary_flow.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating 13 new visual narrative charts...")

    print("1/13: memory_mechanisms_diagram.pdf")
    create_memory_mechanisms_diagram()

    print("2/13: rnn_gradient_vanishing.pdf")
    create_rnn_gradient_vanishing()

    print("3/13: forget_gate_flow.pdf")
    create_forget_gate_flow()

    print("4/13: input_gate_flow.pdf")
    create_input_gate_flow()

    print("5/13: output_gate_flow.pdf")
    create_output_gate_flow()

    print("6/13: cell_state_sequence.pdf")
    create_cell_state_sequence()

    print("7/13: forward_pass_flowchart.pdf")
    create_forward_pass_flowchart()

    print("8/13: sigmoid_curve.pdf")
    create_sigmoid_curve()

    print("9/13: tanh_curve.pdf")
    create_tanh_curve()

    print("10/13: elementwise_operations.pdf")
    create_elementwise_operations()

    print("11/13: equation_anatomy.pdf")
    create_equation_anatomy()

    print("12/13: bptt_visualization.pdf")
    create_bptt_visualization()

    print("13/13: lstm_summary_flow.pdf")
    create_lstm_summary_flow()

    print("\nAll 13 visual narrative charts generated successfully!")
    print("Charts saved to: ../figures/")