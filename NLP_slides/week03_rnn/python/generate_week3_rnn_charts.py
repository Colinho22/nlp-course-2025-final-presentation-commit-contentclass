import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrow
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# 1. RNN Unfolding Through Time
def plot_rnn_unfolding():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Rolled RNN
    ax1.set_title('RNN (Rolled View)', fontsize=14, fontweight='bold')
    
    # Draw the RNN cell
    rnn_cell = FancyBboxPatch((0.3, 0.3), 0.4, 0.4, 
                              boxstyle="round,pad=0.05",
                              facecolor='lightblue',
                              edgecolor='black',
                              linewidth=2)
    ax1.add_patch(rnn_cell)
    ax1.text(0.5, 0.5, 'RNN', ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Self-loop
    ax1.annotate('', xy=(0.5, 0.7), xytext=(0.5, 0.7),
                arrowprops=dict(arrowstyle='->', lw=2, 
                              connectionstyle="arc3,rad=.5"))
    ax1.text(0.5, 0.85, 'h', ha='center', fontsize=12)
    
    # Input arrow
    ax1.arrow(0.5, 0.1, 0, 0.15, head_width=0.05, head_length=0.03, fc='black', ec='black')
    ax1.text(0.5, 0.05, 'x', ha='center', fontsize=12)
    
    # Output arrow
    ax1.arrow(0.5, 0.73, 0, 0.15, head_width=0.05, head_length=0.03, fc='black', ec='black')
    ax1.text(0.5, 0.95, 'y', ha='center', fontsize=12)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Right: Unrolled RNN
    ax2.set_title('RNN Unfolded Through Time', fontsize=14, fontweight='bold')
    
    # Draw sequence of RNN cells
    words = ['The', 'cat', 'sat', 'on']
    positions = [0.15, 0.35, 0.55, 0.75]
    
    for i, (word, pos) in enumerate(zip(words, positions)):
        # RNN cell
        cell = FancyBboxPatch((pos-0.08, 0.4), 0.16, 0.2,
                             boxstyle="round,pad=0.02",
                             facecolor='lightblue',
                             edgecolor='black',
                             linewidth=2)
        ax2.add_patch(cell)
        ax2.text(pos, 0.5, 'RNN', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Input
        ax2.arrow(pos, 0.2, 0, 0.15, head_width=0.02, head_length=0.02, fc='black', ec='black')
        ax2.text(pos, 0.15, word, ha='center', fontsize=10)
        
        # Output
        ax2.arrow(pos, 0.62, 0, 0.13, head_width=0.02, head_length=0.02, fc='black', ec='black')
        ax2.text(pos, 0.8, f'y_{i}', ha='center', fontsize=10)
        
        # Hidden state connection
        if i < len(words) - 1:
            ax2.arrow(pos + 0.08, 0.5, positions[i+1] - pos - 0.16, 0,
                     head_width=0.03, head_length=0.02, fc='red', ec='red')
            ax2.text((pos + positions[i+1])/2, 0.52, f'h_{i}', ha='center', fontsize=9, color='red')
    
    # Initial hidden state
    ax2.arrow(0.02, 0.5, 0.05, 0, head_width=0.03, head_length=0.02, fc='red', ec='red')
    ax2.text(0.04, 0.52, 'h_0', ha='center', fontsize=9, color='red')
    
    ax2.set_xlim(0, 0.9)
    ax2.set_ylim(0, 0.9)
    ax2.axis('off')
    
    # Add note
    ax2.text(0.45, 0.05, 'Same weights used at each time step!', 
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('../figures/rnn_unfolding.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Vanishing Gradient Visualization
def plot_vanishing_gradient():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sentence with long-range dependency
    words = ['The', 'student', 'who', 'the', 'professor', 'who', 'won', 
             'the', 'Nobel', 'Prize', 'taught', 'was', 'brilliant']
    
    # Gradient strength (exponential decay)
    gradient_strength = [0.9 ** (12-i) for i in range(13)]
    
    # Plot bars
    bars = ax.bar(range(len(words)), gradient_strength, color='red', alpha=0.7)
    
    # Color code the important dependency
    bars[1].set_color('blue')  # student
    bars[11].set_color('blue')  # was
    
    # Add word labels
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right')
    
    # Add arrow showing dependency
    ax.annotate('', xy=(11, 0.5), xytext=(1, 0.5),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.text(6, 0.52, 'Must remember "student" to predict "was"', 
            ha='center', fontsize=10, color='blue',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    # Labels
    ax.set_ylabel('Gradient Magnitude', fontsize=12)
    ax.set_xlabel('Words in Sequence', fontsize=12)
    ax.set_title('Vanishing Gradient Problem in RNNs\nGradient decays exponentially through time', 
                 fontsize=14, fontweight='bold')
    
    # Add exponential decay formula
    ax.text(0.02, 0.95, r'Gradient $\approx 0.9^t$ (after $t$ steps)', 
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))
    
    plt.tight_layout()
    plt.savefig('../figures/vanishing_gradient.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 3. RNN vs LSTM Performance
def plot_rnn_vs_lstm_performance():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Gradient flow comparison
    steps = np.arange(0, 100)
    rnn_gradient = 0.9 ** steps  # Exponential decay
    lstm_gradient = 0.98 ** steps  # Much slower decay
    
    ax1.plot(steps, rnn_gradient, label='Vanilla RNN', linewidth=2, color='red')
    ax1.plot(steps, lstm_gradient, label='LSTM', linewidth=2, color='blue')
    ax1.axhline(y=0.01, color='gray', linestyle='--', alpha=0.7)
    ax1.text(50, 0.02, 'Gradient too small to learn', ha='center', fontsize=9, color='gray')
    
    # Mark effective learning range
    ax1.axvspan(0, 10, alpha=0.2, color='red', label='RNN effective range')
    ax1.axvspan(0, 100, alpha=0.1, color='blue', label='LSTM effective range')
    
    ax1.set_xlabel('Time Steps', fontsize=12)
    ax1.set_ylabel('Gradient Magnitude', fontsize=12)
    ax1.set_title('Gradient Flow Comparison', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Task performance
    sequence_lengths = [10, 20, 50, 100, 200]
    rnn_accuracy = [0.95, 0.85, 0.60, 0.40, 0.30]
    lstm_accuracy = [0.98, 0.95, 0.90, 0.85, 0.75]
    
    ax2.plot(sequence_lengths, rnn_accuracy, marker='o', label='Vanilla RNN', 
             linewidth=2, markersize=8, color='red')
    ax2.plot(sequence_lengths, lstm_accuracy, marker='s', label='LSTM', 
             linewidth=2, markersize=8, color='blue')
    
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Task Accuracy', fontsize=12)
    ax2.set_title('Performance on Long Sequences', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    # Add annotations
    ax2.annotate('RNN fails\nbeyond 20 steps', xy=(50, 0.60), xytext=(70, 0.70),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    ax2.annotate('LSTM maintains\nperformance', xy=(100, 0.85), xytext=(120, 0.90),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue')
    
    plt.suptitle('Why LSTMs Revolutionized Sequential Processing', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/rnn_vs_lstm_performance.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 4. LSTM Gates Visualization
def plot_lstm_gates():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # LSTM cell components
    cell_x, cell_y = 0.5, 0.5
    cell_width, cell_height = 0.6, 0.4
    
    # Main cell state
    cell_state = Rectangle((cell_x - cell_width/2, cell_y - cell_height/2), 
                          cell_width, cell_height,
                          facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(cell_state)
    ax.text(cell_x, cell_y, 'Cell State\n(Long-term Memory)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Gates
    gate_width, gate_height = 0.12, 0.08
    gates = [
        ('Forget\nGate', cell_x - 0.2, cell_y - 0.3, 'red'),
        ('Input\nGate', cell_x, cell_y - 0.3, 'green'),
        ('Output\nGate', cell_x + 0.2, cell_y - 0.3, 'blue')
    ]
    
    for name, x, y, color in gates:
        gate = FancyBboxPatch((x - gate_width/2, y - gate_height/2), 
                             gate_width, gate_height,
                             boxstyle="round,pad=0.02",
                             facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(gate)
        ax.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Arrows to cell state
        ax.arrow(x, y + gate_height/2, 0, 0.15, 
                head_width=0.02, head_length=0.02, fc=color, ec=color)
    
    # Input and output
    ax.arrow(cell_x - 0.4, cell_y - 0.5, 0, 0.08, 
            head_width=0.03, head_length=0.02, fc='black', ec='black')
    ax.text(cell_x - 0.4, cell_y - 0.55, 'Input (x_t, h_{t-1})', ha='center', fontsize=10)
    
    ax.arrow(cell_x, cell_y + cell_height/2, 0, 0.1, 
            head_width=0.03, head_length=0.02, fc='black', ec='black')
    ax.text(cell_x, cell_y + 0.35, 'Output (h_t)', ha='center', fontsize=10)
    
    # Gate functions
    gate_functions = [
        ('What to forget\nfrom past', cell_x - 0.2, cell_y - 0.15),
        ('What to store\nfrom input', cell_x, cell_y - 0.15),
        ('What to output\nnow', cell_x + 0.2, cell_y - 0.15)
    ]
    
    for text, x, y in gate_functions:
        ax.text(x, y, text, ha='center', va='center', fontsize=9, 
                style='italic', color='gray')
    
    # Title
    ax.text(cell_x, cell_y + 0.4, 'LSTM: Gated Memory Architecture', 
            ha='center', fontsize=16, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/lstm_gates.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 5. RNN Applications Comparison
def plot_rnn_applications():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Data for comparison
    applications = ['Speech\nRecognition', 'Machine\nTranslation', 'Time Series\nForecasting', 
                   'Text\nGeneration', 'Sentiment\nAnalysis', 'Music\nGeneration']
    rnn_performance = [0.7, 0.6, 0.9, 0.7, 0.6, 0.85]
    transformer_performance = [0.85, 0.95, 0.7, 0.95, 0.9, 0.6]
    
    x = np.arange(len(applications))
    width = 0.35
    
    # Bars
    bars1 = ax.bar(x - width/2, rnn_performance, width, label='RNN/LSTM', color='#4ECDC4')
    bars2 = ax.bar(x + width/2, transformer_performance, width, label='Transformer', color='#FF6B6B')
    
    # Highlight where RNNs win
    for i, (rnn, trans) in enumerate(zip(rnn_performance, transformer_performance)):
        if rnn > trans:
            bars1[i].set_color('#2ECC71')  # Green for RNN wins
            bars1[i].set_edgecolor('black')
            bars1[i].set_linewidth(2)
    
    # Labels
    ax.set_xlabel('Application', fontsize=12)
    ax.set_ylabel('Relative Performance', fontsize=12)
    ax.set_title('RNNs vs Transformers: Where RNNs Still Excel (2024)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(applications)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add annotations for RNN advantages
    ax.text(2, 0.92, 'Sequential nature\ncrucial', ha='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    ax.text(5, 0.87, 'Step-by-step\ngeneration', ha='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # Add note
    ax.text(0.5, 0.02, 'Note: RNNs excel when true sequential processing matters or resources are constrained', 
            transform=ax.transAxes, ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    plt.tight_layout()
    plt.savefig('../figures/rnn_applications.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all Week 3 visualizations
if __name__ == "__main__":
    print("Generating Week 3 RNN visualizations...")
    plot_rnn_unfolding()
    print("- RNN unfolding visualization created")
    plot_vanishing_gradient()
    print("- Vanishing gradient visualization created")
    plot_rnn_vs_lstm_performance()
    print("- RNN vs LSTM performance comparison created")
    plot_lstm_gates()
    print("- LSTM gates visualization created")
    plot_rnn_applications()
    print("- RNN applications comparison created")
    print("\nWeek 3 visualizations completed!")