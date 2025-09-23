"""
Generate Week 3 overview visualizations for RNN/LSTM/GRU
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrow
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Educational colors
COLOR_CURRENT = '#FF6B6B'  # Red
COLOR_CONTEXT = '#4ECDC4'  # Teal  
COLOR_PREDICT = '#95E77E'  # Green
COLOR_NEUTRAL = '#E0E0E0'  # Gray

def create_lstm_gates_flow():
    """Create LSTM gates and information flow visualization"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # LSTM cell components
    cell_width = 0.15
    cell_height = 0.12
    
    # Time steps
    time_steps = 3
    x_spacing = 0.3
    
    for t in range(time_steps):
        x_base = 0.1 + t * x_spacing
        y_base = 0.5
        
        # Cell state
        cell_state = Rectangle((x_base - 0.05, y_base + 0.15), cell_width + 0.1, cell_height,
                              facecolor=COLOR_CONTEXT, alpha=0.7, edgecolor='white', linewidth=2)
        ax.add_patch(cell_state)
        ax.text(x_base + cell_width/2, y_base + 0.21, f'C_{t-1+t}', fontsize=10,
               ha='center', va='center', fontweight='bold', color='white')
        
        # Hidden state
        hidden_state = Rectangle((x_base - 0.05, y_base - 0.1), cell_width + 0.1, cell_height,
                               facecolor=COLOR_PREDICT, alpha=0.7, edgecolor='white', linewidth=2)
        ax.add_patch(hidden_state)
        ax.text(x_base + cell_width/2, y_base - 0.04, f'h_{t-1+t}', fontsize=10,
               ha='center', va='center', fontweight='bold', color='white')
        
        # Gates
        gates = [
            {'name': 'Forget', 'y': 0.35, 'color': '#FF6B6B'},
            {'name': 'Input', 'y': 0.25, 'color': '#4ECDC4'},
            {'name': 'Output', 'y': 0.15, 'color': '#95E77E'}
        ]
        
        for gate in gates:
            gate_circle = Circle((x_base + cell_width/2, gate['y']), 0.03,
                               facecolor=gate['color'], edgecolor='white', linewidth=2)
            ax.add_patch(gate_circle)
            ax.text(x_base + cell_width/2, gate['y'] - 0.06, gate['name'],
                   fontsize=8, ha='center', fontweight='bold')
        
        # Arrows showing flow
        if t < time_steps - 1:
            # Cell state flow
            ax.arrow(x_base + cell_width + 0.05, y_base + 0.21, x_spacing - cell_width - 0.1, 0,
                    head_width=0.02, head_length=0.02, fc=COLOR_CONTEXT, alpha=0.5)
            # Hidden state flow
            ax.arrow(x_base + cell_width + 0.05, y_base - 0.04, x_spacing - cell_width - 0.1, 0,
                    head_width=0.02, head_length=0.02, fc=COLOR_PREDICT, alpha=0.5)
    
    # Title and labels
    ax.text(0.5, 0.9, 'LSTM Information Flow Through Time', fontsize=14,
           fontweight='bold', ha='center')
    
    ax.text(0.1, 0.75, 't-1', fontsize=11, ha='center', fontweight='bold')
    ax.text(0.4, 0.75, 't', fontsize=11, ha='center', fontweight='bold')
    ax.text(0.7, 0.75, 't+1', fontsize=11, ha='center', fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=COLOR_CONTEXT, label='Cell State', alpha=0.7),
        mpatches.Patch(color=COLOR_PREDICT, label='Hidden State', alpha=0.7),
        mpatches.Patch(color='#FF6B6B', label='Forget Gate', alpha=0.7),
        mpatches.Patch(color='#4ECDC4', label='Input Gate', alpha=0.7),
        mpatches.Patch(color='#95E77E', label='Output Gate', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/week3_lstm_gates_flow.pdf', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()

def create_gradient_flow():
    """Create gradient flow and vanishing gradient visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: Vanilla RNN gradient flow
    ax1.set_title('Vanilla RNN: Vanishing Gradients', fontsize=12, fontweight='bold')
    
    time_steps = 10
    gradient_values_rnn = np.exp(-np.arange(time_steps) * 0.5)  # Exponential decay
    
    ax1.bar(range(time_steps), gradient_values_rnn, color=COLOR_CURRENT, alpha=0.7)
    ax1.set_xlabel('Time Steps Back', fontsize=10)
    ax1.set_ylabel('Gradient Magnitude', fontsize=10)
    ax1.set_xticks(range(0, time_steps, 2))
    ax1.set_xticklabels([f't-{i}' for i in range(0, time_steps, 2)])
    
    # Add vanishing indication
    ax1.axhline(y=0.01, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax1.text(5, 0.05, 'Vanishing threshold', fontsize=9, color='red', ha='center')
    
    # Right: LSTM gradient flow
    ax2.set_title('LSTM: Preserved Gradients', fontsize=12, fontweight='bold')
    
    gradient_values_lstm = np.exp(-np.arange(time_steps) * 0.1)  # Much slower decay
    
    ax2.bar(range(time_steps), gradient_values_lstm, color=COLOR_PREDICT, alpha=0.7)
    ax2.set_xlabel('Time Steps Back', fontsize=10)
    ax2.set_ylabel('Gradient Magnitude', fontsize=10)
    ax2.set_xticks(range(0, time_steps, 2))
    ax2.set_xticklabels([f't-{i}' for i in range(0, time_steps, 2)])
    
    # Add preserved indication
    ax2.axhline(y=0.5, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax2.text(5, 0.55, 'Gradient preserved', fontsize=9, color='green', ha='center')
    
    # Add comparison annotation
    fig.text(0.5, 0.02, 'LSTM gates allow gradients to flow through time without vanishing',
            fontsize=11, ha='center', fontweight='bold', color='darkblue')
    
    plt.tight_layout()
    plt.savefig('../figures/week3_gradient_flow.pdf', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()

def create_sequence_performance():
    """Create performance comparison on sequence tasks"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Task categories
    tasks = ['Sentiment\nAnalysis', 'Named Entity\nRecognition', 'Part-of-Speech\nTagging',
             'Machine\nTranslation', 'Text\nGeneration']
    
    # Model performance (accuracy %)
    simple_rnn = [72, 68, 85, 45, 55]
    lstm = [88, 85, 92, 75, 78]
    gru = [86, 83, 91, 73, 76]
    bidirectional = [91, 89, 94, 78, 80]
    
    x = np.arange(len(tasks))
    width = 0.2
    
    # Create bars
    bars1 = ax.bar(x - width*1.5, simple_rnn, width, label='Simple RNN',
                  color='#808080', alpha=0.7)
    bars2 = ax.bar(x - width/2, lstm, width, label='LSTM',
                  color=COLOR_CURRENT, alpha=0.7)
    bars3 = ax.bar(x + width/2, gru, width, label='GRU',
                  color=COLOR_CONTEXT, alpha=0.7)
    bars4 = ax.bar(x + width*1.5, bidirectional, width, label='Bi-LSTM',
                  color=COLOR_PREDICT, alpha=0.7)
    
    # Add value labels
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{int(height)}%', ha='center', va='bottom',
                   fontsize=8, fontweight='bold')
    
    ax.set_xlabel('NLP Tasks', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('RNN Variants Performance on Sequence Tasks', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    
    # Add insight box
    insight_text = 'LSTM/GRU solve vanishing gradients\nBidirectional models leverage full context'
    ax.text(0.98, 0.5, insight_text, transform=ax.transAxes,
           fontsize=10, ha='right', va='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('../figures/week3_sequence_performance.pdf', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    create_lstm_gates_flow()
    create_gradient_flow()
    create_sequence_performance()
    print("Week 3 overview visualizations generated successfully!")