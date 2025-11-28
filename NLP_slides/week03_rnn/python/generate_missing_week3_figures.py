import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Arrow
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

def plot_rnn_unfolding():
    """Visualize RNN unfolding through time"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Colors
    cell_color = '#4A90E2'
    input_color = '#7ED321'
    hidden_color = '#F5A623'
    output_color = '#D0021B'
    
    # Time steps
    time_steps = 5
    cell_width = 1.2
    cell_height = 0.8
    vertical_spacing = 2.5
    horizontal_spacing = 2.0
    
    # Draw RNN cells for each time step
    for t in range(time_steps):
        x_pos = t * horizontal_spacing
        
        # RNN cell
        cell = FancyBboxPatch(
            (x_pos - cell_width/2, -cell_height/2),
            cell_width, cell_height,
            boxstyle="round,pad=0.1",
            facecolor=cell_color,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(cell)
        
        # Cell label
        ax.text(x_pos, 0, f'RNN\nt={t+1}', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
        
        # Input x_t
        input_box = Rectangle(
            (x_pos - 0.3, -vertical_spacing),
            0.6, 0.4,
            facecolor=input_color,
            edgecolor='black'
        )
        ax.add_patch(input_box)
        ax.text(x_pos, -vertical_spacing + 0.2, f'x_{t+1}', ha='center', va='center', 
                fontsize=9, fontweight='bold')
        
        # Arrow from input to cell
        ax.arrow(x_pos, -vertical_spacing + 0.4, 0, vertical_spacing - cell_height/2 - 0.4,
                head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Output y_t
        output_box = Rectangle(
            (x_pos - 0.3, vertical_spacing - 0.2),
            0.6, 0.4,
            facecolor=output_color,
            edgecolor='black'
        )
        ax.add_patch(output_box)
        ax.text(x_pos, vertical_spacing, f'y_{t+1}', ha='center', va='center', 
                fontsize=9, fontweight='bold', color='white')
        
        # Arrow from cell to output
        ax.arrow(x_pos, cell_height/2, 0, vertical_spacing - cell_height/2 - 0.4,
                head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Hidden state connection to next time step
        if t < time_steps - 1:
            # Hidden state h_t
            hidden_circle = Circle(
                (x_pos + horizontal_spacing/2, cell_height/2 + 0.8),
                0.2,
                facecolor=hidden_color,
                edgecolor='black'
            )
            ax.add_patch(hidden_circle)
            ax.text(x_pos + horizontal_spacing/2, cell_height/2 + 0.8, 
                   f'h_{t+1}', ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Arrow from current cell to hidden state
            ax.arrow(x_pos + cell_width/2, cell_height/4, 
                    horizontal_spacing/2 - cell_width/2 - 0.2, 0.3,
                    head_width=0.08, head_length=0.08, fc='black', ec='black')
            
            # Arrow from hidden state to next cell
            ax.arrow(x_pos + horizontal_spacing/2 + 0.2, cell_height/2 + 0.8 - 0.3, 
                    horizontal_spacing/2 - 0.2, -0.3,
                    head_width=0.08, head_length=0.08, fc='black', ec='black')
    
    # Add weight sharing annotation
    ax.text(horizontal_spacing * (time_steps-1) / 2, -vertical_spacing - 0.8,
           'Same weights W shared across all time steps!', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # Labels and title
    ax.set_title('RNN Unfolding Through Time', fontsize=16, fontweight='bold', pad=20)
    ax.text(-1, -vertical_spacing, 'Inputs:', fontsize=12, fontweight='bold', ha='right')
    ax.text(-1, vertical_spacing, 'Outputs:', fontsize=12, fontweight='bold', ha='right')
    
    # Set limits and remove axes
    ax.set_xlim(-1.5, (time_steps-1) * horizontal_spacing + 1.5)
    ax.set_ylim(-vertical_spacing - 1.2, vertical_spacing + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/rnn_unfolding.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_vanishing_gradient():
    """Visualize vanishing gradient problem"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Gradient magnitude over time steps
    time_steps = np.arange(1, 21)
    vanilla_rnn_gradient = np.exp(-0.4 * time_steps)  # Exponential decay
    lstm_gradient = np.exp(-0.05 * time_steps)  # Much slower decay
    
    # Plot gradient magnitudes
    ax1.semilogy(time_steps, vanilla_rnn_gradient, 'r-', linewidth=3, 
                label='Vanilla RNN', marker='o', markersize=6)
    ax1.semilogy(time_steps, lstm_gradient, 'b-', linewidth=3, 
                label='LSTM', marker='s', markersize=6)
    
    # Add threshold line
    ax1.axhline(y=0.01, color='orange', linestyle='--', linewidth=2, 
               label='Effective Learning Threshold')
    
    ax1.set_xlabel('Time Steps Back', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Gradient Magnitude (log scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Vanishing Gradient Problem in RNNs', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add annotations
    ax1.annotate('RNN forgets after ~10 steps', 
                xy=(10, vanilla_rnn_gradient[9]), xytext=(13, 0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red')
    
    ax1.annotate('LSTM maintains gradients', 
                xy=(15, lstm_gradient[14]), xytext=(17, 0.3),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=11, fontweight='bold', color='blue')
    
    # Memory retention visualization
    sentences = ['The', 'student', 'who', 'the', 'professor', 'taught', 'was', 'brilliant']
    positions = np.arange(len(sentences))
    
    # RNN memory strength (decaying)
    rnn_memory = np.exp(-0.3 * positions)
    lstm_memory = np.exp(-0.03 * positions)
    
    width = 0.35
    bars1 = ax2.bar(positions - width/2, rnn_memory, width, 
                   label='RNN Memory', color='red', alpha=0.7)
    bars2 = ax2.bar(positions + width/2, lstm_memory, width,
                   label='LSTM Memory', color='blue', alpha=0.7)
    
    ax2.set_xlabel('Word Position in Sentence', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Memory Strength', fontsize=12, fontweight='bold')
    ax2.set_title('Memory Retention: "The student who the professor taught was brilliant"', 
                 fontsize=14, fontweight='bold')
    ax2.set_xticks(positions)
    ax2.set_xticklabels(sentences, rotation=45, ha='right')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Highlight the connection
    ax2.annotate('', xy=(0, 0.9), xytext=(6, 0.9),
                arrowprops=dict(arrowstyle='<->', color='green', lw=3))
    ax2.text(3, 0.95, 'Long-distance dependency', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='green')
    
    plt.tight_layout()
    plt.savefig('../figures/vanishing_gradient.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_rnn_vs_lstm_performance():
    """Compare RNN vs LSTM performance across different metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Memory Length Performance
    memory_lengths = np.array([5, 10, 20, 50, 100, 200])
    rnn_accuracy = np.array([0.85, 0.78, 0.65, 0.45, 0.25, 0.15])
    lstm_accuracy = np.array([0.87, 0.85, 0.82, 0.78, 0.72, 0.65])
    
    ax1.plot(memory_lengths, rnn_accuracy, 'r-o', linewidth=3, markersize=8, 
            label='Vanilla RNN')
    ax1.plot(memory_lengths, lstm_accuracy, 'b-s', linewidth=3, markersize=8, 
            label='LSTM')
    ax1.set_xlabel('Required Memory Length (steps)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Task Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Performance vs Memory Requirements', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. Training Convergence
    epochs = np.arange(1, 51)
    rnn_loss = 2.5 * np.exp(-0.08 * epochs) + 0.8  # Plateau quickly
    lstm_loss = 2.5 * np.exp(-0.12 * epochs) + 0.3  # Better final loss
    
    ax2.plot(epochs, rnn_loss, 'r-', linewidth=3, label='Vanilla RNN')
    ax2.plot(epochs, lstm_loss, 'b-', linewidth=3, label='LSTM')
    ax2.set_xlabel('Training Epochs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Training Convergence Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. Dataset Size Scaling
    dataset_sizes = np.array([1000, 5000, 10000, 50000, 100000, 500000])
    rnn_performance = np.log(dataset_sizes) * 8 + 45  # Slower scaling
    lstm_performance = np.log(dataset_sizes) * 12 + 40  # Better scaling
    
    ax3.semilogx(dataset_sizes, rnn_performance, 'r-o', linewidth=3, markersize=8,
                label='Vanilla RNN')
    ax3.semilogx(dataset_sizes, lstm_performance, 'b-s', linewidth=3, markersize=8,
                label='LSTM')
    ax3.set_xlabel('Dataset Size (samples)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
    ax3.set_title('Scaling with Dataset Size', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 4. Resource Usage Comparison
    categories = ['Memory Usage', 'Training Time', 'Inference Speed', 'Gradient Flow']
    rnn_scores = [1.0, 1.0, 1.2, 0.3]  # Normalized scores
    lstm_scores = [3.2, 2.8, 0.9, 1.0]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, rnn_scores, width, label='Vanilla RNN', 
                   color='red', alpha=0.7)
    bars2 = ax4.bar(x + width/2, lstm_scores, width, label='LSTM', 
                   color='blue', alpha=0.7)
    
    ax4.set_xlabel('Resource Metrics', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Relative Score', fontsize=12, fontweight='bold')
    ax4.set_title('Resource Usage Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, rotation=15, ha='right')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('RNN vs LSTM: Comprehensive Performance Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../figures/rnn_vs_lstm_performance.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Generating missing Week 3 RNN figures...")
    
    print("1. Creating RNN unfolding visualization...")
    plot_rnn_unfolding()
    
    print("2. Creating vanishing gradient visualization...")
    plot_vanishing_gradient()
    
    print("3. Creating RNN vs LSTM performance comparison...")
    plot_rnn_vs_lstm_performance()
    
    print("Week 3 figures generated successfully!")

if __name__ == "__main__":
    main()