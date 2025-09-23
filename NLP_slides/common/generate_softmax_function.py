"""
Generate x vs softmax(x) function visualization for Week 2 Neural Language Models
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set consistent style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def softmax(x, temperature=1.0):
    """Compute softmax with optional temperature parameter"""
    x = x / temperature
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)

def create_softmax_function_plot():
    """Create a clean x vs softmax(x) function visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Single variable softmax behavior
    ax1 = axes[0]
    
    # Create range of x values
    x_single = np.linspace(-5, 5, 100)
    
    # For single variable, softmax is just sigmoid
    sigmoid = 1 / (1 + np.exp(-x_single))
    
    ax1.plot(x_single, sigmoid, linewidth=3, color='#4ECDC4', label='σ(x) = 1/(1+e^(-x))')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Mark key points
    ax1.plot(0, 0.5, 'ro', markersize=10, label='(0, 0.5)')
    ax1.plot(2, 1/(1+np.exp(-2)), 'go', markersize=8)
    ax1.plot(-2, 1/(1+np.exp(2)), 'go', markersize=8)
    
    ax1.set_xlabel('x', fontsize=14)
    ax1.set_ylabel('Sigmoid(x)', fontsize=14)
    ax1.set_title('Single Variable: Sigmoid Function', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=12)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-0.05, 1.05)
    
    # Add annotations
    ax1.annotate('Saturates at 1', xy=(4, 0.98), xytext=(3, 0.85),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
                fontsize=11)
    ax1.annotate('Saturates at 0', xy=(-4, 0.02), xytext=(-3, 0.15),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
                fontsize=11)
    
    # Right plot: Multi-class softmax
    ax2 = axes[1]
    
    # Create a fixed set of scores
    fixed_scores = np.array([0.0, 0.0, 0.0])  # Three classes
    x_range = np.linspace(-3, 3, 50)
    
    # Colors for each class
    colors = ['#FF6B6B', '#4ECDC4', '#95E77E']
    
    # Plot how probability changes as we vary one score
    for class_idx in range(3):
        probs = []
        for x_val in x_range:
            scores = fixed_scores.copy()
            scores[class_idx] = x_val
            prob = softmax(scores)[class_idx]
            probs.append(prob)
        
        ax2.plot(x_range, probs, linewidth=2.5, color=colors[class_idx], 
                label=f'P(class {class_idx+1}) when x_{class_idx+1} varies')
    
    ax2.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='Equal probability')
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Score value (x_i)', fontsize=14)
    ax2.set_ylabel('Softmax probability', fontsize=14)
    ax2.set_title('Multi-class Softmax: Varying One Score', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(0, 1)
    
    # Add formula annotation
    ax2.text(0.5, 0.95, r'$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$',
            transform=ax2.transAxes, fontsize=12, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
    
    plt.suptitle('Softmax Function Behavior', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/softmax_function_plot.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_temperature_effect_plot():
    """Create visualization showing temperature effect on softmax"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Fixed scores for demonstration
    scores = np.array([2.0, 1.0, 0.5, -0.5, -1.0])
    x_labels = ['Score 2.0', 'Score 1.0', 'Score 0.5', 'Score -0.5', 'Score -1.0']
    x_pos = np.arange(len(scores))
    
    # Different temperatures
    temperatures = [0.5, 1.0, 2.0, 5.0]
    colors = ['#FF6B6B', '#4ECDC4', '#95E77E', '#FFE66D']
    markers = ['o', 's', '^', 'D']
    
    for temp, color, marker in zip(temperatures, colors, markers):
        probs = softmax(scores, temperature=temp)
        ax.plot(x_pos, probs, marker=marker, markersize=10, linewidth=2.5,
               color=color, label=f'T = {temp}', alpha=0.8)
    
    # Add vertical lines for each score position
    for i in x_pos:
        ax.axvline(x=i, color='gray', linestyle=':', alpha=0.3)
    
    ax.set_xlabel('Input Scores', fontsize=14)
    ax.set_ylabel('Softmax Probability', fontsize=14)
    ax.set_title('Temperature Effect on Softmax Distribution', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.legend(loc='upper right', fontsize=12, title='Temperature', title_fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    # Add text annotations
    ax.text(0.02, 0.98, 'Low T → Sharp distribution\n(confident predictions)',
           transform=ax.transAxes, fontsize=11, va='top',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE5E5', alpha=0.5))
    
    ax.text(0.98, 0.98, 'High T → Uniform distribution\n(uncertain predictions)',
           transform=ax.transAxes, fontsize=11, va='top', ha='right',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#E5F5FF', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('../figures/softmax_temperature_effect.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating softmax function visualizations...")
    
    print("1. Creating x vs softmax(x) function plot...")
    create_softmax_function_plot()
    
    print("2. Creating temperature effect visualization...")
    create_temperature_effect_plot()
    
    print("All softmax function visualizations generated successfully!")