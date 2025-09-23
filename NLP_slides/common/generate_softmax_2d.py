"""
Generate 2D visualization of softmax function for Week 2 Neural Language Models
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set consistent style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def softmax(x, temperature=1.0):
    """Compute softmax with optional temperature parameter"""
    x = x / temperature
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)

def create_softmax_visualization():
    """Create comprehensive 2D softmax visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Top-left: Basic softmax transformation
    ax1 = axes[0, 0]
    
    # Example vocabulary
    words = ['cat', 'dog', 'car', 'tree', 'house', 'water', 'fire', 'book']
    raw_scores = np.array([2.1, 1.8, -0.5, 0.3, -1.2, 0.9, -0.3, 0.1])
    probabilities = softmax(raw_scores)
    
    x_pos = np.arange(len(words))
    
    # Plot raw scores
    bars1 = ax1.bar(x_pos - 0.2, raw_scores, 0.4, label='Raw Scores', color='#FF6B6B', alpha=0.7)
    # Plot probabilities
    bars2 = ax1.bar(x_pos + 0.2, probabilities, 0.4, label='Softmax Output', color='#4ECDC4', alpha=0.7)
    
    ax1.set_xlabel('Words', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Softmax Transformation: Scores → Probabilities', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(words, rotation=45, ha='right')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, raw_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    for bar, val in zip(bars2, probabilities):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Top-right: Temperature effect
    ax2 = axes[0, 1]
    
    temperatures = [0.5, 1.0, 2.0, 5.0]
    colors_temp = ['#FF6B6B', '#4ECDC4', '#95E77E', '#FFE66D']
    
    for temp, color in zip(temperatures, colors_temp):
        probs = softmax(raw_scores, temperature=temp)
        ax2.plot(x_pos, probs, 'o-', label=f'T={temp}', color=color, linewidth=2, markersize=8)
    
    ax2.set_xlabel('Words', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title('Temperature Effect on Softmax Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(words, rotation=45, ha='right')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 0.8])
    
    # 3. Bottom-left: 2D heatmap of softmax
    ax3 = axes[1, 0]
    
    # Create 2D grid of input values
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    
    # For each point, compute softmax assuming 3 classes
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            scores = np.array([X[i,j], Y[i,j], 0])  # Third class fixed at 0
            probs = softmax(scores)
            Z[i,j] = probs[0]  # Probability of first class
    
    im = ax3.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax3.set_xlabel('Score for Class 1', fontsize=12)
    ax3.set_ylabel('Score for Class 2', fontsize=12)
    ax3.set_title('2D Softmax: P(Class 1) with Class 3 fixed at 0', fontsize=14, fontweight='bold')
    
    # Add colorbar
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='P(Class 1)')
    
    # Add contour lines
    CS = ax3.contour(X, Y, Z, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors='white', alpha=0.5, linewidths=1)
    ax3.clabel(CS, inline=True, fontsize=8)
    
    # 4. Bottom-right: Mathematical formula and example
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Add mathematical formula
    formula_text = r'$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$'
    ax4.text(0.5, 0.85, 'Softmax Formula:', fontsize=14, fontweight='bold', ha='center')
    ax4.text(0.5, 0.75, formula_text, fontsize=16, ha='center')
    
    # Add concrete example
    ax4.text(0.5, 0.6, 'Example Calculation:', fontsize=12, fontweight='bold', ha='center')
    
    example_text = """Input scores: [2.0, 1.0, 0.1]

Step 1: Exponentiate
    exp(2.0) = 7.39
    exp(1.0) = 2.72
    exp(0.1) = 1.11
    
Step 2: Sum = 7.39 + 2.72 + 1.11 = 11.22

Step 3: Normalize
    P1 = 7.39/11.22 = 0.66
    P2 = 2.72/11.22 = 0.24
    P3 = 1.11/11.22 = 0.10
    
Output: [0.66, 0.24, 0.10] (sum = 1.0)"""
    
    ax4.text(0.5, 0.25, example_text, fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    # Add properties
    properties_text = """Key Properties:
• Output range: [0, 1]
• Sum to 1.0
• Preserves order
• Differentiable"""
    
    ax4.text(0.9, 0.05, properties_text, fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))
    
    plt.suptitle('Softmax Function: Converting Scores to Probabilities', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/softmax_2d_visual.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_simple_softmax_diagram():
    """Create a simple diagram showing softmax in action"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Input scores
    words = ['king', 'queen', 'man', 'woman', 'child']
    scores = np.array([3.2, 2.8, 1.5, 1.3, 0.2])
    probs = softmax(scores)
    
    # Create visual flow
    y_positions = np.linspace(0.8, 0.2, len(words))
    
    # Draw input scores
    for i, (word, score) in enumerate(zip(words, scores)):
        # Input box
        rect = FancyBboxPatch((0.1, y_positions[i]-0.05), 0.15, 0.08,
                              boxstyle="round,pad=0.01",
                              facecolor='#FFE5B4', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(0.175, y_positions[i], f'{word}\n{score:.1f}', 
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Arrow to softmax
        ax.arrow(0.25, y_positions[i], 0.15, 0, head_width=0.02, 
                head_length=0.02, fc='gray', ec='gray')
    
    # Softmax box
    rect = FancyBboxPatch((0.4, 0.1), 0.2, 0.8,
                          boxstyle="round,pad=0.02",
                          facecolor='#E0E0E0', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(0.5, 0.5, 'SOFTMAX\nFUNCTION', ha='center', va='center',
           fontsize=12, fontweight='bold')
    
    # Mathematical formula in the middle
    ax.text(0.5, 0.05, r'$p_i = \frac{e^{s_i}}{\sum_j e^{s_j}}$',
           ha='center', va='center', fontsize=11, style='italic')
    
    # Draw output probabilities
    for i, (word, prob) in enumerate(zip(words, probs)):
        # Arrow from softmax
        ax.arrow(0.6, y_positions[i], 0.15, 0, head_width=0.02,
                head_length=0.02, fc='gray', ec='gray')
        
        # Output box with bar
        rect = FancyBboxPatch((0.75, y_positions[i]-0.05), 0.15, 0.08,
                              boxstyle="round,pad=0.01",
                              facecolor='#B4E5D4', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Probability bar
        bar_width = prob * 0.13
        bar_rect = plt.Rectangle((0.76, y_positions[i]-0.04), bar_width, 0.06,
                                 facecolor='#4ECDC4', alpha=0.7)
        ax.add_patch(bar_rect)
        
        ax.text(0.825, y_positions[i], f'{word}\n{prob:.3f}',
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add sum annotation
    ax.text(0.95, 0.5, f'Σ = 1.000', ha='left', va='center',
           fontsize=11, fontweight='bold', color='green')
    
    # Labels
    ax.text(0.175, 0.95, 'Input Scores', ha='center', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.95, 'Transformation', ha='center', fontsize=12, fontweight='bold')
    ax.text(0.825, 0.95, 'Output Probabilities', ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.title('Softmax in Neural Language Models: From Scores to Word Probabilities',
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('../figures/softmax_simple_flow.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating softmax visualizations...")
    
    print("1. Creating comprehensive 2D softmax visualization...")
    create_softmax_visualization()
    
    print("2. Creating simple softmax flow diagram...")
    create_simple_softmax_diagram()
    
    print("All softmax visualizations generated successfully!")