import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import matplotlib.patches as mpatches

# Set the minimalist style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Minimalist color palette
COLOR_MAIN = '#404040'      # RGB(64,64,64)
COLOR_ACCENT = '#B4B4B4'    # RGB(180,180,180)
COLOR_LIGHT = '#F0F0F0'     # RGB(240,240,240)
COLOR_TRUE = '#4682B4'       # Blue for true function
COLOR_APPROX = '#FF6B6B'     # Red for approximation
COLOR_NEURONS = '#95E77E'    # Green for neurons

def generate_function_approx_basics():
    """Show what function approximation means with a simple example"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # True function (house prices)
    ax = axes[0]
    ax.set_title('Real World: House Prices', fontsize=10, color=COLOR_MAIN)

    # Generate some realistic house data
    sizes = np.array([50, 75, 100, 120, 150, 180, 200, 250])
    prices = np.array([150, 180, 250, 280, 350, 400, 450, 550])

    # Scatter plot of data points
    ax.scatter(sizes, prices, s=50, c=COLOR_TRUE, alpha=0.6, label='Real houses')
    ax.set_xlabel('Size (m²)', fontsize=9, color=COLOR_MAIN)
    ax.set_ylabel('Price (k€)', fontsize=9, color=COLOR_MAIN)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper left', fontsize=8)

    # Mathematical function
    ax = axes[1]
    ax.set_title('Mathematical View', fontsize=10, color=COLOR_MAIN)

    # Show as continuous function
    x_smooth = np.linspace(40, 260, 100)
    y_true = 100 + 1.8 * x_smooth + 0.0005 * x_smooth**2
    ax.plot(x_smooth, y_true, '-', color=COLOR_TRUE, linewidth=2, label='True function f(x)')
    ax.scatter(sizes, prices, s=30, c=COLOR_TRUE, alpha=0.3)

    # Add question mark for unknown function
    ax.text(150, 250, '?', fontsize=40, color=COLOR_ACCENT, alpha=0.3, ha='center')
    ax.text(150, 200, 'f(size) = ?', fontsize=9, color=COLOR_MAIN, ha='center', style='italic')

    ax.set_xlabel('Input (x)', fontsize=9, color=COLOR_MAIN)
    ax.set_ylabel('Output f(x)', fontsize=9, color=COLOR_MAIN)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper left', fontsize=8)

    # Neural network approximation
    ax = axes[2]
    ax.set_title('Neural Network Solution', fontsize=10, color=COLOR_MAIN)

    # Show NN approximation (slightly different)
    y_approx = 105 + 1.75 * x_smooth + 0.0006 * x_smooth**2 + 5*np.sin(0.05*x_smooth)
    ax.plot(x_smooth, y_true, '--', color=COLOR_TRUE, linewidth=1.5, alpha=0.5, label='True f(x)')
    ax.plot(x_smooth, y_approx, '-', color=COLOR_APPROX, linewidth=2, label='NN approximation')
    ax.scatter(sizes, prices, s=30, c=COLOR_TRUE, alpha=0.3)

    # Show it learns from examples
    for s, p in zip(sizes, prices):
        ax.plot([s, s], [p-5, p+5], 'g-', alpha=0.3, linewidth=4)

    ax.set_xlabel('Input (x)', fontsize=9, color=COLOR_MAIN)
    ax.set_ylabel('Output NN(x)', fontsize=9, color=COLOR_MAIN)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper left', fontsize=8)

    plt.suptitle('Function Approximation: Learning Patterns from Examples',
                 fontsize=12, color=COLOR_MAIN, y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/function_approx_basics.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_nn_building_blocks():
    """Show how NNs build complex functions from simple pieces"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    x = np.linspace(-3, 3, 200)

    # Row 1: Individual neurons
    # Neuron 1 - Step function
    ax = axes[0, 0]
    ax.set_title('Neuron 1: Step Function', fontsize=9, color=COLOR_MAIN)
    y1 = (x > 0).astype(float)
    ax.plot(x, y1, color=COLOR_NEURONS, linewidth=2)
    ax.fill_between(x, 0, y1, alpha=0.2, color=COLOR_NEURONS)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('Output', fontsize=8)

    # Neuron 2 - Another step
    ax = axes[0, 1]
    ax.set_title('Neuron 2: Shifted Step', fontsize=9, color=COLOR_MAIN)
    y2 = (x > -1).astype(float) * 0.5
    ax.plot(x, y2, color=COLOR_NEURONS, linewidth=2)
    ax.fill_between(x, 0, y2, alpha=0.2, color=COLOR_NEURONS)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('x', fontsize=8)

    # Neuron 3 - Negative step
    ax = axes[0, 2]
    ax.set_title('Neuron 3: Negative Step', fontsize=9, color=COLOR_MAIN)
    y3 = (x < 1).astype(float) * 0.7
    ax.plot(x, y3, color=COLOR_NEURONS, linewidth=2)
    ax.fill_between(x, 0, y3, alpha=0.2, color=COLOR_NEURONS)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('x', fontsize=8)

    # Row 2: Combinations
    # Simple sum
    ax = axes[1, 0]
    ax.set_title('Add Them Up', fontsize=9, color=COLOR_MAIN)
    y_sum = y1 + y2 + y3
    ax.plot(x, y_sum, color=COLOR_ACCENT, linewidth=2)
    ax.fill_between(x, 0, y_sum, alpha=0.2, color=COLOR_ACCENT)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('Combined Output', fontsize=8)

    # Weighted sum (smooth approximation)
    ax = axes[1, 1]
    ax.set_title('Smooth with Sigmoid', fontsize=9, color=COLOR_MAIN)
    # Use sigmoid for smoothing
    sigmoid = lambda z: 1 / (1 + np.exp(-2*z))
    y1_smooth = sigmoid(x)
    y2_smooth = sigmoid(x + 1) * 0.5
    y3_smooth = sigmoid(-x + 1) * 0.7
    y_smooth = y1_smooth + y2_smooth + y3_smooth
    ax.plot(x, y_smooth, color=COLOR_APPROX, linewidth=2)
    ax.fill_between(x, 0, y_smooth, alpha=0.2, color=COLOR_APPROX)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('x', fontsize=8)

    # Complex function
    ax = axes[1, 2]
    ax.set_title('Any Function!', fontsize=9, color=COLOR_MAIN)
    # Create a more complex target
    y_target = np.sin(x) + 0.3*np.cos(2*x) + 1.5
    ax.plot(x, y_target, '--', color=COLOR_TRUE, linewidth=1.5, alpha=0.5, label='Target')
    # Approximate with many neurons
    y_approx = sum([sigmoid((x - i*0.5)) * np.random.uniform(-0.5, 0.5) for i in range(-4, 5)]) + 1.5
    ax.plot(x, y_approx, color=COLOR_APPROX, linewidth=2, label='NN approx')
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('x', fontsize=8)
    ax.legend(fontsize=7)

    plt.suptitle('Building Complex Functions from Simple Neurons',
                 fontsize=12, color=COLOR_MAIN, y=0.98)

    # Add arrow annotations
    fig.text(0.5, 0.52, '↓ Combine ↓', ha='center', fontsize=10, color=COLOR_ACCENT)

    plt.tight_layout()
    plt.savefig('../figures/nn_building_blocks.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_universal_approximation():
    """Visualize the universal approximation theorem"""
    fig = plt.figure(figsize=(12, 6))

    # Create main plot area
    ax_main = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)

    # Create side panels
    ax_width = plt.subplot2grid((2, 3), (0, 2))
    ax_depth = plt.subplot2grid((2, 3), (1, 2))

    # Main plot - show approximation quality
    x = np.linspace(-2, 2, 200)
    y_true = np.sin(2*x) * np.exp(-0.5*np.abs(x)) + 0.3*x

    ax_main.plot(x, y_true, 'k-', linewidth=3, alpha=0.7, label='Any continuous function')

    # Show different approximation levels
    neurons_counts = [2, 5, 10, 20]
    colors = ['#FFB6B6', '#FF8B8B', '#FF6B6B', '#FF4B4B']

    for n, color in zip(neurons_counts, colors):
        # Simulate approximation with n neurons
        np.random.seed(42)
        y_approx = np.zeros_like(x)
        for i in range(n):
            center = np.random.uniform(-2, 2)
            width = 4.0 / n
            weight = np.random.uniform(-1, 1)
            y_approx += weight * np.exp(-(x - center)**2 / (2 * width**2))

        # Scale to match target roughly
        y_approx = y_approx * (np.max(y_true) - np.min(y_true)) / (np.max(y_approx) - np.min(y_approx) + 1e-6)
        y_approx += np.mean(y_true) - np.mean(y_approx)

        if n == 20:
            ax_main.plot(x, y_approx, color=color, linewidth=2, alpha=0.8, label=f'{n} neurons')
        else:
            ax_main.plot(x, y_approx, color=color, linewidth=1, alpha=0.4)

    ax_main.set_xlabel('Input', fontsize=10, color=COLOR_MAIN)
    ax_main.set_ylabel('Output', fontsize=10, color=COLOR_MAIN)
    ax_main.set_title('Universal Approximation: More Neurons = Better Fit', fontsize=11, color=COLOR_MAIN)
    ax_main.grid(True, alpha=0.2)
    ax_main.legend(loc='upper left', fontsize=9)

    # Width diagram
    ax_width.set_title('Network Width', fontsize=9, color=COLOR_MAIN)
    ax_width.set_xlim(0, 10)
    ax_width.set_ylim(0, 10)
    ax_width.axis('off')

    # Draw neurons
    for i in range(5):
        circle = Circle((2, 8-i*1.5), 0.4, color=COLOR_NEURONS, alpha=0.3)
        ax_width.add_patch(circle)
    for i in range(8):
        circle = Circle((5, 8.5-i), 0.3, color=COLOR_NEURONS, alpha=0.5)
        ax_width.add_patch(circle)
    for i in range(12):
        circle = Circle((8, 9-i*0.7), 0.2, color=COLOR_NEURONS, alpha=0.7)
        ax_width.add_patch(circle)

    ax_width.text(2, 0.5, 'Few', ha='center', fontsize=8, color=COLOR_MAIN)
    ax_width.text(5, 0.5, 'More', ha='center', fontsize=8, color=COLOR_MAIN)
    ax_width.text(8, 0.5, 'Many', ha='center', fontsize=8, color=COLOR_MAIN)
    ax_width.arrow(1, -0.2, 6, 0, head_width=0.3, head_length=0.3, fc=COLOR_ACCENT, ec=COLOR_ACCENT)

    # Depth diagram
    ax_depth.set_title('Network Depth', fontsize=9, color=COLOR_MAIN)
    ax_depth.set_xlim(0, 10)
    ax_depth.set_ylim(0, 10)
    ax_depth.axis('off')

    # Draw layers
    layer_x = [2, 4, 6, 8]
    layer_sizes = [3, 4, 4, 2]
    for x_pos, size in zip(layer_x, layer_sizes):
        for i in range(size):
            y_pos = 5 + (i - size/2) * 1.2
            circle = Circle((x_pos, y_pos), 0.3, color=COLOR_NEURONS, alpha=0.5)
            ax_depth.add_patch(circle)

    # Connect layers
    for i in range(len(layer_x)-1):
        ax_depth.plot([layer_x[i], layer_x[i+1]], [5, 5], 'k-', alpha=0.2, linewidth=1)

    ax_depth.text(5, 1, 'More layers = More complex patterns', ha='center', fontsize=8,
                  color=COLOR_MAIN, style='italic')

    plt.suptitle('The Universal Approximation Theorem (Cybenko, 1989)',
                 fontsize=12, color=COLOR_MAIN, weight='bold')
    fig.text(0.5, 0.02, '"A neural network with enough neurons can approximate ANY continuous function to ANY desired accuracy"',
             ha='center', fontsize=9, color=COLOR_ACCENT, style='italic')

    plt.tight_layout()
    plt.savefig('../figures/universal_approximation.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all figures
print("Generating function approximation figures...")
generate_function_approx_basics()
print("Generated: function_approx_basics.pdf")

generate_nn_building_blocks()
print("Generated: nn_building_blocks.pdf")

generate_universal_approximation()
print("Generated: universal_approximation.pdf")

print("\nAll function approximation figures generated successfully!")