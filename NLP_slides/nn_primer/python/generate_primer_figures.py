"""
Generate figures for Neural Networks Primer presentation
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as patches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color scheme - minimalist monochromatic
COLOR_MAIN = '#404040'     # RGB(64,64,64)
COLOR_ACCENT = '#B4B4B4'   # RGB(180,180,180)
COLOR_LIGHT = '#F0F0F0'    # RGB(240,240,240)

# Educational colors for specific visualizations
COLOR_CURRENT = '#FF6B6B'  # Red - focus/current
COLOR_CONTEXT = '#4ECDC4'  # Teal - context
COLOR_PREDICT = '#95E77E'  # Green - output

def create_various_a_styles():
    """Show different styles of letter A to illustrate pattern recognition challenge"""
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.patch.set_facecolor('white')

    # Different A styles
    a_styles = [
        ('Arial', 'normal', 'A'),
        ('Times New Roman', 'italic', 'A'),
        ('Courier', 'normal', 'A'),
        ('Comic Sans MS', 'normal', 'A'),
        ('Handwritten 1', 'normal', 'A'),
        ('Handwritten 2', 'normal', 'a'),
        ('Rotated', 'normal', 'A'),
        ('Partial', 'normal', 'A')
    ]

    for idx, ax in enumerate(axes.flat):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        if idx < len(a_styles):
            font, style, char = a_styles[idx]
            # Simulate different styles with available fonts
            if idx < 4:
                ax.text(0.5, 0.5, char, fontsize=60, ha='center', va='center',
                       fontstyle=style, color=COLOR_MAIN)
            elif idx == 4:  # Handwritten simulation
                ax.text(0.5, 0.5, 'A', fontsize=50, ha='center', va='center',
                       rotation=5, color=COLOR_MAIN)
            elif idx == 5:  # Lowercase
                ax.text(0.5, 0.5, 'a', fontsize=50, ha='center', va='center',
                       color=COLOR_MAIN)
            elif idx == 6:  # Rotated
                ax.text(0.5, 0.5, 'A', fontsize=50, ha='center', va='center',
                       rotation=30, color=COLOR_MAIN)
            elif idx == 7:  # Partial
                ax.text(0.5, 0.5, 'Λ', fontsize=50, ha='center', va='center',
                       color=COLOR_ACCENT)

    plt.suptitle('The Challenge: Infinite Variations of "A"', fontsize=14, color=COLOR_MAIN)
    plt.tight_layout()
    plt.savefig('../figures/various_a_styles.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_xor_visualization():
    """Visualize why XOR is not linearly separable"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('white')

    # XOR data points
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    # Left plot: The problem
    ax1.scatter(X[y==0, 0], X[y==0, 1], s=200, c=COLOR_CURRENT, marker='o', label='Class 0', edgecolor='black', linewidth=2)
    ax1.scatter(X[y==1, 0], X[y==1, 1], s=200, c=COLOR_CONTEXT, marker='s', label='Class 1', edgecolor='black', linewidth=2)

    # Try to draw a line (impossible)
    x_line = np.linspace(-0.5, 1.5, 100)
    y_line = 1 - x_line
    ax1.plot(x_line, y_line, '--', color=COLOR_ACCENT, linewidth=2, alpha=0.7)

    ax1.set_xlim(-0.2, 1.2)
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_xlabel('Input 1', fontsize=12)
    ax1.set_ylabel('Input 2', fontsize=12)
    ax1.set_title('XOR: No Single Line Can Separate', fontsize=12, color=COLOR_MAIN)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right plot: The solution with hidden layer
    ax2.scatter(X[y==0, 0], X[y==0, 1], s=200, c=COLOR_CURRENT, marker='o', edgecolor='black', linewidth=2)
    ax2.scatter(X[y==1, 0], X[y==1, 1], s=200, c=COLOR_CONTEXT, marker='s', edgecolor='black', linewidth=2)

    # Draw two lines
    ax2.plot([0.5, 0.5], [-0.5, 1.5], '-', color=COLOR_PREDICT, linewidth=2, alpha=0.7, label='Hidden 1')
    ax2.plot([-0.5, 1.5], [0.5, 0.5], '-', color=COLOR_PREDICT, linewidth=2, alpha=0.7, label='Hidden 2')

    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_xlabel('Input 1', fontsize=12)
    ax2.set_ylabel('Input 2', fontsize=12)
    ax2.set_title('Solution: Two Lines (Hidden Layer)', fontsize=12, color=COLOR_MAIN)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle('The XOR Crisis (1969)', fontsize=14, fontweight='bold', color=COLOR_MAIN)
    plt.tight_layout()
    plt.savefig('../figures/xor_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_multilayer_network():
    """Create a simple multi-layer network visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.axis('off')

    # Network architecture
    layers = [2, 3, 2, 1]  # nodes in each layer
    layer_names = ['Input\n(x₁, x₂)', 'Hidden 1\n(Features)', 'Hidden 2\n(Combinations)', 'Output\n(Decision)']

    # Positions
    x_positions = [0.2, 0.4, 0.6, 0.8]

    # Draw nodes
    nodes = []
    for layer_idx, (n_nodes, x_pos, name) in enumerate(zip(layers, x_positions, layer_names)):
        layer_nodes = []
        y_positions = np.linspace(0.2, 0.8, n_nodes)

        for y_pos in y_positions:
            if layer_idx == 0:
                color = COLOR_CONTEXT
            elif layer_idx == len(layers) - 1:
                color = COLOR_PREDICT
            else:
                color = COLOR_ACCENT

            circle = Circle((x_pos, y_pos), 0.03, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            layer_nodes.append((x_pos, y_pos))

        nodes.append(layer_nodes)

        # Add layer label
        ax.text(x_pos, 0.05, name, ha='center', fontsize=10, color=COLOR_MAIN)

    # Draw connections
    for layer_idx in range(len(layers) - 1):
        for node1 in nodes[layer_idx]:
            for node2 in nodes[layer_idx + 1]:
                ax.plot([node1[0], node2[0]], [node1[1], node2[1]],
                       'k-', alpha=0.2, linewidth=0.5)

    # Add annotations
    ax.text(0.5, 0.95, 'Information Flow →', ha='center', fontsize=12,
           fontweight='bold', color=COLOR_MAIN)
    ax.text(0.5, 0.9, 'Each connection has a weight (w), each node has a bias (b)',
           ha='center', fontsize=10, color=COLOR_ACCENT)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.title('Multi-Layer Network: Solving Complex Problems', fontsize=14,
             fontweight='bold', pad=20, color=COLOR_MAIN)
    plt.savefig('../figures/multilayer_network.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_cnn_feature_hierarchy():
    """Show CNN feature hierarchy from edges to objects"""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    fig.patch.set_facecolor('white')

    titles = ['Layer 1: Edges', 'Layer 2: Corners', 'Layer 3: Parts', 'Layer 4: Objects']

    for idx, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')

        if idx == 0:  # Edges
            ax.plot([2, 8], [5, 5], 'k-', linewidth=3)
            ax.plot([5, 5], [2, 8], 'k-', linewidth=3)
            ax.plot([2, 8], [2, 8], 'k-', linewidth=3)
        elif idx == 1:  # Corners
            ax.plot([2, 5, 5], [5, 5, 8], 'k-', linewidth=3)
            ax.plot([5, 8, 8], [2, 2, 5], 'k-', linewidth=3)
        elif idx == 2:  # Parts
            circle = plt.Circle((5, 5), 2, fill=False, linewidth=3, color='black')
            ax.add_patch(circle)
            ax.plot([3, 7], [3, 3], 'k-', linewidth=3)
        elif idx == 3:  # Object (simple car)
            # Car body
            rect = plt.Rectangle((2, 4), 6, 3, fill=False, linewidth=3, color='black')
            ax.add_patch(rect)
            # Wheels
            circle1 = plt.Circle((3.5, 3.5), 0.8, fill=False, linewidth=2, color='black')
            circle2 = plt.Circle((6.5, 3.5), 0.8, fill=False, linewidth=2, color='black')
            ax.add_patch(circle1)
            ax.add_patch(circle2)

        ax.set_title(title, fontsize=11, color=COLOR_MAIN)

    plt.suptitle('CNN: Building Complex Recognition from Simple Features',
                fontsize=13, fontweight='bold', color=COLOR_MAIN)
    plt.tight_layout()
    plt.savefig('../figures/cnn_feature_hierarchy.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_gradient_descent_visualization():
    """Visualize gradient descent as hiking down a mountain"""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create loss surface
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (X**2 + 2*Y**2) + 0.1*X*Y + np.sin(X)*0.2

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.6, edgecolor='none')

    # Gradient descent path
    path_x = [2.5]
    path_y = [2.5]
    learning_rate = 0.2

    for _ in range(15):
        # Approximate gradient
        grad_x = path_x[-1] + 0.1 * np.sin(path_x[-1])
        grad_y = 2 * path_y[-1]

        # Update position
        new_x = path_x[-1] - learning_rate * grad_x
        new_y = path_y[-1] - learning_rate * grad_y

        path_x.append(new_x)
        path_y.append(new_y)

    path_z = [0.5 * (x**2 + 2*y**2) + 0.1*x*y + np.sin(x)*0.2 for x, y in zip(path_x, path_y)]

    # Plot path
    ax.plot(path_x, path_y, path_z, 'r-', linewidth=3, label='Gradient Descent Path')
    ax.scatter(path_x[0], path_y[0], path_z[0], color='green', s=100, label='Start')
    ax.scatter(path_x[-1], path_y[-1], path_z[-1], color='red', s=100, label='Minimum')

    ax.set_xlabel('Weight 1', fontsize=10)
    ax.set_ylabel('Weight 2', fontsize=10)
    ax.set_zlabel('Loss', fontsize=10)
    ax.set_title('Gradient Descent: Finding the Lowest Point', fontsize=12, color=COLOR_MAIN)
    ax.legend()

    plt.savefig('../figures/gradient_descent.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_activation_functions():
    """Plot common activation functions"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor('white')

    x = np.linspace(-5, 5, 100)

    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x))
    axes[0].plot(x, sigmoid, linewidth=3, color=COLOR_CURRENT)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Sigmoid: Smooth 0-1', fontsize=11, color=COLOR_MAIN)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('σ(x)')
    axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # ReLU
    relu = np.maximum(0, x)
    axes[1].plot(x, relu, linewidth=3, color=COLOR_CONTEXT)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('ReLU: Simple and Fast', fontsize=11, color=COLOR_MAIN)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('ReLU(x)')
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Tanh
    tanh = np.tanh(x)
    axes[2].plot(x, tanh, linewidth=3, color=COLOR_PREDICT)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Tanh: Centered -1 to 1', fontsize=11, color=COLOR_MAIN)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('tanh(x)')
    axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[2].axvline(x=0, color='k', linestyle='-', alpha=0.3)

    plt.suptitle('Activation Functions: Adding Non-linearity', fontsize=13,
                fontweight='bold', color=COLOR_MAIN)
    plt.tight_layout()
    plt.savefig('../figures/activation_functions.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_2d_classification_evolution():
    """Show evolution of 2D classification boundary during training"""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    fig.patch.set_facecolor('white')

    # Generate sample data
    np.random.seed(42)
    n_samples = 100

    # Class 0 (red) - bottom left and top right
    class0_1 = np.random.randn(n_samples//2, 2) * 0.5 + [-1.5, -1.5]
    class0_2 = np.random.randn(n_samples//2, 2) * 0.5 + [1.5, 1.5]
    class0 = np.vstack([class0_1, class0_2])

    # Class 1 (blue) - top left and bottom right
    class1_1 = np.random.randn(n_samples//2, 2) * 0.5 + [-1.5, 1.5]
    class1_2 = np.random.randn(n_samples//2, 2) * 0.5 + [1.5, -1.5]
    class1 = np.vstack([class1_1, class1_2])

    epochs = [1, 10, 50, 100]

    for idx, (ax, epoch) in enumerate(zip(axes, epochs)):
        # Plot data points
        ax.scatter(class0[:, 0], class0[:, 1], c=COLOR_CURRENT, s=20, alpha=0.6, label='Class 0')
        ax.scatter(class1[:, 0], class1[:, 1], c=COLOR_CONTEXT, s=20, alpha=0.6, label='Class 1')

        # Create decision boundary (simulated)
        x_min, x_max = -3, 3
        y_min, y_max = -3, 3
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                             np.linspace(y_min, y_max, 50))

        if idx == 0:  # Random boundary
            Z = np.random.randn(*xx.shape) * 0.1
        elif idx == 1:  # Rough separation
            Z = (xx + yy) * 0.3 + np.random.randn(*xx.shape) * 0.05
        elif idx == 2:  # Good boundary
            Z = np.sin(xx*0.5) * np.cos(yy*0.5) + (xx*yy)*0.1
        else:  # Perfect fit
            Z = np.sin(xx*0.5) * np.cos(yy*0.5) + (xx*yy)*0.1
            Z = Z * (1 + 0.1*np.exp(-((xx**2 + yy**2))))

        ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
        ax.contourf(xx, yy, Z, levels=[-1000, 0, 1000], colors=[COLOR_CURRENT, COLOR_CONTEXT], alpha=0.1)

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title(f'Epoch {epoch}', fontsize=11, color=COLOR_MAIN)
        ax.grid(True, alpha=0.2)

        if idx == 0:
            ax.set_ylabel('Feature 2', fontsize=10)
        ax.set_xlabel('Feature 1', fontsize=10)

    plt.suptitle('Learning to Classify: Decision Boundary Evolution',
                fontsize=13, fontweight='bold', color=COLOR_MAIN)
    plt.tight_layout()
    plt.savefig('../figures/2d_classification_evolution.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_overfitting_visualization():
    """Show underfitting, good fit, and overfitting"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor('white')

    # Generate data
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y_true = np.sin(x) + np.random.randn(50) * 0.3

    # Underfitting (linear)
    z1 = np.polyfit(x, y_true, 1)
    p1 = np.poly1d(z1)

    # Good fit (polynomial degree 3)
    z3 = np.polyfit(x, y_true, 3)
    p3 = np.poly1d(z3)

    # Overfitting (polynomial degree 15)
    z15 = np.polyfit(x, y_true, 15)
    p15 = np.poly1d(z15)

    x_smooth = np.linspace(0, 10, 200)

    titles = ['Underfitting', 'Good Fit', 'Overfitting']
    models = [p1, p3, p15]
    colors = [COLOR_ACCENT, COLOR_PREDICT, COLOR_CURRENT]

    for ax, title, model, color in zip(axes, titles, models, colors):
        ax.scatter(x, y_true, alpha=0.6, s=20, color='gray', label='Training Data')
        ax.plot(x_smooth, model(x_smooth), linewidth=2, color=color, label='Model')
        ax.set_title(title, fontsize=11, color=COLOR_MAIN)
        ax.set_xlabel('Input', fontsize=10)
        ax.set_ylabel('Output', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(-3, 3)

    plt.suptitle('The Goldilocks Problem: Finding the Right Complexity',
                fontsize=13, fontweight='bold', color=COLOR_MAIN)
    plt.tight_layout()
    plt.savefig('../figures/overfitting_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_resnet_skip_connection():
    """Visualize ResNet skip connections"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')

    # Traditional deep network (left)
    ax1.axis('off')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Draw layers
    layer_positions = np.linspace(0.1, 0.9, 6)
    for i, pos in enumerate(layer_positions):
        rect = FancyBboxPatch((0.3, pos-0.04), 0.4, 0.08,
                              boxstyle="round,pad=0.01",
                              facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN, linewidth=2)
        ax1.add_patch(rect)
        ax1.text(0.5, pos, f'Layer {i+1}', ha='center', va='center', fontsize=10, color=COLOR_MAIN)

        if i < len(layer_positions) - 1:
            arrow = FancyArrowPatch((0.5, pos+0.04), (0.5, layer_positions[i+1]-0.04),
                                  arrowstyle='->', mutation_scale=20, linewidth=2,
                                  color=COLOR_ACCENT)
            ax1.add_patch(arrow)

    ax1.text(0.5, 0.02, 'Traditional Deep Network', ha='center', fontsize=11,
            fontweight='bold', color=COLOR_MAIN)
    ax1.text(0.1, 0.5, '←', ha='center', fontsize=20, color='red')
    ax1.text(0.05, 0.5, 'Gradient\nvanishes', ha='center', fontsize=9, color='red')

    # ResNet with skip connections (right)
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # Draw layers with skip connections
    for i, pos in enumerate(layer_positions):
        rect = FancyBboxPatch((0.3, pos-0.04), 0.4, 0.08,
                              boxstyle="round,pad=0.01",
                              facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN, linewidth=2)
        ax2.add_patch(rect)
        ax2.text(0.5, pos, f'F(x) + x', ha='center', va='center', fontsize=10, color=COLOR_MAIN)

        if i < len(layer_positions) - 1:
            # Regular connection
            arrow = FancyArrowPatch((0.5, pos+0.04), (0.5, layer_positions[i+1]-0.04),
                                  arrowstyle='->', mutation_scale=20, linewidth=2,
                                  color=COLOR_ACCENT)
            ax2.add_patch(arrow)

            # Skip connection
            if i % 2 == 0 and i < len(layer_positions) - 2:
                skip_arrow = FancyArrowPatch((0.72, pos), (0.72, layer_positions[i+2]),
                                           arrowstyle='->', mutation_scale=15, linewidth=2,
                                           color=COLOR_PREDICT, linestyle='dashed')
                ax2.add_patch(skip_arrow)

    ax2.text(0.5, 0.02, 'ResNet with Skip Connections', ha='center', fontsize=11,
            fontweight='bold', color=COLOR_MAIN)
    ax2.text(0.85, 0.5, 'Skip\nConnections', ha='center', fontsize=9, color=COLOR_PREDICT)

    plt.suptitle('ResNet Innovation: Solving the Vanishing Gradient',
                fontsize=13, fontweight='bold', color=COLOR_MAIN)
    plt.tight_layout()
    plt.savefig('../figures/resnet_skip_connection.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_scale_growth_chart():
    """Show exponential growth in model parameters"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')

    # Data
    years = [1957, 1989, 1998, 2012, 2018, 2020, 2023]
    models = ['Perceptron', 'NetTalk', 'LeNet', 'AlexNet', 'BERT', 'GPT-3', 'GPT-4']
    params = [20, 18000, 60000, 60000000, 340000000, 175000000000, 1800000000000]

    # Convert to log scale
    log_params = np.log10(params)

    # Plot
    ax.plot(years, log_params, 'o-', linewidth=3, markersize=10, color=COLOR_CURRENT)

    # Add model labels
    for year, model, log_param in zip(years, models, log_params):
        ax.annotate(model, (year, log_param), textcoords="offset points",
                   xytext=(0,10), ha='center', fontsize=9, color=COLOR_MAIN)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Parameters (log₁₀ scale)', fontsize=12)
    ax.set_title('The Exponential Growth of Neural Networks', fontsize=14,
                fontweight='bold', color=COLOR_MAIN)

    # Custom y-axis labels
    y_labels = ['10¹', '10⁴', '10⁵', '10⁷', '10⁸', '10¹¹', '10¹²⁺']
    ax.set_yticks([1, 4, 5, 7, 8, 11, 12])
    ax.set_yticklabels(y_labels)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(1950, 2030)

    # Add annotations
    ax.text(1990, 11, 'Each 10x increase\nunlocks new capabilities',
           fontsize=10, color=COLOR_ACCENT, style='italic')

    plt.tight_layout()
    plt.savefig('../figures/scale_growth_chart.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all figures for the Neural Networks primer"""
    print("Generating Neural Networks Primer figures...")

    # Create all figures
    create_various_a_styles()
    print("- Various A styles")

    create_xor_visualization()
    print("- XOR visualization")

    create_multilayer_network()
    print("- Multi-layer network")

    create_cnn_feature_hierarchy()
    print("- CNN feature hierarchy")

    create_gradient_descent_visualization()
    print("- Gradient descent")

    create_activation_functions()
    print("- Activation functions")

    create_2d_classification_evolution()
    print("- 2D classification evolution")

    create_overfitting_visualization()
    print("- Overfitting visualization")

    create_resnet_skip_connection()
    print("- ResNet skip connections")

    create_scale_growth_chart()
    print("- Scale growth chart")

    print("\nAll figures generated successfully!")

if __name__ == "__main__":
    main()