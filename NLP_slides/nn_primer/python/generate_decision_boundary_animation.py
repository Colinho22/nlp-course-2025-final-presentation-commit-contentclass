"""
Generate decision boundary animation showing how neural networks learn
Creates static frames that show the evolution of decision boundaries during training
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# ML Theme Colors
ML_BLUE = '#0066CC'
ML_ORANGE = '#FF7F0E'
ML_GREEN = '#2CA02C'
ML_RED = '#D62728'
ML_PURPLE = '#9467BD'

def generate_spiral_data(n_points=100, noise=0.2):
    """Generate two-class spiral dataset"""
    np.random.seed(42)
    n = n_points // 2

    # Class 1: clockwise spiral
    theta1 = np.linspace(0, 4 * np.pi, n) + np.random.randn(n) * noise
    r1 = np.linspace(0.5, 2, n) + np.random.randn(n) * noise * 0.1
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)

    # Class 2: counter-clockwise spiral
    theta2 = np.linspace(0, 4 * np.pi, n) + np.random.randn(n) * noise
    r2 = np.linspace(0.5, 2, n) + np.random.randn(n) * noise * 0.1
    x2 = -r2 * np.cos(theta2)
    y2 = -r2 * np.sin(theta2)

    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.hstack([np.zeros(n), np.ones(n)])

    return X, y

def simple_neural_net_decision(X, weights_hidden, bias_hidden, weights_output, bias_output):
    """Simple 2-layer neural network decision function"""
    # Hidden layer with ReLU
    hidden = np.maximum(0, np.dot(X, weights_hidden) + bias_hidden)
    # Output layer with sigmoid
    output = 1 / (1 + np.exp(-(np.dot(hidden, weights_output) + bias_output)))
    return output

def create_decision_boundary_evolution():
    """Create frames showing decision boundary evolution during training"""

    # Generate dataset
    X_train, y_train = generate_spiral_data(200)

    # Create figure with subplots for different training stages
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Training epochs to visualize
    epochs = [0, 1, 5, 10, 20, 50, 100, 200]

    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    for idx, epoch in enumerate(epochs):
        ax = axes[idx]

        # Simulate weight evolution (in practice, these would come from actual training)
        # Start with random weights and gradually improve them
        np.random.seed(42)

        if epoch == 0:
            # Random initialization
            W1 = np.random.randn(2, 16) * 0.5
            b1 = np.zeros(16)
            W2 = np.random.randn(16, 1) * 0.5
            b2 = 0
        else:
            # Simulate learning by adjusting weights
            # This is simplified - real training uses backpropagation
            progress = min(1.0, epoch / 100)

            # Gradually learn spiral pattern
            W1 = np.random.randn(2, 16) * (2 - 1.5 * progress)

            # Add structure to weights based on epoch
            for i in range(16):
                angle = 2 * np.pi * i / 16
                W1[0, i] = np.cos(angle) * (1 + 0.5 * progress)
                W1[1, i] = np.sin(angle) * (1 + 0.5 * progress)

            b1 = np.linspace(-1, 1, 16) * progress
            W2 = np.random.randn(16, 1) * (0.5 + progress)
            b2 = -0.5 * progress

        # Calculate decision boundary
        Z = simple_neural_net_decision(
            np.c_[xx.ravel(), yy.ravel()],
            W1, b1, W2.ravel(), b2
        )
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        contour = ax.contourf(xx, yy, Z, levels=[0, 0.5, 1],
                              colors=['#FFE6E6', '#E6E6FF'], alpha=0.6)
        boundary = ax.contour(xx, yy, Z, levels=[0.5], colors=['black'],
                             linewidths=2, linestyles='--')

        # Plot training points
        scatter1 = ax.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1],
                             c=ML_RED, s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
        scatter2 = ax.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1],
                             c=ML_BLUE, s=20, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Calculate and display accuracy
        predictions = simple_neural_net_decision(X_train, W1, b1, W2.ravel(), b2)
        accuracy = np.mean((predictions > 0.5) == y_train)

        ax.set_title(f'Epoch {epoch}\nAccuracy: {accuracy:.1%}', fontweight='bold')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Feature x₁')
        if idx % 4 == 0:
            ax.set_ylabel('Feature x₂')
        ax.grid(True, alpha=0.3)

        # Add text annotation for first and last
        if epoch == 0:
            ax.text(0.02, 0.98, 'Random\ninitialization',
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        elif epoch == 200:
            ax.text(0.02, 0.98, 'Converged\nsolution',
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Add main title
    fig.suptitle('Neural Network Learning: Decision Boundary Evolution',
                fontsize=16, fontweight='bold')

    # Add legend
    legend_elements = [
        plt.scatter([], [], c=ML_RED, s=50, edgecolors='black', label='Class 0'),
        plt.scatter([], [], c=ML_BLUE, s=50, edgecolors='black', label='Class 1'),
        plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Decision boundary')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
              bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig('../figures/decision_boundary_evolution.pdf', dpi=300, bbox_inches='tight')
    print("Saved: decision_boundary_evolution.pdf")

def create_xor_decision_boundaries():
    """Create visualization of XOR problem and solution with hidden layer"""

    # XOR data
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Create mesh
    h = 0.01
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Plot 1: Single perceptron (fails)
    ax = axes[0]
    # Best linear boundary (will fail for XOR)
    W_single = np.array([1, -1])
    b_single = 0
    Z_single = np.dot(np.c_[xx.ravel(), yy.ravel()], W_single) + b_single
    Z_single = Z_single.reshape(xx.shape)

    ax.contourf(xx, yy, Z_single > 0, levels=[0, 0.5, 1],
               colors=['#FFE6E6', '#E6E6FF'], alpha=0.6)
    ax.contour(xx, yy, Z_single, levels=[0], colors=['black'],
              linewidths=2, linestyles='--')

    # Plot XOR points
    colors = [ML_RED if y == 0 else ML_BLUE for y in y_xor]
    ax.scatter(X_xor[:, 0], X_xor[:, 1], c=colors, s=200,
              edgecolors='black', linewidth=2, zorder=5)

    # Add labels
    for i, (x, y) in enumerate(X_xor):
        ax.annotate(f'({int(x)},{int(y)})', (x, y),
                   xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax.set_title('Single Perceptron\n(Cannot solve XOR)', fontweight='bold', color='red')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Input x₁')
    ax.set_ylabel('Input x₂')
    ax.grid(True, alpha=0.3)

    # Plot 2: Two hidden neurons
    ax = axes[1]

    # Hidden neuron 1: (0,1) detector
    W1_h1 = np.array([-2, 2])
    b1_h1 = 1
    Z1 = np.dot(np.c_[xx.ravel(), yy.ravel()], W1_h1) + b1_h1

    # Hidden neuron 2: (1,0) detector
    W1_h2 = np.array([2, -2])
    b1_h2 = 1
    Z2 = np.dot(np.c_[xx.ravel(), yy.ravel()], W1_h2) + b1_h2

    # Show both hidden neuron boundaries
    Z1 = Z1.reshape(xx.shape)
    Z2 = Z2.reshape(xx.shape)

    ax.contour(xx, yy, Z1, levels=[0], colors=[ML_GREEN],
              linewidths=2, linestyles='-', alpha=0.7, label='Hidden 1')
    ax.contour(xx, yy, Z2, levels=[0], colors=[ML_PURPLE],
              linewidths=2, linestyles='-', alpha=0.7, label='Hidden 2')

    # Plot XOR points
    ax.scatter(X_xor[:, 0], X_xor[:, 1], c=colors, s=200,
              edgecolors='black', linewidth=2, zorder=5)

    ax.set_title('Two Hidden Neurons\n(Create regions)', fontweight='bold')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Input x₁')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 3: Combined solution
    ax = axes[2]

    # Combine hidden neurons with OR-like output
    H1 = 1 / (1 + np.exp(-Z1.ravel()))
    H2 = 1 / (1 + np.exp(-Z2.ravel()))
    Z_combined = H1 + H2 - 2 * H1 * H2  # XOR combination
    Z_combined = Z_combined.reshape(xx.shape)

    ax.contourf(xx, yy, Z_combined > 0.5, levels=[0, 0.5, 1],
               colors=['#FFE6E6', '#E6E6FF'], alpha=0.6)
    ax.contour(xx, yy, Z_combined, levels=[0.5], colors=['black'],
              linewidths=3, linestyles='-')

    # Plot XOR points
    ax.scatter(X_xor[:, 0], X_xor[:, 1], c=colors, s=200,
              edgecolors='black', linewidth=2, zorder=5)

    # Add checkmarks for correct classification
    for i, (x, y) in enumerate(X_xor):
        ax.annotate('✓', (x, y), xytext=(0, -25),
                   textcoords='offset points', fontsize=16,
                   color='green', fontweight='bold', ha='center')

    ax.set_title('Combined Output\n(XOR solved!)', fontweight='bold', color='green')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Input x₁')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Solving XOR: Why We Need Hidden Layers',
                fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/xor_solution_visualization.pdf', dpi=300, bbox_inches='tight')
    print("Saved: xor_solution_visualization.pdf")

def create_gradient_flow_visualization():
    """Visualize gradient flow in a neural network"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Network architecture
    layers = [3, 4, 3, 2]  # neurons per layer

    # Plot 1: Normal gradient flow
    ax = axes[0]

    # Draw network
    layer_spacing = 1.0
    neuron_spacing = 1.0

    positions = []
    for i, n_neurons in enumerate(layers):
        x = i * layer_spacing
        y_start = -(n_neurons - 1) * neuron_spacing / 2
        for j in range(n_neurons):
            y = y_start + j * neuron_spacing
            positions.append((x, y, i, j))

            # Draw neuron
            circle = plt.Circle((x, y), 0.15, color='lightblue',
                               edgecolor='darkblue', linewidth=2)
            ax.add_patch(circle)

    # Draw connections with gradient strength
    np.random.seed(42)
    for i in range(len(layers) - 1):
        for j in range(layers[i]):
            for k in range(layers[i + 1]):
                x1 = i * layer_spacing
                y1 = -(layers[i] - 1) * neuron_spacing / 2 + j * neuron_spacing
                x2 = (i + 1) * layer_spacing
                y2 = -(layers[i + 1] - 1) * neuron_spacing / 2 + k * neuron_spacing

                # Simulate gradient magnitude (decreases with depth)
                gradient_strength = 0.8 ** i * (0.5 + 0.5 * np.random.rand())

                ax.plot([x1, x2], [y1, y2], 'b-',
                       alpha=gradient_strength * 0.7,
                       linewidth=gradient_strength * 3)

    ax.set_title('Healthy Gradient Flow', fontweight='bold', color='green', fontsize=14)
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_ylim(-2.5, 2.5)
    ax.axis('off')

    # Add layer labels
    for i, n in enumerate(layers):
        ax.text(i * layer_spacing, -2.2, f'Layer {i}\n({n} neurons)',
               ha='center', fontsize=10)

    # Add gradient strength indicator
    ax.text(0.5, 2.8, 'Gradient Strength:', fontsize=10, fontweight='bold')
    for i in range(5):
        strength = 1.0 - i * 0.2
        ax.plot([1.5 + i * 0.3, 1.7 + i * 0.3], [2.8, 2.8], 'b-',
               alpha=strength * 0.7, linewidth=strength * 3)

    # Plot 2: Vanishing gradient problem
    ax = axes[1]

    # Draw same network
    for i, n_neurons in enumerate(layers):
        x = i * layer_spacing
        y_start = -(n_neurons - 1) * neuron_spacing / 2
        for j in range(n_neurons):
            y = y_start + j * neuron_spacing

            # Color neurons based on gradient
            gradient_at_layer = 0.1 ** (i + 1)
            color = plt.cm.Blues(0.3 + 0.7 * gradient_at_layer)
            circle = plt.Circle((x, y), 0.15, color=color,
                               edgecolor='darkblue', linewidth=2)
            ax.add_patch(circle)

    # Draw connections with vanishing gradients
    for i in range(len(layers) - 1):
        for j in range(layers[i]):
            for k in range(layers[i + 1]):
                x1 = i * layer_spacing
                y1 = -(layers[i] - 1) * neuron_spacing / 2 + j * neuron_spacing
                x2 = (i + 1) * layer_spacing
                y2 = -(layers[i + 1] - 1) * neuron_spacing / 2 + k * neuron_spacing

                # Exponentially decreasing gradient
                gradient_strength = 0.3 ** (i + 1) * (0.5 + 0.5 * np.random.rand())

                ax.plot([x1, x2], [y1, y2], 'b-',
                       alpha=max(0.1, gradient_strength * 0.7),
                       linewidth=max(0.5, gradient_strength * 3))

    ax.set_title('Vanishing Gradient Problem', fontweight='bold', color='red', fontsize=14)
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_ylim(-2.5, 2.5)
    ax.axis('off')

    # Add layer labels with gradient values
    for i, n in enumerate(layers):
        gradient_val = 0.1 ** (i + 1)
        ax.text(i * layer_spacing, -2.2, f'Layer {i}\nGrad: {gradient_val:.4f}',
               ha='center', fontsize=10)

    # Add warning text
    ax.text(1.5, -3.2, '⚠ Gradients vanish exponentially!',
           ha='center', fontsize=11, color='red', fontweight='bold')

    fig.suptitle('Gradient Flow in Neural Networks', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/gradient_flow_visualization.pdf', dpi=300, bbox_inches='tight')
    print("Saved: gradient_flow_visualization.pdf")

if __name__ == "__main__":
    # Generate all visualizations
    print("Generating decision boundary visualizations...")

    create_decision_boundary_evolution()
    create_xor_decision_boundaries()
    create_gradient_flow_visualization()

    print("\nAll decision boundary visualizations generated successfully!")
    print("Files created:")
    print("  - decision_boundary_evolution.pdf")
    print("  - xor_solution_visualization.pdf")
    print("  - gradient_flow_visualization.pdf")