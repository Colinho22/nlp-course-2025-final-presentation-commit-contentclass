"""
Generate 3D visualization of neuron functions with and without activation
Shows how activation functions transform the linear decision boundary
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.patches as mpatches

# Set style for consistent appearance
plt.style.use('seaborn-v0_8-whitegrid')

# ML Theme Color definitions from template
ML_BLUE = '#0066CC'
ML_PURPLE = '#3333B2'
ML_LAVENDER = '#ADADE0'
ML_ORANGE = '#FF7F0E'
ML_GREEN = '#2CA02C'

def linear_neuron(x1, x2, w1=1.0, w2=0.5, b=-0.5):
    """Linear combination without activation"""
    return w1 * x1 + w2 * x2 + b

def sigmoid_activation(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def relu_activation(z):
    """ReLU activation function"""
    return np.maximum(0, z)

def tanh_activation(z):
    """Tanh activation function"""
    return np.tanh(z)

def create_3d_neuron_visualization():
    """Create 3D visualization showing neuron with different activations"""

    # Create grid for visualization
    x1 = np.linspace(-3, 3, 50)
    x2 = np.linspace(-3, 3, 50)
    X1, X2 = np.meshgrid(x1, x2)

    # Calculate linear combination
    Z_linear = linear_neuron(X1, X2)

    # Apply different activation functions
    Z_sigmoid = sigmoid_activation(Z_linear)
    Z_relu = relu_activation(Z_linear)
    Z_tanh = tanh_activation(Z_linear)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Linear (no activation)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X1, X2, Z_linear, cmap='viridis',
                             alpha=0.8, edgecolor='none')
    ax1.set_xlabel('Input x₁', fontsize=10)
    ax1.set_ylabel('Input x₂', fontsize=10)
    ax1.set_zlabel('Output z', fontsize=10)
    ax1.set_title('Linear (No Activation)\nz = w₁x₁ + w₂x₂ + b', fontsize=12, fontweight='bold')
    ax1.view_init(elev=20, azim=45)

    # Add decision boundary line
    x_boundary = np.linspace(-3, 3, 100)
    y_boundary = (-1.0 * x_boundary + 0.5) / 0.5  # From w1*x1 + w2*x2 + b = 0
    z_boundary = np.zeros_like(x_boundary)
    ax1.plot(x_boundary, y_boundary, z_boundary, 'r-', linewidth=3, label='Decision boundary')

    # Plot 2: Sigmoid activation
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X1, X2, Z_sigmoid, cmap='plasma',
                             alpha=0.8, edgecolor='none')
    ax2.set_xlabel('Input x₁', fontsize=10)
    ax2.set_ylabel('Input x₂', fontsize=10)
    ax2.set_zlabel('Output y', fontsize=10)
    ax2.set_title('Sigmoid Activation\ny = 1/(1 + e⁻ᶻ)', fontsize=12, fontweight='bold')
    ax2.view_init(elev=20, azim=45)

    # Plot 3: ReLU activation
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(X1, X2, Z_relu, cmap='coolwarm',
                             alpha=0.8, edgecolor='none')
    ax3.set_xlabel('Input x₁', fontsize=10)
    ax3.set_ylabel('Input x₂', fontsize=10)
    ax3.set_zlabel('Output y', fontsize=10)
    ax3.set_title('ReLU Activation\ny = max(0, z)', fontsize=12, fontweight='bold')
    ax3.view_init(elev=20, azim=45)

    # Plot 4: Tanh activation
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    surf4 = ax4.plot_surface(X1, X2, Z_tanh, cmap='twilight',
                             alpha=0.8, edgecolor='none')
    ax4.set_xlabel('Input x₁', fontsize=10)
    ax4.set_ylabel('Input x₂', fontsize=10)
    ax4.set_zlabel('Output y', fontsize=10)
    ax4.set_title('Tanh Activation\ny = tanh(z)', fontsize=12, fontweight='bold')
    ax4.view_init(elev=20, azim=45)

    # Plot 5: Comparison of activation functions (2D)
    ax5 = fig.add_subplot(2, 3, 5)
    z_range = np.linspace(-5, 5, 1000)
    ax5.plot(z_range, z_range, label='Linear', linewidth=2, color=ML_BLUE)
    ax5.plot(z_range, sigmoid_activation(z_range), label='Sigmoid', linewidth=2, color=ML_ORANGE)
    ax5.plot(z_range, relu_activation(z_range), label='ReLU', linewidth=2, color=ML_GREEN)
    ax5.plot(z_range, tanh_activation(z_range), label='Tanh', linewidth=2, color=ML_PURPLE)
    ax5.set_xlabel('Input (z)', fontsize=11)
    ax5.set_ylabel('Output', fontsize=11)
    ax5.set_title('Activation Functions Comparison', fontsize=12, fontweight='bold')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax5.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Plot 6: Decision boundaries in 2D
    ax6 = fig.add_subplot(2, 3, 6)

    # Create contour plots for each activation
    contour_linear = ax6.contour(X1, X2, Z_linear, levels=[0], colors=ML_BLUE, linewidths=2)
    contour_sigmoid = ax6.contour(X1, X2, Z_sigmoid, levels=[0.5], colors=ML_ORANGE, linewidths=2)
    contour_relu = ax6.contour(X1, X2, Z_relu, levels=[0.01], colors=ML_GREEN, linewidths=2)
    contour_tanh = ax6.contour(X1, X2, Z_tanh, levels=[0], colors=ML_PURPLE, linewidths=2)

    # Add some sample points
    np.random.seed(42)
    n_points = 30
    points_class1 = np.random.randn(n_points, 2) - [1, 1]
    points_class2 = np.random.randn(n_points, 2) + [1, 1]

    ax6.scatter(points_class1[:, 0], points_class1[:, 1], c='blue', alpha=0.5, s=30)
    ax6.scatter(points_class2[:, 0], points_class2[:, 1], c='red', alpha=0.5, s=30)

    ax6.set_xlabel('Input x₁', fontsize=11)
    ax6.set_ylabel('Input x₂', fontsize=11)
    ax6.set_title('Decision Boundaries in 2D', fontsize=12, fontweight='bold')
    ax6.set_xlim(-3, 3)
    ax6.set_ylim(-3, 3)

    # Create legend
    legend_elements = [
        mpatches.Patch(color=ML_BLUE, label='Linear'),
        mpatches.Patch(color=ML_ORANGE, label='Sigmoid'),
        mpatches.Patch(color=ML_GREEN, label='ReLU'),
        mpatches.Patch(color=ML_PURPLE, label='Tanh')
    ]
    ax6.legend(handles=legend_elements, loc='upper right')
    ax6.grid(True, alpha=0.3)

    # Add main title
    fig.suptitle('Single Neuron Visualization: Effect of Activation Functions',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save the figure
    plt.savefig('../figures/neuron_3d_visualization.pdf', dpi=300, bbox_inches='tight')
    print("Saved: neuron_3d_visualization.pdf")

    # Create a second figure showing network depth effect
    create_network_depth_visualization()

def create_network_depth_visualization():
    """Create visualization showing how network depth affects function complexity"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Create more complex decision boundaries with multiple neurons
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    # Single neuron (linear boundary)
    Z1 = linear_neuron(X, Y, 1, 1, 0)
    axes[0, 0].contourf(X, Y, sigmoid_activation(Z1), levels=20, cmap='RdBu', alpha=0.7)
    axes[0, 0].set_title('1 Neuron\n(Linear boundary)', fontweight='bold')
    axes[0, 0].set_xlabel('x₁')
    axes[0, 0].set_ylabel('x₂')

    # Two neurons (can create corner)
    Z2a = linear_neuron(X, Y, 1, 0, -1)
    Z2b = linear_neuron(X, Y, 0, 1, -1)
    Z2 = sigmoid_activation(Z2a) + sigmoid_activation(Z2b)
    axes[0, 1].contourf(X, Y, Z2, levels=20, cmap='RdBu', alpha=0.7)
    axes[0, 1].set_title('2 Neurons\n(Corner boundary)', fontweight='bold')
    axes[0, 1].set_xlabel('x₁')

    # Four neurons (can create box)
    Z3a = linear_neuron(X, Y, 1, 0, -1)
    Z3b = linear_neuron(X, Y, -1, 0, -1)
    Z3c = linear_neuron(X, Y, 0, 1, -1)
    Z3d = linear_neuron(X, Y, 0, -1, -1)
    Z3 = sigmoid_activation(Z3a) + sigmoid_activation(Z3b) + sigmoid_activation(Z3c) + sigmoid_activation(Z3d)
    axes[0, 2].contourf(X, Y, Z3, levels=20, cmap='RdBu', alpha=0.7)
    axes[0, 2].set_title('4 Neurons\n(Box boundary)', fontweight='bold')
    axes[0, 2].set_xlabel('x₁')

    # Create XOR pattern (requires hidden layer)
    # Hidden layer neurons
    H1 = sigmoid_activation(linear_neuron(X, Y, 1, 1, -0.5))  # AND-like
    H2 = sigmoid_activation(linear_neuron(X, Y, 1, 1, -1.5))  # NAND-like
    # Output layer
    Z_xor = sigmoid_activation(linear_neuron(H1, -H2, 1, 1, -0.5))
    axes[1, 0].contourf(X, Y, Z_xor, levels=20, cmap='RdBu', alpha=0.7)
    axes[1, 0].set_title('2 Hidden + 1 Output\n(XOR pattern)', fontweight='bold')
    axes[1, 0].set_xlabel('x₁')
    axes[1, 0].set_ylabel('x₂')

    # Create circular pattern
    R = np.sqrt(X**2 + Y**2)
    Z_circle = sigmoid_activation(2 - R)
    axes[1, 1].contourf(X, Y, Z_circle, levels=20, cmap='RdBu', alpha=0.7)
    axes[1, 1].set_title('Many Neurons\n(Circular boundary)', fontweight='bold')
    axes[1, 1].set_xlabel('x₁')

    # Create complex pattern
    Z_complex = np.sin(X) * np.cos(Y) + 0.5 * np.sin(2*X) * np.cos(2*Y)
    Z_complex = sigmoid_activation(3 * Z_complex)
    axes[1, 2].contourf(X, Y, Z_complex, levels=20, cmap='RdBu', alpha=0.7)
    axes[1, 2].set_title('Deep Network\n(Complex pattern)', fontweight='bold')
    axes[1, 2].set_xlabel('x₁')

    # Add colorbar
    plt.colorbar(axes[1, 2].collections[0], ax=axes, label='Network Output')

    fig.suptitle('How Network Complexity Grows with Neurons and Layers',
                fontsize=16, fontweight='bold')

    plt.tight_layout()

    # Save the figure
    plt.savefig('../figures/network_complexity_visualization.pdf', dpi=300, bbox_inches='tight')
    print("Saved: network_complexity_visualization.pdf")

if __name__ == "__main__":
    # Create the visualizations
    create_3d_neuron_visualization()

    print("\nGenerated 3D neuron visualizations successfully!")
    print("Files created:")
    print("  - neuron_3d_visualization.pdf")
    print("  - network_complexity_visualization.pdf")