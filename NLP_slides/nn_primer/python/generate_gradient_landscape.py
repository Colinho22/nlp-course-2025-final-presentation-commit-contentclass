"""
Generate 3D gradient descent landscape visualizations
Shows optimization surface and the path taken by gradient descent
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# ML Theme Colors
ML_BLUE = '#0066CC'
ML_ORANGE = '#FF7F0E'
ML_GREEN = '#2CA02C'
ML_RED = '#D62728'

def create_3d_loss_landscape():
    """Create 3D loss landscape with gradient descent path"""

    fig = plt.figure(figsize=(16, 10))

    # Create subplots
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    # Create meshgrid for 3D surface
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    # Define different loss functions
    # 1. Simple convex (bowl-shaped)
    Z1 = X**2 + Y**2

    # 2. Saddle point
    Z2 = X**2 - Y**2

    # 3. Multiple local minima
    Z3 = np.sin(X) * np.cos(Y) + 0.1 * (X**2 + Y**2)

    # Plot 1: Convex surface with gradient descent
    surf1 = ax1.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.6, edgecolor='none')
    ax1.set_title('Ideal Case: Convex Loss\n(Easy optimization)', fontweight='bold')

    # Gradient descent path on convex
    gd_path_x = [2.5]
    gd_path_y = [2.5]
    gd_path_z = [Z1[85, 85]]
    lr = 0.1
    for i in range(20):
        grad_x = 2 * gd_path_x[-1]
        grad_y = 2 * gd_path_y[-1]
        new_x = gd_path_x[-1] - lr * grad_x
        new_y = gd_path_y[-1] - lr * grad_y
        gd_path_x.append(new_x)
        gd_path_y.append(new_y)
        gd_path_z.append(new_x**2 + new_y**2)

    ax1.plot(gd_path_x, gd_path_y, gd_path_z, 'r-o', linewidth=3, markersize=5, label='GD path')
    ax1.scatter([0], [0], [0], color='green', s=200, marker='*', label='Global minimum')
    ax1.set_xlabel('Weight 1')
    ax1.set_ylabel('Weight 2')
    ax1.set_zlabel('Loss')
    ax1.view_init(elev=20, azim=45)

    # Plot 2: Saddle point
    surf2 = ax2.plot_surface(X, Y, Z2, cmap='coolwarm', alpha=0.6, edgecolor='none')
    ax2.set_title('Saddle Point Problem\n(Gradient = 0, not minimum)', fontweight='bold')

    # Show saddle point
    ax2.scatter([0], [0], [0], color='red', s=200, marker='X', label='Saddle point')

    # Gradient descent path on saddle (gets stuck)
    gd_path_x2 = [0.5]
    gd_path_y2 = [0.1]
    gd_path_z2 = [0.5**2 - 0.1**2]
    lr = 0.1
    for i in range(15):
        grad_x = 2 * gd_path_x2[-1]
        grad_y = -2 * gd_path_y2[-1]
        new_x = gd_path_x2[-1] - lr * grad_x
        new_y = gd_path_y2[-1] - lr * grad_y
        gd_path_x2.append(new_x)
        gd_path_y2.append(new_y)
        gd_path_z2.append(new_x**2 - new_y**2)

    ax2.plot(gd_path_x2, gd_path_y2, gd_path_z2, 'r-o', linewidth=3, markersize=5)
    ax2.set_xlabel('Weight 1')
    ax2.set_ylabel('Weight 2')
    ax2.set_zlabel('Loss')
    ax2.view_init(elev=20, azim=45)

    # Plot 3: Multiple local minima
    surf3 = ax3.plot_surface(X, Y, Z3, cmap='plasma', alpha=0.6, edgecolor='none')
    ax3.set_title('Multiple Local Minima\n(Can get stuck)', fontweight='bold')

    # Show multiple minima
    local_minima = [(1.57, 1.57), (-1.57, -1.57), (1.57, -1.57), (-1.57, 1.57)]
    for xm, ym in local_minima:
        zm = np.sin(xm) * np.cos(ym) + 0.1 * (xm**2 + ym**2)
        ax3.scatter([xm], [ym], [zm], color='orange', s=100, marker='o')

    # Global minimum at origin
    ax3.scatter([0], [0], [0], color='green', s=200, marker='*', label='Global minimum')

    ax3.set_xlabel('Weight 1')
    ax3.set_ylabel('Weight 2')
    ax3.set_zlabel('Loss')
    ax3.view_init(elev=20, azim=45)

    # Plot 4: Contour plot of convex surface
    contour1 = ax4.contour(X, Y, Z1, levels=20, cmap='viridis')
    ax4.clabel(contour1, inline=True, fontsize=8)
    ax4.plot(gd_path_x, gd_path_y, 'r-o', linewidth=2, markersize=4, label='GD path')
    ax4.scatter([0], [0], color='green', s=200, marker='*', zorder=5)
    ax4.set_title('Convex Loss (Top View)', fontweight='bold')
    ax4.set_xlabel('Weight 1')
    ax4.set_ylabel('Weight 2')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Learning rate comparison
    learning_rates = [0.01, 0.1, 0.5]
    colors = [ML_BLUE, ML_GREEN, ML_ORANGE]

    for lr, color in zip(learning_rates, colors):
        path_x = [2.5]
        path_y = [2.5]
        for i in range(30):
            grad_x = 2 * path_x[-1]
            grad_y = 2 * path_y[-1]
            new_x = path_x[-1] - lr * grad_x
            new_y = path_y[-1] - lr * grad_y
            if abs(new_x) > 5 or abs(new_y) > 5:  # Diverged
                break
            path_x.append(new_x)
            path_y.append(new_y)

        ax5.plot(path_x, path_y, '-o', linewidth=2, markersize=3,
                color=color, label=f'LR = {lr}', alpha=0.7)

    # Draw contours
    contour5 = ax5.contour(X, Y, Z1, levels=15, cmap='gray', alpha=0.3)
    ax5.scatter([0], [0], color='green', s=200, marker='*', zorder=5)
    ax5.set_title('Effect of Learning Rate', fontweight='bold')
    ax5.set_xlabel('Weight 1')
    ax5.set_ylabel('Weight 2')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-3, 3)
    ax5.set_ylim(-3, 3)

    # Plot 6: Loss over iterations
    iterations = list(range(len(gd_path_z)))
    ax6.plot(iterations, gd_path_z, 'o-', linewidth=2, markersize=5, color=ML_BLUE)
    ax6.set_title('Loss Decrease Over Iterations', fontweight='bold')
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Loss')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Global minimum')
    ax6.legend()

    # Add main title
    fig.suptitle('Gradient Descent: Navigating the Loss Landscape',
                fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/gradient_landscape_3d.pdf', dpi=300, bbox_inches='tight')
    print("Saved: gradient_landscape_3d.pdf")

def create_optimization_comparison():
    """Compare different optimization algorithms"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Create loss landscape
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = (X - 1)**2 + 2 * (Y - 0.5)**2  # Elongated bowl

    # Starting point
    start_x, start_y = -2.5, 2.0

    # 1. Vanilla Gradient Descent
    ax = axes[0, 0]
    contour = ax.contour(X, Y, Z, levels=20, cmap='gray', alpha=0.3)

    path_x = [start_x]
    path_y = [start_y]
    lr = 0.05
    for i in range(50):
        grad_x = 2 * (path_x[-1] - 1)
        grad_y = 4 * (path_y[-1] - 0.5)
        path_x.append(path_x[-1] - lr * grad_x)
        path_y.append(path_y[-1] - lr * grad_y)

    ax.plot(path_x, path_y, 'o-', linewidth=2, markersize=3, color=ML_BLUE, label='SGD')
    ax.scatter([1], [0.5], color='green', s=200, marker='*', zorder=5)
    ax.set_title('Vanilla SGD\n(Slow, oscillates)', fontweight='bold')
    ax.legend()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    # 2. SGD with Momentum
    ax = axes[0, 1]
    contour = ax.contour(X, Y, Z, levels=20, cmap='gray', alpha=0.3)

    path_x = [start_x]
    path_y = [start_y]
    velocity_x, velocity_y = 0, 0
    lr = 0.05
    momentum = 0.9
    for i in range(50):
        grad_x = 2 * (path_x[-1] - 1)
        grad_y = 4 * (path_y[-1] - 0.5)
        velocity_x = momentum * velocity_x - lr * grad_x
        velocity_y = momentum * velocity_y - lr * grad_y
        path_x.append(path_x[-1] + velocity_x)
        path_y.append(path_y[-1] + velocity_y)

    ax.plot(path_x, path_y, 'o-', linewidth=2, markersize=3, color=ML_GREEN, label='Momentum')
    ax.scatter([1], [0.5], color='green', s=200, marker='*', zorder=5)
    ax.set_title('SGD + Momentum\n(Faster, smoother)', fontweight='bold')
    ax.legend()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    # 3. RMSprop
    ax = axes[1, 0]
    contour = ax.contour(X, Y, Z, levels=20, cmap='gray', alpha=0.3)

    path_x = [start_x]
    path_y = [start_y]
    squared_grad_x, squared_grad_y = 0, 0
    lr = 0.1
    decay = 0.9
    epsilon = 1e-8
    for i in range(50):
        grad_x = 2 * (path_x[-1] - 1)
        grad_y = 4 * (path_y[-1] - 0.5)
        squared_grad_x = decay * squared_grad_x + (1 - decay) * grad_x**2
        squared_grad_y = decay * squared_grad_y + (1 - decay) * grad_y**2
        path_x.append(path_x[-1] - lr * grad_x / (np.sqrt(squared_grad_x) + epsilon))
        path_y.append(path_y[-1] - lr * grad_y / (np.sqrt(squared_grad_y) + epsilon))

    ax.plot(path_x, path_y, 'o-', linewidth=2, markersize=3, color=ML_ORANGE, label='RMSprop')
    ax.scatter([1], [0.5], color='green', s=200, marker='*', zorder=5)
    ax.set_title('RMSprop\n(Adaptive learning rates)', fontweight='bold')
    ax.legend()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    # 4. Adam
    ax = axes[1, 1]
    contour = ax.contour(X, Y, Z, levels=20, cmap='gray', alpha=0.3)

    path_x = [start_x]
    path_y = [start_y]
    m_x, m_y = 0, 0  # First moment
    v_x, v_y = 0, 0  # Second moment
    lr = 0.1
    beta1, beta2 = 0.9, 0.999
    epsilon = 1e-8
    for t in range(1, 51):
        grad_x = 2 * (path_x[-1] - 1)
        grad_y = 4 * (path_y[-1] - 0.5)

        m_x = beta1 * m_x + (1 - beta1) * grad_x
        m_y = beta1 * m_y + (1 - beta1) * grad_y
        v_x = beta2 * v_x + (1 - beta2) * grad_x**2
        v_y = beta2 * v_y + (1 - beta2) * grad_y**2

        m_x_hat = m_x / (1 - beta1**t)
        m_y_hat = m_y / (1 - beta1**t)
        v_x_hat = v_x / (1 - beta2**t)
        v_y_hat = v_y / (1 - beta2**t)

        path_x.append(path_x[-1] - lr * m_x_hat / (np.sqrt(v_x_hat) + epsilon))
        path_y.append(path_y[-1] - lr * m_y_hat / (np.sqrt(v_y_hat) + epsilon))

    ax.plot(path_x, path_y, 'o-', linewidth=2, markersize=3, color=ML_RED, label='Adam')
    ax.scatter([1], [0.5], color='green', s=200, marker='*', zorder=5)
    ax.set_title('Adam\n(Best of both worlds)', fontweight='bold')
    ax.legend()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    fig.suptitle('Optimization Algorithm Comparison', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/optimizer_paths_comparison.pdf', dpi=300, bbox_inches='tight')
    print("Saved: optimizer_paths_comparison.pdf")

if __name__ == "__main__":
    print("Generating gradient landscape visualizations...")

    create_3d_loss_landscape()
    create_optimization_comparison()

    print("\nAll gradient landscape visualizations generated successfully!")
    print("Files created:")
    print("  - gradient_landscape_3d.pdf")
    print("  - optimizer_paths_comparison.pdf")