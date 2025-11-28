"""
Generate additional figures for expanded Neural Networks Primer presentation
15 new visualizations for the 45-slide version
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

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
COLOR_WARNING = '#FFD93D'  # Yellow - warning
COLOR_ERROR = '#FF6B6B'    # Red - error

def create_historical_timeline():
    """Create comprehensive timeline from 1943 to 2023"""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')

    # Timeline data
    events = [
        (1943, "McCulloch-Pitts", "First neural model", "bottom"),
        (1957, "Perceptron", "Learning machine", "top"),
        (1969, "XOR Crisis", "AI Winter begins", "bottom"),
        (1986, "Backpropagation", "Multi-layer learning", "top"),
        (1987, "NetTalk", "Practical success", "bottom"),
        (1989, "Universal Approx", "Mathematical proof", "top"),
        (1998, "LeNet", "CNN for digits", "bottom"),
        (2006, "Deep Belief Nets", "Deep learning revival", "top"),
        (2012, "AlexNet", "ImageNet breakthrough", "bottom"),
        (2015, "ResNet", "152 layers", "top"),
        (2017, "Transformer", "Attention revolution", "bottom"),
        (2018, "BERT/GPT", "Language understanding", "top"),
        (2020, "GPT-3", "175B parameters", "bottom"),
        (2023, "GPT-4", "Multimodal AI", "top"),
    ]

    # Draw main timeline
    ax.plot([1940, 2025], [0, 0], 'k-', linewidth=3)

    # Plot events
    for year, name, desc, pos in events:
        y = 0.5 if pos == "top" else -0.5

        # Event marker
        ax.scatter(year, 0, s=150, c=COLOR_CURRENT if year in [1957, 1986, 2012, 2017] else COLOR_ACCENT,
                  zorder=5, edgecolor='black', linewidth=2)

        # Event line
        ax.plot([year, year], [0, y*0.8], 'k-', linewidth=1, alpha=0.5)

        # Event text
        ax.text(year, y, f"{name}\n({year})", ha='center',
               va='bottom' if pos == "top" else 'top',
               fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
               facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN, linewidth=1))

        # Description
        ax.text(year, y*1.2, desc, ha='center',
               va='bottom' if pos == "top" else 'top',
               fontsize=7, style='italic', color=COLOR_ACCENT)

    # Era labels
    ax.axvspan(1940, 1970, alpha=0.1, color=COLOR_CONTEXT, label='Foundations')
    ax.axvspan(1970, 1985, alpha=0.1, color=COLOR_WARNING, label='First Winter')
    ax.axvspan(1985, 2000, alpha=0.1, color=COLOR_PREDICT, label='Revival')
    ax.axvspan(2000, 2012, alpha=0.1, color=COLOR_WARNING, label='Second Winter')
    ax.axvspan(2012, 2025, alpha=0.1, color=COLOR_CURRENT, label='Deep Learning Era')

    ax.set_xlim(1940, 2025)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_title('Neural Networks: 80 Years of Evolution', fontsize=16, fontweight='bold', color=COLOR_MAIN)
    ax.axis('off')
    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig('../figures/historical_timeline.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_perceptron_hardware():
    """Visualize the Mark I Perceptron machine"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('white')

    # Left: Conceptual diagram
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')

    # Photocells (input layer)
    for i in range(5):
        for j in range(5):
            circle = Circle((1 + i*0.5, 2 + j*0.5), 0.15,
                          color=COLOR_CONTEXT, edgecolor='black')
            ax1.add_patch(circle)

    ax1.text(1.5, 0.5, '400 Photocells\n(20×20 retina)', ha='center', fontsize=10)

    # Potentiometers (weights)
    for i in range(3):
        rect = Rectangle((5, 3 + i*1.5), 2, 0.8,
                        facecolor=COLOR_ACCENT, edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(6, 3.4 + i*1.5, f'Motor {i+1}', ha='center', fontsize=9)

    ax1.text(6, 1, '512 Motor-driven\nPotentiometers', ha='center', fontsize=10)

    # Output
    circle = Circle((8.5, 5), 0.5, color=COLOR_PREDICT, edgecolor='black', linewidth=2)
    ax1.add_patch(circle)
    ax1.text(8.5, 5, 'OUT', ha='center', fontsize=10, fontweight='bold')

    # Connections
    for i in range(5):
        ax1.arrow(3, 2.5 + i*0.5, 2, 0.5 + i*0.2, head_width=0.1,
                 head_length=0.1, fc=COLOR_LIGHT, ec=COLOR_MAIN)

    for i in range(3):
        ax1.arrow(7, 3.4 + i*1.5, 1, 1.6 - i*0.5, head_width=0.1,
                 head_length=0.1, fc=COLOR_LIGHT, ec=COLOR_MAIN)

    ax1.set_title('Mark I Perceptron Architecture', fontsize=12, fontweight='bold')

    # Right: Learning process
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    # Learning steps
    steps = [
        "1. Present pattern to photocells",
        "2. Calculate weighted sum",
        "3. Compare to desired output",
        "4. Motors adjust potentiometers",
        "5. Repeat until converged"
    ]

    for i, step in enumerate(steps):
        y_pos = 8 - i*1.5
        # Step box
        rect = FancyBboxPatch((0.5, y_pos-0.4), 9, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor=COLOR_LIGHT if i % 2 == 0 else 'white',
                             edgecolor=COLOR_MAIN, linewidth=1.5)
        ax2.add_patch(rect)
        ax2.text(5, y_pos, step, ha='center', va='center', fontsize=10)

        if i < len(steps) - 1:
            ax2.arrow(5, y_pos-0.5, 0, -0.5, head_width=0.2,
                     head_length=0.1, fc=COLOR_ACCENT, ec=COLOR_ACCENT)

    ax2.set_title('Physical Learning Process', fontsize=12, fontweight='bold')

    plt.suptitle('The Mark I Perceptron (1957): A Physical Learning Machine',
                fontsize=14, fontweight='bold', color=COLOR_MAIN)
    plt.tight_layout()
    plt.savefig('../figures/perceptron_hardware.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_xor_solution_steps():
    """Show step-by-step XOR solution with hidden layer"""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 4, figure=fig)
    fig.patch.set_facecolor('white')

    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    # Step 1: Problem
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(X[y==0, 0], X[y==0, 1], s=200, c=COLOR_CURRENT, marker='o', edgecolor='black', linewidth=2)
    ax1.scatter(X[y==1, 0], X[y==1, 1], s=200, c=COLOR_CONTEXT, marker='s', edgecolor='black', linewidth=2)
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_title('Step 1: XOR Problem', fontsize=10, fontweight='bold')
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.grid(True, alpha=0.3)

    # Step 2: Hidden neuron 1
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(X[y==0, 0], X[y==0, 1], s=200, c=COLOR_CURRENT, marker='o', edgecolor='black', linewidth=2, alpha=0.3)
    ax2.scatter(X[y==1, 0], X[y==1, 1], s=200, c=COLOR_CONTEXT, marker='s', edgecolor='black', linewidth=2, alpha=0.3)
    ax2.plot([0.5, 0.5], [-0.5, 1.5], '--', color=COLOR_PREDICT, linewidth=2)
    ax2.fill_between([0.5, 1.5], [-0.5, -0.5], [1.5, 1.5], alpha=0.2, color=COLOR_PREDICT)
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_title('Step 2: Hidden 1\n(x₁ > 0.5)', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Step 3: Hidden neuron 2
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(X[y==0, 0], X[y==0, 1], s=200, c=COLOR_CURRENT, marker='o', edgecolor='black', linewidth=2, alpha=0.3)
    ax3.scatter(X[y==1, 0], X[y==1, 1], s=200, c=COLOR_CONTEXT, marker='s', edgecolor='black', linewidth=2, alpha=0.3)
    ax3.plot([-0.5, 1.5], [0.5, 0.5], '--', color=COLOR_WARNING, linewidth=2)
    ax3.fill_between([-0.5, 1.5], [0.5, 0.5], [1.5, 1.5], alpha=0.2, color=COLOR_WARNING)
    ax3.set_xlim(-0.5, 1.5)
    ax3.set_ylim(-0.5, 1.5)
    ax3.set_title('Step 3: Hidden 2\n(x₂ > 0.5)', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Step 4: Combined solution
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.scatter(X[y==0, 0], X[y==0, 1], s=200, c=COLOR_CURRENT, marker='o', edgecolor='black', linewidth=2)
    ax4.scatter(X[y==1, 0], X[y==1, 1], s=200, c=COLOR_CONTEXT, marker='s', edgecolor='black', linewidth=2)
    ax4.plot([0.5, 0.5], [-0.5, 1.5], '-', color=COLOR_PREDICT, linewidth=2, label='Hidden 1')
    ax4.plot([-0.5, 1.5], [0.5, 0.5], '-', color=COLOR_WARNING, linewidth=2, label='Hidden 2')
    ax4.set_xlim(-0.5, 1.5)
    ax4.set_ylim(-0.5, 1.5)
    ax4.set_title('Step 4: Solution\n(H1 XOR H2)', fontsize=10, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Network diagram
    ax5 = fig.add_subplot(gs[1, :])
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 5)
    ax5.axis('off')

    # Input nodes
    ax5.add_patch(Circle((1, 3), 0.3, color=COLOR_LIGHT, edgecolor='black', linewidth=2))
    ax5.add_patch(Circle((1, 1.5), 0.3, color=COLOR_LIGHT, edgecolor='black', linewidth=2))
    ax5.text(1, 3, 'x₁', ha='center', fontsize=10, fontweight='bold')
    ax5.text(1, 1.5, 'x₂', ha='center', fontsize=10, fontweight='bold')

    # Hidden nodes
    ax5.add_patch(Circle((5, 3.5), 0.3, color=COLOR_PREDICT, edgecolor='black', linewidth=2))
    ax5.add_patch(Circle((5, 1), 0.3, color=COLOR_WARNING, edgecolor='black', linewidth=2))
    ax5.text(5, 3.5, 'H₁', ha='center', fontsize=10, fontweight='bold')
    ax5.text(5, 1, 'H₂', ha='center', fontsize=10, fontweight='bold')

    # Output node
    ax5.add_patch(Circle((9, 2.25), 0.3, color=COLOR_CONTEXT, edgecolor='black', linewidth=2))
    ax5.text(9, 2.25, 'OUT', ha='center', fontsize=10, fontweight='bold')

    # Connections with weights
    connections = [
        ((1.3, 3), (4.7, 3.5), '1'),
        ((1.3, 3), (4.7, 1), '-1'),
        ((1.3, 1.5), (4.7, 3.5), '-1'),
        ((1.3, 1.5), (4.7, 1), '1'),
        ((5.3, 3.5), (8.7, 2.25), '1'),
        ((5.3, 1), (8.7, 2.25), '1'),
    ]

    for start, end, weight in connections:
        ax5.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=1.5, alpha=0.5)
        mid_x, mid_y = (start[0] + end[0])/2, (start[1] + end[1])/2
        ax5.text(mid_x, mid_y + 0.2, weight, ha='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor=COLOR_ACCENT))

    ax5.set_title('Network Architecture: 2 → 2 → 1', fontsize=12, fontweight='bold')

    plt.suptitle('Solving XOR with a Hidden Layer', fontsize=14, fontweight='bold', color=COLOR_MAIN)
    plt.tight_layout()
    plt.savefig('../figures/xor_solution_steps.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_weight_update_animation():
    """Visualize weight updates during learning"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.patch.set_facecolor('white')

    # Epochs to show
    epochs = [0, 1, 5, 10, 50, 100]

    for idx, (ax, epoch) in enumerate(zip(axes.flat, epochs)):
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')

        # Generate sample weights evolution
        np.random.seed(42)
        if epoch == 0:
            weights = np.random.randn(10, 2) * 2
        else:
            target = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0],
                              [2, 0], [0, 2], [-2, 0], [0, -2], [1.5, 1.5]])
            noise = np.random.randn(10, 2) * (0.5 / np.sqrt(epoch + 1))
            weights = target + noise

        # Plot weight vectors
        for w in weights:
            ax.arrow(0, 0, w[0], w[1], head_width=0.1, head_length=0.1,
                    fc=COLOR_CONTEXT, ec=COLOR_MAIN, alpha=0.6, linewidth=1.5)

        # Add grid
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

        # Title with loss
        if epoch == 0:
            loss = 2.5
        else:
            loss = 2.5 * np.exp(-epoch/20) + 0.1

        ax.set_title(f'Epoch {epoch}\nLoss: {loss:.3f}', fontsize=10)
        ax.set_xlabel('w₁')
        ax.set_ylabel('w₂')

    plt.suptitle('Weight Evolution During Training', fontsize=14, fontweight='bold', color=COLOR_MAIN)
    plt.tight_layout()
    plt.savefig('../figures/weight_update_animation.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_vanishing_gradient_flow():
    """Visualize vanishing gradient problem"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('white')

    # Left: Gradient magnitude through layers
    ax1.set_xlim(0, 11)
    ax1.set_ylim(0, 1.2)

    layers = np.arange(1, 11)

    # Different activation functions
    sigmoid_grad = 0.25 ** (layers - 1)  # Max derivative of sigmoid is 0.25
    tanh_grad = 1.0 * 0.65 ** (layers - 1)  # Tanh slightly better
    relu_grad = np.ones_like(layers)  # ReLU preserves gradient

    ax1.plot(layers, sigmoid_grad, 'o-', label='Sigmoid', color=COLOR_ERROR, linewidth=2, markersize=8)
    ax1.plot(layers, tanh_grad, 's-', label='Tanh', color=COLOR_WARNING, linewidth=2, markersize=8)
    ax1.plot(layers, relu_grad, '^-', label='ReLU', color=COLOR_PREDICT, linewidth=2, markersize=8)

    ax1.set_xlabel('Layer Depth', fontsize=11)
    ax1.set_ylabel('Gradient Magnitude', fontsize=11)
    ax1.set_title('Gradient Flow Through Layers', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Add danger zone
    ax1.axhspan(0, 0.01, alpha=0.2, color='red', label='Vanishing Zone')
    ax1.text(8, 0.005, 'Vanishing\nZone', fontsize=9, color='red', fontweight='bold')

    # Right: Network visualization
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    # Draw network with gradient flow
    n_layers = 6
    layer_x = np.linspace(1, 9, n_layers)

    for i, x in enumerate(layer_x):
        # Layer representation
        for y in np.linspace(3, 7, 3):
            # Gradient magnitude affects color intensity
            if i == 0:
                color = COLOR_PREDICT
                size = 150
            else:
                gradient_mag = 0.25 ** i
                alpha = max(0.1, gradient_mag)
                color = COLOR_CURRENT
                size = 150 * (0.5 + 0.5 * gradient_mag)

            circle = Circle((x, y), 0.2, color=color, alpha=alpha if i > 0 else 1,
                          edgecolor='black', linewidth=1)
            ax2.add_patch(circle)

        # Layer label
        ax2.text(x, 2, f'Layer {i+1}', ha='center', fontsize=9)

        # Gradient magnitude label
        if i > 0:
            grad_mag = 0.25 ** i
            ax2.text(x, 8, f'∇: {grad_mag:.3f}', ha='center', fontsize=8,
                    color=COLOR_ERROR if grad_mag < 0.01 else COLOR_MAIN)

    # Draw connections
    for i in range(n_layers - 1):
        x1, x2 = layer_x[i], layer_x[i+1]
        for y in np.linspace(3, 7, 3):
            alpha = max(0.1, 0.25 ** i)
            ax2.plot([x1 + 0.2, x2 - 0.2], [y, y], 'k-', alpha=alpha, linewidth=1)

    ax2.set_title('Gradient Signal Decay in Deep Networks', fontsize=12, fontweight='bold')

    plt.suptitle('The Vanishing Gradient Problem', fontsize=14, fontweight='bold', color=COLOR_MAIN)
    plt.tight_layout()
    plt.savefig('../figures/vanishing_gradient_flow.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_hierarchy_progression():
    """Show how features build from simple to complex"""
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    fig.patch.set_facecolor('white')

    titles = ['Input', 'Layer 1: Edges', 'Layer 2: Parts', 'Layer 3: Objects', 'Output']

    for idx, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')

        if idx == 0:  # Input - digit 8
            # Draw pixelated 8
            pixels = np.zeros((10, 10))
            # Top circle
            pixels[2:4, 3:7] = 1
            pixels[1, 4:6] = 1
            # Middle
            pixels[4:6, 3:7] = 1
            # Bottom circle
            pixels[6:8, 3:7] = 1
            pixels[8, 4:6] = 1
            # Sides
            pixels[2:4, 3] = pixels[2:4, 6] = 1
            pixels[6:8, 3] = pixels[6:8, 6] = 1

            ax.imshow(pixels, cmap='gray', origin='lower')

        elif idx == 1:  # Edges
            # Horizontal edges
            ax.plot([2, 8], [7, 7], 'k-', linewidth=3)
            ax.plot([2, 8], [5, 5], 'k-', linewidth=3)
            ax.plot([2, 8], [3, 3], 'k-', linewidth=3)
            # Vertical edges
            ax.plot([3, 3], [2, 8], 'k-', linewidth=2)
            ax.plot([7, 7], [2, 8], 'k-', linewidth=2)

        elif idx == 2:  # Parts
            # Top circle
            circle1 = plt.Circle((5, 7.5), 1.5, fill=False, linewidth=3, color='black')
            ax.add_patch(circle1)
            # Bottom circle
            circle2 = plt.Circle((5, 2.5), 1.5, fill=False, linewidth=3, color='black')
            ax.add_patch(circle2)
            # Connection
            ax.plot([3.5, 3.5], [4, 6], 'k-', linewidth=3)
            ax.plot([6.5, 6.5], [4, 6], 'k-', linewidth=3)

        elif idx == 3:  # Complete object
            # Draw complete 8
            ax.text(5, 5, '8', fontsize=80, ha='center', va='center', fontweight='bold')

        elif idx == 4:  # Output
            # Classification
            ax.text(5, 5, 'Class: 8\nConfidence: 0.97', ha='center', va='center',
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.5",
                   facecolor=COLOR_PREDICT, edgecolor=COLOR_MAIN, linewidth=2))

        ax.set_title(title, fontsize=10, fontweight='bold')

    # Add arrows between stages
    for i in range(4):
        fig.text(0.16 + i*0.175, 0.5, '→', fontsize=20, ha='center', color=COLOR_ACCENT)

    plt.suptitle('Hierarchical Feature Learning in Neural Networks',
                fontsize=14, fontweight='bold', color=COLOR_MAIN)
    plt.tight_layout()
    plt.savefig('../figures/feature_hierarchy_progression.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_training_dynamics_dashboard():
    """Multi-panel training metrics dashboard"""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    fig.patch.set_facecolor('white')

    epochs = np.arange(0, 100)

    # Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    train_loss = 2.5 * np.exp(-epochs/20) + 0.1 + np.random.randn(100) * 0.02
    val_loss = 2.5 * np.exp(-epochs/25) + 0.15 + np.random.randn(100) * 0.03

    ax1.plot(epochs, train_loss, label='Training', color=COLOR_CONTEXT, linewidth=2)
    ax1.plot(epochs, val_loss, label='Validation', color=COLOR_CURRENT, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2 = fig.add_subplot(gs[0, 1])
    train_acc = 1 - np.exp(-epochs/15) + np.random.randn(100) * 0.01
    train_acc = np.clip(train_acc, 0, 1)
    val_acc = 1 - np.exp(-epochs/20) - 0.05 + np.random.randn(100) * 0.015
    val_acc = np.clip(val_acc, 0, 0.95)

    ax2.plot(epochs, train_acc * 100, label='Training', color=COLOR_CONTEXT, linewidth=2)
    ax2.plot(epochs, val_acc * 100, label='Validation', color=COLOR_CURRENT, linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Learning rate schedule
    ax3 = fig.add_subplot(gs[0, 2])
    lr = 0.01 * np.exp(-epochs/30)
    ax3.plot(epochs, lr, color=COLOR_PREDICT, linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule', fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # Gradient norm
    ax4 = fig.add_subplot(gs[1, 0])
    grad_norm = 10 * np.exp(-epochs/10) + np.random.randn(100) * 0.5
    grad_norm = np.abs(grad_norm)
    ax4.plot(epochs, grad_norm, color=COLOR_WARNING, linewidth=2)
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax4.text(50, 1.5, 'Healthy Range', fontsize=9, color='red')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Gradient Norm')
    ax4.set_title('Gradient Magnitude', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Weight distribution
    ax5 = fig.add_subplot(gs[1, 1])
    for epoch in [0, 20, 50, 99]:
        weights = np.random.normal(0, 1 - epoch/200, 1000)
        ax5.hist(weights, bins=30, alpha=0.5, label=f'Epoch {epoch}', density=True)
    ax5.set_xlabel('Weight Value')
    ax5.set_ylabel('Density')
    ax5.set_title('Weight Distribution Evolution', fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Confusion matrix (final)
    ax6 = fig.add_subplot(gs[1, 2])
    confusion = np.random.rand(5, 5) * 0.1
    np.fill_diagonal(confusion, 0.85 + np.random.rand(5) * 0.1)
    confusion = confusion / confusion.sum(axis=1, keepdims=True)

    im = ax6.imshow(confusion, cmap='Blues')
    ax6.set_xlabel('Predicted')
    ax6.set_ylabel('Actual')
    ax6.set_title('Confusion Matrix (Final)', fontweight='bold')
    ax6.set_xticks(range(5))
    ax6.set_yticks(range(5))
    ax6.set_xticklabels(['A', 'B', 'C', 'D', 'E'])
    ax6.set_yticklabels(['A', 'B', 'C', 'D', 'E'])

    # Add text annotations
    for i in range(5):
        for j in range(5):
            text = ax6.text(j, i, f'{confusion[i, j]:.2f}',
                          ha="center", va="center", color="white" if confusion[i, j] > 0.5 else "black")

    plt.suptitle('Training Dynamics Dashboard', fontsize=16, fontweight='bold', color=COLOR_MAIN)
    plt.tight_layout()
    plt.savefig('../figures/training_dynamics_dashboard.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_depth_width_comparison():
    """Compare deep vs wide networks"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('white')

    # Left: Performance comparison
    params = np.logspace(4, 8, 50)

    # Wide network (more neurons per layer)
    wide_acc = 0.95 * (1 - np.exp(-params/1e6))
    # Deep network (more layers)
    deep_acc = 0.98 * (1 - np.exp(-params/5e5))

    ax1.semilogx(params, wide_acc * 100, label='Wide (Few Layers)',
                color=COLOR_WARNING, linewidth=2)
    ax1.semilogx(params, deep_acc * 100, label='Deep (Many Layers)',
                color=COLOR_PREDICT, linewidth=2)

    ax1.set_xlabel('Number of Parameters', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Deep vs Wide: Performance', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add annotations
    ax1.axvline(x=1e6, color='gray', linestyle='--', alpha=0.5)
    ax1.text(1e6, 70, '1M params', rotation=90, fontsize=9, color='gray')

    # Right: Architecture visualization
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    # Wide network
    wide_x = [1, 3, 5]
    for i, x in enumerate(wide_x):
        for y in np.linspace(1, 5, 8):
            circle = Circle((x, y), 0.15, color=COLOR_WARNING, alpha=0.7,
                          edgecolor='black', linewidth=1)
            ax2.add_patch(circle)

    ax2.text(3, 0.2, 'Wide Network\n(3 layers × 8 neurons)', ha='center', fontsize=10, fontweight='bold')

    # Deep network
    deep_x = np.linspace(6, 9.5, 8)
    for i, x in enumerate(deep_x):
        for y in np.linspace(7, 9, 3):
            circle = Circle((x, y), 0.15, color=COLOR_PREDICT, alpha=0.7,
                          edgecolor='black', linewidth=1)
            ax2.add_patch(circle)

    ax2.text(7.75, 6.2, 'Deep Network\n(8 layers × 3 neurons)', ha='center', fontsize=10, fontweight='bold')

    ax2.set_title('Architecture Comparison', fontsize=12, fontweight='bold')

    plt.suptitle('Depth vs Width: Why Deep Networks Win', fontsize=14, fontweight='bold', color=COLOR_MAIN)
    plt.tight_layout()
    plt.savefig('../figures/depth_width_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_data_scaling_laws():
    """Visualize data scaling laws"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('white')

    # Left: Scaling laws
    data_size = np.logspace(3, 9, 100)

    # Different model sizes
    small_perf = 0.6 * (1 - np.exp(-data_size/1e5))
    medium_perf = 0.8 * (1 - np.exp(-data_size/1e6))
    large_perf = 0.95 * (1 - np.exp(-data_size/1e7))

    ax1.loglog(data_size, small_perf, label='Small Model (1M)', color=COLOR_ACCENT, linewidth=2)
    ax1.loglog(data_size, medium_perf, label='Medium Model (100M)', color=COLOR_WARNING, linewidth=2)
    ax1.loglog(data_size, large_perf, label='Large Model (10B)', color=COLOR_PREDICT, linewidth=2)

    ax1.set_xlabel('Training Tokens', fontsize=11)
    ax1.set_ylabel('Performance', fontsize=11)
    ax1.set_title('Neural Scaling Laws', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')

    # Add power law line
    power_law = data_size ** 0.08 / 1000
    ax1.loglog(data_size, power_law, 'k--', alpha=0.5, label='Power Law')

    # Right: Compute-optimal frontier
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    model_size = np.logspace(6, 12, 50)
    optimal_data = model_size * 20  # Chinchilla ratio

    # Plot regions
    ax2.fill_between(model_size, model_size * 5, model_size * 100,
                     alpha=0.2, color=COLOR_WARNING, label='Suboptimal')
    ax2.plot(model_size, optimal_data, 'k-', linewidth=3, label='Compute-Optimal')

    # Add real models
    models = [
        (175e9, 300e9, 'GPT-3'),
        (1.3e9, 33e9, 'GPT-2'),
        (340e6, 250e9, 'BERT'),
        (70e9, 1.4e12, 'Chinchilla'),
    ]

    for params, tokens, name in models:
        ax2.scatter(params, tokens, s=100, zorder=5)
        ax2.text(params, tokens * 1.5, name, fontsize=9, ha='center')

    ax2.set_xlabel('Model Parameters', fontsize=11)
    ax2.set_ylabel('Training Tokens', fontsize=11)
    ax2.set_title('Compute-Optimal Training', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.suptitle('Data Scaling Laws (Chinchilla, 2022)', fontsize=14, fontweight='bold', color=COLOR_MAIN)
    plt.tight_layout()
    plt.savefig('../figures/data_scaling_laws.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_optimizer_comparison():
    """Compare different optimizers"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor('white')

    iterations = np.arange(0, 200)

    # Generate loss curves for different optimizers
    np.random.seed(42)

    # SGD - slow, steady
    sgd_loss = 2.0 * np.exp(-iterations/80) + 0.1 + np.cumsum(np.random.randn(200) * 0.01)
    sgd_loss = np.maximum(sgd_loss, 0.1)

    # Momentum - faster convergence
    momentum_loss = 2.0 * np.exp(-iterations/50) + 0.1 + np.cumsum(np.random.randn(200) * 0.008)
    momentum_loss = np.maximum(momentum_loss, 0.1)

    # Adam - adaptive, fast
    adam_loss = 2.0 * np.exp(-iterations/30) + 0.1 + np.cumsum(np.random.randn(200) * 0.005)
    adam_loss = np.maximum(adam_loss, 0.1)

    # RMSprop
    rmsprop_loss = 2.0 * np.exp(-iterations/40) + 0.1 + np.cumsum(np.random.randn(200) * 0.007)
    rmsprop_loss = np.maximum(rmsprop_loss, 0.1)

    # Plot all together
    ax_all = axes[0, 0]
    ax_all.plot(iterations, sgd_loss, label='SGD', color=COLOR_ACCENT, linewidth=2)
    ax_all.plot(iterations, momentum_loss, label='Momentum', color=COLOR_WARNING, linewidth=2)
    ax_all.plot(iterations, adam_loss, label='Adam', color=COLOR_PREDICT, linewidth=2)
    ax_all.plot(iterations, rmsprop_loss, label='RMSprop', color=COLOR_CONTEXT, linewidth=2)

    ax_all.set_xlabel('Iteration')
    ax_all.set_ylabel('Loss')
    ax_all.set_title('Optimizer Comparison', fontweight='bold')
    ax_all.legend()
    ax_all.grid(True, alpha=0.3)

    # Individual characteristics
    optimizers = [
        ('SGD', sgd_loss, COLOR_ACCENT),
        ('Momentum', momentum_loss, COLOR_WARNING),
        ('Adam', adam_loss, COLOR_PREDICT)
    ]

    for idx, (name, loss, color) in enumerate(optimizers):
        ax = axes.flat[idx + 1]
        ax.plot(iterations, loss, color=color, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(name, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add characteristics
        if name == 'SGD':
            ax.text(100, 1.5, 'Simple\nConsistent\nSlow', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_LIGHT))
        elif name == 'Momentum':
            ax.text(100, 1.3, 'Accelerates\nOscillates less\nFaster', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_LIGHT))
        elif name == 'Adam':
            ax.text(100, 1.1, 'Adaptive LR\nPer-parameter\nFastest', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_LIGHT))

    plt.suptitle('Modern Optimizers: From SGD to Adam', fontsize=14, fontweight='bold', color=COLOR_MAIN)
    plt.tight_layout()
    plt.savefig('../figures/optimizer_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_emergent_abilities_threshold():
    """Show sudden capability emergence with scale"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('white')

    # Left: Phase transition
    params = np.logspace(8, 12, 100)

    # Different abilities emerge at different scales
    arithmetic = 1 / (1 + np.exp(-(np.log10(params) - 10.5) * 5))
    reasoning = 1 / (1 + np.exp(-(np.log10(params) - 11) * 5))
    coding = 1 / (1 + np.exp(-(np.log10(params) - 10.7) * 5))

    ax1.semilogx(params, arithmetic * 100, label='3-digit Arithmetic', color=COLOR_CONTEXT, linewidth=2)
    ax1.semilogx(params, reasoning * 100, label='Chain-of-Thought', color=COLOR_WARNING, linewidth=2)
    ax1.semilogx(params, coding * 100, label='Code Generation', color=COLOR_PREDICT, linewidth=2)

    ax1.set_xlabel('Model Parameters', fontsize=11)
    ax1.set_ylabel('Task Performance (%)', fontsize=11)
    ax1.set_title('Emergent Abilities: Phase Transitions', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add threshold lines
    ax1.axvline(x=10**10.5, color='gray', linestyle='--', alpha=0.3)
    ax1.axvline(x=10**11, color='gray', linestyle='--', alpha=0.3)

    # Right: Capability map
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    # Model sizes and capabilities
    models = [
        (2, 'Small\n(<1B)', ['Basic QA', 'Simple classification']),
        (5, 'Medium\n(1-10B)', ['Translation', 'Summarization', 'Named entities']),
        (8, 'Large\n(>10B)', ['Arithmetic', 'Code', 'Reasoning', 'Few-shot'])
    ]

    for x, name, capabilities in models:
        # Model circle
        size = x * 100
        circle = Circle((x, 7), x/4, color=COLOR_LIGHT, edgecolor=COLOR_MAIN, linewidth=2)
        ax2.add_patch(circle)
        ax2.text(x, 7, name, ha='center', fontsize=10, fontweight='bold')

        # Capabilities
        for i, cap in enumerate(capabilities):
            y = 4 - i * 0.7
            rect = FancyBboxPatch((x-1.5, y-0.25), 3, 0.5,
                                 boxstyle="round,pad=0.05",
                                 facecolor=COLOR_PREDICT if x > 5 else COLOR_WARNING if x > 2 else COLOR_ACCENT,
                                 alpha=0.5, edgecolor=COLOR_MAIN)
            ax2.add_patch(rect)
            ax2.text(x, y, cap, ha='center', fontsize=8)

    ax2.set_title('Capability Emergence by Scale', fontsize=12, fontweight='bold')

    plt.suptitle('Emergent Abilities in Large Language Models', fontsize=14, fontweight='bold', color=COLOR_MAIN)
    plt.tight_layout()
    plt.savefig('../figures/emergent_abilities_threshold.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_debugging_workflow_diagram():
    """Systematic debugging process flowchart"""
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Debugging steps
    steps = [
        (5, 11, "Loss Not Decreasing?", COLOR_ERROR),
        (2.5, 9.5, "Check Learning Rate", COLOR_WARNING),
        (7.5, 9.5, "Check Data/Labels", COLOR_WARNING),
        (5, 8, "Loss NaN/Inf?", COLOR_ERROR),
        (2.5, 6.5, "Gradient Clipping", COLOR_PREDICT),
        (7.5, 6.5, "Reduce Learning Rate", COLOR_PREDICT),
        (5, 5, "Overfitting?", COLOR_ERROR),
        (2.5, 3.5, "Add Regularization", COLOR_PREDICT),
        (7.5, 3.5, "Data Augmentation", COLOR_PREDICT),
        (5, 2, "Poor Generalization?", COLOR_ERROR),
        (5, 0.5, "Architecture/Hyperparameter Tuning", COLOR_CONTEXT),
    ]

    # Draw boxes and connections
    for x, y, text, color in steps:
        # Box
        width = 3 if 'Check' in text or 'Architecture' in text else 2.5
        rect = FancyBboxPatch((x-width/2, y-0.4), width, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor=color, alpha=0.3,
                             edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows
    connections = [
        ((5, 10.6), (2.5, 9.9)),  # Loss not decreasing → LR
        ((5, 10.6), (7.5, 9.9)),  # Loss not decreasing → Data
        ((2.5, 9.1), (5, 8.4)),   # LR → NaN check
        ((7.5, 9.1), (5, 8.4)),   # Data → NaN check
        ((5, 7.6), (2.5, 6.9)),   # NaN → Clipping
        ((5, 7.6), (7.5, 6.9)),   # NaN → Reduce LR
        ((2.5, 6.1), (5, 5.4)),   # Clipping → Overfitting
        ((7.5, 6.1), (5, 5.4)),   # Reduce LR → Overfitting
        ((5, 4.6), (2.5, 3.9)),   # Overfitting → Regularization
        ((5, 4.6), (7.5, 3.9)),   # Overfitting → Augmentation
        ((2.5, 3.1), (5, 2.4)),   # Regularization → Generalization
        ((7.5, 3.1), (5, 2.4)),   # Augmentation → Generalization
        ((5, 1.6), (5, 0.9)),     # Generalization → Architecture
    ]

    for start, end in connections:
        ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                head_width=0.15, head_length=0.1, fc=COLOR_ACCENT, ec=COLOR_ACCENT, alpha=0.5)

    plt.title('Neural Network Debugging Workflow', fontsize=16, fontweight='bold', color=COLOR_MAIN, pad=20)
    plt.tight_layout()
    plt.savefig('../figures/debugging_workflow_diagram.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all additional figures"""
    print("Generating additional Neural Networks Primer figures...")

    # Generate all new figures
    create_historical_timeline()
    print("- Historical timeline")

    create_perceptron_hardware()
    print("- Perceptron hardware")

    create_xor_solution_steps()
    print("- XOR solution steps")

    create_weight_update_animation()
    print("- Weight update animation")

    create_vanishing_gradient_flow()
    print("- Vanishing gradient flow")

    create_feature_hierarchy_progression()
    print("- Feature hierarchy progression")

    create_training_dynamics_dashboard()
    print("- Training dynamics dashboard")

    create_depth_width_comparison()
    print("- Depth vs width comparison")

    create_data_scaling_laws()
    print("- Data scaling laws")

    create_optimizer_comparison()
    print("- Optimizer comparison")

    create_emergent_abilities_threshold()
    print("- Emergent abilities threshold")

    create_debugging_workflow_diagram()
    print("- Debugging workflow diagram")

    print("\nAll additional figures generated successfully!")

if __name__ == "__main__":
    main()