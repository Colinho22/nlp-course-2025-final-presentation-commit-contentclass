"""
Generate visualization charts for mathematical formulas in the embeddings presentation.
Each formula gets a dedicated chart that helps explain the mathematical concept visually.

Author: Joerg R. Osterrieder
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.special import gamma
from scipy.stats import multivariate_normal
import os

# Set style and create output directory
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define consistent color scheme (matching visual_theme.tex)
COLORS = {
    'embedding': '#4682B4',  # Steel Blue (RGB: 70, 130, 180)
    'training': '#FF6B6B',   # Coral Red (RGB: 255, 107, 107)
    'attention': '#4ECDC4',  # Teal (RGB: 78, 205, 196)
    'dimension': '#95E77E',  # Light Green (RGB: 149, 231, 126)
    'architecture': '#FFD93D', # Gold (RGB: 255, 217, 61)
}

OUTPUT_DIR = '../figures/embeddings/formulas'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_figure(name, dpi=150):
    """Save figure with consistent settings"""
    plt.savefig(f'{OUTPUT_DIR}/{name}.pdf', format='pdf', bbox_inches='tight', dpi=dpi)
    plt.close()

# 1. Skip-gram Loss Surface (3D)
def generate_skipgram_loss_surface():
    """3D loss surface showing optimization landscape with gradient descent path"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create loss surface
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Simulate loss surface (combination of convex and non-convex regions)
    Z = 0.5 * (X**2 + Y**2) + 0.5 * np.sin(2*X) * np.cos(2*Y) + 5
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
    
    # Add gradient descent path
    path_steps = 50
    path_x = 2.5 * np.exp(-0.1 * np.arange(path_steps)) * np.cos(0.3 * np.arange(path_steps))
    path_y = 2.5 * np.exp(-0.1 * np.arange(path_steps)) * np.sin(0.3 * np.arange(path_steps))
    path_z = 0.5 * (path_x**2 + path_y**2) + 0.5 * np.sin(2*path_x) * np.cos(2*path_y) + 5
    
    ax.plot(path_x, path_y, path_z, 'r-', linewidth=3, label='Gradient Descent Path')
    ax.scatter(path_x[0], path_y[0], path_z[0], color='red', s=100, label='Start')
    ax.scatter(path_x[-1], path_y[-1], path_z[-1], color='green', s=100, label='Convergence')
    
    # Labels and formatting
    ax.set_xlabel('Embedding Dimension 1', fontsize=11)
    ax.set_ylabel('Embedding Dimension 2', fontsize=11)
    ax.set_zlabel('Loss J(θ)', fontsize=11)
    ax.set_title('Skip-gram Objective Function: Loss Surface', fontsize=13, fontweight='bold')
    
    # Add formula annotation
    formula = r'$J(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{j} \log p(w_{t+j} | w_t)$'
    ax.text2D(0.05, 0.95, formula, transform=ax.transAxes, fontsize=11, 
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend()
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    save_figure('skipgram_loss_surface')

# 2. Softmax Probability Heatmap
def generate_softmax_probability_heatmap():
    """Heatmap showing probability distributions for different context words"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create vocabulary of 20 words for visualization
    vocab_size = 20
    context_words = 5
    
    # Simulate dot products (before softmax)
    np.random.seed(42)
    raw_scores = np.random.randn(context_words, vocab_size) * 2
    
    # Apply softmax
    exp_scores = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
    softmax_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Plot raw scores
    im1 = ax1.imshow(raw_scores, cmap='RdBu_r', aspect='auto', vmin=-4, vmax=4)
    ax1.set_title('Raw Dot Products: $v\'_{w_O}^T v_{w_I}$', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Output Words', fontsize=11)
    ax1.set_ylabel('Context Words', fontsize=11)
    ax1.set_xticks(range(vocab_size))
    ax1.set_xticklabels([f'w{i}' for i in range(vocab_size)], rotation=45)
    ax1.set_yticks(range(context_words))
    ax1.set_yticklabels([f'context{i}' for i in range(context_words)])
    plt.colorbar(im1, ax=ax1)
    
    # Plot softmax probabilities
    im2 = ax2.imshow(softmax_probs, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.3)
    ax2.set_title('After Softmax: $p(w_O | w_I)$', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Output Words', fontsize=11)
    ax2.set_ylabel('Context Words', fontsize=11)
    ax2.set_xticks(range(vocab_size))
    ax2.set_xticklabels([f'w{i}' for i in range(vocab_size)], rotation=45)
    ax2.set_yticks(range(context_words))
    ax2.set_yticklabels([f'context{i}' for i in range(context_words)])
    plt.colorbar(im2, ax=ax2)
    
    # Add formula
    formula = r'Softmax Probability Distribution'
    fig.suptitle(formula, fontsize=12, y=1.05)
    
    plt.tight_layout()
    save_figure('softmax_probability_heatmap')

# 3. Gradient Vector Field
def generate_gradient_vector_field():
    """Vector field showing gradient flow in embedding space"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create grid
    x = np.linspace(-3, 3, 15)
    y = np.linspace(-3, 3, 15)
    X, Y = np.meshgrid(x, y)
    
    # Define a loss function with interesting gradient field
    # Loss has minimum at origin with some local structure
    U = -Y + 0.5 * np.sin(X)  # Gradient in X direction
    V = X + 0.5 * np.cos(Y)   # Gradient in Y direction
    
    # Normalize for better visualization
    N = np.sqrt(U**2 + V**2)
    U2, V2 = U/N, V/N
    
    # Plot vector field
    Q = ax.quiver(X, Y, U2, V2, N, cmap='coolwarm', scale=20, width=0.005)
    
    # Add contour lines for loss function
    loss = X**2 + Y**2 + np.sin(X) * np.cos(Y)
    contour = ax.contour(X, Y, loss, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Add some trajectory paths
    for start_x, start_y in [(2.5, 2.5), (-2.5, 2), (2, -2.5)]:
        traj_x, traj_y = [start_x], [start_y]
        for _ in range(20):
            # Simple gradient descent step
            x_idx = np.argmin(np.abs(x - traj_x[-1]))
            y_idx = np.argmin(np.abs(y - traj_y[-1]))
            if 0 <= x_idx < len(x) and 0 <= y_idx < len(y):
                grad_x = U[y_idx, x_idx] * 0.1
                grad_y = V[y_idx, x_idx] * 0.1
                traj_x.append(traj_x[-1] - grad_x)
                traj_y.append(traj_y[-1] - grad_y)
        ax.plot(traj_x, traj_y, 'g-', linewidth=2, alpha=0.7)
        ax.scatter(traj_x[0], traj_y[0], color='red', s=50, zorder=5)
    
    # Labels
    ax.set_xlabel('Embedding Dimension 1', fontsize=11)
    ax.set_ylabel('Embedding Dimension 2', fontsize=11)
    ax.set_title('Gradient Flow in Embedding Space', fontsize=13, fontweight='bold')
    
    # Add formula
    formula = r'Gradient Flow: dJ/dv'
    ax.text(0.02, 0.98, formula, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            verticalalignment='top')
    
    plt.colorbar(Q, ax=ax, label='Gradient Magnitude')
    save_figure('gradient_vector_field')

# 4. Negative Sampling Comparison
def generate_negative_sampling_comparison():
    """Bar chart showing computational savings"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Vocabulary sizes to compare
    vocab_sizes = [1000, 5000, 10000, 50000, 100000]
    k_values = [5, 10, 20]  # negative samples
    
    # Computational cost
    full_softmax = vocab_sizes
    
    # Plot 1: Cost comparison
    x_pos = np.arange(len(vocab_sizes))
    width = 0.2
    
    ax1.bar(x_pos - width, full_softmax, width, label='Full Softmax O(W)', 
            color=COLORS['training'])
    
    for i, k in enumerate(k_values):
        neg_sampling = [k + 1 for _ in vocab_sizes]
        ax1.bar(x_pos + i*width, neg_sampling, width, 
                label=f'Neg. Sampling (k={k})', alpha=0.7)
    
    ax1.set_xlabel('Vocabulary Size', fontsize=11)
    ax1.set_ylabel('Operations per Update', fontsize=11)
    ax1.set_title('Computational Cost: Softmax vs Negative Sampling', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(vocab_sizes)
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speed-up factor
    speedup_data = []
    for k in k_values:
        speedup = [v / (k + 1) for v in vocab_sizes]
        speedup_data.append(speedup)
    
    for i, (k, speedup) in enumerate(zip(k_values, speedup_data)):
        ax2.plot(vocab_sizes, speedup, 'o-', label=f'k={k}', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Vocabulary Size', fontsize=11)
    ax2.set_ylabel('Speed-up Factor', fontsize=11)
    ax2.set_title('Training Speed-up with Negative Sampling', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add formula
    formula = r'Negative Sampling Objective'
    fig.suptitle(formula, fontsize=11, y=1.02)
    
    plt.tight_layout()
    save_figure('negative_sampling_comparison')

# 5. GloVe Co-occurrence Matrix
def generate_glove_cooccurrence_matrix():
    """Matrix heatmap showing co-occurrence patterns"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create sample co-occurrence matrix
    words = ['ice', 'steam', 'solid', 'gas', 'water', 'fashion', 'clothes', 'style']
    n_words = len(words)
    
    # Simulate co-occurrence counts
    np.random.seed(42)
    X = np.zeros((n_words, n_words))
    
    # Create semantic clusters
    # Water-related cluster
    for i in range(5):
        for j in range(5):
            if i != j:
                X[i, j] = np.random.poisson(50) + 20
    
    # Fashion-related cluster  
    for i in range(5, 8):
        for j in range(5, 8):
            if i != j:
                X[i, j] = np.random.poisson(40) + 15
    
    # Add some noise
    X += np.random.poisson(2, (n_words, n_words))
    X = (X + X.T) / 2  # Make symmetric
    np.fill_diagonal(X, 0)
    
    # Plot co-occurrence matrix
    im1 = ax1.imshow(X, cmap='YlOrRd', aspect='equal')
    ax1.set_title('Co-occurrence Matrix $X_{ij}$', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(n_words))
    ax1.set_xticklabels(words, rotation=45, ha='right')
    ax1.set_yticks(range(n_words))
    ax1.set_yticklabels(words)
    plt.colorbar(im1, ax=ax1)
    
    # Add text annotations for high values
    for i in range(n_words):
        for j in range(n_words):
            if X[i, j] > 40:
                ax1.text(j, i, f'{int(X[i, j])}', ha='center', va='center', 
                        color='white' if X[i, j] > 60 else 'black', fontsize=8)
    
    # Plot probability ratios
    # Select probe words
    probe_idx = 4  # 'water'
    ratios = np.zeros((n_words, n_words))
    
    for i in range(n_words):
        for j in range(n_words):
            if i != j and X[i, probe_idx] > 0 and X[j, probe_idx] > 0:
                ratios[i, j] = (X[i, probe_idx] / np.sum(X[i, :])) / \
                               (X[j, probe_idx] / np.sum(X[j, :]))
    
    # Log scale for better visualization
    log_ratios = np.log(ratios + 1e-8)
    log_ratios[ratios == 0] = 0
    
    im2 = ax2.imshow(log_ratios, cmap='RdBu_r', aspect='equal', vmin=-2, vmax=2)
    ax2.set_title(f'Log Probability Ratios (probe: "{words[probe_idx]}")', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(n_words))
    ax2.set_xticklabels(words, rotation=45, ha='right')
    ax2.set_yticks(range(n_words))
    ax2.set_yticklabels(words)
    plt.colorbar(im2, ax=ax2, label='log(P_ik/P_jk)')
    
    # Add formula
    formula = r'$\frac{P_{ik}}{P_{jk}} = \frac{X_{ik}/X_i}{X_{jk}/X_j}$'
    fig.suptitle(formula, fontsize=12, y=1.02)
    
    plt.tight_layout()
    save_figure('glove_cooccurrence_matrix')

# 6. GloVe Weighting Function
def generate_glove_weighting_curves():
    """Line plot showing weighting function with different alpha values"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Define x range
    x = np.linspace(0, 200, 1000)
    x_max = 100
    alphas = [0.5, 0.75, 1.0]
    
    # Plot weighting functions
    for alpha in alphas:
        f = np.where(x < x_max, (x/x_max)**alpha, 1.0)
        ax1.plot(x, f, linewidth=2, label=f'α = {alpha}')
    
    # Mark x_max
    ax1.axvline(x=x_max, color='red', linestyle='--', alpha=0.5, label='$x_{max}$')
    ax1.fill_between([x_max, 200], 0, 1, alpha=0.1, color='gray')
    
    ax1.set_xlabel('Co-occurrence Count $X_{ij}$', fontsize=11)
    ax1.set_ylabel('Weight $f(X_{ij})$', fontsize=11)
    ax1.set_title('GloVe Weighting Function', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.1)
    
    # Add formula annotation
    formula = r'f(x) = (x/x_max)^alpha if x < x_max, else 1'
    ax1.text(0.5, 0.3, formula, transform=ax1.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Plot 2: Effect on loss contribution
    x_vals = [1, 10, 50, 100, 150]
    alpha = 0.75
    weights = [min((x_val/x_max)**alpha, 1.0) for x_val in x_vals]
    
    colors = [COLORS['embedding'] if x_val < x_max else COLORS['training'] for x_val in x_vals]
    bars = ax2.bar(range(len(x_vals)), weights, color=colors, alpha=0.7)
    
    ax2.set_xlabel('Co-occurrence Frequency', fontsize=11)
    ax2.set_ylabel('Loss Contribution Weight', fontsize=11)
    ax2.set_title(f'Loss Weighting Effect (α={alpha}, $x_{{max}}$={x_max})', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(x_vals)))
    ax2.set_xticklabels([f'X={x}' for x in x_vals])
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{weight:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_figure('glove_weighting_curves')

# 7. Attention Matrix Visualization
def generate_attention_matrix_visual():
    """Interactive attention matrix visualization"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create sample sentence
    tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    n_tokens = len(tokens)
    
    # Simulate attention scores (QK^T)
    np.random.seed(42)
    raw_attention = np.random.randn(n_tokens, n_tokens) * 2
    
    # Make it more realistic (tokens attend to nearby tokens more)
    for i in range(n_tokens):
        for j in range(n_tokens):
            distance = abs(i - j)
            raw_attention[i, j] += 3 * np.exp(-distance/2)
    
    # Plot 1: Raw scores
    im1 = ax1.imshow(raw_attention, cmap='Blues', aspect='equal')
    ax1.set_title('Step 1: $QK^T$ (Raw Scores)', fontsize=11, fontweight='bold')
    ax1.set_xticks(range(n_tokens))
    ax1.set_xticklabels(tokens, rotation=45)
    ax1.set_yticks(range(n_tokens))
    ax1.set_yticklabels(tokens)
    ax1.set_xlabel('Keys', fontsize=10)
    ax1.set_ylabel('Queries', fontsize=10)
    plt.colorbar(im1, ax=ax1)
    
    # Plot 2: Scaled scores
    d_k = 64  # dimension
    scaled_attention = raw_attention / np.sqrt(d_k)
    im2 = ax2.imshow(scaled_attention, cmap='Blues', aspect='equal')
    ax2.set_title(f'Step 2: $QK^T/\\sqrt{{d_k}}$ (d_k={d_k})', fontsize=11, fontweight='bold')
    ax2.set_xticks(range(n_tokens))
    ax2.set_xticklabels(tokens, rotation=45)
    ax2.set_yticks(range(n_tokens))
    ax2.set_yticklabels(tokens)
    ax2.set_xlabel('Keys', fontsize=10)
    ax2.set_ylabel('Queries', fontsize=10)
    plt.colorbar(im2, ax=ax2)
    
    # Plot 3: After softmax
    attention_weights = np.exp(scaled_attention)
    attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)
    
    im3 = ax3.imshow(attention_weights, cmap='Reds', aspect='equal', vmin=0, vmax=0.5)
    ax3.set_title('Step 3: Softmax (Attention Weights)', fontsize=11, fontweight='bold')
    ax3.set_xticks(range(n_tokens))
    ax3.set_xticklabels(tokens, rotation=45)
    ax3.set_yticks(range(n_tokens))
    ax3.set_yticklabels(tokens)
    ax3.set_xlabel('Keys', fontsize=10)
    ax3.set_ylabel('Queries', fontsize=10)
    plt.colorbar(im3, ax=ax3)
    
    # Add value annotations for high attention
    for i in range(n_tokens):
        for j in range(n_tokens):
            if attention_weights[i, j] > 0.2:
                ax3.text(j, i, f'{attention_weights[i, j]:.2f}', 
                        ha='center', va='center', color='white', fontsize=8)
    
    # Add formula
    formula = r'$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$'
    fig.suptitle(formula, fontsize=12, y=1.02)
    
    plt.tight_layout()
    save_figure('attention_matrix_visual')

# 8. Positional Encoding Patterns
def generate_positional_encoding_patterns():
    """Sinusoidal pattern visualization"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Parameters
    max_pos = 50
    d_model = 128
    
    # Generate positional encodings
    positions = np.arange(max_pos)
    dimensions = np.arange(d_model)
    
    pos_encoding = np.zeros((max_pos, d_model))
    
    for pos in positions:
        for i in range(0, d_model, 2):
            pos_encoding[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    
    # Plot 1: Heatmap of positional encodings
    im1 = ax1.imshow(pos_encoding.T, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax1.set_title('Positional Encoding Pattern', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Position', fontsize=11)
    ax1.set_ylabel('Dimension', fontsize=11)
    ax1.set_yticks([0, 32, 64, 96, 127])
    ax1.set_yticklabels(['0', '32', '64', '96', '128'])
    plt.colorbar(im1, ax=ax1)
    
    # Plot 2: Individual dimension patterns
    dims_to_plot = [0, 1, 10, 11, 50, 51, 100, 101]
    colors_cycle = plt.cm.tab10(np.linspace(0, 1, len(dims_to_plot)))
    
    for idx, (dim, color) in enumerate(zip(dims_to_plot, colors_cycle)):
        label = f'dim {dim} ({"sin" if dim % 2 == 0 else "cos"})'
        ax2.plot(positions[:30], pos_encoding[:30, dim], 
                label=label, color=color, linewidth=1.5)
    
    ax2.set_xlabel('Position', fontsize=11)
    ax2.set_ylabel('Encoding Value', fontsize=11)
    ax2.set_title('Sinusoidal Patterns at Different Dimensions', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', ncol=2, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.2, 1.2)
    
    # Add formulas
    formula_sin = r'$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$'
    formula_cos = r'$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$'
    
    ax2.text(0.02, 0.95, formula_sin, transform=ax2.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(0.02, 0.85, formula_cos, transform=ax2.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    save_figure('positional_encoding_patterns')

# 9. Distance Concentration Formula Visualization
def generate_distance_concentration_formula():
    """Enhanced distance concentration with formula overlay"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Generate data for different dimensions
    dimensions = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000])
    
    # Theoretical ratio for Gaussian points
    ratio_theoretical = 1 / np.sqrt(1 + 2/dimensions)
    
    # Simulate actual ratios
    np.random.seed(42)
    ratio_simulated = []
    
    for d in dimensions:
        if d <= 100:  # Only simulate for smaller dimensions
            points = np.random.randn(1000, d)
            distances = np.linalg.norm(points[None, :, :] - points[:, None, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            
            dist_min = np.min(distances)
            dist_max = np.max(distances[distances != np.inf])
            dist_mean = np.mean(distances[distances != np.inf])
            
            ratio_simulated.append((dist_max - dist_min) / dist_mean)
        else:
            # Use theoretical for large dimensions
            ratio_simulated.append(1 / np.sqrt(1 + 2/d))
    
    # Plot 1: Convergence of ratio
    ax1.plot(dimensions, ratio_theoretical, 'r-', linewidth=2, 
            label='Theoretical: $1/\\sqrt{1 + 2/d}$')
    ax1.plot(dimensions[:len(ratio_simulated)], ratio_simulated, 'bo-', 
            markersize=6, label='Simulated')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Dimension $d$', fontsize=11)
    ax1.set_ylabel('Distance Ratio', fontsize=11)
    ax1.set_title('Distance Concentration: $(d_{max} - d_{min})/d_{mean} \\to 0$', 
                 fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add annotations for key dimensions
    for d, ratio in [(10, ratio_theoretical[2]), (100, ratio_theoretical[5]), 
                     (1000, ratio_theoretical[8])]:
        ax1.annotate(f'd={d}\nratio={ratio:.3f}',
                    xy=(d, ratio), xytext=(d*1.5, ratio+0.1),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5),
                    fontsize=9)
    
    # Plot 2: Distribution of distances
    d_examples = [2, 10, 100]
    colors = [COLORS['embedding'], COLORS['attention'], COLORS['training']]
    
    for d, color in zip(d_examples, colors):
        # Simulate distance distribution
        points = np.random.randn(500, d)
        distances = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                distances.append(np.linalg.norm(points[i] - points[j]))
        
        # Plot histogram
        ax2.hist(distances, bins=30, alpha=0.5, label=f'd={d}', 
                color=color, density=True)
    
    ax2.set_xlabel('Pairwise Distance', fontsize=11)
    ax2.set_ylabel('Probability Density', fontsize=11)
    ax2.set_title('Distance Distribution in Different Dimensions', 
                 fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add key insight
    ax2.text(0.5, 0.85, 'Higher dimensions →\nNarrower distribution', 
            transform=ax2.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
            ha='center')
    
    plt.tight_layout()
    save_figure('distance_concentration_formula')

# 10. Volume Decomposition
def generate_volume_decomposition():
    """Decomposition view of sphere volume formula"""
    fig = plt.figure(figsize=(14, 8))
    
    # Create subplots
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 3), (1, 0))
    ax3 = plt.subplot2grid((2, 3), (1, 1))
    ax4 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
    
    # Plot 1: Volume formula components
    dims = np.arange(1, 31)
    
    # Components
    pi_term = np.pi ** (dims / 2)
    gamma_term = np.array([gamma(d/2 + 1) for d in dims])
    volume = pi_term / gamma_term
    
    ax1.semilogy(dims, pi_term, 'b-', linewidth=2, label='$\\pi^{d/2}$ (numerator)')
    ax1.semilogy(dims, gamma_term, 'r-', linewidth=2, label='$\\Gamma(d/2 + 1)$ (denominator)')
    ax1.semilogy(dims, volume, 'g-', linewidth=3, label='$V_d = \\pi^{d/2}/\\Gamma(d/2 + 1)$')
    
    ax1.set_xlabel('Dimension', fontsize=11)
    ax1.set_ylabel('Value (log scale)', fontsize=11)
    ax1.set_title('Volume Formula Components', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=5, color='orange', linestyle='--', alpha=0.5)
    ax1.text(5.5, 1e10, 'Peak at d=5', fontsize=9, color='orange')
    
    # Plot 2: Growth rates
    growth_pi = np.diff(np.log(pi_term))
    growth_gamma = np.diff(np.log(gamma_term))
    
    ax2.plot(dims[1:], growth_pi, 'b-', linewidth=2, label='$\\pi^{d/2}$ growth')
    ax2.plot(dims[1:], growth_gamma, 'r-', linewidth=2, label='$\\Gamma$ growth')
    
    ax2.set_xlabel('Dimension', fontsize=10)
    ax2.set_ylabel('Log Growth Rate', fontsize=10)
    ax2.set_title('Growth Rate Comparison', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Stirling approximation
    n_vals = np.arange(5, 26)
    exact_factorial = np.array([gamma(n + 1) for n in n_vals])
    stirling_approx = np.sqrt(2 * np.pi * n_vals) * (n_vals / np.e) ** n_vals
    
    ax3.semilogy(n_vals, exact_factorial, 'g-', linewidth=2, label='Exact $n!$')
    ax3.semilogy(n_vals, stirling_approx, 'r--', linewidth=2, label='Stirling approx')
    
    ax3.set_xlabel('n', fontsize=10)
    ax3.set_ylabel('Value (log scale)', fontsize=10)
    ax3.set_title("Stirling's Approximation", fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Volume decay visualization
    dims_extended = np.arange(1, 51)
    volumes_extended = [np.pi**(d/2) / gamma(d/2 + 1) for d in dims_extended]
    
    ax4.plot(dims_extended, volumes_extended, 'g-', linewidth=2)
    ax4.fill_between(dims_extended, 0, volumes_extended, alpha=0.3, color='green')
    
    ax4.set_xlabel('Dimension', fontsize=11)
    ax4.set_ylabel('Unit Sphere Volume', fontsize=11)
    ax4.set_title('Rapid Volume Decay', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add annotations
    max_idx = np.argmax(volumes_extended)
    ax4.scatter(dims_extended[max_idx], volumes_extended[max_idx], 
               color='red', s=100, zorder=5)
    ax4.annotate(f'Max at d={dims_extended[max_idx]}\nV={volumes_extended[max_idx]:.3f}',
                xy=(dims_extended[max_idx], volumes_extended[max_idx]),
                xytext=(10, 4), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # Add explanation text
    ax4.text(30, 3, 'Volume → 0\nas d → ∞', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Main formula
    fig.suptitle('$V_d = \\frac{\\pi^{d/2}}{\\Gamma(d/2 + 1)}$ - Why Factorial Wins', 
                fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    save_figure('volume_decomposition')

# 11. Training Loss Formula Visualization
def generate_training_loss_formula():
    """Training loss dynamics with formula overlay"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Parameters
    epochs = np.linspace(0, 100, 500)
    L0 = 10  # Initial loss
    alpha = 0.05  # Decay rate
    
    # Different training scenarios
    scenarios = {
        'Ideal (α=0.1)': {'alpha': 0.1, 'noise': 0.05},
        'Normal (α=0.05)': {'alpha': 0.05, 'noise': 0.1},
        'Slow (α=0.02)': {'alpha': 0.02, 'noise': 0.15}
    }
    
    # Plot 1: Loss curves
    for name, params in scenarios.items():
        # Exponential decay + noise
        loss = L0 * np.exp(-params['alpha'] * epochs)
        noise = np.random.randn(len(epochs)) * params['noise'] * np.exp(-epochs/50)
        loss_with_noise = loss + np.abs(noise)
        
        ax1.plot(epochs, loss_with_noise, linewidth=2, label=name, alpha=0.8)
        # Plot clean exponential
        ax1.plot(epochs, loss, '--', linewidth=1, alpha=0.3, color='gray')
    
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss: $L(t) = L_0 \\cdot e^{-\\alpha t} + \\epsilon(t)$', 
                 fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 11)
    
    # Add phase annotations
    ax1.axvspan(0, 20, alpha=0.1, color='green', label='Phase 1: Rapid')
    ax1.axvspan(20, 60, alpha=0.1, color='yellow', label='Phase 2: Refinement')
    ax1.axvspan(60, 100, alpha=0.1, color='red', label='Phase 3: Convergence')
    
    # Plot 2: Learning rate and gradient norm
    # Simulated gradient norms
    grad_norm = np.sqrt(300) * np.exp(-epochs/30) + np.random.randn(len(epochs)) * 0.5
    grad_norm = np.maximum(grad_norm, 0.1)
    
    # Learning rate schedule
    lr = 0.01 * np.ones_like(epochs)
    lr[epochs > 30] = 0.005
    lr[epochs > 60] = 0.001
    
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(epochs, grad_norm, 'b-', linewidth=2, label='Gradient Norm')
    line2 = ax2_twin.plot(epochs, lr, 'r-', linewidth=2, label='Learning Rate')
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Gradient Norm', fontsize=11, color='b')
    ax2_twin.set_ylabel('Learning Rate', fontsize=11, color='r')
    ax2.set_title('Optimization Dynamics', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    
    # Add formula annotations
    formula_details = [
        '$L_0$: Initial loss (~8-10 for 10K vocab)',
        '$\\alpha$: Decay rate (0.02-0.1)',
        '$\\epsilon(t)$: SGD noise (decreases over time)'
    ]
    
    for i, detail in enumerate(formula_details):
        ax1.text(0.02, 0.85 - i*0.06, detail, transform=ax1.transAxes, 
                fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    save_figure('training_loss_formula')

# 12. BERT MLM Masking Pattern
def generate_bert_mlm_masking():
    """BERT masked language model visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sample sentence
    tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
    n_tokens = len(tokens)
    
    # Masking strategy
    mask_prob = 0.15
    masked_indices = [1, 4, 7]  # 'quick', 'jumps', 'lazy'
    
    # Plot 1: Original sentence
    ax1 = axes[0, 0]
    for i, token in enumerate(tokens):
        color = 'lightblue'
        if i in masked_indices:
            color = 'lightcoral'
        rect = FancyBboxPatch((i-0.4, 0), 0.8, 1, 
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='black', linewidth=1)
        ax1.add_patch(rect)
        ax1.text(i, 0.5, token, ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlim(-0.5, n_tokens-0.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_title('Step 1: Original Sequence (15% selected for masking)', fontsize=11, fontweight='bold')
    ax1.set_xticks(range(n_tokens))
    ax1.set_xticklabels(range(n_tokens))
    ax1.set_xlabel('Position', fontsize=10)
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    # Plot 2: Masked input
    ax2 = axes[0, 1]
    masked_tokens = tokens.copy()
    for idx in masked_indices:
        if np.random.random() < 0.8:  # 80% [MASK]
            masked_tokens[idx] = '[MASK]'
            color = 'yellow'
        elif np.random.random() < 0.5:  # 10% random
            masked_tokens[idx] = 'random'
            color = 'lightgreen'
        else:  # 10% unchanged
            color = 'lightgray'
        
        rect = FancyBboxPatch((idx-0.4, 0), 0.8, 1,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='black', linewidth=1)
        ax2.add_patch(rect)
    
    for i, token in enumerate(masked_tokens):
        if i not in masked_indices:
            rect = FancyBboxPatch((i-0.4, 0), 0.8, 1,
                                  boxstyle="round,pad=0.1",
                                  facecolor='lightblue', edgecolor='black', linewidth=1)
            ax2.add_patch(rect)
        ax2.text(i, 0.5, token, ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax2.set_xlim(-0.5, n_tokens-0.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_title('Step 2: Masked Input (80% [MASK], 10% random, 10% unchanged)', 
                  fontsize=11, fontweight='bold')
    ax2.set_xticks(range(n_tokens))
    ax2.set_xticklabels(range(n_tokens))
    ax2.set_xlabel('Position', fontsize=10)
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    # Plot 3: Prediction probabilities
    ax3 = axes[1, 0]
    
    # Simulate prediction probabilities for masked positions
    vocab_sample = ['quick', 'fast', 'slow', 'brown', 'jumps', 'leaps', 
                   'runs', 'lazy', 'sleepy', 'tired']
    
    # Create probability distributions
    probs_dict = {
        1: [0.7, 0.15, 0.05] + [0.01] * 7,  # 'quick' position
        4: [0.05, 0.02, 0.03, 0.01, 0.6, 0.2, 0.05] + [0.01] * 3,  # 'jumps' position
        7: [0.01] * 7 + [0.7, 0.15, 0.1]  # 'lazy' position
    }
    
    # Plot probability bars for position 1 (quick)
    pos_to_show = 1
    probs = probs_dict[pos_to_show]
    bars = ax3.bar(range(len(vocab_sample)), probs, color=COLORS['attention'], alpha=0.7)
    
    # Highlight correct prediction
    bars[0].set_color(COLORS['embedding'])
    bars[0].set_alpha(1.0)
    
    ax3.set_xlabel('Vocabulary Words', fontsize=10)
    ax3.set_ylabel('Probability', fontsize=10)
    ax3.set_title(f'Step 3: Prediction for Position {pos_to_show} (Original: "quick")', 
                  fontsize=11, fontweight='bold')
    ax3.set_xticks(range(len(vocab_sample)))
    ax3.set_xticklabels(vocab_sample, rotation=45, ha='right', fontsize=9)
    ax3.set_ylim(0, 0.8)
    ax3.grid(True, alpha=0.3)
    
    # Add probability values on bars
    for bar, prob in zip(bars[:3], probs[:3]):
        height = bar.get_height()
        if height > 0.05:
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Loss calculation
    ax4 = axes[1, 1]
    
    # Show loss components
    positions = ['Pos 1\n(quick)', 'Pos 4\n(jumps)', 'Pos 7\n(lazy)']
    true_probs = [0.7, 0.6, 0.7]  # Predicted probabilities for true words
    losses = [-np.log(p) for p in true_probs]
    
    bars = ax4.bar(range(len(positions)), losses, color=[COLORS['embedding'], 
                                                          COLORS['attention'], 
                                                          COLORS['training']], alpha=0.7)
    
    ax4.set_xlabel('Masked Positions', fontsize=10)
    ax4.set_ylabel('Loss: $-\\log P(x_i | x_{\\setminus M})$', fontsize=10)
    ax4.set_title('Step 4: MLM Loss Calculation', fontsize=11, fontweight='bold')
    ax4.set_xticks(range(len(positions)))
    ax4.set_xticklabels(positions)
    ax4.grid(True, alpha=0.3)
    
    # Add loss values
    for bar, loss, prob in zip(bars, losses, true_probs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'L={loss:.2f}\n(P={prob:.2f})', ha='center', va='bottom', fontsize=9)
    
    # Add total loss
    total_loss = np.mean(losses)
    ax4.axhline(y=total_loss, color='red', linestyle='--', alpha=0.5)
    ax4.text(1.5, total_loss + 0.05, f'Avg Loss = {total_loss:.3f}', 
            fontsize=10, color='red')
    
    # Add formula
    formula = r'$\mathcal{L}_{MLM} = -\mathbb{E} \sum_{i \in \mathcal{M}} \log P(x_i | \mathbf{x}_{\setminus \mathcal{M}})$'
    fig.suptitle(formula, fontsize=12, y=0.98)
    
    plt.tight_layout()
    save_figure('bert_mlm_masking')

# Generate all charts
def main():
    """Generate all formula visualization charts"""
    print("Generating formula visualization charts...")
    
    chart_generators = [
        ("Skip-gram Loss Surface", generate_skipgram_loss_surface),
        ("Softmax Probability Heatmap", generate_softmax_probability_heatmap),
        ("Gradient Vector Field", generate_gradient_vector_field),
        ("Negative Sampling Comparison", generate_negative_sampling_comparison),
        ("GloVe Co-occurrence Matrix", generate_glove_cooccurrence_matrix),
        ("GloVe Weighting Curves", generate_glove_weighting_curves),
        ("Attention Matrix Visual", generate_attention_matrix_visual),
        ("Positional Encoding Patterns", generate_positional_encoding_patterns),
        ("Distance Concentration Formula", generate_distance_concentration_formula),
        ("Volume Decomposition", generate_volume_decomposition),
        ("Training Loss Formula", generate_training_loss_formula),
        ("BERT MLM Masking", generate_bert_mlm_masking)
    ]
    
    for name, generator in chart_generators:
        print(f"  Generating: {name}...")
        try:
            generator()
            print(f"    Y {name} generated successfully")
        except Exception as e:
            print(f"    N Error generating {name}: {e}")
    
    print(f"\nAll charts saved to: {OUTPUT_DIR}/")
    print("Charts generated successfully!")

if __name__ == "__main__":
    main()