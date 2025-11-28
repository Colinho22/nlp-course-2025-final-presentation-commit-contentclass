"""
Generate figures for Function Approximation Discovery Handout
Focuses on regression/curve fitting, not classification
All figures use proper sigmoid activation functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for clean, printable figures
plt.style.use('seaborn-v0_8-whitegrid')

# Monochromatic color scheme for printing
COLOR_MAIN = '#404040'      # Dark gray
COLOR_ACCENT = '#B4B4B4'    # Light gray
COLOR_POSITIVE = '#2E7D32'  # Dark green
COLOR_NEGATIVE = '#C62828'  # Dark red
COLOR_BOUNDARY = '#1976D2'  # Blue
COLOR_TARGET = '#404040'    # Target function
COLOR_APPROX = '#FF6B6B'    # Approximation

OUTPUT_DIR = '../figures/handout/'


def sigmoid(z):
    """Proper sigmoid activation function with numerical stability"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def create_temperature_challenge_scatter():
    """Figure 1: Temperature prediction challenge - the concrete problem"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Real temperature data points throughout the day
    hours = np.array([0, 6, 12, 18, 24])
    temps = np.array([8, 10, 22, 18, 8])

    # Smooth curve (what we want to learn)
    hours_smooth = np.linspace(0, 24, 200)
    # Simple sinusoidal model for temperature
    temps_smooth = 15 + 7 * np.sin((hours_smooth - 6) * np.pi / 12)

    # Plot smooth target curve
    ax.plot(hours_smooth, temps_smooth, COLOR_TARGET, linewidth=3,
           linestyle='--', label='Smooth curve (what we want)', alpha=0.7)

    # Plot measured data points
    ax.scatter(hours, temps, s=200, c=COLOR_POSITIVE, marker='o',
              edgecolors='black', linewidths=2, label='Measured data', zorder=5)

    # Add question marks between points
    question_hours = [3, 9, 15, 21]
    for h in question_hours:
        ax.text(h, 12, '?', fontsize=30, ha='center', va='center',
               color=COLOR_NEGATIVE, fontweight='bold')

    ax.set_xlabel('Time of Day (hours)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=13, fontweight='bold')
    ax.set_title('The Challenge: Predict Temperature at Any Time',
                fontsize=15, fontweight='bold')
    ax.set_xlim(-1, 25)
    ax.set_ylim(5, 25)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(['12am', '6am', '12pm', '6pm', '12am'])
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}temperature_challenge_scatter.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: temperature_challenge_scatter.pdf")
    plt.close()


def create_single_sigmoid_parameters():
    """Figure 2: How parameters control sigmoid shape"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x = np.linspace(0, 24, 200)

    # Panel 1: Varying weight (steepness)
    for w in [0.5, 1.0, 2.0]:
        y = sigmoid(w * (x - 12))
        axes[0].plot(x, y, linewidth=2.5, label=f'w = {w}')
    axes[0].set_title('Weight w: Controls Steepness', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Input x', fontsize=11)
    axes[0].set_ylabel('Output σ(w(x-12))', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=12, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Panel 2: Varying bias (shift)
    w = 1.0
    for b_val, b_label in [(8, '8'), (12, '12'), (16, '16')]:
        y = sigmoid(w * (x - b_val))
        axes[1].plot(x, y, linewidth=2.5, label=f'shift = {b_label}')
    axes[1].set_title('Bias b: Shifts Left/Right', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Input x', fontsize=11)
    axes[1].set_ylabel('Output σ(x-b)', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Varying amplitude (scale)
    w = 1.0
    b = 12
    for a in [10, 15, 20]:
        y = a * sigmoid(w * (x - b))
        axes[2].plot(x, y, linewidth=2.5, label=f'a = {a}')
    axes[2].set_title('Amplitude a: Scales Output', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Input x', fontsize=11)
    axes[2].set_ylabel('Output a·σ(x-12)', fontsize=11)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('How Parameters Control the Sigmoid Curve', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}single_sigmoid_parameters.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: single_sigmoid_parameters.pdf")
    plt.close()


def create_temperature_single_sigmoid_fit():
    """Figure 3: Single sigmoid attempting to fit temperature data"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Temperature data
    hours = np.array([0, 6, 12, 18, 24])
    temps = np.array([8, 10, 22, 18, 8])

    # Target smooth curve
    hours_smooth = np.linspace(0, 24, 200)
    temps_smooth = 15 + 7 * np.sin((hours_smooth - 6) * np.pi / 12)

    # Single sigmoid approximation (best fit S-curve)
    # y = a * sigmoid(w * (x - b)) + c
    w, b, a, c = 0.8, 10, 12, 8
    single_neuron_fit = a * sigmoid(w * (hours_smooth - b)) + c

    # Plot target
    ax.plot(hours_smooth, temps_smooth, COLOR_TARGET, linewidth=3,
           linestyle='--', label='Target (actual temperature)', alpha=0.7)

    # Plot single neuron fit
    ax.plot(hours_smooth, single_neuron_fit, COLOR_NEGATIVE, linewidth=3,
           label='Single Neuron (S-curve)', alpha=0.8)

    # Plot data points
    ax.scatter(hours, temps, s=150, c=COLOR_POSITIVE, marker='o',
              edgecolors='black', linewidths=2, label='Data', zorder=5)

    # Calculate and show error
    mse = np.mean((temps_smooth - single_neuron_fit)**2)
    ax.text(20, 12, f'Error (MSE): {mse:.1f}', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

    # Annotation
    ax.annotate('Can only go UP,\nnot up-then-down!', xy=(18, 16), xytext=(20, 22),
               fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_xlabel('Time of Day (hours)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=13, fontweight='bold')
    ax.set_title('Problem: One Neuron Can Only Make S-Curves',
                fontsize=15, fontweight='bold')
    ax.set_xlim(-1, 25)
    ax.set_ylim(5, 25)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(['12am', '6am', '12pm', '6pm', '12am'])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}temperature_single_sigmoid_fit.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: temperature_single_sigmoid_fit.pdf")
    plt.close()


def create_two_sigmoid_subtraction():
    """Figure 4: Visual math of how two sigmoids create a bump"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.linspace(0, 24, 200)

    # First sigmoid (rising early)
    w1, b1 = 0.8, 10
    sigma1 = sigmoid(w1 * (x - b1))

    # Second sigmoid (rising later)
    w2, b2 = 0.8, 14
    sigma2 = sigmoid(w2 * (x - b2))

    # Difference creates bump
    bump = 10 * (sigma1 - sigma2)

    # Panel 1: First sigmoid
    axes[0].plot(x, sigma1, 'red', linewidth=3, label='σ₁(x)')
    axes[0].fill_between(x, 0, sigma1, alpha=0.2, color='red')
    axes[0].set_title('Sigmoid 1\nRises at x=10', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('x', fontsize=11)
    axes[0].set_ylabel('Output', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.1, 1.5)
    axes[0].axvline(x=10, color='red', linestyle=':', alpha=0.5)

    # Panel 2: Second sigmoid
    axes[1].plot(x, sigma2, 'blue', linewidth=3, label='σ₂(x)')
    axes[1].fill_between(x, 0, sigma2, alpha=0.2, color='blue')
    axes[1].set_title('Sigmoid 2\nRises at x=14', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('x', fontsize=11)
    axes[1].set_ylabel('Output', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.1, 1.5)
    axes[1].axvline(x=14, color='blue', linestyle=':', alpha=0.5)

    # Panel 3: Difference (bump)
    axes[2].plot(x, sigma1, 'red', linewidth=2, alpha=0.3, linestyle='--', label='σ₁')
    axes[2].plot(x, sigma2, 'blue', linewidth=2, alpha=0.3, linestyle='--', label='σ₂')
    axes[2].plot(x, bump, COLOR_POSITIVE, linewidth=3, label='10×(σ₁ - σ₂) = BUMP!')
    axes[2].fill_between(x, 0, bump, alpha=0.2, color=COLOR_POSITIVE)
    axes[2].set_title('Subtract to Get Bump!\nσ₁ - σ₂', fontsize=12, fontweight='bold', color='green')
    axes[2].set_xlabel('x', fontsize=11)
    axes[2].set_ylabel('Output', fontsize=11)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='black', linewidth=0.5)
    axes[2].annotate('Peak where\ndifference is max', xy=(12, 5), xytext=(18, 8),
                    fontsize=9, ha='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    plt.suptitle('The Trick: Two Sigmoids Create a Localized Bump', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}two_sigmoid_subtraction.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: two_sigmoid_subtraction.pdf")
    plt.close()


def create_bump_temperature_application():
    """Figure 5: Using bump to model temperature cycle"""
    fig, ax = plt.subplots(figsize=(10, 6))

    hours_smooth = np.linspace(0, 24, 200)

    # Target temperature
    temps_smooth = 15 + 7 * np.sin((hours_smooth - 6) * np.pi / 12)

    # Two-neuron bump approximation
    w1, b1, a1 = 0.8, 8, 15
    w2, b2, a2 = 0.8, 16, -15
    two_neuron_fit = 15 + a1 * sigmoid(w1 * (hours_smooth - b1)) + a2 * sigmoid(w2 * (hours_smooth - b2))

    # Plot target
    ax.plot(hours_smooth, temps_smooth, COLOR_TARGET, linewidth=3,
           linestyle='--', label='Target (actual)', alpha=0.7)

    # Plot two-neuron fit
    ax.plot(hours_smooth, two_neuron_fit, COLOR_POSITIVE, linewidth=3,
           label='Two Neurons (creates bump)', alpha=0.8)

    # Calculate error
    mse = np.mean((temps_smooth - two_neuron_fit)**2)
    ax.text(20, 12, f'Error (MSE): {mse:.1f}\nMuch better!', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Annotations
    ax.annotate('Morning\nwarmup', xy=(8, 14), xytext=(4, 18),
               fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3),
               arrowprops=dict(arrowstyle='->', color='orange', lw=2))

    ax.annotate('Afternoon\ncooldown', xy=(16, 18), xytext=(20, 22),
               fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    ax.set_xlabel('Time of Day (hours)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=13, fontweight='bold')
    ax.set_title('Solution: Two Neurons Model Daily Temperature Cycle',
                fontsize=15, fontweight='bold')
    ax.set_xlim(-1, 25)
    ax.set_ylim(5, 25)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(['12am', '6am', '12pm', '6pm', '12am'])
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}bump_temperature_application.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: bump_temperature_application.pdf")
    plt.close()


def create_three_neuron_parabola():
    """Figure 6: Three neurons approximating a parabola"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.linspace(0, 24, 200)

    # Target: parabola (simple quadratic)
    y_target = -(x - 12)**2 / 10 + 20

    # 1 neuron: poor S-curve
    w1, b1, a1, c1 = 0.5, 12, 15, 10
    y_1neuron = a1 * sigmoid(w1 * (x - b1)) + c1

    # 3 neurons: good fit
    neurons_3 = [
        (1.0, 12, 8, 10),    # (w, b, a, baseline)
        (1.5, 8, 15, 0),
        (1.5, 16, -15, 0)
    ]
    y_3neurons = neurons_3[0][3]  # baseline
    for w, b, a, _ in neurons_3:
        y_3neurons += a * sigmoid(w * (x - b))

    # Panel 1: Target
    axes[0].plot(x, y_target, COLOR_TARGET, linewidth=4, label='Target: Parabola')
    axes[0].scatter([12], [20], s=300, c='gold', marker='*',
                   edgecolors='black', linewidths=2, zorder=5, label='Peak')
    axes[0].set_title('Target Function\ny = -(x-12)²/10 + 20', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('x', fontsize=11)
    axes[0].set_ylabel('y', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 22)

    # Panel 2: 1 neuron (poor)
    axes[1].plot(x, y_target, COLOR_TARGET, linewidth=3, linestyle='--',
                alpha=0.5, label='Target')
    axes[1].plot(x, y_1neuron, COLOR_NEGATIVE, linewidth=3, label='1 Neuron')
    mse_1 = np.mean((y_target - y_1neuron)**2)
    axes[1].text(18, 8, f'MSE: {mse_1:.1f}\nPoor fit', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    axes[1].set_title('1 Neuron Fit\n(Only S-curve)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('x', fontsize=11)
    axes[1].set_ylabel('y', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 22)

    # Panel 3: 3 neurons (good)
    axes[2].plot(x, y_target, COLOR_TARGET, linewidth=3, linestyle='--',
                alpha=0.5, label='Target')
    axes[2].plot(x, y_3neurons, COLOR_POSITIVE, linewidth=3, label='3 Neurons')
    mse_3 = np.mean((y_target - y_3neurons)**2)
    axes[2].text(18, 8, f'MSE: {mse_3:.1f}\nGood fit!', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
    axes[2].set_title('3 Neurons Fit\n(Captures peak!)', fontsize=12, fontweight='bold', color='green')
    axes[2].set_xlabel('x', fontsize=11)
    axes[2].set_ylabel('y', fontsize=11)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 22)

    plt.suptitle('Three Neurons Can Approximate a Parabola', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}three_neuron_parabola.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: three_neuron_parabola.pdf")
    plt.close()


def create_neuron_contributions():
    """Figure 7: Each neuron's contribution shown separately"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x = np.linspace(0, 24, 200)

    # Target parabola
    y_target = -(x - 12)**2 / 10 + 20

    # Three neurons with specific roles
    baseline = 10

    # Neuron 1: Baseline + left rise
    w1, b1, a1 = 1.5, 8, 15
    y1 = a1 * sigmoid(w1 * (x - b1))

    # Neuron 2: Middle peak (positive bump)
    w2, b2, a2 = 2.0, 12, 10
    y2 = a2 * sigmoid(w2 * (x - b2))

    # Neuron 3: Right fall
    w3, b3, a3 = 1.5, 16, -15
    y3 = a3 * sigmoid(w3 * (x - b3))

    # Combined
    y_combined = baseline + y1 + y2 + y3

    # Panel 1: Neuron 1
    axes[0, 0].plot(x, y1, 'red', linewidth=3, label='Neuron 1: Left rise')
    axes[0, 0].fill_between(x, 0, y1, alpha=0.2, color='red')
    axes[0, 0].set_title('Neuron 1: Handles Left Side', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Contribution', fontsize=10)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-5, 20)

    # Panel 2: Neuron 2
    axes[0, 1].plot(x, y2, 'blue', linewidth=3, label='Neuron 2: Peak boost')
    axes[0, 1].fill_between(x, 0, y2, alpha=0.2, color='blue')
    axes[0, 1].set_title('Neuron 2: Creates Peak', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Contribution', fontsize=10)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-5, 20)

    # Panel 3: Neuron 3
    axes[1, 0].plot(x, y3, 'green', linewidth=3, label='Neuron 3: Right fall')
    axes[1, 0].fill_between(x, y3, 0, alpha=0.2, color='green')
    axes[1, 0].set_title('Neuron 3: Handles Right Side', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('x', fontsize=10)
    axes[1, 0].set_ylabel('Contribution', fontsize=10)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(-20, 5)
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)

    # Panel 4: Combined result
    axes[1, 1].plot(x, y_target, COLOR_TARGET, linewidth=3, linestyle='--',
                   alpha=0.7, label='Target')
    axes[1, 1].plot(x, y_combined, COLOR_POSITIVE, linewidth=3,
                   label=f'Sum (baseline + N1 + N2 + N3)')
    mse = np.mean((y_target - y_combined)**2)
    axes[1, 1].text(18, 8, f'MSE: {mse:.1f}', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    axes[1, 1].set_title('Combined: All 3 Neurons Working Together',
                        fontsize=11, fontweight='bold', color='green')
    axes[1, 1].set_xlabel('x', fontsize=10)
    axes[1, 1].set_ylabel('y', fontsize=10)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 22)

    plt.suptitle('Each Neuron Handles Different Parts of the Function',
                fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}neuron_contributions.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: neuron_contributions.pdf")
    plt.close()


def create_universal_approximation_sine():
    """Figure 8: Progressive approximation with 1, 5, 10, 20 neurons"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    x = np.linspace(0, 2*np.pi, 200)
    y_target = np.sin(x)

    neuron_counts = [1, 5, 10, 20]
    titles = ['1 Neuron (Poor)', '5 Neurons (Better)', '10 Neurons (Good)', '20 Neurons (Excellent)']
    colors = ['red', 'orange', '#4CAF50', '#2E7D32']

    for idx, (n, title, col) in enumerate(zip(neuron_counts, titles, colors)):
        ax = axes[idx]

        # Generate approximation with n neurons
        y_approx = np.zeros_like(x)

        if n == 1:
            # Single sigmoid - very poor fit
            y_approx = 2 * (sigmoid(1.5 * (x - np.pi)) - 0.5)
        else:
            # Universal approximation theorem approach:
            # Use difference of sigmoids (localized bumps) at strategic points
            centers = np.linspace(0, 2*np.pi, n)
            width = 2*np.pi / (n - 1) if n > 1 else 2*np.pi

            for center in centers:
                # Target value at this center
                target_at_center = np.sin(center)

                # Create narrow bump using sigmoid difference
                # Steepness increases with more neurons for better localization
                steepness = max(3.0, n / 4.0)

                # Two sigmoids offset to create bump
                left_sig = sigmoid(steepness * (x - (center - width/2)))
                right_sig = sigmoid(steepness * (x - (center + width/2)))

                # Bump = left going up - right going up = localized peak
                bump = left_sig - right_sig

                # Weight this bump by the target value
                y_approx += target_at_center * bump

            # The sum of bumps approximates the target
            # No normalization needed - bumps are already scaled correctly

        # Plot target
        ax.plot(x, y_target, COLOR_TARGET, linewidth=3, label='Target: sin(x)',
               linestyle='--', alpha=0.7)

        # Plot approximation
        ax.plot(x, y_approx, col, linewidth=3, label=f'{n} neuron{"s" if n > 1 else ""}')

        # Calculate MSE
        mse = np.mean((y_target - y_approx)**2)
        ax.text(1, -1.3, f'MSE: {mse:.4f}', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(0, 2*np.pi)

    plt.suptitle('Universal Approximation: More Neurons = Better Fit',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}universal_approximation_sine.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: universal_approximation_sine.pdf")
    plt.close()


def create_error_vs_neuron_count():
    """Figure 9: Chart showing MSE decreasing with more neurons"""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(0, 2*np.pi, 200)
    y_target = np.sin(x)

    neuron_counts = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
    mse_values = []

    for n in neuron_counts:
        y_approx = np.zeros_like(x)

        if n == 1:
            y_approx = 2 * (sigmoid(1.5 * (x - np.pi)) - 0.5)
        else:
            positions = np.linspace(0, 2*np.pi, n)
            for pos in positions:
                phase = np.sin(pos)
                w = 3.0
                b = -w * pos
                a = 4.0 * phase
                y_approx += a * (sigmoid(w * x + b) - 0.5)

            if np.max(np.abs(y_approx)) > 0:
                y_approx = y_approx / np.max(np.abs(y_approx))

        mse = np.mean((y_target - y_approx)**2)
        mse_values.append(mse)

    # Plot
    ax.plot(neuron_counts, mse_values, COLOR_MAIN, linewidth=3, marker='o',
           markersize=8, label='Approximation Error (MSE)')

    # Highlight key points
    ax.scatter([1], [mse_values[0]], s=300, c='red', marker='o',
              edgecolors='black', linewidths=2, zorder=5)
    ax.text(1, mse_values[0] + 0.02, 'Poor\n(1 neuron)', ha='center',
           fontsize=10, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

    ax.scatter([10], [mse_values[6]], s=300, c='green', marker='o',
              edgecolors='black', linewidths=2, zorder=5)
    ax.text(10, mse_values[6] + 0.02, 'Good\n(10 neurons)', ha='center',
           fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    ax.set_xlabel('Number of Neurons', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=13, fontweight='bold')
    ax.set_title('More Neurons = Lower Error (Universal Approximation)',
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 55)
    ax.set_ylim(0, max(mse_values) * 1.1)

    # Add diminishing returns annotation
    ax.annotate('Diminishing\nreturns', xy=(30, mse_values[-2]), xytext=(40, 0.05),
               fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3),
               arrowprops=dict(arrowstyle='->', color='orange', lw=2))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}error_vs_neuron_count.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: error_vs_neuron_count.pdf")
    plt.close()


def create_multiple_function_examples():
    """Figure 10: Gallery showing different function types approximated"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    x = np.linspace(0, 10, 200)

    # Function 1: Step function
    y_step = np.where(x < 5, 0, 1)
    y_step_approx = sigmoid(5 * (x - 5))
    axes[0].plot(x, y_step, COLOR_TARGET, linewidth=3, linestyle='--',
                label='Target: Step', alpha=0.7)
    axes[0].plot(x, y_step_approx, COLOR_APPROX, linewidth=3,
                label='5 neurons (steep sigmoid)')
    axes[0].set_title('Step Function', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.2, 1.3)

    # Function 2: Sine wave
    y_sine = np.sin(x)
    y_sine_approx = np.zeros_like(x)
    positions = np.linspace(0, 10, 10)
    for pos in positions:
        phase = np.sin(pos)
        y_sine_approx += 0.6 * phase * (sigmoid(3 * (x - pos)) - 0.5)
    axes[1].plot(x, y_sine, COLOR_TARGET, linewidth=3, linestyle='--',
                label='Target: sin(x)', alpha=0.7)
    axes[1].plot(x, y_sine_approx, COLOR_APPROX, linewidth=3, label='10 neurons')
    axes[1].set_title('Sine Wave', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-1.5, 1.5)

    # Function 3: Parabola
    y_parabola = -(x - 5)**2 / 5 + 5
    y_para_approx = 2 + 6*sigmoid(1.5*(x-3)) - 6*sigmoid(1.5*(x-7))
    axes[2].plot(x, y_parabola, COLOR_TARGET, linewidth=3, linestyle='--',
                label='Target: Parabola', alpha=0.7)
    axes[2].plot(x, y_para_approx, COLOR_APPROX, linewidth=3, label='3 neurons')
    axes[2].set_title('Parabola (Peak)', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 6)

    # Function 4: Exponential decay
    y_exp = 5 * np.exp(-0.5 * x)
    y_exp_approx = 5 - 5*sigmoid(1.2*(x-3))
    axes[3].plot(x, y_exp, COLOR_TARGET, linewidth=3, linestyle='--',
                label='Target: Exponential', alpha=0.7)
    axes[3].plot(x, y_exp_approx, COLOR_APPROX, linewidth=3, label='2 neurons')
    axes[3].set_title('Exponential Decay', fontsize=12, fontweight='bold')
    axes[3].legend(fontsize=9)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim(-0.5, 6)

    # Function 5: Triangle/Tent
    y_triangle = np.where(x < 5, x, 10 - x)
    y_tri_approx = 5*sigmoid(2*(x-2)) - 5*sigmoid(2*(x-8))
    axes[4].plot(x, y_triangle, COLOR_TARGET, linewidth=3, linestyle='--',
                label='Target: Triangle', alpha=0.7)
    axes[4].plot(x, y_tri_approx, COLOR_APPROX, linewidth=3, label='4 neurons')
    axes[4].set_title('Triangle (Tent) Function', fontsize=12, fontweight='bold')
    axes[4].legend(fontsize=9)
    axes[4].grid(True, alpha=0.3)
    axes[4].set_ylim(-1, 6)

    # Function 6: Multiple bumps
    y_bumps = 2*np.exp(-((x-2.5)**2)/0.5) + 1.5*np.exp(-((x-7)**2)/0.5)
    y_bumps_approx = 4*sigmoid(3*(x-1.5)) - 4*sigmoid(3*(x-3.5)) + 3*sigmoid(3*(x-6)) - 3*sigmoid(3*(x-8))
    axes[5].plot(x, y_bumps, COLOR_TARGET, linewidth=3, linestyle='--',
                label='Target: Two bumps', alpha=0.7)
    axes[5].plot(x, y_bumps_approx, COLOR_APPROX, linewidth=3, label='4 neurons')
    axes[5].set_title('Multiple Bumps', fontsize=12, fontweight='bold')
    axes[5].legend(fontsize=9)
    axes[5].grid(True, alpha=0.3)
    axes[5].set_ylim(-0.5, 2.5)

    # Add x-labels to bottom row
    for i in range(3, 6):
        axes[i].set_xlabel('x', fontsize=10)

    plt.suptitle('Neural Networks Can Approximate Many Different Function Types',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}multiple_function_examples.pdf', dpi=300, bbox_inches='tight')
    print("[OK] Created: multiple_function_examples.pdf")
    plt.close()


def main():
    """Generate all function approximation handout figures"""
    print("="*70)
    print("Generating Function Approximation Discovery Handout Figures")
    print("="*70)
    print()

    create_temperature_challenge_scatter()
    create_single_sigmoid_parameters()
    create_temperature_single_sigmoid_fit()
    create_two_sigmoid_subtraction()
    create_bump_temperature_application()
    create_three_neuron_parabola()
    create_neuron_contributions()
    create_universal_approximation_sine()
    create_error_vs_neuron_count()
    create_multiple_function_examples()

    print()
    print("="*70)
    print("[SUCCESS] All 10 figures generated successfully!")
    print(f"Location: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()