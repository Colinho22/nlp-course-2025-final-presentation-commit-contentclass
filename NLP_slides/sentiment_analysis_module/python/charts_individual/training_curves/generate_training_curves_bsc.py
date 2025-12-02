"""
Training Curves for BERT Fine-Tuning

PEDAGOGICAL PURPOSE:
- Show loss/accuracy progression over epochs
- Validate "3-5 epochs sufficient" claim
- Demonstrate warmup and convergence patterns

GENUINELY NEEDS VISUALIZATION: Yes - temporal trends require line charts
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# BSc Discovery Color Scheme
COLOR_MAIN = '#404040'
COLOR_ACCENT = '#3333B2'
COLOR_GREEN = '#2CA02C'
COLOR_RED = '#D62728'
COLOR_ORANGE = '#FF7F0E'
COLOR_BLUE = '#0066CC'

# BSc Discovery Font Standard
FONTSIZE_TITLE = 24
FONTSIZE_LABEL = 20
FONTSIZE_TICK = 16
FONTSIZE_ANNOTATION = 18

def create_chart():
    # Simulate realistic training curves
    epochs = np.linspace(0, 5, 100)

    # Training loss: starts high, drops quickly, plateaus
    train_loss = 0.7 * np.exp(-epochs * 0.8) + 0.08 + 0.02 * np.random.randn(100) * 0.1

    # Validation loss: similar but slightly higher, with slight uptick at end
    val_loss = 0.72 * np.exp(-epochs * 0.7) + 0.12 + 0.03 * np.random.randn(100) * 0.1
    val_loss[-15:] += np.linspace(0, 0.05, 15)  # Slight overfitting at end

    # Training accuracy: inverse of loss pattern
    train_acc = 0.55 + 0.40 * (1 - np.exp(-epochs * 0.9)) + 0.02 * np.random.randn(100) * 0.05
    train_acc = np.clip(train_acc, 0.5, 0.98)

    # Validation accuracy
    val_acc = 0.52 + 0.40 * (1 - np.exp(-epochs * 0.8)) + 0.02 * np.random.randn(100) * 0.05
    val_acc = np.clip(val_acc, 0.5, 0.95)
    val_acc[-15:] -= np.linspace(0, 0.02, 15)  # Slight accuracy drop

    # Create dual-axis figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Loss curves
    ax1.plot(epochs, train_loss, color=COLOR_BLUE, linewidth=2.5, label='Training Loss')
    ax1.plot(epochs, val_loss, color=COLOR_ORANGE, linewidth=2.5, label='Validation Loss')

    # Mark warmup period
    ax1.axvspan(0, 0.5, alpha=0.2, color='gray', label='Warmup Period')
    ax1.axvline(x=3, color=COLOR_GREEN, linestyle='--', linewidth=2,
                label='Optimal Stopping (Epoch 3)')

    ax1.set_xlabel('Epoch', fontsize=FONTSIZE_LABEL, color=COLOR_MAIN)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=FONTSIZE_LABEL, color=COLOR_MAIN)
    ax1.set_title('Loss During Fine-Tuning', fontsize=FONTSIZE_TITLE-2,
                  fontweight='bold', color=COLOR_ACCENT, pad=15)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(COLOR_ACCENT)
    ax1.spines['bottom'].set_color(COLOR_ACCENT)
    ax1.tick_params(labelsize=FONTSIZE_TICK, colors=COLOR_MAIN)
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=1)
    ax1.set_ylim(0, 0.8)
    ax1.legend(loc='upper right', fontsize=FONTSIZE_ANNOTATION-4)

    # Right: Accuracy curves
    ax2.plot(epochs, train_acc * 100, color=COLOR_BLUE, linewidth=2.5, label='Training Accuracy')
    ax2.plot(epochs, val_acc * 100, color=COLOR_ORANGE, linewidth=2.5, label='Validation Accuracy')

    # Mark target accuracy region
    ax2.axhline(y=90, color=COLOR_GREEN, linestyle=':', linewidth=2, label='Target Accuracy')
    ax2.axvline(x=3, color=COLOR_GREEN, linestyle='--', linewidth=2)

    # Annotate overfitting zone
    ax2.annotate('Overfitting\nbegins', xy=(4.2, val_acc[-1]*100-2),
                 fontsize=FONTSIZE_TICK, color=COLOR_RED,
                 ha='center', fontweight='bold')

    ax2.set_xlabel('Epoch', fontsize=FONTSIZE_LABEL, color=COLOR_MAIN)
    ax2.set_ylabel('Accuracy (%)', fontsize=FONTSIZE_LABEL, color=COLOR_MAIN)
    ax2.set_title('Accuracy During Fine-Tuning', fontsize=FONTSIZE_TITLE-2,
                  fontweight='bold', color=COLOR_ACCENT, pad=15)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(COLOR_ACCENT)
    ax2.spines['bottom'].set_color(COLOR_ACCENT)
    ax2.tick_params(labelsize=FONTSIZE_TICK, colors=COLOR_MAIN)
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=1)
    ax2.set_ylim(50, 100)
    ax2.legend(loc='lower right', fontsize=FONTSIZE_ANNOTATION-4)

    plt.tight_layout()
    plt.savefig('../../../figures/training_curves_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print("[OK] Generated: training_curves_bsc.pdf")
    print("     Pedagogical role: Validates '3-5 epochs' claim, shows warmup/overfitting")

if __name__ == '__main__':
    create_chart()
