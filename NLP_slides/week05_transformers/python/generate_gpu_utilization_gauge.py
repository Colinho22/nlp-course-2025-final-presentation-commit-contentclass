"""
Generate GPU Utilization Gauge Charts
Week 5 Transformers - For Slide 21 (Experimental Validation)
Speedometer-style gauges showing before/after comparison
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge, Circle, FancyBboxPatch
import numpy as np

# Template colors
COLOR_MAIN = '#333366'
COLOR_RED = '#D62728'
COLOR_ORANGE = '#FF7F0E'
COLOR_GREEN = '#44A044'
COLOR_LIGHT_BG = '#F0F0F0'

# Create figure
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor('white')
fig.suptitle('GPU Utilization: Before and After Transformers', fontsize=14, fontweight='bold', y=0.96)

# Helper function to draw gauge
def draw_gauge(ax, value, max_val, title, color, subtitle=''):
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')

    # Background arc (gray)
    theta = np.linspace(0, np.pi, 100)
    x_bg = np.cos(theta)
    y_bg = np.sin(theta)
    ax.fill_between(x_bg, 0, y_bg, color='#CCCCCC', alpha=0.3)

    # Colored zones
    # Red zone: 0-30%
    theta_red = np.linspace(0, np.pi * 0.3, 50)
    x_red = np.cos(theta_red)
    y_red = np.sin(theta_red)
    ax.fill_between(x_red, 0, y_red, color='#FFE6E6', alpha=0.8)

    # Orange zone: 30-70%
    theta_orange = np.linspace(np.pi * 0.3, np.pi * 0.7, 50)
    x_orange = np.cos(theta_orange)
    y_orange = np.sin(theta_orange)
    ax.fill_between(x_orange, 0, y_orange, color='#FFE6CC', alpha=0.8)

    # Green zone: 70-100%
    theta_green = np.linspace(np.pi * 0.7, np.pi, 50)
    x_green = np.cos(theta_green)
    y_green = np.sin(theta_green)
    ax.fill_between(x_green, 0, y_green, color='#E6FFE6', alpha=0.8)

    # Needle
    angle = np.pi * (1 - value / max_val)
    needle_x = [0, 0.9 * np.cos(angle)]
    needle_y = [0, 0.9 * np.sin(angle)]
    ax.plot(needle_x, needle_y, color=color, linewidth=6)
    ax.plot(needle_x, needle_y, 'ko-', markersize=8)

    # Center circle
    center = Circle((0, 0), 0.08, facecolor='black', zorder=10)
    ax.add_patch(center)

    # Value text
    ax.text(0, -0.25, f'{value}%', ha='center', va='center',
            fontsize=24, fontweight='bold', color=color)

    # Title
    ax.text(0, 1.25, title, ha='center', va='center',
            fontsize=12, fontweight='bold', color=color)

    # Subtitle
    if subtitle:
        ax.text(0, 1.05, subtitle, ha='center', va='center',
                fontsize=9, color=color)

    # Tick marks
    for pct in [0, 25, 50, 75, 100]:
        angle_tick = np.pi * (1 - pct / 100)
        x_tick = 1.1 * np.cos(angle_tick)
        y_tick = 1.1 * np.sin(angle_tick)
        ax.text(x_tick, y_tick, f'{pct}', ha='center', va='center',
                fontsize=8, fontweight='bold')

# TOP ROW: GPU Utilization Gauges
ax_rnn = plt.subplot(2, 2, 1)
draw_gauge(ax_rnn, 2, 100, 'RNN (Sequential)', COLOR_RED, 'GPU mostly idle')

ax_transformer = plt.subplot(2, 2, 2)
draw_gauge(ax_transformer, 92, 100, 'Transformer (Parallel)', COLOR_GREEN, 'Full GPU power!')

# BOTTOM ROW: Training Time and Cost
ax_bottom_left = plt.subplot(2, 2, 3)
ax_bottom_left.set_xlim(0, 10)
ax_bottom_left.set_ylim(0, 10)
ax_bottom_left.axis('off')
ax_bottom_left.set_title('Training Time Reduction', fontsize=12, fontweight='bold',
                          color=COLOR_MAIN, pad=10)

# Before bar
before_bar = FancyBboxPatch((1, 3), 3, 4,
                            boxstyle="round,pad=0.1",
                            facecolor=COLOR_RED, edgecolor=COLOR_MAIN,
                            linewidth=2, alpha=0.5)
ax_bottom_left.add_patch(before_bar)
ax_bottom_left.text(2.5, 5, '90 days', ha='center', va='center',
                    fontsize=16, fontweight='bold', color='white')
ax_bottom_left.text(2.5, 2, 'RNN', ha='center', va='center',
                    fontsize=11, fontweight='bold')

# After bar
after_bar = FancyBboxPatch((6, 3), 3, 0.4,
                           boxstyle="round,pad=0.1",
                           facecolor=COLOR_GREEN, edgecolor=COLOR_MAIN,
                           linewidth=2, alpha=0.5)
ax_bottom_left.add_patch(after_bar)
ax_bottom_left.text(7.5, 3.2, '1 day', ha='center', va='center',
                    fontsize=16, fontweight='bold', color='white')
ax_bottom_left.text(7.5, 2, 'Transformer', ha='center', va='center',
                    fontsize=11, fontweight='bold')

# 90x speedup label
ax_bottom_left.text(5, 8.5, '90x Faster!', ha='center', va='center',
                    fontsize=14, fontweight='bold', color=COLOR_GREEN,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#E6FFE6',
                             edgecolor=COLOR_GREEN, linewidth=2))

ax_bottom_right = plt.subplot(2, 2, 4)
ax_bottom_right.set_xlim(0, 10)
ax_bottom_right.set_ylim(0, 10)
ax_bottom_right.axis('off')
ax_bottom_right.set_title('Training Cost Reduction', fontsize=12, fontweight='bold',
                           color=COLOR_MAIN, pad=10)

# Before cost
before_cost = FancyBboxPatch((1, 3), 3, 4,
                             boxstyle="round,pad=0.1",
                             facecolor=COLOR_RED, edgecolor=COLOR_MAIN,
                             linewidth=2, alpha=0.5)
ax_bottom_right.add_patch(before_cost)
ax_bottom_right.text(2.5, 5, '$45K', ha='center', va='center',
                     fontsize=16, fontweight='bold', color='white')
ax_bottom_right.text(2.5, 2, 'RNN', ha='center', va='center',
                     fontsize=11, fontweight='bold')

# After cost
after_cost = FancyBboxPatch((6, 3), 3, 0.1,
                            boxstyle="round,pad=0.1",
                            facecolor=COLOR_GREEN, edgecolor=COLOR_MAIN,
                            linewidth=2, alpha=0.5)
ax_bottom_right.add_patch(after_cost)
ax_bottom_right.text(7.5, 3.05, '$500', ha='center', va='center',
                     fontsize=16, fontweight='bold', color='white')
ax_bottom_right.text(7.5, 2, 'Transformer', ha='center', va='center',
                     fontsize=11, fontweight='bold')

# Cost reduction label
ax_bottom_right.text(5, 8.5, '90x Cheaper!', ha='center', va='center',
                     fontsize=14, fontweight='bold', color=COLOR_GREEN,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#E6FFE6',
                              edgecolor=COLOR_GREEN, linewidth=2))

plt.tight_layout(rect=[0, 0, 1, 0.94])

# Save figure
output_path = '../figures/sr_23_gpu_utilization_gauge.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')

plt.close()
