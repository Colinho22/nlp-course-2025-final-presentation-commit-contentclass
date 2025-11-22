"""
Generate BSc-level charts for Week 10: Fine-tuning & Prompt Engineering
Educational framework: Zero pre-knowledge, discovery-based pedagogy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrow, Ellipse, Wedge, Arc
from matplotlib.path import Path
import matplotlib.patheffects as path_effects
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Color scheme (matching educational framework)
MLPURPLE = '#3333B2'    # Primary accent
DARKGRAY = '#404040'    # Main text
MIDGRAY = '#B4B4B4'     # Secondary elements
LIGHTGRAY = '#F0F0F0'   # Backgrounds

# Educational color scheme
COLOR_QUERY = '#FF6B6B'    # Red for query/focus
COLOR_KEY = '#4ECDC4'      # Teal for keys/context
COLOR_VALUE = '#95E77E'    # Green for values/output
COLOR_ATTENTION = '#FFE66D' # Yellow for attention weights

# Create figures directory
os.makedirs('../figures', exist_ok=True)

def set_minimalist_style(ax):
    """Apply minimalist style to axis"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(MIDGRAY)
    ax.spines['bottom'].set_color(MIDGRAY)
    ax.tick_params(colors=DARKGRAY, which='both')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_facecolor('white')


# 1. Cost vs Performance Scatter (Enhanced Hook)
def plot_cost_performance_scatter():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Methods with (cost, performance, flexibility, label)
    methods = [
        ('Zero-shot\nPrompting', 0, 60, 1000, 'green'),
        ('Few-shot\nPrompting', 5, 75, 1200, 'lightgreen'),
        ('Prompt\nEngineering', 10, 78, 1100, 'yellow'),
        ('LoRA\nFine-tuning', 500, 93, 2000, 'orange'),
        ('Full\nFine-tuning', 50000, 95, 1500, 'red'),
        ('RLHF', 100000, 97, 1000, 'darkred')
    ]

    for name, cost, perf, size, color in methods:
        ax.scatter(cost, perf, s=size, c=color, alpha=0.7,
                  edgecolors='black', linewidth=2, zorder=3)

        # Annotate
        if cost == 0:
            ax.annotate(name, (cost, perf), xytext=(15, 15),
                       textcoords='offset points', ha='left',
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
        elif cost < 100:
            ax.annotate(name, (cost, perf), xytext=(0, -35),
                       textcoords='offset points', ha='center',
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
        else:
            ax.annotate(name, (cost, perf), xytext=(0, 15),
                       textcoords='offset points', ha='center',
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))

    # Highlight sweet spot
    sweet_ellipse = Ellipse((500, 93), 1000, 8, facecolor='lightblue',
                            alpha=0.2, edgecolor='blue', linewidth=2, linestyle='--')
    ax.add_patch(sweet_ellipse)
    ax.text(500, 86, 'Sweet Spot:\n0.1% cost, 98% performance',
            ha='center', fontsize=12, fontweight='bold', color='blue',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    # Axes
    ax.set_xlabel('Cost (USD)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (Accuracy %)', fontsize=14, fontweight='bold')
    ax.set_title('The $50,000 Question Has a $500 Answer\nModel Adaptation: Cost vs Performance',
                fontsize=16, fontweight='bold', pad=15)

    ax.set_xscale('symlog')
    ax.set_xlim(-10, 200000)
    ax.set_ylim(55, 100)

    set_minimalist_style(ax)

    # Add insight box
    ax.text(0.98, 0.02, 'Key Insight: LoRA achieves 93% accuracy (98% of full FT) at 1% of the cost',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow'))

    plt.tight_layout()
    plt.savefig('../figures/cost_performance_scatter_enhanced.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("1/8: cost_performance_scatter_enhanced.pdf created")


# 2. Transfer Learning Visual
def plot_transfer_learning_visual():
    fig, ax = plt.subplots(figsize=(14, 8))

    # Base model (frozen)
    base_rect = FancyBboxPatch((0.05, 0.3), 0.35, 0.5,
                               boxstyle="round,pad=0.02",
                               facecolor='lightblue',
                               edgecolor='black',
                               linewidth=3)
    ax.add_patch(base_rect)
    ax.text(0.225, 0.55, 'Pre-trained Model\n(175B parameters)',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # Knowledge boxes inside
    knowledge = ['Grammar', 'Facts', 'Reasoning', 'Common Sense']
    y_positions = [0.7, 0.6, 0.5, 0.4]
    for know, y in zip(knowledge, y_positions):
        ax.text(0.225, y, f'✓ {know}', ha='center', va='center',
                fontsize=11, color='darkblue')

    # Freeze symbol
    ax.text(0.225, 0.35, '🔒 FROZEN', ha='center', va='center',
            fontsize=12, fontweight='bold', color='blue')

    # Arrow
    arrow = FancyArrow(0.42, 0.55, 0.13, 0, width=0.03,
                      head_width=0.06, head_length=0.03,
                      facecolor=MLPURPLE, edgecolor='black', linewidth=2)
    ax.add_patch(arrow)
    ax.text(0.485, 0.6, 'Transfer', ha='center', fontsize=12,
            fontweight='bold', color=MLPURPLE)

    # Task-specific adaptation (trainable)
    task_rect = FancyBboxPatch((0.6, 0.3), 0.35, 0.5,
                               boxstyle="round,pad=0.02",
                               facecolor='lightcoral',
                               edgecolor='black',
                               linewidth=3)
    ax.add_patch(task_rect)
    ax.text(0.775, 0.55, 'Task-Specific\nAdaptation',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # New knowledge
    new_knowledge = ['Medical Terms', 'Legal Jargon', 'Code Syntax', 'Domain Facts']
    for know, y in zip(new_knowledge, y_positions):
        ax.text(0.775, y, f'+ {know}', ha='center', va='center',
                fontsize=11, color='darkred')

    # Trainable symbol
    ax.text(0.775, 0.35, '🔓 TRAINABLE', ha='center', va='center',
            fontsize=12, fontweight='bold', color='red')

    # Bottom comparison
    comparison_box = FancyBboxPatch((0.1, 0.05), 0.8, 0.15,
                                    boxstyle="round,pad=0.02",
                                    facecolor='lightyellow',
                                    edgecolor='black',
                                    linewidth=2)
    ax.add_patch(comparison_box)

    ax.text(0.3, 0.125, 'Traditional Approach:\nTrain from scratch\n175B params, $5M+',
            ha='center', va='center', fontsize=10, color='black')
    ax.text(0.5, 0.125, '→', ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(0.7, 0.125, 'Transfer Learning:\nFreeze base, adapt top\n0.1-10% params, $100-$5K',
            ha='center', va='center', fontsize=10, color='green', fontweight='bold')

    # Title
    ax.text(0.5, 0.95, 'Transfer Learning: Reuse General Knowledge, Adapt for Specifics',
            ha='center', va='center', fontsize=16, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/transfer_learning_visual.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("2/8: transfer_learning_visual.pdf created")


# 3. Parameter Update Spectrum
def plot_parameter_spectrum():
    fig, ax = plt.subplots(figsize=(14, 6))

    # Spectrum bar
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap='RdYlGn_r', extent=[0, 100, 0, 1])

    # Methods on spectrum
    methods = [
        (0, 'Zero-Shot\n0%\nNo training', 'green'),
        (5, 'Few-Shot\n0%\nNo training', 'lightgreen'),
        (10, 'Prompting\n0%\nNo training', 'yellow'),
        (30, 'Adapters\n0.5-2%\nFew layers', 'orange'),
        (50, 'LoRA\n0.1-1%\nLow-rank', 'darkorange'),
        (85, 'Full FT\n100%\nAll weights', 'red')
    ]

    for x, label, color in methods:
        # Vertical line
        ax.plot([x, x], [1, 1.5], 'k-', linewidth=2)

        # Marker
        ax.scatter(x, 1, s=300, c=color, edgecolors='black',
                  linewidth=2, zorder=3, marker='D')

        # Label
        ax.text(x, 1.8, label, ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))

    # Axes
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.2, 2.5)
    ax.set_xlabel('Parameters Updated (%)', fontsize=14, fontweight='bold')
    ax.set_title('The Parameter Update Spectrum: From Zero Training to Full Fine-tuning',
                fontsize=16, fontweight='bold', pad=20)

    # Remove y-axis
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Add zones
    ax.axvspan(0, 15, alpha=0.1, color='green')
    ax.text(7.5, -0.15, 'No Training Zone\n(Prompt Engineering)',
            ha='center', fontsize=9, style='italic', color='green')

    ax.axvspan(15, 70, alpha=0.1, color='orange')
    ax.text(42.5, -0.15, 'Efficient Zone\n(PEFT Methods)',
            ha='center', fontsize=9, style='italic', color='darkorange')

    ax.axvspan(70, 100, alpha=0.1, color='red')
    ax.text(85, -0.15, 'Full Update Zone\n(Traditional Fine-tuning)',
            ha='center', fontsize=9, style='italic', color='red')

    plt.tight_layout()
    plt.savefig('../figures/parameter_spectrum.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("3/8: parameter_spectrum.pdf created")


# 4. Data-Performance Curves
def plot_data_performance_curves():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Data points (examples)
    data_sizes = np.array([0, 10, 50, 100, 500, 1000, 5000, 10000])

    # Performance curves (realistic)
    zero_shot = np.array([60, 60, 60, 60, 60, 60, 60, 60])
    few_shot = np.array([60, 70, 75, 76, 76, 76, 76, 76])
    lora = np.array([60, 65, 75, 82, 88, 91, 93, 94])
    full_ft = np.array([60, 60, 70, 78, 87, 92, 95, 96])

    # Plot curves
    ax.plot(data_sizes, zero_shot, 'o-', linewidth=3, label='Zero-Shot Prompting',
            color='green', markersize=8)
    ax.plot(data_sizes, few_shot, 's-', linewidth=3, label='Few-Shot Prompting',
            color='lightgreen', markersize=8)
    ax.plot(data_sizes, lora, 'D-', linewidth=3, label='LoRA Fine-tuning',
            color='orange', markersize=8)
    ax.plot(data_sizes, full_ft, '^-', linewidth=3, label='Full Fine-tuning',
            color='red', markersize=8)

    # Highlight zones
    ax.axvspan(0, 100, alpha=0.1, color='green')
    ax.text(50, 98, 'Prompting Zone\n(<100 examples)', ha='center',
            fontsize=10, fontweight='bold', color='green')

    ax.axvspan(100, 1000, alpha=0.1, color='orange')
    ax.text(550, 98, 'LoRA Sweet Spot\n(100-1K examples)', ha='center',
            fontsize=10, fontweight='bold', color='darkorange')

    ax.axvspan(1000, 10000, alpha=0.1, color='red')
    ax.text(5500, 98, 'Full FT Zone\n(1K+ examples)', ha='center',
            fontsize=10, fontweight='bold', color='darkred')

    # Axes
    ax.set_xlabel('Number of Training Examples', fontsize=14, fontweight='bold')
    ax.set_ylabel('Task Performance (Accuracy %)', fontsize=14, fontweight='bold')
    ax.set_title('Data Efficiency: Performance vs Training Data Size',
                fontsize=16, fontweight='bold', pad=15)

    ax.set_xscale('symlog')
    ax.set_xlim(-5, 12000)
    ax.set_ylim(55, 100)

    ax.legend(loc='lower right', fontsize=12)
    set_minimalist_style(ax)

    # Add insight
    ax.text(0.02, 0.98, 'Key Insight: LoRA excels with 100-1K examples\nFull FT needs 1K+ to avoid overfitting',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow'))

    plt.tight_layout()
    plt.savefig('../figures/data_performance_curves.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("4/8: data_performance_curves.pdf created")


# 5. Adapter Architecture
def plot_adapter_architecture():
    fig, ax = plt.subplots(figsize=(10, 12))

    # Transformer layers (simplified)
    layer_height = 0.12
    layer_width = 0.4
    n_layers = 6

    for i in range(n_layers):
        y = 0.15 + i * 0.13

        # Main transformer layer (frozen)
        main_rect = FancyBboxPatch((0.3, y), layer_width, layer_height,
                                   boxstyle="round,pad=0.01",
                                   facecolor='lightblue',
                                   edgecolor='black',
                                   linewidth=2)
        ax.add_patch(main_rect)
        ax.text(0.5, y + layer_height/2, f'Transformer Layer {i+1}\n(FROZEN)',
                ha='center', va='center', fontsize=9, fontweight='bold')

        # Adapter (trainable) - only on some layers
        if i % 2 == 0:  # Adapters on even layers
            adapter_rect = FancyBboxPatch((0.15, y + 0.02), 0.1, layer_height - 0.04,
                                         boxstyle="round,pad=0.01",
                                         facecolor='lightcoral',
                                         edgecolor='black',
                                         linewidth=2)
            ax.add_patch(adapter_rect)
            ax.text(0.2, y + layer_height/2, 'Adapt\n✓',
                    ha='center', va='center', fontsize=8, fontweight='bold', color='red')

            # Connection arrows
            ax.arrow(0.25, y + layer_height/2, 0.03, 0,
                    head_width=0.015, head_length=0.01, fc='red', ec='red')

    # Input/Output
    ax.text(0.5, 0.05, 'Input', ha='center', fontsize=12, fontweight='bold')
    ax.arrow(0.5, 0.08, 0, 0.05, head_width=0.02, head_length=0.01,
             fc='black', ec='black', linewidth=2)

    ax.text(0.5, 0.95, 'Output', ha='center', fontsize=12, fontweight='bold')
    ax.arrow(0.5, 0.93, 0, -0.05, head_width=0.02, head_length=0.01,
             fc='black', ec='black', linewidth=2)

    # Legend
    legend_box = FancyBboxPatch((0.75, 0.3), 0.2, 0.3,
                                boxstyle="round,pad=0.02",
                                facecolor='lightyellow',
                                edgecolor='black',
                                linewidth=2)
    ax.add_patch(legend_box)

    ax.text(0.85, 0.55, 'Legend:', ha='center', fontsize=11, fontweight='bold')

    # Frozen indicator
    frozen_mini = FancyBboxPatch((0.77, 0.48), 0.06, 0.03,
                                 facecolor='lightblue',
                                 edgecolor='black',
                                 linewidth=1)
    ax.add_patch(frozen_mini)
    ax.text(0.88, 0.495, 'Frozen\n(0 train)', ha='left', va='center', fontsize=9)

    # Trainable indicator
    train_mini = FancyBboxPatch((0.77, 0.42), 0.06, 0.03,
                                facecolor='lightcoral',
                                edgecolor='black',
                                linewidth=1)
    ax.add_patch(train_mini)
    ax.text(0.88, 0.435, 'Trainable\n(adapt)', ha='left', va='center', fontsize=9)

    # Stats
    ax.text(0.85, 0.35, 'Parameters:\nBase: 99%\nAdapters: 1%',
            ha='center', fontsize=9, style='italic')

    # Title
    ax.text(0.5, 0.98, 'Adapter Architecture: Small Trainable Modules\nInserted Between Frozen Layers',
            ha='center', va='top', fontsize=14, fontweight='bold')

    # Bottom insight
    insight_box = FancyBboxPatch((0.1, 0.02), 0.8, 0.06,
                                 boxstyle="round,pad=0.01",
                                 facecolor='lightgreen',
                                 edgecolor='black',
                                 linewidth=2)
    ax.add_patch(insight_box)
    ax.text(0.5, 0.05, 'Key: Freeze expensive pre-trained weights, train cheap adapters (1-2% parameters)',
            ha='center', va='center', fontsize=10, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/adapter_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("5/8: adapter_architecture.pdf created")


# 6. Catastrophic Forgetting
def plot_catastrophic_forgetting():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Training steps
    steps = np.linspace(0, 100, 100)

    # Performance curves
    # Target task performance
    target_frozen = np.full_like(steps, 60)  # Stays low
    target_lora = 60 + 35 * (1 - np.exp(-steps/20))  # Improves steadily
    target_full = 60 + 35 * (1 - np.exp(-steps/15))  # Improves faster

    # Original task performance
    original_frozen = np.full_like(steps, 90)  # Stays high
    original_lora = 90 - 5 * (1 - np.exp(-steps/30))  # Slight degradation
    original_full = 90 - 30 * (1 - np.exp(-steps/15))  # Catastrophic forgetting

    # Plot
    ax.plot(steps, target_frozen, '--', linewidth=2, label='Target (Frozen)',
            color='lightblue', alpha=0.7)
    ax.plot(steps, target_lora, '-', linewidth=3, label='Target (LoRA)',
            color='orange')
    ax.plot(steps, target_full, '-', linewidth=3, label='Target (Full FT)',
            color='red')

    ax.plot(steps, original_frozen, '--', linewidth=2, label='Original (Frozen)',
            color='lightgreen', alpha=0.7)
    ax.plot(steps, original_lora, '-', linewidth=3, label='Original (LoRA)',
            color='green')
    ax.plot(steps, original_full, '-', linewidth=3, label='Original (Full FT)',
            color='darkred')

    # Highlight catastrophic forgetting zone
    ax.axvspan(50, 100, alpha=0.15, color='red')
    ax.text(75, 70, 'Catastrophic\nForgetting Zone', ha='center',
            fontsize=11, fontweight='bold', color='darkred',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Annotations
    ax.annotate('LoRA: Minimal forgetting', xy=(100, 85), xytext=(80, 92),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=11, fontweight='bold', color='green')

    ax.annotate('Full FT: Loses 30%\non original task', xy=(100, 60), xytext=(70, 50),
                arrowprops=dict(arrowstyle='->', lw=2, color='darkred'),
                fontsize=11, fontweight='bold', color='darkred')

    # Axes
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (Accuracy %)', fontsize=14, fontweight='bold')
    ax.set_title('Catastrophic Forgetting: Full Fine-tuning vs LoRA\nOriginal Task Performance Degrades with Full Updates',
                fontsize=16, fontweight='bold', pad=15)

    ax.set_xlim(0, 100)
    ax.set_ylim(50, 100)
    ax.legend(loc='center left', fontsize=11)
    set_minimalist_style(ax)

    # Insight box
    ax.text(0.98, 0.02, 'Key Insight: LoRA preserves original capabilities\nFull FT overwrites pre-trained knowledge',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow'))

    plt.tight_layout()
    plt.savefig('../figures/catastrophic_forgetting.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("6/8: catastrophic_forgetting.pdf created")


# 7. Applications 2024
def plot_applications_2024():
    fig, ax = plt.subplots(figsize=(14, 10))

    # Applications with method and results
    applications = [
        {
            'name': 'Bloomberg GPT',
            'method': 'Full Fine-tuning',
            'domain': 'Finance',
            'data': '363B tokens',
            'cost': '$2.7M+',
            'result': '50B params, SOTA finance',
            'y': 0.85,
            'color': 'lightcoral'
        },
        {
            'name': 'Med-PaLM 2',
            'method': 'Instruction Tuning',
            'domain': 'Medical',
            'data': 'Medical QA datasets',
            'cost': '$500K+',
            'result': '85% on USMLE',
            'y': 0.7,
            'color': 'lightblue'
        },
        {
            'name': 'Code Llama',
            'method': 'LoRA',
            'domain': 'Programming',
            'data': '500B code tokens',
            'cost': '$50K',
            'result': '53% HumanEval',
            'y': 0.55,
            'color': 'lightgreen'
        },
        {
            'name': 'GPT-4 Custom',
            'method': 'Prompt Engineering',
            'domain': 'Customer Service',
            'data': '0 training',
            'cost': '$0',
            'result': '90% satisfaction',
            'y': 0.4,
            'color': 'lightyellow'
        },
        {
            'name': 'LegalBERT',
            'method': 'Domain Fine-tuning',
            'domain': 'Legal',
            'data': '12GB legal docs',
            'cost': '$10K',
            'result': '89% on legal NER',
            'y': 0.25,
            'color': 'lavender'
        },
        {
            'name': 'Llama-2-Chat',
            'method': 'RLHF',
            'domain': 'Conversational',
            'data': '27K preference pairs',
            'cost': '$100K+',
            'result': 'Human-preferred',
            'y': 0.1,
            'color': 'peachpuff'
        }
    ]

    for app in applications:
        # Box
        box = FancyBboxPatch((0.05, app['y']-0.05), 0.9, 0.12,
                            boxstyle="round,pad=0.02",
                            facecolor=app['color'],
                            edgecolor='black',
                            linewidth=2)
        ax.add_patch(box)

        # Name
        ax.text(0.08, app['y']+0.04, app['name'], ha='left', va='center',
                fontsize=13, fontweight='bold')

        # Method badge
        ax.text(0.08, app['y'], f"[{app['method']}]", ha='left', va='center',
                fontsize=10, style='italic', color='darkblue')

        # Domain
        ax.text(0.35, app['y']+0.04, f"Domain: {app['domain']}", ha='left', va='center',
                fontsize=11)

        # Data
        ax.text(0.35, app['y']+0.01, f"Data: {app['data']}", ha='left', va='center',
                fontsize=10)

        # Cost
        ax.text(0.35, app['y']-0.02, f"Cost: {app['cost']}", ha='left', va='center',
                fontsize=10, fontweight='bold', color='red')

        # Result
        ax.text(0.75, app['y'], f"Result:\n{app['result']}", ha='center', va='center',
                fontsize=10, fontweight='bold', color='green',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Title
    ax.text(0.5, 0.98, 'Real-World Applications 2024: Model Adaptation in Production',
            ha='center', va='top', fontsize=16, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/applications_2024.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("7/8: applications_2024.pdf created")


# 8. Prompt Anatomy
def plot_prompt_anatomy():
    fig, ax = plt.subplots(figsize=(12, 10))

    # Prompt components with colors
    components = [
        {
            'name': '1. Role Definition',
            'content': 'You are an expert data scientist with 10 years\nof experience in machine learning and statistics.',
            'y': 0.85,
            'height': 0.1,
            'color': 'lightblue',
            'purpose': 'Sets expertise context'
        },
        {
            'name': '2. Task Specification',
            'content': 'Your task is to analyze this customer churn dataset\nand identify the top 3 factors driving customer loss.',
            'y': 0.72,
            'height': 0.1,
            'color': 'lightgreen',
            'purpose': 'Defines clear objective'
        },
        {
            'name': '3. Context Provision',
            'content': 'Background: E-commerce company, 50K customers, 15% churn rate.\nConstraints: Must complete analysis in 2 hours.',
            'y': 0.59,
            'height': 0.1,
            'color': 'lightyellow',
            'purpose': 'Provides relevant details'
        },
        {
            'name': '4. Output Format',
            'content': 'Provide your response in this format:\n- Finding 1: [Factor] (Impact: X%)\n- Finding 2: [Factor] (Impact: X%)\n- Finding 3: [Factor] (Impact: X%)',
            'y': 0.43,
            'height': 0.13,
            'color': 'lightcoral',
            'purpose': 'Structures response'
        },
        {
            'name': '5. Examples (Few-shot)',
            'content': 'Example: For telecom churn:\nFinding 1: High monthly charges (Impact: 35%)\nFinding 2: Short tenure (Impact: 28%)',
            'y': 0.28,
            'height': 0.12,
            'color': 'lavender',
            'purpose': 'Shows desired output'
        },
        {
            'name': '6. Constraints/Tone',
            'content': 'Use non-technical language.\nFocus on actionable insights.\nAvoid complex statistical jargon.',
            'y': 0.15,
            'height': 0.1,
            'color': 'peachpuff',
            'purpose': 'Guides style'
        }
    ]

    for comp in components:
        # Box
        box = FancyBboxPatch((0.05, comp['y']-comp['height']), 0.6, comp['height'],
                            boxstyle="round,pad=0.01",
                            facecolor=comp['color'],
                            edgecolor='black',
                            linewidth=2)
        ax.add_patch(box)

        # Name
        ax.text(0.07, comp['y']-0.01, comp['name'], ha='left', va='top',
                fontsize=11, fontweight='bold')

        # Content
        ax.text(0.07, comp['y']-0.03, comp['content'], ha='left', va='top',
                fontsize=9, style='italic')

        # Purpose annotation
        ax.text(0.68, comp['y']-comp['height']/2, comp['purpose'], ha='left', va='center',
                fontsize=10, color='darkblue',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='blue', linewidth=1))

        # Arrow
        ax.arrow(0.65, comp['y']-comp['height']/2, -0.03, 0,
                head_width=0.015, head_length=0.01, fc='blue', ec='blue')

    # Title
    ax.text(0.5, 0.98, 'Anatomy of an Effective Prompt: 6 Essential Components',
            ha='center', va='top', fontsize=15, fontweight='bold')

    # Best practice box
    practice_box = FancyBboxPatch((0.05, 0.01), 0.9, 0.08,
                                  boxstyle="round,pad=0.02",
                                  facecolor='lightgreen',
                                  edgecolor='black',
                                  linewidth=3)
    ax.add_patch(practice_box)
    ax.text(0.5, 0.05, 'Best Practice: Combine ALL components for maximum effectiveness\nOrder matters: Role → Task → Context → Format → Examples → Constraints',
            ha='center', va='center', fontsize=10, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/prompt_anatomy.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("8/8: prompt_anatomy.pdf created")


# Generate all charts
if __name__ == "__main__":
    print("\nGenerating Week 10 BSc-level charts (Educational Framework)...")
    print("="*60)

    plot_cost_performance_scatter()
    plot_transfer_learning_visual()
    plot_parameter_spectrum()
    plot_data_performance_curves()
    plot_adapter_architecture()
    plot_catastrophic_forgetting()
    plot_applications_2024()
    plot_prompt_anatomy()

    print("="*60)
    print("All 8 BSc charts generated successfully!")
    print("Location: NLP_slides/week10_finetuning/figures/")
