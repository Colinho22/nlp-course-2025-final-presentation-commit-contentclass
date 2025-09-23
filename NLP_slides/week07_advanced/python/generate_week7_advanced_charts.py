import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrow
import seaborn as sns
import os
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# 1. Model Scale Timeline
def plot_model_scale_timeline():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Model data (name, year, parameters in billions, color)
    models = [
        ('BERT-Large', 2018, 0.34, '#3498db'),
        ('GPT-2', 2019, 1.5, '#e74c3c'),
        ('T5', 2019, 11, '#2ecc71'),
        ('GPT-3', 2020, 175, '#e74c3c'),
        ('Gopher', 2021, 280, '#9b59b6'),
        ('PaLM', 2022, 540, '#f39c12'),
        ('GPT-4', 2023, 1760, '#e74c3c'),  # Estimated
    ]
    
    # Extract data
    years = [m[1] for m in models]
    params = [m[2] for m in models]
    names = [m[0] for m in models]
    colors = [m[3] for m in models]
    
    # Plot on log scale
    ax.scatter(years, params, s=300, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add labels
    for i, (name, year, param, color) in enumerate(models):
        if param < 100:
            ax.annotate(f'{name}\n{param}B', (year, param), 
                       xytext=(0, 20), textcoords='offset points',
                       ha='center', fontsize=10, fontweight='bold')
        else:
            ax.annotate(f'{name}\n{param}B', (year, param), 
                       xytext=(0, -30), textcoords='offset points',
                       ha='center', fontsize=10, fontweight='bold')
    
    # Add trend line
    z = np.polyfit(years, np.log10(params), 1)
    p = np.poly1d(z)
    years_extended = np.linspace(2018, 2024, 100)
    ax.plot(years_extended, 10**p(years_extended), '--', color='gray', alpha=0.5, linewidth=2)
    
    # Annotations for scale comparisons
    ax.text(2020.5, 600, '1000x\nin 5 years!', fontsize=14, color='red', 
            fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
    
    # Add book comparison annotations
    book_annotations = [
        (2018, 0.34, '1 book'),
        (2019, 11, 'Small library'),
        (2020, 175, 'Library of Congress'),
        (2023, 1760, 'All books ever written')
    ]
    
    for year, param, text in book_annotations:
        if param in params:
            idx = params.index(param)
            ax.annotate(text, (year, param), 
                       xytext=(40, 0), textcoords='offset points',
                       ha='left', fontsize=9, style='italic',
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    
    ax.set_yscale('log')
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Parameters (Billions)', fontsize=14, fontweight='bold')
    ax.set_title('The Exponential Growth of Language Models\n"Just Make It Bigger" Works!', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2017.5, 2024)
    ax.set_ylim(0.1, 3000)
    
    plt.tight_layout()
    plt.savefig('../figures/model_scale_timeline.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Parameter vs Capabilities (Scaling Laws Visualization)
def plot_scaling_laws():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Loss vs parameters
    params = np.logspace(7, 12, 100)  # 10M to 1T
    loss = 2.5 * params**(-0.05)  # Simplified scaling law
    
    ax1.loglog(params, loss, linewidth=3, color='blue')
    ax1.set_xlabel('Model Parameters', fontsize=12)
    ax1.set_ylabel('Test Loss', fontsize=12)
    ax1.set_title('Scaling Law: Loss vs Parameters', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Mark actual models
    model_params = [3.4e8, 1.5e9, 1.75e11, 5.4e11]  # BERT, GPT-2, GPT-3, PaLM
    model_names = ['BERT', 'GPT-2', 'GPT-3', 'PaLM']
    model_losses = [2.5 * p**(-0.05) for p in model_params]
    
    ax1.scatter(model_params, model_losses, s=100, c='red', zorder=5)
    for p, l, n in zip(model_params, model_losses, model_names):
        ax1.annotate(n, (p, l), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Add formula
    ax1.text(0.05, 0.95, r'Loss $\propto N^{-0.05}$', 
            transform=ax1.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    # Right: Compute-optimal frontier
    compute_budget = np.logspace(20, 26, 100)  # FLOPs
    optimal_params = (compute_budget / 6)**0.5  # Chinchilla scaling
    
    ax2.loglog(compute_budget, optimal_params, linewidth=3, color='green', 
               label='Compute-optimal')
    
    # Show over/under-trained regions
    ax2.fill_between(compute_budget, optimal_params*0.1, optimal_params, 
                     alpha=0.2, color='red', label='Under-trained')
    ax2.fill_between(compute_budget, optimal_params, optimal_params*10, 
                     alpha=0.2, color='blue', label='Over-parameterized')
    
    ax2.set_xlabel('Compute Budget (FLOPs)', fontsize=12)
    ax2.set_ylabel('Optimal Parameters', fontsize=12)
    ax2.set_title('Chinchilla Scaling: Compute-Optimal Training', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Mark GPT-3 and Chinchilla
    gpt3_compute = 3.14e23
    gpt3_params = 1.75e11
    chinchilla_params = 7e10
    
    ax2.scatter([gpt3_compute], [gpt3_params], s=100, c='red', marker='x', linewidths=3)
    ax2.annotate('GPT-3\n(over-parameterized)', (gpt3_compute, gpt3_params), 
                xytext=(-50, 20), textcoords='offset points', fontsize=9)
    
    ax2.scatter([gpt3_compute], [chinchilla_params], s=100, c='green', marker='o')
    ax2.annotate('Chinchilla\n(optimal)', (gpt3_compute, chinchilla_params), 
                xytext=(20, -30), textcoords='offset points', fontsize=9)
    
    plt.suptitle('Scaling Laws: Predictable Progress with Scale', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/scaling_laws.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Task Performance Chart
def plot_gpt3_capabilities():
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Task categories and performance
    categories = ['Translation', 'Q&A', 'Arithmetic', 'Code\nGeneration', 
                  'Common\nSense', 'Reading\nComp.', 'Summarization', 'Analogy']
    
    # Performance scores (out of 100)
    gpt2_scores = [45, 55, 10, 20, 65, 70, 60, 40]
    gpt3_scores = [85, 88, 75, 82, 90, 92, 88, 85]
    human_baseline = [95, 90, 98, 85, 95, 95, 90, 90]
    
    x = np.arange(len(categories))
    width = 0.25
    
    # Create bars
    bars1 = ax.bar(x - width, gpt2_scores, width, label='GPT-2 (1.5B)', color='#3498db')
    bars2 = ax.bar(x, gpt3_scores, width, label='GPT-3 (175B)', color='#2ecc71')
    bars3 = ax.bar(x + width, human_baseline, width, label='Human Baseline', 
                   color='#e74c3c', alpha=0.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height}', ha='center', va='bottom', fontsize=9)
    
    # Highlight emergent abilities
    emergent_tasks = [2, 3]  # Arithmetic and Code Generation
    for idx in emergent_tasks:
        ax.add_patch(Rectangle((idx-0.5, 0), 1, 100, 
                              facecolor='yellow', alpha=0.2, zorder=0))
        ax.text(idx, 95, 'Emergent!', ha='center', fontsize=10, 
               color='red', fontweight='bold')
    
    ax.set_xlabel('Task Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance Score', fontsize=14, fontweight='bold')
    ax.set_title('GPT-3 Capabilities: Emergent Abilities at Scale\nNo Task-Specific Training!', 
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add note about few-shot
    ax.text(0.5, 0.02, 'All evaluations done with few-shot prompting (no fine-tuning)', 
            transform=ax.transAxes, ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    plt.tight_layout()
    plt.savefig('../figures/gpt3_capabilities.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all Week 7 visualizations
if __name__ == "__main__":
    print("Generating Week 7 Advanced Transformer visualizations...")
    plot_model_scale_timeline()
    print("- Model scaling curve created")
    plot_scaling_laws()
    print("- Parameter vs capabilities chart created")
    plot_gpt3_capabilities()
    print("- Task performance chart created")
    print("\nWeek 7 visualizations completed!")