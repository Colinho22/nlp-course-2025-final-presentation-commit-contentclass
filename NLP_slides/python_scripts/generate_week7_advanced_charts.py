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

# 2. Scaling Laws Visualization
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

# 3. Few-shot Performance
def plot_few_shot_performance():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Number of examples
    n_shots = [0, 1, 2, 4, 8, 16, 32]
    
    # Performance for different model sizes (simplified)
    small_model = [52, 58, 61, 64, 66, 67, 68]  # Plateaus quickly
    medium_model = [65, 72, 76, 79, 82, 84, 85]
    large_model = [78, 85, 88, 91, 93, 94, 95]  # GPT-3 scale
    
    # Plot lines
    ax.plot(n_shots, small_model, 'o-', label='Small (350M)', linewidth=2, markersize=8)
    ax.plot(n_shots, medium_model, 's-', label='Medium (13B)', linewidth=2, markersize=8)
    ax.plot(n_shots, large_model, '^-', label='Large (175B)', linewidth=2, markersize=8)
    
    # Highlight zero-shot and few-shot regions
    ax.axvspan(-0.5, 0.5, alpha=0.1, color='red', label='Zero-shot')
    ax.axvspan(0.5, 8.5, alpha=0.1, color='green', label='Few-shot')
    ax.axvspan(8.5, 35, alpha=0.1, color='blue', label='Many-shot')
    
    # Annotations
    ax.annotate('Large models excel\nat zero-shot!', xy=(0, 78), xytext=(0.5, 70),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=11, color='red')
    
    ax.annotate('Diminishing returns\nfor small models', xy=(16, 67), xytext=(20, 60),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=11, color='blue')
    
    ax.set_xlabel('Number of Examples', fontsize=14, fontweight='bold')
    ax.set_ylabel('Task Performance (%)', fontsize=14, fontweight='bold')
    ax.set_title('Few-Shot Learning: Large Models Learn from Examples\nNo Fine-tuning Required!', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 35)
    ax.set_ylim(50, 100)
    
    # Add note
    ax.text(0.5, 0.02, 'Performance on SuperGLUE benchmark tasks (illustrative)', 
            transform=ax.transAxes, ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    plt.tight_layout()
    plt.savefig('../figures/few_shot_performance.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Model Parallelism Visualization
def plot_model_parallelism():
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # GPU layout
    gpus_per_row = 4
    n_rows = 2
    gpu_width = 0.15
    gpu_height = 0.25
    
    # Draw GPUs
    for row in range(n_rows):
        for col in range(gpus_per_row):
            gpu_idx = row * gpus_per_row + col
            x = 0.1 + col * 0.22
            y = 0.6 - row * 0.35
            
            # GPU box
            gpu_box = FancyBboxPatch((x, y), gpu_width, gpu_height,
                                     boxstyle="round,pad=0.02",
                                     facecolor='lightgray',
                                     edgecolor='black',
                                     linewidth=2)
            ax.add_patch(gpu_box)
            ax.text(x + gpu_width/2, y + gpu_height + 0.02, f'GPU {gpu_idx}',
                   ha='center', fontsize=10, fontweight='bold')
    
    # Show different parallelism strategies
    # 1. Data Parallelism
    ax.text(0.25, 0.95, 'Data Parallelism', fontsize=14, fontweight='bold', color='blue')
    for col in range(gpus_per_row):
        x = 0.1 + col * 0.22
        y = 0.6
        ax.add_patch(Rectangle((x+0.01, y+0.18), gpu_width-0.02, 0.05, 
                              facecolor='blue', alpha=0.5))
        ax.text(x + gpu_width/2, y + 0.205, 'Batch', ha='center', fontsize=8)
    
    # 2. Model Parallelism (layers)
    ax.text(0.75, 0.95, 'Pipeline Parallelism', fontsize=14, fontweight='bold', color='green')
    layer_colors = ['#ff9999', '#99ff99', '#9999ff', '#ffff99']
    for col in range(gpus_per_row):
        x = 0.1 + col * 0.22
        y = 0.6
        ax.add_patch(Rectangle((x+0.01, y+0.1), gpu_width-0.02, 0.07, 
                              facecolor=layer_colors[col], alpha=0.7))
        ax.text(x + gpu_width/2, y + 0.135, f'Layers\n{col*3}-{col*3+2}', 
               ha='center', fontsize=8)
    
    # 3. Tensor Parallelism
    for row in range(n_rows):
        for col in range(gpus_per_row):
            x = 0.1 + col * 0.22
            y = 0.6 - row * 0.35
            if row == 1:
                # Show matrix sharding
                ax.add_patch(Rectangle((x+0.02, y+0.05), gpu_width/2-0.02, 0.15, 
                                      facecolor='red', alpha=0.5))
                ax.add_patch(Rectangle((x+gpu_width/2, y+0.05), gpu_width/2-0.02, 0.15, 
                                      facecolor='orange', alpha=0.5))
                ax.text(x + gpu_width/2, y + 0.125, 'Matrix\nShard', ha='center', fontsize=7)
    
    ax.text(0.5, 0.18, 'Tensor Parallelism', fontsize=14, fontweight='bold', color='red')
    
    # Communication patterns
    # Pipeline communication
    for col in range(gpus_per_row-1):
        x1 = 0.1 + col * 0.22 + gpu_width
        x2 = 0.1 + (col+1) * 0.22
        y = 0.735
        ax.arrow(x1, y, x2-x1-0.01, 0, head_width=0.02, head_length=0.01,
                fc='green', ec='green', linewidth=2)
    
    # Add title and labels
    ax.text(0.5, 1.05, 'Model Parallelism Strategies for 175B Parameters', 
            ha='center', fontsize=16, fontweight='bold')
    
    # Add strategy comparison
    strategies = [
        ('Data Parallel:', '• Same model copy\n• Different batches\n• High communication'),
        ('Pipeline Parallel:', '• Split by layers\n• Sequential processing\n• Memory efficient'),
        ('Tensor Parallel:', '• Split matrices\n• Parallel computation\n• Low latency')
    ]
    
    y_pos = 0.5
    for strategy, details in strategies:
        ax.text(0.92, y_pos, strategy, fontsize=11, fontweight='bold', ha='right')
        ax.text(0.93, y_pos-0.01, details, fontsize=9, ha='left', va='top')
        y_pos -= 0.15
    
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/model_parallelism.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 5. GPT-3 Capabilities Chart
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
    print("- Model scale timeline created")
    plot_scaling_laws()
    print("- Scaling laws visualization created")
    plot_few_shot_performance()
    print("- Few-shot performance visualization created")
    plot_model_parallelism()
    print("- Model parallelism visualization created")
    plot_gpt3_capabilities()
    print("- GPT-3 capabilities chart created")
    print("\nWeek 7 visualizations completed!")