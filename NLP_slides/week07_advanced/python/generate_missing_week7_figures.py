import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Arrow, FancyArrowPatch
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

def plot_model_scale_timeline():
    """Plot the evolution of model scale over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Model data (year, name, parameters in millions, color)
    models = [
        (2017, 'Transformer', 65, 'blue'),
        (2018, 'BERT-Base', 110, 'green'),
        (2018, 'BERT-Large', 340, 'green'),
        (2019, 'GPT-1', 117, 'red'),
        (2019, 'GPT-2', 1500, 'red'),
        (2020, 'T5-Large', 770, 'orange'),
        (2020, 'GPT-3', 175000, 'red'),
        (2021, 'Switch-T', 1600000, 'purple'),
        (2022, 'PaLM', 540000, 'brown'),
        (2022, 'Chinchilla', 70000, 'pink'),
        (2023, 'GPT-4', 1760000, 'red'),
        (2024, 'Claude-3', 1000000, 'cyan')
    ]
    
    # Extract data
    years = [model[0] for model in models]
    names = [model[1] for model in models]
    params = [model[2] for model in models]
    colors = [model[3] for model in models]
    
    # Plot 1: Parameter count over time (log scale)
    scatter = ax1.scatter(years, params, s=[50 + np.log10(p)*20 for p in params], 
                         c=colors, alpha=0.7)
    
    # Add model names as annotations
    for year, name, param, color in models:
        if param > 1000:  # Only label large models to avoid clutter
            ax1.annotate(name, (year, param), xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    ax1.set_title('Language Model Scale Evolution', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(2016.5, 2024.5)
    
    # Add Moore's Law-like trend line
    trend_years = np.array([2017, 2024])
    trend_params = np.array([65, 1760000])
    ax1.plot(trend_years, trend_params, 'k--', alpha=0.5, linewidth=2, 
            label='Exponential Growth Trend')
    ax1.legend(fontsize=11)
    
    # Plot 2: Training cost and compute
    training_costs = [0.01, 0.05, 0.2, 0.1, 10, 5, 12000, 50000, 20000, 8000, 100000, 30000]  # Million USD
    
    bars = ax2.bar(range(len(models)), training_costs, color=colors, alpha=0.7)
    ax2.set_xlabel('Models (Chronological)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Cost (Million USD)', fontsize=12, fontweight='bold')
    ax2.set_title('Training Cost Evolution', fontsize=16, fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([f"{name}\n({year})" for year, name, _, _ in models], 
                       rotation=45, ha='right', fontsize=10)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Add cost labels on significant bars
    for i, (bar, cost) in enumerate(zip(bars, training_costs)):
        if cost > 1000:  # Only label expensive models
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
                    f'${cost/1000:.0f}M', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
    
    plt.suptitle('The Scale Revolution: From Millions to Trillions', 
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/model_scale_timeline.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_scaling_laws():
    """Plot scaling laws for neural language models"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Loss vs Parameters (Kaplan scaling law)
    params = np.logspace(6, 12, 50)  # 1M to 1T parameters
    # Kaplan et al: L(N) ∝ N^(-0.076)
    kaplan_loss = 10 * (params / 1e6) ** (-0.076)
    # Chinchilla revision: needs more data
    chinchilla_loss = 8 * (params / 1e6) ** (-0.095)
    
    ax1.loglog(params / 1e6, kaplan_loss, 'b-', linewidth=3, label='Kaplan et al. (2020)')
    ax1.loglog(params / 1e6, chinchilla_loss, 'r-', linewidth=3, label='Chinchilla (2022)')
    
    # Add actual model points
    model_params = [110, 340, 1500, 175000, 540000, 1760000]  # millions
    model_losses = [3.2, 2.8, 2.1, 1.2, 0.9, 0.7]  # estimated
    model_names = ['BERT-Base', 'BERT-Large', 'GPT-2', 'GPT-3', 'PaLM', 'GPT-4']
    
    ax1.scatter(model_params, model_losses, s=100, c='green', alpha=0.8, 
               zorder=5, label='Actual Models')
    
    for param, loss, name in zip(model_params, model_losses, model_names):
        if param > 1000:  # Only label large models
            ax1.annotate(name, (param, loss), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Scaling Laws: Loss vs Parameters', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Optimal Model Size vs Compute Budget
    compute_budget = np.logspace(18, 25, 50)  # FLOPs
    # Chinchilla optimal: N ∝ C^0.5
    optimal_params = (compute_budget / 1e20) ** 0.5 * 1000  # millions
    
    ax2.loglog(compute_budget, optimal_params, 'g-', linewidth=3, 
              label='Chinchilla Optimal')
    
    # Add GPT series trend (over-parameterized)
    gpt_compute = [1e20, 3e23, 3e25]
    gpt_params = [117, 1500, 175000]
    gpt_names = ['GPT-1', 'GPT-2', 'GPT-3']
    ax2.scatter(gpt_compute, gpt_params, s=100, c='red', alpha=0.8, 
               label='GPT Series')
    
    for compute, param, name in zip(gpt_compute, gpt_params, gpt_names):
        ax2.annotate(name, (compute, param), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Compute Budget (FLOPs)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Optimal Parameters (Millions)', fontsize=12, fontweight='bold')
    ax2.set_title('Compute-Optimal Model Size', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance vs Scale on downstream tasks
    model_sizes = [110, 340, 1500, 175000]
    # Normalized performance scores across tasks
    reading_comp = [65, 72, 78, 89]
    math_reasoning = [35, 42, 55, 78]
    code_generation = [20, 25, 45, 72]
    
    x = np.arange(len(model_sizes))
    width = 0.25
    
    bars1 = ax3.bar(x - width, reading_comp, width, label='Reading Comprehension', 
                   color='blue', alpha=0.7)
    bars2 = ax3.bar(x, math_reasoning, width, label='Math Reasoning', 
                   color='red', alpha=0.7)
    bars3 = ax3.bar(x + width, code_generation, width, label='Code Generation', 
                   color='green', alpha=0.7)
    
    ax3.set_xlabel('Model Size (Parameters)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
    ax3.set_title('Task Performance vs Scale', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{size}M' if size < 1000 else f'{size//1000}B' 
                        for size in model_sizes])
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 4. Emergence threshold visualization
    param_range = np.logspace(8, 12, 100)  # 100M to 1T
    
    # Different capabilities emerge at different scales
    arithmetic = 1 / (1 + np.exp(-(np.log10(param_range) - 10.5) * 2))  # ~300B threshold
    reasoning = 1 / (1 + np.exp(-(np.log10(param_range) - 11.0) * 3))   # ~1T threshold
    creativity = 1 / (1 + np.exp(-(np.log10(param_range) - 11.2) * 4))  # ~1.5T threshold
    
    ax4.semilogx(param_range / 1e6, arithmetic * 100, 'b-', linewidth=3, 
                label='Arithmetic')
    ax4.semilogx(param_range / 1e6, reasoning * 100, 'r-', linewidth=3, 
                label='Complex Reasoning')
    ax4.semilogx(param_range / 1e6, creativity * 100, 'g-', linewidth=3, 
                label='Creative Writing')
    
    # Mark emergence thresholds
    ax4.axvline(x=300000, color='blue', linestyle='--', alpha=0.5)
    ax4.axvline(x=1000000, color='red', linestyle='--', alpha=0.5)
    ax4.axvline(x=1500000, color='green', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Capability Strength (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Emergent Capabilities', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)
    
    plt.suptitle('Neural Scaling Laws: The Science of Scale', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/scaling_laws.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_few_shot_performance():
    """Plot few-shot learning performance"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Few-shot performance vs model size
    model_sizes = [125, 350, 760, 1300, 2700, 6700, 13000, 175000]  # millions
    model_names = ['125M', '350M', '760M', '1.3B', '2.7B', '6.7B', '13B', '175B']
    
    # Performance on different tasks (normalized scores)
    arithmetic = [20, 25, 30, 35, 45, 55, 65, 80]
    reading_comp = [40, 45, 52, 58, 65, 72, 78, 89]
    translation = [35, 42, 48, 55, 62, 68, 74, 85]
    
    ax1.semilogx(model_sizes, arithmetic, 'r-o', linewidth=3, markersize=6, 
                label='Arithmetic')
    ax1.semilogx(model_sizes, reading_comp, 'b-s', linewidth=3, markersize=6, 
                label='Reading Comprehension')
    ax1.semilogx(model_sizes, translation, 'g-^', linewidth=3, markersize=6, 
                label='Translation')
    
    ax1.set_xlabel('Model Size (Parameters)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Few-shot Performance', fontsize=12, fontweight='bold')
    ax1.set_title('Few-shot Learning vs Scale', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # 2. Shot count comparison for GPT-3
    shot_counts = [0, 1, 5, 10, 50]
    
    # Performance with different numbers of examples
    task1_perf = [45, 62, 78, 85, 89]  # Translation
    task2_perf = [30, 55, 72, 80, 83]  # Math word problems
    task3_perf = [25, 45, 68, 75, 78]  # Code generation
    
    ax2.plot(shot_counts, task1_perf, 'b-o', linewidth=3, markersize=8, 
            label='Translation')
    ax2.plot(shot_counts, task2_perf, 'r-s', linewidth=3, markersize=8, 
            label='Math Problems')
    ax2.plot(shot_counts, task3_perf, 'g-^', linewidth=3, markersize=8, 
            label='Code Generation')
    
    ax2.set_xlabel('Number of Examples (Shots)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Task Performance', fontsize=12, fontweight='bold')
    ax2.set_title('Performance vs Examples (GPT-3)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(20, 95)
    
    # 3. GPT-3 vs Fine-tuned models
    tasks = ['Sentiment\nAnalysis', 'Question\nAnswering', 'Translation', 
             'Summarization', 'Math\nProblems']
    
    fine_tuned_scores = [94, 89, 87, 92, 65]  # State-of-art fine-tuned
    gpt3_few_shot = [89, 85, 85, 88, 78]     # GPT-3 few-shot
    
    x = np.arange(len(tasks))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, fine_tuned_scores, width, 
                   label='Fine-tuned SOTA', color='blue', alpha=0.7)
    bars2 = ax3.bar(x + width/2, gpt3_few_shot, width, 
                   label='GPT-3 Few-shot', color='red', alpha=0.7)
    
    ax3.set_xlabel('Tasks', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
    ax3.set_title('Few-shot vs Fine-tuned Performance', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(tasks)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(50, 100)
    
    # Add difference annotations
    for i, (ft, fs) in enumerate(zip(fine_tuned_scores, gpt3_few_shot)):
        diff = fs - ft
        if abs(diff) > 2:
            color = 'green' if diff > 0 else 'red'
            ax3.text(i + width/2, fs + 1, f'{diff:+.0f}', 
                    ha='center', va='bottom', fontweight='bold', color=color)
    
    # 4. In-context learning demonstration
    context_lengths = [0, 50, 100, 200, 500, 1000, 2000]
    # Pattern recognition task performance
    pattern_perf = [20, 45, 62, 75, 85, 90, 92]
    
    ax4.plot(context_lengths, pattern_perf, 'purple', marker='o', 
            linewidth=3, markersize=8)
    ax4.set_xlabel('Context Length (tokens)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Pattern Recognition Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title('In-context Learning: More Context = Better Performance', 
                 fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    
    # Add annotation
    ax4.annotate('Learning without\nparameter updates!', 
                xy=(1000, 90), xytext=(600, 70),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2),
                fontsize=11, fontweight='bold', color='purple',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    plt.suptitle('Few-shot Learning: The Power of Large Language Models', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/few_shot_performance.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_parallelism():
    """Visualize model parallelism strategies"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Data Parallelism
    ax1.set_title('Data Parallelism', fontsize=14, fontweight='bold')
    
    # Draw 4 GPUs with copies of the model
    for i in range(4):
        x_pos = i * 2
        # GPU box
        gpu_box = FancyBboxPatch((x_pos, 0), 1.5, 2,
                                boxstyle="round,pad=0.1",
                                facecolor='lightblue', edgecolor='black')
        ax1.add_patch(gpu_box)
        ax1.text(x_pos + 0.75, 1, f'GPU {i+1}\nFull Model\nCopy', 
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Data batch
        data_box = Rectangle((x_pos + 0.25, -1), 1, 0.5,
                           facecolor='lightgreen', edgecolor='black')
        ax1.add_patch(data_box)
        ax1.text(x_pos + 0.75, -0.75, f'Batch {i+1}', 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Gradient aggregation
    ax1.text(3.5, -2, 'Gradients aggregated across GPUs', 
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
    
    ax1.set_xlim(-0.5, 7.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.axis('off')
    
    # 2. Model Parallelism (Tensor)
    ax2.set_title('Tensor Parallelism', fontsize=14, fontweight='bold')
    
    # Single layer split across GPUs
    layer_width = 6
    layer_height = 1.5
    colors = ['red', 'blue', 'green', 'orange']
    
    for i in range(4):
        x_start = i * layer_width / 4
        # Layer partition
        partition = Rectangle((x_start, 0), layer_width/4, layer_height,
                            facecolor=colors[i], alpha=0.7, edgecolor='black')
        ax2.add_patch(partition)
        ax2.text(x_start + layer_width/8, layer_height/2, f'GPU {i+1}\nLayer Part', 
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    ax2.text(layer_width/2, -0.5, 'Single layer split across GPUs', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax2.set_xlim(-0.5, 6.5)
    ax2.set_ylim(-1, 2)
    ax2.axis('off')
    
    # 3. Pipeline Parallelism
    ax3.set_title('Pipeline Parallelism', fontsize=14, fontweight='bold')
    
    # Stack of layers across GPUs
    layer_names = ['Embedding', 'Layers 1-6', 'Layers 7-12', 'Output Head']
    
    for i, name in enumerate(layer_names):
        y_pos = 3 - i * 0.8
        layer_box = FancyBboxPatch((0, y_pos - 0.3), 4, 0.6,
                                  boxstyle="round,pad=0.05",
                                  facecolor=colors[i], alpha=0.7, edgecolor='black')
        ax3.add_patch(layer_box)
        ax3.text(2, y_pos, f'GPU {i+1}: {name}', 
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # Arrow to next stage
        if i < len(layer_names) - 1:
            ax3.arrow(2, y_pos - 0.4, 0, -0.3,
                     head_width=0.1, head_length=0.05, fc='black', ec='black')
    
    ax3.set_xlim(-0.5, 4.5)
    ax3.set_ylim(-0.5, 3.5)
    ax3.axis('off')
    
    # 4. Performance Comparison
    parallelism_types = ['Data\nParallel', 'Tensor\nParallel', 'Pipeline\nParallel', 'Hybrid\n(All)']
    
    # Metrics (normalized)
    throughput = [100, 85, 75, 120]  # Samples/sec
    memory_efficiency = [25, 95, 90, 98]  # % of optimal
    communication_overhead = [20, 60, 30, 45]  # Relative overhead
    
    x = np.arange(len(parallelism_types))
    width = 0.25
    
    bars1 = ax4.bar(x - width, throughput, width, label='Throughput', 
                   color='blue', alpha=0.7)
    bars2 = ax4.bar(x, memory_efficiency, width, label='Memory Efficiency', 
                   color='green', alpha=0.7)
    bars3 = ax4.bar(x + width, communication_overhead, width, label='Comm. Overhead', 
                   color='red', alpha=0.7)
    
    ax4.set_xlabel('Parallelism Strategy', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Relative Performance', fontsize=12, fontweight='bold')
    ax4.set_title('Parallelism Strategy Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(parallelism_types)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Model Parallelism: Training at Scale', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/model_parallelism.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_gpt3_capabilities():
    """Plot GPT-3 capabilities across different tasks"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Task performance comparison (GPT-3 vs state-of-art)
    tasks = ['Reading\nComp.', 'Translation', 'Question\nAnswering', 'Math\nReasoning', 
             'Code\nGeneration', 'Creative\nWriting']
    
    sota_scores = [92, 89, 88, 75, 70, 85]  # State-of-art specialized models
    gpt3_scores = [89, 85, 82, 78, 72, 90]  # GPT-3 zero/few-shot
    
    x = np.arange(len(tasks))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, sota_scores, width, label='Specialized SOTA', 
                   color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, gpt3_scores, width, label='GPT-3 (Zero/Few-shot)', 
                   color='red', alpha=0.7)
    
    ax1.set_xlabel('Task Categories', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
    ax1.set_title('GPT-3: Generalist vs Specialists', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(60, 100)
    
    # Highlight where GPT-3 exceeds specialists
    for i, (sota, gpt3) in enumerate(zip(sota_scores, gpt3_scores)):
        if gpt3 > sota:
            ax1.text(i + width/2, gpt3 + 1, '★', ha='center', va='bottom',
                    fontsize=16, color='gold')
    
    # 2. Model size vs capability emergence
    model_sizes = [125, 350, 760, 1300, 2700, 6700, 13000, 175000]  # millions
    
    # Different capabilities (normalized 0-100)
    basic_language = [60, 70, 75, 80, 83, 85, 87, 89]
    reasoning = [10, 15, 25, 35, 45, 55, 70, 78]
    creativity = [5, 10, 20, 30, 40, 60, 75, 90]
    code_gen = [2, 5, 15, 25, 35, 50, 65, 72]
    
    ax2.semilogx(model_sizes, basic_language, 'b-o', linewidth=3, 
                label='Basic Language', markersize=6)
    ax2.semilogx(model_sizes, reasoning, 'r-s', linewidth=3, 
                label='Complex Reasoning', markersize=6)
    ax2.semilogx(model_sizes, creativity, 'g-^', linewidth=3, 
                label='Creative Writing', markersize=6)
    ax2.semilogx(model_sizes, code_gen, 'm-d', linewidth=3, 
                label='Code Generation', markersize=6)
    
    ax2.set_xlabel('Model Size (Millions of Parameters)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Capability Strength', fontsize=12, fontweight='bold')
    ax2.set_title('Capability Emergence with Scale', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # 3. Few-shot learning examples
    example_tasks = ['Arithmetic', 'Translation', 'Analogies', 'Code Debug', 'Summarization']
    zero_shot = [45, 72, 68, 35, 78]
    few_shot = [78, 85, 89, 72, 88]
    
    x = np.arange(len(example_tasks))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, zero_shot, width, label='Zero-shot', 
                   color='orange', alpha=0.7)
    bars2 = ax3.bar(x + width/2, few_shot, width, label='Few-shot (5 examples)', 
                   color='green', alpha=0.7)
    
    ax3.set_xlabel('Task Types', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
    ax3.set_title('Zero-shot vs Few-shot Learning', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(example_tasks, rotation=15, ha='right')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # Add improvement percentages
    for i, (zero, few) in enumerate(zip(zero_shot, few_shot)):
        improvement = ((few - zero) / zero) * 100
        ax3.text(i + width/2, few + 2, f'+{improvement:.0f}%', 
                ha='center', va='bottom', fontweight='bold', color='green')
    
    # 4. Applications timeline
    months = np.arange(12)  # 12 months after GPT-3 release
    applications = ['Research\nPapers', 'Code\nAssistants', 'Writing\nTools', 
                   'Educational\nApps', 'Business\nApps', 'Creative\nTools']
    
    # Cumulative adoption
    research_adoption = np.cumsum([100, 200, 150, 100, 80, 60, 40, 30, 25, 20, 15, 10])
    code_adoption = np.cumsum([50, 80, 120, 200, 180, 150, 100, 80, 60, 40, 30, 20])
    writing_adoption = np.cumsum([20, 40, 80, 150, 200, 180, 160, 140, 120, 100, 80, 60])
    
    ax4.plot(months, research_adoption, 'b-o', linewidth=3, label='Research Papers')
    ax4.plot(months, code_adoption, 'r-s', linewidth=3, label='Code Assistants')
    ax4.plot(months, writing_adoption, 'g-^', linewidth=3, label='Writing Tools')
    
    ax4.set_xlabel('Months After GPT-3 Release', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cumulative Applications', fontsize=12, fontweight='bold')
    ax4.set_title('GPT-3 Application Growth', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('GPT-3: The Generalist AI That Changed Everything', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/gpt3_capabilities.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Generating missing Week 7 Advanced Transformer figures...")
    
    print("1. Creating model scale timeline...")
    plot_model_scale_timeline()
    
    print("2. Creating scaling laws visualization...")
    plot_scaling_laws()
    
    print("3. Creating few-shot performance analysis...")
    plot_few_shot_performance()
    
    print("4. Creating model parallelism strategies...")
    plot_model_parallelism()
    
    print("5. Creating GPT-3 capabilities overview...")
    plot_gpt3_capabilities()
    
    print("Week 7 figures generated successfully!")

if __name__ == "__main__":
    main()