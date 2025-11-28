import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrow, Ellipse
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# 1. Model Compression Landscape
def plot_model_compression_landscape():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Model sizes and performance
    models = [
        ('GPT-3\n175B', 175000, 0.95, 'red', 350),
        ('GPT-2\n1.5B', 1500, 0.85, 'orange', 3),
        ('BERT-Large\n340M', 340, 0.80, 'yellow', 1.3),
        ('BERT-Base\n110M', 110, 0.75, 'lightgreen', 0.44),
        ('DistilBERT\n66M', 66, 0.72, 'green', 0.26),
        ('TinyBERT\n15M', 15, 0.68, 'cyan', 0.06),
        ('MobileBERT\n25M', 25, 0.70, 'blue', 0.1),
        ('Edge Model\n5M', 5, 0.60, 'purple', 0.02)
    ]
    
    # Plot models
    for name, params, perf, color, size_gb in models:
        # Size based on params (log scale)
        bubble_size = np.log10(params) * 100
        
        # Position based on size vs performance
        x = np.log10(params)
        y = perf
        
        ax.scatter(x, y, s=bubble_size, c=color, alpha=0.7, 
                  edgecolors='black', linewidth=2)
        ax.annotate(f'{name}\n{size_gb}GB', (x, y), 
                   xytext=(0, -30), textcoords='offset points', 
                   ha='center', fontsize=9, fontweight='bold')
    
    # Add deployment zones
    ax.axvspan(0, 2, alpha=0.1, color='green', label='Mobile')
    ax.axvspan(2, 3.5, alpha=0.1, color='yellow', label='Edge')
    ax.axvspan(3.5, 6, alpha=0.1, color='red', label='Cloud Only')
    
    # Add performance threshold
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
    ax.text(3, 0.71, 'Practical Performance Threshold', fontsize=10, color='green')
    
    ax.set_xlabel('Model Size (log10 parameters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (Relative to GPT-3)', fontsize=14, fontweight='bold')
    ax.set_title('Model Compression Landscape: Size vs Performance Trade-off', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.55, 1.0)
    
    plt.tight_layout()
    plt.savefig('../figures/model_compression_landscape.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Quantization Levels
def plot_quantization_levels():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Quantization bit widths
    bit_widths = ['FP32\n(32 bits)', 'FP16\n(16 bits)', 'INT8\n(8 bits)', 
                  'INT4\n(4 bits)', 'Binary\n(1 bit)']
    memory_size = [100, 50, 25, 12.5, 3.125]  # Relative to FP32
    accuracy = [100, 99.9, 99, 97, 85]  # Relative accuracy
    
    # Memory savings
    x = np.arange(len(bit_widths))
    bars1 = ax1.bar(x, memory_size, color=['red', 'orange', 'yellow', 'lightgreen', 'green'], 
                     alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, size in zip(bars1, memory_size):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{size:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('Memory Usage (% of FP32)', fontsize=12, fontweight='bold')
    ax1.set_title('Memory Savings with Quantization', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bit_widths)
    ax1.set_ylim(0, 110)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Accuracy retention
    bars2 = ax2.bar(x, accuracy, color=['red', 'orange', 'yellow', 'lightgreen', 'green'], 
                     alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, acc in zip(bars2, accuracy):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc}%', ha='center', va='bottom', fontweight='bold')
    
    # Add acceptable threshold
    ax2.axhline(y=95, color='red', linestyle='--', alpha=0.5)
    ax2.text(2.5, 96, 'Typical Acceptable Threshold', ha='center', color='red')
    
    ax2.set_ylabel('Accuracy (% of FP32)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Impact of Quantization', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bit_widths)
    ax2.set_ylim(80, 105)
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('Quantization: Trading Precision for Efficiency', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/quantization_levels.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Knowledge Distillation
def plot_knowledge_distillation():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Teacher model
    teacher_box = FancyBboxPatch((0.1, 0.5), 0.2, 0.3,
                                boxstyle="round,pad=0.02",
                                facecolor='lightblue',
                                edgecolor='black',
                                linewidth=3)
    ax.add_patch(teacher_box)
    ax.text(0.2, 0.65, 'Teacher\nModel\n(Large)', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax.text(0.2, 0.45, '175B params\n350GB', ha='center', va='center', fontsize=9)
    
    # Student model
    student_box = FancyBboxPatch((0.7, 0.55), 0.15, 0.2,
                                boxstyle="round,pad=0.02",
                                facecolor='lightgreen',
                                edgecolor='black',
                                linewidth=3)
    ax.add_patch(student_box)
    ax.text(0.775, 0.65, 'Student\nModel\n(Small)', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(0.775, 0.52, '1.5B params\n3GB', ha='center', va='center', fontsize=9)
    
    # Knowledge transfer arrows
    # Soft labels
    arrow1 = FancyArrow(0.31, 0.7, 0.37, 0, head_width=0.03, head_length=0.02,
                       fc='orange', ec='orange', linewidth=2)
    ax.add_patch(arrow1)
    ax.text(0.5, 0.73, 'Soft Labels\n(Probabilities)', ha='center', va='bottom', 
            fontsize=10, fontweight='bold', color='orange')
    
    # Feature matching
    arrow2 = FancyArrow(0.31, 0.65, 0.37, 0, head_width=0.03, head_length=0.02,
                       fc='green', ec='green', linewidth=2)
    ax.add_patch(arrow2)
    ax.text(0.5, 0.62, 'Hidden States', ha='center', va='top', 
            fontsize=10, fontweight='bold', color='green')
    
    # Attention transfer
    arrow3 = FancyArrow(0.31, 0.6, 0.37, 0, head_width=0.03, head_length=0.02,
                       fc='blue', ec='blue', linewidth=2)
    ax.add_patch(arrow3)
    ax.text(0.5, 0.57, 'Attention Maps', ha='center', va='top', 
            fontsize=10, fontweight='bold', color='blue')
    
    # Training data
    data_box = FancyBboxPatch((0.4, 0.1), 0.2, 0.15,
                             boxstyle="round,pad=0.02",
                             facecolor='lightyellow',
                             edgecolor='black',
                             linewidth=2)
    ax.add_patch(data_box)
    ax.text(0.5, 0.175, 'Training\nData', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Arrows from data
    ax.arrow(0.4, 0.25, -0.08, 0.23, head_width=0.02, head_length=0.02,
             fc='gray', ec='gray', linewidth=2, alpha=0.5)
    ax.arrow(0.6, 0.25, 0.08, 0.28, head_width=0.02, head_length=0.02,
             fc='gray', ec='gray', linewidth=2, alpha=0.5)
    
    # Key insight
    ax.text(0.5, 0.9, 'Knowledge Distillation: Teaching Small Models to Think Like Large Ones',
            ha='center', fontsize=16, fontweight='bold')
    
    # Benefits box
    benefits = ['95% performance', '10x smaller', '50x faster', 'Mobile-ready']
    for i, benefit in enumerate(benefits):
        benefit_box = FancyBboxPatch((0.05 + i*0.23, 0.35), 0.2, 0.08,
                                    boxstyle="round,pad=0.01",
                                    facecolor='lightgray',
                                    edgecolor='black',
                                    alpha=0.7)
        ax.add_patch(benefit_box)
        ax.text(0.15 + i*0.23, 0.39, benefit, ha='center', va='center', fontsize=9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/knowledge_distillation.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Pruning Strategies
def plot_pruning_strategies():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Magnitude Pruning
    ax = axes[0, 0]
    np.random.seed(42)
    weights = np.random.randn(10, 10) * 2
    mask = np.abs(weights) > 0.5
    pruned_weights = weights * mask
    
    im1 = ax.imshow(weights, cmap='RdBu', vmin=-3, vmax=3)
    ax.set_title('Original Weights', fontsize=12, fontweight='bold')
    ax.set_xlabel('Output Neurons')
    ax.set_ylabel('Input Neurons')
    
    # 2. After Magnitude Pruning
    ax = axes[0, 1]
    im2 = ax.imshow(pruned_weights, cmap='RdBu', vmin=-3, vmax=3)
    ax.set_title('After Magnitude Pruning (70% removed)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Output Neurons')
    ax.set_ylabel('Input Neurons')
    
    # 3. Structured Pruning
    ax = axes[1, 0]
    structured_mask = np.ones((10, 10))
    structured_mask[:, [2, 5, 8]] = 0  # Remove entire columns
    structured_mask[[1, 6], :] = 0     # Remove entire rows
    structured_pruned = weights * structured_mask
    
    im3 = ax.imshow(structured_pruned, cmap='RdBu', vmin=-3, vmax=3)
    ax.set_title('Structured Pruning (channels removed)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Output Neurons')
    ax.set_ylabel('Input Neurons')
    
    # 4. Pruning Impact
    ax = axes[1, 1]
    sparsity_levels = [0, 50, 70, 90, 95, 99]
    accuracy = [100, 99.5, 99, 97, 93, 80]
    speedup = [1, 1.5, 2.1, 4.5, 8, 20]
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(sparsity_levels, accuracy, 'b-o', linewidth=2, markersize=8, label='Accuracy')
    line2 = ax2.plot(sparsity_levels, speedup, 'r-s', linewidth=2, markersize=8, label='Speedup')
    
    ax.set_xlabel('Sparsity (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='blue')
    ax2.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold', color='red')
    ax.set_title('Pruning: Accuracy vs Speedup Trade-off', fontsize=12, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center left')
    
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add colorbar for weight matrices
    fig.colorbar(im1, ax=axes[0, :], orientation='horizontal', pad=0.1, 
                 label='Weight Value', shrink=0.8)
    
    plt.suptitle('Pruning Strategies: Removing Redundant Connections', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/pruning_strategies.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Deployment Pipeline
def plot_deployment_pipeline():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Pipeline stages
    stages = [
        ('Pre-trained\nModel', 0.1, 0.7, 'lightblue', '440MB\nBERT-base'),
        ('Distillation', 0.3, 0.7, 'lightgreen', '66MB\nDistilBERT'),
        ('Quantization', 0.5, 0.7, 'yellow', '17MB\nINT8'),
        ('Pruning', 0.7, 0.7, 'orange', '5MB\nSparse'),
        ('Deployment', 0.9, 0.7, 'lightcoral', 'Mobile\nReady')
    ]
    
    # Draw stages
    for i, (name, x, y, color, details) in enumerate(stages):
        box = FancyBboxPatch((x-0.08, y-0.1), 0.16, 0.2,
                            boxstyle="round,pad=0.02",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=2)
        ax.add_patch(box)
        ax.text(x, y+0.05, name, ha='center', va='center', 
                fontsize=11, fontweight='bold')
        ax.text(x, y-0.05, details, ha='center', va='center', 
                fontsize=9, style='italic')
        
        # Arrows between stages
        if i < len(stages) - 1:
            arrow = FancyArrow(x+0.08, y, 0.12, 0, head_width=0.03, head_length=0.02,
                              fc='gray', ec='gray', linewidth=2)
            ax.add_patch(arrow)
    
    # Performance metrics
    metrics_y = 0.4
    metrics = [
        ('Size', [440, 66, 17, 5, 5], 'MB'),
        ('Latency', [100, 80, 40, 20, 15], 'ms'),
        ('Accuracy', [100, 97, 96, 94, 94], '%')
    ]
    
    for i, (metric, values, unit) in enumerate(metrics):
        y_pos = metrics_y - i * 0.1
        ax.text(0.02, y_pos, f'{metric}:', fontsize=10, fontweight='bold')
        
        for j, (stage, x, _, _, _) in enumerate(stages):
            ax.text(x, y_pos, f'{values[j]}{unit}', ha='center', fontsize=9)
    
    # Hardware targets
    targets_y = 0.25
    ax.text(0.5, targets_y, 'Deployment Targets:', ha='center', fontsize=12, fontweight='bold')
    
    target_boxes = [
        ('Cloud GPU', 0.2, targets_y-0.08, 'lightblue'),
        ('Edge Server', 0.4, targets_y-0.08, 'lightgreen'),
        ('Mobile Phone', 0.6, targets_y-0.08, 'yellow'),
        ('IoT Device', 0.8, targets_y-0.08, 'orange')
    ]
    
    for target, x, y, color in target_boxes:
        box = FancyBboxPatch((x-0.07, y-0.03), 0.14, 0.06,
                            boxstyle="round,pad=0.01",
                            facecolor=color,
                            edgecolor='black',
                            alpha=0.7)
        ax.add_patch(box)
        ax.text(x, y, target, ha='center', va='center', fontsize=9)
    
    ax.set_title('Model Deployment Pipeline: From Research to Production', 
                fontsize=16, fontweight='bold')
    ax.text(0.5, 0.05, 'Each stage trades model size for deployment flexibility', 
            ha='center', fontsize=11, style='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.85)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/deployment_pipeline.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 6. Energy Efficiency Trends
def plot_energy_efficiency_trends():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Energy consumption over time
    years = np.array([2018, 2019, 2020, 2021, 2022, 2023, 2024])
    
    # Model sizes (in billions of parameters)
    model_sizes = [0.1, 0.3, 1.5, 17, 175, 540, 1000]
    
    # Energy per inference (normalized)
    energy_naive = model_sizes
    energy_optimized = [0.1, 0.2, 0.5, 2, 10, 20, 30]
    
    ax1.semilogy(years, model_sizes, 'r-o', linewidth=2, markersize=8, 
                 label='Model Size (B params)')
    ax1.semilogy(years, energy_naive, 'b--s', linewidth=2, markersize=8, 
                 label='Naive Energy Use')
    ax1.semilogy(years, energy_optimized, 'g-^', linewidth=2, markersize=8, 
                 label='Optimized Energy Use')
    
    ax1.fill_between(years, energy_optimized, energy_naive, alpha=0.2, color='yellow',
                     label='Efficiency Gains')
    
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Relative Scale (log)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Size vs Energy Consumption Trends', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Carbon footprint comparison
    models = ['GPT-3\nTraining', 'GPT-3\nInference\n(1M queries)', 
              'BERT\nTraining', 'DistilBERT\nTraining', 
              'Mobile Model\nTraining', 'Edge Inference\n(1M queries)']
    carbon_kg = [552000, 150, 1438, 144, 14, 0.5]
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen']
    
    bars = ax2.bar(range(len(models)), carbon_kg, color=colors, alpha=0.7, 
                    edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, carbon in zip(bars, carbon_kg):
        height = bar.get_height()
        if carbon > 1000:
            label = f'{carbon/1000:.0f}k kg'
        else:
            label = f'{carbon:.1f} kg'
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                label, ha='center', va='bottom', fontweight='bold')
    
    ax2.set_yscale('log')
    ax2.set_ylabel('CO2 Emissions (kg)', fontsize=12, fontweight='bold')
    ax2.set_title('Carbon Footprint of Different Models', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add reference line
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5)
    ax2.text(2.5, 120, 'Car trip NYC-SF', ha='center', color='red', fontsize=9)
    
    plt.suptitle('The Environmental Cost of AI: Why Efficiency Matters', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/energy_efficiency_trends.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all Week 11 visualizations
if __name__ == "__main__":
    print("Generating Week 11 Efficiency & Deployment visualizations...")
    plot_model_compression_landscape()
    print("- Speed-accuracy frontier visualization created")
    plot_quantization_levels()
    print("- Memory reduction through quantization created")
    plot_deployment_pipeline()
    print("- Latency benchmarks across deployment stages created")
    print("\nWeek 11 visualizations completed!")