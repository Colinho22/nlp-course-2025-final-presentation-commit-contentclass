import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrow
import seaborn as sns
import os
from matplotlib.patches import ConnectionPatch
import networkx as nx

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# 1. Decoding Landscape
def plot_decoding_landscape():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define strategies with positions (safety, creativity)
    strategies = [
        ('Greedy', 0.9, 0.1, 'red', 'Always pick highest probability'),
        ('Beam Search', 0.7, 0.3, 'orange', 'Track top-k paths'),
        ('Top-k=10', 0.5, 0.5, 'yellow', 'Sample from top 10'),
        ('Top-p=0.9', 0.4, 0.6, 'lightgreen', 'Dynamic threshold'),
        ('Temperature=1.2', 0.2, 0.8, 'cyan', 'High randomness'),
        ('Pure Sampling', 0.1, 0.9, 'blue', 'Complete randomness')
    ]
    
    # Plot strategies
    for name, safety, creativity, color, desc in strategies:
        ax.scatter(safety, creativity, s=300, c=color, alpha=0.7, 
                  edgecolors='black', linewidth=2)
        ax.annotate(name, (safety, creativity), xytext=(0, -20), 
                   textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    
    # Add quadrant labels
    ax.text(0.75, 0.9, 'Safe but\nBoring', fontsize=12, ha='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.5))
    ax.text(0.25, 0.9, 'Creative but\nRisky', fontsize=12, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))
    ax.text(0.5, 0.5, 'Sweet\nSpot', fontsize=14, ha='center', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # Draw ideal region
    circle = plt.Circle((0.45, 0.55), 0.15, color='green', fill=False, 
                       linewidth=3, linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    
    # Add arrows showing trade-off
    ax.arrow(0.1, 0.05, 0.8, 0, head_width=0.03, head_length=0.03, 
             fc='gray', ec='gray', alpha=0.5)
    ax.arrow(0.05, 0.1, 0, 0.8, head_width=0.03, head_length=0.03, 
             fc='gray', ec='gray', alpha=0.5)
    
    ax.set_xlabel('Safety / Predictability →', fontsize=14, fontweight='bold')
    ax.set_ylabel('Creativity / Diversity →', fontsize=14, fontweight='bold')
    ax.set_title('The Decoding Strategy Landscape\nChoose Your Trade-off', 
                fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add example outputs
    examples = [
        (0.9, 0.15, '"The weather is nice today."'),
        (0.1, 0.85, '"The weather is purple dancing!"')
    ]
    
    for x, y, text in examples:
        ax.text(x, y, text, fontsize=9, style='italic', ha='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('../figures/decoding_landscape.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Beam Search Tree
def plot_beam_search_tree():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create tree structure
    # Format: (word, probability, level, parent_idx)
    nodes = [
        # Level 0 (start)
        ('START', 1.0, 0, None),
        # Level 1
        ('The', 0.4, 1, 0),
        ('A', 0.3, 1, 0),
        ('I', 0.2, 1, 0),
        ('It', 0.1, 1, 0),
        # Level 2 (only top 3 expanded - beam size 3)
        ('cat', 0.5, 2, 1),
        ('dog', 0.3, 2, 1),
        ('quick', 0.4, 2, 2),
        ('nice', 0.3, 2, 2),
        ('think', 0.6, 2, 3),
        ('love', 0.3, 2, 3),
        # Level 3 (beam continues)
        ('sat', 0.7, 3, 4),
        ('ran', 0.2, 3, 4),
        ('brown', 0.6, 3, 6),
        ('that', 0.8, 3, 8),
    ]
    
    # Positions
    level_width = [1, 4, 6, 4]
    x_positions = {}
    y_positions = [0.8, 0.6, 0.4, 0.2]
    
    # Calculate x positions
    for i, (word, prob, level, parent) in enumerate(nodes):
        if level == 0:
            x_positions[i] = 0.5
        else:
            level_nodes = [j for j, n in enumerate(nodes) if n[2] == level]
            idx_in_level = level_nodes.index(i)
            spacing = 0.8 / (level_width[level] - 1) if level_width[level] > 1 else 0
            x_positions[i] = 0.1 + idx_in_level * spacing
    
    # Draw edges
    for i, (word, prob, level, parent) in enumerate(nodes):
        if parent is not None:
            # Edge from parent to current
            x1, y1 = x_positions[parent], y_positions[nodes[parent][2]]
            x2, y2 = x_positions[i], y_positions[level]
            
            # Color based on whether it's in beam
            in_beam = prob >= 0.3
            color = 'green' if in_beam else 'lightgray'
            width = 2 if in_beam else 1
            
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.7)
            
            # Add probability on edge
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            if in_beam:
                ax.text(mid_x, mid_y, f'{prob:.2f}', fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Draw nodes
    for i, (word, prob, level, parent) in enumerate(nodes):
        x, y = x_positions[i], y_positions[level]
        
        # Node color based on beam membership
        in_beam = prob >= 0.3 or level == 0
        color = 'lightgreen' if in_beam else 'lightcoral'
        
        circle = Circle((x, y), 0.04, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y-0.06, word, ha='center', fontsize=10, fontweight='bold')
    
    # Highlight best path
    best_path = [0, 1, 4, 10, 12]  # Example best path
    for i in range(len(best_path)-1):
        x1, y1 = x_positions[best_path[i]], y_positions[nodes[best_path[i]][2]]
        x2, y2 = x_positions[best_path[i+1]], y_positions[nodes[best_path[i+1]][2]]
        ax.plot([x1, x2], [y1, y2], color='red', linewidth=4, alpha=0.5)
    
    # Add beam size indicator
    ax.text(0.95, 0.5, 'Beam Size = 3', fontsize=12, ha='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
    
    # Labels
    ax.set_title('Beam Search: Exploring Multiple Paths\nRed = Final selected path', 
                fontsize=16, fontweight='bold')
    ax.text(0.5, 0.05, 'Only top 3 hypotheses kept at each step (beam size = 3)', 
            ha='center', fontsize=11, style='italic')
    
    # Add pruning annotation
    ax.text(0.7, 0.45, 'Pruned\n(low score)', fontsize=9, color='red',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral', alpha=0.5))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.9)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/beam_search_tree.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Temperature Effects
def plot_temperature_effects():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Vocabulary for example
    words = ['the', 'a', 'cat', 'dog', 'elephant', 'xyz', 'zzz']
    base_logits = np.array([5.0, 4.5, 3.0, 2.8, 1.0, -2.0, -3.0])
    
    temperatures = [0.5, 1.0, 1.5]
    titles = ['Low Temperature (0.5)\nMore Focused', 
              'Normal Temperature (1.0)\nBalanced', 
              'High Temperature (1.5)\nMore Random']
    
    for ax, temp, title in zip(axes, temperatures, titles):
        # Apply temperature
        scaled_logits = base_logits / temp
        probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))
        
        # Create bar plot
        bars = ax.bar(words, probs, color=plt.cm.coolwarm(probs/max(probs)))
        
        # Add value labels
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylim(0, max(0.8, max(probs) * 1.1))
        ax.set_ylabel('Probability', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # Add entropy value
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        ax.text(0.5, 0.9, f'Entropy: {entropy:.2f}', transform=ax.transAxes,
                ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    plt.suptitle('Temperature Controls Probability Distribution Sharpness', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/temperature_effects.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Sampling Strategies Comparison
def plot_sampling_strategies():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create probability distribution
    vocab_size = 50
    words = [f'w{i}' for i in range(vocab_size)]
    
    # Zipf-like distribution
    probs = 1.0 / (np.arange(vocab_size) + 1) ** 0.7
    probs = probs / probs.sum()
    
    # Top-k visualization
    k = 10
    ax1.bar(range(vocab_size), probs, color='lightblue', alpha=0.7)
    ax1.bar(range(k), probs[:k], color='darkblue', alpha=0.9)
    ax1.axvline(x=k-0.5, color='red', linestyle='--', linewidth=2)
    ax1.text(k, max(probs)*0.8, f'Top-{k} cutoff', fontsize=11, color='red', 
             rotation=90, va='bottom')
    
    ax1.set_xlabel('Word Rank', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title('Top-k Sampling (k=10)', fontsize=14, fontweight='bold')
    ax1.set_xlim(-1, 30)
    
    # Add annotation
    selected_mass = sum(probs[:k])
    ax1.text(0.5, 0.9, f'Selected probability mass: {selected_mass:.2f}',
             transform=ax1.transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    # Top-p visualization
    p = 0.9
    cumsum = np.cumsum(probs)
    cutoff_idx = np.argmax(cumsum >= p) + 1
    
    ax2.bar(range(vocab_size), probs, color='lightgreen', alpha=0.7)
    ax2.bar(range(cutoff_idx), probs[:cutoff_idx], color='darkgreen', alpha=0.9)
    ax2.axvline(x=cutoff_idx-0.5, color='red', linestyle='--', linewidth=2)
    ax2.text(cutoff_idx, max(probs)*0.8, f'Top-p={p} cutoff', fontsize=11, 
             color='red', rotation=90, va='bottom')
    
    # Show cumulative line
    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(vocab_size), cumsum, 'orange', linewidth=2, label='Cumulative')
    ax2_twin.axhline(y=p, color='orange', linestyle=':', alpha=0.7)
    ax2_twin.set_ylabel('Cumulative Probability', fontsize=12, color='orange')
    ax2_twin.set_ylim(0, 1.05)
    
    ax2.set_xlabel('Word Rank', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title(f'Top-p (Nucleus) Sampling (p={p})', fontsize=14, fontweight='bold')
    ax2.set_xlim(-1, 30)
    
    # Add annotation
    ax2.text(0.5, 0.9, f'Dynamic cutoff at {cutoff_idx} words',
             transform=ax2.transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    plt.suptitle('Top-k vs Top-p: Fixed vs Dynamic Vocabulary', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/sampling_strategies.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Decoding Quality Metrics
def plot_decoding_quality_metrics():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Methods
    methods = ['Greedy', 'Beam-5', 'Top-k=40', 'Top-p=0.9', 'Temp=1.2']
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'cyan']
    
    # 1. Quality vs Diversity trade-off
    quality_scores = [0.92, 0.88, 0.75, 0.82, 0.65]
    diversity_scores = [0.15, 0.25, 0.60, 0.72, 0.85]
    
    ax1.scatter(diversity_scores, quality_scores, s=200, c=colors, alpha=0.7, 
                edgecolors='black', linewidth=2)
    
    for i, method in enumerate(methods):
        ax1.annotate(method, (diversity_scores[i], quality_scores[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax1.set_xlabel('Diversity Score', fontsize=12)
    ax1.set_ylabel('Quality Score', fontsize=12)
    ax1.set_title('Quality vs Diversity Trade-off', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Ideal region
    circle = plt.Circle((0.7, 0.82), 0.08, color='green', fill=False, 
                       linewidth=2, linestyle='--')
    ax1.add_patch(circle)
    
    # 2. Repetition rates
    repetition_rates = [45, 25, 10, 8, 15]
    bars = ax2.bar(methods, repetition_rates, color=colors, alpha=0.7)
    ax2.set_ylabel('Repetition Rate (%)', fontsize=12)
    ax2.set_title('Repetition in Generated Text', fontsize=14, fontweight='bold')
    ax2.axhline(y=10, color='green', linestyle='--', alpha=0.5)
    ax2.text(2, 12, 'Acceptable level', ha='center', fontsize=9, color='green')
    
    # 3. Human preference
    human_pref = [15, 20, 25, 35, 5]
    ax3.pie(human_pref, labels=methods, colors=colors, autopct='%1.1f%%',
            startangle=90)
    ax3.set_title('Human Preference Study', fontsize=14, fontweight='bold')
    
    # 4. Task-specific performance
    tasks = ['Translation', 'Dialogue', 'Story', 'Code']
    greedy_perf = [95, 60, 40, 85]
    sampling_perf = [75, 85, 90, 70]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    ax4.bar(x - width/2, greedy_perf, width, label='Greedy/Beam', color='coral')
    ax4.bar(x + width/2, sampling_perf, width, label='Sampling', color='lightblue')
    
    ax4.set_xlabel('Task Type', fontsize=12)
    ax4.set_ylabel('Performance Score', fontsize=12)
    ax4.set_title('Best Strategy Depends on Task', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(tasks)
    ax4.legend()
    ax4.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('Decoding Strategy Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/decoding_quality_metrics.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 6. Decoding Selection Guide
def plot_decoding_selection_guide():
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Task categories with recommended settings
    tasks = [
        ('Factual Q&A', 'Greedy', 'T=0.1', 'Maximum accuracy', 0.1, 0.9),
        ('Code Generation', 'Beam-5', 'T=0.2, p=0.95', 'Syntactic correctness', 0.2, 0.8),
        ('Translation', 'Beam-4', 'T=0.3', 'Preserve meaning', 0.2, 0.7),
        ('Summarization', 'Top-p=0.9', 'T=0.5', 'Balance accuracy/fluency', 0.4, 0.6),
        ('Dialogue', 'Top-p=0.9', 'T=0.7', 'Natural conversation', 0.6, 0.5),
        ('Creative Writing', 'Top-k=50', 'T=0.9, p=0.95', 'Maximum creativity', 0.8, 0.3),
        ('Poetry', 'Top-p=0.95', 'T=1.0+', 'Artistic expression', 0.9, 0.2),
    ]
    
    # Create flowchart-style guide
    y_pos = 0.9
    for task, method, params, goal, creativity, accuracy in tasks:
        # Task box
        task_box = FancyBboxPatch((0.05, y_pos-0.05), 0.2, 0.08,
                                 boxstyle="round,pad=0.02",
                                 facecolor='lightblue',
                                 edgecolor='black',
                                 linewidth=2)
        ax.add_patch(task_box)
        ax.text(0.15, y_pos, task, ha='center', va='center', 
                fontsize=11, fontweight='bold')
        
        # Arrow
        ax.arrow(0.26, y_pos, 0.08, 0, head_width=0.02, head_length=0.02,
                fc='gray', ec='gray')
        
        # Method box
        method_box = FancyBboxPatch((0.35, y_pos-0.05), 0.15, 0.08,
                                   boxstyle="round,pad=0.02",
                                   facecolor='lightgreen',
                                   edgecolor='black',
                                   linewidth=2)
        ax.add_patch(method_box)
        ax.text(0.425, y_pos, method, ha='center', va='center', fontsize=10)
        
        # Parameters
        ax.text(0.55, y_pos, params, ha='left', va='center', fontsize=9,
                style='italic')
        
        # Goal
        ax.text(0.75, y_pos, goal, ha='left', va='center', fontsize=9)
        
        # Meters
        # Creativity meter
        meter_x = 0.9
        ax.add_patch(Rectangle((meter_x, y_pos-0.02), 0.08, 0.04, 
                              facecolor='lightgray', edgecolor='black'))
        ax.add_patch(Rectangle((meter_x, y_pos-0.02), 0.08*creativity, 0.04, 
                              facecolor='orange', alpha=0.7))
        
        y_pos -= 0.12
    
    # Add legend
    ax.text(0.9, 0.95, 'Creativity', ha='center', fontsize=10, fontweight='bold')
    
    # Title
    ax.text(0.5, 0.98, 'Decoding Strategy Selection Guide', 
            ha='center', fontsize=16, fontweight='bold')
    
    # General advice
    ax.text(0.5, 0.05, 'General Rule: Start conservative, increase randomness until output quality drops',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/decoding_selection_guide.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all Week 9 visualizations
if __name__ == "__main__":
    print("Generating Week 9 Decoding Strategies visualizations...")
    plot_decoding_selection_guide()
    print("- Decoding decision tree/guide created")
    plot_temperature_effects()
    print("- Output probability distributions created")
    plot_decoding_quality_metrics()
    print("- Evaluation scores comparison created")
    print("\nWeek 9 visualizations completed!")