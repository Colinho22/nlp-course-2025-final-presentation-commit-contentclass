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

# 1. Adaptation Methods Comparison
def plot_adaptation_methods():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Methods with properties (cost, performance, flexibility, time)
    methods = [
        ('Zero-shot', 0.1, 0.4, 0.9, 0.95, 'lightcoral'),
        ('Few-shot', 0.2, 0.6, 0.8, 0.9, 'orange'),
        ('Prompt\nEngineering', 0.15, 0.7, 0.85, 0.85, 'yellow'),
        ('LoRA\nFine-tuning', 0.4, 0.85, 0.6, 0.6, 'lightgreen'),
        ('Full\nFine-tuning', 0.9, 0.95, 0.3, 0.2, 'lightblue'),
        ('RLHF', 0.95, 0.98, 0.2, 0.1, 'purple')
    ]
    
    # Plot methods
    for name, cost, perf, flex, time, color in methods:
        # Size based on performance
        size = perf * 1000
        # Position based on cost vs time
        ax.scatter(cost, time, s=size, c=color, alpha=0.7, 
                  edgecolors='black', linewidth=2)
        ax.annotate(name, (cost, time), xytext=(0, -25), 
                   textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    
    # Add ideal region
    ellipse = Ellipse((0.3, 0.7), 0.3, 0.3, facecolor='green', alpha=0.1)
    ax.add_patch(ellipse)
    ax.text(0.3, 0.7, 'Sweet\nSpot', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='green')
    
    # Axes and labels
    ax.set_xlabel('Implementation Cost →', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speed to Deploy →', fontsize=14, fontweight='bold')
    ax.set_title('Adaptation Methods: Cost vs Speed vs Performance\n(Bubble size = Performance)', 
                fontsize=16, fontweight='bold')
    
    # Add performance legend
    for perf, y in [(0.4, 0.15), (0.7, 0.1), (0.95, 0.05)]:
        ax.scatter(0.85, y, s=perf*1000, c='gray', alpha=0.5)
        ax.text(0.88, y, f'{int(perf*100)}%', va='center', fontsize=9)
    ax.text(0.9, 0.2, 'Performance', ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()  # Fast at top
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/adaptation_methods.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 2. LoRA Explanation
def plot_lora_explanation():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Original weight matrix
    orig_x, orig_y = 0.1, 0.4
    orig_size = 0.25
    
    # Draw original weight matrix W
    orig_rect = FancyBboxPatch((orig_x, orig_y), orig_size, orig_size,
                              boxstyle="round,pad=0.02",
                              facecolor='lightblue',
                              edgecolor='black',
                              linewidth=2)
    ax.add_patch(orig_rect)
    ax.text(orig_x + orig_size/2, orig_y + orig_size/2, 'W\n(d×d)\nFrozen', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(orig_x + orig_size/2, orig_y - 0.05, 'Original Weights\n(e.g., 4096×4096)', 
            ha='center', fontsize=10)
    
    # LoRA decomposition
    lora_x = 0.5
    
    # Matrix A
    a_rect = FancyBboxPatch((lora_x, orig_y + 0.1), 0.05, orig_size - 0.1,
                           boxstyle="round,pad=0.01",
                           facecolor='lightcoral',
                           edgecolor='black',
                           linewidth=2)
    ax.add_patch(a_rect)
    ax.text(lora_x + 0.025, orig_y + orig_size/2, 'A\n(d×r)', 
            ha='center', va='center', fontsize=11, fontweight='bold', rotation=90)
    
    # Matrix B  
    b_rect = FancyBboxPatch((lora_x + 0.08, orig_y + 0.15), orig_size - 0.1, 0.05,
                           boxstyle="round,pad=0.01",
                           facecolor='lightgreen',
                           edgecolor='black',
                           linewidth=2)
    ax.add_patch(b_rect)
    ax.text(lora_x + 0.08 + (orig_size-0.1)/2, orig_y + 0.175, 'B (r×d)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Multiplication symbol
    ax.text(lora_x + 0.065, orig_y + orig_size/2, '×', 
            fontsize=20, ha='center', va='center')
    
    # Arrows showing flow
    ax.arrow(orig_x + orig_size + 0.02, orig_y + orig_size/2, 0.1, 0,
            head_width=0.03, head_length=0.02, fc='gray', ec='gray')
    ax.text(orig_x + orig_size + 0.07, orig_y + orig_size/2 + 0.05, 'Freeze', 
            ha='center', fontsize=10, style='italic')
    
    # Plus symbol
    ax.text(0.75, orig_y + orig_size/2, '+', fontsize=30, ha='center', va='center')
    
    # Result
    result_rect = FancyBboxPatch((0.8, orig_y), orig_size, orig_size,
                                boxstyle="round,pad=0.02",
                                facecolor='gold',
                                edgecolor='black',
                                linewidth=3)
    ax.add_patch(result_rect)
    ax.text(0.8 + orig_size/2, orig_y + orig_size/2, 'W + ΔW\n(d×d)\nAdapted', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Key insight box
    ax.text(0.5, 0.8, 'LoRA: Low-Rank Adaptation', 
            ha='center', fontsize=18, fontweight='bold')
    ax.text(0.5, 0.75, 'Instead of updating 16M parameters, update only 32K!', 
            ha='center', fontsize=12, style='italic')
    
    # Example numbers
    example_box = FancyBboxPatch((0.15, 0.05), 0.7, 0.15,
                                boxstyle="round,pad=0.02",
                                facecolor='lightyellow',
                                edgecolor='black')
    ax.add_patch(example_box)
    ax.text(0.5, 0.125, 'Example: d=4096, r=8\nOriginal: 4096×4096 = 16,777,216 parameters\n' + 
            'LoRA: (4096×8) + (8×4096) = 65,536 parameters (0.39%!)',
            ha='center', va='center', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.85)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/lora_explanation.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Prompt Patterns
def plot_prompt_patterns():
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Prompt components
    patterns = [
        ('Role Definition', 0.1, 0.8, 'lightblue', 
         'You are an expert [DOMAIN] professional with [N] years of experience...'),
        ('Task Specification', 0.1, 0.65, 'lightgreen',
         'Your task is to [SPECIFIC ACTION] for [TARGET AUDIENCE]...'),
        ('Context Provision', 0.1, 0.5, 'lightyellow',
         'Background: [RELEVANT INFORMATION]\nConstraints: [LIMITATIONS]...'),
        ('Output Format', 0.1, 0.35, 'lightcoral',
         'Provide your response in [FORMAT]:\n- Bullet points\n- Max [N] words...'),
        ('Examples (Few-shot)', 0.1, 0.2, 'lightgray',
         'Example 1: Input: ... Output: ...\nExample 2: Input: ... Output: ...')
    ]
    
    # Draw pattern boxes
    for name, x, y, color, template in patterns:
        box = FancyBboxPatch((x, y-0.06), 0.35, 0.12,
                            boxstyle="round,pad=0.02",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=2)
        ax.add_patch(box)
        ax.text(x + 0.175, y, name, ha='center', va='center', 
                fontsize=12, fontweight='bold')
        
        # Template text
        ax.text(x + 0.38, y, template, ha='left', va='center', 
                fontsize=9, style='italic')
    
    # Combination example
    combo_box = FancyBboxPatch((0.55, 0.15), 0.4, 0.75,
                              boxstyle="round,pad=0.02",
                              facecolor='lightsteelblue',
                              edgecolor='black',
                              linewidth=3)
    ax.add_patch(combo_box)
    
    ax.text(0.75, 0.85, 'Complete Example', ha='center', fontsize=14, fontweight='bold')
    
    example_text = """You are an expert data scientist with 10 years 
of experience in machine learning.

Your task is to analyze this dataset and provide 
actionable insights for business stakeholders.

Context: The company is an e-commerce platform 
with 1M daily users. We need to improve conversion.

Please provide your analysis in the following format:
1. Key findings (3-5 bullet points)
2. Recommendations (numbered list)
3. Next steps (brief paragraph)

Keep technical jargon to a minimum."""
    
    ax.text(0.57, 0.5, example_text, ha='left', va='center', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))
    
    # Title
    ax.text(0.5, 0.95, 'Effective Prompt Engineering Patterns', 
            ha='center', fontsize=16, fontweight='bold')
    
    # Best practice note
    ax.text(0.5, 0.05, 'Best Practice: Combine multiple patterns for optimal results', 
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/prompt_patterns.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Fine-tuning vs Prompting Comparison
def plot_finetuning_vs_prompting():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Metrics for comparison
    metrics = ['Accuracy', 'Cost', 'Time to Deploy', 'Flexibility', 
               'Data Required', 'Maintenance']
    
    # Scores (normalized to 0-1)
    prompting_scores = [0.7, 0.95, 0.98, 0.9, 0.95, 0.8]
    lora_scores = [0.85, 0.7, 0.7, 0.7, 0.6, 0.6]
    full_ft_scores = [0.95, 0.2, 0.3, 0.3, 0.2, 0.4]
    
    # Radar chart setup
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    prompting_scores += prompting_scores[:1]
    lora_scores += lora_scores[:1]
    full_ft_scores += full_ft_scores[:1]
    angles += angles[:1]
    
    # Plot
    ax.plot(angles, prompting_scores, 'o-', linewidth=2, label='Prompt Engineering', color='green')
    ax.fill(angles, prompting_scores, alpha=0.15, color='green')
    
    ax.plot(angles, lora_scores, 's-', linewidth=2, label='LoRA Fine-tuning', color='blue')
    ax.fill(angles, lora_scores, alpha=0.15, color='blue')
    
    ax.plot(angles, full_ft_scores, '^-', linewidth=2, label='Full Fine-tuning', color='red')
    ax.fill(angles, full_ft_scores, alpha=0.15, color='red')
    
    # Fix axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.grid(True, alpha=0.3)
    
    # Title and legend
    ax.set_title('Fine-tuning vs Prompting: Multi-dimensional Comparison', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Add insights
    insights = [
        (0.5, -0.15, 'Prompting excels at flexibility and speed'),
        (0.5, -0.2, 'LoRA balances performance and efficiency'),
        (0.5, -0.25, 'Full fine-tuning for maximum accuracy')
    ]
    
    for x, y, text in insights:
        ax.text(x, y, text, transform=ax.transAxes, ha='center', 
                fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('../figures/finetuning_vs_prompting.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 5. RLHF Process
def plot_rlhf_process():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Process steps
    steps = [
        ('1. Initial Model', 0.1, 0.7, 0.15, 0.15),
        ('2. Generate Outputs', 0.3, 0.7, 0.15, 0.15),
        ('3. Human Ranking', 0.5, 0.7, 0.15, 0.15),
        ('4. Reward Model', 0.7, 0.7, 0.15, 0.15),
        ('5. RL Training', 0.9, 0.7, 0.15, 0.15),
        ('6. Aligned Model', 0.5, 0.3, 0.2, 0.15)
    ]
    
    # Draw steps
    for i, (name, x, y, w, h) in enumerate(steps):
        if i < 5:
            color = plt.cm.viridis(i/5)
        else:
            color = 'gold'
            
        box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                            boxstyle="round,pad=0.02",
                            facecolor=color,
                            alpha=0.7,
                            edgecolor='black',
                            linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, name.split('.')[1].strip(), ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white' if i < 3 else 'black')
    
    # Arrows
    arrow_pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
    for start, end in arrow_pairs:
        x1, y1 = steps[start][1], steps[start][2]
        x2, y2 = steps[end][1], steps[end][2]
        ax.arrow(x1 + 0.075, y1, x2 - x1 - 0.15, 0,
                head_width=0.03, head_length=0.02, fc='gray', ec='gray', linewidth=2)
    
    # Feedback loop
    ax.arrow(0.9, 0.6, 0, -0.15, head_width=0.03, head_length=0.02, 
             fc='red', ec='red', linewidth=2)
    ax.arrow(0.9, 0.4, -0.35, 0, head_width=0.03, head_length=0.02, 
             fc='red', ec='red', linewidth=2)
    ax.arrow(0.5, 0.4, 0, -0.05, head_width=0.03, head_length=0.02, 
             fc='red', ec='red', linewidth=2)
    
    # Annotations
    ax.text(0.3, 0.85, 'Multiple\ncompletions', ha='center', fontsize=9, style='italic')
    ax.text(0.5, 0.85, 'Humans\nrank quality', ha='center', fontsize=9, style='italic')
    ax.text(0.7, 0.85, 'Learn human\npreferences', ha='center', fontsize=9, style='italic')
    ax.text(0.95, 0.5, 'Optimize\nfor rewards', ha='left', fontsize=9, 
            style='italic', color='red')
    
    # Title
    ax.text(0.5, 0.95, 'RLHF: Reinforcement Learning from Human Feedback', 
            ha='center', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.05, 'Iterative process aligns model outputs with human preferences', 
            ha='center', fontsize=11, style='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/rlhf_process.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 6. Adaptation Decision Tree
def plot_adaptation_decision_tree():
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Decision nodes
    nodes = {
        'root': (0.5, 0.9, 'How much labeled\ndata do you have?'),
        '<100': (0.2, 0.7, '<100 examples'),
        '100-1K': (0.5, 0.7, '100-1K examples'),
        '>1K': (0.8, 0.7, '>1K examples'),
        'prompt': (0.1, 0.5, 'Prompt\nEngineering'),
        'few_shot': (0.3, 0.5, 'Few-shot +\nPrompting'),
        'lora': (0.5, 0.5, 'LoRA\nFine-tuning'),
        'full_ft': (0.7, 0.5, 'Full\nFine-tuning'),
        'instruct': (0.9, 0.5, 'Instruction\nTuning'),
        'accuracy': (0.5, 0.3, 'Need >95%\naccuracy?'),
        'yes': (0.3, 0.1, 'Consider full\nfine-tuning'),
        'no': (0.7, 0.1, 'LoRA is\nsufficient')
    }
    
    # Draw nodes
    for key, (x, y, text) in nodes.items():
        if key == 'root':
            color = 'lightblue'
            size = 0.15
        elif key in ['<100', '100-1K', '>1K', 'accuracy']:
            color = 'lightyellow'
            size = 0.12
        elif key in ['yes', 'no']:
            color = 'lightgray'
            size = 0.1
        else:
            color = 'lightgreen'
            size = 0.1
            
        circle = Circle((x, y), size/2, facecolor=color, 
                       edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw connections
    connections = [
        ('root', '<100'), ('root', '100-1K'), ('root', '>1K'),
        ('<100', 'prompt'), ('100-1K', 'few_shot'), ('100-1K', 'lora'),
        ('>1K', 'lora'), ('>1K', 'full_ft'), ('>1K', 'instruct'),
        ('lora', 'accuracy'), ('accuracy', 'yes'), ('accuracy', 'no')
    ]
    
    for start, end in connections:
        x1, y1, _ = nodes[start]
        x2, y2, _ = nodes[end]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.5)
    
    # Add recommendations
    recs = [
        (0.1, 0.4, 'Fast iteration\nNo training cost', 'blue'),
        (0.3, 0.4, 'Good balance\nQuick results', 'blue'),
        (0.5, 0.4, 'Best overall\nEfficient & effective', 'green'),
        (0.7, 0.4, 'Maximum accuracy\nHigh cost', 'orange'),
        (0.9, 0.4, 'Generalization\nMulti-task', 'purple')
    ]
    
    for x, y, text, color in recs:
        ax.text(x, y, text, ha='center', va='top', fontsize=8, 
                style='italic', color=color)
    
    # Title
    ax.text(0.5, 0.98, 'Adaptation Strategy Decision Tree', 
            ha='center', fontsize=16, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/adaptation_decision_tree.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all Week 10 visualizations
if __name__ == "__main__":
    print("Generating Week 10 Fine-tuning & Prompt Engineering visualizations...")
    plot_lora_explanation()
    print("- Parameter efficiency (LoRA) visualization created")
    plot_adaptation_decision_tree()
    print("- Fine-tuning flowchart/decision tree created")
    plot_finetuning_vs_prompting()
    print("- Performance gains comparison created")
    print("\nWeek 10 visualizations completed!")