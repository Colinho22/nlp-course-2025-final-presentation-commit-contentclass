import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrow, Ellipse, Wedge
import seaborn as sns
import os
import networkx as nx

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# 1. AI Ethics Landscape
def plot_ai_ethics_landscape():
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create network of ethical concerns
    G = nx.Graph()
    
    # Core ethical areas
    core_areas = {
        'Fairness': (0.5, 0.8),
        'Privacy': (0.2, 0.6),
        'Safety': (0.8, 0.6),
        'Transparency': (0.2, 0.3),
        'Accountability': (0.8, 0.3),
        'Sustainability': (0.5, 0.1)
    }
    
    # Add nodes
    for area, pos in core_areas.items():
        G.add_node(area, pos=pos)
    
    # Add connections
    connections = [
        ('Fairness', 'Transparency'),
        ('Fairness', 'Accountability'),
        ('Privacy', 'Safety'),
        ('Privacy', 'Transparency'),
        ('Safety', 'Accountability'),
        ('Transparency', 'Accountability'),
        ('Sustainability', 'Fairness'),
        ('Sustainability', 'Safety')
    ]
    
    G.add_edges_from(connections)
    
    # Draw network
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=2, edge_color='gray')
    
    # Draw nodes with different sizes based on importance
    node_sizes = {
        'Fairness': 3000,
        'Privacy': 2500,
        'Safety': 2500,
        'Transparency': 2000,
        'Accountability': 2000,
        'Sustainability': 1500
    }
    
    node_colors = {
        'Fairness': 'lightblue',
        'Privacy': 'lightgreen',
        'Safety': 'lightcoral',
        'Transparency': 'lightyellow',
        'Accountability': 'lightgray',
        'Sustainability': 'lightseagreen'
    }
    
    for node, (x, y) in pos.items():
        circle = Circle((x, y), radius=0.08, 
                       facecolor=node_colors[node],
                       edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, node, ha='center', va='center', 
                fontsize=11, fontweight='bold')
    
    # Add specific challenges around each area
    challenges = {
        'Fairness': ['Gender bias', 'Racial bias', 'Economic bias'],
        'Privacy': ['Data leaks', 'Re-identification', 'Surveillance'],
        'Safety': ['Misuse', 'Adversarial attacks', 'Alignment'],
        'Transparency': ['Black box', 'Explainability', 'Interpretability'],
        'Accountability': ['Liability', 'Governance', 'Regulation'],
        'Sustainability': ['Carbon footprint', 'Resource use', 'E-waste']
    }
    
    for area, (x, y) in core_areas.items():
        for i, challenge in enumerate(challenges[area]):
            angle = 2 * np.pi * i / len(challenges[area])
            cx = x + 0.15 * np.cos(angle)
            cy = y + 0.15 * np.sin(angle)
            ax.text(cx, cy, challenge, ha='center', va='center', 
                   fontsize=8, style='italic',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    ax.set_title('The AI Ethics Landscape: Interconnected Challenges', 
                fontsize=16, fontweight='bold')
    ax.text(0.5, 0.95, 'Each ethical dimension affects and is affected by others', 
            ha='center', transform=ax.transAxes, fontsize=11, style='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/ai_ethics_landscape.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Bias Sources
def plot_bias_sources():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Pipeline stages
    stages = [
        ('Historical\nBias', 0.1, 0.5, 'Society reflects\npast injustices'),
        ('Data\nCollection', 0.3, 0.5, 'Who/what\nis included?'),
        ('Annotation\nBias', 0.5, 0.5, 'Labeler\nsubjectivity'),
        ('Model\nBias', 0.7, 0.5, 'Algorithm\nchoices'),
        ('Deployment\nBias', 0.9, 0.5, 'Usage\ncontext')
    ]
    
    # Draw pipeline
    for i, (stage, x, y, desc) in enumerate(stages):
        # Stage box
        box = FancyBboxPatch((x-0.08, y-0.1), 0.16, 0.2,
                            boxstyle="round,pad=0.02",
                            facecolor='lightblue',
                            edgecolor='black',
                            linewidth=2)
        ax.add_patch(box)
        ax.text(x, y+0.05, stage, ha='center', va='center', 
                fontsize=11, fontweight='bold')
        ax.text(x, y-0.05, desc, ha='center', va='center', 
                fontsize=8, style='italic')
        
        # Arrow to next stage
        if i < len(stages) - 1:
            arrow = FancyArrow(x+0.08, y, 0.12, 0, head_width=0.03, head_length=0.02,
                              fc='gray', ec='gray', linewidth=2)
            ax.add_patch(arrow)
    
    # Add examples of bias at each stage
    examples = [
        (0.1, 0.25, 'Example: Crime data\nreflects policing bias'),
        (0.3, 0.25, 'Example: Internet text\nexcludes many voices'),
        (0.5, 0.25, 'Example: "Professional"\nmeans different things'),
        (0.7, 0.25, 'Example: Objective functions\nencode values'),
        (0.9, 0.25, 'Example: Hiring tool\namplifies inequality')
    ]
    
    for x, y, text in examples:
        ax.text(x, y, text, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    # Feedback loop
    ax.add_patch(FancyArrow(0.9, 0.4, 0, -0.15, head_width=0.02, head_length=0.02,
                           fc='red', ec='red', linewidth=2))
    ax.add_patch(FancyArrow(0.9, 0.2, -0.8, 0, head_width=0.02, head_length=0.02,
                           fc='red', ec='red', linewidth=2))
    ax.add_patch(FancyArrow(0.1, 0.2, 0, 0.15, head_width=0.02, head_length=0.02,
                           fc='red', ec='red', linewidth=2))
    
    ax.text(0.5, 0.15, 'Feedback Loop: Biased outputs reinforce societal biases', 
            ha='center', fontsize=10, color='red', fontweight='bold')
    
    ax.set_title('How Bias Enters AI Systems: From Society to Model to Impact', 
                fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 0.7)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/bias_sources.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Fairness Techniques
def plot_fairness_techniques():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Timeline of interventions
    timeline_y = 0.5
    stages = [
        ('Pre-processing', 0.2, 'lightgreen', ['Data augmentation', 'Re-sampling', 'Synthetic data']),
        ('In-processing', 0.5, 'yellow', ['Fair objectives', 'Adversarial debiasing', 'Constraints']),
        ('Post-processing', 0.8, 'lightcoral', ['Output calibration', 'Threshold optimization', 'Fair ranking'])
    ]
    
    # Draw timeline
    ax.plot([0.1, 0.9], [timeline_y, timeline_y], 'k-', linewidth=3)
    
    for stage, x, color, techniques in stages:
        # Stage marker
        circle = Circle((x, timeline_y), 0.03, facecolor=color, 
                       edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        # Stage label
        ax.text(x, timeline_y + 0.08, stage, ha='center', fontsize=12, fontweight='bold')
        
        # Techniques box
        box_height = 0.15
        box = FancyBboxPatch((x-0.1, timeline_y-0.25), 0.2, box_height,
                            boxstyle="round,pad=0.02",
                            facecolor=color,
                            alpha=0.7,
                            edgecolor='black')
        ax.add_patch(box)
        
        # List techniques
        for i, tech in enumerate(techniques):
            ax.text(x, timeline_y-0.18+i*0.04, f'• {tech}', ha='center', 
                   va='center', fontsize=8)
    
    # Add metrics
    ax.text(0.5, 0.8, 'Fairness Metrics', ha='center', fontsize=14, fontweight='bold')
    
    metrics = [
        ('Demographic Parity', 0.2, 0.7),
        ('Equal Opportunity', 0.4, 0.7),
        ('Equalized Odds', 0.6, 0.7),
        ('Individual Fairness', 0.8, 0.7)
    ]
    
    for metric, x, y in metrics:
        metric_box = FancyBboxPatch((x-0.08, y-0.03), 0.16, 0.06,
                                   boxstyle="round,pad=0.01",
                                   facecolor='lightblue',
                                   edgecolor='black')
        ax.add_patch(metric_box)
        ax.text(x, y, metric, ha='center', va='center', fontsize=9)
    
    # Trade-offs note
    ax.text(0.5, 0.1, 'Note: Different fairness metrics often conflict - perfect fairness is mathematically impossible', 
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.set_title('Fairness Interventions Across the ML Pipeline', 
                fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.85)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/fairness_techniques.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Privacy Protection
def plot_privacy_protection():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Central model
    model_box = FancyBboxPatch((0.4, 0.4), 0.2, 0.2,
                              boxstyle="round,pad=0.02",
                              facecolor='lightcoral',
                              edgecolor='black',
                              linewidth=3)
    ax.add_patch(model_box)
    ax.text(0.5, 0.5, 'Language\nModel', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # Privacy threats (incoming arrows)
    threats = [
        ('Training Data\nExtraction', 0.1, 0.7, 'red'),
        ('Membership\nInference', 0.1, 0.5, 'red'),
        ('Model\nInversion', 0.1, 0.3, 'red'),
        ('Attribute\nInference', 0.3, 0.8, 'red')
    ]
    
    for threat, x, y, color in threats:
        ax.arrow(x, y, 0.28, 0.5-y, head_width=0.03, head_length=0.02,
                fc=color, ec=color, linewidth=2, alpha=0.7)
        ax.text(x, y, threat, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))
    
    # Privacy defenses (outgoing shields)
    defenses = [
        ('Differential\nPrivacy', 0.9, 0.7, 'green'),
        ('Federated\nLearning', 0.9, 0.5, 'green'),
        ('Secure\nAggregation', 0.9, 0.3, 'green'),
        ('Data\nMinimization', 0.7, 0.8, 'green')
    ]
    
    for defense, x, y, color in defenses:
        # Shield shape
        shield = Wedge((x-0.05, y), 0.08, 180, 360, 
                      facecolor=color, alpha=0.7, edgecolor='black', linewidth=2)
        ax.add_patch(shield)
        ax.text(x, y-0.1, defense, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # Key principles
    ax.text(0.5, 0.9, 'Privacy-Preserving NLP: Protecting User Data', 
            ha='center', fontsize=16, fontweight='bold')
    
    principles = ['Collect only what\'s needed', 'Protect data in transit & rest', 
                  'Limit access & retention', 'Enable user control']
    for i, principle in enumerate(principles):
        ax.text(0.5, 0.15 - i*0.03, f'✓ {principle}', ha='center', fontsize=9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/privacy_protection.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Carbon Footprint
def plot_carbon_footprint():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Carbon emissions by model
    models = ['BERT\nBase', 'GPT-2', 'T5', 'GPT-3', 'PaLM', 'GPT-4\n(est)']
    carbon_tons = [0.65, 1.5, 47, 552, 1400, 3000]
    
    bars = ax1.bar(models, carbon_tons, color=['green', 'yellow', 'orange', 'red', 'darkred', 'purple'],
                    alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add reference lines
    ax1.axhline(y=5, color='blue', linestyle='--', alpha=0.5)
    ax1.text(2.5, 6, 'Average US car/year', ha='center', color='blue', fontsize=9)
    
    ax1.axhline(y=300, color='red', linestyle='--', alpha=0.5)
    ax1.text(2.5, 320, 'Lifetime emissions of 5 cars', ha='center', color='red', fontsize=9)
    
    ax1.set_ylabel('CO2 Emissions (tons)', fontsize=12, fontweight='bold')
    ax1.set_title('Carbon Footprint of Training Large Models', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Mitigation strategies
    strategies = ['Model\nReuse', 'Efficient\nArchitectures', 'Green\nData Centers', 
                  'Carbon\nOffsets', 'Compute\nOptimization']
    impact = [90, 70, 60, 30, 50]  # Percentage reduction potential
    
    ax2.barh(strategies, impact, color='lightgreen', alpha=0.7, 
             edgecolor='black', linewidth=2)
    
    for i, (strategy, pct) in enumerate(zip(strategies, impact)):
        ax2.text(pct + 2, i, f'{pct}%', va='center', fontweight='bold')
    
    ax2.set_xlabel('Potential Carbon Reduction (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Green AI: Mitigation Strategies', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.grid(True, axis='x', alpha=0.3)
    
    plt.suptitle('Environmental Impact of Large Language Models', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/carbon_footprint.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 6. NLP Future Timeline
def plot_nlp_future_timeline():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Timeline
    years = [2024, 2026, 2028, 2030, 2035, 2040]
    y_base = 0.5
    
    # Draw main timeline
    ax.plot([years[0]-0.5, years[-1]+0.5], [y_base, y_base], 'k-', linewidth=3)
    
    # Milestones
    milestones = [
        (2024, 'Now', ['1T parameters', 'Multimodal', 'Code generation'], 'lightblue'),
        (2026, 'Near', ['Real-time translation', 'Personal AI assistants', 'Scientific discovery'], 'lightgreen'),
        (2028, 'Soon', ['AGI debates', 'Autonomous research', 'Creative partners'], 'yellow'),
        (2030, 'Future', ['Brain interfaces?', 'Consciousness?', 'Singularity?'], 'orange'),
        (2035, 'Far', ['Post-human AI?', 'Space exploration', 'Unknown unknowns'], 'red'),
        (2040, '???', ['Beyond prediction', '...', '...'], 'purple')
    ]
    
    for year, label, items, color in milestones:
        # Year marker
        ax.plot(year, y_base, 'o', markersize=15, color=color, 
                markeredgecolor='black', markeredgewidth=2)
        
        # Year label
        ax.text(year, y_base-0.08, str(year), ha='center', fontsize=11, fontweight='bold')
        ax.text(year, y_base-0.12, label, ha='center', fontsize=9, style='italic')
        
        # Capabilities box
        box_y = y_base + 0.15 if milestones.index((year, label, items, color)) % 2 == 0 else y_base - 0.35
        
        box = FancyBboxPatch((year-0.15, box_y), 0.3, 0.15,
                            boxstyle="round,pad=0.02",
                            facecolor=color,
                            alpha=0.7,
                            edgecolor='black')
        ax.add_patch(box)
        
        # List capabilities
        for i, item in enumerate(items):
            ax.text(year, box_y + 0.12 - i*0.04, item, ha='center', 
                   va='center', fontsize=8)
        
        # Connect to timeline
        if box_y > y_base:
            ax.plot([year, year], [y_base+0.02, box_y], 'k--', alpha=0.5)
        else:
            ax.plot([year, year], [y_base-0.02, box_y+0.15], 'k--', alpha=0.5)
    
    # Uncertainty cone
    uncertainty_x = np.array([2024, 2030, 2040])
    uncertainty_y_upper = np.array([0.05, 0.2, 0.4])
    uncertainty_y_lower = np.array([-0.05, -0.2, -0.4])
    
    ax.fill_between(uncertainty_x, y_base + uncertainty_y_upper, 
                    y_base + uncertainty_y_lower, alpha=0.1, color='gray')
    ax.text(2032, y_base+0.25, 'Cone of Uncertainty', ha='center', 
            fontsize=10, style='italic', color='gray')
    
    ax.set_title('The Future of NLP: From Predictions to Possibilities', 
                fontsize=16, fontweight='bold')
    ax.set_xlim(2023, 2041)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/nlp_future_timeline.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 7. AI for Good
def plot_ai_for_good():
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create hexagonal arrangement of positive applications
    applications = [
        ('Education', 0.5, 0.8, 'Personal tutors\nfor every child'),
        ('Healthcare', 0.25, 0.65, 'Diagnosis\n& treatment'),
        ('Accessibility', 0.75, 0.65, 'Equal access\nfor all'),
        ('Science', 0.2, 0.4, 'Accelerated\ndiscovery'),
        ('Environment', 0.8, 0.4, 'Climate\nsolutions'),
        ('Communication', 0.5, 0.25, 'No language\nbarriers')
    ]
    
    # Draw connections between all applications
    for i, (app1, x1, y1, _) in enumerate(applications):
        for j, (app2, x2, y2, _) in enumerate(applications[i+1:], i+1):
            ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.2, linewidth=1)
    
    # Draw application nodes
    for app, x, y, desc in applications:
        # Hexagon shape
        hexagon = plt.Circle((x, y), 0.12, facecolor='lightgreen', 
                            edgecolor='darkgreen', linewidth=3, alpha=0.8)
        ax.add_patch(hexagon)
        
        ax.text(x, y+0.02, app, ha='center', va='center', 
                fontsize=11, fontweight='bold')
        ax.text(x, y-0.02, desc, ha='center', va='center', 
                fontsize=8, style='italic')
    
    # Central vision
    center_circle = Circle((0.5, 0.5), 0.08, facecolor='gold', 
                          edgecolor='black', linewidth=3)
    ax.add_patch(center_circle)
    ax.text(0.5, 0.5, 'AI for\nGood', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # Impact statistics
    stats = [
        '1B+ people helped',
        '100+ languages',
        '50% cost reduction',
        '10x faster research'
    ]
    
    for i, stat in enumerate(stats):
        ax.text(0.1 + i*0.2, 0.1, stat, ha='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.set_title('Positive Vision: AI Empowering Humanity', 
                fontsize=16, fontweight='bold')
    ax.text(0.5, 0.95, 'Technology amplifying human potential for good', 
            ha='center', transform=ax.transAxes, fontsize=11, style='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/ai_for_good.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all Week 12 visualizations
if __name__ == "__main__":
    print("Generating Week 12 Ethics & Future Directions visualizations...")
    plot_ai_ethics_landscape()
    print("- AI ethics landscape created")
    plot_bias_sources()
    print("- Bias sources visualization created")
    plot_fairness_techniques()
    print("- Fairness techniques diagram created")
    plot_privacy_protection()
    print("- Privacy protection visualization created")
    plot_carbon_footprint()
    print("- Carbon footprint analysis created")
    plot_nlp_future_timeline()
    print("- NLP future timeline created")
    plot_ai_for_good()
    print("- AI for good vision created")
    print("\nWeek 12 visualizations completed!")