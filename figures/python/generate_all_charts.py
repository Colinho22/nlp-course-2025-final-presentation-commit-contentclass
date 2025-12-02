"""
Generate all charts for NLP Final Lecture: The Next Frontier
Charts are saved as PDF for inclusion in Beamer slides.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import os

# Set up output directory
OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color scheme matching Beamer template
COLORS = {
    'mlpurple': '#3333B2',
    'mlblue': '#0066CC',
    'mllavender': '#ADADE0',
    'mllavender2': '#C1C1E8',
    'mllavender3': '#CCCCEB',
    'mlorange': '#FF7F0E',
    'mlgreen': '#2CA02C',
    'mlred': '#D62728',
    'mlgray': '#7F7F7F',
    'lightgray': '#F0F0F0',
}

# Font settings
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14


def create_rag_architecture():
    """Create RAG pipeline architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Box style
    box_style = dict(boxstyle='round,pad=0.3', facecolor=COLORS['mllavender2'],
                     edgecolor=COLORS['mlpurple'], linewidth=2)

    # Main components (left to right flow)
    components = [
        (1.5, 4, 'User\nQuery'),
        (4, 4, 'Embedding\nModel'),
        (7, 6, 'Vector\nDatabase'),
        (7, 2, 'Top-K\nDocuments'),
        (10, 4, 'LLM'),
        (12.5, 4, 'Grounded\nAnswer'),
    ]

    for x, y, label in components:
        bbox = FancyBboxPatch((x-0.9, y-0.6), 1.8, 1.2,
                              boxstyle='round,pad=0.1',
                              facecolor=COLORS['mllavender2'],
                              edgecolor=COLORS['mlpurple'],
                              linewidth=2)
        ax.add_patch(bbox)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold')

    # Arrows
    arrow_style = dict(arrowstyle='->', color=COLORS['mlpurple'], lw=2,
                       connectionstyle='arc3,rad=0')

    # Query to Embedding
    ax.annotate('', xy=(3.1, 4), xytext=(2.4, 4),
                arrowprops=dict(arrowstyle='->', color=COLORS['mlpurple'], lw=2))

    # Embedding to Vector DB
    ax.annotate('', xy=(6.1, 5.7), xytext=(4.9, 4.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['mlpurple'], lw=2))
    ax.text(5.2, 5.3, 'Query\nVector', fontsize=10, ha='center', color=COLORS['mlgray'])

    # Vector DB to Documents
    ax.annotate('', xy=(7, 2.6), xytext=(7, 5.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['mlpurple'], lw=2))
    ax.text(7.5, 4, 'Similarity\nSearch', fontsize=10, ha='left', color=COLORS['mlgray'])

    # Documents + Query to LLM
    ax.annotate('', xy=(9.1, 4.2), xytext=(7.9, 2.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['mlpurple'], lw=2))
    ax.annotate('', xy=(9.1, 3.8), xytext=(4.9, 3.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['mlorange'], lw=2, ls='--'))
    ax.text(7, 3.3, 'Context +\nQuery', fontsize=10, ha='center', color=COLORS['mlgray'])

    # LLM to Answer
    ax.annotate('', xy=(11.6, 4), xytext=(10.9, 4),
                arrowprops=dict(arrowstyle='->', color=COLORS['mlpurple'], lw=2))

    # Title
    ax.text(7, 7.5, 'RAG (Retrieval-Augmented Generation) Architecture',
            fontsize=18, ha='center', fontweight='bold', color=COLORS['mlpurple'])

    # Legend
    ax.plot([], [], '-', color=COLORS['mlpurple'], lw=2, label='Data flow')
    ax.plot([], [], '--', color=COLORS['mlorange'], lw=2, label='Original query (preserved)')
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rag_architecture.pdf'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print('Created: rag_architecture.pdf')


def create_agent_loop():
    """Create ReAct agent loop visualization."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Central cycle - four main states
    center_x, center_y = 6, 5.5
    radius = 2.5

    states = [
        (center_x, center_y + radius, 'THOUGHT', 'Reasoning about\nwhat to do next'),
        (center_x + radius, center_y, 'ACTION', 'Execute tool\nor respond'),
        (center_x, center_y - radius, 'OBSERVATION', 'Receive feedback\nfrom environment'),
        (center_x - radius, center_y, 'UPDATE', 'Update internal\nstate/memory'),
    ]

    # Draw state boxes
    for x, y, title, desc in states:
        # Main box
        bbox = FancyBboxPatch((x-1.2, y-0.8), 2.4, 1.6,
                              boxstyle='round,pad=0.1',
                              facecolor=COLORS['mllavender2'],
                              edgecolor=COLORS['mlpurple'],
                              linewidth=2)
        ax.add_patch(bbox)
        ax.text(x, y+0.3, title, ha='center', va='center',
                fontsize=12, fontweight='bold', color=COLORS['mlpurple'])
        ax.text(x, y-0.3, desc, ha='center', va='center',
                fontsize=9, color=COLORS['mlgray'])

    # Draw circular arrows connecting states
    arrow_positions = [
        ((center_x+0.8, center_y+radius-0.5), (center_x+radius-0.5, center_y+0.8)),  # Thought -> Action
        ((center_x+radius-0.5, center_y-0.8), (center_x+0.8, center_y-radius+0.5)),  # Action -> Observation
        ((center_x-0.8, center_y-radius+0.5), (center_x-radius+0.5, center_y-0.8)),  # Observation -> Update
        ((center_x-radius+0.5, center_y+0.8), (center_x-0.8, center_y+radius-0.5)),  # Update -> Thought
    ]

    for start, end in arrow_positions:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=COLORS['mlpurple'],
                                    lw=2.5, connectionstyle='arc3,rad=0.3'))

    # Tools panel on the right
    tools_x = 10.5
    tools_y = 5.5

    ax.text(tools_x, tools_y + 2, 'TOOLS', ha='center', fontsize=12,
            fontweight='bold', color=COLORS['mlpurple'])

    tools = ['Web Search', 'Calculator', 'Code Exec', 'File I/O']
    for i, tool in enumerate(tools):
        y_pos = tools_y + 1 - i * 0.8
        bbox = FancyBboxPatch((tools_x-0.8, y_pos-0.25), 1.6, 0.5,
                              boxstyle='round,pad=0.05',
                              facecolor=COLORS['mllavender3'],
                              edgecolor=COLORS['mlblue'],
                              linewidth=1)
        ax.add_patch(bbox)
        ax.text(tools_x, y_pos, tool, ha='center', va='center', fontsize=9)

    # Arrow from Action to Tools
    ax.annotate('', xy=(tools_x-0.8, tools_y), xytext=(center_x+radius+1.2, center_y),
                arrowprops=dict(arrowstyle='<->', color=COLORS['mlblue'],
                                lw=2, connectionstyle='arc3,rad=0'))

    # Final Answer box at bottom
    ax.text(center_x, 1.2, 'FINAL ANSWER', ha='center', fontsize=12,
            fontweight='bold', color=COLORS['mlgreen'])
    bbox = FancyBboxPatch((center_x-1.5, 0.5), 3, 1,
                          boxstyle='round,pad=0.1',
                          facecolor='#E8F5E9',
                          edgecolor=COLORS['mlgreen'],
                          linewidth=2)
    ax.add_patch(bbox)
    ax.text(center_x, 1, 'Return response\nto user', ha='center', va='center', fontsize=9)

    # Dashed arrow from Action to Final Answer
    ax.annotate('', xy=(center_x+0.5, 1.5), xytext=(center_x+radius-0.5, center_y-1),
                arrowprops=dict(arrowstyle='->', color=COLORS['mlgreen'],
                                lw=2, ls='--', connectionstyle='arc3,rad=-0.3'))
    ax.text(center_x+2.2, 2.5, 'When task\ncomplete', fontsize=9,
            color=COLORS['mlgreen'], ha='center')

    # Title
    ax.text(6, 9.5, 'ReAct Agent Loop: Reasoning + Acting',
            fontsize=18, ha='center', fontweight='bold', color=COLORS['mlpurple'])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'agent_loop.pdf'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print('Created: agent_loop.pdf')


def create_test_time_scaling():
    """Create test-time compute scaling visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # X-axis: Compute (log scale)
    x = np.logspace(0, 4, 100)

    # Curves representing different scaling strategies
    # Pre-training scaling (logarithmic growth)
    y_pretrain = 50 + 12 * np.log10(x)

    # Test-time scaling on easy problems (saturates quickly)
    y_testtime_easy = 75 + 15 * (1 - np.exp(-x/50))

    # Test-time scaling on hard problems (continues to improve)
    y_testtime_hard = 40 + 25 * np.log10(x + 1)

    # Combined optimal (test + train)
    y_combined = 60 + 18 * np.log10(x)

    ax.semilogx(x, y_pretrain, '-', color=COLORS['mlblue'], lw=3,
                label='Pre-training scaling (more parameters)')
    ax.semilogx(x, y_testtime_easy, '--', color=COLORS['mlgreen'], lw=3,
                label='Test-time scaling (easy tasks)')
    ax.semilogx(x, y_testtime_hard, '-.', color=COLORS['mlorange'], lw=3,
                label='Test-time scaling (hard tasks)')
    ax.semilogx(x, y_combined, ':', color=COLORS['mlpurple'], lw=3,
                label='Combined optimal')

    # Mark crossover points
    crossover_x = 100
    ax.axvline(x=crossover_x, color=COLORS['mlgray'], ls=':', alpha=0.5)
    ax.text(crossover_x*1.2, 95, 'Crossover:\nTest-time > Pre-train\nfor hard problems',
            fontsize=10, color=COLORS['mlgray'])

    ax.set_xlabel('Compute (FLOPs, log scale)', fontsize=14)
    ax.set_ylabel('Performance (% accuracy)', fontsize=14)
    ax.set_title('Test-Time vs Pre-Training Compute Scaling',
                 fontsize=18, fontweight='bold', color=COLORS['mlpurple'])

    ax.set_ylim(30, 100)
    ax.set_xlim(1, 10000)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add insight box
    ax.text(0.02, 0.98, 'Key Insight: For hard problems,\nletting the model "think longer"\ncan be more effective than\nmaking it bigger.',
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor=COLORS['mllavender3'], alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'test_time_scaling.pdf'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print('Created: test_time_scaling.pdf')


def create_rlhf_vs_dpo():
    """Create RLHF vs DPO pipeline comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

    # Left panel: RLHF
    ax1 = axes[0]
    ax1.text(5, 9.5, 'RLHF Pipeline', fontsize=16, ha='center',
             fontweight='bold', color=COLORS['mlpurple'])
    ax1.text(5, 8.8, '(3 stages, 3 models)', fontsize=11, ha='center',
             color=COLORS['mlgray'])

    rlhf_stages = [
        (5, 7.5, 'Stage 1: SFT', 'Supervised\nFine-Tuning'),
        (5, 5.5, 'Stage 2: RM', 'Reward Model\nTraining'),
        (5, 3.5, 'Stage 3: PPO', 'Policy\nOptimization'),
        (5, 1.5, 'Aligned Model', ''),
    ]

    for x, y, title, desc in rlhf_stages:
        if y == 1.5:  # Final output
            color = COLORS['mlgreen']
            facecolor = '#E8F5E9'
        else:
            color = COLORS['mlpurple']
            facecolor = COLORS['mllavender2']

        bbox = FancyBboxPatch((x-1.8, y-0.6), 3.6, 1.2,
                              boxstyle='round,pad=0.1',
                              facecolor=facecolor,
                              edgecolor=color,
                              linewidth=2)
        ax1.add_patch(bbox)
        ax1.text(x, y+0.2 if desc else y, title, ha='center', va='center',
                fontsize=11, fontweight='bold', color=color)
        if desc:
            ax1.text(x, y-0.25, desc, ha='center', va='center',
                    fontsize=9, color=COLORS['mlgray'])

    # Arrows for RLHF
    for i in range(3):
        start_y = 7.5 - i * 2 - 0.6
        end_y = 7.5 - (i+1) * 2 + 0.6
        ax1.annotate('', xy=(5, end_y), xytext=(5, start_y),
                    arrowprops=dict(arrowstyle='->', color=COLORS['mlpurple'], lw=2))

    # Side annotations for RLHF
    ax1.text(8, 7.5, 'Human\ndemos', fontsize=9, ha='left', color=COLORS['mlblue'])
    ax1.annotate('', xy=(6.8, 7.5), xytext=(7.8, 7.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['mlblue'], lw=1.5))

    ax1.text(8, 5.5, 'Preference\npairs', fontsize=9, ha='left', color=COLORS['mlblue'])
    ax1.annotate('', xy=(6.8, 5.5), xytext=(7.8, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['mlblue'], lw=1.5))

    ax1.text(8, 3.5, 'RM scores\n+ KL penalty', fontsize=9, ha='left', color=COLORS['mlblue'])
    ax1.annotate('', xy=(6.8, 3.5), xytext=(7.8, 3.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['mlblue'], lw=1.5))

    # Right panel: DPO
    ax2 = axes[1]
    ax2.text(5, 9.5, 'DPO Pipeline', fontsize=16, ha='center',
             fontweight='bold', color=COLORS['mlpurple'])
    ax2.text(5, 8.8, '(2 stages, 1 model)', fontsize=11, ha='center',
             color=COLORS['mlgray'])

    dpo_stages = [
        (5, 7, 'Stage 1: SFT', 'Supervised\nFine-Tuning'),
        (5, 4, 'Stage 2: DPO', 'Direct Preference\nOptimization'),
        (5, 1.5, 'Aligned Model', ''),
    ]

    for x, y, title, desc in dpo_stages:
        if y == 1.5:  # Final output
            color = COLORS['mlgreen']
            facecolor = '#E8F5E9'
        else:
            color = COLORS['mlpurple']
            facecolor = COLORS['mllavender2']

        bbox = FancyBboxPatch((x-1.8, y-0.6), 3.6, 1.2,
                              boxstyle='round,pad=0.1',
                              facecolor=facecolor,
                              edgecolor=color,
                              linewidth=2)
        ax2.add_patch(bbox)
        ax2.text(x, y+0.2 if desc else y, title, ha='center', va='center',
                fontsize=11, fontweight='bold', color=color)
        if desc:
            ax2.text(x, y-0.25, desc, ha='center', va='center',
                    fontsize=9, color=COLORS['mlgray'])

    # Arrows for DPO
    ax2.annotate('', xy=(5, 4.6), xytext=(5, 6.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['mlpurple'], lw=2))
    ax2.annotate('', xy=(5, 2.1), xytext=(5, 3.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['mlpurple'], lw=2))

    # Side annotations for DPO
    ax2.text(8, 7, 'Human\ndemos', fontsize=9, ha='left', color=COLORS['mlblue'])
    ax2.annotate('', xy=(6.8, 7), xytext=(7.8, 7),
                arrowprops=dict(arrowstyle='->', color=COLORS['mlblue'], lw=1.5))

    ax2.text(8, 4, 'Preference\npairs only', fontsize=9, ha='left', color=COLORS['mlblue'])
    ax2.annotate('', xy=(6.8, 4), xytext=(7.8, 4),
                arrowprops=dict(arrowstyle='->', color=COLORS['mlblue'], lw=1.5))

    # Crossed out "Reward Model"
    ax2.text(2, 5.5, 'No Reward\nModel!', fontsize=11, ha='center',
             color=COLORS['mlgreen'], fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor=COLORS['mlgreen']))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rlhf_vs_dpo.pdf'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print('Created: rlhf_vs_dpo.pdf')


def create_convergence_diagram():
    """Create three pillars convergence diagram."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Three pillars at top
    pillars = [
        (3, 7, 'RAG + Agents', 'USEFUL', 'Knowledge &\nAction'),
        (7, 7, 'Reasoning', 'SMART', 'Chain-of-Thought &\nTest-Time Compute'),
        (11, 7, 'Alignment', 'SAFE', 'RLHF, DPO &\nConstitutional AI'),
    ]

    pillar_colors = [COLORS['mlblue'], COLORS['mlorange'], COLORS['mlgreen']]

    for (x, y, title, subtitle, desc), color in zip(pillars, pillar_colors):
        # Main pillar box
        bbox = FancyBboxPatch((x-1.5, y-1), 3, 2,
                              boxstyle='round,pad=0.1',
                              facecolor=COLORS['mllavender2'],
                              edgecolor=color,
                              linewidth=3)
        ax.add_patch(bbox)
        ax.text(x, y+0.5, title, ha='center', va='center',
                fontsize=14, fontweight='bold', color=color)
        ax.text(x, y, subtitle, ha='center', va='center',
                fontsize=18, fontweight='bold', color=color)
        ax.text(x, y-0.6, desc, ha='center', va='center',
                fontsize=9, color=COLORS['mlgray'])

    # Convergence arrows pointing down
    for x, color in zip([3, 7, 11], pillar_colors):
        ax.annotate('', xy=(7, 3.5), xytext=(x, 5.9),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=3, connectionstyle='arc3,rad=0'))

    # Central convergence box
    bbox = FancyBboxPatch((4, 1.5), 6, 2,
                          boxstyle='round,pad=0.2',
                          facecolor=COLORS['mlpurple'],
                          edgecolor=COLORS['mlpurple'],
                          linewidth=3)
    ax.add_patch(bbox)
    ax.text(7, 2.9, 'Modern AI Systems', ha='center', va='center',
            fontsize=16, fontweight='bold', color='white')
    ax.text(7, 2.2, 'Useful + Smart + Safe', ha='center', va='center',
            fontsize=14, color=COLORS['mllavender2'])

    # Title
    ax.text(7, 8.5, 'The Convergence: Three Pillars of Modern NLP',
            fontsize=18, ha='center', fontweight='bold', color=COLORS['mlpurple'])

    # Examples at bottom
    examples_text = 'Examples: ChatGPT, Claude, GPT-4, Gemini, DeepSeek-R1'
    ax.text(7, 0.5, examples_text, ha='center', fontsize=11,
            color=COLORS['mlgray'], style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'convergence_diagram.pdf'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print('Created: convergence_diagram.pdf')


def create_course_journey():
    """Create course journey timeline."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Timeline
    weeks = [
        (1, 'Weeks 1-2', 'N-grams\nEmbeddings'),
        (3, 'Week 3', 'RNN\nLSTM'),
        (5, 'Weeks 4-5', 'Seq2Seq\nTransformers'),
        (7, 'Weeks 6-7', 'BERT/GPT\nAdvanced'),
        (9, 'Weeks 8-10', 'Tokenization\nDecoding\nFine-tuning'),
        (11, 'Weeks 11-12', 'Efficiency\nEthics'),
        (13, 'TODAY', 'The\nFrontier'),
    ]

    # Draw timeline line
    ax.plot([0.5, 13.5], [3, 3], '-', color=COLORS['mllavender'], lw=4)

    # Draw week boxes
    for x, label, content in weeks:
        if label == 'TODAY':
            color = COLORS['mlgreen']
            facecolor = '#E8F5E9'
        else:
            color = COLORS['mlpurple']
            facecolor = COLORS['mllavender2']

        # Circle on timeline
        circle = Circle((x, 3), 0.15, facecolor=color, edgecolor=color)
        ax.add_patch(circle)

        # Box above
        bbox = FancyBboxPatch((x-0.8, 3.5), 1.6, 1.8,
                              boxstyle='round,pad=0.05',
                              facecolor=facecolor,
                              edgecolor=color,
                              linewidth=2)
        ax.add_patch(bbox)
        ax.text(x, 4.9, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color=color)
        ax.text(x, 4.1, content, ha='center', va='center',
                fontsize=8, color=COLORS['mlgray'])

    # Title
    ax.text(7, 5.7, 'Your NLP Journey This Semester',
            fontsize=16, ha='center', fontweight='bold', color=COLORS['mlpurple'])

    # Bottom label
    ax.text(7, 1.5, 'From predicting the next word... to building AI that is USEFUL, SMART, and SAFE',
            fontsize=12, ha='center', color=COLORS['mlgray'], style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'course_journey.pdf'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print('Created: course_journey.pdf')


def create_ai_timeline():
    """Create AI breakthroughs timeline (2022-2025)."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Timeline events
    events = [
        (1.5, 'Nov 2022', 'ChatGPT\nLaunches', '100M users\nin 2 months'),
        (4, 'Mar 2023', 'GPT-4', 'Passes bar exam,\nmultimodal'),
        (6.5, 'Dec 2023', 'Gemini', 'Google enters\nthe race'),
        (9, 'Sep 2024', 'o1', 'Reasoning\nbreakthrough'),
        (11.5, 'Jan 2025', 'DeepSeek-R1', 'Open-source\nreasoning'),
    ]

    # Draw timeline line
    ax.plot([0.5, 13], [3.5, 3.5], '-', color=COLORS['mllavender'], lw=6)

    for i, (x, date, title, desc) in enumerate(events):
        # Alternate above/below
        y_offset = 1.8 if i % 2 == 0 else -1.8
        y_box = 3.5 + y_offset

        # Vertical connector
        ax.plot([x, x], [3.5, y_box - 0.7 * np.sign(y_offset)], '-',
                color=COLORS['mlpurple'], lw=2)

        # Circle on timeline
        circle = Circle((x, 3.5), 0.2, facecolor=COLORS['mlpurple'],
                        edgecolor=COLORS['mlpurple'])
        ax.add_patch(circle)

        # Event box
        bbox = FancyBboxPatch((x-1.2, y_box-0.7), 2.4, 1.4,
                              boxstyle='round,pad=0.1',
                              facecolor=COLORS['mllavender2'],
                              edgecolor=COLORS['mlpurple'],
                              linewidth=2)
        ax.add_patch(bbox)

        # Date above box
        date_y = y_box + 0.9 if y_offset > 0 else y_box - 0.9
        ax.text(x, date_y, date, ha='center', va='center',
                fontsize=10, fontweight='bold', color=COLORS['mlblue'])

        ax.text(x, y_box+0.2, title, ha='center', va='center',
                fontsize=11, fontweight='bold', color=COLORS['mlpurple'])
        ax.text(x, y_box-0.3, desc, ha='center', va='center',
                fontsize=8, color=COLORS['mlgray'])

    # Title
    ax.text(7, 6.5, 'The 18 Months That Changed AI',
            fontsize=18, ha='center', fontweight='bold', color=COLORS['mlpurple'])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ai_timeline.pdf'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print('Created: ai_timeline.pdf')


if __name__ == '__main__':
    print(f'Generating charts in: {OUTPUT_DIR}')
    print('=' * 50)

    create_rag_architecture()
    create_agent_loop()
    create_test_time_scaling()
    create_rlhf_vs_dpo()
    create_convergence_diagram()
    create_course_journey()
    create_ai_timeline()

    print('=' * 50)
    print('All charts generated successfully!')
