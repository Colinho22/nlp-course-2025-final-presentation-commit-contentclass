import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import os

# Create output directory
os.makedirs('../figures', exist_ok=True)

# Educational color scheme (minimalist gray palette)
COLOR_MAIN = '#404040'      # RGB(64,64,64)
COLOR_ACCENT = '#B4B4B4'    # RGB(180,180,180)
COLOR_LIGHT = '#F0F0F0'     # RGB(240,240,240)
COLOR_HIGHLIGHT = '#FF6B6B' # Red for emphasis

# Chart 1: Human-like Paraphrasing Visual (Discovery Hook)
def plot_human_paraphrasing():
    fig, ax = plt.subplots(figsize=(12, 7))

    # Original text box
    original = "The company reported strong financial results\nwith revenue increasing by 25% year-over-year\nand profit margins expanding to 18%."

    # Three paraphrasing styles
    extractive = "The company reported strong financial results\nwith revenue increasing by 25%\nand profit margins expanding to 18%."

    abstractive_bad = "The company reported strong financial results.\nRevenue increasing by 25% year-over-year."

    abstractive_llm = "The firm achieved impressive growth,\nwith sales up 25% and profitability\nimproving to 18% margins."

    # Draw boxes
    ax.add_patch(FancyBboxPatch((0.05, 0.7), 0.9, 0.25, boxstyle="round,pad=0.02",
                                 edgecolor=COLOR_MAIN, facecolor=COLOR_LIGHT, linewidth=2))
    ax.text(0.5, 0.92, "Original Text", ha='center', fontsize=13, weight='bold', color=COLOR_MAIN)
    ax.text(0.5, 0.82, original, ha='center', va='center', fontsize=10, color=COLOR_MAIN,
            family='monospace', linespacing=1.5)

    # Extractive (copy-paste)
    ax.add_patch(FancyBboxPatch((0.05, 0.45), 0.25, 0.2, boxstyle="round,pad=0.02",
                                 edgecolor=COLOR_ACCENT, facecolor='#FFE6E6', linewidth=1.5))
    ax.text(0.175, 0.63, "Extractive\n(Copy-Paste)", ha='center', fontsize=10, weight='bold', color=COLOR_MAIN)
    ax.text(0.175, 0.54, extractive, ha='center', va='center', fontsize=7, color=COLOR_MAIN,
            family='monospace', linespacing=1.3)
    ax.text(0.175, 0.42, "❌ Identical words", ha='center', fontsize=8, color='#CC0000', style='italic')

    # Bad abstractive
    ax.add_patch(FancyBboxPatch((0.375, 0.45), 0.25, 0.2, boxstyle="round,pad=0.02",
                                 edgecolor=COLOR_ACCENT, facecolor='#FFF8E6', linewidth=1.5))
    ax.text(0.5, 0.63, "Abstractive\n(Old Models)", ha='center', fontsize=10, weight='bold', color=COLOR_MAIN)
    ax.text(0.5, 0.54, abstractive_bad, ha='center', va='center', fontsize=7, color=COLOR_MAIN,
            family='monospace', linespacing=1.3)
    ax.text(0.5, 0.42, "⚠ Incomplete", ha='center', fontsize=8, color='#CC8800', style='italic')

    # LLM paraphrasing (best)
    ax.add_patch(FancyBboxPatch((0.7, 0.45), 0.25, 0.2, boxstyle="round,pad=0.02",
                                 edgecolor=COLOR_HIGHLIGHT, facecolor='#E6FFE6', linewidth=2.5))
    ax.text(0.825, 0.63, "LLM Abstractive\n(GPT, Claude)", ha='center', fontsize=10, weight='bold', color=COLOR_MAIN)
    ax.text(0.825, 0.54, abstractive_llm, ha='center', va='center', fontsize=7, color=COLOR_MAIN,
            family='monospace', linespacing=1.3)
    ax.text(0.825, 0.42, "✓ Natural rephrasing", ha='center', fontsize=8, color='#00AA00', style='italic', weight='bold')

    # Arrows
    ax.annotate('', xy=(0.175, 0.68), xytext=(0.175, 0.72),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_ACCENT))
    ax.annotate('', xy=(0.5, 0.68), xytext=(0.5, 0.72),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_ACCENT))
    ax.annotate('', xy=(0.825, 0.68), xytext=(0.825, 0.72),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=COLOR_HIGHLIGHT))

    # Bottom insight
    ax.text(0.5, 0.28, "Key Insight: LLMs can paraphrase like humans - not just copy sentences",
            ha='center', fontsize=12, weight='bold', color=COLOR_MAIN,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFACD', edgecolor=COLOR_MAIN, linewidth=2))

    ax.text(0.5, 0.15, "Different words ('firm' vs 'company'), same meaning ('impressive growth' vs '25% increase')",
            ha='center', fontsize=10, color=COLOR_ACCENT, style='italic')

    ax.text(0.5, 0.05, "This enables creative, human-like summarization that goes beyond sentence extraction",
            ha='center', fontsize=10, color=COLOR_ACCENT)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.savefig('../figures/human_paraphrasing_visual_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("1/12: Human paraphrasing visual")

# Chart 2: LLM Summarization Pipeline
def plot_llm_pipeline():
    fig, ax = plt.subplots(figsize=(14, 6))

    stages = [
        ("Long\nDocument", 0.1),
        ("Prompt\nEngineering", 0.27),
        ("LLM\nProcessing", 0.5),
        ("Decoding\nControl", 0.73),
        ("Summary\nOutput", 0.9)
    ]

    for i, (stage, x) in enumerate(stages):
        # Box styling
        if i == 2:  # LLM processing (central)
            color = COLOR_HIGHLIGHT
            height = 0.35
            y_pos = 0.5
            linewidth = 3
        else:
            color = COLOR_MAIN
            height = 0.25
            y_pos = 0.5
            linewidth = 2

        ax.add_patch(FancyBboxPatch((x - 0.06, y_pos - height/2), 0.12, height,
                                     boxstyle="round,pad=0.01",
                                     edgecolor=color, facecolor=COLOR_LIGHT, linewidth=linewidth))
        ax.text(x, y_pos, stage, ha='center', va='center', fontsize=10, weight='bold', color=color)

        # Arrows between stages
        if i < len(stages) - 1:
            ax.annotate('', xy=(stages[i+1][1] - 0.07, y_pos), xytext=(x + 0.07, y_pos),
                       arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_MAIN))

    # Annotations below
    annotations = [
        (0.1, "Input text\n(up to 100K tokens)", 0.25),
        (0.27, "System prompt +\nfew-shot examples", 0.25),
        (0.5, "GPT-3.5/4, Claude,\nLLaMA, Mistral", 0.18),
        (0.73, "Temperature, top-p,\nmax_tokens", 0.25),
        (0.9, "Concise summary\n(target length)", 0.25)
    ]

    for x, text, y in annotations:
        ax.text(x, y, text, ha='center', va='top', fontsize=8, color=COLOR_ACCENT,
                style='italic', linespacing=1.4)

    # Title
    ax.text(0.5, 0.85, "LLM-Based Summarization Pipeline", ha='center', fontsize=14,
            weight='bold', color=COLOR_MAIN)

    # Bottom note
    ax.text(0.5, 0.08, "Three control points: prompt design, model selection, decoding parameters",
            ha='center', fontsize=10, color=COLOR_ACCENT)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.savefig('../figures/llm_pipeline_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("2/12: LLM summarization pipeline")

# Chart 3: Zero-Shot Prompt Example
def plot_zero_shot_prompt():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Input prompt box
    prompt_text = '''USER PROMPT:
"Summarize the following article in 3 sentences:

[Long article about climate change: 800 words]

Focus on main findings and policy implications."'''

    ax.add_patch(FancyBboxPatch((0.05, 0.55), 0.9, 0.38, boxstyle="round,pad=0.02",
                                 edgecolor=COLOR_MAIN, facecolor=COLOR_LIGHT, linewidth=2))
    ax.text(0.5, 0.90, "Zero-Shot Prompt (No Examples)", ha='center', fontsize=13,
            weight='bold', color=COLOR_MAIN)
    ax.text(0.5, 0.72, prompt_text, ha='center', va='center', fontsize=10,
            color=COLOR_MAIN, family='monospace', linespacing=1.6)

    # Arrow
    ax.annotate('', xy=(0.5, 0.52), xytext=(0.5, 0.55),
                arrowprops=dict(arrowstyle='->', lw=3, color=COLOR_HIGHLIGHT))
    ax.text(0.52, 0.535, "LLM processes", fontsize=9, color=COLOR_HIGHLIGHT, style='italic')

    # Output box
    output_text = '''LLM OUTPUT:
"The study finds global temperatures have risen 1.2°C since pre-industrial
times, with severe impacts on ecosystems and weather patterns. Scientists
recommend immediate carbon emission reductions of 50% by 2030 to limit
warming to 1.5°C. Policy proposals include carbon pricing, renewable
energy investment, and international cooperation frameworks."'''

    ax.add_patch(FancyBboxPatch((0.05, 0.15), 0.9, 0.35, boxstyle="round,pad=0.02",
                                 edgecolor=COLOR_HIGHLIGHT, facecolor='#E6FFE6', linewidth=2.5))
    ax.text(0.5, 0.48, "Generated Summary", ha='center', fontsize=13,
            weight='bold', color=COLOR_HIGHLIGHT)
    ax.text(0.5, 0.32, output_text, ha='left', va='center', fontsize=9,
            color=COLOR_MAIN, family='monospace', linespacing=1.5)

    # Bottom note
    ax.text(0.5, 0.05, "Zero-shot: No training examples needed - just clear instructions",
            ha='center', fontsize=11, color=COLOR_ACCENT, weight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.savefig('../figures/zero_shot_prompt_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("3/12: Zero-shot prompt example")

# Chart 4: Few-Shot Prompt with Examples
def plot_few_shot_prompt():
    fig, ax = plt.subplots(figsize=(12, 9))

    # System + examples
    prompt_text = '''SYSTEM: You are a professional summarizer. Condense articles to 2-3 sentences.

EXAMPLE 1:
Article: [500 words on AI breakthroughs]
Summary: "Recent AI advances include GPT-4 and multimodal models. Applications
span healthcare, education, and creative work."

EXAMPLE 2:
Article: [600 words on renewable energy]
Summary: "Solar and wind capacity grew 40% in 2024. Cost reductions make
renewables competitive with fossil fuels."

NOW YOUR TURN:
Article: [New article about quantum computing: 700 words]
Summary:'''

    ax.add_patch(FancyBboxPatch((0.05, 0.4), 0.9, 0.55, boxstyle="round,pad=0.02",
                                 edgecolor=COLOR_MAIN, facecolor=COLOR_LIGHT, linewidth=2))
    ax.text(0.5, 0.93, "Few-Shot Prompt (With Examples)", ha='center', fontsize=13,
            weight='bold', color=COLOR_MAIN)
    ax.text(0.08, 0.67, prompt_text, ha='left', va='center', fontsize=9,
            color=COLOR_MAIN, family='monospace', linespacing=1.5)

    # Highlight examples
    ax.add_patch(Rectangle((0.06, 0.72), 0.88, 0.16, fill=False,
                           edgecolor='#FF6B6B', linewidth=2, linestyle='--'))
    ax.text(0.92, 0.80, "Training examples\nshow desired format",
            ha='left', fontsize=9, color='#FF6B6B', weight='bold')

    # Arrow
    ax.annotate('', xy=(0.5, 0.37), xytext=(0.5, 0.4),
                arrowprops=dict(arrowstyle='->', lw=3, color=COLOR_HIGHLIGHT))

    # Output
    output_text = '''LLM OUTPUT:
"Quantum computers achieved error correction breakthroughs in 2024, enabling
practical applications. IBM and Google demonstrate quantum advantage in
optimization and simulation tasks."'''

    ax.add_patch(FancyBboxPatch((0.05, 0.12), 0.9, 0.22, boxstyle="round,pad=0.02",
                                 edgecolor=COLOR_HIGHLIGHT, facecolor='#E6FFE6', linewidth=2.5))
    ax.text(0.5, 0.32, "Generated Summary (Follows Pattern)", ha='center', fontsize=12,
            weight='bold', color=COLOR_HIGHLIGHT)
    ax.text(0.08, 0.22, output_text, ha='left', va='center', fontsize=9,
            color=COLOR_MAIN, family='monospace', linespacing=1.5)

    # Bottom note
    ax.text(0.5, 0.04, "Few-shot: Provide 2-5 examples → LLM learns your style",
            ha='center', fontsize=11, color=COLOR_ACCENT, weight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.savefig('../figures/few_shot_prompt_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("4/12: Few-shot prompt example")

# Chart 5: Chain-of-Thought for Long Documents
def plot_chain_of_thought():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Long document problem
    ax.add_patch(FancyBboxPatch((0.05, 0.78), 0.25, 0.18, boxstyle="round,pad=0.01",
                                 edgecolor=COLOR_MAIN, facecolor='#FFE6E6', linewidth=2))
    ax.text(0.175, 0.94, "Problem", ha='center', fontsize=11, weight='bold', color=COLOR_MAIN)
    ax.text(0.175, 0.87, "50-page report\n(20K tokens)", ha='center', fontsize=9, color=COLOR_MAIN)
    ax.text(0.175, 0.80, "Too long for\none summary", ha='center', fontsize=8,
            color='#CC0000', style='italic')

    # Arrow to CoT
    ax.annotate('', xy=(0.35, 0.87), xytext=(0.3, 0.87),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_MAIN))

    # Chain-of-Thought strategy
    ax.add_patch(FancyBboxPatch((0.35, 0.5), 0.6, 0.46, boxstyle="round,pad=0.02",
                                 edgecolor=COLOR_HIGHLIGHT, facecolor=COLOR_LIGHT, linewidth=2.5))
    ax.text(0.65, 0.94, "Chain-of-Thought Strategy", ha='center', fontsize=12,
            weight='bold', color=COLOR_HIGHLIGHT)

    # Steps
    steps = [
        ("Step 1: Extract sections", "Split into Introduction, Methods, Results, Discussion", 0.85),
        ("Step 2: Summarize each", "Generate 2-3 sentences per section", 0.73),
        ("Step 3: Identify key findings", "List main results and conclusions", 0.61),
        ("Step 4: Synthesize", "Combine into coherent final summary", 0.52)
    ]

    for i, (title, desc, y) in enumerate(steps, 1):
        ax.text(0.38, y, f"{title}", ha='left', fontsize=10, weight='bold', color=COLOR_MAIN)
        ax.text(0.38, y-0.05, f"   {desc}", ha='left', fontsize=8, color=COLOR_ACCENT, style='italic')

        # Arrows between steps
        if i < 4:
            ax.annotate('', xy=(0.37, y-0.08), xytext=(0.37, y-0.01),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_ACCENT))

    # Arrow to output
    ax.annotate('', xy=(0.5, 0.47), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=COLOR_HIGHLIGHT))

    # Final output
    ax.add_patch(FancyBboxPatch((0.05, 0.18), 0.9, 0.25, boxstyle="round,pad=0.02",
                                 edgecolor=COLOR_HIGHLIGHT, facecolor='#E6FFE6', linewidth=2))
    ax.text(0.5, 0.41, "Final Comprehensive Summary", ha='center', fontsize=11,
            weight='bold', color=COLOR_HIGHLIGHT)

    summary_text = '''"This study analyzed 10,000 patients over 5 years to evaluate treatment X.
Results show 30% improvement in primary outcomes with minimal side effects.
The hierarchical analysis reveals sustained benefits across all subgroups,
suggesting broad clinical applicability. Recommendations include protocol
adoption for patients meeting inclusion criteria."'''

    ax.text(0.5, 0.28, summary_text, ha='center', va='center', fontsize=8,
            color=COLOR_MAIN, family='monospace', linespacing=1.5)

    # Bottom note
    ax.text(0.5, 0.08, "CoT: Break complex documents into steps → better reasoning → accurate summaries",
            ha='center', fontsize=10, color=COLOR_ACCENT, weight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.savefig('../figures/chain_of_thought_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("5/12: Chain-of-thought prompting")

# Chart 6: Temperature Effect
def plot_temperature_effect():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Original text
    ax.text(0.5, 0.95, "Same Input Text, Different Temperatures", ha='center',
            fontsize=13, weight='bold', color=COLOR_MAIN)

    input_box = "Article: The Federal Reserve raised interest rates by 0.25% to combat inflation..."
    ax.add_patch(FancyBboxPatch((0.1, 0.85), 0.8, 0.08, boxstyle="round,pad=0.01",
                                 edgecolor=COLOR_MAIN, facecolor=COLOR_LIGHT, linewidth=1.5))
    ax.text(0.5, 0.89, input_box, ha='center', fontsize=9, color=COLOR_MAIN, family='monospace')

    # Three temperature outputs
    temps = [
        ("T = 0.3 (Conservative)",
         '''"The Federal Reserve increased rates by 0.25 percentage points to address
rising inflation. This marks the third consecutive rate hike this year."''',
         0.68, '#E6F3FF'),

        ("T = 0.7 (Balanced)",
         '''"The Fed raised borrowing costs by a quarter point in its ongoing effort to
cool inflation. Markets responded positively to the measured approach."''',
         0.45, '#FFF8E6'),

        ("T = 1.0 (Creative)",
         '''"Central bank officials opted for modest tightening amid persistent price
pressures. The cautious move reflects concerns about economic growth."''',
         0.22, '#FFE6F3')
    ]

    for title, text, y, bg_color in temps:
        ax.add_patch(FancyBboxPatch((0.05, y-0.02), 0.9, 0.16, boxstyle="round,pad=0.015",
                                     edgecolor=COLOR_MAIN, facecolor=bg_color, linewidth=2))
        ax.text(0.5, y+0.13, title, ha='center', fontsize=11, weight='bold', color=COLOR_MAIN)
        ax.text(0.5, y+0.06, text, ha='center', va='center', fontsize=8.5,
                color=COLOR_MAIN, family='monospace', linespacing=1.5)

    # Annotations
    ax.text(0.03, 0.75, "Safe,\npredictable", ha='left', fontsize=9,
            color='#0066CC', weight='bold', rotation=0)
    ax.text(0.03, 0.52, "Natural,\nvaried", ha='left', fontsize=9,
            color='#CC8800', weight='bold')
    ax.text(0.03, 0.29, "Diverse,\nunpredictable", ha='left', fontsize=9,
            color='#CC0066', weight='bold')

    # Bottom insight
    ax.text(0.5, 0.08, "Temperature controls randomness: Low = deterministic, High = creative",
            ha='center', fontsize=11, color=COLOR_ACCENT, weight='bold')

    ax.text(0.5, 0.02, "For summarization: T=0.3-0.5 (factual accuracy) | T=0.7-1.0 (varied phrasing)",
            ha='center', fontsize=9, color=COLOR_ACCENT, style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.savefig('../figures/temperature_effect_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("6/12: Temperature effect on creativity")

# Chart 7: Top-p Nucleus Sampling
def plot_nucleus_sampling():
    fig, ax = plt.subplots(figsize=(12, 7))

    # Title
    ax.text(0.5, 0.95, "Top-p (Nucleus) Sampling: Dynamic Cutoff", ha='center',
            fontsize=13, weight='bold', color=COLOR_MAIN)

    # Probability distribution
    words = ["growth", "increase", "rise", "gain", "surge", "uptick", "boost", "jump"]
    probs = np.array([0.35, 0.25, 0.15, 0.10, 0.07, 0.04, 0.03, 0.01])
    cumsum = np.cumsum(probs)

    # Bar chart
    colors = ['#66BB66' if cumsum[i] <= 0.9 else '#CCCCCC' for i in range(len(words))]
    bars = ax.bar(range(len(words)), probs, color=colors, edgecolor=COLOR_MAIN, linewidth=1.5)

    # Cumulative line
    ax2 = ax.twinx()
    ax2.plot(range(len(words)), cumsum, 'r-', linewidth=2.5, marker='o', markersize=8, label='Cumulative')
    ax2.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='p=0.9 cutoff')
    ax2.set_ylabel('Cumulative Probability', fontsize=11, color='red')
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='y', labelcolor='red')

    # Labels
    ax.set_xlabel('Next Word Candidates', fontsize=11, color=COLOR_MAIN)
    ax.set_ylabel('Probability', fontsize=11, color=COLOR_MAIN)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right')
    ax.set_ylim(0, 0.45)

    # Annotations
    ax.text(2, 0.40, "Nucleus: Top 90%\ncumulative probability",
            fontsize=10, color='#00AA00', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E6FFE6', edgecolor='#00AA00'))

    ax.text(6, 0.35, "Excluded:\nToo unlikely",
            fontsize=9, color='#888888', style='italic')

    # Bottom note
    ax.text(0.5, -0.28, "Top-p: Include words until cumulative probability reaches p (e.g., 0.9)",
            ha='center', fontsize=10, color=COLOR_ACCENT, weight='bold', transform=ax.transAxes)

    ax.text(0.5, -0.35, "Adapts to distribution: narrow → few words, flat → many words",
            ha='center', fontsize=9, color=COLOR_ACCENT, style='italic', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig('../figures/nucleus_sampling_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("7/12: Top-p nucleus sampling visual")

# Chart 8: Max Tokens and Truncation
def plot_max_tokens():
    fig, ax = plt.subplots(figsize=(12, 6))

    # Title
    ax.text(0.5, 0.93, "Max Tokens: Length Control", ha='center',
            fontsize=13, weight='bold', color=COLOR_MAIN)

    # Three scenarios
    scenarios = [
        ("max_tokens=50\n(Too Short)",
         '''"The study examined...[TRUNCATED]"''',
         0.73, '#FFE6E6', 'X'),

        ("max_tokens=150\n(Just Right)",
         '''"The study examined treatment efficacy in 1000
patients. Results showed 25% improvement with
minimal side effects. Recommended for clinical use."''',
         0.45, '#E6FFE6', 'OK'),

        ("max_tokens=500\n(Too Verbose)",
         '''"The comprehensive longitudinal study meticulously
examined the treatment efficacy across multiple patient
cohorts totaling approximately 1000 individuals.
The results demonstrated a statistically significant
improvement of approximately 25%..."''',
         0.17, '#FFF8E6', '?')
    ]

    for title, text, y, bg_color, symbol in scenarios:
        ax.add_patch(FancyBboxPatch((0.08, y-0.02), 0.84, 0.18, boxstyle="round,pad=0.015",
                                     edgecolor=COLOR_MAIN, facecolor=bg_color, linewidth=2))
        ax.text(0.5, y+0.15, title, ha='center', fontsize=11, weight='bold', color=COLOR_MAIN)
        ax.text(0.5, y+0.065, text, ha='center', va='center', fontsize=8.5,
                color=COLOR_MAIN, family='monospace', linespacing=1.5)

        # Symbol
        symbol_colors = {'X': '#CC0000', 'OK': '#00AA00', '?': '#CC8800'}
        ax.text(0.05, y+0.08, symbol, ha='center', fontsize=20, weight='bold',
                color=symbol_colors[symbol])

    # Bottom note
    ax.text(0.5, 0.04, "Set max_tokens based on desired summary length (typically 100-200 for news articles)",
            ha='center', fontsize=10, color=COLOR_ACCENT, weight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.savefig('../figures/max_tokens_control_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("8/12: Max tokens and truncation")

# Chart 9: Repetition Penalty Effect
def plot_repetition_penalty():
    fig, ax = plt.subplots(figsize=(12, 7))

    # Title
    ax.text(0.5, 0.95, "Repetition Penalty: Avoid Redundancy", ha='center',
            fontsize=13, weight='bold', color=COLOR_MAIN)

    # Without penalty
    ax.add_patch(FancyBboxPatch((0.05, 0.65), 0.9, 0.25, boxstyle="round,pad=0.02",
                                 edgecolor='#CC0000', facecolor='#FFE6E6', linewidth=2.5))
    ax.text(0.5, 0.88, "Without Repetition Penalty (penalty=1.0)", ha='center',
            fontsize=11, weight='bold', color='#CC0000')

    bad_text = '''"The company reported strong results. The company announced strong earnings.
The company's financial performance was strong. The company showed strong growth.
The company demonstrated strong performance in Q4."'''

    ax.text(0.5, 0.77, bad_text, ha='center', va='center', fontsize=9,
            color=COLOR_MAIN, family='monospace', linespacing=1.5)

    # Highlight repetitions
    ax.text(0.06, 0.67, "❌ 'company' x5\n❌ 'strong' x5", ha='left', fontsize=9,
            color='#CC0000', weight='bold')

    # Arrow
    ax.annotate('', xy=(0.5, 0.62), xytext=(0.5, 0.65),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=COLOR_MAIN))
    ax.text(0.52, 0.635, "Apply penalty=1.2", fontsize=9, color=COLOR_MAIN, style='italic')

    # With penalty
    ax.add_patch(FancyBboxPatch((0.05, 0.35), 0.9, 0.23, boxstyle="round,pad=0.02",
                                 edgecolor='#00AA00', facecolor='#E6FFE6', linewidth=2.5))
    ax.text(0.5, 0.56, "With Repetition Penalty (penalty=1.2)", ha='center',
            fontsize=11, weight='bold', color='#00AA00')

    good_text = '''"The firm reported strong Q4 results, with revenue increasing 15% year-over-year.
This performance exceeded analyst expectations and demonstrated effective cost
management. Leadership attributes success to strategic initiatives and market positioning."'''

    ax.text(0.5, 0.44, good_text, ha='center', va='center', fontsize=9,
            color=COLOR_MAIN, family='monospace', linespacing=1.5)

    # Highlight diversity
    ax.text(0.06, 0.37, "✓ Varied vocabulary\n✓ Natural flow", ha='left', fontsize=9,
            color='#00AA00', weight='bold')

    # Bottom note
    ax.text(0.5, 0.22, "Repetition penalty reduces probability of recently used tokens",
            ha='center', fontsize=11, color=COLOR_ACCENT, weight='bold')

    ax.text(0.5, 0.15, "Typical values: 1.0 (none) | 1.1 (mild) | 1.2 (moderate) | 1.5+ (aggressive)",
            ha='center', fontsize=9, color=COLOR_ACCENT, style='italic')

    ax.text(0.5, 0.08, "For summarization: Use 1.1-1.2 to encourage vocabulary diversity without awkwardness",
            ha='center', fontsize=9, color=COLOR_ACCENT)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.savefig('../figures/repetition_penalty_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("9/12: Repetition penalty effect")

# Chart 10: Chunking Strategy
def plot_chunking_strategy():
    fig, ax = plt.subplots(figsize=(12, 7))

    # Long document
    ax.add_patch(FancyBboxPatch((0.05, 0.75), 0.15, 0.2, boxstyle="round,pad=0.01",
                                 edgecolor=COLOR_MAIN, facecolor='#FFE6E6', linewidth=2))
    ax.text(0.125, 0.93, "Long Doc", ha='center', fontsize=10, weight='bold', color=COLOR_MAIN)
    ax.text(0.125, 0.85, "20,000\ntokens", ha='center', fontsize=9, color=COLOR_MAIN)
    ax.text(0.125, 0.77, "Exceeds\ncontext limit", ha='center', fontsize=7,
            color='#CC0000', style='italic')

    # Arrow to splitting
    ax.annotate('', xy=(0.25, 0.85), xytext=(0.2, 0.85),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_MAIN))
    ax.text(0.225, 0.88, "Split", fontsize=8, color=COLOR_MAIN)

    # Chunks
    chunk_colors = ['#E6F3FF', '#E6FFE6', '#FFF8E6', '#FFE6F3']
    for i in range(4):
        y = 0.75 - i * 0.18
        ax.add_patch(FancyBboxPatch((0.28, y-0.03), 0.15, 0.14, boxstyle="round,pad=0.01",
                                     edgecolor=COLOR_MAIN, facecolor=chunk_colors[i], linewidth=1.5))
        ax.text(0.355, y+0.09, f"Chunk {i+1}", ha='center', fontsize=9, weight='bold', color=COLOR_MAIN)
        ax.text(0.355, y+0.03, "5,000\ntokens", ha='center', fontsize=7, color=COLOR_MAIN)

        # Arrow to summary
        ax.annotate('', xy=(0.48, y+0.04), xytext=(0.43, y+0.04),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_ACCENT))

    # Individual summaries
    for i in range(4):
        y = 0.75 - i * 0.18
        ax.add_patch(FancyBboxPatch((0.5, y-0.025), 0.18, 0.11, boxstyle="round,pad=0.008",
                                     edgecolor=COLOR_ACCENT, facecolor=COLOR_LIGHT, linewidth=1))
        ax.text(0.59, y+0.07, f"Summary {i+1}", ha='center', fontsize=8, color=COLOR_ACCENT)
        ax.text(0.59, y+0.02, "2-3 sent.", ha='center', fontsize=6, color=COLOR_ACCENT, style='italic')

    # Combine arrow
    ax.annotate('', xy=(0.75, 0.50), xytext=(0.70, 0.65),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=COLOR_HIGHLIGHT))
    ax.annotate('', xy=(0.75, 0.50), xytext=(0.70, 0.57),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=COLOR_HIGHLIGHT))
    ax.annotate('', xy=(0.75, 0.50), xytext=(0.70, 0.39),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=COLOR_HIGHLIGHT))
    ax.annotate('', xy=(0.75, 0.50), xytext=(0.70, 0.21),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=COLOR_HIGHLIGHT))
    ax.text(0.72, 0.52, "Merge", fontsize=9, color=COLOR_HIGHLIGHT, weight='bold')

    # Final summary
    ax.add_patch(FancyBboxPatch((0.73, 0.38), 0.22, 0.25, boxstyle="round,pad=0.015",
                                 edgecolor=COLOR_HIGHLIGHT, facecolor='#E6FFE6', linewidth=2.5))
    ax.text(0.84, 0.60, "Final\nSummary", ha='center', fontsize=11, weight='bold', color=COLOR_HIGHLIGHT)
    ax.text(0.84, 0.50, "Coherent\ncomprehensive\nsummary of\nfull document",
            ha='center', fontsize=8, color=COLOR_MAIN, linespacing=1.4)

    # Bottom note
    ax.text(0.5, 0.15, "Chunking Strategy: Split → Summarize each → Merge summaries",
            ha='center', fontsize=11, color=COLOR_ACCENT, weight='bold')

    ax.text(0.5, 0.08, "Essential for documents exceeding model context window (typically 4K-32K tokens)",
            ha='center', fontsize=9, color=COLOR_ACCENT, style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.savefig('../figures/chunking_strategy_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("10/12: Chunking strategy diagram")

# Chart 11: Map-Reduce Summarization
def plot_map_reduce():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Title
    ax.text(0.5, 0.95, "Map-Reduce Summarization: Parallel Processing", ha='center',
            fontsize=13, weight='bold', color=COLOR_MAIN)

    # Input documents
    doc_positions = [0.15, 0.35, 0.55, 0.75]
    doc_colors = ['#E6F3FF', '#FFE6F3', '#E6FFE6', '#FFF8E6']
    for i, (x, color) in enumerate(zip(doc_positions, doc_colors)):
        ax.add_patch(FancyBboxPatch((x-0.06, 0.78), 0.12, 0.12, boxstyle="round,pad=0.01",
                                     edgecolor=COLOR_MAIN, facecolor=color, linewidth=1.5))
        ax.text(x, 0.87, f"Doc {i+1}", ha='center', fontsize=9, weight='bold', color=COLOR_MAIN)
        ax.text(x, 0.81, f"{3+i*2}K\ntokens", ha='center', fontsize=7, color=COLOR_MAIN)

    ax.text(0.45, 0.92, "Multiple documents or large sections", ha='center', fontsize=9,
            color=COLOR_ACCENT, style='italic')

    # MAP phase arrows
    for x in doc_positions:
        ax.annotate('', xy=(x, 0.65), xytext=(x, 0.73),
                    arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_MAIN))

    ax.add_patch(FancyBboxPatch((0.05, 0.52), 0.9, 0.08, boxstyle="round,pad=0.01",
                                 edgecolor=COLOR_HIGHLIGHT, facecolor='#FFFFCC', linewidth=2))
    ax.text(0.5, 0.56, "MAP PHASE: Summarize each document independently (parallel)",
            ha='center', fontsize=10, weight='bold', color=COLOR_HIGHLIGHT)

    # Individual summaries
    for i, x in enumerate(doc_positions):
        ax.annotate('', xy=(x, 0.48), xytext=(x, 0.52),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_ACCENT))
        ax.add_patch(FancyBboxPatch((x-0.055, 0.36), 0.11, 0.1, boxstyle="round,pad=0.008",
                                     edgecolor=COLOR_ACCENT, facecolor=COLOR_LIGHT, linewidth=1))
        ax.text(x, 0.44, f"Sum {i+1}", ha='center', fontsize=8, color=COLOR_ACCENT)
        ax.text(x, 0.38, "Brief", ha='center', fontsize=6, color=COLOR_ACCENT, style='italic')

    # Gather arrows
    for x in doc_positions:
        ax.annotate('', xy=(0.5, 0.28), xytext=(x, 0.32),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_MAIN))

    ax.add_patch(FancyBboxPatch((0.05, 0.16), 0.9, 0.08, boxstyle="round,pad=0.01",
                                 edgecolor=COLOR_HIGHLIGHT, facecolor='#FFFFCC', linewidth=2))
    ax.text(0.5, 0.20, "REDUCE PHASE: Combine summaries into final coherent output",
            ha='center', fontsize=10, weight='bold', color=COLOR_HIGHLIGHT)

    # Final output
    ax.annotate('', xy=(0.5, 0.12), xytext=(0.5, 0.16),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=COLOR_HIGHLIGHT))

    ax.add_patch(FancyBboxPatch((0.25, 0.02), 0.5, 0.08, boxstyle="round,pad=0.01",
                                 edgecolor=COLOR_HIGHLIGHT, facecolor='#E6FFE6', linewidth=2.5))
    ax.text(0.5, 0.06, "Final Unified Summary", ha='center', fontsize=11,
            weight='bold', color=COLOR_HIGHLIGHT)

    # Bottom note (outside main area)
    fig.text(0.5, 0.01, "Map-Reduce: Process independently → Combine results (scales to many documents)",
             ha='center', fontsize=10, color=COLOR_ACCENT, weight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.savefig('../figures/map_reduce_summarization_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("11/12: Map-reduce summarization flow")

# Chart 12: Recursive Hierarchical Summarization
def plot_recursive_hierarchical():
    fig, ax = plt.subplots(figsize=(12, 9))

    # Title
    ax.text(0.5, 0.97, "Recursive Hierarchical Summarization", ha='center',
            fontsize=13, weight='bold', color=COLOR_MAIN)

    # Level 0: Original sections (bottom)
    sections = ["Intro", "Methods", "Results 1", "Results 2", "Results 3", "Discussion"]
    x_positions = np.linspace(0.1, 0.9, len(sections))

    for i, (sec, x) in enumerate(zip(sections, x_positions)):
        color = ['#E6F3FF', '#FFE6F3', '#E6FFE6', '#E6FFE6', '#E6FFE6', '#FFF8E6'][i]
        ax.add_patch(FancyBboxPatch((x-0.045, 0.05), 0.09, 0.12, boxstyle="round,pad=0.008",
                                     edgecolor=COLOR_MAIN, facecolor=color, linewidth=1))
        ax.text(x, 0.14, sec, ha='center', fontsize=7, weight='bold', color=COLOR_MAIN)
        ax.text(x, 0.08, f"{2+i}K\ntokens", ha='center', fontsize=6, color=COLOR_MAIN)

    ax.text(0.5, 0.19, "Level 0: Original sections", ha='center', fontsize=8,
            color=COLOR_ACCENT, style='italic')

    # Arrows up
    for x in x_positions:
        ax.annotate('', xy=(x, 0.25), xytext=(x, 0.2),
                    arrowprops=dict(arrowstyle='->', lw=1, color=COLOR_ACCENT))

    # Level 1: First-level summaries
    level1_x = [0.2, 0.5, 0.8]
    level1_labels = ["Intro+Methods", "All Results", "Discussion"]
    for x, label in zip(level1_x, level1_labels):
        ax.add_patch(FancyBboxPatch((x-0.065, 0.32), 0.13, 0.12, boxstyle="round,pad=0.01",
                                     edgecolor=COLOR_ACCENT, facecolor='#F0F0F0', linewidth=1.5))
        ax.text(x, 0.41, label, ha='center', fontsize=8, weight='bold', color=COLOR_ACCENT)
        ax.text(x, 0.35, "Summary", ha='center', fontsize=6, color=COLOR_ACCENT, style='italic')

    # Connect Level 0 to Level 1
    ax.annotate('', xy=(0.2, 0.30), xytext=(0.1, 0.23),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_ACCENT))
    ax.annotate('', xy=(0.2, 0.30), xytext=(0.3, 0.23),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_ACCENT))

    ax.annotate('', xy=(0.5, 0.30), xytext=(0.38, 0.23),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_ACCENT))
    ax.annotate('', xy=(0.5, 0.30), xytext=(0.5, 0.23),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_ACCENT))
    ax.annotate('', xy=(0.5, 0.30), xytext=(0.62, 0.23),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_ACCENT))

    ax.annotate('', xy=(0.8, 0.30), xytext=(0.9, 0.23),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_ACCENT))

    ax.text(0.5, 0.47, "Level 1: Group and summarize related sections", ha='center', fontsize=8,
            color=COLOR_ACCENT, style='italic')

    # Arrows up to Level 2
    for x in level1_x:
        ax.annotate('', xy=(x, 0.53), xytext=(x, 0.48),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_MAIN))

    # Level 2: Second-level summary
    ax.add_patch(FancyBboxPatch((0.35, 0.60), 0.3, 0.14, boxstyle="round,pad=0.015",
                                 edgecolor=COLOR_HIGHLIGHT, facecolor='#FFFACD', linewidth=2))
    ax.text(0.5, 0.71, "Complete Study Summary", ha='center', fontsize=9,
            weight='bold', color=COLOR_HIGHLIGHT)
    ax.text(0.5, 0.64, "Coherent overview\ncombining all aspects", ha='center', fontsize=7,
            color=COLOR_MAIN, linespacing=1.4)

    # Connect Level 1 to Level 2
    for x in level1_x:
        ax.annotate('', xy=(0.5, 0.59), xytext=(x, 0.50),
                    arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_MAIN))

    ax.text(0.5, 0.77, "Level 2: Final synthesis", ha='center', fontsize=8,
            color=COLOR_HIGHLIGHT, style='italic')

    # Arrow to final output
    ax.annotate('', xy=(0.5, 0.85), xytext=(0.5, 0.80),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=COLOR_HIGHLIGHT))

    # Final output
    ax.add_patch(FancyBboxPatch((0.15, 0.86), 0.7, 0.08, boxstyle="round,pad=0.01",
                                 edgecolor=COLOR_HIGHLIGHT, facecolor='#E6FFE6', linewidth=2.5))
    ax.text(0.5, 0.90, "\"This study analyzed X using Y methods, finding Z results with W implications.\"",
            ha='center', fontsize=8, color=COLOR_MAIN, family='monospace', weight='bold')

    # Bottom note
    fig.text(0.5, 0.01, "Hierarchical: Bottom-up summarization → Preserve document structure → Better coherence",
             ha='center', fontsize=10, color=COLOR_ACCENT, weight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.savefig('../figures/recursive_hierarchical_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("12/12: Recursive hierarchical summarization")

# Generate all charts
if __name__ == "__main__":
    print("Generating 12 LLM summarization charts...")
    plot_human_paraphrasing()
    plot_llm_pipeline()
    plot_zero_shot_prompt()
    plot_few_shot_prompt()
    plot_chain_of_thought()
    plot_temperature_effect()
    plot_nucleus_sampling()
    plot_max_tokens()
    plot_repetition_penalty()
    plot_chunking_strategy()
    plot_map_reduce()
    plot_recursive_hierarchical()
    print("\nAll 12 charts generated successfully!")
    print("Output: NLP_slides/summarization_module/figures/")
