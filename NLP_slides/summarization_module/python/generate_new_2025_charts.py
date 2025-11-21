#!/usr/bin/env python3
"""
Generate NEW charts for LLM Summarization Module Redesign (2024-2025)
Creates 5 new charts for the discovery-based presentation
BSc Discovery color scheme
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np
import seaborn as sns
import os
import graphviz

# Create output directory
os.makedirs('../figures', exist_ok=True)

# BSc Discovery Color Scheme (consistent with existing charts)
COLOR_MAIN = '#404040'      # Main elements (dark gray)
COLOR_ACCENT = '#3333B2'    # Key concepts (purple)
COLOR_GRAY = '#B4B4B4'      # Secondary elements
COLOR_LIGHT = '#F0F0F0'     # Backgrounds
COLOR_GREEN = '#2CA02C'     # Success/positive
COLOR_RED = '#D62728'       # Error/negative
COLOR_ORANGE = '#FF7F0E'    # Warning/attention
COLOR_BLUE = '#1F77B4'      # Information

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Font sizes (consistent with existing module)
FONT_TITLE = 20
FONT_HEADER = 14
FONT_LABEL = 12
FONT_BODY = 11
FONT_ANNOTATION = 10

def set_chart_style(ax):
    """Apply consistent BSc Discovery styling to charts"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_GRAY)
    ax.spines['bottom'].set_color(COLOR_GRAY)
    ax.tick_params(colors=COLOR_MAIN, which='both', labelsize=FONT_LABEL)
    ax.grid(True, alpha=0.1, linestyle='--', color=COLOR_GRAY)
    ax.set_facecolor('white')

# Chart 1: Information Overload Growth (Slide 2)
def generate_information_overload_growth():
    """Time series showing exponential growth in text production"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Data: 2010-2025
    years = np.arange(2010, 2026)

    # Emails per day (billions)
    emails = 200 * (1.08 ** (years - 2010))  # 8% growth/year

    # Research papers per year (millions)
    papers = 2.0 * (1.12 ** (years - 2010))  # 12% growth/year

    # News articles per day (thousands)
    news = 100 * (1.20 ** (years - 2010))  # 20% growth/year

    # Plot lines with different styles
    ax.plot(years, emails, color=COLOR_ACCENT, linewidth=3,
            marker='o', markersize=6, label='Emails/day (billions)')
    ax.plot(years, papers * 100, color=COLOR_BLUE, linewidth=2.5,
            marker='s', markersize=5, label='Research papers/year (×100K)')
    ax.plot(years, news, color=COLOR_ORANGE, linewidth=2.5,
            marker='^', markersize=5, label='News articles/day (thousands)')

    # Highlight key milestones
    ax.axvline(x=2020, color=COLOR_RED, linestyle='--', alpha=0.3, linewidth=1)
    ax.text(2020.2, 100, 'COVID-19\nInfo Surge', fontsize=FONT_ANNOTATION,
            color=COLOR_RED, style='italic')

    ax.axvline(x=2023, color=COLOR_GREEN, linestyle='--', alpha=0.3, linewidth=1)
    ax.text(2023.2, 150, 'ChatGPT\nEra', fontsize=FONT_ANNOTATION,
            color=COLOR_GREEN, style='italic')

    # Add cost annotation
    ax.text(2024, 380, f'Manual summarization\ncost: $50B/year',
            fontsize=FONT_HEADER, color=COLOR_RED, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_LIGHT, edgecolor=COLOR_RED))

    # Styling
    ax.set_xlabel('Year', fontsize=FONT_LABEL, color=COLOR_MAIN)
    ax.set_ylabel('Volume (log scale)', fontsize=FONT_LABEL, color=COLOR_MAIN)
    ax.set_title('Information Overload Crisis: Exponential Growth in Text Production',
                fontsize=FONT_TITLE, color=COLOR_MAIN, weight='bold', pad=20)

    # Use log scale for better visualization
    ax.set_yscale('log')
    ax.set_ylim(50, 500)

    # Grid and styling
    ax.grid(True, alpha=0.2, linestyle='--', which='both')
    ax.legend(loc='upper left', fontsize=FONT_LABEL, framealpha=0.95)

    set_chart_style(ax)
    plt.tight_layout()
    plt.savefig('../figures/information_overload_growth_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: information_overload_growth_bsc.pdf")

# Chart 2: Traditional vs. LLM Approaches Matrix (Slide 6.5)
def generate_traditional_vs_llm_matrix():
    """2x2 matrix showing evolution of summarization techniques"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Hide axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Draw quadrant lines
    ax.axhline(y=5, color=COLOR_GRAY, linewidth=2)
    ax.axvline(x=5, color=COLOR_GRAY, linewidth=2)

    # Axis labels
    ax.text(5, -0.5, 'Rule-based ← → Neural', fontsize=FONT_HEADER,
            ha='center', weight='bold', color=COLOR_MAIN)
    ax.text(-0.5, 5, 'Extractive\n↕\nAbstractive', fontsize=FONT_HEADER,
            ha='center', va='center', weight='bold', color=COLOR_MAIN, rotation=90)

    # Quadrant 1: Rule-based Extractive (bottom-left)
    q1_box = FancyBboxPatch((0.5, 0.5), 4, 4, boxstyle="round,pad=0.1",
                            facecolor=COLOR_LIGHT, edgecolor=COLOR_GRAY, linewidth=1)
    ax.add_patch(q1_box)
    ax.text(2.5, 3.5, 'Extractive + Rule-based', fontsize=FONT_HEADER,
            ha='center', weight='bold', color=COLOR_MAIN)
    ax.text(2.5, 2.8, '• TF-IDF ranking', fontsize=FONT_BODY, ha='center')
    ax.text(2.5, 2.3, '• Keyword extraction', fontsize=FONT_BODY, ha='center')
    ax.text(2.5, 1.8, '• TextRank (2004)', fontsize=FONT_BODY, ha='center')
    ax.text(2.5, 1.2, 'Era: 2000-2010', fontsize=FONT_ANNOTATION,
            ha='center', style='italic', color=COLOR_GRAY)

    # Quadrant 2: Rule-based Abstractive (top-left)
    q2_box = FancyBboxPatch((0.5, 5.5), 4, 4, boxstyle="round,pad=0.1",
                            facecolor='#FFF0F0', edgecolor=COLOR_GRAY, linewidth=1)
    ax.add_patch(q2_box)
    ax.text(2.5, 8.5, 'Abstractive + Rule-based', fontsize=FONT_HEADER,
            ha='center', weight='bold', color=COLOR_MAIN)
    ax.text(2.5, 7.8, '• Template filling', fontsize=FONT_BODY, ha='center')
    ax.text(2.5, 7.3, '• Sentence fusion', fontsize=FONT_BODY, ha='center')
    ax.text(2.5, 6.8, '• Limited success', fontsize=FONT_BODY, ha='center', color=COLOR_RED)
    ax.text(2.5, 6.2, 'Era: Rarely used', fontsize=FONT_ANNOTATION,
            ha='center', style='italic', color=COLOR_GRAY)

    # Quadrant 3: Neural Extractive (bottom-right)
    q3_box = FancyBboxPatch((5.5, 0.5), 4, 4, boxstyle="round,pad=0.1",
                            facecolor='#F0F0FF', edgecolor=COLOR_BLUE, linewidth=1.5)
    ax.add_patch(q3_box)
    ax.text(7.5, 3.5, 'Extractive + Neural', fontsize=FONT_HEADER,
            ha='center', weight='bold', color=COLOR_MAIN)
    ax.text(7.5, 2.8, '• BERT sentence ranking', fontsize=FONT_BODY, ha='center')
    ax.text(7.5, 2.3, '• Neural sentence selection', fontsize=FONT_BODY, ha='center')
    ax.text(7.5, 1.8, '• BertSum (2019)', fontsize=FONT_BODY, ha='center')
    ax.text(7.5, 1.2, 'Era: 2018-2020', fontsize=FONT_ANNOTATION,
            ha='center', style='italic', color=COLOR_GRAY)

    # Quadrant 4: Neural Abstractive - HIGHLIGHTED (top-right)
    q4_box = FancyBboxPatch((5.5, 5.5), 4, 4, boxstyle="round,pad=0.1",
                            facecolor='#E8E8FF', edgecolor=COLOR_ACCENT, linewidth=3)
    ax.add_patch(q4_box)
    ax.text(7.5, 8.5, 'Abstractive + Neural', fontsize=FONT_HEADER,
            ha='center', weight='bold', color=COLOR_ACCENT)
    ax.text(7.5, 7.8, '• Transformer models', fontsize=FONT_BODY, ha='center', color=COLOR_ACCENT)
    ax.text(7.5, 7.3, '• LLM generation', fontsize=FONT_BODY, ha='center', color=COLOR_ACCENT)
    ax.text(7.5, 6.8, '• GPT/Claude/T5', fontsize=FONT_BODY, ha='center', color=COLOR_ACCENT)
    ax.text(7.5, 6.2, 'Era: 2020+', fontsize=FONT_ANNOTATION,
            ha='center', weight='bold', color=COLOR_ACCENT)

    # Add evolution arrow
    arrow = FancyArrowPatch((2.5, 2.5), (7.5, 7.5),
                           connectionstyle="arc3,rad=0.3",
                           arrowstyle='->', mutation_scale=30,
                           linewidth=3, color=COLOR_GREEN, alpha=0.7)
    ax.add_patch(arrow)
    ax.text(5, 4, 'Evolution', fontsize=FONT_LABEL, ha='center',
            color=COLOR_GREEN, weight='bold', rotation=45)

    # Title
    ax.text(5, 10.5, 'Evolution of Summarization Techniques',
            fontsize=FONT_TITLE, ha='center', weight='bold', color=COLOR_MAIN)

    # Star on the winning quadrant
    ax.plot(8.5, 8.5, marker='*', markersize=20, color=COLOR_ACCENT)
    ax.text(8.5, 9, 'BEST', fontsize=FONT_ANNOTATION, ha='center',
            weight='bold', color=COLOR_ACCENT)

    plt.tight_layout()
    plt.savefig('../figures/traditional_vs_llm_matrix_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: traditional_vs_llm_matrix_bsc.pdf")

# Chart 3: Chain-of-Thought Summarization Process (Slide 22)
def generate_cot_summarization_process():
    """Flowchart showing CoT reasoning steps for summarization"""
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(dpi='300', rankdir='TB', bgcolor='white')
    dot.attr('node', shape='box', style='rounded,filled', fontsize='14', height='0.8')
    dot.attr('edge', fontsize='12')

    # Title node
    dot.node('title', 'Chain-of-Thought Summarization Process',
             fillcolor=COLOR_ACCENT, fontcolor='white', shape='box',
             style='filled,bold', fontsize='16')

    # Step 1: Instruction
    dot.node('instruction',
             'INSTRUCTION\n"Let\'s identify the main points\nstep-by-step before writing the summary"',
             fillcolor='#E8E8FF', color=COLOR_BLUE)

    # Step 2: Reasoning steps (purple)
    dot.node('reason1',
             'STEP 1: Main Claim\n"Study found 30% reduction\nin patient complications"',
             fillcolor='#F0E8FF', color=COLOR_ACCENT)

    dot.node('reason2',
             'STEP 2: Supporting Evidence\n"1,000 patients over 5 years\nRandomized controlled trial"',
             fillcolor='#F0E8FF', color=COLOR_ACCENT)

    dot.node('reason3',
             'STEP 3: Methodology\n"Double-blind study\nStatistically significant (p<0.01)"',
             fillcolor='#F0E8FF', color=COLOR_ACCENT)

    dot.node('reason4',
             'STEP 4: Conclusion\n"New treatment protocol\nrecommended for adoption"',
             fillcolor='#F0E8FF', color=COLOR_ACCENT)

    # Step 3: Synthesis
    dot.node('synthesis',
             'SYNTHESIS\n"Now I\'ll write a coherent summary\ncombining these key points"',
             fillcolor='#E8FFE8', color=COLOR_GREEN)

    # Step 4: Output
    dot.node('output',
             'OUTPUT\n"Study of 1,000 patients over 5 years\nshows 30% reduction in complications\nwith new treatment (p<0.01).\nDouble-blind RCT recommends adoption."',
             fillcolor='#E8FFE8', color=COLOR_GREEN, shape='box', style='filled,bold')

    # Edges
    dot.edge('title', 'instruction', label='Start', color=COLOR_GRAY)
    dot.edge('instruction', 'reason1', label='Extract', color=COLOR_BLUE)
    dot.edge('reason1', 'reason2', color=COLOR_ACCENT)
    dot.edge('reason2', 'reason3', color=COLOR_ACCENT)
    dot.edge('reason3', 'reason4', color=COLOR_ACCENT)
    dot.edge('reason4', 'synthesis', label='Combine', color=COLOR_ACCENT)
    dot.edge('synthesis', 'output', label='Generate', color=COLOR_GREEN)

    # Save
    dot.render('../figures/cot_summarization_process_bsc', cleanup=True)
    print("Generated: cot_summarization_process_bsc.pdf")

# Chart 4: Chain-of-Thought Variants Comparison (Slide 23)
def generate_cot_variants_comparison():
    """Table comparing different CoT variants from 2024-2025 research"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # Table data
    columns = ['Variant', 'Key Feature', 'Best For', 'Example Prompt']

    data = [
        ['Standard CoT',
         '"Let\'s think\nstep-by-step"',
         'General reasoning\nAll summaries',
         '"Identify main points,\nthen summarize"'],

        ['Contrastive CoT',
         'Show wrong\nexample too',
         'Avoiding specific\nerrors',
         '"Good: factual summary\nBad: hallucinated facts\nNow summarize correctly"'],

        ['Thread-of-\nThought',
         'Multi-part\nprocessing',
         'Long RAG\ncontexts',
         '"Walk through document\nin parts, summarizing\nas we go"'],

        ['Faithful CoT',
         'Verify each\nstep',
         'Critical accuracy\n(medical/legal)',
         '"Extract claim.\nVerify in source.\nThen summarize."']
    ]

    # Create table with colors
    colors = []
    for i in range(len(data)):
        row_colors = []
        if i == 0:  # Standard CoT
            row_colors = [COLOR_LIGHT, COLOR_LIGHT, COLOR_LIGHT, COLOR_LIGHT]
        elif i == 1:  # Contrastive
            row_colors = ['#FFF0F0', '#FFF0F0', '#FFF0F0', '#FFF0F0']
        elif i == 2:  # Thread
            row_colors = ['#F0F0FF', '#F0F0FF', '#F0F0FF', '#F0F0FF']
        else:  # Faithful
            row_colors = ['#F0FFF0', '#F0FFF0', '#F0FFF0', '#F0FFF0']
        colors.append(row_colors)

    # Create table
    table = ax.table(cellText=data, colLabels=columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.18, 0.22, 0.25, 0.35],
                    cellColours=colors,
                    colColours=[COLOR_ACCENT, COLOR_ACCENT, COLOR_ACCENT, COLOR_ACCENT])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Color header cells white text
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor(COLOR_ACCENT)

    # Bold first column
    for i in range(1, len(data) + 1):
        cell = table[(i, 0)]
        cell.set_text_props(weight='bold')

    # Title
    plt.title('Chain-of-Thought Variants Comparison (2024-2025 Research)',
             fontsize=FONT_TITLE, weight='bold', color=COLOR_MAIN, pad=20)

    # Annotation
    plt.text(0.5, -0.1,
            'Note: These variants emerged from recent research on improving LLM reasoning',
            transform=ax.transAxes, ha='center', fontsize=FONT_ANNOTATION,
            style='italic', color=COLOR_GRAY)

    plt.tight_layout()
    plt.savefig('../figures/cot_variants_comparison_2024_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: cot_variants_comparison_2024_bsc.pdf")

# Chart 5: Modern Metrics Comparison (Slide 31)
def generate_modern_metrics_comparison():
    """Comprehensive comparison of evaluation metrics"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis('tight')
    ax.axis('off')

    # Table data
    columns = ['Metric', 'What it Measures', 'Human\nCorrelation', 'Cost', 'Best For']

    data = [
        ['ROUGE', 'Word overlap', '0.45', 'Free', 'Baseline screening'],
        ['BLEU', 'N-gram precision', '0.38', 'Free', 'Translation\n(not summarization)'],
        ['BERTScore', 'Semantic similarity', '0.72', 'Low\n($0.01/eval)', 'Quality filtering'],
        ['METEOR', 'Stemming +\nsynonyms', '0.55', 'Free', 'Improved ROUGE'],
        ['G-eval', 'LLM rates quality\n(multi-aspect)', '0.85', 'Medium\n($0.10/eval)', 'Detailed evaluation'],
        ['GPT-4 Judge', 'Overall quality\nassessment', '0.92', 'High\n($0.30/eval)', 'Final validation'],
        ['Faithfulness', 'Fact verification\nagainst source', '0.88', 'High\n($0.25/eval)', 'Critical applications'],
        ['Human Eval', 'Gold standard\n(definition)', '1.00', 'Very High\n($5-20/eval)', 'Ground truth']
    ]

    # Create color map for correlation column
    correlations = [0.45, 0.38, 0.72, 0.55, 0.85, 0.92, 0.88, 1.00]

    # Cell colors based on correlation
    colors = []
    for i, corr in enumerate(correlations):
        row_colors = []
        for j in range(5):
            if j == 2:  # Correlation column
                if corr >= 0.85:
                    row_colors.append('#D4F4DD')  # Green
                elif corr >= 0.70:
                    row_colors.append('#FFF4D4')  # Yellow
                else:
                    row_colors.append('#FFD4D4')  # Red
            else:
                row_colors.append('white')
        colors.append(row_colors)

    # Create table
    table = ax.table(cellText=data, colLabels=columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.22, 0.15, 0.18, 0.30],
                    cellColours=colors,
                    colColours=[COLOR_ACCENT, COLOR_ACCENT, COLOR_ACCENT, COLOR_ACCENT, COLOR_ACCENT])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)

    # Color header cells
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor(COLOR_ACCENT)

    # Bold metric names and color correlation values
    for i in range(1, len(data) + 1):
        # Bold metric name
        cell = table[(i, 0)]
        cell.set_text_props(weight='bold')

        # Color correlation value
        cell = table[(i, 2)]
        corr_value = correlations[i-1]
        if corr_value >= 0.85:
            cell.set_text_props(weight='bold', color=COLOR_GREEN)
        elif corr_value >= 0.70:
            cell.set_text_props(weight='bold', color=COLOR_ORANGE)
        else:
            cell.set_text_props(weight='bold', color=COLOR_RED)

    # Title
    plt.title('Modern Evaluation Metrics Comparison (2024-2025)',
             fontsize=FONT_TITLE, weight='bold', color=COLOR_MAIN, pad=20)

    # Legend for correlation colors
    legend_elements = [
        mpatches.Patch(color='#D4F4DD', label='High Correlation (≥0.85)'),
        mpatches.Patch(color='#FFF4D4', label='Medium Correlation (0.70-0.84)'),
        mpatches.Patch(color='#FFD4D4', label='Low Correlation (<0.70)')
    ]
    plt.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=FONT_ANNOTATION)

    # Annotation
    plt.text(0.5, -0.08,
            'Higher correlation with human judgment = better proxy for quality assessment',
            transform=ax.transAxes, ha='center', fontsize=FONT_ANNOTATION,
            style='italic', color=COLOR_GRAY)

    plt.tight_layout()
    plt.savefig('../figures/modern_metrics_comparison_2024_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: modern_metrics_comparison_2024_bsc.pdf")

# Main execution
def main():
    """Generate all 5 new charts"""
    print("\n" + "="*60)
    print("Generating NEW Charts for LLM Summarization Redesign")
    print("="*60 + "\n")

    # Generate each chart
    print("Generating Chart 1: Information Overload Growth...")
    generate_information_overload_growth()

    print("Generating Chart 2: Traditional vs LLM Matrix...")
    generate_traditional_vs_llm_matrix()

    print("Generating Chart 3: CoT Summarization Process...")
    generate_cot_summarization_process()

    print("Generating Chart 4: CoT Variants Comparison...")
    generate_cot_variants_comparison()

    print("Generating Chart 5: Modern Metrics Comparison...")
    generate_modern_metrics_comparison()

    print("\n" + "="*60)
    print("SUCCESS: All 5 new charts generated successfully!")
    print("Location: NLP_slides/summarization_module/figures/")
    print("="*60 + "\n")

    # List generated files
    generated_files = [
        "information_overload_growth_bsc.pdf",
        "traditional_vs_llm_matrix_bsc.pdf",
        "cot_summarization_process_bsc.pdf",
        "cot_variants_comparison_2024_bsc.pdf",
        "modern_metrics_comparison_2024_bsc.pdf"
    ]

    print("Generated files:")
    for i, filename in enumerate(generated_files, 1):
        print(f"  {i}. {filename}")

    print("\nThese charts are ready to be included in the LaTeX presentation.")
    print("Total charts now available: 87 (82 existing + 5 new)")

if __name__ == "__main__":
    main()