"""
Context Matters Chart - Clue #1 in the Investigation

PEDAGOGICAL PURPOSE:
- Make abstract "context" concrete with 3 specific failure patterns
- Create cognitive dissonance: "These seem obvious to humans, why do models fail?"
- Establish NEED for bidirectional understanding before introducing solution

EMOTIONAL HOOK: Students recognize these patterns from real reviews
PREVENTS MISCONCEPTION: "Just add more features" or "weight words better"
SCAFFOLDS: From BOW independence assumption to need for relational semantics
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# BSc Discovery Color Scheme
COLOR_MAIN = '#404040'
COLOR_ACCENT = '#3333B2'
COLOR_GRAY = '#B4B4B4'
COLOR_GREEN = '#2CA02C'
COLOR_RED = '#D62728'
COLOR_ORANGE = '#FF7F0E'
COLOR_LIGHT = '#F0F0F0'

# BSc Discovery Font Standard
FONTSIZE_TITLE = 24
FONTSIZE_LABEL = 20
FONTSIZE_TICK = 16
FONTSIZE_ANNOTATION = 18
FONTSIZE_TEXT = 20

def create_chart():
    """Generate context matters visualization with 3 failure examples."""

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Clue #1: Why Traditional Methods Fail',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', color=COLOR_ACCENT)

    # Define examples (text, BOW prediction, actual sentiment)
    examples = [
        {
            'title': 'Sarcasm',
            'text': '"Great, another boring\\nsuperhero movie"',
            'bow_pred': 'POSITIVE',
            'actual': 'NEGATIVE',
            'explanation': 'BOW sees "Great"\\nignores context'
        },
        {
            'title': 'Negation',
            'text': '"This is not a\\nbad film"',
            'bow_pred': 'NEGATIVE',
            'actual': 'POSITIVE',
            'explanation': 'BOW sees "bad"\\nignores "not"'
        },
        {
            'title': 'Intensity',
            'text': '"Absolutely incredible"\\nvs\\n"Somewhat good"',
            'bow_pred': 'BOTH POSITIVE',
            'actual': 'STRONG vs WEAK',
            'explanation': "BOW can't measure\\nstrength"
        }
    ]

    for idx, example in enumerate(examples):
        ax_left = axes[idx, 0]
        ax_right = axes[idx, 1]

        # Left column: The review text
        ax_left.axis('off')
        ax_left.set_xlim(0, 1)
        ax_left.set_ylim(0, 1)

        # Title
        ax_left.text(0.5, 0.85, example['title'], ha='center', va='top',
                    fontsize=FONTSIZE_LABEL, fontweight='bold', color=COLOR_ACCENT)

        # Review text box
        box = FancyBboxPatch((0.1, 0.35), 0.8, 0.4, boxstyle="round,pad=0.02",
                            edgecolor=COLOR_GRAY, facecolor=COLOR_LIGHT, linewidth=2)
        ax_left.add_patch(box)
        ax_left.text(0.5, 0.55, example['text'], ha='center', va='center',
                    fontsize=FONTSIZE_ANNOTATION, color=COLOR_MAIN, style='italic')

        # Bottom label
        ax_left.text(0.5, 0.15, 'Actual: ' + example['actual'], ha='center', va='center',
                    fontsize=FONTSIZE_ANNOTATION, fontweight='bold', color=COLOR_GREEN)

        # Right column: BOW failure
        ax_right.axis('off')
        ax_right.set_xlim(0, 1)
        ax_right.set_ylim(0, 1)

        # BOW Prediction (wrong)
        box_wrong = FancyBboxPatch((0.1, 0.55), 0.8, 0.25, boxstyle="round,pad=0.02",
                                  edgecolor=COLOR_RED, facecolor='#FFE6E6', linewidth=3)
        ax_right.add_patch(box_wrong)
        ax_right.text(0.5, 0.75, 'BOW Predicts:', ha='center', va='center',
                     fontsize=FONTSIZE_ANNOTATION, color=COLOR_GRAY)
        ax_right.text(0.5, 0.62, example['bow_pred'], ha='center', va='center',
                     fontsize=FONTSIZE_LABEL, fontweight='bold', color=COLOR_RED)

        # Explanation box
        box_explain = FancyBboxPatch((0.1, 0.15), 0.8, 0.30, boxstyle="round,pad=0.02",
                                    edgecolor=COLOR_ORANGE, facecolor='#FFF4E6', linewidth=2)
        ax_right.add_patch(box_explain)
        ax_right.text(0.5, 0.35, 'Why BOW Fails:', ha='center', va='top',
                     fontsize=FONTSIZE_ANNOTATION-2, color=COLOR_ORANGE, fontweight='bold')
        ax_right.text(0.5, 0.25, example['explanation'], ha='center', va='center',
                     fontsize=FONTSIZE_TICK, color=COLOR_MAIN)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path = '../../../figures/context_matters_bsc.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Generated: {output_path}")
    print(f"Pedagogical role: Establish concrete clues for why context matters")
    print(f"Font sizes: title={FONTSIZE_TITLE}pt, label={FONTSIZE_LABEL}pt, annotation={FONTSIZE_ANNOTATION}pt")

if __name__ == '__main__':
    create_chart()
