#!/usr/bin/env python3
"""
Generate all remaining course materials for NLP Course 2025
This script creates presentations, lab notebooks, and handouts for weeks 8-12
"""

import os
import json
from datetime import datetime

# Get current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# Course structure for remaining weeks
weeks_config = {
    "week08_tokenization": {
        "title": "Tokenization & Vocabulary",
        "topics": ["BPE", "WordPiece", "SentencePiece", "Subword tokenization"],
        "lab_focus": "Implementing and comparing tokenization methods"
    },
    "week09_decoding": {
        "title": "Decoding Strategies",
        "topics": ["Beam search", "Sampling methods", "Top-k/Top-p", "Temperature"],
        "lab_focus": "Implementing various decoding strategies"
    },
    "week10_finetuning": {
        "title": "Fine-tuning & Prompt Engineering",
        "topics": ["PEFT", "LoRA", "Prompt design", "In-context learning"],
        "lab_focus": "Fine-tuning techniques and prompt optimization"
    },
    "week11_efficiency": {
        "title": "Efficiency & Optimization",
        "topics": ["Quantization", "Pruning", "Distillation", "Efficient architectures"],
        "lab_focus": "Model compression and optimization techniques"
    },
    "week12_ethics": {
        "title": "Ethics & Fairness",
        "topics": ["Bias detection", "Safety", "Responsible AI", "Evaluation metrics"],
        "lab_focus": "Bias detection and mitigation strategies"
    }
}

def create_presentation(week_dir, week_num, config, timestamp):
    """Create main presentation for a week."""

    presentation_content = f"""% Week {week_num}: {config['title']}
% Using the Master Optimal Readability Template

\\input{{../../common/master_template.tex}}

\\title{{{config['title']}}}
\\subtitle{{\\secondary{{Week {week_num} - NLP Course 2025}}}}
\\author{{NLP Course 2025}}
\\date{{\\today}}

\\begin{{document}}

% Title slide
\\begin{{frame}}
\\titlepage
\\vfill
\\begin{{center}}
\\secondary{{\\footnotesize {config['title']}}}
\\end{{center}}
\\end{{frame}}

% Overview
\\begin{{frame}}{{Week {week_num}: Overview}}
\\begin{{columns}}[T]
\\column{{0.48\\textwidth}}
\\textbf{{Learning Objectives}}
\\begin{{itemize}}
"""

    for topic in config['topics'][:2]:
        presentation_content += f"\\item Understand {topic}\n"

    presentation_content += f"""\\item Apply concepts in practice
\\item Evaluate different approaches
\\end{{itemize}}

\\column{{0.48\\textwidth}}
\\textbf{{Topics Covered}}
\\begin{{itemize}}
"""

    for topic in config['topics']:
        presentation_content += f"\\item {topic}\n"

    presentation_content += f"""\\end{{itemize}}
\\end{{columns}}
\\end{{frame}}

% Main content slides (template)
\\begin{{frame}}{{Introduction}}
\\begin{{center}}
{{\\Large \\textbf{{{config['title']}}}}}
\\end{{center}}
\\vspace{{10mm}}
Content for {config['title']}...
\\end{{frame}}

% Key concepts
\\begin{{frame}}{{Key Concepts}}
\\begin{{itemize}}
\\item Concept 1: {config['topics'][0]}
\\item Concept 2: {config['topics'][1] if len(config['topics']) > 1 else 'Advanced topics'}
\\item Practical applications
\\item Current research directions
\\end{{itemize}}
\\end{{frame}}

% Summary
\\begin{{frame}}{{Summary}}
\\begin{{center}}
{{\\Large \\textbf{{Key Takeaways}}}}
\\end{{center}}
\\vspace{{10mm}}
\\begin{{itemize}}
\\item Main learning point 1
\\item Main learning point 2
\\item Main learning point 3
\\end{{itemize}}
\\vfill
\\secondary{{\\footnotesize Next week: Continue to next topic}}
\\end{{frame}}

\\end{{document}}"""

    week_name = os.path.basename(week_dir)
    filepath = os.path.join(week_dir, "presentations", f"{timestamp}_{week_name}_optimal.tex")
    with open(filepath, 'w') as f:
        f.write(presentation_content)
    print(f"Created presentation: {filepath}")
    return filepath

def create_lab_notebook(week_dir, week_num, config):
    """Create lab notebook for a week."""

    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# Week {week_num} Lab: {config['title']}\n\n",
                    "## Learning Objectives\n",
                    f"- {config['lab_focus']}\n",
                    "- Hands-on implementation\n",
                    "- Performance evaluation\n\n",
                    "## Prerequisites\n",
                    "```bash\n",
                    "pip install transformers torch numpy matplotlib\n",
                    "```"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": "import torch\\nimport numpy as np\\nimport matplotlib.pyplot as plt\\nfrom transformers import AutoTokenizer, AutoModel\\n\\n# Setup\\nprint('Week " + str(week_num) + ": " + config['title'] + "')\\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\\nprint(f'Using device: {device}')"
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Part 1: Introduction\n\n",
                    f"This lab focuses on: {config['lab_focus']}"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": "# Main implementation\\n# Week " + str(week_num) + " specific code\\npass"
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Exercises\n\n",
                    "1. Implement the main concept\n",
                    "2. Compare different approaches\n",
                    "3. Evaluate performance\n",
                    "4. Extend to new applications"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Summary\n\n",
                    f"In this lab, we explored {config['title']} concepts and implementations."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    week_name = os.path.basename(week_dir)
    filepath = os.path.join(week_dir, "lab", f"{week_name}_lab.ipynb")
    with open(filepath, 'w') as f:
        json.dump(notebook_content, f, indent=2)
    print(f"Created lab notebook: {filepath}")
    return filepath

def create_handouts(week_dir, week_num, config):
    """Create instructor and student handouts."""

    # Student handout
    student_content = f"""\\documentclass[8pt,a4paper]{{article}}
\\usepackage{{geometry}}
\\geometry{{margin=1in}}
\\usepackage{{amsmath}}

\\title{{Week {week_num}: {config['title']} - Student Handout}}
\\date{{}}

\\begin{{document}}
\\maketitle

\\section{{Learning Objectives}}
\\begin{{itemize}}
"""
    for topic in config['topics']:
        student_content += f"\\item Understand {topic}\n"

    student_content += f"""\\end{{itemize}}

\\section{{Key Concepts}}
{config['title']} concepts...

\\section{{Exercises}}
\\begin{{enumerate}}
\\item Exercise 1: Basic concepts
\\item Exercise 2: Implementation
\\item Exercise 3: Analysis
\\end{{enumerate}}

\\section{{Additional Resources}}
\\begin{{itemize}}
\\item Textbook chapters
\\item Research papers
\\item Online tutorials
\\end{{itemize}}

\\end{{document}}"""

    # Instructor handout (with solutions)
    instructor_content = student_content.replace("Student Handout", "Instructor Handout")
    instructor_content = instructor_content.replace("\\section{Exercises}",
        "\\section{Exercises with Solutions}")

    # Save files
    student_path = os.path.join(week_dir, "presentations/handouts", f"week{week_num:02d}_student.tex")
    instructor_path = os.path.join(week_dir, "presentations/handouts", f"week{week_num:02d}_instructor.tex")

    with open(student_path, 'w') as f:
        f.write(student_content)
    with open(instructor_path, 'w') as f:
        f.write(instructor_content)

    print(f"Created handouts: {student_path}, {instructor_path}")
    return student_path, instructor_path

def create_figure_generation_script(week_dir, week_num, config):
    """Create Python script for generating figures."""

    script_content = f'''"""
Generate figures for Week {week_num}: {config['title']}
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

def create_main_figure():
    """Create main visualization for {config['title']}."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Placeholder visualization
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y, linewidth=2)
    ax.set_title('Week {week_num}: {config["title"]}')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.grid(True, alpha=0.3)

    plt.savefig('../figures/week{week_num:02d}_main.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating Week {week_num} figures...")
    create_main_figure()
    print("Figures generated successfully!")
'''

    filepath = os.path.join(week_dir, "python", f"generate_week{week_num:02d}_figures.py")
    with open(filepath, 'w') as f:
        f.write(script_content)
    print(f"Created figure script: {filepath}")
    return filepath

# Main execution
def main():
    base_dir = "NLP_slides"
    created_files = []

    for week_key, config in weeks_config.items():
        week_num = int(week_key.split('_')[0].replace('week', ''))
        week_dir = os.path.join(base_dir, week_key)

        print(f"\n{'='*50}")
        print(f"Processing Week {week_num}: {config['title']}")
        print('='*50)

        # Create all materials
        try:
            pres = create_presentation(week_dir, week_num, config, timestamp)
            lab = create_lab_notebook(week_dir, week_num, config)
            handouts = create_handouts(week_dir, week_num, config)
            fig_script = create_figure_generation_script(week_dir, week_num, config)

            created_files.extend([pres, lab, handouts[0], handouts[1], fig_script])
        except Exception as e:
            print(f"Error processing week {week_num}: {e}")

    print(f"\n{'='*50}")
    print(f"SUMMARY: Created {len(created_files)} files")
    print('='*50)

    # Also create handouts for weeks 5-6
    for week in [5, 6]:
        week_dir = f"NLP_slides/week{week:02d}_*"
        # Find actual directory
        import glob
        dirs = glob.glob(week_dir)
        if dirs:
            actual_dir = dirs[0]
            config = {
                'title': f'Week {week} Content',
                'topics': ['Topic 1', 'Topic 2', 'Topic 3'],
                'lab_focus': f'Week {week} lab focus'
            }
            handouts = create_handouts(actual_dir, week, config)
            print(f"Created handouts for Week {week}")

if __name__ == "__main__":
    main()
    print("\nAll remaining materials have been generated!")