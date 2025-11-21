"""
Prepare QuantLet Submission for Week 9 Decoding Scripts
========================================================
Creates ONE QuantLet folder per chart (not per script).

If a script generates multiple charts, it creates multiple folders,
each with a modified script that generates only ONE chart.

Usage:
    python prepare_quantlet_submission.py              # Process all
    python prepare_quantlet_submission.py --list       # List only
"""

import os
import re
import ast
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Configuration
SOURCE_DIR = Path("python")
FIGURES_DIR = Path("figures")
OUTPUT_DIR = Path("quantlet_submission")

# Author info
AUTHOR_NAME = "Joerg Osterrieder"
PUBLISHED_IN = "NLP Course - Language Model Decoding Strategies (Week 9)"

# Map of script -> list of (output_file, function_name, description)
# Only include scripts we want to submit
SCRIPT_CHARTS = {
    'generate_beam_search_graphviz.py': [
        ('beam_search_tree_graphviz.pdf', None, 'Beam search tree visualization with pruning using graphviz'),
    ],
    'generate_full_exploration_graphviz.py': [
        ('full_exploration_tree_graphviz.pdf', None, 'Full exploration explosion tree showing exponential growth'),
    ],
    'generate_greedy_suboptimal_comparison.py': [
        ('greedy_suboptimal_comparison.pdf', None, 'Greedy vs optimal path comparison showing suboptimal choices'),
    ],
    'generate_probability_distribution_animation.py': [
        ('temperature_effect.pdf', None, 'Temperature effect on probability distribution visualization'),
    ],
    'generate_weeks_pipeline_graphviz.py': [
        ('prediction_to_text_pipeline.pdf', None, 'NLP pipeline from model predictions to text output'),
    ],
    'generate_beam_search_step_by_step.py': [
        ('beam_search_step_by_step.pdf', None, 'Step-by-step beam search visualization with scores'),
    ],
    'generate_decoding_live_demo.py': [
        ('decoding_strategy_comparison.pdf', None, 'Side-by-side comparison of decoding strategies'),
    ],
    'generate_weeks_pipeline_chart.py': [
        ('weeks_pipeline.pdf', None, 'NLP course pipeline showing weeks 1-8 leading to decoding'),
    ],
}

# Standard library modules
STDLIB_MODULES = {
    'os', 'sys', 're', 'ast', 'math', 'random', 'collections', 'itertools',
    'functools', 'datetime', 'time', 'json', 'csv', 'io', 'pathlib',
    'subprocess', 'shutil', 'copy', 'typing', 'warnings', 'pickle',
    'tempfile', 'glob', 'string', 'textwrap', 'abc', 'dataclasses',
    'contextlib', 'enum', 'operator', 'logging', 'argparse', 'hashlib'
}

PACKAGE_MAPPINGS = {
    'cv2': 'opencv-python',
    'PIL': 'Pillow',
    'sklearn': 'scikit-learn',
    'mpl_toolkits': 'matplotlib',
    'yaml': 'pyyaml',
}


def extract_imports(filepath):
    """Extract external package imports."""
    imports = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except:
        pass

    external = []
    for imp in sorted(imports):
        if imp not in STDLIB_MODULES:
            external.append(PACKAGE_MAPPINGS.get(imp, imp))
    return list(set(external))


def create_quantlet_name(chart_name):
    """Create clean QuantLet folder name from chart filename."""
    name = chart_name.replace('.pdf', '').replace('.png', '')
    name = re.sub(r'_bsc$', '', name)

    # Convert to CamelCase
    parts = name.split('_')
    camel = ''.join(p.capitalize() for p in parts if p)

    return f"NLPDecoding_{camel}"


def fix_script_paths(source_code, output_filename):
    """Fix paths to output in current directory with correct filename."""
    fixed = source_code

    # Replace ../figures/ paths with ./
    fixed = re.sub(r"'\.\./figures/[^']*'", f"'./{output_filename}'", fixed)
    fixed = re.sub(r'"\.\./figures/[^"]*"', f'"./{output_filename}"', fixed)

    # Fix makedirs
    fixed = re.sub(r"os\.makedirs\s*\([^)]+\)", "os.makedirs('.', exist_ok=True)", fixed)

    # Add missing import for mpatches if needed
    if 'mpatches' in fixed and 'import matplotlib.patches' not in fixed.split('if __name__')[0]:
        if 'import matplotlib.pyplot as plt' in fixed:
            fixed = fixed.replace(
                'import matplotlib.pyplot as plt',
                'import matplotlib.pyplot as plt\nimport matplotlib.patches as mpatches'
            )

    return fixed


def create_metainfo(quantlet_name, description, keywords):
    """Create YAML-formatted Metainfo.txt."""
    keyword_str = ', '.join(keywords)
    if description:
        description = description.replace("'", "''")

    return f"""Name of QuantLet: {quantlet_name}

Published in: {PUBLISHED_IN}

Description: '{description}'

Keywords: '{keyword_str}'

Author: {AUTHOR_NAME}

Submitted: {datetime.now().strftime('%Y-%m-%d')}
"""


def process_chart(script_path, chart_info, output_base):
    """Process one chart from a script."""
    output_file, func_name, description = chart_info
    quantlet_name = create_quantlet_name(output_file)
    output_dir = output_base / quantlet_name

    print(f"\n  {quantlet_name}")
    print(f"    Source: {script_path.name}")
    print(f"    Output: {output_file}")

    # Create folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read and fix script
    with open(script_path, 'r', encoding='utf-8') as f:
        source = f.read()

    fixed_source = fix_script_paths(source, output_file)

    # Write script
    dest_script = output_dir / f"{quantlet_name}.py"
    with open(dest_script, 'w', encoding='utf-8') as f:
        f.write(fixed_source)

    # Create Metainfo.txt
    keywords = ['NLP', 'decoding', 'language model', 'text generation', 'visualization']
    if 'beam' in output_file.lower():
        keywords.extend(['beam search', 'search algorithm'])
    if 'greedy' in output_file.lower():
        keywords.extend(['greedy decoding', 'argmax'])
    if 'temperature' in output_file.lower():
        keywords.extend(['temperature scaling', 'softmax'])
    if 'pipeline' in output_file.lower():
        keywords.extend(['NLP pipeline', 'workflow'])

    metainfo = create_metainfo(quantlet_name, description, keywords)
    with open(output_dir / "Metainfo.txt", 'w', encoding='utf-8') as f:
        f.write(metainfo)

    # Create require_py.txt
    imports = extract_imports(script_path)
    if not imports:
        imports = ['numpy', 'matplotlib']
    with open(output_dir / "require_py.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(set(imports))))

    # Copy output PDF if it exists
    for fig_dir in [FIGURES_DIR, Path('presentations/figures')]:
        for pattern in [output_file, output_file.replace('.pdf', '_bsc.pdf')]:
            fig_path = fig_dir / pattern
            if fig_path.exists():
                shutil.copy2(fig_path, output_dir / output_file)
                print(f"    Copied: {output_file}")
                break

    return quantlet_name


def main():
    print("=" * 60)
    print("QuantLet Submission - One Chart Per Folder")
    print("Week 9: Decoding Strategies")
    print("=" * 60)

    list_mode = '--list' in sys.argv

    if not SOURCE_DIR.exists():
        print(f"ERROR: Run from NLP_slides/week09_decoding/")
        sys.exit(1)

    # Count charts
    total_charts = sum(len(charts) for charts in SCRIPT_CHARTS.values())
    print(f"\n{len(SCRIPT_CHARTS)} scripts -> {total_charts} charts (one folder each)")

    if list_mode:
        print("\nCharts to create:")
        for script, charts in SCRIPT_CHARTS.items():
            for chart_info in charts:
                output_file = chart_info[0]
                name = create_quantlet_name(output_file)
                print(f"  {name}")
                print(f"    <- {script}")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    created = []
    for script_name, charts in SCRIPT_CHARTS.items():
        script_path = SOURCE_DIR / script_name
        if not script_path.exists():
            print(f"\n  SKIP: {script_name} not found")
            continue

        for chart_info in charts:
            try:
                name = process_chart(script_path, chart_info, OUTPUT_DIR)
                created.append(name)
            except Exception as e:
                print(f"    ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"Created {len(created)} QuantLet folders (one chart each)")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_DIR.absolute()}")


if __name__ == '__main__':
    main()
