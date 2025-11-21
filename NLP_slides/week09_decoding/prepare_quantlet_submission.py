"""
Prepare QuantLet Submission for Week 9 Decoding Scripts
========================================================
Automatically creates QuantLet-compatible folder structure for all Python scripts.

Usage:
    python prepare_quantlet_submission.py              # Process all scripts
    python prepare_quantlet_submission.py --test       # Test with single script
    python prepare_quantlet_submission.py --list       # List scripts only

Output: quantlet_submission/ folder with one subfolder per script.

QuantLet Structure per folder:
    NLPDecoding_ScriptName/
    ├── Metainfo.txt          (YAML format for auto-README generation)
    ├── NLPDecoding_ScriptName.py
    ├── require_py.txt        (extracted dependencies)
    └── *.pdf (optional)      (generated figures if found)
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
QUANTLET_PREFIX = "NLPDecoding"

# Author info - UPDATE THIS
AUTHOR_NAME = "Joerg Osterrieder"
PUBLISHED_IN = "NLP Course - Language Model Decoding Strategies (Week 9)"

# Filter: only include scripts that START with "generate_" and DON'T contain "fix"
INCLUDE_PATTERN = r'^generate_'
# Exclude: fix scripts, CLEAN versions, and duplicate/redundant versions
EXCLUDE_PATTERNS = [
    r'fix',              # Fix scripts
    r'_CLEAN',           # Clean versions
    r'_bsc\.py$',        # Base BSC (keep enhanced)
    r'week09_figures',   # Redundant with enhanced
    r'week9_decoding_charts',  # Old version
    r'all_charts_graphviz',    # Meta-script
    r'missing_charts',         # Patch script
]

# Standard library modules (not external dependencies)
STDLIB_MODULES = {
    'os', 'sys', 're', 'ast', 'math', 'random', 'collections', 'itertools',
    'functools', 'datetime', 'time', 'json', 'csv', 'io', 'pathlib',
    'subprocess', 'shutil', 'copy', 'typing', 'warnings', 'pickle',
    'tempfile', 'glob', 'string', 'textwrap', 'abc', 'dataclasses',
    'contextlib', 'enum', 'operator', 'logging', 'argparse', 'hashlib'
}

# Package name mappings (import name -> pip package name)
PACKAGE_MAPPINGS = {
    'cv2': 'opencv-python',
    'PIL': 'Pillow',
    'sklearn': 'scikit-learn',
    'mpl_toolkits': 'matplotlib',
    'yaml': 'pyyaml',
}


def extract_docstring(filepath):
    """Extract the module docstring from a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)
        docstring = ast.get_docstring(tree)
        if docstring:
            # Clean up: take first meaningful paragraph
            lines = docstring.strip().split('\n')
            # Get non-empty lines
            meaningful = [l.strip() for l in lines if l.strip()]
            # Return first 2-3 sentences or 200 chars
            desc = ' '.join(meaningful[:3])
            if len(desc) > 250:
                desc = desc[:247] + '...'
            return desc
        return None
    except Exception as e:
        print(f"  Warning: Could not parse {filepath}: {e}")
        return None


def extract_imports(filepath):
    """Extract import statements and return list of external packages."""
    imports = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    imports.add(module)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split('.')[0]
                    imports.add(module)
    except Exception as e:
        print(f"  Warning: Could not parse imports from {filepath}: {e}")

    # Filter to external packages only
    external = []
    for imp in sorted(imports):
        if imp not in STDLIB_MODULES:
            # Map to correct pip package name
            pkg_name = PACKAGE_MAPPINGS.get(imp, imp)
            external.append(pkg_name)

    return list(set(external))  # Remove duplicates from mappings


def extract_keywords(filepath, description):
    """Generate keywords based on filename and description."""
    keywords = ['decoding', 'NLP', 'language model', 'text generation']

    filename = filepath.stem.lower()

    # Add keywords based on filename patterns
    keyword_patterns = {
        'beam': ['beam search', 'search algorithm'],
        'greedy': ['greedy decoding', 'argmax'],
        'temperature': ['temperature scaling', 'softmax'],
        'top_k': ['top-k sampling', 'truncation'],
        'nucleus': ['nucleus sampling', 'top-p'],
        'graphviz': ['visualization', 'graph'],
        'chart': ['visualization', 'matplotlib'],
        'animation': ['animation', 'visualization'],
        'probability': ['probability distribution', 'softmax'],
        'exploration': ['search space', 'exploration'],
    }

    for pattern, kws in keyword_patterns.items():
        if pattern in filename:
            keywords.extend(kws)

    # Add based on description content
    if description:
        desc_lower = description.lower()
        if 'scatter' in desc_lower:
            keywords.append('scatter plot')
        if 'tradeoff' in desc_lower or 'trade-off' in desc_lower:
            keywords.append('tradeoff analysis')

    return list(set(keywords))


def find_output_figures(script_path):
    """Find figures that might have been generated by this script."""
    figures = []
    script_name = script_path.stem.lower()

    if not FIGURES_DIR.exists():
        return figures

    # Common patterns in the script name to match figures
    name_parts = script_name.replace('generate_', '').replace('_', ' ').split()

    for fig_file in FIGURES_DIR.glob('*.pdf'):
        fig_lower = fig_file.stem.lower()
        # Check if any significant words match
        matches = sum(1 for part in name_parts if part in fig_lower and len(part) > 3)
        if matches >= 1:
            figures.append(fig_file)

    # Limit to 5 most relevant
    return figures[:5]


def create_quantlet_name(script_path):
    """Create QuantLet name from script filename."""
    # Remove generate_ prefix and .py suffix
    name = script_path.stem
    name = re.sub(r'^generate_', '', name)
    name = re.sub(r'_?charts?$', '', name)
    name = re.sub(r'_?bsc$', '', name)

    # Convert to CamelCase
    parts = name.split('_')
    camel = ''.join(p.capitalize() for p in parts if p)

    return f"{QUANTLET_PREFIX}_{camel}"


def create_metainfo(quantlet_name, description, keywords, author=AUTHOR_NAME):
    """Create YAML-formatted Metainfo.txt content."""
    keyword_str = ', '.join(keywords)

    # Escape single quotes in description
    if description:
        description = description.replace("'", "''")
    else:
        description = "Python script for NLP decoding visualization"

    metainfo = f"""Name of QuantLet: {quantlet_name}

Published in: {PUBLISHED_IN}

Description: '{description}'

Keywords: '{keyword_str}'

Author: {author}

Submitted: {datetime.now().strftime('%Y-%m-%d')}
"""
    return metainfo


def create_require_py(packages):
    """Create require_py.txt content."""
    if not packages:
        packages = ['numpy', 'matplotlib']  # Default minimum
    return '\n'.join(sorted(set(packages)))


def process_script(script_path, output_base, dry_run=False):
    """Process a single Python script into QuantLet format."""
    quantlet_name = create_quantlet_name(script_path)
    output_dir = output_base / quantlet_name

    print(f"\n  Processing: {script_path.name}")
    print(f"    QuantLet name: {quantlet_name}")

    # Extract info
    description = extract_docstring(script_path)
    imports = extract_imports(script_path)
    keywords = extract_keywords(script_path, description)
    figures = find_output_figures(script_path)

    print(f"    Description: {description[:60] if description else 'None'}...")
    print(f"    Dependencies: {imports}")
    print(f"    Keywords: {keywords[:5]}...")
    print(f"    Related figures: {len(figures)}")

    if dry_run:
        print("    [DRY RUN - not creating files]")
        return quantlet_name

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy Python script with QuantLet name
    dest_script = output_dir / f"{quantlet_name}.py"
    shutil.copy2(script_path, dest_script)
    print(f"    Created: {dest_script.name}")

    # 2. Create Metainfo.txt
    metainfo = create_metainfo(quantlet_name, description, keywords)
    metainfo_path = output_dir / "Metainfo.txt"
    with open(metainfo_path, 'w', encoding='utf-8') as f:
        f.write(metainfo)
    print(f"    Created: Metainfo.txt")

    # 3. Create require_py.txt
    require_content = create_require_py(imports)
    require_path = output_dir / "require_py.txt"
    with open(require_path, 'w', encoding='utf-8') as f:
        f.write(require_content)
    print(f"    Created: require_py.txt")

    # 4. Copy related figures (optional)
    for fig in figures[:3]:  # Limit to 3 figures per QuantLet
        dest_fig = output_dir / fig.name
        shutil.copy2(fig, dest_fig)
        print(f"    Copied: {fig.name}")

    return quantlet_name


def main():
    """Main entry point."""
    print("=" * 60)
    print("QuantLet Submission Preparation Tool")
    print("Week 9: Decoding Strategies")
    print("=" * 60)

    # Parse arguments
    test_mode = '--test' in sys.argv
    list_mode = '--list' in sys.argv

    # Find all Python scripts
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        print("Run this script from: NLP_slides/week09_decoding/")
        sys.exit(1)

    all_scripts = sorted(SOURCE_DIR.glob('*.py'))

    # Filter scripts: must match INCLUDE_PATTERN and not match any EXCLUDE_PATTERNS
    scripts = []
    excluded = []
    for s in all_scripts:
        name = s.name
        if re.match(INCLUDE_PATTERN, name):
            if not any(re.search(pat, name, re.IGNORECASE) for pat in EXCLUDE_PATTERNS):
                scripts.append(s)
            else:
                excluded.append(s.name)
        else:
            excluded.append(s.name)

    print(f"\nFound {len(all_scripts)} Python scripts in {SOURCE_DIR}/")
    print(f"Filtered to {len(scripts)} chart-generating scripts")
    if excluded:
        print(f"Excluded {len(excluded)}: {', '.join(excluded[:5])}{'...' if len(excluded) > 5 else ''}")

    if list_mode:
        print("\nScripts to process:")
        for i, s in enumerate(scripts, 1):
            print(f"  {i:2d}. {s.name}")
        return

    # Create output directory
    if not list_mode:
        OUTPUT_DIR.mkdir(exist_ok=True)
        print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")

    # Process scripts
    if test_mode:
        print("\n*** TEST MODE: Processing first script only ***")
        scripts = scripts[:1]

    quantlets_created = []
    for script in scripts:
        try:
            name = process_script(script, OUTPUT_DIR, dry_run=list_mode)
            quantlets_created.append(name)
        except Exception as e:
            print(f"  ERROR processing {script.name}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: Created {len(quantlets_created)} QuantLet folders")
    print("=" * 60)

    if quantlets_created and not list_mode:
        print(f"\nOutput location: {OUTPUT_DIR.absolute()}")
        print("\nNext steps:")
        print("  1. Review generated Metainfo.txt files for accuracy")
        print("  2. Push to your GitHub repository")
        print("  3. Contact QuantLet org to request inclusion:")
        print("     - Open issue at: https://github.com/QuantLet/Styleguide-and-FAQ")
        print("     - Or fork to QuantLet org if you have member access")


if __name__ == '__main__':
    main()
