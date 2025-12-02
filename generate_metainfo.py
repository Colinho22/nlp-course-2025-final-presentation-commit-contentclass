"""
Generate Quantlet metainfo.txt files for all chart folders.

This script:
1. Scans all numbered chart folders (01-20)
2. Extracts metadata from CHART_METADATA in Python files
3. Generates standard Quantlet metainfo.txt in each folder

Standard Quantlet metainfo format:
- Name of Quantlet
- Published in
- Description
- Keywords
- Author
- Submitted
- Datafile
- Input
- Output
- Example
"""
import ast
import re
from pathlib import Path
from datetime import datetime


def extract_metadata_from_chart(py_file):
    """Extract CHART_METADATA from a chart Python file."""
    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find CHART_METADATA dictionary
        pattern = r'CHART_METADATA\s*=\s*(\{[^}]+\})'
        match = re.search(pattern, content, re.DOTALL)

        if match:
            dict_str = match.group(1)
            metadata = ast.literal_eval(dict_str)
            return metadata
        return {}

    except Exception as e:
        print(f"    ERROR extracting metadata: {e}")
        return {}


def detect_files_in_folder(folder):
    """Detect input/output files in chart folder."""
    files = {
        'python': list(folder.glob('*.py')),
        'pdf': list(folder.glob('*.pdf')),
        'png': list(folder.glob('*.png')),
        'csv': list(folder.glob('*.csv')),
        'json': list(folder.glob('*.json'))
    }
    return files


def generate_keywords_from_name(name, description):
    """Generate keywords from name and description."""
    keywords = []

    # Common ML/AI keywords
    ml_keywords = ['neural network', 'machine learning', 'deep learning', 'visualization',
                   'gradient descent', 'backpropagation', 'activation function', 'loss function',
                   'overfitting', 'prediction', 'classification', 'regression']

    name_lower = name.lower()
    desc_lower = description.lower()

    for keyword in ml_keywords:
        if keyword in name_lower or keyword in desc_lower:
            keywords.append(keyword)

    # Add specific keywords from folder name
    if 'neuron' in name_lower:
        keywords.append('neuron')
    if 'loss' in name_lower or 'landscape' in name_lower:
        keywords.extend(['loss function', 'optimization'])
    if 'gradient' in name_lower:
        keywords.append('gradient descent')
    if 'forward' in name_lower or 'propagation' in name_lower:
        keywords.append('forward propagation')
    if 'activation' in name_lower:
        keywords.append('activation function')

    # Remove duplicates and limit to 5
    keywords = list(dict.fromkeys(keywords))[:5]

    # Always include these core keywords
    if 'neural network' not in keywords:
        keywords.insert(0, 'neural network')
    if 'visualization' not in keywords and len(keywords) < 5:
        keywords.append('visualization')

    return keywords


def create_metainfo(folder, metadata, files):
    """Create Quantlet metainfo.txt content."""

    # Extract metadata fields
    name = metadata.get('name', folder.name)
    description = metadata.get('description', f'Visualization for {name}')
    author = metadata.get('author', 'Digital-AI-Finance')
    created = metadata.get('created', datetime.now().strftime('%Y-%m-%d'))
    license_type = metadata.get('license', 'MIT License')

    # Generate keywords
    keywords = generate_keywords_from_name(name, description)
    keywords_str = ', '.join(keywords)

    # Detect files
    python_files = [f.name for f in files['python']]
    pdf_files = [f.name for f in files['pdf']]
    png_files = [f.name for f in files['png']]
    csv_files = [f.name for f in files['csv']]

    # Determine datafile
    datafile = ', '.join(csv_files) if csv_files else 'None'

    # Input files (Python scripts)
    input_files = ', '.join(python_files) if python_files else 'None'

    # Output files (PDFs and PNGs)
    output_list = pdf_files + png_files
    output_files = ', '.join(output_list) if output_list else 'None'

    # Example usage
    if python_files:
        example = f'python {python_files[0]}'
    else:
        example = 'See README for usage'

    # Create metainfo content
    metainfo = f"""Name of Quantlet: '{folder.name}'

Published in: 'Neural Networks for Business Applications'

Description: '{description}'

Keywords: '{keywords_str}'

Author: '{author}'

Submitted: '{created}'

Datafile: '{datafile}'

Input: '{input_files}'

Output: '{output_files}'

Example: '{example}'
"""

    return metainfo


def generate_all_metainfo():
    """Generate metainfo.txt for all chart folders."""
    print("Generating Quantlet metainfo.txt files...\n")

    # Find all chart folders (numbered)
    chart_folders = sorted([f for f in Path('.').iterdir()
                           if f.is_dir() and f.name[0:2].isdigit()])

    if not chart_folders:
        print("ERROR: No chart folders found")
        return

    print(f"Found {len(chart_folders)} chart folders\n")

    success_count = 0

    for folder in chart_folders:
        print(f"Processing: {folder.name}")

        # Find Python file
        py_files = list(folder.glob('*.py'))

        if not py_files:
            print(f"    WARNING: No Python file found, skipping")
            continue

        # Extract metadata
        metadata = extract_metadata_from_chart(py_files[0])

        if not metadata:
            print(f"    WARNING: No CHART_METADATA found")
            metadata = {
                'name': folder.name.replace('_', ' ').title(),
                'description': f'Chart for {folder.name}',
                'author': 'Digital-AI-Finance',
                'created': datetime.now().strftime('%Y-%m-%d')
            }

        # Detect files
        files = detect_files_in_folder(folder)

        # Generate metainfo content
        metainfo_content = create_metainfo(folder, metadata, files)

        # Write to file
        metainfo_path = folder / 'metainfo.txt'
        with open(metainfo_path, 'w', encoding='utf-8') as f:
            f.write(metainfo_content)

        print(f"    >> Created: {metainfo_path}")
        success_count += 1
        print()

    print("="*78)
    print(f"COMPLETE: Generated {success_count}/{len(chart_folders)} metainfo.txt files")
    print("="*78)


if __name__ == '__main__':
    generate_all_metainfo()
