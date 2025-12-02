"""
Add Quantlet branding to Final Lecture slides.

This script:
1. Adds CHART_METADATA with URLs to FinalLecture Python scripts
2. Generates QR codes for each chart
3. Modifies the LaTeX file to add logo + QR + URL overlay on chart slides
"""
import re
import os
import shutil
from pathlib import Path
from datetime import datetime

# Try to import qrcode
try:
    import qrcode
    HAS_QRCODE = True
except ImportError:
    HAS_QRCODE = False
    print("WARNING: qrcode not installed. Run: pip install qrcode[pil]")

# Configuration
REPO_URL = "https://github.com/Digital-AI-Finance/Natural-Language-Processing"
LOGO_PATH = "logo/quantlet.png"

# Chart mapping: PDF filename -> FinalLecture folder
CHART_MAPPING = {
    'ann_math_concept.pdf': '01_ann_math_concept',
    'hnsw_explanation.pdf': '02_hnsw_explanation',
    'hnsw_cities_example.pdf': '03_hnsw_cities_example',
    'hybrid_search_flow.pdf': '04_hybrid_search_flow',
    'rag_failures_flowchart.pdf': '05_rag_failures',
    'rag_conditional_probs.pdf': '06_rag_conditional_probs',
    'rag_venn_diagrams.pdf': '07_rag_venn_diagrams',
    'vector_db_architecture.pdf': '08_vector_db_architecture',
}


def add_chart_metadata_to_scripts():
    """Add CHART_METADATA with URLs to all FinalLecture Python scripts."""
    print("\n1. Adding CHART_METADATA to Python scripts...")
    print("-" * 60)

    final_lecture_dir = Path('FinalLecture')

    for pdf_name, folder_name in CHART_MAPPING.items():
        folder_path = final_lecture_dir / folder_name
        if not folder_path.exists():
            print(f"  [SKIP] {folder_name}: folder not found")
            continue

        # Find Python script
        py_files = list(folder_path.glob('generate_*.py'))
        if not py_files:
            print(f"  [SKIP] {folder_name}: no Python script")
            continue

        py_file = py_files[0]

        # Read current content
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if CHART_METADATA already exists
        if 'CHART_METADATA' in content:
            print(f"  [EXISTS] {folder_name}: CHART_METADATA already present")
            continue

        # Create URL
        chart_url = f"{REPO_URL}/tree/main/FinalLecture/{folder_name}"

        # Create CHART_METADATA block
        metadata_block = f'''
# Quantlet metadata for branding
CHART_METADATA = {{
    'name': '{folder_name}',
    'url': '{chart_url}'
}}
'''

        # Insert after imports (find last import line)
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_idx = i + 1

        # Insert metadata
        lines.insert(insert_idx, metadata_block)
        new_content = '\n'.join(lines)

        # Write back
        with open(py_file, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"  [OK] {folder_name}: Added CHART_METADATA")

    print()


def generate_qr_codes():
    """Generate QR codes for each chart folder."""
    print("\n2. Generating QR codes...")
    print("-" * 60)

    if not HAS_QRCODE:
        print("  [ERROR] qrcode module not available")
        return False

    final_lecture_dir = Path('FinalLecture')

    for pdf_name, folder_name in CHART_MAPPING.items():
        folder_path = final_lecture_dir / folder_name
        if not folder_path.exists():
            continue

        # Create URL
        chart_url = f"{REPO_URL}/tree/main/FinalLecture/{folder_name}"

        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=2,
        )
        qr.add_data(chart_url)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        # Save QR code
        qr_path = folder_path / "qr_code.png"
        img.save(qr_path)

        print(f"  [OK] {folder_name}/qr_code.png")

    print()
    return True


def create_branding_tikz(chart_folder, chart_url):
    """Create tikz overlay code for branding."""
    short_path = chart_folder.replace('_', r'\_')

    tikz_code = f'''
  % Quantlet branding (auto-generated)
  \\begin{{tikzpicture}}[remember picture,overlay]
    % Logo (clickable)
    \\node[anchor=south east,xshift=-0.3cm,yshift=0.5cm] at (current page.south east) {{
      \\href{{{chart_url}}}{{\\includegraphics[width=0.7cm]{{{LOGO_PATH}}}}}
    }};
    % QR Code (clickable)
    \\node[anchor=south east,xshift=-1.2cm,yshift=0.5cm,opacity=0.85] at (current page.south east) {{
      \\href{{{chart_url}}}{{\\includegraphics[width=0.55cm]{{FinalLecture/{chart_folder}/qr_code.png}}}}
    }};
    % URL text
    \\node[anchor=south east,xshift=-0.3cm,yshift=0.15cm] at (current page.south east) {{
      \\href{{{chart_url}}}{{\\tiny\\texttt{{\\textcolor{{gray}}{{{short_path}}}}}}}
    }};
  \\end{{tikzpicture}}'''

    return tikz_code


def add_branding_to_latex():
    """Add branding to LaTeX file for chart slides."""
    print("\n3. Adding branding to LaTeX slides...")
    print("-" * 60)

    # Find the latest tex file
    presentations_dir = Path('presentations')
    tex_files = sorted(presentations_dir.glob('*.tex'), reverse=True)

    if not tex_files:
        print("  [ERROR] No .tex file found in presentations/")
        return None

    tex_file = tex_files[0]
    print(f"  Processing: {tex_file.name}")

    # Read content
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if tikz package is loaded
    if r'\usepackage{tikz}' not in content:
        # Add tikz package after documentclass
        content = content.replace(
            r'\documentclass',
            r'\usepackage{tikz}' + '\n' + r'\documentclass'
        )
        # Actually, insert after other packages
        if r'\usepackage{hyperref}' in content:
            content = content.replace(
                r'\usepackage{hyperref}',
                r'\usepackage{hyperref}' + '\n' + r'\usepackage{tikz}'
            )
        print("  [INFO] Added tikz package")

    # Find frames with charts from figures/ folder and add branding
    branding_count = 0

    # Pattern to match frames with includegraphics
    frame_pattern = r'(\\begin\{frame\}.*?)(\\end\{frame\})'

    def add_branding_to_frame(match):
        nonlocal branding_count
        frame_start = match.group(1)
        frame_end = match.group(2)

        # Check if frame contains a chart we have in FinalLecture
        for pdf_name, folder_name in CHART_MAPPING.items():
            if pdf_name in frame_start:
                # Check if branding already added
                if 'Quantlet branding' in frame_start:
                    return match.group(0)

                chart_url = f"{REPO_URL}/tree/main/FinalLecture/{folder_name}"
                tikz_code = create_branding_tikz(folder_name, chart_url)

                branding_count += 1
                print(f"  [OK] Added branding for: {pdf_name}")

                return frame_start + tikz_code + '\n' + frame_end

        return match.group(0)

    # Apply branding
    new_content = re.sub(frame_pattern, add_branding_to_frame, content, flags=re.DOTALL)

    if branding_count == 0:
        print("  [INFO] No matching chart frames found")
        return None

    # Create backup
    backup_dir = Path('previous')
    backup_dir.mkdir(exist_ok=True)
    shutil.copy2(tex_file, backup_dir / tex_file.name)
    print(f"  [BACKUP] {backup_dir / tex_file.name}")

    # Save new file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    new_filename = f"{timestamp}_final_lecture.tex"
    new_path = presentations_dir / new_filename

    with open(new_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"\n  [SAVED] {new_path}")
    print(f"  Added branding to {branding_count} chart slides")

    return new_path


def main():
    """Main execution."""
    print("=" * 60)
    print("QUANTLET BRANDING FOR NLP FINAL LECTURE")
    print("=" * 60)
    print(f"Repository: {REPO_URL}")

    # Step 1: Add CHART_METADATA
    add_chart_metadata_to_scripts()

    # Step 2: Generate QR codes
    if HAS_QRCODE:
        generate_qr_codes()
    else:
        print("\n2. [SKIP] QR code generation (install qrcode module)")

    # Step 3: Add branding to LaTeX
    new_tex = add_branding_to_latex()

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)

    if new_tex:
        print(f"\nNext step: Compile {new_tex}")
        print("  pdflatex " + str(new_tex))

    return new_tex


if __name__ == '__main__':
    main()
