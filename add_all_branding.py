"""
Add Quantlet branding to ALL chart slides in the Final Lecture.
"""
import re
import shutil
from pathlib import Path
from datetime import datetime

# Configuration
REPO_URL = "https://github.com/Digital-AI-Finance/Natural-Language-Processing"
LOGO_PATH = "../logo/quantlet.png"

# Build mapping from PDF name to Quantlet folder
def build_chart_mapping():
    """Scan FinalLecture folder and build PDF -> folder mapping."""
    mapping = {}
    final_lecture_dir = Path('FinalLecture')

    for folder in final_lecture_dir.iterdir():
        if folder.is_dir():
            for pdf in folder.glob('*.pdf'):
                mapping[pdf.name] = folder.name

    return mapping


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
      \\href{{{chart_url}}}{{\\includegraphics[width=0.55cm]{{../FinalLecture/{chart_folder}/qr_code.png}}}}
    }};
    % URL text
    \\node[anchor=south east,xshift=-0.3cm,yshift=0.15cm] at (current page.south east) {{
      \\href{{{chart_url}}}{{\\tiny\\texttt{{\\textcolor{{gray}}{{{short_path}}}}}}}
    }};
  \\end{{tikzpicture}}'''

    return tikz_code


def add_branding_to_latex():
    """Add branding to LaTeX file for ALL chart slides."""
    print("Adding Quantlet branding to ALL chart slides...")
    print("=" * 60)

    # Build chart mapping
    chart_mapping = build_chart_mapping()
    print(f"Found {len(chart_mapping)} charts in FinalLecture/\n")

    # Find the latest tex file (use the one we already created)
    presentations_dir = Path('presentations')

    # Use the original file without branding as base
    original_file = presentations_dir / '20251128_0945_final_lecture.tex'
    if not original_file.exists():
        # Try previous folder
        original_file = Path('previous') / '20251128_0945_final_lecture.tex'

    if not original_file.exists():
        print("ERROR: Original tex file not found")
        return None

    print(f"Using base file: {original_file}\n")

    # Read content
    with open(original_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Ensure tikz package is loaded
    if r'\usepackage{tikz}' not in content:
        content = content.replace(
            r'\usepackage{graphicx}',
            r'\usepackage{graphicx}' + '\n' + r'\usepackage{tikz}'
        )
        print("[INFO] Added tikz package\n")

    # Find all frames and add branding
    branding_count = 0

    # Pattern to find PDF includes
    pdf_pattern = r'\\includegraphics\[.*?\]\{[^}]*?([^/}]+\.pdf)\}'

    # Process frame by frame
    frame_pattern = r'(\\begin\{frame\}.*?)(\\end\{frame\})'

    def process_frame(match):
        nonlocal branding_count
        frame_content = match.group(1)
        frame_end = match.group(2)

        # Find PDF in this frame
        pdf_match = re.search(pdf_pattern, frame_content)
        if not pdf_match:
            return match.group(0)

        pdf_name = pdf_match.group(1)

        # Check if we have a Quantlet folder for this PDF
        if pdf_name not in chart_mapping:
            return match.group(0)

        # Check if branding already exists
        if 'Quantlet branding' in frame_content:
            return match.group(0)

        folder_name = chart_mapping[pdf_name]
        chart_url = f"{REPO_URL}/tree/main/FinalLecture/{folder_name}"
        tikz_code = create_branding_tikz(folder_name, chart_url)

        branding_count += 1
        print(f"  [{branding_count:2d}] {pdf_name} -> {folder_name}")

        return frame_content + tikz_code + '\n' + frame_end

    # Apply branding
    new_content = re.sub(frame_pattern, process_frame, content, flags=re.DOTALL)

    if branding_count == 0:
        print("\n[INFO] No new branding added")
        return None

    # Save new file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    new_filename = f"{timestamp}_final_lecture.tex"
    new_path = presentations_dir / new_filename

    with open(new_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"\n[SAVED] {new_path}")
    print(f"Added branding to {branding_count} chart slides")
    print("=" * 60)

    return new_path


if __name__ == '__main__':
    result = add_branding_to_latex()
    if result:
        print(f"\nNext: pdflatex {result}")
