"""
Extract slides from LaTeX beamer presentation to JSON for React
Parses frame environments and extracts content, layout, and figures
"""

import re
import json
from pathlib import Path

def parse_latex_slides(tex_file_path):
    """Parse LaTeX beamer file and extract all slides"""

    with open(tex_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    slides = []
    slide_id = 0

    # Find all \begin{frame}...\end{frame} blocks
    frame_pattern = r'\\begin{frame}(?:\[([^\]]*)\])?\s*(?:{([^}]*)})?(.+?)\\end{frame}'
    frames = re.findall(frame_pattern, content, re.DOTALL)

    for options, title, frame_content in frames:
        slide_id += 1

        slide = {
            'id': slide_id,
            'options': options.strip() if options else '',
            'title': title.strip() if title else '',
            'content': {},
            'layout': determine_layout(frame_content),
            'figures': extract_figures(frame_content),
            'formulas': extract_formulas(frame_content),
            'lists': extract_lists(frame_content),
            'text': extract_plain_text(frame_content),
            'bottomNote': extract_bottom_note(frame_content),
            'hasColumns': '\\begin{columns}' in frame_content,
            'hasPause': '\\pause' in frame_content,
            'hasColorbox': 'beamercolorbox' in frame_content or 'colorbox' in frame_content
        }

        # Extract column content if two-column layout
        if slide['hasColumns']:
            slide['columns'] = extract_columns(frame_content)

        slides.append(slide)

    return slides

def determine_layout(content):
    """Determine slide layout type"""
    if '[plain]' in content or 'beamercolorbox' in content:
        return 'title'
    elif '\\begin{columns}' in content:
        return 'two-column'
    elif 'includegraphics' in content and not any(x in content for x in ['itemize', 'enumerate']):
        return 'figure-only'
    elif '$$' in content or '\\[' in content:
        return 'formula'
    else:
        return 'single-column'

def extract_figures(content):
    r"""Extract all \includegraphics commands"""
    figures = []
    pattern = r'\\includegraphics(?:\[([^\]]*)\])?\{([^}]+)\}'
    matches = re.findall(pattern, content)

    for options, path in matches:
        # Parse width from options
        width = 0.75  # default
        if 'width=' in options:
            width_match = re.search(r'width=([0-9.]+)\\textwidth', options)
            if width_match:
                width = float(width_match.group(1))

        # Clean path (remove ../figures/ prefix)
        clean_path = path.replace('../figures/', '').replace('.pdf', '')

        figures.append({
            'path': clean_path + '.pdf',  # Keep original PDF
            'width': width,
            'options': options
        })

    return figures

def extract_formulas(content):
    """Extract mathematical formulas"""
    formulas = []

    # Display math: $$...$$
    display_math = re.findall(r'\$\$(.*?)\$\$', content, re.DOTALL)
    for formula in display_math:
        formulas.append({
            'type': 'display',
            'content': formula.strip()
        })

    # Display math: \[...\]
    bracket_math = re.findall(r'\\\[(.*?)\\\]', content, re.DOTALL)
    for formula in bracket_math:
        formulas.append({
            'type': 'display',
            'content': formula.strip()
        })

    return formulas

def extract_lists(content):
    """Extract itemize and enumerate environments"""
    lists = []

    # Itemize
    itemize_pattern = r'\\begin{itemize}(.*?)\\end{itemize}'
    itemize_blocks = re.findall(itemize_pattern, content, re.DOTALL)
    for block in itemize_blocks:
        items = extract_list_items(block)
        lists.append({'type': 'itemize', 'items': items})

    # Enumerate
    enumerate_pattern = r'\\begin{enumerate}(.*?)\\end{enumerate}'
    enumerate_blocks = re.findall(enumerate_pattern, content, re.DOTALL)
    for block in enumerate_blocks:
        items = extract_list_items(block)
        lists.append({'type': 'enumerate', 'items': items})

    return lists

def extract_list_items(list_content):
    r"""Extract \item entries from list"""
    items = []
    item_pattern = r'\\item\s+([^\n]+(?:\n(?!\\item)[^\n]+)*)'
    matches = re.findall(item_pattern, list_content)
    for item in matches:
        # Clean up LaTeX commands
        clean_item = clean_latex(item.strip())
        items.append(clean_item)
    return items

def extract_plain_text(content):
    """Extract plain text paragraphs (not in lists)"""
    # Remove all LaTeX environments
    text = content
    for env in ['itemize', 'enumerate', 'columns', 'center', 'figure']:
        text = re.sub(rf'\\begin{{{env}}}.*?\\end{{{env}}}', '', text, flags=re.DOTALL)

    # Remove commands
    text = re.sub(r'\\includegraphics.*', '', text)
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\\bottomnote.*', '', text)

    # Clean up
    text = clean_latex(text)
    text = ' '.join(text.split())  # Normalize whitespace

    return text.strip() if text.strip() else None

def extract_bottom_note(content):
    """Extract \bottomnote{} content"""
    match = re.search(r'\\bottomnote\{([^}]+)\}', content)
    if match:
        return clean_latex(match.group(1))
    return None

def extract_columns(content):
    """Extract two-column layout content"""
    columns = []
    column_pattern = r'\\column\{([^}]+)\}(.*?)(?=\\column|\\end{columns})'
    matches = re.findall(column_pattern, content, re.DOTALL)

    for width, col_content in matches:
        columns.append({
            'width': width,
            'content': col_content.strip(),
            'figures': extract_figures(col_content),
            'lists': extract_lists(col_content),
            'text': extract_plain_text(col_content)
        })

    return columns

def clean_latex(text):
    """Clean LaTeX commands from text"""
    # Remove common commands
    text = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', text)  # Bold to markdown
    text = re.sub(r'\\textit\{([^}]+)\}', r'*\1*', text)    # Italic to markdown
    text = re.sub(r'\\texttt\{([^}]+)\}', r'`\1`', text)    # Code to markdown
    text = re.sub(r'\\emph\{([^}]+)\}', r'*\1*', text)      # Emphasis to markdown

    # Remove other commands
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\[a-zA-Z]+\s*', '', text)

    # Clean up special characters
    text = text.replace('\\&', '&')
    text = text.replace('\\%', '%')
    text = text.replace('\\_', '_')
    text = text.replace('~', ' ')

    return text.strip()

def assign_sections(slides):
    """Assign section names to slides based on content/position"""
    sections = [
        {'name': 'intro', 'start': 1, 'end': 3},
        {'name': 'extremes', 'start': 4, 'end': 10},
        {'name': 'toolbox', 'start': 11, 'end': 43},
        {'name': 'quiz1', 'start': 44, 'end': 44},
        {'name': 'problems', 'start': 45, 'end': 51},
        {'name': 'quiz2', 'start': 52, 'end': 52},
        {'name': 'integration', 'start': 53, 'end': 56},
        {'name': 'quiz3', 'start': 57, 'end': 57},
        {'name': 'conclusion', 'start': 58, 'end': 59},
        {'name': 'appendix', 'start': 60, 'end': 90}
    ]

    for slide in slides:
        for section in sections:
            if section['start'] <= slide['id'] <= section['end']:
                slide['section'] = section['name']
                break

    return slides

def main():
    # Path to LaTeX file
    tex_file = Path(__file__).parent / 'presentations' / '20251119_1100_week09_decoding_fixed.tex'

    if not tex_file.exists():
        print(f"Error: {tex_file} not found")
        return

    print(f"Parsing {tex_file}...")
    slides = parse_latex_slides(tex_file)

    print(f"Extracted {len(slides)} slides")

    # Assign sections
    slides = assign_sections(slides)

    # Create sections summary
    sections_summary = {}
    for slide in slides:
        section = slide.get('section', 'unknown')
        if section not in sections_summary:
            sections_summary[section] = []
        sections_summary[section].append(slide['id'])

    # Output JSON
    output = {
        'totalSlides': len(slides),
        'sections': sections_summary,
        'slides': slides
    }

    output_file = Path(__file__).parent / 'react-app' / 'src' / 'data' / 'extractedSlides.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nExtracted data saved to {output_file}")
    print(f"\nSection breakdown:")
    for section, slide_ids in sections_summary.items():
        print(f"  {section}: {len(slide_ids)} slides (IDs: {min(slide_ids)}-{max(slide_ids)})")

if __name__ == '__main__':
    main()
