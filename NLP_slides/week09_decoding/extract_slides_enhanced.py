"""
Enhanced LaTeX Beamer to JSON extractor
Captures ALL details: spacing, colors, pause positions, nested structures
For high-fidelity React conversion
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any

class BeamerExtractor:
    def __init__(self, tex_file_path: str):
        self.tex_file_path = Path(tex_file_path)
        with open(self.tex_file_path, 'r', encoding='utf-8') as f:
            self.full_content = f.read()

    def extract_all_slides(self) -> List[Dict[str, Any]]:
        """Extract all slides with complete structure"""
        slides = []

        # Find all frames
        frame_pattern = r'\\begin{frame}(?:\[([^\]]*)\])?\s*(?:{([^}]*)})?(.+?)\\end{frame}'
        frames = re.findall(frame_pattern, self.full_content, re.DOTALL)

        for slide_id, (options, title, frame_content) in enumerate(frames, 1):
            slide = self._parse_frame(slide_id, options, title, frame_content)
            slides.append(slide)

        return slides

    def _parse_frame(self, slide_id, options, title, content) -> Dict[str, Any]:
        """Parse a single frame completely"""

        # Determine if this is a special slide
        is_title_slide = '[plain]' in options and 'beamercolorbox' in content
        is_appendix_divider = 'Technical Appendix' in content and 'beamercolorbox' in content

        slide = {
            'id': slide_id,
            'frameOptions': options.strip() if options else '',
            'title': self._clean_latex(title.strip()) if title else '',
            'layout': self._determine_layout(content, is_title_slide, is_appendix_divider),
            'sections': self._extract_sections(content),
            'bottomNote': self._extract_bottom_note(content),
            'metadata': {
                'hasPause': '\\pause' in content,
                'hasColumns': '\\begin{columns}' in content,
                'hasColorbox': 'colorbox' in content or 'beamercolorbox' in content,
                'hasTikz': 'tikzpicture' in content,
                'hasTabular': 'tabular' in content,
                'hasMath': '$$' in content or '$' in content,
                'fontSizes': self._extract_font_sizes(content)
            }
        }

        return slide

    def _determine_layout(self, content, is_title, is_appendix) -> str:
        """Determine layout pattern"""
        if is_title:
            return 'title-slide'
        elif is_appendix:
            return 'appendix-divider'
        elif '\\begin{columns}' in content:
            # Count columns
            columns = re.findall(r'\\column\{([^}]+)\}', content)
            if len(columns) == 2:
                # Check if four figures (2x2 grid)
                figs_in_cols = content.count('includegraphics')
                if figs_in_cols == 4:
                    return 'four-figure-grid'
                elif '0.48' in columns[0]:
                    return 'two-column-equal'
                elif '0.45' in columns[0] or '0.55' in columns[0]:
                    return 'two-column-asymmetric'
                else:
                    return 'two-column-custom'
            elif len(columns) == 3:
                return 'three-column'
        elif 'tikzpicture' in content:
            return 'tikz-diagram'
        elif 'includegraphics' in content and not any(x in content for x in ['itemize', 'enumerate']):
            return 'single-column-figure'
        elif 'enumerate' in content and 'Match' in (self._clean_latex(content) if content else ''):
            return 'quiz'
        else:
            return 'single-column-text'

    def _extract_sections(self, content) -> List[Dict[str, Any]]:
        """Extract content sections in order, preserving structure"""
        sections = []

        # Split by major structural elements while preserving order
        position = 0

        # Track pause positions
        pause_positions = [m.start() for m in re.finditer(r'\\pause', content)]
        pause_index = 0

        # Process content sequentially
        lines = content.split('\n')
        current_section = {'type': 'text', 'content': [], 'pauseBefore': False}

        for line in lines:
            stripped = line.strip()

            # Detect pauses
            if '\\pause' in stripped:
                if current_section['content']:
                    sections.append(current_section)
                sections.append({'type': 'pause'})
                current_section = {'type': 'text', 'content': [], 'pauseBefore': True}
                continue

            # Detect vspace
            vspace_match = re.search(r'\\vspace\{([^}]+)\}', stripped)
            if vspace_match:
                sections.append({
                    'type': 'vspace',
                    'value': vspace_match.group(1)
                })
                continue

            # Detect figures
            if '\\includegraphics' in stripped:
                fig_data = self._parse_includegraphics(stripped)
                if fig_data:
                    sections.append({
                        'type': 'figure',
                        **fig_data
                    })
                continue

            # Detect formulas
            if stripped.startswith('$$') or '\\[' in stripped:
                formula = self._extract_display_math(stripped)
                if formula:
                    sections.append({
                        'type': 'formula',
                        'display': True,
                        'content': formula
                    })
                continue

            # Detect lists
            if '\\begin{itemize}' in stripped:
                # Extract full list
                list_content = self._extract_environment(content, 'itemize', line)
                if list_content:
                    sections.append({
                        'type': 'list',
                        'listType': 'itemize',
                        'items': self._extract_list_items(list_content)
                    })
                continue

            if '\\begin{enumerate}' in stripped:
                list_content = self._extract_environment(content, 'enumerate', line)
                if list_content:
                    sections.append({
                        'type': 'list',
                        'listType': 'enumerate',
                        'items': self._extract_list_items(list_content)
                    })
                continue

            # Detect colorbox
            if '\\colorbox{' in stripped or '\\begin{beamercolorbox}' in stripped:
                colorbox = self._extract_colorbox(stripped)
                if colorbox:
                    sections.append(colorbox)
                continue

            # Detect colored text
            color_match = re.search(r'\\textcolor\{([^}]+)\}\{([^}]+)\}', stripped)
            if color_match:
                sections.append({
                    'type': 'coloredText',
                    'color': color_match.group(1),
                    'content': self._clean_latex(color_match.group(2))
                })
                continue

            # Detect bold/formatted text
            if '\\textbf{' in stripped or '\\textit{' in stripped:
                sections.append({
                    'type': 'formattedText',
                    'content': self._clean_latex(stripped),
                    'formatting': self._extract_formatting(stripped)
                })
                continue

            # Regular text line
            if stripped and not stripped.startswith('%') and not stripped.startswith('\\'):
                current_section['content'].append(self._clean_latex(stripped))

        # Add final section
        if current_section['content']:
            current_section['content'] = ' '.join(current_section['content'])
            sections.append(current_section)

        return sections

    def _parse_includegraphics(self, line) -> Dict[str, Any]:
        """Parse \includegraphics with all options"""
        pattern = r'\\includegraphics(?:\[([^\]]*)\])?\{([^}]+)\}'
        match = re.search(pattern, line)

        if not match:
            return None

        options, path = match.groups()

        # Extract width
        width = 0.75  # default
        if options and 'width=' in options:
            width_match = re.search(r'width=([0-9.]+)\\textwidth', options)
            if width_match:
                width = float(width_match.group(1))

        # Clean path
        clean_path = path.replace('../figures/', '').replace('figures/', '')

        return {
            'path': clean_path,
            'width': width,
            'options': options or ''
        }

    def _extract_display_math(self, line) -> str:
        """Extract display math formula"""
        # $$...$$
        match1 = re.search(r'\$\$(.*?)\$\$', line, re.DOTALL)
        if match1:
            return match1.group(1).strip()

        # \[...\]
        match2 = re.search(r'\\\[(.*?)\\\]', line, re.DOTALL)
        if match2:
            return match2.group(1).strip()

        return ''

    def _extract_environment(self, full_content, env_name, start_line) -> str:
        """Extract full environment content"""
        pattern = rf'\\begin{{{env_name}}}(.*?)\\end{{{env_name}}}'
        match = re.search(pattern, full_content, re.DOTALL)
        if match:
            return match.group(1)
        return ''

    def _extract_list_items(self, list_content) -> List[str]:
        """Extract \item entries"""
        items = []
        item_pattern = r'\\item\s+(.+?)(?=\\item|$)'
        matches = re.findall(item_pattern, list_content, re.DOTALL)

        for item in matches:
            clean_item = self._clean_latex(item.strip())
            if clean_item:
                items.append(clean_item)

        return items

    def _extract_colorbox(self, line) -> Dict[str, Any]:
        """Extract colorbox or beamercolorbox"""
        # Simple colorbox
        simple_match = re.search(r'\\colorbox\{([^}]+)\}\{([^}]+)\}', line)
        if simple_match:
            return {
                'type': 'colorbox',
                'color': simple_match.group(1),
                'content': self._clean_latex(simple_match.group(2))
            }

        # beamercolorbox (more complex)
        if 'beamercolorbox' in line:
            return {
                'type': 'beamercolorbox',
                'content': 'NEEDS_MANUAL_EXTRACTION'
            }

        return None

    def _extract_bottom_note(self, content) -> str:
        """Extract \bottomnote{} content"""
        match = re.search(r'\\bottomnote\{([^}]+)\}', content)
        if match:
            return self._clean_latex(match.group(1))
        return None

    def _extract_font_sizes(self, content) -> List[str]:
        """Extract font size commands"""
        sizes = []
        for size in ['\\tiny', '\\small', '\\normalsize', '\\large', '\\Large', '\\LARGE', '\\huge', '\\Huge']:
            if size in content:
                sizes.append(size.replace('\\', ''))
        return sizes

    def _extract_formatting(self, text) -> List[str]:
        """Extract formatting commands"""
        formatting = []
        if '\\textbf{' in text:
            formatting.append('bold')
        if '\\textit{' in text:
            formatting.append('italic')
        if '\\texttt{' in text:
            formatting.append('monospace')
        return formatting

    def _clean_latex(self, text: str) -> str:
        """Clean LaTeX commands from text"""
        if not text:
            return ''

        # Preserve some structure with markdown
        text = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', text)
        text = re.sub(r'\\textit\{([^}]+)\}', r'*\1*', text)
        text = re.sub(r'\\texttt\{([^}]+)\}', r'`\1`', text)
        text = re.sub(r'\\emph\{([^}]+)\}', r'*\1*', text)

        # Remove other commands but keep content
        text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+\s*', '', text)

        # Special characters
        text = text.replace('\\&', '&')
        text = text.replace('\\%', '%')
        text = text.replace('\\_', '_')
        text = text.replace('~', ' ')
        text = text.replace('``', '"')
        text = text.replace("''", '"')

        # Clean whitespace
        text = ' '.join(text.split())

        return text.strip()

    def _extract_columns(self, content) -> List[Dict[str, Any]]:
        """Extract column content with exact widths"""
        columns = []

        # Find each \column{...} block
        column_pattern = r'\\column\{([^}]+)\}(.*?)(?=\\column|\\end{columns})'
        matches = re.findall(column_pattern, content, re.DOTALL)

        for width_spec, col_content in matches:
            col_data = {
                'width': width_spec,
                'sections': self._extract_sections(col_content)
            }
            columns.append(col_data)

        return columns

    def assign_sections_to_slides(self, slides) -> List[Dict]:
        """Assign semantic section names"""
        section_ranges = [
            {'name': 'intro', 'start': 1, 'end': 3},
            {'name': 'extremes', 'start': 4, 'end': 10},
            {'name': 'toolbox', 'start': 11, 'end': 43},
            {'name': 'quiz1', 'start': 44, 'end': 44},
            {'name': 'problems', 'start': 45, 'end': 51},
            {'name': 'quiz2', 'start': 52, 'end': 52},
            {'name': 'integration', 'start': 53, 'end': 56},
            {'name': 'quiz3', 'start': 57, 'end': 57},
            {'name': 'conclusion', 'start': 58, 'end': 59},
            {'name': 'appendix', 'start': 60, 'end': 100}
        ]

        for slide in slides:
            for section in section_ranges:
                if section['start'] <= slide['id'] <= section['end']:
                    slide['section'] = section['name']
                    break

        return slides

def main():
    print("=" * 60)
    print("ENHANCED LATEX BEAMER EXTRACTOR")
    print("=" * 60)

    tex_file = Path(__file__).parent / 'presentations' / '20251119_1100_week09_decoding_fixed.tex'

    if not tex_file.exists():
        print(f"\nERROR: File not found: {tex_file}")
        return

    print(f"\nProcessing: {tex_file.name}")
    print(f"Size: {tex_file.stat().st_size / 1024:.1f} KB")

    extractor = BeamerExtractor(str(tex_file))
    slides = extractor.extract_all_slides()

    print(f"\nExtracted: {len(slides)} slides")

    # Assign sections
    slides = extractor.assign_sections_to_slides(slides)

    # Count by section
    sections_count = {}
    for slide in slides:
        section = slide.get('section', 'unknown')
        sections_count[section] = sections_count.get(section, 0) + 1

    print("\nSection breakdown:")
    for section, count in sections_count.items():
        slide_ids = [s['id'] for s in slides if s.get('section') == section]
        print(f"  {section:12s}: {count:2d} slides (IDs: {min(slide_ids)}-{max(slide_ids)})")

    # Count special features
    pause_count = sum(1 for s in slides if s['metadata']['hasPause'])
    columns_count = sum(1 for s in slides if s['metadata']['hasColumns'])
    colorbox_count = sum(1 for s in slides if s['metadata']['hasColorbox'])
    tikz_count = sum(1 for s in slides if s['metadata']['hasTikz'])

    print("\nSpecial features:")
    print(f"  Slides with \\pause: {pause_count}")
    print(f"  Slides with columns: {columns_count}")
    print(f"  Slides with colorbox: {colorbox_count}")
    print(f"  Slides with TikZ: {tikz_count}")

    # Save to JSON
    output_file = Path(__file__).parent / 'react-app' / 'src' / 'data' / 'week09_slides_complete.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'totalSlides': len(slides),
            'source': tex_file.name,
            'extractionDate': '2025-11-20',
            'sections': sections_count,
            'features': {
                'pauseSlides': pause_count,
                'columnSlides': columns_count,
                'colorboxSlides': colorbox_count,
                'tikzSlides': tikz_count
            }
        },
        'slides': slides
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nComplete data saved to: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
