"""
Detect potential overflow issues in Beamer LaTeX presentation.
Analyzes figure sizes, content density, and common overflow patterns.
"""

import re
import os

TEX_FILE = r"D:\Joerg\Research\slides\NLPFinalLecture\presentations\20251127_1324_final_lecture.tex"

def analyze_presentation(tex_path):
    """Analyze presentation for potential overflow issues."""

    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into frames
    frame_pattern = r'\\begin\{frame\}(?:\[.*?\])?\{([^}]*)\}(.*?)\\end\{frame\}'
    frames = re.findall(frame_pattern, content, re.DOTALL)

    issues = []

    for i, (title, body) in enumerate(frames, 1):
        frame_issues = []

        # Check figure sizes
        figure_matches = re.findall(r'\\includegraphics\[width=([0-9.]+)\\textwidth\]', body)
        for width in figure_matches:
            w = float(width)
            if w > 0.8:
                frame_issues.append(f"Large figure width: {w}\\textwidth (recommend <= 0.8)")

        # Check for multiple figures
        fig_count = len(re.findall(r'\\includegraphics', body))
        if fig_count > 1:
            frame_issues.append(f"Multiple figures: {fig_count} images")

        # Check itemize/enumerate depth and item count
        items = re.findall(r'\\item', body)
        if len(items) > 8:
            frame_issues.append(f"Many items: {len(items)} (recommend <= 8)")

        # Check for nested lists
        nested = re.findall(r'\\begin\{itemize\}.*?\\begin\{itemize\}', body, re.DOTALL)
        if nested:
            frame_issues.append("Nested itemize detected")

        # Check content length (rough estimate)
        text_content = re.sub(r'\\[a-zA-Z]+(\{[^}]*\}|\[[^\]]*\])*', '', body)
        text_content = re.sub(r'[{}\\]', '', text_content)
        words = len(text_content.split())
        if words > 150:
            frame_issues.append(f"Dense content: ~{words} words")

        # Check for equations that might be large
        equations = re.findall(r'\\begin\{(equation|align|gather)\*?\}', body)
        if len(equations) > 2:
            frame_issues.append(f"Multiple equation environments: {len(equations)}")

        # Check columns environment
        if '\\begin{columns}' in body:
            col_widths = re.findall(r'\\begin\{column\}\{([0-9.]+)\\textwidth\}', body)
            total_width = sum(float(w) for w in col_widths)
            if total_width > 1.0:
                frame_issues.append(f"Column widths sum to {total_width} (should be <= 1.0)")

        # Check for vspace or vfill (signs of manual adjustment)
        vspace_count = len(re.findall(r'\\vspace', body))
        if vspace_count > 2:
            frame_issues.append(f"Many vspace commands: {vspace_count}")

        if frame_issues:
            issues.append({
                'frame_num': i,
                'title': title.strip(),
                'issues': frame_issues,
                'body_preview': body[:200].strip()
            })

    return issues, len(frames)


def main():
    print("=" * 70)
    print("BEAMER OVERFLOW DETECTION REPORT")
    print("=" * 70)
    print(f"\nAnalyzing: {TEX_FILE}\n")

    issues, total_frames = analyze_presentation(TEX_FILE)

    if not issues:
        print("No potential overflow issues detected!")
        return

    print(f"Found {len(issues)} frames with potential issues out of {total_frames} total frames:\n")
    print("-" * 70)

    for issue in issues:
        print(f"\nFrame {issue['frame_num']}: {issue['title']}")
        print("-" * 40)
        for prob in issue['issues']:
            print(f"  [!] {prob}")

    print("\n" + "=" * 70)
    print("SUMMARY OF LARGE FIGURES (width > 0.8)")
    print("=" * 70)

    # Re-read to get specific figure info
    with open(TEX_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all large figures with context
    lines = content.split('\n')
    current_frame = "Unknown"
    large_figures = []

    for i, line in enumerate(lines):
        # Track current frame
        frame_match = re.search(r'\\begin\{frame\}(?:\[.*?\])?\{([^}]*)\}', line)
        if frame_match:
            current_frame = frame_match.group(1).strip()

        # Check for large figures
        fig_match = re.search(r'\\includegraphics\[width=([0-9.]+)\\textwidth\]\{([^}]+)\}', line)
        if fig_match:
            width = float(fig_match.group(1))
            filename = fig_match.group(2)
            if width > 0.7:
                large_figures.append({
                    'line': i + 1,
                    'frame': current_frame,
                    'width': width,
                    'file': filename
                })

    if large_figures:
        print(f"\nFound {len(large_figures)} figures with width > 0.7:\n")
        for fig in large_figures:
            print(f"  Line {fig['line']:4d}: width={fig['width']} in '{fig['frame']}'")
            print(f"           File: {fig['file']}")
    else:
        print("\nNo figures with width > 0.7 found.")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
    1. Reduce figure widths to 0.65-0.75 for single figures
    2. Use 0.45-0.48 for side-by-side figures in columns
    3. Limit items per slide to 6-8
    4. Split dense slides into multiple frames
    5. Use smaller font sizes sparingly (\\small, \\footnotesize)
    """)


if __name__ == "__main__":
    main()
