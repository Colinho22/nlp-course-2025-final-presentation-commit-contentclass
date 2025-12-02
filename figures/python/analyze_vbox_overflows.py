"""
Analyze vbox overflows by line number and identify slides.
"""

import re

TEX_FILE = r"D:\Joerg\Research\slides\NLPFinalLecture\presentations\20251127_1321_final_lecture.tex"

# Overflow lines from pdflatex output
OVERFLOW_LINES = [287, 314, 485, 990, 1074, 1083, 1481, 1939]

def find_frame_for_line(content_lines, target_line):
    """Find which frame contains the given line number."""
    current_frame = None
    frame_start = 0

    for i, line in enumerate(content_lines, 1):
        frame_match = re.search(r'\\begin\{frame\}(?:\[.*?\])?\{([^}]*)\}', line)
        if frame_match:
            current_frame = frame_match.group(1).strip()
            frame_start = i

        if i == target_line:
            return {
                'frame': current_frame,
                'frame_start': frame_start,
                'line': target_line,
                'content': line.strip()[:80]
            }

    return None


def main():
    with open(TEX_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print("=" * 70)
    print("VBOX OVERFLOW ANALYSIS")
    print("=" * 70)
    print(f"\nAnalyzing {len(OVERFLOW_LINES)} overflow locations:\n")

    for overflow_line in OVERFLOW_LINES:
        info = find_frame_for_line(lines, overflow_line)
        if info:
            print(f"Line {overflow_line}: Frame '{info['frame']}'")
            print(f"  Frame starts at line {info['frame_start']}")
            print(f"  Content: {info['content']}")
            print()


if __name__ == "__main__":
    main()
