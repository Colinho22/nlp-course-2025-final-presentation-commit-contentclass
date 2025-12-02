#!/usr/bin/env python3
"""
Requirements Verification Script for NLP Course Site
Checks all 5 requirements and reports pass/fail status
"""

import os
import re
from pathlib import Path

def check_requirement_1_content_based_naming(docs_dir):
    """
    Requirement 1: Rename weeks to content-based names
    - NO "Week 01", "Week 02" labels in main titles
    - Use topic names instead
    """
    print("\n" + "="*60)
    print("REQUIREMENT 1: Content-Based Naming")
    print("="*60)

    issues = []

    # Check index.html for "WEEK XX" badges (these are OK in small badges but not main titles)
    index_file = docs_dir / 'index.html'
    if index_file.exists():
        content = index_file.read_text(encoding='utf-8')

        # Check sidebar - should NOT have "01. N-gram" format
        old_pattern = re.findall(r'>\d{2}\.\s+[A-Z]', content)
        if old_pattern:
            issues.append(f"index.html: Found old numbered format in sidebar: {old_pattern[:3]}...")

        # Check topic cards - should show topic names, not "Week 01"
        week_title_pattern = re.findall(r'<span class="topic-title">Week \d+', content)
        if week_title_pattern:
            issues.append(f"index.html: Found 'Week X' in topic titles: {week_title_pattern}")

    # Check week pages - should have content-based titles
    weeks_dir = docs_dir / 'weeks'
    week_files = list(weeks_dir.glob('*.html'))

    content_based_names = ['ngrams', 'embeddings', 'rnn-lstm', 'seq2seq', 'transformers',
                           'pretrained', 'scaling', 'tokenization', 'decoding',
                           'finetuning', 'efficiency', 'ethics']

    found_content_based = []
    found_old_format = []

    for wf in week_files:
        name = wf.stem
        if name in content_based_names:
            found_content_based.append(name)
        elif name.startswith('week'):
            found_old_format.append(name)

    if found_old_format:
        issues.append(f"Found old 'weekXX.html' files: {found_old_format}")

    print(f"  Content-based files found: {len(found_content_based)}/12")
    print(f"  Files: {', '.join(sorted(found_content_based))}")

    if found_old_format:
        print(f"  [FAIL] Old format files still exist: {found_old_format}")

    if not issues:
        print("  [PASS] All naming is content-based")
        return True
    else:
        for issue in issues:
            print(f"  [ISSUE] {issue}")
        return False

def check_requirement_2_link_verification(docs_dir):
    """
    Requirement 2: Python scripts to verify all links work
    """
    print("\n" + "="*60)
    print("REQUIREMENT 2: Link Verification Script")
    print("="*60)

    verify_script = docs_dir / 'verify_links.py'

    if not verify_script.exists():
        print("  [FAIL] verify_links.py does not exist")
        return False

    content = verify_script.read_text(encoding='utf-8')

    # Check for key functions
    has_find_html = 'find_html_files' in content
    has_extract_links = 'extract_links' in content
    has_check_external = 'check_external_url' in content

    print(f"  Script exists: Yes")
    print(f"  Has find_html_files: {has_find_html}")
    print(f"  Has extract_links: {has_extract_links}")
    print(f"  Has check_external_url: {has_check_external}")

    if has_find_html and has_extract_links:
        print("  [PASS] Link verification script is complete")
        return True
    else:
        print("  [FAIL] Link verification script is incomplete")
        return False

def check_requirement_3_consistent_layout(docs_dir):
    """
    Requirement 3: Same layout everywhere on sub pages
    """
    print("\n" + "="*60)
    print("REQUIREMENT 3: Consistent Layout")
    print("="*60)

    # Key CSS patterns that should appear in all subpages
    required_css_patterns = [
        '.top-nav',
        '.hero',
        '.container',
        '.section',
        '.footer'
    ]

    weeks_dir = docs_dir / 'weeks'
    modules_dir = docs_dir / 'modules'

    all_subpages = list(weeks_dir.glob('*.html')) + list(modules_dir.glob('*.html'))

    consistent = True
    for page in all_subpages:
        content = page.read_text(encoding='utf-8')
        missing = []
        for pattern in required_css_patterns:
            if pattern not in content:
                missing.append(pattern)

        if missing:
            print(f"  [ISSUE] {page.name}: Missing {missing}")
            consistent = False

    print(f"  Subpages checked: {len(all_subpages)}")

    if consistent:
        print("  [PASS] All subpages have consistent layout")
        return True
    else:
        print("  [FAIL] Some pages have inconsistent layout")
        return False

def check_requirement_4_top_navigation(docs_dir):
    """
    Requirement 4: Top navigation on all pages
    """
    print("\n" + "="*60)
    print("REQUIREMENT 4: Top Navigation")
    print("="*60)

    nav_links = ['Home', 'Topics', 'Charts', 'Modules', 'GitHub']

    all_html = list(docs_dir.glob('*.html')) + \
               list((docs_dir / 'weeks').glob('*.html')) + \
               list((docs_dir / 'modules').glob('*.html'))

    pages_with_nav = 0
    pages_missing_nav = []

    for page in all_html:
        content = page.read_text(encoding='utf-8')

        has_nav = '<nav class="top-nav">' in content or 'class="top-nav"' in content

        if has_nav:
            pages_with_nav += 1
        else:
            pages_missing_nav.append(page.name)

    print(f"  Total pages: {len(all_html)}")
    print(f"  Pages with top nav: {pages_with_nav}")

    if pages_missing_nav:
        print(f"  [FAIL] Pages missing navigation: {pages_missing_nav}")
        return False
    else:
        print("  [PASS] All pages have top navigation")
        return True

def check_requirement_5_complete_execution(docs_dir):
    """
    Requirement 5: Complete autonomous execution
    """
    print("\n" + "="*60)
    print("REQUIREMENT 5: Complete Execution")
    print("="*60)

    # Check all expected files exist
    expected_files = {
        'index.html': docs_dir / 'index.html',
        'generate_site.py': docs_dir / 'generate_site.py',
        'verify_links.py': docs_dir / 'verify_links.py',
    }

    expected_weeks = ['ngrams', 'embeddings', 'rnn-lstm', 'seq2seq', 'transformers',
                      'pretrained', 'scaling', 'tokenization', 'decoding',
                      'finetuning', 'efficiency', 'ethics']

    expected_modules = ['embeddings', 'summarization', 'sentiment', 'lstm-primer']

    missing = []

    # Check main files
    for name, path in expected_files.items():
        if not path.exists():
            missing.append(name)

    # Check week pages
    for week in expected_weeks:
        path = docs_dir / 'weeks' / f'{week}.html'
        if not path.exists():
            missing.append(f'weeks/{week}.html')

    # Check module pages
    for module in expected_modules:
        path = docs_dir / 'modules' / f'{module}.html'
        if not path.exists():
            missing.append(f'modules/{module}.html')

    print(f"  Expected files: {len(expected_files) + len(expected_weeks) + len(expected_modules)}")
    print(f"  Week pages: {len(expected_weeks)}")
    print(f"  Module pages: {len(expected_modules)}")

    if missing:
        print(f"  [FAIL] Missing files: {missing}")
        return False
    else:
        print("  [PASS] All expected files exist")
        return True

def main():
    docs_dir = Path(__file__).parent

    print("="*60)
    print("NLP COURSE SITE - REQUIREMENTS VERIFICATION")
    print("="*60)

    results = {}

    results['1_content_naming'] = check_requirement_1_content_based_naming(docs_dir)
    results['2_link_verification'] = check_requirement_2_link_verification(docs_dir)
    results['3_consistent_layout'] = check_requirement_3_consistent_layout(docs_dir)
    results['4_top_navigation'] = check_requirement_4_top_navigation(docs_dir)
    results['5_complete_execution'] = check_requirement_5_complete_execution(docs_dir)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for req, status in results.items():
        status_str = "[PASS]" if status else "[FAIL]"
        print(f"  {status_str} {req.replace('_', ' ').title()}")

    print(f"\nTotal: {passed}/{total} requirements passed")

    if passed == total:
        print("\n*** ALL REQUIREMENTS SATISFIED ***")
        return 0
    else:
        print(f"\n*** {total - passed} REQUIREMENTS NEED FIXING ***")
        return 1

if __name__ == '__main__':
    exit(main())
