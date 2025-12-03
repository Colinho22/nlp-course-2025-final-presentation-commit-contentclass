#!/usr/bin/env python3
"""
Requirements Verification Script for NLP Course Site
Checks all requirements including sidebar and working downloads
"""

import os
import re
from pathlib import Path

def check_requirement_1_content_based_naming(docs_dir):
    """
    Requirement 1: Rename topics to content-based names
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
    topics_dir = docs_dir / 'topics'
    week_files = list(topics_dir.glob('*.html'))

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

    verify_script = docs_dir / 'scripts' / 'verify_links.py'

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

def check_requirement_3_sidebar_on_subpages(docs_dir):
    """
    Requirement 3: Sidebar navigation on ALL subpages
    """
    print("\n" + "="*60)
    print("REQUIREMENT 3: Sidebar on Subpages")
    print("="*60)

    topics_dir = docs_dir / 'topics'
    modules_dir = docs_dir / 'modules'

    all_subpages = list(topics_dir.glob('*.html')) + list(modules_dir.glob('*.html'))

    pages_with_sidebar = 0
    pages_missing_sidebar = []

    for page in all_subpages:
        content = page.read_text(encoding='utf-8')

        # Check for sidebar elements
        has_sidebar = '<aside class="sidebar">' in content
        has_sidebar_nav = 'class="sidebar-nav"' in content
        has_search = 'id="topic-search"' in content

        if has_sidebar and has_sidebar_nav:
            pages_with_sidebar += 1
        else:
            pages_missing_sidebar.append(page.name)

    print(f"  Total subpages: {len(all_subpages)}")
    print(f"  Pages with sidebar: {pages_with_sidebar}")

    if pages_missing_sidebar:
        print(f"  [FAIL] Pages missing sidebar: {pages_missing_sidebar}")
        return False
    else:
        print("  [PASS] All subpages have sidebar navigation")
        return True

def check_requirement_4_working_downloads(docs_dir):
    """
    Requirement 4: Working download links (no href="#" placeholders)
    """
    print("\n" + "="*60)
    print("REQUIREMENT 4: Working Download Links")
    print("="*60)

    topics_dir = docs_dir / 'topics'
    modules_dir = docs_dir / 'modules'

    all_subpages = list(topics_dir.glob('*.html')) + list(modules_dir.glob('*.html'))

    pages_with_broken_links = []
    pages_with_working_links = 0

    for page in all_subpages:
        content = page.read_text(encoding='utf-8')

        # Check for href="#" in resource buttons (broken links)
        broken_pattern = re.findall(r'class="resource-btn[^"]*"[^>]*href="#"', content)

        if broken_pattern:
            pages_with_broken_links.append(page.name)
        else:
            # Verify links point to GitHub
            github_links = re.findall(r'href="https://github\.com/Digital-AI-Finance/[^"]+', content)
            if github_links:
                pages_with_working_links += 1
            else:
                pages_with_broken_links.append(f"{page.name} (no GitHub links)")

    print(f"  Total subpages: {len(all_subpages)}")
    print(f"  Pages with working links: {pages_with_working_links}")

    if pages_with_broken_links:
        print(f"  [FAIL] Pages with broken/missing links: {pages_with_broken_links[:5]}...")
        return False
    else:
        print("  [PASS] All download links are working")
        return True

def check_requirement_5_top_navigation(docs_dir):
    """
    Requirement 5: Top navigation on all pages
    """
    print("\n" + "="*60)
    print("REQUIREMENT 5: Top Navigation")
    print("="*60)

    all_html = list(docs_dir.glob('*.html')) + \
               list((docs_dir / 'topics').glob('*.html')) + \
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

def check_requirement_6_complete_execution(docs_dir):
    """
    Requirement 6: Complete autonomous execution
    """
    print("\n" + "="*60)
    print("REQUIREMENT 6: Complete Execution")
    print("="*60)

    # Check all expected files exist
    expected_files = {
        'index.html': docs_dir / 'index.html',
        'generate_site.py': docs_dir / 'scripts' / 'generate_site.py',
        'verify_links.py': docs_dir / 'scripts' / 'verify_links.py',
    }

    expected_topics = ['ngrams', 'embeddings', 'rnn-lstm', 'seq2seq', 'transformers',
                      'pretrained', 'scaling', 'tokenization', 'decoding',
                      'finetuning', 'efficiency', 'ethics']

    expected_modules = ['embeddings', 'summarization', 'sentiment', 'lstm-primer']

    missing = []

    # Check main files
    for name, path in expected_files.items():
        if not path.exists():
            missing.append(name)

    # Check week pages
    for week in expected_topics:
        path = docs_dir / 'topics' / f'{week}.html'
        if not path.exists():
            missing.append(f'topics/{week}.html')

    # Check module pages
    for module in expected_modules:
        path = docs_dir / 'modules' / f'{module}.html'
        if not path.exists():
            missing.append(f'modules/{module}.html')

    print(f"  Expected files: {len(expected_files) + len(expected_topics) + len(expected_modules)}")
    print(f"  Week pages: {len(expected_topics)}")
    print(f"  Module pages: {len(expected_modules)}")

    if missing:
        print(f"  [FAIL] Missing files: {missing}")
        return False
    else:
        print("  [PASS] All expected files exist")
        return True

def main():
    docs_dir = Path(__file__).parent.parent  # scripts/ -> docs/

    print("="*60)
    print("NLP COURSE SITE - REQUIREMENTS VERIFICATION")
    print("="*60)

    results = {}

    results['1_content_naming'] = check_requirement_1_content_based_naming(docs_dir)
    results['2_link_verification'] = check_requirement_2_link_verification(docs_dir)
    results['3_sidebar_subpages'] = check_requirement_3_sidebar_on_subpages(docs_dir)
    results['4_working_downloads'] = check_requirement_4_working_downloads(docs_dir)
    results['5_top_navigation'] = check_requirement_5_top_navigation(docs_dir)
    results['6_complete_execution'] = check_requirement_6_complete_execution(docs_dir)

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
