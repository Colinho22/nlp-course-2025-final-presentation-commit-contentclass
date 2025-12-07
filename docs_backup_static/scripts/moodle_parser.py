"""
Moodle Course Export Parser
Parses Moodle HTML export and extracts structured course data.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import html


def detect_resource_type(folder_name: str) -> str:
    """
    Detect resource type from folder name.

    Args:
        folder_name: Folder name like 'File_Exercise_Ngrams_Intro_.1238443'

    Returns:
        Resource type: 'file', 'assignment', 'page', 'forum', 'external_tool'
    """
    folder_name_lower = folder_name.lower()

    if folder_name_lower.startswith('file_'):
        return 'file'
    elif folder_name_lower.startswith('assignment_'):
        return 'assignment'
    elif folder_name_lower.startswith('page_'):
        return 'page'
    elif folder_name_lower.startswith('forum_'):
        return 'forum'
    elif folder_name_lower.startswith('external_tool_'):
        return 'external_tool'
    else:
        return 'unknown'


def extract_moodle_id(folder_name: str) -> str:
    """
    Extract Moodle ID from folder name.

    Args:
        folder_name: Folder name like 'File_Exercise_Ngrams_Intro_.1238443'

    Returns:
        Moodle ID like '1238443'
    """
    match = re.search(r'\.(\d+)$', folder_name)
    if match:
        return match.group(1)
    return ''


def parse_resource_folder(folder_path: Path) -> Dict:
    """
    Parse a single resource folder to extract metadata and files.

    Args:
        folder_path: Path to resource folder

    Returns:
        Dictionary with resource metadata
    """
    folder_name = folder_path.name
    resource_type = detect_resource_type(folder_name)
    moodle_id = extract_moodle_id(folder_name)

    # Initialize result
    result = {
        'type': resource_type,
        'moodle_id': moodle_id,
        'folder_name': folder_name,
        'title': '',
        'description': '',
        'files': []
    }

    # Parse index.html if exists
    index_path = folder_path / 'index.html'
    if index_path.exists():
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')

            # Extract title from h2
            h2_tag = soup.find('h2')
            if h2_tag:
                result['title'] = html.unescape(h2_tag.get_text(strip=True))

            # Extract description from description div or first p tag
            desc_div = soup.find('div', class_='description')
            if desc_div:
                result['description'] = html.unescape(desc_div.get_text(strip=True))
            elif soup.find('p'):
                result['description'] = html.unescape(soup.find('p').get_text(strip=True))

        except Exception as e:
            print(f"Warning: Could not parse {index_path}: {e}")

    # List files in content/ subfolder
    content_path = folder_path / 'content'
    if content_path.exists() and content_path.is_dir():
        for file_path in content_path.iterdir():
            if file_path.is_file():
                file_info = {
                    'filename': file_path.name,
                    'type': file_path.suffix.lstrip('.').lower(),
                    'path': str(file_path.relative_to(folder_path.parent.parent))
                }
                result['files'].append(file_info)

    return result


def extract_schedule(page_html: str) -> List[Dict]:
    """
    Extract schedule from Semesterbeschreibung page HTML.

    Args:
        page_html: HTML content of the page

    Returns:
        List of schedule items with date and topic
    """
    soup = BeautifulSoup(page_html, 'html.parser')
    schedule = []

    # Look for table with schedule information
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        for row in rows[1:]:  # Skip header
            cols = row.find_all(['td', 'th'])
            if len(cols) >= 2:
                date_text = html.unescape(cols[0].get_text(strip=True))
                topic_text = html.unescape(cols[1].get_text(strip=True))

                if date_text and topic_text:
                    schedule.append({
                        'date': date_text,
                        'topic': topic_text
                    })

    # If no table found, look for list items
    if not schedule:
        list_items = soup.find_all('li')
        for item in list_items:
            text = html.unescape(item.get_text(strip=True))
            # Try to extract date pattern (e.g., "Week 1: Topic")
            match = re.match(r'(Week\s+\d+|Woche\s+\d+|\d{1,2}\.\d{1,2}\.\d{4}):?\s*(.+)', text)
            if match:
                schedule.append({
                    'date': match.group(1),
                    'topic': match.group(2)
                })

    return schedule


def parse_course_index(index_path: Path) -> List[Dict]:
    """
    Parse the main course index.html to extract sections and links.

    Args:
        index_path: Path to course index.html

    Returns:
        List of sections with items
    """
    with open(index_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    sections = []
    current_section = None

    # Find all h3 headers (section titles)
    for element in soup.find_all(['h3', 'a']):
        if element.name == 'h3':
            # New section
            if current_section:
                sections.append(current_section)

            section_title = html.unescape(element.get_text(strip=True))
            section_id = re.sub(r'[^a-z0-9]+', '_', section_title.lower()).strip('_')

            current_section = {
                'id': section_id,
                'title': section_title,
                'items': []
            }

        elif element.name == 'a' and current_section is not None:
            # Link to resource
            href = element.get('href', '')
            link_text = html.unescape(element.get_text(strip=True))

            # Extract folder name from href
            folder_match = re.search(r'([^/]+)/index\.html$', href)
            if folder_match:
                folder_name = folder_match.group(1)
                current_section['items'].append({
                    'folder_name': folder_name,
                    'link_text': link_text
                })

    # Add last section
    if current_section:
        sections.append(current_section)

    return sections


def extract_assignments(course_data: Dict) -> List[Dict]:
    """
    Extract assessment/assignment information from course data.

    Args:
        course_data: Parsed course data

    Returns:
        List of assignment dictionaries
    """
    assignments = []

    for section in course_data.get('sections', []):
        for item in section.get('items', []):
            if item.get('type') == 'assignment':
                assignments.append({
                    'title': item.get('title', ''),
                    'description': item.get('description', ''),
                    'moodle_id': item.get('moodle_id', ''),
                    'section': section.get('title', ''),
                    'files': item.get('files', [])
                })

    return assignments


def parse_moodle_export(source_path: str) -> Dict:
    """
    Main entry point: Parse Moodle course export.

    Args:
        source_path: Path to Moodle export folder

    Returns:
        Dictionary with complete course data
    """
    source_path = Path(source_path)

    # Initialize course data
    course_data = {
        'course_name': 'Text Analytics (dv-) HS25',
        'moodle_url': 'https://moodle.fhgr.ch/course/view.php?id=18956',
        'sections': [],
        'all_pdfs': [],
        'all_notebooks': [],
        'schedule': [],
        'assignments': []
    }

    # Parse main index.html
    index_path = source_path / 'index.html'
    if index_path.exists():
        print(f"Parsing course index: {index_path}")
        sections_structure = parse_course_index(index_path)

        # Parse each resource folder
        for section in sections_structure:
            section_data = {
                'id': section['id'],
                'title': section['title'],
                'items': []
            }

            for item_link in section['items']:
                folder_name = item_link['folder_name']
                folder_path = source_path / folder_name

                if folder_path.exists():
                    resource_data = parse_resource_folder(folder_path)
                    section_data['items'].append(resource_data)

                    # Collect PDFs and notebooks
                    for file_info in resource_data.get('files', []):
                        if file_info['type'] == 'pdf':
                            course_data['all_pdfs'].append({
                                'filename': file_info['filename'],
                                'path': file_info['path'],
                                'title': resource_data['title'],
                                'section': section['title']
                            })
                        elif file_info['type'] in ['ipynb', 'py']:
                            course_data['all_notebooks'].append({
                                'filename': file_info['filename'],
                                'path': file_info['path'],
                                'title': resource_data['title'],
                                'section': section['title']
                            })

                    # Extract schedule if this is the Semesterbeschreibung page
                    if 'semesterbeschreibung' in folder_name.lower():
                        page_html_path = folder_path / 'index.html'
                        if page_html_path.exists():
                            with open(page_html_path, 'r', encoding='utf-8') as f:
                                page_html = f.read()
                                course_data['schedule'] = extract_schedule(page_html)

            course_data['sections'].append(section_data)

    # Extract assignments
    course_data['assignments'] = extract_assignments(course_data)

    return course_data


def main():
    """Run the parser and save results."""
    # Paths
    moodle_export_path = r'D:\Joerg\Research\slides\2025_NLP_16\moodle\TA_(dv-)_HS25_1764742483\Course_Text_Analytics_dv-_HS25_.1182614'
    output_path = r'D:\Joerg\Research\slides\2025_NLP_16\docs_backup_static\scripts\moodle_data.json'

    print("=" * 80)
    print("MOODLE COURSE EXPORT PARSER")
    print("=" * 80)
    print(f"\nSource: {moodle_export_path}")
    print(f"Output: {output_path}\n")

    # Parse the export
    course_data = parse_moodle_export(moodle_export_path)

    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(course_data, f, indent=2, ensure_ascii=False)

    # Report statistics
    print("\n" + "=" * 80)
    print("PARSING RESULTS")
    print("=" * 80)
    print(f"\nSections found: {len(course_data['sections'])}")
    print(f"PDFs found: {len(course_data['all_pdfs'])}")
    print(f"Notebooks found: {len(course_data['all_notebooks'])}")
    print(f"Assignments found: {len(course_data['assignments'])}")
    print(f"Schedule items: {len(course_data['schedule'])}")

    # Show section breakdown
    print("\n" + "-" * 80)
    print("SECTION BREAKDOWN")
    print("-" * 80)
    for section in course_data['sections']:
        print(f"\n{section['title']} ({section['id']})")
        print(f"  Items: {len(section['items'])}")
        for item in section['items'][:3]:  # Show first 3
            print(f"    - [{item['type']}] {item['title']}")
        if len(section['items']) > 3:
            print(f"    ... and {len(section['items']) - 3} more")

    # Show sample PDFs
    if course_data['all_pdfs']:
        print("\n" + "-" * 80)
        print("SAMPLE PDFs (first 5)")
        print("-" * 80)
        for pdf in course_data['all_pdfs'][:5]:
            print(f"  - {pdf['filename']}")
            print(f"    Title: {pdf['title']}")
            print(f"    Section: {pdf['section']}")

    # Show assignments
    if course_data['assignments']:
        print("\n" + "-" * 80)
        print("ASSIGNMENTS")
        print("-" * 80)
        for i, assignment in enumerate(course_data['assignments'], 1):
            print(f"\n{i}. {assignment['title']}")
            print(f"   Section: {assignment['section']}")
            if assignment['description']:
                desc_preview = assignment['description'][:100]
                print(f"   Description: {desc_preview}...")

    # Show schedule
    if course_data['schedule']:
        print("\n" + "-" * 80)
        print("SCHEDULE (first 5 items)")
        print("-" * 80)
        for item in course_data['schedule'][:5]:
            print(f"  {item['date']}: {item['topic']}")

    print("\n" + "=" * 80)
    print(f"Data saved to: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
