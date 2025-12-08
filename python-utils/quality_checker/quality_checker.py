#!/usr/bin/env python3
"""
GitHub Pages Quality Checker

Evaluates website quality by comparing against a reference site.
Produces similarity score + detailed pass/fail checklist.

Usage:
    python quality_checker.py https://target-site.github.io/repo
    python quality_checker.py https://target.io --reference https://ref.io
    python quality_checker.py https://target.io --threshold 80 --output ./reports
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Add parent directory to path for link_checker import
sys.path.insert(0, str(Path(__file__).parent.parent / "link_checker"))
try:
    from link_checker import crawl_and_check
    LINK_CHECKER_AVAILABLE = True
except ImportError:
    LINK_CHECKER_AVAILABLE = False


# Default reference site
DEFAULT_REFERENCE = "https://digital-ai-finance.github.io/neural-networks/"

# Category weights
WEIGHTS = {
    "structure": 0.30,
    "visual": 0.30,
    "technical": 0.20,
    "links": 0.20
}


@dataclass
class CheckResult:
    """Result of a single quality check."""
    name: str
    passed: bool
    score: float  # 0.0 - 1.0
    details: str
    category: str  # structure/visual/technical/links


@dataclass
class PageReport:
    """Quality report for a single page."""
    url: str
    score: float
    category_scores: dict
    passed: bool
    checks: list = field(default_factory=list)


@dataclass
class QualityReport:
    """Complete quality report for a site (potentially multiple pages)."""
    url: str
    reference_url: str
    timestamp: str
    overall_score: float
    category_scores: dict
    passed: bool
    threshold: float
    similarity_score: float = 0.0  # Similarity to reference site
    checks: list = field(default_factory=list)  # Main page checks (backward compat)
    reference_checks: list = field(default_factory=list)
    # Multi-page support
    page_reports: list = field(default_factory=list)  # List of PageReport
    worst_page_url: str = ""
    pages_checked: int = 1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Check GitHub Pages site quality against a reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python quality_checker.py https://site.github.io/repo
    python quality_checker.py https://site.io --reference https://ref.io
    python quality_checker.py https://site.io --threshold 80 --output ./reports
        """
    )
    parser.add_argument("url", help="Target URL to check")
    parser.add_argument(
        "--reference", "-r",
        default=DEFAULT_REFERENCE,
        help=f"Reference URL to compare against (default: {DEFAULT_REFERENCE})"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=70.0,
        help="Minimum passing score percentage (default: 70)"
    )
    parser.add_argument(
        "--output", "-o",
        default=".",
        help="Output directory for reports (default: current directory)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--skip-links",
        action="store_true",
        help="Skip link checking (faster but incomplete)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Crawl depth for subpages (default: 1, main page only)"
    )
    return parser.parse_args()


def fetch_page(url: str, timeout: int = 30) -> tuple[Optional[str], Optional[requests.Response], float]:
    """
    Fetch a page and return HTML, response, and load time.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) QualityChecker/1.0"
    }
    try:
        start = time.time()
        response = requests.get(url, headers=headers, timeout=timeout)
        load_time = time.time() - start
        response.raise_for_status()
        return response.text, response, load_time
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None, None, 0.0


def extract_css(soup: BeautifulSoup, base_url: str, timeout: int = 10) -> str:
    """Extract all CSS from page (inline + external stylesheets)."""
    css_content = []

    # Ensure base_url has trailing slash for proper urljoin behavior
    base_url = base_url.rstrip('/') + '/'

    # Inline styles
    for style in soup.find_all("style"):
        if style.string:
            css_content.append(style.string)

    # External stylesheets
    for link in soup.find_all("link", rel="stylesheet"):
        href = link.get("href")
        if href:
            css_url = urljoin(base_url, href)
            try:
                resp = requests.get(css_url, timeout=timeout)
                if resp.status_code == 200:
                    css_content.append(resp.text)
            except Exception:
                pass

    return "\n".join(css_content)


def discover_subpages(start_url: str, max_depth: int, timeout: int) -> list[str]:
    """Discover all internal HTML pages up to max_depth.

    Uses BFS to crawl internal links, returning unique page URLs.
    """
    from collections import deque

    base_domain = urlparse(start_url).netloc
    base_path = urlparse(start_url).path.rstrip("/")

    # Queue: (url, depth)
    queue = deque([(start_url, 0)])
    visited = set()
    pages = []

    # Extensions to skip
    skip_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg",
                      ".css", ".js", ".zip", ".tar", ".gz", ".mp4",
                      ".mp3", ".wav", ".webp", ".ico", ".woff", ".woff2",
                      ".xml", ".json", ".txt"}

    print(f"\nDiscovering subpages (depth {max_depth})...")

    while queue:
        current_url, depth = queue.popleft()

        # Normalize URL
        current_url = current_url.rstrip("/")
        if current_url in visited:
            continue

        visited.add(current_url)

        # Check if internal
        parsed = urlparse(current_url)
        if parsed.netloc != base_domain:
            continue

        # Check if it's under the base path (for repo-specific sites)
        if base_path and not parsed.path.startswith(base_path):
            continue

        # Skip non-HTML resources
        path_lower = parsed.path.lower()
        if any(path_lower.endswith(ext) for ext in skip_extensions):
            continue

        # Fetch page
        html, _, _ = fetch_page(current_url, timeout)
        if not html:
            continue

        pages.append(current_url)
        print(f"  [{len(pages)}] Depth {depth}: {current_url}")

        # Don't crawl deeper than max_depth
        if depth >= max_depth:
            continue

        # Extract links for next level
        soup = BeautifulSoup(html, "html.parser")
        for link in soup.find_all("a", href=True):
            href = link["href"]

            # Skip anchors, mailto, javascript
            if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
                continue

            # Make absolute
            absolute_url = urljoin(current_url, href).split("#")[0].rstrip("/")

            if absolute_url not in visited:
                queue.append((absolute_url, depth + 1))

    print(f"  Found {len(pages)} pages total")
    return pages


# =============================================================================
# STRUCTURE CHECKS (30%)
# =============================================================================

def check_navigation(soup: BeautifulSoup) -> CheckResult:
    """Check for top navigation bar."""
    nav = soup.find("nav")
    header = soup.find("header")
    header_links = header.find_all("a") if header else []

    has_nav = nav is not None or len(header_links) >= 3
    nav_links = len(nav.find_all("a")) if nav else len(header_links)

    return CheckResult(
        name="Top Navigation",
        passed=has_nav,
        score=1.0 if has_nav else 0.0,
        details=f"Found nav with {nav_links} links" if has_nav else "No navigation found",
        category="structure"
    )


def check_sidebar(soup: BeautifulSoup) -> CheckResult:
    """Check for sidebar navigation."""
    # Look for aside, or divs with sidebar-related classes
    aside = soup.find("aside")
    sidebar_div = soup.find(class_=re.compile(r"sidebar|side-nav|sidenav", re.I))

    has_sidebar = aside is not None or sidebar_div is not None
    element = aside or sidebar_div

    if has_sidebar and element:
        links = len(element.find_all("a"))
        return CheckResult(
            name="Sidebar",
            passed=True,
            score=1.0,
            details=f"Sidebar found with {links} links",
            category="structure"
        )

    return CheckResult(
        name="Sidebar",
        passed=False,
        score=0.0,
        details="No sidebar found",
        category="structure"
    )


def check_hero_section(soup: BeautifulSoup) -> CheckResult:
    """Check for hero/banner section."""
    hero = soup.find(class_=re.compile(r"hero|banner|jumbotron|masthead", re.I))
    # Also check for large heading at top
    first_h1 = soup.find("h1")

    has_hero = hero is not None or first_h1 is not None

    if hero:
        return CheckResult(
            name="Hero Section",
            passed=True,
            score=1.0,
            details="Hero/banner section found",
            category="structure"
        )
    elif first_h1:
        return CheckResult(
            name="Hero Section",
            passed=True,
            score=0.7,
            details="H1 heading found (partial hero)",
            category="structure"
        )

    return CheckResult(
        name="Hero Section",
        passed=False,
        score=0.0,
        details="No hero section found",
        category="structure"
    )


def check_content_grids(soup: BeautifulSoup) -> CheckResult:
    """Check for content grids/card layouts."""
    # Look for grid/flex containers
    grid_div = soup.find(class_=re.compile(r"grid|cards|card-container|topics|modules", re.I))
    # Count card-like elements
    cards = soup.find_all(class_=re.compile(r"card|topic|module|item", re.I))

    has_grid = grid_div is not None or len(cards) >= 3

    return CheckResult(
        name="Content Grids",
        passed=has_grid,
        score=1.0 if has_grid else 0.0,
        details=f"Found {len(cards)} card elements" if has_grid else "No content grids found",
        category="structure"
    )


def check_footer(soup: BeautifulSoup) -> CheckResult:
    """Check for footer."""
    footer = soup.find("footer")

    if footer:
        links = len(footer.find_all("a"))
        text_len = len(footer.get_text(strip=True))
        return CheckResult(
            name="Footer",
            passed=True,
            score=1.0,
            details=f"Footer found with {links} links, {text_len} chars",
            category="structure"
        )

    return CheckResult(
        name="Footer",
        passed=False,
        score=0.0,
        details="No footer found",
        category="structure"
    )


def check_search(soup: BeautifulSoup) -> CheckResult:
    """Check for search functionality."""
    search_input = soup.find("input", {"type": "search"})
    search_form = soup.find("form", class_=re.compile(r"search", re.I))
    search_div = soup.find(class_=re.compile(r"search", re.I))

    has_search = search_input or search_form or search_div

    return CheckResult(
        name="Search Functionality",
        passed=has_search is not None,
        score=1.0 if has_search else 0.0,
        details="Search found" if has_search else "No search functionality",
        category="structure"
    )


def check_collapsible(soup: BeautifulSoup) -> CheckResult:
    """Check for collapsible/accordion sections."""
    details_elements = soup.find_all("details")
    accordion = soup.find(class_=re.compile(r"accordion|collapse|expandable", re.I))

    has_collapsible = len(details_elements) > 0 or accordion is not None

    return CheckResult(
        name="Collapsible Sections",
        passed=has_collapsible,
        score=1.0 if has_collapsible else 0.0,
        details=f"Found {len(details_elements)} details elements" if details_elements else (
            "Accordion found" if accordion else "No collapsible sections"
        ),
        category="structure"
    )


def check_journey_tracker(soup: BeautifulSoup) -> CheckResult:
    """Check for journey/progress tracker."""
    progress = soup.find("progress")
    steps = soup.find(class_=re.compile(r"step|journey|progress|timeline|tracker", re.I))
    # Look for numbered steps
    numbered_items = soup.find_all(class_=re.compile(r"step-\d|part-\d", re.I))

    has_tracker = progress is not None or steps is not None or len(numbered_items) >= 2

    return CheckResult(
        name="Journey/Progress Tracker",
        passed=has_tracker,
        score=1.0 if has_tracker else 0.0,
        details="Progress tracker found" if has_tracker else "No progress tracker",
        category="structure"
    )


def check_structure(soup: BeautifulSoup) -> list[CheckResult]:
    """Run all structure checks."""
    return [
        check_navigation(soup),
        check_sidebar(soup),
        check_hero_section(soup),
        check_content_grids(soup),
        check_footer(soup),
        check_search(soup),
        check_collapsible(soup),
        check_journey_tracker(soup),
    ]


# =============================================================================
# VISUAL CHECKS (30%)
# =============================================================================

def check_stylesheet(soup: BeautifulSoup) -> CheckResult:
    """Check for stylesheets."""
    external_css = soup.find_all("link", rel="stylesheet")
    inline_css = soup.find_all("style")

    total = len(external_css) + len(inline_css)
    has_css = total > 0

    return CheckResult(
        name="Has Stylesheet",
        passed=has_css,
        score=1.0 if has_css else 0.0,
        details=f"{len(external_css)} external, {len(inline_css)} inline stylesheets",
        category="visual"
    )


def check_responsive_meta(soup: BeautifulSoup) -> CheckResult:
    """Check for responsive viewport meta tag."""
    viewport = soup.find("meta", {"name": "viewport"})

    return CheckResult(
        name="Responsive Meta",
        passed=viewport is not None,
        score=1.0 if viewport else 0.0,
        details="Viewport meta found" if viewport else "No viewport meta tag",
        category="visual"
    )


def check_media_queries(css: str) -> CheckResult:
    """Check for responsive media queries."""
    media_pattern = r"@media\s*\([^)]+\)"
    matches = re.findall(media_pattern, css)

    has_media = len(matches) > 0
    # Score based on number of breakpoints
    score = min(1.0, len(matches) / 3)  # 3+ breakpoints = full score

    return CheckResult(
        name="Media Queries",
        passed=has_media,
        score=score if has_media else 0.0,
        details=f"Found {len(matches)} media queries" if has_media else "No media queries",
        category="visual"
    )


def check_images(soup: BeautifulSoup) -> CheckResult:
    """Check for images."""
    images = soup.find_all("img")
    with_src = [img for img in images if img.get("src")]

    has_images = len(with_src) >= 3
    score = min(1.0, len(with_src) / 10)  # 10+ images = full score

    return CheckResult(
        name="Images Present",
        passed=has_images,
        score=score,
        details=f"Found {len(with_src)} images",
        category="visual"
    )


def check_chart_gallery(soup: BeautifulSoup) -> CheckResult:
    """Check for chart/image gallery."""
    gallery = soup.find(class_=re.compile(r"gallery|chart|visualization|figures", re.I))
    # Look for grid of images
    img_grids = soup.find_all(class_=re.compile(r"grid|flex", re.I))
    img_count_in_grids = sum(len(g.find_all("img")) for g in img_grids)

    has_gallery = gallery is not None or img_count_in_grids >= 4

    return CheckResult(
        name="Chart/Image Gallery",
        passed=has_gallery,
        score=1.0 if has_gallery else 0.0,
        details=f"Gallery found with {img_count_in_grids} images in grids" if has_gallery else "No gallery found",
        category="visual"
    )


def check_gradients(css: str) -> CheckResult:
    """Check for gradient usage."""
    gradient_pattern = r"(linear-gradient|radial-gradient)\s*\("
    matches = re.findall(gradient_pattern, css)

    has_gradients = len(matches) > 0

    return CheckResult(
        name="Gradients",
        passed=has_gradients,
        score=1.0 if has_gradients else 0.0,
        details=f"Found {len(matches)} gradients" if has_gradients else "No gradients",
        category="visual"
    )


def check_shadows(css: str) -> CheckResult:
    """Check for box shadows."""
    shadow_pattern = r"box-shadow\s*:"
    matches = re.findall(shadow_pattern, css)

    has_shadows = len(matches) > 0

    return CheckResult(
        name="Box Shadows",
        passed=has_shadows,
        score=1.0 if has_shadows else 0.0,
        details=f"Found {len(matches)} shadow declarations" if has_shadows else "No shadows",
        category="visual"
    )


def check_transitions(css: str) -> CheckResult:
    """Check for CSS transitions."""
    transition_pattern = r"transition\s*:"
    matches = re.findall(transition_pattern, css)

    has_transitions = len(matches) > 0

    return CheckResult(
        name="Transitions",
        passed=has_transitions,
        score=1.0 if has_transitions else 0.0,
        details=f"Found {len(matches)} transitions" if has_transitions else "No transitions",
        category="visual"
    )


def check_hover_effects(css: str) -> CheckResult:
    """Check for hover effects."""
    hover_pattern = r":hover\s*\{"
    matches = re.findall(hover_pattern, css)

    has_hover = len(matches) > 0

    return CheckResult(
        name="Hover Effects",
        passed=has_hover,
        score=1.0 if has_hover else 0.0,
        details=f"Found {len(matches)} hover rules" if has_hover else "No hover effects",
        category="visual"
    )


def check_color_palette(css: str) -> CheckResult:
    """Check for consistent color usage."""
    # Extract all color formats
    hex_colors = set(re.findall(r"#[0-9a-fA-F]{3,6}\b", css))
    rgb_colors = set(re.findall(r"rgba?\s*\([^)]+\)", css))
    hsl_colors = set(re.findall(r"hsla?\s*\([^)]+\)", css))
    css_vars = set(re.findall(r"var\s*\(\s*--[a-zA-Z0-9-]+", css))

    total_colors = len(hex_colors) + len(rgb_colors) + len(hsl_colors)
    has_css_vars = len(css_vars) > 0

    # Having a reasonable palette (not too few, not too many unique colors)
    if 5 <= total_colors <= 25 or has_css_vars:
        score = 1.0
        status = "Good color palette"
    elif 3 <= total_colors < 5 or 25 < total_colors <= 35:
        score = 0.7
        status = "Acceptable color palette"
    else:
        score = 0.4
        status = "Color palette needs work"

    details = f"{len(hex_colors)} hex, {len(rgb_colors)} rgb, {len(hsl_colors)} hsl"
    if has_css_vars:
        details += f", {len(css_vars)} CSS vars"
    details += f" - {status}"

    return CheckResult(
        name="Color Palette",
        passed=total_colors >= 3 or has_css_vars,
        score=score,
        details=details,
        category="visual"
    )


def check_visual(soup: BeautifulSoup, css: str) -> list[CheckResult]:
    """Run all visual checks."""
    return [
        check_stylesheet(soup),
        check_responsive_meta(soup),
        check_media_queries(css),
        check_images(soup),
        check_chart_gallery(soup),
        check_gradients(css),
        check_shadows(css),
        check_transitions(css),
        check_hover_effects(css),
        check_color_palette(css),
    ]


# =============================================================================
# TECHNICAL CHECKS (20%)
# =============================================================================

def check_html_structure(soup: BeautifulSoup, html: str) -> CheckResult:
    """Check basic HTML structure."""
    has_doctype = html.strip().lower().startswith("<!doctype")
    has_html = soup.find("html") is not None
    has_head = soup.find("head") is not None
    has_body = soup.find("body") is not None

    score = sum([has_doctype, has_html, has_head, has_body]) / 4
    passed = score >= 0.75

    missing = []
    if not has_doctype:
        missing.append("doctype")
    if not has_html:
        missing.append("html")
    if not has_head:
        missing.append("head")
    if not has_body:
        missing.append("body")

    return CheckResult(
        name="HTML Structure",
        passed=passed,
        score=score,
        details="Valid structure" if passed else f"Missing: {', '.join(missing)}",
        category="technical"
    )


def check_semantic_html(soup: BeautifulSoup) -> CheckResult:
    """Check for semantic HTML5 elements."""
    semantic_tags = ["nav", "main", "article", "section", "aside", "header", "footer"]
    found = []

    for tag in semantic_tags:
        if soup.find(tag):
            found.append(tag)

    score = len(found) / len(semantic_tags)
    passed = score >= 0.5

    return CheckResult(
        name="Semantic HTML",
        passed=passed,
        score=score,
        details=f"Found {len(found)}/{len(semantic_tags)} semantic tags: {', '.join(found)}",
        category="technical"
    )


def check_alt_texts(soup: BeautifulSoup) -> CheckResult:
    """Check images have alt attributes."""
    images = soup.find_all("img")
    if not images:
        return CheckResult(
            name="Alt Texts",
            passed=True,
            score=1.0,
            details="No images to check",
            category="technical"
        )

    with_alt = [img for img in images if img.get("alt") is not None]
    score = len(with_alt) / len(images)
    passed = score >= 0.8

    return CheckResult(
        name="Alt Texts",
        passed=passed,
        score=score,
        details=f"{len(with_alt)}/{len(images)} images have alt text ({score*100:.0f}%)",
        category="technical"
    )


def check_aria_labels_contextual(soup: BeautifulSoup, semantic_score: float) -> CheckResult:
    """Check for ARIA labels, with partial credit if semantic HTML is good.

    If a site uses good semantic HTML, ARIA is less critical.
    """
    aria_elements = soup.find_all(attrs={"aria-label": True})
    aria_roles = soup.find_all(attrs={"role": True})

    total = len(aria_elements) + len(aria_roles)
    has_aria = total > 0

    if has_aria:
        # Has ARIA attributes
        score = min(1.0, total / 5)
        details = f"Found {len(aria_elements)} aria-labels, {len(aria_roles)} roles"
        passed = True
    elif semantic_score >= 0.7:
        # No ARIA but good semantic HTML - partial credit
        score = 0.7
        details = "No ARIA, but semantic HTML compensates"
        passed = True
    else:
        # No ARIA and poor semantic HTML
        score = 0.0
        details = "No ARIA labels and weak semantic HTML"
        passed = False

    return CheckResult(
        name="ARIA/Accessibility",
        passed=passed,
        score=score,
        details=details,
        category="technical"
    )


def check_heading_hierarchy(soup: BeautifulSoup) -> CheckResult:
    """Check heading hierarchy (h1 -> h2 -> h3 etc)."""
    headings = []
    for i in range(1, 7):
        for h in soup.find_all(f"h{i}"):
            headings.append(i)

    if not headings:
        return CheckResult(
            name="Heading Hierarchy",
            passed=False,
            score=0.0,
            details="No headings found",
            category="technical"
        )

    # Check if headings follow proper hierarchy
    violations = 0
    for i in range(1, len(headings)):
        if headings[i] > headings[i - 1] + 1:
            violations += 1

    has_h1 = 1 in headings
    h1_count = headings.count(1)

    score = 1.0
    issues = []

    if not has_h1:
        score -= 0.3
        issues.append("no h1")
    if h1_count > 1:
        score -= 0.2
        issues.append(f"{h1_count} h1s")
    if violations > 0:
        score -= min(0.3, violations * 0.1)
        issues.append(f"{violations} skips")

    score = max(0, score)
    passed = score >= 0.7

    return CheckResult(
        name="Heading Hierarchy",
        passed=passed,
        score=score,
        details=f"Issues: {', '.join(issues)}" if issues else "Good heading structure",
        category="technical"
    )


def check_load_time(load_time: float) -> CheckResult:
    """Check page load time."""
    if load_time < 1.0:
        score = 1.0
        status = "Excellent"
    elif load_time < 2.0:
        score = 0.8
        status = "Good"
    elif load_time < 3.0:
        score = 0.6
        status = "Acceptable"
    elif load_time < 5.0:
        score = 0.4
        status = "Slow"
    else:
        score = 0.2
        status = "Very slow"

    return CheckResult(
        name="Page Load Time",
        passed=score >= 0.6,
        score=score,
        details=f"{load_time:.2f}s - {status}",
        category="technical"
    )


def check_assets(soup: BeautifulSoup) -> CheckResult:
    """Check number of render-blocking assets (scripts + CSS only, not images)."""
    scripts = soup.find_all("script", src=True)
    stylesheets = soup.find_all("link", rel="stylesheet")
    images = soup.find_all("img", src=True)

    # Only scripts and CSS affect performance, images are content
    blocking_assets = len(scripts) + len(stylesheets)

    if blocking_assets <= 5:
        score = 1.0
        status = "Excellent"
    elif blocking_assets <= 10:
        score = 0.8
        status = "Good"
    elif blocking_assets <= 20:
        score = 0.6
        status = "Moderate"
    else:
        score = 0.4
        status = "Heavy"

    return CheckResult(
        name="Render-Blocking Assets",
        passed=score >= 0.6,
        score=score,
        details=f"{len(scripts)} scripts, {len(stylesheets)} CSS ({blocking_assets} blocking), {len(images)} images - {status}",
        category="technical"
    )


def check_meta_tags(soup: BeautifulSoup) -> CheckResult:
    """Check for essential meta tags."""
    title = soup.find("title")
    description = soup.find("meta", {"name": "description"})
    charset = soup.find("meta", {"charset": True}) or soup.find("meta", {"http-equiv": "Content-Type"})

    found = sum([title is not None, description is not None, charset is not None])
    score = found / 3

    missing = []
    if not title:
        missing.append("title")
    if not description:
        missing.append("description")
    if not charset:
        missing.append("charset")

    return CheckResult(
        name="Meta Tags",
        passed=score >= 0.66,
        score=score,
        details=f"Found {found}/3 essential meta tags" + (f" - missing: {', '.join(missing)}" if missing else ""),
        category="technical"
    )


def check_favicon(soup: BeautifulSoup) -> CheckResult:
    """Check for favicon."""
    # Multiple ways to specify favicon
    icon_link = soup.find("link", rel=re.compile(r"icon", re.I))
    shortcut = soup.find("link", rel="shortcut icon")

    has_favicon = icon_link is not None or shortcut is not None

    return CheckResult(
        name="Favicon",
        passed=has_favicon,
        score=1.0 if has_favicon else 0.0,
        details="Favicon found" if has_favicon else "No favicon",
        category="technical"
    )


def check_lang_attribute(soup: BeautifulSoup) -> CheckResult:
    """Check for lang attribute on html tag."""
    html_tag = soup.find("html")
    has_lang = html_tag and html_tag.get("lang")

    lang_value = html_tag.get("lang", "") if html_tag else ""

    return CheckResult(
        name="Lang Attribute",
        passed=has_lang is not None and has_lang != "",
        score=1.0 if has_lang else 0.0,
        details=f"lang=\"{lang_value}\"" if has_lang else "No lang attribute on html",
        category="technical"
    )


def check_open_graph(soup: BeautifulSoup) -> CheckResult:
    """Check for Open Graph meta tags."""
    og_title = soup.find("meta", property="og:title")
    og_description = soup.find("meta", property="og:description")
    og_image = soup.find("meta", property="og:image")

    found = sum([og_title is not None, og_description is not None, og_image is not None])
    score = found / 3

    missing = []
    if not og_title:
        missing.append("og:title")
    if not og_description:
        missing.append("og:description")
    if not og_image:
        missing.append("og:image")

    return CheckResult(
        name="Open Graph Tags",
        passed=found >= 2,
        score=score,
        details=f"Found {found}/3 OG tags" + (f" - missing: {', '.join(missing)}" if missing else ""),
        category="technical"
    )


def check_technical(soup: BeautifulSoup, html: str, load_time: float) -> list[CheckResult]:
    """Run all technical checks."""
    # Get semantic HTML score first for ARIA context
    semantic_result = check_semantic_html(soup)

    return [
        check_html_structure(soup, html),
        semantic_result,
        check_alt_texts(soup),
        check_aria_labels_contextual(soup, semantic_result.score),
        check_heading_hierarchy(soup),
        check_load_time(load_time),
        check_assets(soup),
        check_meta_tags(soup),
        check_favicon(soup),
        check_lang_attribute(soup),
        check_open_graph(soup),
    ]


# =============================================================================
# LINK CHECKS (20%)
# =============================================================================

def check_links_health(url: str, timeout: int = 10) -> list[CheckResult]:
    """Check link health using link_checker."""
    if not LINK_CHECKER_AVAILABLE:
        return [CheckResult(
            name="Link Health",
            passed=True,
            score=0.5,
            details="link_checker not available - skipped",
            category="links"
        )]

    try:
        print("\nChecking links (this may take a moment)...")
        results = crawl_and_check(url, max_depth=1, timeout=timeout, num_workers=5)

        total = len(results)
        if total == 0:
            return [CheckResult(
                name="Link Health",
                passed=True,
                score=1.0,
                details="No links found to check",
                category="links"
            )]

        broken = sum(1 for r in results if r.status_code is None or r.status_code >= 400)
        working = total - broken
        score = working / total

        return [CheckResult(
            name="Link Health",
            passed=score >= 0.9,
            score=score,
            details=f"{working}/{total} links working ({score*100:.1f}%)",
            category="links"
        )]

    except Exception as e:
        return [CheckResult(
            name="Link Health",
            passed=False,
            score=0.0,
            details=f"Error checking links: {str(e)[:100]}",
            category="links"
        )]


# =============================================================================
# REFERENCE COMPARISON
# =============================================================================

def analyze_site(url: str, timeout: int, skip_links: bool) -> tuple[list[CheckResult], float]:
    """Analyze a site and return checks and load time."""
    html, response, load_time = fetch_page(url, timeout)
    if not html:
        return [], 0.0

    soup = BeautifulSoup(html, "html.parser")
    css = extract_css(soup, url, timeout)

    all_checks = []
    all_checks.extend(check_structure(soup))
    all_checks.extend(check_visual(soup, css))
    all_checks.extend(check_technical(soup, html, load_time))

    if not skip_links:
        all_checks.extend(check_links_health(url, timeout))

    return all_checks, load_time


def calculate_similarity(target_checks: list[CheckResult], ref_checks: list[CheckResult]) -> float:
    """Calculate similarity score between target and reference site.

    Compares each check that exists in both and calculates how similar the scores are.
    Returns 0-100 percentage.
    """
    if not target_checks or not ref_checks:
        return 0.0

    # Create lookup by check name
    ref_lookup = {c.name: c.score for c in ref_checks}

    matches = 0
    total = 0

    for check in target_checks:
        if check.name in ref_lookup:
            ref_score = ref_lookup[check.name]
            # Similarity is based on how close the scores are
            # Both passed (both > 0.5): high similarity
            # Both failed (both <= 0.5): medium similarity
            # One passed, one failed: low similarity
            if check.score >= 0.5 and ref_score >= 0.5:
                matches += 1  # Both pass
            elif check.score < 0.5 and ref_score < 0.5:
                matches += 0.5  # Both fail (partial credit)
            else:
                matches += 0  # Mismatch

            total += 1

    if total == 0:
        return 0.0

    return round((matches / total) * 100, 1)


# =============================================================================
# SCORING AND REPORTING
# =============================================================================

def calculate_scores(checks: list[CheckResult], threshold: float, skip_links: bool = False) -> tuple[float, dict, bool]:
    """Calculate overall and category scores.

    If skip_links is True, excludes links category and redistributes weight.
    """
    category_scores = {}

    # Determine active weights (exclude links if skipped)
    if skip_links:
        active_weights = {k: v for k, v in WEIGHTS.items() if k != "links"}
        # Redistribute link weight proportionally
        total_weight = sum(active_weights.values())
        active_weights = {k: v / total_weight for k, v in active_weights.items()}
    else:
        active_weights = WEIGHTS.copy()

    for category in WEIGHTS.keys():
        cat_checks = [c for c in checks if c.category == category]
        if cat_checks:
            avg = sum(c.score for c in cat_checks) / len(cat_checks)
            category_scores[category] = round(avg * 100, 1)
        else:
            category_scores[category] = 0.0

    # Calculate overall using active weights only
    overall = sum(category_scores[c] * active_weights.get(c, 0) for c in WEIGHTS)
    passed = overall >= threshold

    return round(overall, 1), category_scores, passed


def print_report(report: QualityReport):
    """Print quality report to console."""
    print("\n" + "=" * 70)
    print("QUALITY CHECK REPORT")
    print("=" * 70)
    print(f"Target URL: {report.url}")
    print(f"Reference:  {report.reference_url}")
    print(f"Timestamp:  {report.timestamp}")
    print(f"Threshold:  {report.threshold}%")
    print(f"Pages:      {report.pages_checked}")

    # Show per-page summary if multiple pages
    if report.pages_checked > 1:
        print("\n" + "-" * 70)
        print("PAGE SCORES (sorted by score)")
        print("-" * 70)
        sorted_pages = sorted(report.page_reports, key=lambda p: p.score)
        for page in sorted_pages:
            status = "[PASS]" if page.passed else "[FAIL]"
            bar = "#" * int(page.score / 5) + "-" * (20 - int(page.score / 5))
            # Truncate URL for display
            display_url = page.url if len(page.url) <= 50 else "..." + page.url[-47:]
            print(f"  {status} [{bar}] {page.score:5.1f}%  {display_url}")

        print(f"\n  Worst page: {report.worst_page_url}")

    print("\n" + "-" * 70)
    print("CATEGORY SCORES (main page)")
    print("-" * 70)
    for category, score in report.category_scores.items():
        weight = int(WEIGHTS[category] * 100)
        bar = "#" * int(score / 5) + "-" * (20 - int(score / 5))
        print(f"  {category.upper():12} ({weight}%): [{bar}] {score:.1f}%")

    print("\n" + "-" * 70)
    print("DETAILED CHECKS (main page)")
    print("-" * 70)

    for category in WEIGHTS.keys():
        cat_checks = [c for c in report.checks if c.category == category]
        if cat_checks:
            print(f"\n  {category.upper()}:")
            for check in cat_checks:
                status = "[PASS]" if check.passed else "[FAIL]"
                print(f"    {status} {check.name}: {check.details}")

    # Show details for worst page if different from main
    if report.pages_checked > 1 and report.worst_page_url != report.url:
        worst = next((p for p in report.page_reports if p.url == report.worst_page_url), None)
        if worst:
            print("\n" + "-" * 70)
            print(f"DETAILED CHECKS (worst page: {worst.url})")
            print("-" * 70)
            for category in WEIGHTS.keys():
                cat_checks = [c for c in worst.checks if c.category == category]
                if cat_checks:
                    print(f"\n  {category.upper()}:")
                    for check in cat_checks:
                        status = "[PASS]" if check.passed else "[FAIL]"
                        print(f"    {status} {check.name}: {check.details}")

    print("\n" + "=" * 70)
    status = "PASSED" if report.passed else "FAILED"
    if report.pages_checked > 1:
        print(f"OVERALL SCORE: {report.overall_score:.1f}% - {status} (worst of {report.pages_checked} pages)")
    else:
        print(f"OVERALL SCORE: {report.overall_score:.1f}% - {status}")

    # Show similarity score if comparing to different reference
    if report.url.rstrip("/") != report.reference_url.rstrip("/"):
        sim_bar = "#" * int(report.similarity_score / 5) + "-" * (20 - int(report.similarity_score / 5))
        print(f"SIMILARITY:    [{sim_bar}] {report.similarity_score:.1f}%")

    print("=" * 70)


def export_json(report: QualityReport, output_dir: str) -> str:
    """Export report to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = output_path / f"quality_report_{timestamp}.json"

    # Convert page reports to dicts
    page_reports_data = []
    for page in report.page_reports:
        page_reports_data.append({
            "url": page.url,
            "score": page.score,
            "category_scores": page.category_scores,
            "passed": page.passed,
            "checks": [asdict(c) for c in page.checks]
        })

    data = {
        "url": report.url,
        "reference_url": report.reference_url,
        "timestamp": report.timestamp,
        "threshold": report.threshold,
        "overall_score": report.overall_score,
        "similarity_score": report.similarity_score,
        "passed": report.passed,
        "category_scores": report.category_scores,
        "pages_checked": report.pages_checked,
        "worst_page_url": report.worst_page_url,
        "checks": [asdict(c) for c in report.checks],
        "page_reports": page_reports_data
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return str(filename)


def export_html(report: QualityReport, output_dir: str) -> str:
    """Export report to HTML file with multi-page support."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = output_path / f"quality_report_{timestamp}.html"

    status_color = "#28a745" if report.passed else "#dc3545"
    status_text = "PASSED" if report.passed else "FAILED"

    # Pages summary table (if multiple pages)
    pages_html = ""
    if report.pages_checked > 1:
        sorted_pages = sorted(report.page_reports, key=lambda p: p.score)
        pages_rows = ""
        for page in sorted_pages:
            is_worst = page.url == report.worst_page_url
            is_main = page.url == report.url
            bg = "#f8d7da" if not page.passed else ("#fff3cd" if is_worst else "#d4edda")
            label = ""
            if is_worst and is_main:
                label = " (main, worst)"
            elif is_worst:
                label = " (worst)"
            elif is_main:
                label = " (main)"
            pages_rows += f"<tr style='background:{bg}'><td><a href='{page.url}'>{page.url}</a>{label}</td><td>{'PASS' if page.passed else 'FAIL'}</td><td>{page.score:.1f}%</td></tr>"

        pages_html = f"""
    <h2>Page Scores ({report.pages_checked} pages)</h2>
    <p style="color:#666;">Overall score is determined by the worst page.</p>
    <table>
        <tr><th>Page URL</th><th>Status</th><th>Score</th></tr>
        {pages_rows}
    </table>
"""

    # Main page checks
    checks_html = ""
    for category in WEIGHTS.keys():
        cat_checks = [c for c in report.checks if c.category == category]
        if cat_checks:
            checks_html += f"<tr><th colspan='3' style='background:#f0f0f0;text-align:left;'>{category.upper()} ({report.category_scores[category]:.1f}%)</th></tr>"
            for check in cat_checks:
                bg = "#d4edda" if check.passed else "#f8d7da"
                checks_html += f"<tr style='background:{bg}'><td>{check.name}</td><td>{'PASS' if check.passed else 'FAIL'}</td><td>{check.details}</td></tr>"

    # Worst page checks (if different from main)
    worst_checks_html = ""
    if report.pages_checked > 1 and report.worst_page_url != report.url:
        worst = next((p for p in report.page_reports if p.url == report.worst_page_url), None)
        if worst:
            worst_checks_html = f"""
    <h2>Detailed Checks - Worst Page</h2>
    <p style="color:#666;"><a href="{worst.url}">{worst.url}</a></p>
    <table>
        <tr><th>Check</th><th>Status</th><th>Details</th></tr>
"""
            for category in WEIGHTS.keys():
                cat_checks = [c for c in worst.checks if c.category == category]
                if cat_checks:
                    cat_score = worst.category_scores.get(category, 0)
                    worst_checks_html += f"<tr><th colspan='3' style='background:#f0f0f0;text-align:left;'>{category.upper()} ({cat_score:.1f}%)</th></tr>"
                    for check in cat_checks:
                        bg = "#d4edda" if check.passed else "#f8d7da"
                        worst_checks_html += f"<tr style='background:{bg}'><td>{check.name}</td><td>{'PASS' if check.passed else 'FAIL'}</td><td>{check.details}</td></tr>"
            worst_checks_html += "</table>"

    similarity_html = ""
    if report.url.rstrip("/") != report.reference_url.rstrip("/"):
        similarity_html = f'<p style="font-size: 18px; color: #666;">Similarity to reference: <strong>{report.similarity_score:.1f}%</strong></p>'

    pages_info = f" (worst of {report.pages_checked} pages)" if report.pages_checked > 1 else ""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Quality Report - {report.url}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1000px; margin: 40px auto; padding: 20px; }}
        h1 {{ margin-bottom: 5px; }}
        .meta {{ color: #666; margin-bottom: 20px; }}
        .score {{ font-size: 48px; font-weight: bold; color: {status_color}; }}
        .status {{ font-size: 24px; color: {status_color}; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; margin-bottom: 30px; }}
        th, td {{ padding: 8px 12px; border: 1px solid #ddd; text-align: left; }}
        th {{ background: #f5f5f5; }}
        a {{ color: #0066cc; }}
    </style>
</head>
<body>
    <h1>Quality Check Report</h1>
    <div class="meta">
        <p><strong>Target:</strong> <a href="{report.url}">{report.url}</a></p>
        <p><strong>Reference:</strong> <a href="{report.reference_url}">{report.reference_url}</a></p>
        <p><strong>Generated:</strong> {report.timestamp}</p>
        <p><strong>Threshold:</strong> {report.threshold}%</p>
        <p><strong>Pages checked:</strong> {report.pages_checked}</p>
    </div>
    <div class="score">{report.overall_score:.1f}%{pages_info}</div>
    <div class="status">{status_text}</div>
    {similarity_html}
{pages_html}
    <h2>Category Scores (Main Page)</h2>
    <table>
        <tr><th>Category</th><th>Weight</th><th>Score</th></tr>
        {"".join(f"<tr><td>{cat}</td><td>{int(WEIGHTS[cat]*100)}%</td><td>{score:.1f}%</td></tr>" for cat, score in report.category_scores.items())}
    </table>

    <h2>Detailed Checks - Main Page</h2>
    <table>
        <tr><th>Check</th><th>Status</th><th>Details</th></tr>
        {checks_html}
    </table>
{worst_checks_html}
</body>
</html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    return str(filename)


# =============================================================================
# MAIN
# =============================================================================

def run_quality_check(url: str, reference_url: str, threshold: float,
                      timeout: int, skip_links: bool, depth: int = 1) -> QualityReport:
    """Run complete quality check on a URL and optionally its subpages."""

    # Ensure URL has trailing slash for proper relative URL resolution
    # urljoin treats /path as a file without trailing slash, breaking relative links
    if not url.endswith('/'):
        url = url + '/'

    # Store normalized version for comparison (without trailing slash)
    url_normalized = url.rstrip('/')

    page_reports = []

    # Discover pages to check
    if depth > 1:
        pages = discover_subpages(url, depth, timeout)
    else:
        pages = [url]

    # Check each page
    print(f"\n{'='*60}")
    print(f"CHECKING {len(pages)} PAGE(S)")
    print('='*60)

    for i, page_url in enumerate(pages, 1):
        print(f"\n[Page {i}/{len(pages)}] {page_url}")
        print("-" * 60)

        html, response, load_time = fetch_page(page_url, timeout)
        if not html:
            print("  Error: Could not fetch page, skipping...")
            continue

        print(f"  Loaded in {load_time:.2f}s")

        soup = BeautifulSoup(html, "html.parser")
        css = extract_css(soup, page_url, timeout)

        all_checks = []

        print("  Checking structure...")
        all_checks.extend(check_structure(soup))

        print("  Checking visual elements...")
        all_checks.extend(check_visual(soup, css))

        print("  Checking technical quality...")
        all_checks.extend(check_technical(soup, html, load_time))

        # Only check links on main page (for performance)
        if not skip_links and page_url.rstrip('/') == url_normalized:
            print("  Checking links...")
            all_checks.extend(check_links_health(page_url, timeout))

        # Calculate scores for this page
        score, category_scores, passed = calculate_scores(all_checks, threshold, skip_links=(skip_links or page_url.rstrip('/') != url_normalized))

        page_report = PageReport(
            url=page_url,
            score=score,
            category_scores=category_scores,
            passed=passed,
            checks=all_checks
        )
        page_reports.append(page_report)

        status = "PASS" if passed else "FAIL"
        print(f"  Score: {score:.1f}% - {status}")

    # Aggregate results - worst page determines overall score
    if page_reports:
        worst_page = min(page_reports, key=lambda p: p.score)
        overall_score = worst_page.score
        worst_page_url = worst_page.url

        # Use main page's category scores for the report
        # Compare with both normalized (no slash) and original (with slash) versions
        main_page = next((p for p in page_reports if p.url.rstrip('/') == url_normalized), page_reports[0])
        main_checks = main_page.checks
        main_category_scores = main_page.category_scores
    else:
        overall_score = 0.0
        worst_page_url = url_normalized
        main_checks = []
        main_category_scores = {}

    passed = overall_score >= threshold

    # Reference comparison (only if target != reference)
    similarity_score = 0.0
    ref_checks = []
    if url_normalized != reference_url.rstrip("/"):
        print(f"\nAnalyzing reference site: {reference_url}...")
        ref_checks, _ = analyze_site(reference_url, timeout, skip_links=True)
        if ref_checks and main_checks:
            similarity_score = calculate_similarity(main_checks, ref_checks)
            print(f"Similarity to reference: {similarity_score}%")
    else:
        similarity_score = 100.0  # Same site

    return QualityReport(
        url=url_normalized,
        reference_url=reference_url,
        timestamp=datetime.now().isoformat(),
        overall_score=overall_score,
        category_scores=main_category_scores,
        passed=passed,
        threshold=threshold,
        similarity_score=similarity_score,
        checks=main_checks,
        reference_checks=ref_checks,
        page_reports=page_reports,
        worst_page_url=worst_page_url,
        pages_checked=len(page_reports)
    )


def main():
    """Main entry point."""
    args = parse_args()

    # Validate URL
    parsed = urlparse(args.url)
    if not parsed.scheme or not parsed.netloc:
        print(f"Error: Invalid URL '{args.url}'")
        sys.exit(1)

    # Run quality check
    report = run_quality_check(
        url=args.url,
        reference_url=args.reference,
        threshold=args.threshold,
        timeout=args.timeout,
        skip_links=args.skip_links,
        depth=args.depth
    )

    # Print console report
    print_report(report)

    # Export reports
    json_path = export_json(report, args.output)
    html_path = export_html(report, args.output)

    print(f"\nReports saved:")
    print(f"  JSON: {json_path}")
    print(f"  HTML: {html_path}")

    # Exit code
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
