# Quality Checker

Evaluates GitHub Pages site quality by comparing against a reference site. Produces similarity score + detailed pass/fail checklist.

## Features

- **Structure checks** (30%): Navigation, sidebar, hero, grids, footer, search, collapsible sections, journey tracker
- **Visual checks** (30%): Stylesheets, responsive design, media queries, images, gradients, shadows, transitions, hover effects, color palette (hex/rgb/hsl/CSS vars)
- **Technical checks** (20%): HTML validity, semantic HTML, alt texts, ARIA/accessibility, heading hierarchy, load time, render-blocking assets, meta tags, favicon, lang attribute, Open Graph tags
- **Link health** (20%): Integrates with link_checker for broken link detection
- **Reference comparison**: Calculates similarity score against a reference site

## Installation

```bash
pip install requests beautifulsoup4 lxml
```

## Usage

```bash
# Basic usage (compares to default reference)
python quality_checker.py https://your-site.github.io/repo

# With custom reference site
python quality_checker.py https://target.io --reference https://reference.io

# With custom threshold and output
python quality_checker.py https://target.io --threshold 80 --output ./reports

# Skip link checking (faster, excludes links from scoring)
python quality_checker.py https://target.io --skip-links

# Check subpages (crawl depth 2)
python quality_checker.py https://target.io --depth 2 --skip-links

# Deep crawl with depth 4
python quality_checker.py https://target.io --depth 4 --skip-links
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--reference`, `-r` | neural-networks | Reference URL to compare against |
| `--threshold`, `-t` | 70 | Minimum passing score percentage |
| `--output`, `-o` | `.` | Output directory for reports |
| `--timeout` | 30 | Request timeout in seconds |
| `--skip-links` | false | Skip link checking (excludes from scoring) |
| `--depth` | 1 | Crawl depth for subpages (1 = main page only) |

## Output

- **Console**: Category breakdown + pass/fail for each check + overall score + similarity score
- **JSON**: `quality_report_YYYYMMDD_HHMM.json` - Full structured report with all page data
- **HTML**: `quality_report_YYYYMMDD_HHMM.html` - Visual report with color-coded results

When using `--depth > 1`, reports include:
- Summary table of all pages with scores
- Detailed checks for main page
- Detailed checks for worst page (if different from main)

## Scoring

| Category | Weight | Checks |
|----------|--------|--------|
| Structure | 30% | 8 checks |
| Visual | 30% | 10 checks |
| Technical | 20% | 11 checks |
| Links | 20% | 1 check (link health ratio) |

When `--skip-links` is used, the links category is excluded and weights are redistributed proportionally.

## Multi-Page Scoring

When `--depth > 1`, the checker crawls internal pages using BFS (breadth-first search):
- Discovers all internal HTML pages within the base path
- Runs all checks on each page (link checking only on main page for performance)
- **Worst page score determines overall score** - this ensures quality is consistent across all pages
- Reports show per-page breakdown sorted by score

## Similarity Score

When comparing to a different reference site, the checker calculates a similarity score (0-100%):
- Compares each check between target and reference
- Both passing = full match
- Both failing = partial match
- One pass, one fail = no match

## Exit Codes

- `0` - Passed (score >= threshold)
- `1` - Failed (score < threshold)

## Integration with link_checker

If `link_checker.py` is available in the parent `link_checker/` folder, it will be used for link health checking. Otherwise, link checks are skipped.
