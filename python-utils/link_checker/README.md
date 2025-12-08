# Link Checker

Validates all links on a website with deep crawling capability. Checks anchor tags, images, scripts, stylesheets, and other resources.

## Features

- Deep recursive crawl of internal pages (configurable depth)
- Checks all link types: `<a>`, `<img>`, `<script>`, `<link>`, `<iframe>`, etc.
- Parallel link checking with configurable workers
- Exports results to CSV and JSON with timestamps
- Console summary with broken link details

## Installation

```bash
pip install requests beautifulsoup4
```

## Usage

```bash
# Basic usage
python link_checker.py https://example.com

# With options
python link_checker.py https://example.com --depth 5 --timeout 15 --output ./reports
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--depth`, `-d` | 3 | Maximum crawl depth for internal pages |
| `--timeout`, `-t` | 10 | Request timeout in seconds |
| `--output`, `-o` | `.` | Output directory for reports |
| `--workers`, `-w` | 10 | Number of concurrent workers |

## Output

- `link_report_YYYYMMDD_HHMM.csv` - Tabular format with all links
- `link_report_YYYYMMDD_HHMM.json` - Structured JSON with metadata and statistics

## Exit Codes

- `0` - All links valid
- `1` - Broken links found
