from playwright.sync_api import sync_playwright
from pathlib import Path

def take_screenshots():
    site_dir = Path(r'D:\Joerg\Research\slides\2025_NLP_16\docs\moodle')
    screenshots_dir = site_dir / 'screenshots'
    screenshots_dir.mkdir(exist_ok=True)

    pages = ['index', 'schedule', 'pdfs', 'notebooks', 'assignments']

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1920, 'height': 1080})

        for name in pages:
            url = f'file:///{site_dir}/{name}.html'
            page.goto(url)
            page.wait_for_load_state('networkidle')
            page.screenshot(path=str(screenshots_dir / f'before_{name}.png'), full_page=True)
            print(f'Captured: before_{name}.png')

        browser.close()

if __name__ == '__main__':
    take_screenshots()
