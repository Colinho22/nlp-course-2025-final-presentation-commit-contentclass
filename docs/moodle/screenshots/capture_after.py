"""
Capture "after" screenshots of all 5 Moodle pages
"""
from playwright.sync_api import sync_playwright
from pathlib import Path
import time

# Paths
base_dir = Path(__file__).parent.parent
screenshot_dir = Path(__file__).parent

pages = [
    ("index.html", "after_index.png"),
    ("schedule.html", "after_schedule.png"),
    ("pdfs.html", "after_pdfs.png"),
    ("notebooks.html", "after_notebooks.png"),
    ("assignments.html", "after_assignments.png"),
]

def capture_screenshots():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1920, 'height': 1080})

        for html_file, screenshot_file in pages:
            file_path = base_dir / html_file
            url = f"file:///{file_path.as_posix()}"

            print(f"Capturing {html_file}...")
            page.goto(url)
            time.sleep(0.5)  # Let page render

            screenshot_path = screenshot_dir / screenshot_file
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"  Saved: {screenshot_file}")

        browser.close()

    print("\nAll after screenshots captured!")

if __name__ == "__main__":
    capture_screenshots()
