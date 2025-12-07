#!/usr/bin/env python3
"""
Take final screenshots of all Moodle site pages (light and dark mode)
"""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

async def take_screenshots():
    """Take screenshots of all pages in both light and dark mode"""
    moodle_dir = Path(__file__).parent.parent.parent / 'docs' / 'moodle'
    screenshots_dir = moodle_dir / 'screenshots'
    screenshots_dir.mkdir(exist_ok=True)

    pages = [
        ('index.html', 'home'),
        ('schedule.html', 'schedule'),
        ('pdfs.html', 'pdfs'),
        ('notebooks.html', 'notebooks'),
        ('assignments.html', 'assignments')
    ]

    async with async_playwright() as p:
        browser = await p.chromium.launch()

        # Light mode screenshots
        print("\nCapturing light mode screenshots...")
        context = await browser.new_context(
            viewport={'width': 1400, 'height': 900},
            color_scheme='light'
        )
        page = await context.new_page()

        for filename, name in pages:
            url = f'file:///{(moodle_dir / filename).as_posix()}'
            await page.goto(url)
            await page.wait_for_load_state('networkidle')
            # Force light mode
            await page.evaluate("document.documentElement.setAttribute('data-theme', 'light')")
            await asyncio.sleep(0.3)
            screenshot_path = screenshots_dir / f'final_light_{name}.png'
            await page.screenshot(path=str(screenshot_path), full_page=False)
            print(f"  Saved: {screenshot_path.name}")

        await context.close()

        # Dark mode screenshots
        print("\nCapturing dark mode screenshots...")
        context = await browser.new_context(
            viewport={'width': 1400, 'height': 900},
            color_scheme='dark'
        )
        page = await context.new_page()

        for filename, name in pages:
            url = f'file:///{(moodle_dir / filename).as_posix()}'
            await page.goto(url)
            await page.wait_for_load_state('networkidle')
            # Force dark mode
            await page.evaluate("document.documentElement.setAttribute('data-theme', 'dark')")
            await asyncio.sleep(0.3)
            screenshot_path = screenshots_dir / f'final_dark_{name}.png'
            await page.screenshot(path=str(screenshot_path), full_page=False)
            print(f"  Saved: {screenshot_path.name}")

        await context.close()
        await browser.close()

    print(f"\nAll screenshots saved to: {screenshots_dir}")

    # List all screenshots
    all_screenshots = list(screenshots_dir.glob('final_*.png'))
    total_size = sum(f.stat().st_size for f in all_screenshots)
    print(f"Total: {len(all_screenshots)} screenshots ({total_size / 1024:.1f} KB)")

if __name__ == '__main__':
    asyncio.run(take_screenshots())
