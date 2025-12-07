"""
Create side-by-side before/after comparison images
"""
from PIL import Image
from pathlib import Path

screenshot_dir = Path(__file__).parent

pages = [
    ("index", "Landing Page"),
    ("schedule", "Weekly Schedule"),
    ("pdfs", "PDF Resources"),
    ("notebooks", "Jupyter Notebooks"),
    ("assignments", "Assignments"),
]

def create_comparison(page_name, page_title):
    before_path = screenshot_dir / f"before_{page_name}.png"
    after_path = screenshot_dir / f"after_{page_name}.png"

    if not before_path.exists() or not after_path.exists():
        print(f"Skipping {page_name}: missing images")
        return

    # Load images
    before = Image.open(before_path)
    after = Image.open(after_path)

    # Resize to same height (use smaller height)
    target_height = min(before.height, after.height)

    before_ratio = before.width / before.height
    after_ratio = after.width / after.height

    before_resized = before.resize((int(target_height * before_ratio), target_height))
    after_resized = after.resize((int(target_height * after_ratio), target_height))

    # Create side-by-side image with gap
    gap = 20
    total_width = before_resized.width + gap + after_resized.width

    comparison = Image.new('RGB', (total_width, target_height), 'white')
    comparison.paste(before_resized, (0, 0))
    comparison.paste(after_resized, (before_resized.width + gap, 0))

    # Save comparison
    output_path = screenshot_dir / f"comparison_{page_name}.png"
    comparison.save(output_path, quality=85)

    print(f"Created: comparison_{page_name}.png ({page_title})")
    print(f"  Before: {before.width}x{before.height} -> {before_resized.width}x{before_resized.height}")
    print(f"  After:  {after.width}x{after.height} -> {after_resized.width}x{after_resized.height}")
    print(f"  Final:  {total_width}x{target_height}")

def main():
    print("Creating before/after comparison images...\n")

    for page_name, page_title in pages:
        create_comparison(page_name, page_title)
        print()

    print("All comparisons created!")

if __name__ == "__main__":
    main()
