#!/usr/bin/env python3
"""
Generate placeholder images for NLP Course Site
These will be replaced by actual thumbnails during CI/CD
"""

import os
from pathlib import Path

# Create assets/images directory
docs_dir = Path(__file__).parent
images_dir = docs_dir / 'assets' / 'images'
images_dir.mkdir(parents=True, exist_ok=True)

# Week data for labels
WEEKS = [
    ('01', 'N-grams', '#1e3a5f'),
    ('02', 'Embeddings', '#2e4a6f'),
    ('03', 'RNN/LSTM', '#3e5a7f'),
    ('04', 'Seq2Seq', '#4e6a8f'),
    ('05', 'Transformers', '#3333B2'),
    ('06', 'Pre-trained', '#4343C2'),
    ('07', 'Scaling', '#5353D2'),
    ('08', 'Tokenization', '#6363E2'),
    ('09', 'Decoding', '#1e5a3f'),
    ('10', 'Fine-tuning', '#2e6a4f'),
    ('11', 'Efficiency', '#3e7a5f'),
    ('12', 'Ethics', '#4e8a6f'),
]

# Gallery chart images
GALLERY_CHARTS = [
    'skipgram_architecture_bsc',
    'cbow_architecture_bsc',
    '3d_transformer_architecture',
    'beam_search_tree_graphviz',
    'temperature_effects_bsc',
    'word_arithmetic_3d_bsc',
    'training_evolution_bsc',
    'negative_sampling_process_bsc',
    'contrastive_vs_nucleus_bsc',
    'degeneration_problem_bsc',
    'topk_example_bsc',
    'vocabulary_probability_bsc',
]

def create_svg_placeholder(filename, label, color='#3333B2', width=400, height=300):
    """Create a simple SVG placeholder image"""
    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{color};stop-opacity:1" />
      <stop offset="100%" style="stop-color:#1e3a5f;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="{width}" height="{height}" fill="url(#grad)"/>
  <text x="50%" y="50%" fill="white" font-family="Arial, sans-serif" font-size="24" font-weight="bold" text-anchor="middle" dominant-baseline="middle">{label}</text>
</svg>'''

    svg_path = images_dir / f'{filename}.svg'
    with open(svg_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    return svg_path

def create_png_placeholder(filename, label, color='#3333B2', width=400, height=300):
    """Create a PNG placeholder using PIL if available, otherwise use SVG"""
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Create gradient background
        img = Image.new('RGB', (width, height))
        pixels = img.load()

        # Parse hex color
        r1 = int(color[1:3], 16)
        g1 = int(color[3:5], 16)
        b1 = int(color[5:7], 16)
        r2, g2, b2 = 30, 58, 95  # #1e3a5f

        for y in range(height):
            for x in range(width):
                # Diagonal gradient
                t = (x + y) / (width + height)
                r = int(r1 * (1 - t) + r2 * t)
                g = int(g1 * (1 - t) + g2 * t)
                b = int(b1 * (1 - t) + b2 * t)
                pixels[x, y] = (r, g, b)

        # Add text
        draw = ImageDraw.Draw(img)

        # Try to use a nice font, fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except:
            font = ImageFont.load_default()

        # Get text bounding box
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center text
        x = (width - text_width) // 2
        y = (height - text_height) // 2

        draw.text((x, y), label, fill='white', font=font)

        png_path = images_dir / f'{filename}.png'
        img.save(png_path, 'PNG')
        return png_path

    except ImportError:
        # Fall back to SVG if PIL not available
        print(f"  PIL not available, creating SVG for {filename}")
        return create_svg_placeholder(filename, label, color, width, height)

def main():
    print("=" * 60)
    print("Generating Placeholder Images")
    print("=" * 60)

    # Try to import PIL for PNG generation
    try:
        from PIL import Image
        use_png = True
        print("Using PIL for PNG generation")
    except ImportError:
        use_png = False
        print("PIL not available, will create SVG placeholders")

    print(f"\nOutput directory: {images_dir}")
    print()

    # Generate week thumbnails
    print("Generating week thumbnails...")
    for num, title, color in WEEKS:
        filename = f'week{num}'
        label = f'Week {num}\n{title}'

        if use_png:
            path = create_png_placeholder(filename, label, color)
        else:
            path = create_svg_placeholder(filename, label, color)
        print(f"  Created: {path.name}")

    # Generate gallery chart placeholders
    print("\nGenerating gallery chart placeholders...")
    for chart_name in GALLERY_CHARTS:
        # Create descriptive label from filename
        label = chart_name.replace('_bsc', '').replace('_', ' ').title()
        if len(label) > 20:
            label = label[:17] + '...'

        if use_png:
            path = create_png_placeholder(chart_name, label, '#3333B2')
        else:
            path = create_svg_placeholder(chart_name, label, '#3333B2')
        print(f"  Created: {path.name}")

    print(f"\n{'=' * 60}")
    print(f"Generated {len(WEEKS) + len(GALLERY_CHARTS)} placeholder images")
    print("=" * 60)

    return 0

if __name__ == '__main__':
    exit(main())
