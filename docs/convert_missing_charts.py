"""
Convert missing PDF charts to PNG for web display.
"""
from pathlib import Path
from pdf2image import convert_from_path

# Source PDFs
SOURCE_DIR = Path(r"D:\Joerg\Research\slides\2025_NLP_16\embeddings\figures")
TARGET_DIR = Path(r"D:\Joerg\Research\slides\2025_NLP_16\docs\charts")

CHARTS = [
    "semantic_clusters_3d.pdf",
    "landmark_analogies_3d.pdf",
    "context_movement_3d.pdf",
    "cultural_analogies_3d.pdf"
]

def main():
    TARGET_DIR.mkdir(exist_ok=True)

    for chart in CHARTS:
        pdf_path = SOURCE_DIR / chart
        if not pdf_path.exists():
            print(f"NOT FOUND: {pdf_path}")
            continue

        png_name = chart.replace(".pdf", ".png")
        png_path = TARGET_DIR / png_name

        print(f"Converting: {chart} -> {png_name}")

        # Convert PDF to PNG (first page only)
        images = convert_from_path(str(pdf_path), dpi=150)
        if images:
            images[0].save(str(png_path), "PNG")
            print(f"  Saved: {png_path}")

    print("\nDone!")

if __name__ == "__main__":
    main()
